from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ijepa_lite.losses.context_loss import context_loss
from ijepa_lite.masking.base import CollateMasker, LatentMasker, MaskOutput
from ijepa_lite.masking.compressor import TokenCompressor
from ijepa_lite.masking.metrics import mask_diagnostics
from ijepa_lite.models.ema import ema_update


class IJEPAModel(nn.Module):
    """
    i-JEPA model.

    Masking strategy
    ----------------
    Two paths, selected automatically — the training loop is identical for both:

    Deterministic (CollateMasker) path:
        Masks are pre-computed in DataLoader workers (CPU) and arrive in the
        batch dict as "context_idx" / "target_idx".  The training loop passes
        them to forward() as the `masks` argument.  No extra encoder forward pass.

    Learned (LatentMasker) path:
        `masks` is None (collate is dumb).  The model:
          1. Runs the target (EMA) encoder with no_grad to obtain patch tokens.
          2. Compresses them via token_compressor.
          3. Runs latent_masker(compressed_tokens) → MaskOutput  [grads flow here].
          4. Uses hard indices for the standard JEPA forward.
          5. Adds latent_masker.aux_loss() to the reconstruction loss to route
             gradients back to masker and compressor parameters.
        The EMA encoder runs once per step — its output is reused for both masking and targets.

    Args
    ----
    context_encoder : ViTTokens encoder for the context (online, trained via grad)
    target_encoder  : ViTTokens encoder for the target (EMA of context, no grad)
    predictor       : Predictor module
    loss_fn         : Token-level reconstruction loss
    ema_momentum    : Initial EMA momentum (updated by training loop)
    mask_generator  : Optional CollateMasker — kept for backward compat / unit tests.
                      Used as fallback when masks=None AND latent_masker=None.
    latent_masker   : Optional LatentMasker — when set, masking happens here on GPU.
    token_compressor: Required when latent_masker is set.  Compresses EMA tokens
                      before passing to the latent masker.
    """

    def __init__(
        self,
        context_encoder: nn.Module,
        target_encoder: nn.Module,
        predictor: nn.Module,
        loss_fn: nn.Module,
        ema_momentum: float,
        # Kept for backward compat / unit-test convenience.
        mask_generator: Optional[CollateMasker] = None,
        # New: learned masker path.
        latent_masker: Optional[LatentMasker] = None,
        token_compressor: Optional[TokenCompressor] = None,
        predict_blocks_jointly: bool = True,
        # Context loss (V-JEPA 2.1-style visible token supervision).
        ctx_loss_weight: float = 0.0,
        ctx_loss_gamma: float = 0.7,
        ctx_loss_warmup_start: int = 0,
        ctx_loss_warmup_end: int = 0,
        grid_size: int = 0,
    ) -> None:
        super().__init__()

        if latent_masker is not None and token_compressor is None:
            raise ValueError(
                "latent_masker is set but token_compressor is None. "
                "A TokenCompressor is required to compress EMA encoder tokens "
                "before passing them to the LatentMasker."
            )

        self.context_encoder = context_encoder
        self.target_encoder = target_encoder
        self.predictor = predictor
        self.loss_fn = loss_fn
        self.ema_momentum = float(ema_momentum)
        self._mask_generator = mask_generator   # fallback only

        self.predict_blocks_jointly = predict_blocks_jointly

        # Context loss config
        self.ctx_loss_weight = float(ctx_loss_weight)
        self.ctx_loss_enabled = self.ctx_loss_weight > 0.0
        self.ctx_loss_gamma = float(ctx_loss_gamma)
        self.ctx_warmup_start = int(ctx_loss_warmup_start)
        self.ctx_warmup_end = int(ctx_loss_warmup_end)
        self.grid_size = int(grid_size)

        # Registered as submodules so their params are checkpointed and optimised.
        self.latent_masker = latent_masker
        self.token_compressor = token_compressor

        for p in self.target_encoder.parameters():
            p.requires_grad = False

    # ------------------------------------------------------------------
    # EMA update — called by the training loop after each step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_target(self) -> None:
        ema_update(self.target_encoder, self.context_encoder, self.ema_momentum)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        images: torch.Tensor,
        masks: Optional[Dict[str, torch.Tensor]] = None,
        compute_agreement: bool = False,
        compute_mask_metrics: bool = False,
        epoch: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            images               : (B, C, H, W)
            masks                : dict with "context_idx" and "target_idx" when using a
                                   CollateMasker.  None when using a LatentMasker (masking
                                   happens here) or when falling back to _mask_generator.
            compute_agreement    : compute encoder agreement diagnostic (extra no-grad ops).
            compute_mask_metrics : compute full mask diagnostics (spatial coverage, batch
                                   IoU, entropy, effective patches, etc.).  Should be
                                   gated on do_log in the training loop — free when False.

        Returns dict keys:
            "loss"                 : total scalar loss (reconstruction + masker aux)
            "reconstruction_loss"  : JEPA reconstruction loss only
            "pred"                 : (B, K, D) or (B, M, K, D) predictor output, detached
            "target"               : same shape as pred, target encoder tokens, detached
            "mask_stats"           : dict[str, float] — all mask/* metrics
            "ctx_tokens_all"       : (B, Nctx, D) context tokens (if compute_agreement)
            "tgt_tokens_all"       : (B, Nctx, D) target tokens at same positions
        """
        # ------------------------------------------------------------------
        # Step 1: Resolve MaskOutput
        #
        # Latent masker path: _resolve_latent_masks runs the EMA encoder once
        # and returns both the MaskOutput and the raw ema_tokens (B, N, D).
        # We reuse those tokens directly as tgt_tokens_all — no second forward.
        #
        # Deterministic path: tgt_tokens_all is produced here as usual.
        # ------------------------------------------------------------------
        cached_ema_tokens: Optional[torch.Tensor] = None

        if self.latent_masker is not None:
            mask_output, cached_ema_tokens = self._resolve_latent_masks(images)
        else:
            mask_output = self._resolve_collate_masks(masks, images)

        ctx_idx = mask_output.context_idx  # (B, Nctx)
        tgt_idx = mask_output.target_idx   # (B, Ntgt) or (B, M, K)

        # ------------------------------------------------------------------
        # Step 2: Context encoder (masked, gradients flow)
        # ------------------------------------------------------------------
        ctx_tokens = self.context_encoder(images, keep_idx=ctx_idx)  # (B, Nctx, D)
        b = images.shape[0]
        d = ctx_tokens.shape[-1]

        # ------------------------------------------------------------------
        # Step 3: Target token bank — reuse cached EMA tokens if available,
        # otherwise run the target encoder now (deterministic masker path).
        # Layer norm is applied in both cases.
        # ------------------------------------------------------------------
        if cached_ema_tokens is not None:
            with torch.no_grad():
                tgt_tokens_all = F.layer_norm(
                    cached_ema_tokens, (cached_ema_tokens.shape[-1],)
                )
        else:
            with torch.no_grad():
                tgt_tokens_all = self.target_encoder(images)   # (B, N, D)
                tgt_tokens_all = F.layer_norm(tgt_tokens_all, (tgt_tokens_all.shape[-1],))

        # Optional encoder-agreement diagnostic (re-uses already-computed tensors)
        ctx_tokens_all: Optional[torch.Tensor] = None
        tgt_at_ctx: Optional[torch.Tensor] = None
        if compute_agreement:
            ctx_tokens_all = ctx_tokens.detach()
            tgt_at_ctx = tgt_tokens_all.gather(
                1, ctx_idx.unsqueeze(-1).expand(-1, -1, d)
            ).detach()

        # ------------------------------------------------------------------
        # Step 4: Predictor + loss (single-block or multi-block)
        # ------------------------------------------------------------------
        if tgt_idx.dim() == 2:
            reconstruction_loss, pred, tgt_tokens, patch_loss, pred_ctx = (
                self._forward_single_block(
                    ctx_tokens, ctx_idx, tgt_idx, tgt_tokens_all, b, d
                )
            )
        elif tgt_idx.dim() == 3:
            reconstruction_loss, pred, tgt_tokens, patch_loss, pred_ctx = (
                self._forward_multi_block(
                    ctx_tokens, ctx_idx, tgt_idx, tgt_tokens_all, b, d
                )
            )
        else:
            raise ValueError(
                f"Unsupported target_idx.dim()={tgt_idx.dim()}, expected 2 or 3."
            )

        # ------------------------------------------------------------------
        # Step 5: Masker auxiliary loss and total loss assembly
        #
        # owns_loss=False (default, 2-way entropy maskers):
        #   total = reconstruction_loss + aux_loss()
        # owns_loss=True (RateDist3WayMasker):
        #   aux_loss() returns D_soft + λ·N·R as the complete objective.
        #   reconstruction_loss (unweighted mean) is preserved for monitoring.
        # ------------------------------------------------------------------
        if self.latent_masker is not None:
            masker_aux = self.latent_masker.aux_loss(
                mask_output, reconstruction_loss, patch_loss=patch_loss
            )
            if self.latent_masker.owns_loss:
                total_loss = masker_aux
            else:
                total_loss = reconstruction_loss + masker_aux
        else:
            masker_aux = reconstruction_loss.new_zeros(())
            total_loss = reconstruction_loss

        # ------------------------------------------------------------------
        # Step 5b: Context loss (V-JEPA 2.1-style visible token supervision)
        # ------------------------------------------------------------------
        ctx_loss_val = None
        if pred_ctx is not None:
            tgt_idx_flat = tgt_idx.reshape(b, -1) if tgt_idx.dim() == 3 else tgt_idx
            tgt_at_ctx_pos = tgt_tokens_all.gather(
                1, ctx_idx.unsqueeze(-1).expand(-1, -1, d)
            )
            alpha = self._ctx_alpha(epoch)
            ctx_l = context_loss(
                pred_ctx, tgt_at_ctx_pos, ctx_idx, tgt_idx_flat,
                self.grid_size, self.loss_fn,
                gamma=self.ctx_loss_gamma, alpha=alpha,
            )
            ctx_loss_val = float(ctx_l.detach().item())
            total_loss = total_loss + self.ctx_loss_weight * ctx_l

        # ------------------------------------------------------------------
        # Step 6: Mask diagnostics
        # mask_diagnostics() always returns the cheap always-on metrics.
        # Full diagnostics (coverage, IoU, entropy, etc.) are gated on
        # compute_mask_metrics so they only run at log steps.
        # Note: for RD masker, aux_loss writes D_soft and R into mask_output.aux
        # before mask_diagnostics runs, so they are picked up automatically.
        # ------------------------------------------------------------------
        num_patches = tgt_tokens_all.shape[1]
        mask_stats = mask_diagnostics(
            mask_output,
            num_patches=num_patches,
            masker_loss=masker_aux,
            full=compute_mask_metrics,
            patch_loss=patch_loss,
        )

        # ------------------------------------------------------------------
        # Output
        # ------------------------------------------------------------------
        out: Dict = {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "pred": pred.detach(),
            "target": tgt_tokens.detach(),
            "patch_loss": patch_loss.detach(),   # (B, K) — for diagnostics / curriculum
            "mask_stats": mask_stats,
            "ctx_loss": ctx_loss_val,
        }
        if compute_agreement:
            out["ctx_tokens_all"] = ctx_tokens_all
            out["tgt_tokens_all"] = tgt_at_ctx
        return out

    # ------------------------------------------------------------------
    # Private: context loss warmup
    # ------------------------------------------------------------------

    def _ctx_alpha(self, epoch: int) -> float:
        if self.ctx_warmup_end <= self.ctx_warmup_start:
            return 1.0
        if epoch < self.ctx_warmup_start:
            return 0.0
        if epoch >= self.ctx_warmup_end:
            return 1.0
        return (epoch - self.ctx_warmup_start) / (
            self.ctx_warmup_end - self.ctx_warmup_start
        )

    # ------------------------------------------------------------------
    # Private: mask resolution
    # ------------------------------------------------------------------

    def _resolve_latent_masks(
        self, images: torch.Tensor
    ) -> tuple[MaskOutput, torch.Tensor]:
        """
        Run EMA encoder → compress → latent masker.

        Returns (mask_output, ema_tokens) so the caller can reuse ema_tokens
        as tgt_tokens_all without a second encoder forward.  The EMA encoder
        sees the full image regardless of compressor mode — compression is
        applied to its output, not its input — so ema_tokens is always (B, N, D)
        and is always valid as the target token bank.

        The target encoder runs without gradients (frozen / EMA-updated).
        The compressor and latent masker run with gradients so their parameters
        receive signal from both the reconstruction loss (via aux_loss) and any
        additional terms the masker defines.
        """
        with torch.no_grad():
            if self.token_compressor.needs_cls:
                cls_token, ema_tokens = self.target_encoder(images, return_cls=True)
            else:
                cls_token = None
                ema_tokens = self.target_encoder(images)   # (B, N, D)

        # Compressor and latent masker are outside no_grad — grads flow normally.
        compressed = self.token_compressor(ema_tokens, cls_token=cls_token)  # (B, M, D)
        if self.latent_masker.needs_full_tokens:
            mask_output = self.latent_masker(compressed, ema_full=ema_tokens)
        else:
            mask_output = self.latent_masker(compressed)
        return mask_output, ema_tokens

    def _resolve_collate_masks(
        self,
        masks: Optional[Dict[str, torch.Tensor]],
        images: torch.Tensor,
    ) -> MaskOutput:
        """
        Reconstruct a MaskOutput from the pre-computed batch dict.
        Falls back to _mask_generator for unit tests / debugging.
        """
        if masks is None:
            if self._mask_generator is None:
                raise ValueError(
                    "masks=None but no mask_generator or latent_masker was provided "
                    "to IJEPAModel. Either pass pre-computed masks, supply a "
                    "mask_generator, or set latent_masker."
                )
            generated = self._mask_generator(batch_size=images.shape[0])
            return MaskOutput(
                context_idx=generated.context_idx.to(images.device, non_blocking=True),
                target_idx=generated.target_idx.to(images.device, non_blocking=True),
            )

        return MaskOutput(
            context_idx=masks["context_idx"].to(images.device, non_blocking=True),
            target_idx=masks["target_idx"].to(images.device, non_blocking=True),
        )

    # ------------------------------------------------------------------
    # Private: predictor + loss (split by tgt_idx dimensionality)
    # ------------------------------------------------------------------

    def _forward_single_block(
        self,
        ctx_tokens: torch.Tensor,     # (B, Nctx, D)
        ctx_idx: torch.Tensor,        # (B, Nctx)
        tgt_idx: torch.Tensor,        # (B, Ntgt)
        tgt_tokens_all: torch.Tensor, # (B, N, D)
        b: int,
        d: int,
    ):
        tgt_tokens = tgt_tokens_all.gather(
            1, tgt_idx.unsqueeze(-1).expand(-1, -1, d)
        )  # (B, Ntgt, D)

        result = self.predictor(
            ctx_tokens, ctx_idx=ctx_idx, tgt_idx=tgt_idx,
            return_ctx_pred=self.ctx_loss_enabled,
        )
        if self.ctx_loss_enabled:
            pred, pred_ctx = result
        else:
            pred, pred_ctx = result, None

        patch_loss = self.loss_fn(pred, tgt_tokens, reduction="none")  # (B, Ntgt)
        loss = patch_loss.mean()
        return loss, pred, tgt_tokens, patch_loss, pred_ctx

    def _forward_multi_block(
        self,
        ctx_tokens: torch.Tensor,     # (B, Nctx, D)
        ctx_idx: torch.Tensor,        # (B, Nctx)
        tgt_idx: torch.Tensor,        # (B, M, K)
        tgt_tokens_all: torch.Tensor, # (B, N, D)
        b: int,
        d: int,
    ):
        m = tgt_idx.shape[1]
        k = tgt_idx.shape[2]

        # -- Separate prediction: each block predicted independently -----------
        if not self.predict_blocks_jointly:
            preds_list = []
            tgts_list = []
            ploss_list = []
            for i in range(m):
                block_idx = tgt_idx[:, i, :]  # (B, K)
                block_tgt = tgt_tokens_all.gather(
                    1, block_idx.unsqueeze(-1).expand(-1, -1, d)
                )  # (B, K, D)
                block_pred = self.predictor(
                    ctx_tokens, ctx_idx=ctx_idx, tgt_idx=block_idx
                )  # (B, K, D)
                block_ploss = self.loss_fn(block_pred, block_tgt, reduction="none")  # (B, K)
                preds_list.append(block_pred)
                tgts_list.append(block_tgt)
                ploss_list.append(block_ploss)

            pred = torch.stack(preds_list, dim=1)        # (B, M, K, D)
            tgt_tokens = torch.stack(tgts_list, dim=1)   # (B, M, K, D)
            patch_loss_cat = torch.cat(ploss_list, dim=1) # (B, M*K)
            loss = patch_loss_cat.mean()
            # ctx_loss not supported with separate prediction
            return loss, pred, tgt_tokens, patch_loss_cat, None

        # -- Joint prediction (default): all blocks concatenated ---------------
        tgt_idx_cat = tgt_idx.reshape(b, m * k)  # (B, M*K)

        tgt_tokens_cat = tgt_tokens_all.gather(
            1, tgt_idx_cat.unsqueeze(-1).expand(-1, -1, d)
        )  # (B, M*K, D)
        tgt_tokens = tgt_tokens_cat.reshape(b, m, k, d)  # (B, M, K, D)

        result = self.predictor(
            ctx_tokens, ctx_idx=ctx_idx, tgt_idx=tgt_idx_cat,
            return_ctx_pred=self.ctx_loss_enabled,
        )
        if self.ctx_loss_enabled:
            pred_cat, pred_ctx = result
        else:
            pred_cat, pred_ctx = result, None

        pred = pred_cat.reshape(b, m, k, d)  # (B, M, K, D)

        patch_loss_cat = self.loss_fn(pred_cat, tgt_tokens_cat, reduction="none")  # (B, M*K)
        loss = patch_loss_cat.mean()
        return loss, pred, tgt_tokens, patch_loss_cat, pred_ctx