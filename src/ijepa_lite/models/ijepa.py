from __future__ import annotations

import inspect
import math
from collections.abc import Sequence
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
    target_encoder  : ViTTokens encoder for the target path. Required when
                      ``target_mode="ema"`` and unused when ``target_mode="shared"``.
    predictor       : Predictor module
    loss_fn         : Token-level reconstruction loss
    ema_momentum    : Initial EMA momentum (updated by training loop)
    mask_generator  : Optional CollateMasker — kept for backward compat / unit tests.
                      Used as fallback when masks=None AND latent_masker=None.
    latent_masker   : Optional LatentMasker — when set, masking happens here on GPU.
    token_compressor: Required when latent_masker is set.  Compresses EMA tokens
                      before passing to the latent masker.
    target_mode     : "ema" (default) or "shared". In shared mode the context
                      encoder also produces the full-image target bank and the
                      full-view path keeps gradients.
    sigreg_loss     : Optional latent regularizer applied to full-view patch
                      tokens. Expected to return ``(loss, logs)``.
    sigreg_weight   : Weight applied to ``sigreg_loss`` when present.
    """

    def __init__(
        self,
        context_encoder: nn.Module,
        target_encoder: nn.Module | None,
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
        target_mode: str = "ema",
        sigreg_loss: nn.Module | None = None,
        sigreg_weight: float | Sequence[float] = 0.0,
        sigreg_weight_schedule: str = "constant",
        total_epochs: int = 0,
    ) -> None:
        super().__init__()

        if latent_masker is not None and token_compressor is None:
            raise ValueError(
                "latent_masker is set but token_compressor is None. "
                "A TokenCompressor is required to compress EMA encoder tokens "
                "before passing them to the LatentMasker."
            )

        target_mode = str(target_mode).lower()
        if target_mode not in {"ema", "shared"}:
            raise ValueError(f"Unsupported target_mode={target_mode!r}. Expected 'ema' or 'shared'.")
        if target_mode == "ema" and target_encoder is None:
            raise ValueError("target_encoder is required when target_mode='ema'.")

        self.context_encoder = context_encoder
        self.target_encoder = target_encoder
        self.predictor = predictor
        self.loss_fn = loss_fn
        self.ema_momentum = float(ema_momentum)
        self._mask_generator = mask_generator   # fallback only
        self.target_mode = target_mode
        self.sigreg_loss = sigreg_loss
        if isinstance(sigreg_weight, Sequence) and not isinstance(sigreg_weight, (str, bytes)):
            if len(sigreg_weight) != 2:
                raise ValueError(
                    "sigreg_weight must be a scalar or a [start, end] pair."
                )
            sigreg_weight_start = float(sigreg_weight[0])
            sigreg_weight_end = float(sigreg_weight[1])
        else:
            sigreg_weight_start = float(sigreg_weight)
            sigreg_weight_end = float(sigreg_weight)
        if sigreg_weight_start < 0.0 or sigreg_weight_end < 0.0:
            raise ValueError("sigreg_weight endpoints must be >= 0.")
        self.sigreg_weight_start = sigreg_weight_start
        self.sigreg_weight_end = sigreg_weight_end
        self.sigreg_weight_schedule = str(sigreg_weight_schedule).lower()
        if self.sigreg_weight_schedule not in {"constant", "linear", "cosine"}:
            raise ValueError(
                "sigreg_weight_schedule must be 'constant', 'linear', or 'cosine'."
            )
        self.total_epochs = int(total_epochs)

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

        if self.target_encoder is not None:
            for p in self.target_encoder.parameters():
                p.requires_grad = False

    @property
    def has_ema_target(self) -> bool:
        return self.target_mode == "ema"

    def get_eval_encoder(self) -> nn.Module:
        if self.target_mode == "shared":
            return self.context_encoder
        return self.target_encoder

    def get_viz_encoder(self) -> nn.Module:
        return self.get_eval_encoder()

    def _uses_projected_prediction_space(self) -> bool:
        return self.sigreg_loss is not None and max(
            self.sigreg_weight_start, self.sigreg_weight_end
        ) > 0.0

    def _project_prediction_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if not self._uses_projected_prediction_space():
            return tokens
        return self.sigreg_loss.project_tokens(tokens)

    def _get_sigreg_weight(self, epoch: int | None) -> float:
        if (
            self.sigreg_weight_start == self.sigreg_weight_end
            or self.sigreg_weight_schedule == "constant"
        ):
            return self.sigreg_weight_start

        if epoch is None or self.total_epochs <= 1:
            return self.sigreg_weight_start

        progress = min(max(float(epoch) / float(self.total_epochs - 1), 0.0), 1.0)
        if self.sigreg_weight_schedule == "linear":
            blend = progress
        else:
            blend = 0.5 * (1.0 - math.cos(math.pi * progress))

        return self.sigreg_weight_start + (
            self.sigreg_weight_end - self.sigreg_weight_start
        ) * blend

    # ------------------------------------------------------------------
    # EMA update — called by the training loop after each step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_target(self) -> None:
        if not self.has_ema_target:
            return
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
        cached_full_tokens: Optional[torch.Tensor] = None

        if self.latent_masker is not None:
            mask_output, cached_full_tokens = self._resolve_latent_masks(images, epoch=epoch)
        else:
            mask_output = self._resolve_collate_masks(masks, images)

        ctx_idx = mask_output.context_idx  # (B, Nctx)
        tgt_idx = mask_output.target_idx   # (B, Ntgt) or (B, M, K)
        target_block_counts = mask_output.aux.get("target_block_counts")

        # ------------------------------------------------------------------
        # Step 2: Context encoder (masked, gradients flow)
        # ------------------------------------------------------------------
        ctx_tokens = self.context_encoder(images, keep_idx=ctx_idx)  # (B, Nctx, D)
        b = images.shape[0]
        d = ctx_tokens.shape[-1]

        # ------------------------------------------------------------------
        # Step 3: Target token bank — reuse cached full-view tokens if available,
        # otherwise run the appropriate full-view encoder now. Layer norm is
        # applied in both cases so target semantics match the current vanilla path.
        # ------------------------------------------------------------------
        if cached_full_tokens is not None:
            full_tokens = cached_full_tokens
        else:
            full_tokens = self._encode_full_view_tokens(images)

        raw_tgt_tokens_all = F.layer_norm(full_tokens, (full_tokens.shape[-1],))
        if self._uses_projected_prediction_space():
            tgt_tokens_all = self.sigreg_loss.project_tokens(full_tokens)
        else:
            tgt_tokens_all = raw_tgt_tokens_all

        # Optional encoder-agreement diagnostic (re-uses already-computed tensors)
        ctx_tokens_all: Optional[torch.Tensor] = None
        tgt_at_ctx: Optional[torch.Tensor] = None
        if compute_agreement:
            ctx_tokens_all = ctx_tokens.detach()
            tgt_at_ctx = raw_tgt_tokens_all.gather(
                1, ctx_idx.unsqueeze(-1).expand(-1, -1, raw_tgt_tokens_all.shape[-1])
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
                    ctx_tokens, ctx_idx, tgt_idx, tgt_tokens_all, b, d,
                    target_block_counts=target_block_counts,
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

        model_stats: dict[str, float] = {}
        sigreg_weight = self._get_sigreg_weight(epoch)
        if self.sigreg_loss is not None and sigreg_weight > 0.0:
            sigreg_val, sigreg_logs = self.sigreg_loss.forward_projected(
                tgt_tokens_all, full_tokens
            )
            total_loss = total_loss + sigreg_weight * sigreg_val
            model_stats.update(sigreg_logs)
            model_stats["sigreg/weight"] = float(sigreg_weight)
            model_stats["sigreg/weight_start"] = float(self.sigreg_weight_start)
            model_stats["sigreg/weight_end"] = float(self.sigreg_weight_end)
            model_stats["sigreg/weight_schedule_id"] = {
                "constant": 0.0,
                "linear": 1.0,
                "cosine": 2.0,
            }[self.sigreg_weight_schedule]
            model_stats["sigreg/loss_weighted"] = float(
                (sigreg_weight * sigreg_val).detach().item()
            )

        # ------------------------------------------------------------------
        # Step 5b: Context loss (V-JEPA 2.1-style visible token supervision)
        # ------------------------------------------------------------------
        ctx_loss_val = None
        if pred_ctx is not None:
            if tgt_idx.dim() == 3:
                tgt_idx_flat = self._flatten_multi_block_target_idx(
                    tgt_idx, b, target_block_counts
                )
            else:
                tgt_idx_flat = tgt_idx
            tgt_at_ctx_pos = tgt_tokens_all.gather(
                1, ctx_idx.unsqueeze(-1).expand(-1, -1, tgt_tokens_all.shape[-1])
            )
            pred_ctx_loss = self._project_prediction_tokens(pred_ctx)
            alpha = self._ctx_alpha(epoch)
            ctx_l = context_loss(
                pred_ctx_loss, tgt_at_ctx_pos, ctx_idx, tgt_idx_flat,
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
            "model_stats": model_stats,
            "ctx_loss": ctx_loss_val,
        }
        if self.sigreg_loss is not None and sigreg_weight > 0.0:
            out["sigreg_loss_weighted"] = sigreg_weight * sigreg_val
        if compute_agreement:
            out["ctx_tokens_all"] = ctx_tokens_all
            out["tgt_tokens_all"] = tgt_at_ctx
        return out

    def _encode_full_view_tokens(self, images: torch.Tensor) -> torch.Tensor:
        if self.target_mode == "shared":
            return self.context_encoder(images)
        with torch.no_grad():
            return self.target_encoder(images)

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
        self, images: torch.Tensor, epoch: Optional[int] = None
    ) -> tuple[MaskOutput, torch.Tensor]:
        """
        Run full target path → compress → latent masker.

        Returns (mask_output, full_tokens) so the caller can reuse full_tokens
        as tgt_tokens_all without a second encoder forward. The full target path
        sees the full image regardless of compressor mode — compression is
        applied to its output, not its input — so full_tokens is always (B, N, D)
        and is always valid as the target token bank.

        In the current v1 integration, learned maskers remain tied to the EMA
        target path; shared-target SIGReg runs only with deterministic maskers.
        The target encoder runs without gradients (frozen / EMA-updated).
        The compressor and latent masker run with gradients so their parameters
        receive signal from both the reconstruction loss (via aux_loss) and any
        additional terms the masker defines.
        """
        with torch.no_grad():
            if self.token_compressor.needs_cls:
                cls_token, full_tokens = self.target_encoder(images, return_cls=True)
            else:
                cls_token = None
                full_tokens = self.target_encoder(images)   # (B, N, D)

        # Compressor and latent masker are outside no_grad — grads flow normally.
        compressed = self.token_compressor(full_tokens, cls_token=cls_token)  # (B, M, D)
        forward_sig = inspect.signature(self.latent_masker.forward)
        latent_kwargs = {}
        if "epoch" in forward_sig.parameters:
            latent_kwargs["epoch"] = epoch
        if self.latent_masker.needs_full_tokens:
            mask_output = self.latent_masker(compressed, ema_full=full_tokens, **latent_kwargs)
        else:
            mask_output = self.latent_masker(compressed, **latent_kwargs)
        return mask_output, full_tokens

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
        _d_ctx: int,
    ):
        d_tgt = tgt_tokens_all.shape[-1]
        tgt_tokens = tgt_tokens_all.gather(
            1, tgt_idx.unsqueeze(-1).expand(-1, -1, d_tgt)
        )  # (B, Ntgt, D)

        result = self.predictor(
            ctx_tokens, ctx_idx=ctx_idx, tgt_idx=tgt_idx,
            return_ctx_pred=self.ctx_loss_enabled,
        )
        if self.ctx_loss_enabled:
            pred, pred_ctx = result
        else:
            pred, pred_ctx = result, None

        pred = self._project_prediction_tokens(pred)
        patch_loss = self.loss_fn(pred, tgt_tokens, reduction="none")  # (B, Ntgt)
        loss = patch_loss.mean()
        return loss, pred, tgt_tokens, patch_loss, pred_ctx

    def _flatten_multi_block_target_idx(
        self,
        tgt_idx: torch.Tensor,        # (B, M, K)
        batch_size: int,
        target_block_counts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if target_block_counts is None:
            return tgt_idx.reshape(batch_size, -1)

        counts = [int(x) for x in target_block_counts.detach().cpu().tolist()]
        parts = [tgt_idx[:, i, :counts[i]] for i in range(tgt_idx.shape[1]) if counts[i] > 0]
        if not parts:
            raise ValueError("Expected at least one valid target block when flattening targets.")
        return torch.cat(parts, dim=1)

    def _forward_multi_block(
        self,
        ctx_tokens: torch.Tensor,     # (B, Nctx, D)
        ctx_idx: torch.Tensor,        # (B, Nctx)
        tgt_idx: torch.Tensor,        # (B, M, K)
        tgt_tokens_all: torch.Tensor, # (B, N, D)
        b: int,
        _d_ctx: int,
        target_block_counts: torch.Tensor | None = None,
    ):
        m = tgt_idx.shape[1]
        k = tgt_idx.shape[2]
        d_tgt = tgt_tokens_all.shape[-1]
        counts = None
        if target_block_counts is not None:
            counts = [int(x) for x in target_block_counts.detach().cpu().tolist()]

        # -- Separate prediction: each block predicted independently -----------
        if not self.predict_blocks_jointly:
            preds_list = []
            tgts_list = []
            ploss_list = []
            max_k = k
            for i in range(m):
                k_i = counts[i] if counts is not None else k
                if k_i <= 0:
                    continue
                block_idx = tgt_idx[:, i, :k_i]  # (B, K_i)
                block_tgt = tgt_tokens_all.gather(
                    1, block_idx.unsqueeze(-1).expand(-1, -1, d_tgt)
                )  # (B, K_i, D)
                block_pred = self.predictor(
                    ctx_tokens, ctx_idx=ctx_idx, tgt_idx=block_idx
                )  # (B, K_i, D)
                block_pred = self._project_prediction_tokens(block_pred)
                block_ploss = self.loss_fn(block_pred, block_tgt, reduction="none")  # (B, K_i)

                pred_pad = block_pred.new_zeros((b, max_k, block_pred.shape[-1]))
                tgt_pad = block_tgt.new_zeros((b, max_k, d_tgt))
                pred_pad[:, :k_i] = block_pred
                tgt_pad[:, :k_i] = block_tgt
                preds_list.append(pred_pad)
                tgts_list.append(tgt_pad)
                ploss_list.append(block_ploss)

            pred = torch.stack(preds_list, dim=1)        # (B, M, K, D)
            tgt_tokens = torch.stack(tgts_list, dim=1)   # (B, M, K, D)
            patch_loss_cat = torch.cat(ploss_list, dim=1) # (B, M*K)
            loss = patch_loss_cat.mean()
            # ctx_loss not supported with separate prediction
            return loss, pred, tgt_tokens, patch_loss_cat, None

        # -- Joint prediction (default): all blocks concatenated ---------------
        tgt_idx_cat = self._flatten_multi_block_target_idx(
            tgt_idx, b, target_block_counts
        )

        tgt_tokens_cat = tgt_tokens_all.gather(
            1, tgt_idx_cat.unsqueeze(-1).expand(-1, -1, d_tgt)
        )  # (B, sum(K_i), D)

        result = self.predictor(
            ctx_tokens, ctx_idx=ctx_idx, tgt_idx=tgt_idx_cat,
            return_ctx_pred=self.ctx_loss_enabled,
        )
        if self.ctx_loss_enabled:
            pred_cat, pred_ctx = result
        else:
            pred_cat, pred_ctx = result, None

        pred_cat = self._project_prediction_tokens(pred_cat)
        patch_loss_cat = self.loss_fn(pred_cat, tgt_tokens_cat, reduction="none")  # (B, sum(K_i))
        if counts is None:
            pred = pred_cat.reshape(b, m, k, pred_cat.shape[-1])  # (B, M, K, D)
            tgt_tokens = tgt_tokens_cat.reshape(b, m, k, d_tgt)  # (B, M, K, D)
        else:
            pred = pred_cat.new_zeros((b, m, k, pred_cat.shape[-1]))
            tgt_tokens = tgt_tokens_cat.new_zeros((b, m, k, d_tgt))
            offset = 0
            for i in range(m):
                k_i = counts[i]
                if k_i <= 0:
                    continue
                pred[:, i, :k_i] = pred_cat[:, offset: offset + k_i]
                tgt_tokens[:, i, :k_i] = tgt_tokens_cat[:, offset: offset + k_i]
                offset += k_i
        loss = patch_loss_cat.mean()
        return loss, pred, tgt_tokens, patch_loss_cat, pred_ctx
