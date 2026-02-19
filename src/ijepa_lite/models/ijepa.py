from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ijepa_lite.models.ema import ema_update


class IJEPAModel(nn.Module):
    """
    i-JEPA model.

    Mask generation has been moved out of this class and into the DataLoader
    collate function (``IJEPACollate``).  Masks must therefore always be
    supplied via the ``masks`` argument to ``forward``; passing ``None`` is
    still accepted as a convenience (e.g. unit tests) but will raise if no
    ``mask_generator`` was provided at construction time.
    """

    def __init__(
        self,
        context_encoder: nn.Module,
        target_encoder: nn.Module,
        predictor: nn.Module,
        loss_fn: nn.Module,
        ema_momentum: float,
        # kept for backward-compat / unit-test convenience; not used in
        # normal training where masks arrive from the DataLoader.
        mask_generator=None,
    ):
        super().__init__()
        self.context_encoder = context_encoder
        self.target_encoder = target_encoder
        self.predictor = predictor
        self.loss_fn = loss_fn
        self.ema_momentum = float(ema_momentum)
        self._mask_generator = mask_generator  # fallback only

        for p in self.target_encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_target(self) -> None:
        ema_update(self.target_encoder, self.context_encoder, self.ema_momentum)

    def forward(
        self,
        images: torch.Tensor,
        masks: Optional[Dict[str, torch.Tensor]] = None,
        compute_agreement: bool = False,
    ) -> Dict[str, torch.Tensor]:
        # ------------------------------------------------------------------
        # Masks come pre-computed from IJEPACollate (CPU, DataLoader workers).
        # Fallback to on-device generation only for unit tests / debugging.
        # ------------------------------------------------------------------
        if masks is None:
            if self._mask_generator is None:
                raise ValueError(
                    "masks=None but no mask_generator was provided to IJEPAModel. "
                    "Either pass pre-computed masks or supply a mask_generator."
                )
            masks = self._mask_generator(
                batch_size=images.shape[0], device=images.device
            )

        ctx_idx = masks["context_idx"].to(images.device, non_blocking=True)  # (B, Nctx)
        tgt_idx = masks["target_idx"].to(
            images.device, non_blocking=True
        )  # (B, Ntgt) or (B, M, K)

        # mask stats
        mask_stats: Dict[str, float] = {"mask/nctx": float(ctx_idx.shape[1])}
        if tgt_idx.dim() == 2:
            mask_stats["mask/ntgt"] = float(tgt_idx.shape[1])
            mask_stats["mask/nblocks"] = 1.0
        elif tgt_idx.dim() == 3:
            m = tgt_idx.shape[1]
            k = tgt_idx.shape[2]
            mask_stats["mask/ntgt_per_block"] = float(k)
            mask_stats["mask/ntgt_total"] = float(m * k)
            mask_stats["mask/nblocks"] = float(m)

        # context encoder (masked)
        ctx_tokens = self.context_encoder(images, keep_idx=ctx_idx)  # (B, Nctx, D)
        b = images.shape[0]
        d = ctx_tokens.shape[-1]

        # target encoder (full image, no grad) + LN
        with torch.no_grad():
            tgt_tokens_all = self.target_encoder(images)  # (B, N, D)
            tgt_tokens_all = F.layer_norm(tgt_tokens_all, (tgt_tokens_all.shape[-1],))

        # optional encoder-agreement diagnostic
        # Re-uses already-computed tensors — no extra forward pass needed.
        ctx_tokens_all: Optional[torch.Tensor] = None
        tgt_at_ctx: Optional[torch.Tensor] = None
        if compute_agreement:
            ctx_tokens_all = ctx_tokens.detach()  # (B, Nctx, D)
            tgt_at_ctx = tgt_tokens_all.gather(
                1, ctx_idx.unsqueeze(-1).expand(-1, -1, d)
            ).detach()  # (B, Nctx, D)

        # single block: tgt_idx (B, Ntgt)
        if tgt_idx.dim() == 2:
            tgt_tokens = tgt_tokens_all.gather(
                1, tgt_idx.unsqueeze(-1).expand(-1, -1, d)
            )  # (B, Ntgt, D)

            pred = self.predictor(
                ctx_tokens, ctx_idx=ctx_idx, tgt_idx=tgt_idx
            )  # (B, Ntgt, D)
            loss = self.loss_fn(pred, tgt_tokens)

            out: Dict = {
                "loss": loss,
                "pred": pred.detach(),
                "target": tgt_tokens.detach(),
                "mask_stats": mask_stats,
            }
            if compute_agreement:
                out["ctx_tokens_all"] = ctx_tokens_all
                out["tgt_tokens_all"] = tgt_at_ctx
            return out

        # multiblock: tgt_idx (B, M, K)
        if tgt_idx.dim() == 3:
            m = tgt_idx.shape[1]
            k = tgt_idx.shape[2]
            tgt_idx_cat = tgt_idx.reshape(b, m * k)  # (B, M*K)

            tgt_tokens_cat = tgt_tokens_all.gather(
                1, tgt_idx_cat.unsqueeze(-1).expand(-1, -1, d)
            )  # (B, M*K, D)
            tgt_tokens = tgt_tokens_cat.reshape(b, m, k, d)  # (B, M, K, D)

            pred_cat = self.predictor(
                ctx_tokens, ctx_idx=ctx_idx, tgt_idx=tgt_idx_cat
            )  # (B, M*K, D)
            pred = pred_cat.reshape(b, m, k, d)  # (B, M, K, D)

            loss = self.loss_fn(pred, tgt_tokens)

            out: Dict = {
                "loss": loss,
                "pred": pred.detach(),
                "target": tgt_tokens.detach(),
                "mask_stats": mask_stats,
            }
            if compute_agreement:
                out["ctx_tokens_all"] = ctx_tokens_all
                out["tgt_tokens_all"] = tgt_at_ctx
            return out

        raise ValueError(
            f"Unsupported target_idx.dim()={tgt_idx.dim()}, expected 2 or 3."
        )
