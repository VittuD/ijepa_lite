"""
MIRateMasker: learned masker with modular compositional loss.

Architecture
------------
  proj_in → [compressed_ctx | selection_queries] → TransformerEncoder → proj_score

The loss is a weighted sum of independently toggleable atomic terms
(see ``losses/terms.py`` and ``losses/composite.py``). Per-step term
weights are sampled from LogUniform distributions with optional warmup.

No (λ, α) conditioning — the transformer sees only positional embeddings
and compressed context tokens, keeping the masker architecture agnostic
to the loss weighting.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ijepa_lite.losses.composite import CompositeMaskerLoss
from ijepa_lite.masking.base import LatentMasker, MaskOutput
from ijepa_lite.masking.registry import register


@register("mi_3way")
class MIRateMasker(LatentMasker):
    """
    MI-rate 3-way masker with modular compositional loss.

    Args
    ----
    dim            : Encoder embedding dim.
    predictor_dim  : Internal transformer dim (matches Predictor).
    depth          : Transformer layers.
    num_heads      : Attention heads.
    mlp_ratio      : FFN expansion.
    dropout        : Dropout.
    num_patches    : N — total patch positions.
    terms          : Nested dict of term configs (see CompositeMaskerLoss).
    ntgt_min       : Hard floor on target count.
    nctx_min       : Hard floor on context count.
    warmup_epochs  : Epochs to grow weight sampling range to full.
    """

    owns_loss: bool = True
    needs_full_tokens: bool = True

    def __init__(
        self,
        dim: int,
        predictor_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        num_patches: int,
        terms: dict,
        ntgt_min: int = 4,
        nctx_min: int = 1,
        warmup_epochs: int = 0,
        # Unused — kept for build.py kwarg filtering
        base_kind: str = "smooth_l1",
        normalize: bool = False,
    ) -> None:
        super().__init__()

        self.num_patches = int(num_patches)
        self.ntgt_min = max(1, int(ntgt_min))
        self.nctx_min = max(1, int(nctx_min))
        self.warmup_epochs = int(warmup_epochs)

        # _progress in [0, 1]; initialised to 1.0 so unit tests use full range.
        self.register_buffer("_progress", torch.tensor(1.0), persistent=False)

        d = predictor_dim

        # ----------------------------------------------------------------
        # Transformer backbone
        # ----------------------------------------------------------------
        self.proj_in = nn.Linear(dim, d)

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d))
        self.selection_token = nn.Parameter(torch.zeros(1, 1, d))

        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=num_heads,
            dim_feedforward=int(d * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            layer, num_layers=depth, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d)

        # 3-way score head: d → [l_ctx, l_tgt, l_ign]
        self.proj_score = nn.Linear(d, 3)

        # Composite loss
        self.composite_loss = CompositeMaskerLoss(terms, num_patches)

        # ----------------------------------------------------------------
        # Initialisation
        # ----------------------------------------------------------------
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.selection_token, std=0.02)
        nn.init.trunc_normal_(self.proj_score.weight, std=0.02)
        nn.init.zeros_(self.proj_score.bias)

    # ------------------------------------------------------------------
    # Warmup progress
    # ------------------------------------------------------------------

    def set_progress(self, fraction: float) -> None:
        """Update warmup progress. Call once per epoch."""
        self._progress.fill_(max(0.0, min(1.0, float(fraction))))

    # ------------------------------------------------------------------
    # Weight sampling
    # ------------------------------------------------------------------

    def _sample_weights(self, device: torch.device) -> dict[str, float]:
        """Sample per-term weights from LogUniform with warmup."""
        p = max(self._progress.item(), 1e-3)
        weights: dict[str, float] = {}

        for name, (lo, hi) in self.composite_loss.weight_ranges.items():
            if lo == hi:
                weights[name] = lo
            else:
                # Temperature-biased LogUniform: u^(1/p) concentrates near lo
                u = torch.rand(1, device=device).item() ** (1.0 / p)
                val = math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
                weights[name] = val

        return weights

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        tokens: torch.Tensor,                     # (B, M, D)
        ema_full: Optional[torch.Tensor] = None,   # (B, N, D)
        rates: Optional[torch.Tensor] = None,      # unused, kept for interface compat
    ) -> MaskOutput:
        B = tokens.shape[0]

        # ----------------------------------------------------------------
        # Transformer input: compressed tokens
        # ----------------------------------------------------------------
        ctx = self.proj_in(tokens)   # (B, M, d)

        # ----------------------------------------------------------------
        # Selection queries + transformer (no rates conditioning)
        # ----------------------------------------------------------------
        queries = self.selection_token.expand(B, self.num_patches, -1) \
                  + self.pos_embed.expand(B, -1, -1)
        seq = torch.cat([ctx, queries], dim=1)   # (B, M+N, d)
        out = self.norm(self.blocks(seq))
        query_out = out[:, -self.num_patches:]   # (B, N, d)

        # ----------------------------------------------------------------
        # 3-way soft assignments
        # ----------------------------------------------------------------
        logits = self.proj_score(query_out)   # (B, N, 3)
        soft = F.softmax(logits, dim=-1)      # (B, N, 3)

        p_ctx = soft[..., 0]
        p_tgt = soft[..., 1]
        p_ign = soft[..., 2]

        # ----------------------------------------------------------------
        # Hard counts — ntgt floored at ntgt_min
        # ----------------------------------------------------------------
        ntgt = max(self.ntgt_min, int(round(p_tgt.sum(dim=-1).mean().item())))
        nctx = max(self.nctx_min, int(round(p_ctx.sum(dim=-1).mean().item())))

        _, tgt_idx = torch.topk(p_tgt, ntgt, dim=-1, sorted=False)
        p_ctx_masked = p_ctx.clone().scatter_(1, tgt_idx, 0.0)
        _, ctx_idx = torch.topk(p_ctx_masked, nctx, dim=-1, sorted=False)

        # Sample weights for this step
        weights = self._sample_weights(tokens.device)

        return MaskOutput(
            context_idx=ctx_idx,
            target_idx=tgt_idx,
            context_soft=p_ctx,
            target_soft=p_tgt,
            aux={
                "weights":  weights,
                "p_ign":    p_ign,
                "logits":   logits.detach(),
                "ema_full": ema_full,
            },
        )

    # ------------------------------------------------------------------
    # Auxiliary loss
    # ------------------------------------------------------------------

    def aux_loss(
        self,
        mask_output: MaskOutput,
        reconstruction_loss: torch.Tensor,
        patch_loss: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        p_ctx    = mask_output.context_soft
        p_tgt    = mask_output.target_soft
        p_ign    = mask_output.aux["p_ign"]
        weights  = mask_output.aux["weights"]
        ema_full = mask_output.aux.get("ema_full")

        if ema_full is None:
            # Unit test fallback — compute only entropy terms
            device = p_ctx.device
            ema_full = torch.zeros(
                p_ctx.shape[0], p_ctx.shape[1], 1, device=device,
            )

        total, logs = self.composite_loss(
            weights=weights,
            p_ctx=p_ctx,
            p_tgt=p_tgt,
            p_ign=p_ign,
            ema_full=ema_full,
        )

        # Write logs into aux for metrics.py to pick up
        mask_output.aux.update(logs)

        return reconstruction_loss + total
