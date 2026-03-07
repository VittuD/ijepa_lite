"""
MI-rate surprise objective for the 3-way learned masker.

Objective
---------
    total = reconstruction_loss
          - α · surprise_soft
          + λ · mi_rate

Where:

    ctx_centroid  = (Σᵢ p_ctx_i · EMA_i  +  (1 − Σ p_ctx).clamp(0) · image_mean)
                   / (Σ p_ctx + (1 − Σ p_ctx).clamp(0))   (B, D)
    BS_all_i      = ||EMA_i - ctx_centroid||²              (B, N)
    surprise_soft = (Σᵢ p_tgt_i · BS_all_i) / max(Σᵢ p_tgt_i, 1)   (scalar)

    soft_3way     = [p_ctx, p_tgt, p_ign]                  (B, N, 3)
    H(Y|n)        = mean per-patch categorical entropy      ∈ [0, log3]
    H(Y)          = entropy of the mean role distribution   ∈ [0, log3]
    mi_rate       = H(Y|n) − H(Y)                          = −I(n; Y)

Where n ~ Uniform({1,...,N}) is a randomly sampled patch position and Y its assigned
role. I(n; Y) is MI between patch *position* and role within a single image — NOT
between patch *content* and role. A positional masker achieves the same I(n;Y) as a
content-adaptive one. mi_rate is a confidence + balance regularizer:
  H(Y|n) → 0   : confident per-patch assignments (self-sharpening)
  H(Y)   → log3 : balanced role usage across patches (collapse prevention)
Content-adaptivity is provided by the surprise term, not mi_rate.

Gradient paths
--------------
surprise_soft → p_tgt : Concrete relaxation — fully differentiable.
surprise_soft → p_ctx : via ctx_centroid (blended with image mean, collapse-safe).
mi_rate       → soft  : penalises high per-patch entropy (encourages hard assignments)
                         and rewards uniform marginal (role balance).

λ and α are per-sample (B,) tensors sampled from LogUniform during pre-training.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class MIRateSurpriseLoss(nn.Module):
    """
    MI-rate + semantic surprise objective. No learned parameters.

    Args
    ----
    num_patches : N — total patch positions (unused in forward but kept for
                  interface parity with RateDistSurpriseLoss).
    """

    def __init__(self, num_patches: int) -> None:
        super().__init__()
        self.num_patches = int(num_patches)

    def forward(
        self,
        reconstruction_loss: torch.Tensor,  # scalar
        p_ctx:               torch.Tensor,  # (B, N)
        p_tgt:               torch.Tensor,  # (B, N)
        p_ign:               torch.Tensor,  # (B, N)
        ema_full:            torch.Tensor,  # (B, N, D)
        lam:                 torch.Tensor,  # (B,)
        alpha:               torch.Tensor,  # (B,)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        total         : scalar loss — the full objective
        surprise_soft : scalar — soft-weighted surprise (for logging)
        mi_rate       : scalar — H(Y|X) − H(Y) (for logging)
        H_cond        : scalar — H(Y|X) conditional entropy (for logging)
        H_marg        : scalar — H(Y) marginal entropy (for logging)
        """
        # ------------------------------------------------------------------
        # Soft context centroid — blended with image mean (collapse-safe)
        # ------------------------------------------------------------------
        image_mean   = ema_full.mean(dim=1)                                   # (B, D)
        p_ctx_sum    = p_ctx.sum(dim=1, keepdim=True)                         # (B, 1)
        ctx_weighted = (p_ctx.unsqueeze(-1) * ema_full).sum(dim=1)            # (B, D)
        virtual_w    = (1.0 - p_ctx_sum).clamp(min=0.0)                      # (B, 1)
        ctx_centroid = (ctx_weighted + virtual_w * image_mean) \
                       / (p_ctx_sum + virtual_w).clamp(min=1e-6)              # (B, D)

        # ------------------------------------------------------------------
        # Bayesian surprise — fully differentiable via Concrete weights
        # ------------------------------------------------------------------
        BS_all        = (ema_full - ctx_centroid.unsqueeze(1)).pow(2).mean(-1) # (B, N)
        p_tgt_sum     = p_tgt.sum(dim=-1).clamp(min=1.0)                      # (B,)
        surprise_soft = ((p_tgt * BS_all).sum(-1) / p_tgt_sum).mean()         # scalar

        # ------------------------------------------------------------------
        # MI rate = H(Y|n) − H(Y) = −I(n; Y)
        # n = patch position (uniform RV over {1,...,N}), Y = role ∈ {ctx,tgt,ign}.
        # Minimising mi_rate maximises I(n;Y) within each image:
        #   H(Y|n) → 0   : confident per-patch assignments (self-sharpening)
        #   H(Y)   → log3 : balanced role usage (collapse prevention)
        # NOTE: I(n;Y) is MI between position and role, not content and role.
        # ------------------------------------------------------------------
        soft_3way = torch.stack([p_ctx, p_tgt, p_ign], dim=-1)                # (B, N, 3)

        # H(Y|n): mean per-patch categorical entropy — minimising this
        # encourages confident (hard) per-patch role assignments.
        H_cond = -(soft_3way * (soft_3way + 1e-8).log()).sum(-1).mean()       # scalar

        # H(Y): entropy of the mean role distribution (marginal over positions) —
        # maximising this prevents degenerate role collapse (all-ign, all-ctx, etc.).
        p_bar  = soft_3way.mean(dim=1)                                        # (B, 3)
        H_marg = -(p_bar * (p_bar + 1e-8).log()).sum(-1).mean()               # scalar

        mi_rate = H_cond - H_marg                                             # scalar = −I(n;Y)

        # ------------------------------------------------------------------
        # Total objective
        # ------------------------------------------------------------------
        total = (
            reconstruction_loss
            - alpha.mean() * surprise_soft
            + lam.mean()   * mi_rate
        )

        return (
            total,
            surprise_soft.detach(),
            mi_rate.detach(),
            H_cond.detach(),
            H_marg.detach(),
        )
