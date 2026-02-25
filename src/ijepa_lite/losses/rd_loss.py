"""
Rate-distortion-surprise objective for the 3-way learned masker.

Objective
---------
    total = reconstruction_loss
          - α · surprise_soft
          + β · ign_tax
          + λ_ctx · N · R_ctx
          + λ_tgt · N · R_tgt

Where:

    ctx_centroid  = (Σᵢ p_ctx_i · EMA_i) / max(Σᵢ p_ctx_i, 1)   (B, D)
    BS_all_i      = ||EMA_i - ctx_centroid||²                       (B, N)

    surprise_soft = (Σᵢ p_tgt_i · BS_all_i) / max(Σᵢ p_tgt_i, 1)   (normalised expectation)
    prior_bs_i    = ||EMA_i - mean(EMA)||²   (B, N)
    ign_tax       = Σᵢ p_ign_i · prior_bs_i  (ignoring surprising patches costs β)
    R_ctx         = (1/N) Σᵢ p_ctx_i
    R_tgt         = (1/N) Σᵢ p_tgt_i

Gradient paths
--------------
surprise_soft  → p_tgt : Concrete relaxation — fully differentiable, no REINFORCE.
surprise_soft  → p_ctx : via ctx_centroid (clamped-sum normalisation, no ε-collapse).
ign_tax        → p_ign : β penalises assigning high p_ign to patches far from mean.
λ_ctx · R_ctx  → p_ctx : penalises using many context tokens.
λ_tgt · R_tgt  → p_tgt : FIRST positive gradient on logit_tgt — prevents tgt collapse.

Centroid stability
------------------
The denominator is clamped to max(Σ p_ctx, 1.0) instead of Σ p_ctx + ε.
When Σ p_ctx < 1, the floor activates and the gradient
    ∂centroid / ∂p_ctx_j = (EMA_j − centroid)
is larger than with a small-ε denominator, providing a genuine restoring force
away from the p_ctx → 0 collapse.

(λ_ctx, α, β, λ_tgt) are per-sample tensors (B,) sampled from independent
LogUniform distributions during pre-training.  At downstream time the caller can
pass a fixed / optimised rates tensor — no mode switch needed inside the masker.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class RateDistSurpriseLoss(nn.Module):
    """
    Post-selection rate-distortion-surprise objective. No learned parameters.

    Args
    ----
    num_patches : N — total patch positions. Used to normalise R into [0, 1].
    """

    def __init__(self, num_patches: int) -> None:
        super().__init__()
        self.num_patches = int(num_patches)

    def forward(
        self,
        reconstruction_loss: torch.Tensor,  # scalar — hard mean over K targets
        p_ctx:               torch.Tensor,  # (B, N)  soft context probs
        p_tgt:               torch.Tensor,  # (B, N)  soft target probs
        p_ign:               torch.Tensor,  # (B, N)  soft ignore probs
        ema_full:            torch.Tensor,  # (B, N, D) full EMA tokens
        lam:                 torch.Tensor,  # (B,)   λ_ctx per sample
        alpha:               torch.Tensor,  # (B,)   α per sample
        beta:                torch.Tensor,  # (B,)   β per sample
        lam_tgt:             torch.Tensor,  # (B,)   λ_tgt per sample
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        total         : scalar loss — the full objective
        surprise_soft : scalar — soft-weighted surprise (for logging)
        R_ctx         : scalar — expected context fraction ∈ [0, 1] (for logging)
        ign_rate      : scalar — soft-weighted ignore tax (for logging)
        """
        # ------------------------------------------------------------------
        # Soft context centroid — clamped-sum normalisation
        #
        # Denominator floor at 1.0 prevents gradient vanishing at p_ctx → 0.
        # When Σ p_ctx < 1, ∂centroid/∂p_ctx_j = (EMA_j − centroid) remains
        # meaningful (restoring force away from the degenerate collapse).
        # ------------------------------------------------------------------
        p_ctx_sum = p_ctx.sum(dim=1, keepdim=True).clamp(min=1.0)          # (B, 1)
        ctx_centroid = (p_ctx.unsqueeze(-1) * ema_full).sum(dim=1) / p_ctx_sum
        # ctx_centroid : (B, D)  — p_ctx_sum is (B,1), broadcasts correctly against (B,D)

        # ------------------------------------------------------------------
        # Bayesian surprise for every patch — fully differentiable
        #
        # BS_all_i = ||EMA_i - ctx_centroid||² averaged over D
        # Gradients flow through p_tgt (Concrete weights) and ctx_centroid.
        # ------------------------------------------------------------------
        BS_all = (ema_full - ctx_centroid.unsqueeze(1)).pow(2).mean(dim=-1)  # (B, N)
        p_tgt_sum = p_tgt.sum(dim=-1).clamp(min=1.0)                        # (B,)
        surprise_soft = ((p_tgt * BS_all).sum(dim=-1) / p_tgt_sum).mean()   # scalar

        # ------------------------------------------------------------------
        # Ignore tax — penalise assigning high p_ign to spatially surprising patches
        #
        # prior_bs_i = ||EMA_i - image_mean||² — auto-scaled to same EMA norm
        # as surprise_soft, so β has the same units as α.
        # ------------------------------------------------------------------
        prior_bs = (ema_full - ema_full.mean(dim=1, keepdim=True)).pow(2).mean(dim=-1)
        # prior_bs : (B, N)
        ign_tax = (p_ign * prior_bs).sum(dim=-1).mean()                       # scalar

        # ------------------------------------------------------------------
        # Rate terms:
        #   λ_ctx · N · R_ctx  penalises context tokens (∂/∂logit_ctx < 0)
        #   λ_tgt · N · R_tgt  penalises target tokens  (∂/∂logit_tgt > 0 — first
        #                       positive gradient on logit_tgt, prevents tgt collapse)
        # ------------------------------------------------------------------
        R_ctx = p_ctx.sum(dim=-1).mean() / self.num_patches
        R_tgt = p_tgt.sum(dim=-1).mean() / self.num_patches
        rate_term = (lam.mean() * self.num_patches * R_ctx
                     + lam_tgt.mean() * self.num_patches * R_tgt)

        # ------------------------------------------------------------------
        # Total: minimise reconstruction + rate + ign_tax, maximise surprise
        # ------------------------------------------------------------------
        total = (
            reconstruction_loss
            - alpha.mean() * surprise_soft
            + beta.mean()  * ign_tax
            + rate_term
        )

        return total, surprise_soft.detach(), R_ctx.detach(), ign_tax.detach()
