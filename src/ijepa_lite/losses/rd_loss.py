"""
Rate-distortion-surprise objective for the 3-way learned masker.

Objective
---------
    total = reconstruction_loss
          - α · (surprise_direct + surprise_reinforce)
          + λ · N · R

Where:

    ctx_centroid       = Σᵢ (p_ctx_i / Σⱼ p_ctx_j) · EMA_i     (B, D)
    BS_k               = ||EMA_tgt_k - ctx_centroid||²            (B, K)

    surprise_direct    = mean(BS)
    surprise_reinforce = mean(BS.detach() · log(p_tgt at tgt_idx))
    R                  = (1/N) Σᵢ p_ctx_i

Gradient paths
--------------
surprise_direct   → p_ctx : masker learns to choose context whose centroid is
                             semantically distant from the selected targets.
                             (direct gradient through soft centroid computation)

surprise_reinforce → p_tgt : masker learns to assign high p_tgt to patches that
                             are far from the current context centroid.
                             (REINFORCE: reward = per-patch BS, policy = p_tgt)

reconstruction_loss → predictor + context encoder, as usual.

rate term → p_ctx : penalises using many context tokens.

Push-pull
---------
-α · surprise:  rewards selecting targets that context *cannot* explain
reconstruction_loss: requires predictor to explain those same targets
λ · N · R:      limits how much context is used

As the predictor improves on selected (ctx, tgt) pairs, reconstruction_loss
drops. The masker must find new (ctx, tgt) pairs where surprise is still high.
The curriculum emerges from the tension between these three terms.

Why not D_soft?
---------------
D_soft (soft-weighted reconstruction) collapsed because p_tgt → 0 makes
weights degenerate regardless of ntgt_min. reconstruction_loss uses the *hard*
mean over ntgt_min-floored targets and is always non-zero — it is structurally
collapse-proof.  D_soft is removed entirely.

Why this is better than the geometry cross-attention version
------------------------------------------------------------
The previous GeometryCrossAttention computed surprise BEFORE context was
decided, making it vacuous with full compressor (M=N → self-attention →
uniform weights → BS=0). This version computes surprise AFTER context
selection, so the centroid is always a strict subset of N patches regardless
of compressor mode. No new parameters are introduced.
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
    alpha       : Weight on the surprise bonus. Scales relative to
                  reconstruction_loss. Start at 0.05–0.1; increase if the
                  masker ignores the surprise signal.
    """

    def __init__(self, num_patches: int, alpha: float = 0.1) -> None:
        super().__init__()
        self.num_patches = int(num_patches)
        self.alpha = float(alpha)

    def forward(
        self,
        reconstruction_loss: torch.Tensor,  # scalar — hard mean over K targets
        p_ctx:               torch.Tensor,  # (B, N)  soft context probs
        p_tgt:               torch.Tensor,  # (B, N)  soft target probs
        tgt_idx:             torch.Tensor,  # (B, K)  hard target indices
        ema_full:            torch.Tensor,  # (B, N, D) full EMA tokens
        lam:                 torch.Tensor,  # (B,)   λ per sample
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        total    : scalar loss — the full objective
        BS_mean  : scalar — mean Bayesian surprise (for logging)
        R        : scalar — expected context fraction ∈ [0, 1] (for logging)
        """
        D = ema_full.shape[-1]

        # ------------------------------------------------------------------
        # Soft context centroid — differentiable w.r.t. p_ctx
        #
        # Normalise p_ctx so centroid is a convex combination of EMA tokens,
        # independent of the raw probability scale.
        # ------------------------------------------------------------------
        p_ctx_norm = p_ctx / (p_ctx.sum(dim=1, keepdim=True) + 1e-10)  # (B, N)
        ctx_centroid = (p_ctx_norm.unsqueeze(-1) * ema_full).sum(dim=1) # (B, D)

        # ------------------------------------------------------------------
        # Bayesian surprise at hard target positions
        #
        # BS_k = ||EMA_tgt_k - ctx_centroid||² averaged over D
        # High BS: target is far from what context represents — genuinely new info
        # ------------------------------------------------------------------
        ema_tgt = ema_full.gather(
            1, tgt_idx.unsqueeze(-1).expand(-1, -1, D)
        )  # (B, K, D)

        BS = (ema_tgt - ctx_centroid.unsqueeze(1)).pow(2).mean(dim=-1)  # (B, K)

        # ------------------------------------------------------------------
        # Gradient path 1: direct gradient to context selector
        #
        # ∂(BS.mean())/∂p_ctx flows through ctx_centroid:
        # moves centroid away from targets → rewards semantically distant context
        # ------------------------------------------------------------------
        surprise_direct = BS.mean()

        # ------------------------------------------------------------------
        # Gradient path 2: REINFORCE gradient to target selector
        #
        # reward = BS (detached — prevents double-counting with surprise_direct)
        # policy = p_tgt at hard target positions
        # Minimising -α * (BS.detach() * log_p_tgt).mean() maximises expected
        # surprise under the target selection policy.
        # ------------------------------------------------------------------
        log_p_tgt = (
            p_tgt.gather(1, tgt_idx).clamp(min=1e-10)
        ).log()  # (B, K)

        surprise_reinforce = (BS.detach() * log_p_tgt).mean()

        # ------------------------------------------------------------------
        # Rate term: λ · N · R, where R = expected context fraction
        # ------------------------------------------------------------------
        R = p_ctx.sum(dim=-1).mean() / self.num_patches
        rate_term = lam.mean() * self.num_patches * R

        # ------------------------------------------------------------------
        # Total: minimise reconstruction + rate, maximise surprise
        # ------------------------------------------------------------------
        total = (
            reconstruction_loss
            - self.alpha * (surprise_direct + surprise_reinforce)
            + rate_term
        )

        return total, BS.mean().detach(), R.detach()