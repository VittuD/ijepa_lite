"""
Atomic masker loss terms for the compositional MI masker.

Each term is an ``nn.Module`` that receives the soft 3-way assignments
(p_ctx, p_tgt, p_ign) and EMA tokens, and returns
``(scalar_to_minimise, {log_key: value})``.

Terms are fully independent — no shared state, no cross-term coupling.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskerTerm(nn.Module):
    """Base class for atomic masker loss terms."""

    name: str  # registry key

    def forward(
        self,
        *,
        p_ctx: torch.Tensor,    # (B, N)
        p_tgt: torch.Tensor,    # (B, N)
        p_ign: torch.Tensor,    # (B, N)
        ema_full: torch.Tensor,  # (B, N, D)
        **kw,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        raise NotImplementedError


# ------------------------------------------------------------------
# H(Y|n) — conditional entropy (minimise → confident assignments)
# ------------------------------------------------------------------

class HCondTerm(MaskerTerm):
    name = "H_cond"

    def forward(self, *, p_ctx, p_tgt, p_ign, ema_full, **kw):
        soft = torch.stack([p_ctx, p_tgt, p_ign], dim=-1)      # (B, N, 3)
        H_cond = -(soft * (soft + 1e-8).log()).sum(-1).mean()   # scalar
        return H_cond, {"entropy_conditional": float(H_cond.detach().item())}


# ------------------------------------------------------------------
# −H(Y) — negative marginal entropy (minimise → maximise H(Y))
# ------------------------------------------------------------------

class NegHMargTerm(MaskerTerm):
    name = "neg_H_marg"

    def forward(self, *, p_ctx, p_tgt, p_ign, ema_full, **kw):
        soft = torch.stack([p_ctx, p_tgt, p_ign], dim=-1)      # (B, N, 3)
        p_bar = soft.mean(dim=1)                                # (B, 3)
        H_marg = -(p_bar * (p_bar + 1e-8).log()).sum(-1).mean()
        neg_H = -H_marg
        return neg_H, {"entropy_marginal": float(H_marg.detach().item())}


# ------------------------------------------------------------------
# −surprise — negative Bayesian surprise (minimise → maximise surprise)
# ------------------------------------------------------------------

class NegSurpriseTerm(MaskerTerm):
    name = "neg_surprise"

    def forward(self, *, p_ctx, p_tgt, p_ign, ema_full, **kw):
        # Soft context centroid blended with image mean (collapse-safe)
        image_mean = ema_full.mean(dim=1)                                    # (B, D)
        p_ctx_sum = p_ctx.sum(dim=1, keepdim=True)                           # (B, 1)
        ctx_weighted = (p_ctx.unsqueeze(-1) * ema_full).sum(dim=1)           # (B, D)
        virtual_w = (1.0 - p_ctx_sum).clamp(min=0.0)                        # (B, 1)
        ctx_centroid = (ctx_weighted + virtual_w * image_mean) \
                       / (p_ctx_sum + virtual_w).clamp(min=1e-6)             # (B, D)

        BS_all = (ema_full - ctx_centroid.unsqueeze(1)).pow(2).mean(-1)      # (B, N)
        p_tgt_sum = p_tgt.sum(dim=-1).clamp(min=1.0)                        # (B,)
        surprise = ((p_tgt * BS_all).sum(-1) / p_tgt_sum).mean()            # scalar

        return -surprise, {"surprise_mean": float(surprise.detach().item())}


# ------------------------------------------------------------------
# Floor penalty — ReLU(h_floor − H(Y|n))²
# ------------------------------------------------------------------

class FloorPenaltyTerm(MaskerTerm):
    name = "floor_penalty"

    def __init__(self, h_floor: float = 0.1):
        super().__init__()
        self.h_floor = float(h_floor)

    def forward(self, *, p_ctx, p_tgt, p_ign, ema_full, **kw):
        soft = torch.stack([p_ctx, p_tgt, p_ign], dim=-1)
        H_cond = -(soft * (soft + 1e-8).log()).sum(-1).mean()
        penalty = F.relu(self.h_floor - H_cond).pow(2)
        return penalty, {"floor_penalty": float(penalty.detach().item())}


# ------------------------------------------------------------------
# Ignore tax — Σ p_ign · prior_BS
# ------------------------------------------------------------------

class IgnoreTaxTerm(MaskerTerm):
    name = "ignore_tax"

    def forward(self, *, p_ctx, p_tgt, p_ign, ema_full, **kw):
        # prior BS = squared distance from image mean
        image_mean = ema_full.mean(dim=1, keepdim=True)             # (B, 1, D)
        prior_bs = (ema_full - image_mean).pow(2).mean(-1)          # (B, N)
        tax = (p_ign * prior_bs).sum(-1).mean()                     # scalar
        ign_rate = float(p_ign.detach().mean().item())
        return tax, {"ign_rate": ign_rate}


# ------------------------------------------------------------------
# Context rate — (1/N) Σ p_ctx
# ------------------------------------------------------------------

class ContextRateTerm(MaskerTerm):
    name = "context_rate"

    def __init__(self, num_patches: int = 1):
        super().__init__()
        self.num_patches = int(num_patches)

    def forward(self, *, p_ctx, p_tgt, p_ign, ema_full, **kw):
        R_ctx = p_ctx.mean()  # mean over batch and patches
        return R_ctx, {"R_ctx": float(R_ctx.detach().item())}


# ------------------------------------------------------------------
# Target rate — (1/N) Σ p_tgt
# ------------------------------------------------------------------

class TargetRateTerm(MaskerTerm):
    name = "target_rate"

    def __init__(self, num_patches: int = 1):
        super().__init__()
        self.num_patches = int(num_patches)

    def forward(self, *, p_ctx, p_tgt, p_ign, ema_full, **kw):
        R_tgt = p_tgt.mean()
        return R_tgt, {"R_tgt": float(R_tgt.detach().item())}


# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------

TERM_REGISTRY: dict[str, type[MaskerTerm]] = {
    "H_cond": HCondTerm,
    "neg_H_marg": NegHMargTerm,
    "neg_surprise": NegSurpriseTerm,
    "floor_penalty": FloorPenaltyTerm,
    "ignore_tax": IgnoreTaxTerm,
    "context_rate": ContextRateTerm,
    "target_rate": TargetRateTerm,
}
