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
# −centroid distance — ||μ_ctx − μ_tgt||² (minimise → maximise)
#
# Same as neg_surprise but without the within-target variance term.
# See bias-variance decomposition:
#   E_tgt[||z_i − μ_ctx||²] = ||μ_tgt − μ_ctx||² + Var_tgt
# ------------------------------------------------------------------

class NegCentroidDistTerm(MaskerTerm):
    name = "neg_centroid_dist"

    def forward(self, *, p_ctx, p_tgt, p_ign, ema_full, **kw):
        # Context centroid (collapse-safe, blended with image mean)
        image_mean = ema_full.mean(dim=1)                                    # (B, D)
        p_ctx_sum = p_ctx.sum(dim=1, keepdim=True)                           # (B, 1)
        ctx_weighted = (p_ctx.unsqueeze(-1) * ema_full).sum(dim=1)           # (B, D)
        virtual_w = (1.0 - p_ctx_sum).clamp(min=0.0)                        # (B, 1)
        ctx_centroid = (ctx_weighted + virtual_w * image_mean) \
                       / (p_ctx_sum + virtual_w).clamp(min=1e-6)             # (B, D)

        # Target centroid (same blending for symmetry)
        p_tgt_sum = p_tgt.sum(dim=1, keepdim=True)                           # (B, 1)
        tgt_weighted = (p_tgt.unsqueeze(-1) * ema_full).sum(dim=1)           # (B, D)
        virtual_w_tgt = (1.0 - p_tgt_sum).clamp(min=0.0)                    # (B, 1)
        tgt_centroid = (tgt_weighted + virtual_w_tgt * image_mean) \
                       / (p_tgt_sum + virtual_w_tgt).clamp(min=1e-6)         # (B, D)

        dist = (ctx_centroid - tgt_centroid).pow(2).mean(-1).mean()          # scalar

        return -dist, {"centroid_dist_mean": float(dist.detach().item())}


# ------------------------------------------------------------------
# Floor penalty — ReLU(h_floor − H(Y|n))²
# ------------------------------------------------------------------

class FloorPenaltyTerm(MaskerTerm):
    name = "floor_penalty"

    def __init__(self, h_floor: float = 0.1, num_tgt_blocks: int = 1):
        super().__init__()
        self.h_floor = float(h_floor)

    def forward(self, *, p_ctx, p_tgt, p_ign, ema_full, **kw):
        soft = kw.get("soft")  # (B, N, M+2) when N-way, None for 3-way
        if soft is None:
            soft = torch.stack([p_ctx, p_tgt, p_ign], dim=-1)  # (B, N, 3)
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
# N-way cross-target surprise
# ------------------------------------------------------------------

class NWayCrossSurpriseTerm(MaskerTerm):
    """Inter-target surprise: Σ_{k≠l} E_{tgt_k}[‖zᵢ − μ_{tgt_l}‖²].

    Only pushes target blocks apart — no ctx-vs-tgt terms (adversarial).
    """
    name = "nway_cross_surprise"

    def __init__(self, num_tgt_blocks: int = 4):
        super().__init__()
        self.M = int(num_tgt_blocks)

    def forward(self, *, p_ctx, p_tgt, p_ign, ema_full, **kw):
        soft = kw.get("soft")  # (B, N, M+2)
        if soft is None:
            return p_ctx.new_zeros(()), {}

        M = self.M
        D = ema_full.shape[-1]
        image_mean = ema_full.mean(dim=1)  # (B, D)

        # --- Vectorized centroids: one bmm instead of M weighted sums ---
        # p_tgts: (B, N, M) → bmm with (B, N, D) → (B, M, D)
        p_tgts = soft[..., 1:M+1]                                           # (B, N, M)
        p_tgts_sum = p_tgts.sum(dim=1)                                      # (B, M)
        weighted = torch.bmm(p_tgts.transpose(1, 2), ema_full)              # (B, M, D)
        virtual_w = (1.0 - p_tgts_sum).clamp(min=0.0).unsqueeze(-1)        # (B, M, 1)
        centroids = (weighted + virtual_w * image_mean.unsqueeze(1)) \
                    / (p_tgts_sum.unsqueeze(-1) + virtual_w).clamp(min=1e-6)  # (B, M, D)

        # --- Vectorized pairwise distances: one bmm instead of M*(M-1) dots ---
        # ‖zᵢ − μ_l‖² = mean_d(zᵢ²) − 2·mean_d(zᵢ·μ_l) + mean_d(μ_l²)
        norm_ema_sq = ema_full.pow(2).mean(-1)                              # (B, N)
        all_dots = torch.bmm(ema_full, centroids.transpose(1, 2)) / D      # (B, N, M)
        all_norm_c = centroids.pow(2).mean(-1).unsqueeze(1)                 # (B, 1, M)
        dist_sq_all = norm_ema_sq.unsqueeze(-1) - 2.0 * all_dots + all_norm_c  # (B, N, M)

        # S_total = Σ_{k≠l} E_k[dist_sq_l]
        #         = Σ_k E_k[Σ_l dist_sq_l − dist_sq_k]   (subtract self-pair)
        dist_cross = dist_sq_all.sum(-1, keepdim=True) - dist_sq_all        # (B, N, M)
        p_tgts_sum_c = p_tgts_sum.clamp(min=1.0)                           # (B, M)
        S_total = ((p_tgts * dist_cross).sum(1) / p_tgts_sum_c).mean(0).sum()

        return -S_total, {"cross_surprise_mean": float(S_total.detach().item())}


# ------------------------------------------------------------------
# N-way full cross-surprise (targets + context)
# ------------------------------------------------------------------

class NWayFullCrossSurpriseTerm(MaskerTerm):
    """Cross-surprise over all M+1 groups (ctx + M targets, ignoring ign).

    S = S_tgt_tgt + ctx_weight · S_ctx_tgt

    where S_tgt_tgt is the inter-target surprise (same as NWayCrossSurpriseTerm)
    and S_ctx_tgt includes ctx↔tgt_k cross-pairs in both directions.
    ctx_weight controls how much the ctx-vs-target terms contribute.
    """
    name = "nway_full_cross_surprise"

    def __init__(self, num_tgt_blocks: int = 4, ctx_weight: float = 1.0):
        super().__init__()
        self.M = int(num_tgt_blocks)
        self.ctx_weight = float(ctx_weight)

    def forward(self, *, p_ctx, p_tgt, p_ign, ema_full, **kw):
        soft = kw.get("soft")  # (B, N, M+2)
        if soft is None:
            return p_ctx.new_zeros(()), {}

        M = self.M
        D = ema_full.shape[-1]
        image_mean = ema_full.mean(dim=1)  # (B, D)

        # --- Vectorized centroids for ctx (idx 0) + M targets (idx 1..M) ---
        p_groups = soft[..., :M+1]                                           # (B, N, M+1)
        p_groups_sum = p_groups.sum(dim=1)                                   # (B, M+1)
        weighted = torch.bmm(p_groups.transpose(1, 2), ema_full)             # (B, M+1, D)
        virtual_w = (1.0 - p_groups_sum).clamp(min=0.0).unsqueeze(-1)       # (B, M+1, 1)
        centroids = (weighted + virtual_w * image_mean.unsqueeze(1)) \
                    / (p_groups_sum.unsqueeze(-1) + virtual_w).clamp(min=1e-6)  # (B, M+1, D)

        # --- Vectorized pairwise distances for all M+1 groups ---
        norm_ema_sq = ema_full.pow(2).mean(-1)                               # (B, N)
        all_dots = torch.bmm(ema_full, centroids.transpose(1, 2)) / D       # (B, N, M+1)
        all_norm_c = centroids.pow(2).mean(-1).unsqueeze(1)                  # (B, 1, M+1)
        dist_sq_all = norm_ema_sq.unsqueeze(-1) - 2.0 * all_dots + all_norm_c  # (B, N, M+1)

        # --- Inter-target surprise: pairs (k,l) both in {1..M} ---
        p_tgts = p_groups[..., 1:]                                           # (B, N, M)
        p_tgts_sum = p_groups_sum[:, 1:].clamp(min=1.0)                     # (B, M)
        dist_sq_tgts = dist_sq_all[..., 1:]                                  # (B, N, M)
        dist_cross_tgts = dist_sq_tgts.sum(-1, keepdim=True) - dist_sq_tgts  # (B, N, M)
        S_tgt = ((p_tgts * dist_cross_tgts).sum(1) / p_tgts_sum).mean(0).sum()

        # --- Ctx↔target surprise: pairs involving ctx (index 0) ---
        p_0 = p_groups[..., 0]                                               # (B, N)
        p_0_sum = p_groups_sum[:, 0].clamp(min=1.0)                          # (B,)
        # ctx → each tgt: E_ctx[dist(z, c_k)] for k in 1..M
        S_ctx_to_tgt = (p_0.unsqueeze(-1) * dist_sq_all[..., 1:]).sum(1) \
                       / p_0_sum.unsqueeze(-1)                               # (B, M)
        # each tgt → ctx: E_{tgt_k}[dist(z, c_ctx)] for k in 1..M
        S_tgt_to_ctx = (p_tgts * dist_sq_all[..., :1]).sum(1) \
                       / p_tgts_sum                                          # (B, M)
        S_ctx = (S_ctx_to_tgt + S_tgt_to_ctx).mean(0).sum()

        S_total = S_tgt + self.ctx_weight * S_ctx

        return -S_total, {
            "full_cross_surprise_mean": float(S_total.detach().item()),
            "cross_surprise_tgt": float(S_tgt.detach().item()),
            "cross_surprise_ctx": float(S_ctx.detach().item()),
        }


# ------------------------------------------------------------------
# N-way negative marginal entropy
# ------------------------------------------------------------------

class NWayNegHMargTerm(MaskerTerm):
    """−H(Y) over (M+2)-dim marginal distribution."""
    name = "nway_neg_H_marg"

    def __init__(self, num_tgt_blocks: int = 4):
        super().__init__()

    def forward(self, *, p_ctx, p_tgt, p_ign, ema_full, **kw):
        soft = kw.get("soft")  # (B, N, M+2)
        if soft is None:
            return p_ctx.new_zeros(()), {}

        p_bar = soft.mean(dim=1)  # (B, M+2)
        H_marg = -(p_bar * (p_bar + 1e-8).log()).sum(-1).mean()
        return -H_marg, {"nway_entropy_marginal": float(H_marg.detach().item())}


# ------------------------------------------------------------------
# Role-alive penalty — ReLU(p_min − p_c(n))² per role per patch
#
# Prevents role death: ensures every role maintains at least p_min
# mass on every patch, so gradients flow to all roles and the masker
# can reassign patches as the encoder evolves.
# ------------------------------------------------------------------

class RoleAliveTerm(MaskerTerm):
    name = "role_alive"

    def __init__(self, p_min: float = 0.02, num_tgt_blocks: int = 1):
        super().__init__()
        self.p_min = float(p_min)

    def forward(self, *, p_ctx, p_tgt, p_ign, ema_full, **kw):
        soft = kw.get("soft")  # (B, N, M+2) when N-way, None for 3-way
        if soft is None:
            soft = torch.stack([p_ctx, p_tgt, p_ign], dim=-1)  # (B, N, 3)
        # Per-role, per-patch: penalise any probability below p_min
        deficit = F.relu(self.p_min - soft)           # (B, N, C)
        penalty = deficit.pow(2).mean()                # scalar
        # Log the fraction of (patch, role) pairs that are below p_min
        dead_frac = float((soft.detach() < self.p_min).float().mean().item())
        return penalty, {
            "role_alive_penalty": float(penalty.detach().item()),
            "role_dead_frac": dead_frac,
        }


# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------
# KL-to-target marginal — KL(p_bar || q(k))
#
# Generalises nway_neg_H_marg.  The target distribution q(k) allocates
# mass k to context, (1-k)/(M+1) to each other role.  At k = 1/(M+2)
# the loss reduces to -H(p_bar) + const (pure entropy maximisation).
# k is provided by the masker via **kw and follows a warmup+cosine
# schedule across epochs.
# ------------------------------------------------------------------

class NWayKLMargTerm(MaskerTerm):
    """KL(p_bar || q(k)) over (M+2)-dim marginal distribution."""
    name = "nway_kl_marg"

    def __init__(self, num_tgt_blocks: int = 4):
        super().__init__()
        self.M = int(num_tgt_blocks)

    def forward(self, *, p_ctx, p_tgt, p_ign, ema_full, **kw):
        soft = kw.get("soft")           # (B, N, M+2)
        k = kw.get("k")                 # float or None
        if soft is None or k is None:
            return p_ctx.new_zeros(()), {}

        p_bar = soft.mean(dim=1)        # (B, M+2)

        # Target: ctx gets k, rest get (1-k)/(M+1)
        q = torch.full_like(p_bar, (1.0 - k) / (self.M + 1))
        q[:, 0] = k

        # KL(p_bar || q)
        kl = (p_bar * (p_bar.clamp(min=1e-8).log() - q.clamp(min=1e-8).log())).sum(-1)
        kl_mean = kl.mean()

        # Also log entropy for comparison
        H_marg = -(p_bar * (p_bar + 1e-8).log()).sum(-1).mean()

        return kl_mean, {
            "kl_marg": float(kl_mean.detach().item()),
            "nway_entropy_marginal": float(H_marg.detach().item()),
            "k_schedule": float(k),
        }


# ------------------------------------------------------------------

# ------------------------------------------------------------------
# Progressive KL marginal — KL(q̄ ‖ p̄) / KL(p̄ ‖ q̄) / sum
#
# p̄ is a scheduled uniform distribution over the currently active
# roles {ctx, tgt₁…tgt_n_active, ign}.  Inactive tgt roles receive
# inactive_eps mass so forward KL stays finite.
#
# n_active_tgt is passed per-step via **kw (set by MINWayMasker).
# global_step is passed per-step via **kw for transition blending.
# direction: "forward" = KL(q̄‖p̄), "reverse" = KL(p̄‖q̄), "sum" = both.
#
# Smooth transitions: when n_active_tgt increases, the target linearly
# blends from the current q_eff to the new target over transition_steps
# steps, avoiding the KL discontinuity from hard phase switches.
# ------------------------------------------------------------------

class NWayProgressiveKLTerm(MaskerTerm):
    """Scheduled-target KL for progressive role unlocking with smooth transitions."""
    name = "nway_progressive_kl"

    def __init__(
        self,
        num_tgt_blocks: int = 4,
        direction: str = "forward",
        inactive_eps: float = 1e-4,
        transition_steps: int = 0,
    ):
        super().__init__()
        self.M = int(num_tgt_blocks)
        assert direction in ("forward", "reverse", "sum"), \
            f"direction must be 'forward', 'reverse', or 'sum', got {direction!r}"
        self.direction = str(direction)
        self.inactive_eps = float(inactive_eps)
        # 0 (default): hard switch — no blending.
        # -1: auto — MINWayMasker sets this to the inter-phase gap on first set_step().
        # >0: explicit smooth transition over that many steps.
        self.transition_steps = int(transition_steps)

        # Transition state — persistent so mid-transition survives checkpoint/resume.
        # _transition_start = -1 means "not yet initialized".
        self.register_buffer("_q_from", torch.zeros(self.M + 2), persistent=True)
        self.register_buffer("_q_to", torch.zeros(self.M + 2), persistent=True)
        self.register_buffer(
            "_transition_start", torch.tensor(-1, dtype=torch.long), persistent=True
        )
        self.register_buffer(
            "_prev_n_active", torch.tensor(-1, dtype=torch.long), persistent=True
        )
        # n_active at the start of the current transition (for smooth logging)
        self.register_buffer(
            "_n_active_from", torch.tensor(0, dtype=torch.long), persistent=True
        )

    def _build_target(self, n_active: int) -> torch.Tensor:
        """Return (M+2,) uniform target distribution for n_active active tgt roles."""
        M = self.M
        n_inactive = M - n_active
        n_active_roles = n_active + 2  # ctx + n_active tgts + ign
        active_mass = (1.0 - self.inactive_eps * n_inactive) / n_active_roles
        q = torch.full((M + 2,), self.inactive_eps, dtype=torch.float32)
        q[0] = active_mass    # ctx
        q[-1] = active_mass   # ign
        for k in range(n_active):
            q[1 + k] = active_mass
        return q

    def _get_q_eff(self, global_step: int) -> tuple[torch.Tensor, float]:
        """Return (q_eff, alpha) blended target at global_step."""
        t_start = int(self._transition_start.item())
        if t_start < 0 or self.transition_steps <= 0:
            return self._q_to.clone(), 1.0
        alpha = float(min(1.0, max(0.0, (global_step - t_start) / self.transition_steps)))
        q_eff = (1.0 - alpha) * self._q_from + alpha * self._q_to
        return q_eff, alpha

    def reset_parameters(self) -> None:
        self._q_from.zero_()
        self._q_to.zero_()
        self._transition_start.fill_(-1)
        self._prev_n_active.fill_(-1)
        self._n_active_from.fill_(0)

    def forward(self, *, p_ctx, p_tgt, p_ign, ema_full, **kw):
        soft = kw.get("soft")              # (B, N, M+2)
        n_active = kw.get("n_active_tgt")  # int
        global_step = int(kw.get("global_step", 0))
        if soft is None or n_active is None:
            return p_ctx.new_zeros(()), {}

        n_active = int(n_active)
        prev_n = int(self._prev_n_active.item())

        # --- Detect phase change (or first call) ---
        if prev_n != n_active:
            device = self._q_from.device
            new_q = self._build_target(n_active).to(device)
            if self._transition_start.item() < 0:
                # First call ever: jump directly (no transition)
                self._q_from.copy_(new_q)
                self._q_to.copy_(new_q)
                self._transition_start.fill_(global_step)
                self._n_active_from.fill_(n_active)   # from == to → smooth value = n_active
            else:
                # Phase advanced: blend from current q_eff to new target
                current_q_eff, _ = self._get_q_eff(global_step)
                self._q_from.copy_(current_q_eff.to(device))
                self._q_to.copy_(new_q)
                self._transition_start.fill_(global_step)
                self._n_active_from.fill_(n_active)    # smooth starts at the just-unlocked count
            self._prev_n_active.fill_(n_active)

        p_bar = soft.mean(dim=1)  # (B, M+2)

        # Blended target distribution
        q_eff, alpha = self._get_q_eff(global_step)
        p_target = q_eff.to(p_bar.device).unsqueeze(0).expand_as(p_bar)

        EPS = 1e-8
        log_q = p_bar.clamp(min=EPS).log()
        log_p = p_target.clamp(min=EPS).log()

        # KL(q̄ ‖ p̄) — zero-forcing: penalises mass in inactive roles
        forward_kl = (p_bar * (log_q - log_p)).sum(-1).mean()
        # KL(p̄ ‖ q̄) — zero-avoiding: no penalty for mass in inactive roles
        reverse_kl = (p_target * (log_p - log_q)).sum(-1).mean()

        if self.direction == "forward":
            loss = forward_kl
        elif self.direction == "reverse":
            loss = reverse_kl
        else:  # sum
            loss = forward_kl + reverse_kl

        n_active_smooth = float(self._n_active_from.item()) + alpha

        return loss, {
            "prog_kl/forward": float(forward_kl.detach().item()),
            "prog_kl/reverse": float(reverse_kl.detach().item()),
            "prog_kl/loss": float(loss.detach().item()),
            "prog_kl/n_active_tgt": n_active_smooth,
            "prog_kl/transition_alpha": alpha,
        }


# ------------------------------------------------------------------

TERM_REGISTRY: dict[str, type[MaskerTerm]] = {
    "H_cond": HCondTerm,
    "neg_H_marg": NegHMargTerm,
    "neg_surprise": NegSurpriseTerm,
    "neg_centroid_dist": NegCentroidDistTerm,
    "floor_penalty": FloorPenaltyTerm,
    "ignore_tax": IgnoreTaxTerm,
    "context_rate": ContextRateTerm,
    "target_rate": TargetRateTerm,
    "nway_cross_surprise": NWayCrossSurpriseTerm,
    "nway_full_cross_surprise": NWayFullCrossSurpriseTerm,
    "nway_neg_H_marg": NWayNegHMargTerm,
    "nway_kl_marg": NWayKLMargTerm,
    "nway_floor_penalty": FloorPenaltyTerm,
    "role_alive": RoleAliveTerm,
    "nway_progressive_kl": NWayProgressiveKLTerm,
}
