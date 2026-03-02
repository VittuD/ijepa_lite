"""
MIRateMasker: learned masker with MI-rate + surprise objective.

Architecture
------------
Identical to RateDist3WayMasker:
  proj_in → [compressed_ctx | selection_queries] → TransformerEncoder → proj_score

Structural differences from RateDist3WayMasker:
  1. rates_proj takes a 2-vector [log_lam, log_alpha] instead of 4
  2. owns_loss=True: aux_loss() returns the full MI-rate objective

Objective (see mi_loss.py for full derivation)
----------------------------------------------
    total = reconstruction_loss
          - α · surprise_soft
          + λ · mi_rate

    mi_rate = H(Y|X) − H(Y)   (= −I(X;Y))

Collapse prevention
-------------------
H(Y|X) → 0 : MI rate reward is self-sharpening — minimising per-patch entropy
              directly drives decisive assignments without an external sharpener.
H(Y)   → 0 : penalised because mi_rate = H_cond − H_marg increases
surprise ↑  : −α · surprise rewards context that is semantically informative

proj_score is initialised with trunc_normal_(std=0.02) so that logits at step 0
are non-uniform (≈ std 0.28 for d=192), breaking the uniform fixed point where
∂H(Y|X)/∂logit = 0 and the entropy gradient is dead.
"""
from __future__ import annotations

import math
import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ijepa_lite.masking.base import LatentMasker, MaskOutput
from ijepa_lite.masking.registry import register
from ijepa_lite.losses.mi_loss import MIRateSurpriseLoss


# ---------------------------------------------------------------------------
# MIRateMasker
# ---------------------------------------------------------------------------

@register("mi_3way")
class MIRateMasker(LatentMasker):
    """
    MI-rate 3-way masker with (λ, α) Lagrange conditioning.

    Args
    ----
    dim           : Encoder embedding dim.
    predictor_dim : Internal transformer dim (matches Predictor).
    depth         : Transformer layers.
    num_heads     : Attention heads.
    mlp_ratio     : FFN expansion.
    dropout       : Dropout.
    num_patches   : N — total patch positions.
    lam_min       : Lower bound of LogUniform λ (MI-rate multiplier).
    lam_max       : Upper bound of LogUniform λ.
    alpha_min     : Lower bound of LogUniform α (surprise bonus multiplier).
    alpha_max     : Upper bound of LogUniform α.
    ntgt_min      : Hard floor on target count.
    nctx_min      : Hard floor on context count.
    base_kind     : Unused; kept for build.py kwarg filtering.
    normalize     : Unused; kept for build.py kwarg filtering.
    """

    owns_loss: bool = True
    needs_full_tokens: bool = True   # ijepa.py passes ema_full=(B, N, D)

    def __init__(
        self,
        dim: int,
        predictor_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        num_patches: int,
        lam_min: float = 1e-2,
        lam_max: float = 1.0,
        alpha_min: float = 0.01,
        alpha_max: float = 0.5,
        ntgt_min: int = 4,
        nctx_min: int = 1,
        h_floor: float = 0.1,          # entropy floor (nats); penalty kicks in below this
        floor_weight: float = 2.0,     # quadratic penalty weight
        lam_warmup_epochs: int = 50,   # epochs to grow λ sampling range to lam_max
        base_kind: str = "smooth_l1",  # unused; kept for build.py compatibility
        normalize: bool = False,       # unused; kept for build.py compatibility
        # ------------------------------------------------------------------
        # Normalised (s, r) scalarization — opt-in, default=False
        # ------------------------------------------------------------------
        normalize_scalarization: bool = False,  # False = current independent LogUniform
        s_min: float = 0.02,      # total strength lower bound (≈ lam_min + alpha_min)
        s_max: float = 1.5,       # total strength upper bound (≈ lam_max + alpha_max)
        ema_decay: float = 0.996,  # EMA momentum; matches encoder ema_momentum[0] default
        ema_init_mi: float = 0.3,       # initial EMA(|mi_rate|)  — typical at random init
        ema_init_surprise: float = 0.05,# initial EMA(|surprise|) — typical at random init
        # ------------------------------------------------------------------
        # Beta (s, r) scalarization — opt-in, default=False
        # Keeps scale/direction decoupling but uses Beta(a,a) for r instead
        # of Uniform, biasing toward balanced (λ≈α) configurations.
        # No EMA normalisation — open-loop, stable.
        # ------------------------------------------------------------------
        beta_scalarization: bool = False,
        beta_concentration: float = 2.0,  # a in Beta(a, a); higher = tighter around 0.5
        # ------------------------------------------------------------------
        # Coupled (s, r) scalarization via logit-normal ratio — opt-in, default=False
        # r = sigmoid(N(0, σ_r²)), concentrated near 0.5; σ_r controls spread.
        # ------------------------------------------------------------------
        coupled_scalarization: bool = False,
        ratio_logit_std: float = 1.0,   # σ_r; controls coupling tightness
    ) -> None:
        super().__init__()

        if lam_min <= 0:
            raise ValueError(f"lam_min must be > 0 for LogUniform, got {lam_min}")
        if lam_max < lam_min:
            raise ValueError(f"lam_max ({lam_max}) must be >= lam_min ({lam_min})")
        if alpha_min <= 0:
            raise ValueError(f"alpha_min must be > 0 for LogUniform, got {alpha_min}")
        if alpha_max < alpha_min:
            raise ValueError(f"alpha_max ({alpha_max}) must be >= alpha_min ({alpha_min})")
        if lam_min == lam_max:
            warnings.warn(
                f"lam_min == lam_max == {lam_min}: λ is fixed (no sampling). "
                "Pareto front exploration is disabled — intended for ablation only.",
                UserWarning, stacklevel=2,
            )
        if alpha_min == alpha_max:
            warnings.warn(
                f"alpha_min == alpha_max == {alpha_min}: α is fixed (no sampling). "
                "Pareto front exploration is disabled — intended for ablation only.",
                UserWarning, stacklevel=2,
            )

        self.num_patches = int(num_patches)
        self.lam_min   = float(lam_min)
        self.lam_max   = float(lam_max)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.ntgt_min  = max(1, int(ntgt_min))
        self.nctx_min  = max(1, int(nctx_min))
        self.h_floor = float(h_floor)
        self.floor_weight = float(floor_weight)
        self.lam_warmup_epochs = int(lam_warmup_epochs)

        self.normalize_scalarization = bool(normalize_scalarization)
        self.beta_scalarization  = bool(beta_scalarization)
        self.beta_concentration  = float(beta_concentration)
        self.coupled_scalarization = bool(coupled_scalarization)
        self.ratio_logit_std     = float(ratio_logit_std)
        self.s_min     = float(s_min)
        self.s_max     = float(s_max)
        self.ema_decay = float(ema_decay)

        # _progress in [0, 1]; initialised to 1.0 so unit tests (no set_progress call)
        # use the full lam range and are unaffected by the warmup logic.
        self.register_buffer("_progress", torch.tensor(1.0), persistent=False)
        # Running EMA of objective magnitudes — used by normalised scalarization.
        # persistent=True so they survive checkpoint save/load.
        self.register_buffer("_ema_mi_rate",  torch.tensor(float(ema_init_mi)),      persistent=True)
        self.register_buffer("_ema_surprise", torch.tensor(float(ema_init_surprise)), persistent=True)

        d = predictor_dim

        # ----------------------------------------------------------------
        # Transformer backbone — identical to RateDist3WayMasker
        # ----------------------------------------------------------------
        self.proj_in = nn.Linear(dim, d)

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d))
        self.selection_token = nn.Parameter(torch.zeros(1, 1, d))

        # (λ, α) conditioning — 2-vector [log_lam, log_alpha]
        self.rates_proj = nn.Linear(2, d)
        self.pos_rates_mlp = nn.Sequential(
            nn.LayerNorm(2 * d),
            nn.Linear(2 * d, d),
            nn.GELU(),
            nn.Linear(d, d),
        )

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

        # MI-rate + surprise loss (no learned parameters)
        self.mi_loss = MIRateSurpriseLoss(num_patches=num_patches)

        # ----------------------------------------------------------------
        # Initialisation
        # ----------------------------------------------------------------
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.selection_token, std=0.02)
        nn.init.normal_(self.rates_proj.weight, std=0.01)
        nn.init.zeros_(self.rates_proj.bias)
        # Zero-init the final linear of pos_rates_mlp so positional embeddings
        # start unperturbed by the rates conditioning.
        nn.init.zeros_(self.pos_rates_mlp[-1].weight)
        nn.init.zeros_(self.pos_rates_mlp[-1].bias)
        # Score head: trunc_normal init breaks the uniform fixed point where
        # ∂H(Y|X)/∂logit = 0 (dead gradient at exactly uniform distribution).
        nn.init.trunc_normal_(self.proj_score.weight, std=0.02)
        nn.init.zeros_(self.proj_score.bias)

    # ------------------------------------------------------------------
    # Warmup progress
    # ------------------------------------------------------------------

    def set_progress(self, fraction: float) -> None:
        """Update warmup progress. Call once per epoch.
        fraction = epoch / lam_warmup_epochs, clamped to [0, 1].
        """
        self._progress.fill_(max(0.0, min(1.0, float(fraction))))

    def set_ema_decay(self, m: float) -> None:
        """Sync the running-stats EMA momentum to the encoder EMA momentum.
        Call once per step, right after core.update_target().
        """
        self.ema_decay = float(m)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        tokens:   torch.Tensor,                    # (B, M, D)
        ema_full: Optional[torch.Tensor] = None,   # (B, N, D)
        rates:    Optional[torch.Tensor] = None,   # (B, 2) [lam, alpha]
    ) -> MaskOutput:
        """
        Args
        ----
        tokens   : (B, M, D) compressed EMA tokens — input to transformer.
        ema_full : (B, N, D) full EMA tokens — stored in aux for aux_loss.
                   When None (unit tests), surprise computation is skipped.
        rates    : (B, 2) tensor with columns [λ, α].
                   When provided, skips LogUniform sampling — intended for
                   downstream use where (λ, α) are learnable parameters.

        Returns MaskOutput with
        -----------------------
        context_idx          : (B, nctx)
        target_idx           : (B, ntgt)  ntgt ≥ ntgt_min
        context_soft         : (B, N)     p_ctx
        target_soft          : (B, N)     p_tgt
        aux["lambda"]        : (B,)
        aux["alpha"]         : (B,)
        aux["p_ign"]         : (B, N)     with grad — needed for H(Y|X) gradient
        aux["logits"]        : (B, N, 3)  detached — for diagnostics
        aux["ema_full"]      : (B, N, D)  — passed to aux_loss for surprise
        """
        B = tokens.shape[0]
        device = tokens.device

        # ----------------------------------------------------------------
        # (λ, α) — sample or use caller-provided rates
        # ----------------------------------------------------------------
        if rates is not None:
            lam   = rates[:, 0]
            alpha = rates[:, 1]
        elif self.beta_scalarization:
            # (s, r) reparametrization with Beta(a, a) ratio sampling.
            #
            # Same scale/direction decoupling as normalize_scalarization but
            # open-loop — no EMA feedback, no normalisation by objective magnitudes.
            # Beta(a, a) concentrates r around 0.5, biasing toward balanced
            # (λ ≈ α) configurations while still covering the full [0, 1] range.
            # Higher beta_concentration → tighter concentration around 0.5.
            p     = max(self._progress.item(), 1e-3)
            u_s   = torch.rand(1, device=device).item() ** (1.0 / p)
            s_val = math.exp(math.log(self.s_min) + u_s * math.log(self.s_max / self.s_min))
            a     = self.beta_concentration
            r_val = torch.distributions.Beta(a, a).sample().item()
            lam   = torch.tensor(s_val * r_val,       device=device).expand(B)
            alpha = torch.tensor(s_val * (1 - r_val), device=device).expand(B)
        elif self.coupled_scalarization:
            # (s, r) reparametrization with logit-normal ratio sampling.
            #
            # s ~ LogUniform(s_min, s_max)  — total regularisation strength
            # r = sigmoid(N(0, σ_r²))       — ratio, concentrated near 0.5
            # λ = s · r,  α = s · (1 − r)
            #
            # σ_r → 0: fixed ratio λ = α; σ_r = 1: r ∈ [0.27, 0.73] at 95%.
            # Open-loop, no EMA feedback, no normalisation.
            p     = max(self._progress.item(), 1e-3)
            u_s   = torch.rand(1, device=device).item() ** (1.0 / p)
            s_val = math.exp(math.log(self.s_min) + u_s * math.log(self.s_max / self.s_min))
            z     = torch.randn(1, device=device).item() * self.ratio_logit_std
            r_val = torch.sigmoid(torch.tensor(z)).item()
            lam   = torch.tensor(s_val * r_val,       device=device).expand(B)
            alpha = torch.tensor(s_val * (1 - r_val), device=device).expand(B)
        elif self.normalize_scalarization:
            # (s, r) reparametrization with normalised scalarization.
            #
            # Sample total strength s and allocation ratio r independently:
            #   λ_eff = s · r       / EMA(|mi_rate|)
            #   α_eff = s · (1 − r) / EMA(|surprise|)
            #
            # This ensures both terms are always on the same gradient scale
            # (λ_eff=1 ≈ MI term contributes as much as reconstruction).
            # The same temperature warmup is applied to s so early training
            # stays in the low-regularisation regime.
            p     = max(self._progress.item(), 1e-3)
            u_s   = torch.rand(1, device=device).item() ** (1.0 / p)
            s_val = math.exp(math.log(self.s_min) + u_s * math.log(self.s_max / self.s_min))
            r_val = torch.rand(1, device=device).item()   # Uniform(0, 1)

            m_mi   = max(self._ema_mi_rate.item(),  1e-6)
            m_surp = max(self._ema_surprise.item(), 1e-6)
            lam   = torch.tensor(s_val * r_val       / m_mi,   device=device).expand(B)
            alpha = torch.tensor(s_val * (1 - r_val) / m_surp, device=device).expand(B)
        else:
            # Temperature-biased sampling: u^(1/p) concentrates near lam_min when p≈0,
            # degrades to standard LogUniform when p=1 (end of warmup).
            p = max(self._progress.item(), 1e-3)          # clamp to avoid u^inf
            u = torch.rand(1, device=device).item()
            u_warmed = u ** (1.0 / p)
            log_lam_range = math.log(self.lam_max) - math.log(self.lam_min)
            lam_val = math.exp(math.log(self.lam_min) + u_warmed * log_lam_range)
            lam = torch.tensor(lam_val, device=device).expand(B)

            # α: standard LogUniform, no temperature (no collapse risk from surprise term)
            alpha = _sample_log_uniform(self.alpha_min, self.alpha_max, device).expand(B)

        # ----------------------------------------------------------------
        # Transformer input: compressed tokens
        # ----------------------------------------------------------------
        ctx = self.proj_in(tokens)   # (B, M, d)

        # ----------------------------------------------------------------
        # (λ, α)-conditioned positional embeddings
        # ----------------------------------------------------------------
        log_rates = torch.stack([lam.log(), alpha.log()], dim=-1)  # (B, 2)
        rates_embed = self.rates_proj(log_rates)                                # (B, d)
        rates_embed_exp = rates_embed.unsqueeze(1).expand(-1, self.num_patches, -1)
        pos_base = self.pos_embed.expand(B, -1, -1)                            # (B, N, d)
        pos_residual = self.pos_rates_mlp(
            torch.cat([pos_base, rates_embed_exp], dim=-1)
        )
        pos_embed_rates = pos_base + pos_residual                               # (B, N, d)

        # ----------------------------------------------------------------
        # Selection queries + transformer
        # ----------------------------------------------------------------
        queries = self.selection_token.expand(B, self.num_patches, -1) + pos_embed_rates
        seq = torch.cat([ctx, queries], dim=1)   # (B, M+N, d)
        out = self.norm(self.blocks(seq))
        query_out = out[:, -self.num_patches:]   # (B, N, d)

        # ----------------------------------------------------------------
        # 3-way soft assignments — H(Y|X) loss self-sharpens over training
        # ----------------------------------------------------------------
        logits = self.proj_score(query_out)   # (B, N, 3)
        soft   = F.softmax(logits, dim=-1)    # (B, N, 3)

        p_ctx = soft[..., 0]   # (B, N)
        p_tgt = soft[..., 1]   # (B, N)
        p_ign = soft[..., 2]   # (B, N)

        # ----------------------------------------------------------------
        # Hard counts — ntgt floored at ntgt_min
        # ----------------------------------------------------------------
        ntgt = max(self.ntgt_min, int(round(p_tgt.sum(dim=-1).mean().item())))
        nctx = max(self.nctx_min, int(round(p_ctx.sum(dim=-1).mean().item())))

        # Targets selected first; context from remaining positions
        _, tgt_idx = torch.topk(p_tgt, ntgt, dim=-1, sorted=False)
        p_ctx_masked = p_ctx.clone().scatter_(1, tgt_idx, 0.0)
        _, ctx_idx = torch.topk(p_ctx_masked, nctx, dim=-1, sorted=False)

        return MaskOutput(
            context_idx=ctx_idx,
            target_idx=tgt_idx,
            context_soft=p_ctx,
            target_soft=p_tgt,
            aux={
                "lambda":   lam,
                "alpha":    alpha,
                "p_ign":    p_ign,              # NOT detached — needed for gradient
                "logits":   logits.detach(),    # for diagnostics
                "ema_full": ema_full,           # (B, N, D) — used in aux_loss
            },
        )

    # ------------------------------------------------------------------
    # Auxiliary loss — MI-rate + surprise objective
    # ------------------------------------------------------------------

    def aux_loss(
        self,
        mask_output: MaskOutput,
        reconstruction_loss: torch.Tensor,
        patch_loss: Optional[torch.Tensor] = None,  # available but unused here
    ) -> torch.Tensor:
        """
        Returns reconstruction_loss − α·surprise_soft + λ·mi_rate
        as the complete training objective.
        """
        p_ctx    = mask_output.context_soft          # (B, N)
        p_tgt    = mask_output.target_soft           # (B, N)
        p_ign    = mask_output.aux["p_ign"]          # (B, N) — with grad
        lam      = mask_output.aux["lambda"]         # (B,)
        alpha    = mask_output.aux["alpha"]          # (B,)
        ema_full = mask_output.aux.get("ema_full")   # (B, N, D) or None

        if ema_full is None:
            # Unit test / fallback — MI rate term only, no surprise
            soft_3way = torch.stack([p_ctx, p_tgt, p_ign], dim=-1)
            H_cond = -(soft_3way * (soft_3way + 1e-8).log()).sum(-1).mean()
            p_bar  = soft_3way.mean(dim=1)
            H_marg = -(p_bar * (p_bar + 1e-8).log()).sum(-1).mean()
            mi_rate = H_cond - H_marg
            floor_penalty = self.floor_weight * F.relu(self.h_floor - H_cond).pow(2)
            if self.normalize_scalarization:
                decay = self.ema_decay
                self._ema_mi_rate.mul_(decay).add_(
                    abs(float(mi_rate.detach().item())) * (1.0 - decay)
                )
                # No surprise in this branch — EMA(|surprise|) left unchanged.
            mask_output.aux["surprise_mean"]       = 0.0
            mask_output.aux["mi_rate"]             = float(mi_rate.detach().item())
            mask_output.aux["entropy_conditional"] = float(H_cond.detach().item())
            mask_output.aux["entropy_marginal"]    = float(H_marg.detach().item())
            mask_output.aux["floor_penalty"]       = float(floor_penalty.detach().item())
            return reconstruction_loss + lam.mean() * mi_rate + floor_penalty

        total, surprise, mi_rate, H_cond, H_marg = self.mi_loss(
            reconstruction_loss=reconstruction_loss,
            p_ctx=p_ctx,
            p_tgt=p_tgt,
            p_ign=p_ign,
            ema_full=ema_full,
            lam=lam,
            alpha=alpha,
        )

        # H_cond returned by mi_loss is .detach()-ed (logging only).
        # Recompute with gradient for the floor penalty so the masker feels it.
        soft_3way = torch.stack([p_ctx, p_tgt, p_ign], dim=-1)   # (B, N, 3)
        H_cond_grad = -(soft_3way * (soft_3way + 1e-8).log()).sum(-1).mean()
        floor_penalty = self.floor_weight * F.relu(self.h_floor - H_cond_grad).pow(2)
        total = total + floor_penalty

        # Update running EMA stats used by normalised scalarization.
        if self.normalize_scalarization:
            decay = self.ema_decay
            self._ema_mi_rate.mul_(decay).add_(
                abs(float(mi_rate.item())) * (1.0 - decay)
            )
            self._ema_surprise.mul_(decay).add_(
                float(surprise.item()) * (1.0 - decay)
            )

        # Write back into aux for mask_diagnostics to pick up
        mask_output.aux["surprise_mean"]       = float(surprise.item())
        mask_output.aux["mi_rate"]             = float(mi_rate.item())
        mask_output.aux["entropy_conditional"] = float(H_cond.item())
        mask_output.aux["entropy_marginal"]    = float(H_marg.item())
        mask_output.aux["floor_penalty"]       = float(floor_penalty.detach().item())

        return total


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _sample_log_uniform(low: float, high: float, device: torch.device) -> torch.Tensor:
    log_val = math.log(low) + (math.log(high) - math.log(low)) * torch.rand(1, device=device)
    return log_val.exp()
