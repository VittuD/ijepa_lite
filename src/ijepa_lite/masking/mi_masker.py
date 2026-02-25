"""
MIRateMasker: learned masker with MI-rate + surprise objective and STanH sharpening.

Architecture
------------
Identical to RateDist3WayMasker:
  proj_in → [compressed_ctx | selection_queries] → TransformerEncoder → proj_score

Structural differences from RateDist3WayMasker:
  1. Gumbel noise replaced by STanH sharpening (deterministic, λ-conditioned)
  2. rates_proj takes a 2-vector [log_lam, log_alpha] instead of 4
  3. owns_loss=True: aux_loss() returns the full MI-rate objective
  4. lambda_to_beta head maps log(λ) → β (steepness), so higher λ forces harder masks

Objective (see mi_loss.py for full derivation)
----------------------------------------------
    total = reconstruction_loss
          - α · surprise_soft
          + λ · mi_rate

    mi_rate = H(Y|X) − H(Y)   (= −I(X;Y))

Sharpening schedule
-------------------
β = softplus(lambda_to_beta(log λ)) + 1  ≥ 1

At λ → small: β ≈ 1.69 (softplus(0) + 1) — mild, near-linear STanH.
At λ → large: β grows — STanH approaches a staircase, assignments become hard.
The lambda_to_beta linear is zero-initialised, so β starts at ≈ 1.69 everywhere.

Collapse prevention
-------------------
H(Y|X) → 0 : MI rate reward (lower H_cond = more decisive assignments)
H(Y)   → 0 : penalised because mi_rate = H_cond − H_marg increases
surprise ↑  : −α · surprise rewards context that is semantically informative
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ijepa_lite.masking.base import LatentMasker, MaskOutput
from ijepa_lite.masking.registry import register
from ijepa_lite.losses.mi_loss import MIRateSurpriseLoss


# ---------------------------------------------------------------------------
# STanH wrapper
# ---------------------------------------------------------------------------

class StanHSharpener(nn.Module):
    """Element-wise STanH applied to any-shape tensor (broadcasting on last dim).

    Wraps NonLinearStanh from the STanH package (EIDOSLAB) and re-applies the
    learned bias points b and weights w to an arbitrary-shape input tensor,
    avoiding the dim=1 convention of the original forward.

    Args
    ----
    beta_init    : Initial steepness (may be overridden per-call by the β arg).
    num_sigmoids : Number of sigmoid / tanh components (= quantisation levels).
    extrema      : Range [−extrema, +extrema] for the learnable bias points b.
    """

    def __init__(
        self,
        beta_init: float,
        num_sigmoids: int,
        extrema: float = 5.0,
    ) -> None:
        super().__init__()
        from compress.quantization.activation import NonLinearStanh  # type: ignore[import]
        self._stanh = NonLinearStanh(
            beta=beta_init,
            num_sigmoids=num_sigmoids,
            extrema=extrema,
        )

    def forward(self, x: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        x    : (...) float tensor — logits to sharpen.
        beta : scalar tensor ≥ 1 — steepness conditioned on λ.

        Returns
        -------
        (...) tensor — sharpened logits, same shape as x.
        """
        b = torch.sort(self._stanh.b)[0].to(x.device)   # (S,)
        w = self._stanh.w.to(x.device)                   # (S,)
        # x.unsqueeze(-1): (..., 1); b: (S,) → diff: (..., S)
        diff = x.unsqueeze(-1) - b                        # (..., S)
        f    = 2.0 * torch.sigmoid(2.0 * beta * diff) - 1.0  # (..., S)
        return (w / 2.0 * f).sum(-1)                      # (...) same shape as x


# ---------------------------------------------------------------------------
# MIRateMasker
# ---------------------------------------------------------------------------

@register("mi_3way")
class MIRateMasker(LatentMasker):
    """
    MI-rate 3-way masker with STanH sharpening and (λ, α) Lagrange conditioning.

    Args
    ----
    dim                : Encoder embedding dim.
    predictor_dim      : Internal transformer dim (matches Predictor).
    depth              : Transformer layers.
    num_heads          : Attention heads.
    mlp_ratio          : FFN expansion.
    dropout            : Dropout.
    num_patches        : N — total patch positions.
    lam_min            : Lower bound of LogUniform λ (MI-rate multiplier).
    lam_max            : Upper bound of LogUniform λ.
    alpha_min          : Lower bound of LogUniform α (surprise bonus multiplier).
    alpha_max          : Upper bound of LogUniform α.
    ntgt_min           : Hard floor on target count.
    stanh_num_sigmoids : Number of STanH sigmoid components.
    stanh_extrema      : Bias point range [−extrema, +extrema].
    stanh_beta_init    : Initial steepness passed to NonLinearStanh constructor.
    base_kind          : Unused; kept for build.py kwarg filtering.
    normalize          : Unused; kept for build.py kwarg filtering.
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
        stanh_num_sigmoids: int = 5,
        stanh_extrema: float = 5.0,
        stanh_beta_init: float = 1.0,
        base_kind: str = "smooth_l1",  # unused; kept for build.py compatibility
        normalize: bool = False,       # unused; kept for build.py compatibility
    ) -> None:
        super().__init__()

        if lam_min <= 0:
            raise ValueError(f"lam_min must be > 0 for LogUniform, got {lam_min}")
        if lam_max <= lam_min:
            raise ValueError(f"lam_max ({lam_max}) must be > lam_min ({lam_min})")
        if alpha_min <= 0:
            raise ValueError(f"alpha_min must be > 0 for LogUniform, got {alpha_min}")
        if alpha_max <= alpha_min:
            raise ValueError(f"alpha_max ({alpha_max}) must be > alpha_min ({alpha_min})")

        self.num_patches = int(num_patches)
        self.lam_min   = float(lam_min)
        self.lam_max   = float(lam_max)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.ntgt_min  = max(1, int(ntgt_min))

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

        # STanH sharpener (has learned b and w parameters)
        self.stanh_sharpener = StanHSharpener(
            beta_init=stanh_beta_init,
            num_sigmoids=stanh_num_sigmoids,
            extrema=stanh_extrema,
        )

        # Maps log(λ) → log(β − 1 + ε) so that β = softplus(·) + 1 ≥ 1
        self.lambda_to_beta = nn.Linear(1, 1)
        nn.init.zeros_(self.lambda_to_beta.weight)
        nn.init.zeros_(self.lambda_to_beta.bias)
        # At init: softplus(0) + 1.0 ≈ 1.69 — mild sharpening, close to linear

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
        # Score head: near-zero → soft-uniform 3-way distribution at init
        nn.init.zeros_(self.proj_score.weight)
        nn.init.zeros_(self.proj_score.bias)

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
        aux["steepness"]     : scalar     β used for sharpening — for diagnostics
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
        else:
            lam   = _sample_log_uniform(self.lam_min,   self.lam_max,   device).expand(B)
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
        # STanH sharpening (replaces Gumbel-softmax)
        # ----------------------------------------------------------------
        logits = self.proj_score(query_out)   # (B, N, 3)

        # Compute β from λ (batch-level scalar — lam is same for all B elements)
        log_lam = lam[0:1].log()                                                # (1,)
        beta = F.softplus(self.lambda_to_beta(log_lam.unsqueeze(0))) + 1.0     # (1, 1) scalar ≥ 1

        sharpened = self.stanh_sharpener(logits, beta=beta)    # (B, N, 3)
        soft      = F.softmax(sharpened, dim=-1)               # (B, N, 3)

        p_ctx = soft[..., 0]   # (B, N)
        p_tgt = soft[..., 1]   # (B, N)
        p_ign = soft[..., 2]   # (B, N)

        # ----------------------------------------------------------------
        # Hard counts — ntgt floored at ntgt_min
        # ----------------------------------------------------------------
        ntgt = max(self.ntgt_min, int(round(p_tgt.sum(dim=-1).mean().item())))
        nctx = max(1, int(round(p_ctx.sum(dim=-1).mean().item())))

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
                "lambda":    lam,
                "alpha":     alpha,
                "p_ign":     p_ign,                       # NOT detached — needed for gradient
                "logits":    logits.detach(),              # pre-sharpening, for diagnostics
                "steepness": beta.detach().squeeze(),      # scalar — for diagnostics
                "ema_full":  ema_full,                     # (B, N, D) — used in aux_loss
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
            mask_output.aux["surprise_mean"]       = 0.0
            mask_output.aux["mi_rate"]             = float(mi_rate.detach().item())
            mask_output.aux["entropy_conditional"] = float(H_cond.detach().item())
            mask_output.aux["entropy_marginal"]    = float(H_marg.detach().item())
            return reconstruction_loss + lam.mean() * mi_rate

        total, surprise, mi_rate, H_cond, H_marg = self.mi_loss(
            reconstruction_loss=reconstruction_loss,
            p_ctx=p_ctx,
            p_tgt=p_tgt,
            p_ign=p_ign,
            ema_full=ema_full,
            lam=lam,
            alpha=alpha,
        )

        # Write back into aux for mask_diagnostics to pick up
        mask_output.aux["surprise_mean"]       = float(surprise.item())
        mask_output.aux["mi_rate"]             = float(mi_rate.item())
        mask_output.aux["entropy_conditional"] = float(H_cond.item())
        mask_output.aux["entropy_marginal"]    = float(H_marg.item())

        return total


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _sample_log_uniform(low: float, high: float, device: torch.device) -> torch.Tensor:
    log_val = math.log(low) + (math.log(high) - math.log(low)) * torch.rand(1, device=device)
    return log_val.exp()
