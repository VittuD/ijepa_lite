"""
RateDist3WayMasker: learned masker with post-selection rate-distortion-surprise
objective and (λ_ctx, α, β, λ_tgt) Lagrange-multiplier conditioning.

Architecture
------------
Identical to PredictorBasedMasker:
  proj_in → [compressed_ctx | selection_queries] → TransformerEncoder → proj_score

Structural differences from PredictorBasedMasker:
  1. proj_score outputs 3 logits (ctx / tgt / ignore) instead of 1
  2. Positional embeddings are conditioned on log(λ_ctx, α, β, λ_tgt) as a 4-vector
  3. owns_loss=True: aux_loss() returns the full objective
  4. Optional `rates: (B, 4)` kwarg — skip internal sampling at downstream time

Objective (see rd_loss.py for full derivation)
----------------------------------------------
    total = reconstruction_loss
          - α · surprise_soft
          + β · ign_tax
          + λ_ctx · N · R_ctx
          + λ_tgt · N · R_tgt

    surprise_soft = Σᵢ p_tgt_i · ||EMA_i − ctx_centroid||²   (Concrete, no REINFORCE)
    ign_tax       = Σᵢ p_ign_i · ||EMA_i − image_mean||²
    ctx_centroid  = (Σᵢ p_ctx_i · EMA_i) / max(Σᵢ p_ctx_i, 1)
    R_ctx         = (1/N) Σᵢ p_ctx_i
    R_tgt         = (1/N) Σᵢ p_tgt_i

Pre-training
------------
(λ_ctx, α, β, λ_tgt) are sampled once per batch from independent LogUniform
distributions.  The masker learns a 4-D R-D surface indexed by the multipliers.

Downstream use
--------------
Pass `rates: (B, 4)` to forward — the masker uses those values as (λ_ctx, α, β, λ_tgt)
directly, enabling the caller to optimise them as nn.Parameter without any
mode switch inside the masker.

Collapse prevention
-------------------
p_ctx → 1   : λ_ctx · R_ctx penalises
p_ctx → 0   : clamped-sum centroid (denominator floor at 1); non-zero surprise gradient
p_tgt → 1   : λ_tgt · R_tgt penalises (first positive gradient on logit_tgt)
p_ign → 1   : β · ign_tax penalises; prior_bs always positive
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ijepa_lite.masking.base import LatentMasker, MaskOutput
from ijepa_lite.masking.registry import register
from ijepa_lite.losses.rd_loss import RateDistSurpriseLoss


@register("rd_3way")
class RateDist3WayMasker(LatentMasker):
    """
    Rate-distortion 3-way masker with post-selection Bayesian surprise and
    (λ, α, β) Lagrange-multiplier conditioning.

    Args
    ----
    dim           : Encoder embedding dim.
    predictor_dim : Internal transformer dim (matches Predictor).
    depth         : Transformer layers.
    num_heads     : Attention heads.
    mlp_ratio     : FFN expansion.
    dropout       : Dropout.
    num_patches   : N — total patch positions.
    temperature   : Gumbel temperature for 3-way categorical.
    lam_min       : Lower bound of LogUniform λ_ctx (context rate multiplier).
    lam_max       : Upper bound of LogUniform λ_ctx.
    alpha_min     : Lower bound of LogUniform α (surprise bonus multiplier).
    alpha_max     : Upper bound of LogUniform α.
    beta_min      : Lower bound of LogUniform β (ignore tax multiplier).
    beta_max      : Upper bound of LogUniform β.
    lam_tgt_min   : Lower bound of LogUniform λ_tgt (target rate multiplier).
    lam_tgt_max   : Upper bound of LogUniform λ_tgt.
    ntgt_min      : Hard floor on target count.
    base_kind     : Distortion function used by VanillaTokenLoss (unused here
                    but kept for registry signature consistency).
    normalize     : Normalise pred/target (unused here, kept for consistency).
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
        temperature: float = 1.0,
        lam_min: float = 1e-3,
        lam_max: float = 1.0,
        alpha_min: float = 0.01,
        alpha_max: float = 0.5,
        beta_min: float = 0.01,
        beta_max: float = 0.5,
        lam_tgt_min: float = 1e-3,
        lam_tgt_max: float = 0.1,
        ntgt_min: int = 4,
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
        if beta_min <= 0:
            raise ValueError(f"beta_min must be > 0 for LogUniform, got {beta_min}")
        if beta_max <= beta_min:
            raise ValueError(f"beta_max ({beta_max}) must be > beta_min ({beta_min})")
        if lam_tgt_min <= 0:
            raise ValueError(f"lam_tgt_min must be > 0 for LogUniform, got {lam_tgt_min}")
        if lam_tgt_max <= lam_tgt_min:
            raise ValueError(f"lam_tgt_max ({lam_tgt_max}) must be > lam_tgt_min ({lam_tgt_min})")

        self.num_patches = int(num_patches)
        self.temperature = float(temperature)
        self.lam_min   = float(lam_min)
        self.lam_max   = float(lam_max)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.beta_min    = float(beta_min)
        self.beta_max    = float(beta_max)
        self.lam_tgt_min = float(lam_tgt_min)
        self.lam_tgt_max = float(lam_tgt_max)
        self.ntgt_min    = max(1, int(ntgt_min))

        d = predictor_dim

        # ----------------------------------------------------------------
        # Transformer backbone — identical to PredictorBasedMasker
        # ----------------------------------------------------------------
        self.proj_in = nn.Linear(dim, d)

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d))
        self.selection_token = nn.Parameter(torch.zeros(1, 1, d))

        # (λ_ctx, α, β, λ_tgt) conditioning — takes log(λ_ctx, α, β, λ_tgt) as a 4-vector
        self.rates_proj = nn.Linear(4, d)
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

        # Surprise-based RD loss (no learned parameters)
        self.rd_loss = RateDistSurpriseLoss(num_patches=num_patches)

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
        rates:    Optional[torch.Tensor] = None,   # (B, 4) [lam_ctx, alpha, beta, lam_tgt]
    ) -> MaskOutput:
        """
        Args:
            tokens   : (B, M, D) compressed EMA tokens — input to transformer.
            ema_full : (B, N, D) full EMA tokens — stored in aux for aux_loss.
                       When None (unit tests), surprise computation is skipped.
            rates    : (B, 4) tensor with columns [λ_ctx, α, β, λ_tgt].
                       When provided, skips LogUniform sampling — intended for
                       downstream use where (λ_ctx, α, β, λ_tgt) are learnable parameters.

        Returns MaskOutput with:
            context_idx          : (B, nctx)
            target_idx           : (B, ntgt)  ntgt ≥ ntgt_min
            context_soft         : (B, N)     p_ctx
            target_soft          : (B, N)     p_tgt
            aux["lambda"]        : (B,)
            aux["alpha"]         : (B,)
            aux["beta"]          : (B,)
            aux["lambda_tgt"]    : (B,)
            aux["p_ign"]         : (B, N)     with grad — needed for β ign_tax
            aux["logits"]        : (B, N, 3)  detached — for diagnostics
            aux["ema_full"]      : (B, N, D)  — passed to aux_loss for surprise
        """
        B = tokens.shape[0]
        device = tokens.device

        # ----------------------------------------------------------------
        # (λ_ctx, α, β, λ_tgt) — sample or use caller-provided rates
        # ----------------------------------------------------------------
        if rates is not None:
            lam     = rates[:, 0]
            alpha   = rates[:, 1]
            beta    = rates[:, 2]
            lam_tgt = rates[:, 3]
        else:
            lam     = _sample_log_uniform(self.lam_min,     self.lam_max,     device).expand(B)
            alpha   = _sample_log_uniform(self.alpha_min,   self.alpha_max,   device).expand(B)
            beta    = _sample_log_uniform(self.beta_min,    self.beta_max,    device).expand(B)
            lam_tgt = _sample_log_uniform(self.lam_tgt_min, self.lam_tgt_max, device).expand(B)

        # ----------------------------------------------------------------
        # Transformer input: compressed tokens (no positional embed)
        # ----------------------------------------------------------------
        ctx = self.proj_in(tokens)   # (B, M, d)

        # ----------------------------------------------------------------
        # (λ, α, β)-conditioned positional embeddings
        # ----------------------------------------------------------------
        log_rates = torch.stack(
            [lam.log(), alpha.log(), beta.log(), lam_tgt.log()], dim=-1
        )  # (B, 4)
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
        # 3-way Gumbel-softmax (Concrete relaxation)
        # ----------------------------------------------------------------
        logits = self.proj_score(query_out)   # (B, N, 3)

        if self.training:
            logits = logits + _sample_gumbel(logits)
        soft = F.softmax(logits / self.temperature, dim=-1)

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
                "lambda":      lam,
                "alpha":       alpha,
                "beta":        beta,
                "lambda_tgt":  lam_tgt,
                "p_ign":       p_ign,           # NOT detached — needed for β gradient
                "logits":      logits.detach(), # pre-Gumbel, for diagnostics
                "ema_full":    ema_full,        # (B, N, D) — used in aux_loss
            },
        )

    # ------------------------------------------------------------------
    # Auxiliary loss — full rate-distortion-surprise objective
    # ------------------------------------------------------------------

    def aux_loss(
        self,
        mask_output: MaskOutput,
        reconstruction_loss: torch.Tensor,
        patch_loss: Optional[torch.Tensor] = None,  # available but unused here
    ) -> torch.Tensor:
        """
        Returns reconstruction_loss - α·surprise_soft + β·ign_tax
              + λ_ctx·N·R_ctx + λ_tgt·N·R_tgt
        as the complete training objective.

        patch_loss is not used in the surprise formulation (reconstruction_loss
        already captures prediction difficulty as a scalar). It remains in the
        signature for interface compatibility and future curriculum work.
        """
        p_ctx    = mask_output.context_soft          # (B, N)
        p_tgt    = mask_output.target_soft           # (B, N)
        p_ign    = mask_output.aux["p_ign"]          # (B, N) — with grad
        lam      = mask_output.aux["lambda"]         # (B,)
        alpha    = mask_output.aux["alpha"]          # (B,)
        beta     = mask_output.aux["beta"]           # (B,)
        lam_tgt  = mask_output.aux["lambda_tgt"]    # (B,)
        ema_full = mask_output.aux.get("ema_full")   # (B, N, D) or None

        if ema_full is None:
            # Unit test / fallback — rate terms only, no surprise
            R_ctx = p_ctx.sum(dim=-1).mean() / self.num_patches
            R_tgt = p_tgt.sum(dim=-1).mean() / self.num_patches
            mask_output.aux["surprise_mean"] = 0.0
            mask_output.aux["R"] = float(R_ctx.item())
            mask_output.aux["ign_rate"] = 0.0
            return (reconstruction_loss
                    + lam.mean() * self.num_patches * R_ctx
                    + lam_tgt.mean() * self.num_patches * R_tgt)

        total, surprise, R_ctx, ign_rate = self.rd_loss(
            reconstruction_loss=reconstruction_loss,
            p_ctx=p_ctx,
            p_tgt=p_tgt,
            p_ign=p_ign,
            ema_full=ema_full,
            lam=lam,
            alpha=alpha,
            beta=beta,
            lam_tgt=lam_tgt,
        )

        # Write back into aux for mask_diagnostics to pick up
        mask_output.aux["surprise_mean"] = float(surprise.item())
        mask_output.aux["R"] = float(R_ctx.item())
        mask_output.aux["ign_rate"] = float(ign_rate.item())

        return total


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _sample_log_uniform(low: float, high: float, device: torch.device) -> torch.Tensor:
    log_val = math.log(low) + (math.log(high) - math.log(low)) * torch.rand(1, device=device)
    return log_val.exp()


def _sample_gumbel(like: torch.Tensor) -> torch.Tensor:
    u = torch.zeros_like(like).uniform_().clamp_(1e-10, 1.0 - 1e-10)
    return -(-u.log()).log()
