"""
RateDist3WayMasker: learned masker with post-selection rate-distortion-surprise
objective.

Architecture
------------
Identical to PredictorBasedMasker:
  proj_in → [compressed_ctx | selection_queries] → TransformerEncoder → proj_score

The only structural difference from PredictorBasedMasker is:
  1. proj_score outputs 3 logits (ctx / tgt / ignore) instead of 1
  2. Option-C λ-conditioned positional residuals on the selection queries
  3. owns_loss=True: aux_loss() returns the full objective

No learned geometry module. Surprise is computed analytically in aux_loss
from the EMA tokens and the soft/hard mask probabilities — zero extra parameters.

Objective (see rd_loss.py for full derivation)
----------------------------------------------
    total = reconstruction_loss
          - α · (surprise_direct + surprise_reinforce)
          + λ · N · R

    surprise_direct    = mean(BS)                               gradient → p_ctx
    surprise_reinforce = mean(BS.detach() · log_p_tgt_at_K)    gradient → p_tgt
    BS_k               = ||EMA_tgt_k - ctx_centroid_soft||²     (B, K)
    ctx_centroid_soft  = Σᵢ (p_ctx_i/Σⱼp_ctx_j) · EMA_i       (B, D)
    R                  = (1/N) Σᵢ p_ctx_i

Push-pull
---------
reconstruction_loss : predictor must predict targets from context
-α · surprise       : masker must pick targets context cannot explain
λ · N · R           : context is expensive; use less of it

Collapse prevention
-------------------
ntgt = max(ntgt_min, round(batch_mean(Σᵢ p_tgt_i)))
ntgt_min (default 4) ensures K ≥ 4 hard targets always exist, so
reconstruction_loss > 0 always, never pulling total to zero.
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
    Rate-distortion 3-way masker with post-selection Bayesian surprise.

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
    lam_min       : Lower bound of LogUniform λ.
    lam_max       : Upper bound of LogUniform λ.
    ntgt_min      : Hard floor on target count.
    alpha         : Surprise bonus weight (see rd_loss.py). Start 0.05–0.1.
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
        ntgt_min: int = 4,
        alpha: float = 0.1,
        base_kind: str = "smooth_l1",  # unused; kept for build.py compatibility
        normalize: bool = False,       # unused; kept for build.py compatibility
    ) -> None:
        super().__init__()

        if lam_min <= 0:
            raise ValueError(f"lam_min must be > 0 for LogUniform, got {lam_min}")
        if lam_max <= lam_min:
            raise ValueError(f"lam_max ({lam_max}) must be > lam_min ({lam_min})")

        self.num_patches = int(num_patches)
        self.temperature = float(temperature)
        self.lam_min = float(lam_min)
        self.lam_max = float(lam_max)
        self.ntgt_min = max(1, int(ntgt_min))

        # ----------------------------------------------------------------
        # Transformer backbone — identical to PredictorBasedMasker
        # ----------------------------------------------------------------
        self.proj_in = nn.Linear(dim, predictor_dim)

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, predictor_dim))
        self.selection_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))

        # Option C: λ-conditioned positional residual
        self.lam_proj = nn.Linear(1, predictor_dim)
        self.pos_lam_mlp = nn.Sequential(
            nn.LayerNorm(2 * predictor_dim),
            nn.Linear(2 * predictor_dim, predictor_dim),
            nn.GELU(),
            nn.Linear(predictor_dim, predictor_dim),
        )

        layer = nn.TransformerEncoderLayer(
            d_model=predictor_dim,
            nhead=num_heads,
            dim_feedforward=int(predictor_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            layer, num_layers=depth, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(predictor_dim)

        # 3-way score head: predictor_dim → [l_ctx, l_tgt, l_ign]
        self.proj_score = nn.Linear(predictor_dim, 3)

        # Surprise-based RD loss (no learned parameters)
        self.rd_loss = RateDistSurpriseLoss(num_patches=num_patches, alpha=alpha)

        # ----------------------------------------------------------------
        # Initialisation
        # ----------------------------------------------------------------
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.selection_token, std=0.02)
        nn.init.normal_(self.lam_proj.weight, std=0.01)
        nn.init.zeros_(self.lam_proj.bias)
        nn.init.zeros_(self.pos_lam_mlp[-1].weight)
        nn.init.zeros_(self.pos_lam_mlp[-1].bias)
        # Score head: near-zero → soft-uniform 3-way distribution at init
        nn.init.zeros_(self.proj_score.weight)
        nn.init.zeros_(self.proj_score.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        tokens: torch.Tensor,                    # (B, M, D)
        ema_full: Optional[torch.Tensor] = None, # (B, N, D)
    ) -> MaskOutput:
        """
        Args:
            tokens   : (B, M, D) compressed EMA tokens — input to transformer.
            ema_full : (B, N, D) full EMA tokens — stored in aux for aux_loss.
                       When None (unit tests), surprise computation is skipped.

        Returns MaskOutput with:
            context_idx  : (B, nctx)
            target_idx   : (B, ntgt)  ntgt ≥ ntgt_min
            context_soft : (B, N)     p_ctx
            target_soft  : (B, N)     p_tgt
            aux["lambda"]: (B,)
            aux["p_ign"] : (B, N)     detached
            aux["logits"]: (B, N, 3)  detached — for diagnostics
            aux["ema_full"]: (B, N, D) — passed to aux_loss for surprise
        """
        B = tokens.shape[0]
        device = tokens.device

        # ----------------------------------------------------------------
        # λ sample — once per batch, identical for all B samples
        # ----------------------------------------------------------------
        lam = _sample_log_uniform(self.lam_min, self.lam_max, device=device).expand(B)

        # ----------------------------------------------------------------
        # Transformer input: compressed tokens (no positional embed)
        # ----------------------------------------------------------------
        ctx = self.proj_in(tokens)   # (B, M, predictor_dim)

        # ----------------------------------------------------------------
        # Option C: λ-conditioned positional embeddings
        # ----------------------------------------------------------------
        lam_embed = self.lam_proj(lam.unsqueeze(-1))                    # (B, predictor_dim)
        lam_embed_exp = lam_embed.unsqueeze(1).expand(-1, self.num_patches, -1)
        pos_base = self.pos_embed.expand(B, -1, -1)                     # (B, N, predictor_dim)
        pos_residual = self.pos_lam_mlp(
            torch.cat([pos_base, lam_embed_exp], dim=-1)
        )
        pos_embed_lam = pos_base + pos_residual                         # (B, N, predictor_dim)

        # ----------------------------------------------------------------
        # Selection queries + transformer
        # ----------------------------------------------------------------
        queries = self.selection_token.expand(B, self.num_patches, -1) + pos_embed_lam
        seq = torch.cat([ctx, queries], dim=1)   # (B, M+N, predictor_dim)
        out = self.norm(self.blocks(seq))
        query_out = out[:, -self.num_patches:]   # (B, N, predictor_dim)

        # ----------------------------------------------------------------
        # 3-way Gumbel-softmax
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
                "lambda":   lam,
                "p_ign":    p_ign.detach(),
                "logits":   logits.detach(),   # pre-Gumbel, for diagnostics
                "ema_full": ema_full,           # (B, N, D) — used in aux_loss
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
        Returns reconstruction_loss - α·surprise + λ·N·R as the complete
        training objective.

        patch_loss is not used in the surprise formulation (reconstruction_loss
        already captures prediction difficulty as a scalar). It remains in the
        signature for interface compatibility and future curriculum work.
        """
        p_ctx    = mask_output.context_soft          # (B, N)
        p_tgt    = mask_output.target_soft           # (B, N)
        tgt_idx  = mask_output.target_idx            # (B, K)
        lam      = mask_output.aux["lambda"]         # (B,)
        ema_full = mask_output.aux.get("ema_full")   # (B, N, D) or None

        if ema_full is None:
            # Unit test / fallback — rate term only, no surprise
            R = p_ctx.sum(dim=-1).mean() / self.num_patches
            rate_term = lam.mean() * self.num_patches * R
            mask_output.aux["surprise_mean"] = 0.0
            mask_output.aux["R"] = float(R.item())
            return reconstruction_loss + rate_term

        total, BS_mean, R = self.rd_loss(
            reconstruction_loss=reconstruction_loss,
            p_ctx=p_ctx,
            p_tgt=p_tgt,
            tgt_idx=tgt_idx,
            ema_full=ema_full,
            lam=lam,
        )

        # Write back into aux for mask_diagnostics to pick up
        mask_output.aux["surprise_mean"] = float(BS_mean.item())
        mask_output.aux["R"] = float(R.item())

        return total


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _sample_log_uniform(low: float, high: float, device: torch.device) -> torch.Tensor:
    log_lam = math.log(low) + (math.log(high) - math.log(low)) * torch.rand(1, device=device)
    return log_lam.exp()


def _sample_gumbel(like: torch.Tensor) -> torch.Tensor:
    u = torch.zeros_like(like).uniform_().clamp_(1e-10, 1.0 - 1e-10)
    return -(-u.log()).log()