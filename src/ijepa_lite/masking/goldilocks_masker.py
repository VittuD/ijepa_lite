"""
GoldilocksTeacherMasker: curriculum masker driven by reconstruction error.

Design
------
Replaces the MI-rate + surprise objective with a *deliberate curriculum teacher*
that uses per-patch reconstruction error (already stop-gradient detached in
ijepa.py) as a direct teaching signal:

    target_i = exp(-ê_i² / 2)   where ê_i is z-scored patch loss
    L_goldilocks = mean((q_i - target_i)²)
    L_total = L_pred + β · L_goldilocks

Gradient isolation is natural:
  - patch_loss.detach() already in ijepa.py — masker never sees ∇_θ
  - encoder never sees ∇_ψ — masker loss not added to reconstruction backward

Context is a contiguous rectangle (like I-JEPA multiblock) selected first;
targets are scored and selected from remaining non-context patches.
The masker receives a binary ctx_flag conditioning so it learns difficulty
relative to what context is available.
"""
from __future__ import annotations

import math
import random
from typing import Optional

import torch
import torch.nn as nn

from ijepa_lite.masking.base import LatentMasker, MaskOutput
from ijepa_lite.masking.registry import register
from ijepa_lite.losses.goldilocks_loss import GoldilocksLoss


# ---------------------------------------------------------------------------
# Content-adaptivity diagnostics helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def _content_adaptivity_metrics(
    p_tgt:   torch.Tensor,   # (B, N) soft scores
    tgt_idx: torch.Tensor,   # (B, K) hard target indices
    N: int,
) -> dict:
    B = p_tgt.shape[0]

    # Binary target assignment map (B, N): 1 if patch selected as target
    is_tgt = torch.zeros(B, N, device=p_tgt.device)
    is_tgt.scatter_(1, tgt_idx, 1.0)

    K = tgt_idx.shape[1]

    # 1. tgt_pos_std — per-position std of binary assignment across batch.
    #    Near 0 → masker always picks the same positions (positional).
    #    Higher  → different images → different targets (content-adaptive).
    tgt_pos_std = is_tgt.std(dim=0).mean().item()

    # 1b. tgt_pos_std_norm — tgt_pos_std divided by its expected value under a
    #     random (content-blind) masker: sqrt(p·(1−p)) where p = K/N.
    #     Ratio ≈ 1 → masker behaves like random selection.
    #     Ratio < 1 → positional collapse (always same patches).
    #     Ratio > 1 → more diverse across images than random.
    p_rand = K / N
    std_rand = math.sqrt(p_rand * (1.0 - p_rand)) if 0 < p_rand < 1 else 1.0
    tgt_pos_std_norm = tgt_pos_std / std_rand

    # 2. p_tgt_score_std — per-position std of the *soft* score across batch.
    #    More sensitive than binary: detects content-dependence before hard selection.
    p_tgt_score_std = p_tgt.std(dim=0).mean().item()

    # 3. tgt_assignment_entropy — per-position binary entropy of target frequency.
    #    H(p) = -p*log(p) - (1-p)*log(1-p), averaged over N positions.
    #    Near 0 → masker always picks or always skips each position.
    #    Near log(2) ≈ 0.693 → near-random assignment per position.
    freq = is_tgt.mean(dim=0).clamp(1e-7, 1 - 1e-7)   # (N,)
    entropy = -(freq * freq.log() + (1 - freq) * (1 - freq).log())
    tgt_assignment_entropy = entropy.mean().item()

    # 4. batch_iou — mean pairwise Jaccard similarity across batch.
    #    Expected value under random masker: K / (2N − K).
    #    Near 1 → masker always selects the same patches (positional collapse).
    #    Near E[iou_random] → content-blind random selection.
    #    Below E[iou_random] → more diverse than random (ideal).
    if B > 1:
        inter = is_tgt @ is_tgt.T                                   # (B, B)
        ntgt_vec = is_tgt.sum(dim=1, keepdim=True)                  # (B, 1)
        union = ntgt_vec + ntgt_vec.T - inter
        iou = inter / union.clamp(min=1.0)
        upper = torch.triu(torch.ones(B, B, device=p_tgt.device, dtype=torch.bool), diagonal=1)
        batch_iou = iou[upper].mean().item()
    else:
        batch_iou = 1.0
    iou_random = K / (2 * N - K)

    return {
        "tgt_pos_std":          tgt_pos_std,
        "tgt_pos_std_norm":     tgt_pos_std_norm,
        "p_tgt_score_std":      p_tgt_score_std,
        "tgt_assignment_entropy": tgt_assignment_entropy,
        "batch_iou":            batch_iou,
        "batch_iou_random":     iou_random,
    }


# ---------------------------------------------------------------------------
# GoldilocksTeacherMasker
# ---------------------------------------------------------------------------

@register("goldilocks")
class GoldilocksTeacherMasker(LatentMasker):
    """
    Goldilocks curriculum masker.

    Scores patches by their predicted difficulty relative to available context,
    driven by a Gaussian target derived from z-scored per-patch reconstruction
    errors. "Not too easy, not too hard" → target score = exp(-ê²/2).

    Args
    ----
    dim           : Encoder embedding dim.
    predictor_dim : Internal transformer dim.
    depth         : Transformer layers.
    num_heads     : Attention heads.
    mlp_ratio     : FFN expansion ratio.
    dropout       : Dropout probability.
    num_patches   : N — total patch positions.
    k_tgt_min     : Lower bound for LogUniform target count.
    k_tgt_max     : Upper bound for LogUniform target count.
    k_ctx_min     : Lower bound for Uniform context count.
    k_ctx_max     : Upper bound for Uniform context count.
    beta          : Weight for Goldilocks loss term.
    z_score_eps   : Floor for z-score denominator.
    """

    owns_loss: bool = True
    needs_full_tokens: bool = True   # receives ema_full (B, N, D)

    # Context rectangle parameters (mirrors multiblock_mask.py defaults)
    ctx_min_scale  = 0.85
    ctx_max_scale  = 1.00
    ctx_min_aspect = 0.75
    ctx_max_aspect = 1.50

    def __init__(
        self,
        dim: int,
        predictor_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        num_patches: int,
        k_tgt_min: int = 4,
        k_tgt_max: int = 72,
        k_ctx_min: int = 32,
        k_ctx_max: int = 96,
        beta: float = 1.0,
        z_score_eps: float = 1e-6,
        # Unused kwargs forwarded by build.py — kept for compatibility
        base_kind: str = "smooth_l1",
        normalize: bool = False,
    ) -> None:
        super().__init__()

        self.num_patches = int(num_patches)
        self.grid = int(math.sqrt(num_patches))
        self.k_tgt_min = int(k_tgt_min)
        self.k_tgt_max = int(k_tgt_max)
        self.k_ctx_min = int(k_ctx_min)
        self.k_ctx_max = int(k_ctx_max)
        self.beta = float(beta)

        # Evaluation-time constants (geometric midpoint for k_tgt, arithmetic for k_ctx)
        self.k_tgt_eval = round(math.exp((math.log(k_tgt_min) + math.log(k_tgt_max)) / 2))
        self.k_ctx_eval = round((k_ctx_min + k_ctx_max) / 2)

        d = predictor_dim

        # ----------------------------------------------------------------
        # Transformer backbone (mirrors mi_masker.py lines 178–219)
        # ----------------------------------------------------------------
        self.proj_in  = nn.Linear(dim, d)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d))

        # Binary context-conditioning: 1 = context patch, 0 = not
        # Zero-init so pos_embed starts unperturbed
        self.ctx_embed = nn.Linear(1, d)

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

        # Scalar score per patch; sigmoid → (0, 1)
        self.proj_score = nn.Linear(d, 1)

        # Goldilocks loss (no learnable parameters)
        self.goldilocks_loss = GoldilocksLoss(z_score_eps=z_score_eps)

        # ----------------------------------------------------------------
        # Initialisation
        # ----------------------------------------------------------------
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.ctx_embed.weight, std=0.02)
        nn.init.zeros_(self.ctx_embed.bias)
        nn.init.trunc_normal_(self.proj_score.weight, std=0.02)
        nn.init.zeros_(self.proj_score.bias)

    # ------------------------------------------------------------------
    # No-op hooks (called by train_loop.py, irrelevant here)
    # ------------------------------------------------------------------

    def set_progress(self, fraction: float) -> None:
        pass

    def set_ema_decay(self, m: float) -> None:
        pass

    # ------------------------------------------------------------------
    # K sampling
    # ------------------------------------------------------------------

    def _sample_k_tgt(self) -> int:
        log_k = torch.empty(1).uniform_(
            math.log(self.k_tgt_min), math.log(self.k_tgt_max)
        )
        return max(self.k_tgt_min, min(self.k_tgt_max, int(round(log_k.exp().item()))))

    def _sample_k_ctx(self) -> int:
        return random.randint(self.k_ctx_min, self.k_ctx_max)

    # ------------------------------------------------------------------
    # Context block sampling — contiguous rectangle on patch grid
    # ------------------------------------------------------------------

    def _sample_ctx_block(self, K_ctx: int, device) -> list:
        """Return exactly K_ctx flat patch indices forming a contiguous rectangle."""
        g = self.grid
        log_aspect = random.uniform(
            math.log(self.ctx_min_aspect), math.log(self.ctx_max_aspect)
        )
        aspect = math.exp(log_aspect)
        h = max(1, min(g, int(round(math.sqrt(K_ctx * aspect)))))
        w = max(1, min(g, int(round(K_ctx / h))))
        top  = random.randint(0, max(0, g - h))
        left = random.randint(0, max(0, g - w))
        rect = [(top + r) * g + (left + c) for r in range(h) for c in range(w)]
        if len(rect) >= K_ctx:
            return random.sample(rect, K_ctx)
        pool = [i for i in range(self.num_patches) if i not in set(rect)]
        rect += random.sample(pool, min(K_ctx - len(rect), len(pool)))
        return rect[:K_ctx]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        tokens:   torch.Tensor,
        ema_full: Optional[torch.Tensor] = None,
    ) -> MaskOutput:
        """
        Args
        ----
        tokens   : (B, M, D) compressed EMA tokens (unused; ema_full used instead).
        ema_full : (B, N, D) full EMA tokens — scored by the masker.

        Returns MaskOutput with
        -----------------------
        context_idx  : (B, K_ctx)
        target_idx   : (B, K_tgt)
        context_soft : None
        target_soft  : (B, N) — full soft scores, stored for aux_loss
        aux["k_tgt"] : int
        aux["k_ctx"] : int
        aux + content-adaptivity metrics
        """
        B, N, D = ema_full.shape
        device = ema_full.device

        # Step 1: sample budgets
        K_ctx = self._sample_k_ctx() if self.training else self.k_ctx_eval
        K_tgt = self._sample_k_tgt() if self.training else self.k_tgt_eval
        K_ctx = min(K_ctx, N - K_tgt)   # safety: leave room for targets

        # Step 2: context blocks — one rectangle per sample
        ctx_idx = torch.tensor(
            [self._sample_ctx_block(K_ctx, device) for _ in range(B)],
            dtype=torch.long, device=device,
        )   # (B, K_ctx)

        # Step 3: binary context conditioning (B, N, 1)
        ctx_flag = torch.zeros(B, N, 1, device=device)
        ctx_flag.scatter_(1, ctx_idx.unsqueeze(-1), 1.0)

        # Step 4: transformer with context conditioning
        x = self.proj_in(ema_full) + self.pos_embed    # (B, N, d)
        x = x + self.ctx_embed(ctx_flag)               # inject context visibility
        x = self.norm(self.blocks(x))                  # (B, N, d)
        p_tgt = torch.sigmoid(self.proj_score(x).squeeze(-1))  # (B, N)

        # Step 5: target selection from non-context patches only
        p_tgt_avail = p_tgt.clone()
        p_tgt_avail.scatter_(1, ctx_idx, float("-inf"))
        _, tgt_idx = torch.topk(p_tgt_avail, K_tgt, dim=-1, sorted=False)  # (B, K_tgt)

        # Step 6: content-adaptivity diagnostics (no_grad)
        aux_metrics = _content_adaptivity_metrics(p_tgt, tgt_idx, N)

        return MaskOutput(
            context_idx=ctx_idx,
            target_idx=tgt_idx,
            context_soft=None,
            target_soft=p_tgt,    # full (B, N), stored for aux_loss
            aux={"k_tgt": K_tgt, "k_ctx": K_ctx, **aux_metrics},
        )

    # ------------------------------------------------------------------
    # Auxiliary loss — delegates to GoldilocksLoss
    # ------------------------------------------------------------------

    def aux_loss(
        self,
        mask_output: MaskOutput,
        reconstruction_loss: torch.Tensor,
        patch_loss: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns reconstruction_loss + β · L_goldilocks.

        patch_loss must already be detached (enforced in ijepa.py).
        Falls back to reconstruction_loss alone when patch_loss is unavailable.
        """
        if patch_loss is None:
            return reconstruction_loss

        L_goldilocks = self.goldilocks_loss(
            mask_output.target_soft,
            mask_output.target_idx,
            patch_loss,
        )
        return reconstruction_loss + self.beta * L_goldilocks
