"""
MIRateMasker: learned masker with modular compositional loss.

Architecture
------------
  proj_in → [compressed_ctx | selection_queries] → TransformerEncoder → proj_score

The loss is a weighted sum of independently toggleable atomic terms
(see ``losses/terms.py`` and ``losses/composite.py``). Per-step term
weights are sampled from LogUniform distributions with optional warmup.

No (λ, α) conditioning — the transformer sees only positional embeddings
and compressed context tokens, keeping the masker architecture agnostic
to the loss weighting.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ijepa_lite.losses.composite import CompositeMaskerLoss
from ijepa_lite.masking.base import LatentMasker, MaskOutput
from ijepa_lite.masking.registry import register


@register("mi_3way")
class MIRateMasker(LatentMasker):
    """
    MI-rate 3-way masker with modular compositional loss.

    Args
    ----
    dim            : Encoder embedding dim.
    predictor_dim  : Internal transformer dim (matches Predictor).
    depth          : Transformer layers.
    num_heads      : Attention heads.
    mlp_ratio      : FFN expansion.
    dropout        : Dropout.
    num_patches    : N — total patch positions.
    terms          : Nested dict of term configs (see CompositeMaskerLoss).
    ntgt_min       : Hard floor on target count.
    nctx_min       : Hard floor on context count.
    warmup_epochs  : Epochs to grow weight sampling range to full.
    """

    owns_loss: bool = True
    needs_full_tokens: bool = True

    def __init__(
        self,
        dim: int,
        predictor_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        num_patches: int,
        terms: dict,
        ntgt_min: int = 4,
        nctx_min: int = 1,
        warmup_epochs: int = 0,
        pos_embed_kind: str = "learned",
        # Unused — kept for build.py kwarg filtering
        base_kind: str = "smooth_l1",
        normalize: bool = False,
    ) -> None:
        super().__init__()

        self.num_patches = int(num_patches)
        self.ntgt_min = max(1, int(ntgt_min))
        self.nctx_min = max(1, int(nctx_min))
        self.warmup_epochs = int(warmup_epochs)

        # _progress in [0, 1]; initialised to 1.0 so unit tests use full range.
        self.register_buffer("_progress", torch.tensor(1.0), persistent=False)

        d = predictor_dim

        from ijepa_lite.models.pos_embed import build_pos_embed_2d

        grid_size = int(math.isqrt(num_patches))

        # ----------------------------------------------------------------
        # Transformer backbone
        # ----------------------------------------------------------------
        self.proj_in = nn.Linear(dim, d)

        self.pos_embed = build_pos_embed_2d(pos_embed_kind, grid_size, d)
        self.selection_token = nn.Parameter(torch.zeros(1, 1, d))

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

        # Composite loss
        self.composite_loss = CompositeMaskerLoss(terms, num_patches)

        # ----------------------------------------------------------------
        # Initialisation
        # ----------------------------------------------------------------
        nn.init.trunc_normal_(self.selection_token, std=0.02)
        nn.init.trunc_normal_(self.proj_score.weight, std=0.02)
        nn.init.zeros_(self.proj_score.bias)

    # ------------------------------------------------------------------
    # Warmup progress
    # ------------------------------------------------------------------

    def set_progress(self, fraction: float) -> None:
        """Update warmup progress. Call once per epoch."""
        self._progress.fill_(max(0.0, min(1.0, float(fraction))))

    # ------------------------------------------------------------------
    # Weight sampling
    # ------------------------------------------------------------------

    def _sample_weights(self, device: torch.device) -> dict[str, float]:
        """Sample per-term weights from LogUniform with warmup."""
        p = max(self._progress.item(), 1e-3)
        weights: dict[str, float] = {}

        for name, (lo, hi) in self.composite_loss.weight_ranges.items():
            if lo == hi:
                weights[name] = lo
            else:
                # Temperature-biased LogUniform: u^(1/p) concentrates near lo
                u = torch.rand(1, device=device).item() ** (1.0 / p)
                val = math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
                weights[name] = val

        return weights

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        tokens: torch.Tensor,                     # (B, M, D)
        ema_full: Optional[torch.Tensor] = None,   # (B, N, D)
        rates: Optional[torch.Tensor] = None,      # unused, kept for interface compat
    ) -> MaskOutput:
        B = tokens.shape[0]

        # ----------------------------------------------------------------
        # Transformer input: compressed tokens
        # ----------------------------------------------------------------
        ctx = self.proj_in(tokens)   # (B, M, d)

        # ----------------------------------------------------------------
        # Selection queries + transformer (no rates conditioning)
        # ----------------------------------------------------------------
        queries = self.selection_token.expand(B, self.num_patches, -1) \
                  + self.pos_embed.expand(B, -1, -1)
        seq = torch.cat([ctx, queries], dim=1)   # (B, M+N, d)
        out = self.norm(self.blocks(seq))
        query_out = out[:, -self.num_patches:]   # (B, N, d)

        # ----------------------------------------------------------------
        # 3-way soft assignments
        # ----------------------------------------------------------------
        logits = self.proj_score(query_out)   # (B, N, 3)
        soft = F.softmax(logits, dim=-1)      # (B, N, 3)

        p_ctx = soft[..., 0]
        p_tgt = soft[..., 1]
        p_ign = soft[..., 2]

        # ----------------------------------------------------------------
        # Hard counts — ntgt floored at ntgt_min
        # ----------------------------------------------------------------
        ntgt = max(self.ntgt_min, int(round(p_tgt.sum(dim=-1).mean().item())))
        nctx = max(self.nctx_min, int(round(p_ctx.sum(dim=-1).mean().item())))

        _, tgt_idx = torch.topk(p_tgt, ntgt, dim=-1, sorted=False)
        p_ctx_masked = p_ctx.clone().scatter_(1, tgt_idx, 0.0)
        _, ctx_idx = torch.topk(p_ctx_masked, nctx, dim=-1, sorted=False)

        # Sample weights for this step
        weights = self._sample_weights(tokens.device)

        return MaskOutput(
            context_idx=ctx_idx,
            target_idx=tgt_idx,
            context_soft=p_ctx,
            target_soft=p_tgt,
            aux={
                "weights":  weights,
                "p_ign":    p_ign,
                "logits":   logits.detach(),
                "ema_full": ema_full,
            },
        )

    # ------------------------------------------------------------------
    # Auxiliary loss
    # ------------------------------------------------------------------

    def aux_loss(
        self,
        mask_output: MaskOutput,
        reconstruction_loss: torch.Tensor,
        patch_loss: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        p_ctx    = mask_output.context_soft
        p_tgt    = mask_output.target_soft
        p_ign    = mask_output.aux["p_ign"]
        weights  = mask_output.aux["weights"]
        ema_full = mask_output.aux.get("ema_full")

        if ema_full is None:
            # Unit test fallback — compute only entropy terms
            device = p_ctx.device
            ema_full = torch.zeros(
                p_ctx.shape[0], p_ctx.shape[1], 1, device=device,
            )

        total, logs = self.composite_loss(
            weights=weights,
            p_ctx=p_ctx,
            p_tgt=p_tgt,
            p_ign=p_ign,
            ema_full=ema_full,
        )

        # Write logs into aux for metrics.py to pick up
        mask_output.aux.update(logs)

        return reconstruction_loss + total


# ======================================================================
# N-way MI masker (cross-target surprise, no ctx-vs-tgt terms)
# ======================================================================


@register("mi_nway")
class MINWayMasker(LatentMasker):
    """
    N-way MI masker with (M+2)-way categorical assignments.

    Assigns each patch to (ctx, tgt₁, ..., tgt_M, ign). Loss terms only
    push target blocks apart (inter-target surprise). Context relevance
    emerges naturally without adversarial ctx-vs-tgt distance terms.
    """

    owns_loss: bool = True
    needs_full_tokens: bool = True

    def __init__(
        self,
        dim: int,
        predictor_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        num_patches: int,
        terms: dict,
        num_tgt_blocks: int = 4,
        ntgt_min_per_block: int = 4,
        nctx_min: int = 1,
        hard_assignment: str = "topk",
        warmup_epochs: int = 0,
        pos_embed_kind: str = "learned",
        # Unused — kept for build.py kwarg filtering
        base_kind: str = "smooth_l1",
        normalize: bool = False,
    ) -> None:
        super().__init__()

        self.num_patches = int(num_patches)
        self.M = int(num_tgt_blocks)
        self.ntgt_min_per_block = max(1, int(ntgt_min_per_block))
        self.nctx_min = max(1, int(nctx_min))
        self.hard_assignment = str(hard_assignment)
        self.warmup_epochs = int(warmup_epochs)

        self.register_buffer("_progress", torch.tensor(1.0), persistent=False)

        d = predictor_dim

        from ijepa_lite.models.pos_embed import build_pos_embed_2d

        grid_size = int(math.isqrt(num_patches))

        # ----------------------------------------------------------------
        # Transformer backbone (same as MIRateMasker)
        # ----------------------------------------------------------------
        self.proj_in = nn.Linear(dim, d)

        self.pos_embed = build_pos_embed_2d(pos_embed_kind, grid_size, d)
        self.selection_token = nn.Parameter(torch.zeros(1, 1, d))

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

        # (M+2)-way score head: d → [l_ctx, l_tgt1, ..., l_tgtM, l_ign]
        self.proj_score = nn.Linear(d, self.M + 2)

        # Composite loss
        self.composite_loss = CompositeMaskerLoss(
            terms, num_patches, num_tgt_blocks=self.M
        )

        # ----------------------------------------------------------------
        # Initialisation
        # ----------------------------------------------------------------
        nn.init.trunc_normal_(self.selection_token, std=0.02)
        nn.init.trunc_normal_(self.proj_score.weight, std=0.02)
        nn.init.zeros_(self.proj_score.bias)

    # ------------------------------------------------------------------
    # Warmup progress
    # ------------------------------------------------------------------

    def set_progress(self, fraction: float) -> None:
        self._progress.fill_(max(0.0, min(1.0, float(fraction))))

    # ------------------------------------------------------------------
    # Weight sampling
    # ------------------------------------------------------------------

    def _sample_weights(self, device: torch.device) -> dict[str, float]:
        p = max(self._progress.item(), 1e-3)
        weights: dict[str, float] = {}

        for name, (lo, hi) in self.composite_loss.weight_ranges.items():
            if lo == hi:
                weights[name] = lo
            else:
                u = torch.rand(1, device=device).item() ** (1.0 / p)
                val = math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
                weights[name] = val

        return weights

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        tokens: torch.Tensor,
        ema_full: Optional[torch.Tensor] = None,
        rates: Optional[torch.Tensor] = None,
    ) -> MaskOutput:
        B = tokens.shape[0]
        M = self.M

        # ----------------------------------------------------------------
        # Transformer input: compressed tokens
        # ----------------------------------------------------------------
        ctx = self.proj_in(tokens)

        # ----------------------------------------------------------------
        # Selection queries + transformer
        # ----------------------------------------------------------------
        queries = self.selection_token.expand(B, self.num_patches, -1) \
                  + self.pos_embed.expand(B, -1, -1)
        seq = torch.cat([ctx, queries], dim=1)
        out = self.norm(self.blocks(seq))
        query_out = out[:, -self.num_patches:]

        # ----------------------------------------------------------------
        # (M+2)-way soft assignments
        # ----------------------------------------------------------------
        logits = self.proj_score(query_out)   # (B, N, M+2)
        soft = F.softmax(logits, dim=-1)      # (B, N, M+2)

        p_ctx  = soft[..., 0]                 # (B, N)
        p_tgts = soft[..., 1:M+1]            # (B, N, M)
        p_ign  = soft[..., -1]                # (B, N)

        # ----------------------------------------------------------------
        # Hard indices → (B, M, K)
        # ----------------------------------------------------------------
        if self.hard_assignment == "argmax":
            # Each patch goes to its argmax role; no topk budget constraint.
            # Blocks may have variable sizes → pad to max for rectangular tensor.
            winners = soft.argmax(dim=-1)  # (B, N)  values in [0, M+1]

            # Collect per-block indices
            tgt_idx_list = []
            for k in range(M):
                mask_k = (winners == (1 + k))  # (B, N)
                # Per-sample indices where this block wins
                batch_indices = []
                for b in range(B):
                    idxs_b = mask_k[b].nonzero(as_tuple=False).squeeze(-1)
                    if idxs_b.numel() < self.ntgt_min_per_block:
                        # Fall back: top ntgt_min_per_block by soft probability
                        _, idxs_b = torch.topk(
                            p_tgts[b, :, k], self.ntgt_min_per_block,
                            dim=-1, sorted=False,
                        )
                    batch_indices.append(idxs_b)
                tgt_idx_list.append(batch_indices)

            # Pad to uniform K across blocks and batch
            K = max(
                self.ntgt_min_per_block,
                max(
                    idxs.numel()
                    for block_indices in tgt_idx_list
                    for idxs in block_indices
                ),
            )
            padded = []
            for k in range(M):
                block_padded = []
                for b in range(B):
                    idxs = tgt_idx_list[k][b]
                    n = idxs.numel()
                    if n < K:
                        # Cycle through genuine indices to pad evenly
                        idxs = idxs.repeat((K + n - 1) // n)[:K]
                    elif n > K:
                        idxs = idxs[:K]
                    block_padded.append(idxs)
                padded.append(torch.stack(block_padded))  # (B, K)
            tgt_idx = torch.stack(padded, dim=1)  # (B, M, K)

            # Context: patches whose argmax is ctx (channel 0)
            ctx_lists = []
            for b in range(B):
                ctx_b = (winners[b] == 0).nonzero(as_tuple=False).squeeze(-1)
                if ctx_b.numel() < self.nctx_min:
                    # Fall back: top nctx_min by p_ctx (excluding targets)
                    tgt_flat_b = tgt_idx[b].reshape(-1)
                    p_ctx_b = p_ctx[b].clone()
                    p_ctx_b.scatter_(0, tgt_flat_b, 0.0)
                    _, ctx_b = torch.topk(
                        p_ctx_b, self.nctx_min, dim=-1, sorted=False,
                    )
                ctx_lists.append(ctx_b)
            # Pad context to uniform size
            nctx = max(self.nctx_min, max(c.numel() for c in ctx_lists))
            ctx_padded = []
            for ctx_b in ctx_lists:
                n = ctx_b.numel()
                if n < nctx:
                    ctx_b = ctx_b.repeat((nctx + n - 1) // n)[:nctx]
                elif n > nctx:
                    ctx_b = ctx_b[:nctx]
                ctx_padded.append(ctx_b)
            ctx_idx = torch.stack(ctx_padded)  # (B, nctx)
        else:
            # topk (default): fixed K per block from soft mass
            per_block_mass = p_tgts.sum(dim=1).mean(dim=0)  # (M,)
            K = max(self.ntgt_min_per_block,
                    int(round(per_block_mass.max().item())))

            tgt_idx_list = []
            for k in range(M):
                _, idx_k = torch.topk(p_tgts[..., k], K, dim=-1, sorted=False)
                tgt_idx_list.append(idx_k)
            tgt_idx = torch.stack(tgt_idx_list, dim=1)  # (B, M, K)

            # Context: topk on p_ctx after zeroing all target positions
            nctx = max(self.nctx_min, int(round(p_ctx.sum(dim=-1).mean().item())))
            tgt_flat = tgt_idx.reshape(B, -1)  # (B, M*K)
            p_ctx_masked = p_ctx.clone().scatter_(1, tgt_flat, 0.0)
            _, ctx_idx = torch.topk(p_ctx_masked, nctx, dim=-1, sorted=False)

        # Sample weights for this step
        weights = self._sample_weights(tokens.device)

        # target_soft = sum across blocks for backward compat metrics
        return MaskOutput(
            context_idx=ctx_idx,       # (B, Nctx)
            target_idx=tgt_idx,        # (B, M, K)
            context_soft=p_ctx,        # (B, N)
            target_soft=p_tgts.sum(-1),  # (B, N) — total target mass
            aux={
                "weights":  weights,
                "p_ign":    p_ign,
                "soft":     soft,
                "logits":   logits.detach(),
                "ema_full": ema_full,
            },
        )

    # ------------------------------------------------------------------
    # Auxiliary loss
    # ------------------------------------------------------------------

    def aux_loss(
        self,
        mask_output: MaskOutput,
        reconstruction_loss: torch.Tensor,
        patch_loss: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        p_ctx    = mask_output.context_soft
        p_tgt    = mask_output.target_soft
        p_ign    = mask_output.aux["p_ign"]
        weights  = mask_output.aux["weights"]
        ema_full = mask_output.aux.get("ema_full")
        soft     = mask_output.aux.get("soft")

        if ema_full is None:
            device = p_ctx.device
            ema_full = torch.zeros(
                p_ctx.shape[0], p_ctx.shape[1], 1, device=device,
            )

        total, logs = self.composite_loss(
            weights=weights,
            p_ctx=p_ctx,
            p_tgt=p_tgt,
            p_ign=p_ign,
            ema_full=ema_full,
            soft=soft,
        )

        mask_output.aux.update(logs)

        return reconstruction_loss + total
