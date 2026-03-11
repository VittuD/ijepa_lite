"""
PredictorBasedMasker: uses the same transformer architecture as the JEPA Predictor
to learn a patch selection distribution conditioned on EMA encoder tokens.

Architecture comparison
-----------------------

Predictor forward:
    ctx_tokens (B, Nctx, D) → proj_in → + ctx_pos_embed
    mask_token.expand(B, K)            → + tgt_pos_embed
    cat [ctx, queries] → transformer → norm → last K → proj_out → (B, K, D)

PredictorBasedMasker forward:
    compressed  (B, M, D)  → proj_in → (no pos: compressor handles spatial)
    selection_token.expand(B, N)       → + pos_embed  (all N patch positions)
    cat [ctx, queries] → transformer → norm → last N → proj_score → (B, N) scalars
    → softmax + Gumbel-TopK → MaskOutput

Differences from Predictor
--------------------------
1. proj_score: predictor_dim → 1 (scalar logit per patch) instead of proj_out → D
2. selection_token plays the role of mask_token (same shape, different semantics)
3. Compressed input tokens receive no positional embedding — they are treated as
   unpositioned context regardless of compressor mode.  The N selection queries
   still get full positional embeddings, so the transformer remains spatially aware.
4. Output: soft scores → Gumbel-TopK → MaskOutput instead of token embeddings.

Why no pos embed on inputs?
---------------------------
For compressor modes where M < N (cls, mean_pool, spatial_stride, attention_pool)
the compressed tokens have no well-defined patch positions.  Treating all compressed
tokens as unpositioned context works uniformly across all modes.  For "full" mode
(M = N) this means sacrificing input positional information, but the selection
queries at all N positions still give the transformer full spatial awareness.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ijepa_lite.masking.base import LatentMasker, MaskOutput
from ijepa_lite.masking.registry import register


@register("predictor_based")
class PredictorBasedMasker(LatentMasker):
    """
    Learned masker whose transformer backbone matches the JEPA Predictor exactly:
    same depth, num_heads, predictor_dim, mlp_ratio, dropout, and positional
    embedding shape.

    By default, build.py pulls predictor_dim / depth / num_heads / mlp_ratio /
    dropout from cfg.predictor, so the masker and predictor have identical
    capacity out of the box.  Any field can be overridden in the latent config.

    Args
    ----
    dim            : Encoder embedding dimension (must match EMA encoder output).
    predictor_dim  : Internal narrow dimension of the transformer (matches Predictor).
    depth          : Number of transformer layers (matches Predictor).
    num_heads      : Attention heads (matches Predictor).
    mlp_ratio      : FFN expansion ratio (matches Predictor).
    dropout        : Dropout (matches Predictor).
    num_patches    : Total patch positions N in the image grid.
    target_ratio   : Fraction of N patches to select as targets.
    context_ratio  : Fraction of N patches to select as context.
    temperature    : Gumbel-softmax temperature.
                     1.0 = standard; anneal toward ~0.5 for sharper decisions.
    entropy_coeff  : Weight on entropy regularisation in aux_loss.
                     Positive = encourage diverse selection.
                     0.0 = pure reconstruction signal drives masker.
    """

    def __init__(
        self,
        dim: int,
        predictor_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        num_patches: int,
        target_ratio: float,
        context_ratio: float,
        temperature: float = 1.0,
        entropy_coeff: float = 0.01,
        pos_embed_kind: str = "learned",
    ) -> None:
        super().__init__()

        self.num_patches = int(num_patches)
        self.ntgt = max(1, int(round(self.num_patches * float(target_ratio))))
        self.nctx = max(1, int(round(self.num_patches * float(context_ratio))))
        self.temperature = float(temperature)
        self.entropy_coeff = float(entropy_coeff)
        self.predictor_dim = int(predictor_dim)

        import math as _math
        from ijepa_lite.models.pos_embed import build_pos_embed_2d

        _grid = int(_math.isqrt(num_patches))

        # ----------------------------------------------------------------
        # Same building blocks as Predictor
        # ----------------------------------------------------------------

        # Bottleneck projection: encoder dim → narrow predictor dim
        self.proj_in = nn.Linear(dim, predictor_dim)

        # Positional embeddings for all N patch positions.
        # Shared with selection queries; not applied to compressed inputs.
        self.pos_embed = build_pos_embed_2d(pos_embed_kind, _grid, predictor_dim)

        # Learned selection query token — the masker analog of mask_token.
        # One shared token expanded to N queries, each placed at its patch position
        # via pos_embed, exactly mirroring how the Predictor places mask tokens.
        self.selection_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))

        # Transformer encoder — identical hyperparameters to Predictor
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

        # Score head: predictor_dim → 1 scalar logit per patch position.
        # This replaces Predictor's proj_out (predictor_dim → D).
        self.proj_score = nn.Linear(predictor_dim, 1)

        # Initialisation — matches Predictor
        nn.init.trunc_normal_(self.selection_token, std=0.02)
        # Score head near-zero init → soft-uniform selection at training start
        nn.init.zeros_(self.proj_score.weight)
        nn.init.zeros_(self.proj_score.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, tokens: torch.Tensor) -> MaskOutput:
        """
        Args:
            tokens : (B, M, D) compressed EMA encoder tokens.

        Returns:
            MaskOutput with hard indices and soft scores.

        Transformer sequence layout (mirrors Predictor exactly):
            [ compressed_context (M tokens) | selection_queries (N tokens) ]
                                             └── last N outputs → scores
        """
        B = tokens.shape[0]

        # ----------------------------------------------------------------
        # Context: compressed tokens projected to predictor_dim.
        # No positional embedding — works uniformly for all compressor modes.
        # ----------------------------------------------------------------
        ctx = self.proj_in(tokens)   # (B, M, predictor_dim)

        # ----------------------------------------------------------------
        # Selection queries: N tokens, one per patch position.
        # Mirrors Predictor: mask_token.expand + tgt_pos_embed.
        # ----------------------------------------------------------------
        queries = self.selection_token.expand(B, self.num_patches, -1)  # (B, N, predictor_dim)
        queries = queries + self.pos_embed                               # (B, N, predictor_dim)

        # ----------------------------------------------------------------
        # Transformer over [context | queries] — same as Predictor
        # ----------------------------------------------------------------
        seq = torch.cat([ctx, queries], dim=1)   # (B, M+N, predictor_dim)
        out = self.blocks(seq)
        out = self.norm(out)

        # Last N outputs correspond to the N selection queries
        query_out = out[:, -self.num_patches:]   # (B, N, predictor_dim)

        # ----------------------------------------------------------------
        # Score each patch position → soft selection distribution
        # ----------------------------------------------------------------
        logits = self.proj_score(query_out).squeeze(-1)   # (B, N)

        if self.training:
            gumbel_noise = _sample_gumbel(logits)
            soft_scores = F.softmax(
                (logits + gumbel_noise) / self.temperature, dim=-1
            )
        else:
            soft_scores = F.softmax(logits / self.temperature, dim=-1)

        # ----------------------------------------------------------------
        # Hard selection: top-ntgt for targets, top-nctx from remainder
        # ----------------------------------------------------------------
        _, tgt_idx = torch.topk(soft_scores, self.ntgt, dim=-1, sorted=False)

        ctx_scores = soft_scores.clone()
        ctx_scores.scatter_(1, tgt_idx, 0.0)
        ctx_scores = ctx_scores / (ctx_scores.sum(dim=-1, keepdim=True) + 1e-10)

        _, ctx_idx = torch.topk(ctx_scores, self.nctx, dim=-1, sorted=False)

        return MaskOutput(
            context_idx=ctx_idx,
            target_idx=tgt_idx,
            context_soft=ctx_scores,
            target_soft=soft_scores,
            aux={"logits": logits.detach()},
        )

    # ------------------------------------------------------------------
    # Auxiliary loss — entropy regularisation (identical to GumbelTopKMasker)
    # ------------------------------------------------------------------

    def aux_loss(
        self,
        mask_output: MaskOutput,
        reconstruction_loss: torch.Tensor,
        patch_loss=None,  # accepted but not used; kept for interface consistency
    ) -> torch.Tensor:
        if mask_output.target_soft is None:
            return reconstruction_loss.new_zeros(())

        soft = mask_output.target_soft   # (B, N)
        entropy = -(soft * (soft + 1e-10).log()).sum(dim=-1).mean()
        return -self.entropy_coeff * entropy


# ------------------------------------------------------------------
# Private helper
# ------------------------------------------------------------------

def _sample_gumbel(like: torch.Tensor) -> torch.Tensor:
    u = torch.zeros_like(like).uniform_().clamp_(1e-10, 1.0 - 1e-10)
    return -(-u.log()).log()