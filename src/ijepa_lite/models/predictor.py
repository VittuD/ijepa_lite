from __future__ import annotations

import torch
import torch.nn as nn


def _gather_pos(pos: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    b, k = idx.shape
    d = pos.shape[-1]
    if pos.shape[0] == 1:
        pos = pos.expand(b, -1, -1)
    return pos.gather(1, idx.unsqueeze(-1).expand(-1, -1, d))


class Predictor(nn.Module):
    """
    Lightweight JEPA-style predictor:
    - takes context tokens at ctx_idx
    - appends learned mask tokens at tgt_idx
    - transformer encoder over [context + target_queries]
    - returns predicted embeddings for target positions
    """

    def __init__(
        self,
        dim: int,  # encoder dim (e.g. 512)
        predictor_dim: int,  # narrow hidden dim (e.g. 256)
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        num_patches: int,
        pos_embed_kind: str = "learned",
    ):
        super().__init__()
        self.dim = dim
        self.predictor_dim = predictor_dim
        self.num_patches = num_patches

        from ijepa_lite.models.pos_embed import build_pos_embed_2d
        import math

        grid_size = int(math.isqrt(num_patches))

        # Bottleneck projections
        self.proj_in = nn.Linear(dim, predictor_dim)
        self.proj_out = nn.Linear(predictor_dim, dim)

        self.pos_embed = build_pos_embed_2d(pos_embed_kind, grid_size, predictor_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))

        layer = nn.TransformerEncoderLayer(
            d_model=predictor_dim,  # narrow
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

        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(
        self,
        ctx_tokens: torch.Tensor,  # (B, Nctx, D)
        ctx_idx: torch.Tensor,  # (B, Nctx)
        tgt_idx: torch.Tensor,  # (B, K)
        return_ctx_pred: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        b, nctx, _ = ctx_tokens.shape
        ntgt = tgt_idx.shape[1]

        # Project down to predictor_dim
        ctx = self.proj_in(ctx_tokens)  # (B, Nctx, predictor_dim)

        ctx_pos = _gather_pos(self.pos_embed, ctx_idx)
        tgt_pos = _gather_pos(self.pos_embed, tgt_idx)

        tgt = self.mask_token.expand(b, ntgt, -1)  # (B, K, predictor_dim)

        seq = torch.cat([ctx + ctx_pos, tgt + tgt_pos], dim=1)
        out = self.blocks(seq)
        out = self.norm(out)

        pred_tgt = self.proj_out(out[:, -ntgt:])  # (B, K, D)
        if return_ctx_pred:
            pred_ctx = self.proj_out(out[:, :nctx])  # (B, Nctx, D)
            return pred_tgt, pred_ctx
        return pred_tgt
