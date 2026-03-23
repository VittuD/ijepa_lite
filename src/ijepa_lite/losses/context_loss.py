"""V-JEPA 2.1-style context loss: distance-weighted supervision on visible tokens."""

from __future__ import annotations

import torch


def context_distance_weights(
    ctx_idx: torch.Tensor,
    tgt_idx: torch.Tensor,
    grid_size: int,
    gamma: float = 0.7,
) -> torch.Tensor:
    """Distance weights λ_i = d_i^{-γ} for each context patch.

    Args:
        ctx_idx:   (B, Nctx) flat patch indices of context tokens.
        tgt_idx:   (B, Ntgt) flat patch indices of target tokens.
        grid_size: patches per side (e.g. 12 for 96×96 image, patch_size=8).
        gamma:     decay exponent (0.7 for images per V-JEPA 2.1).

    Returns:
        (B, Nctx) distance weights.
    """
    g = grid_size
    ctx_r, ctx_c = ctx_idx // g, ctx_idx % g        # (B, Nctx)
    tgt_r, tgt_c = tgt_idx // g, tgt_idx % g        # (B, Ntgt)

    dr = ctx_r.unsqueeze(2) - tgt_r.unsqueeze(1)    # (B, Nctx, Ntgt)
    dc = ctx_c.unsqueeze(2) - tgt_c.unsqueeze(1)
    dist = (dr.float() ** 2 + dc.float() ** 2).sqrt()

    d_min = dist.min(dim=-1).values.clamp(min=1.0)  # (B, Nctx)
    return d_min.pow(-gamma)


def context_loss(
    pred_ctx: torch.Tensor,
    tgt_ctx: torch.Tensor,
    ctx_idx: torch.Tensor,
    tgt_idx: torch.Tensor,
    grid_size: int,
    loss_fn: torch.nn.Module,
    gamma: float = 0.7,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Distance-weighted reconstruction loss on context (visible) tokens.

    Args:
        pred_ctx:  (B, Nctx, D) predictor output at context positions.
        tgt_ctx:   (B, Nctx, D) EMA encoder output at context positions (detached).
        ctx_idx:   (B, Nctx) flat patch indices.
        tgt_idx:   (B, Ntgt) flat patch indices (all targets, flattened if multi-block).
        grid_size: patches per side.
        loss_fn:   token-level loss (smooth_l1 / mse / cosine_mse) with reduction="none".
        gamma:     distance decay exponent.
        alpha:     warmup scale (0→1).

    Returns:
        Scalar loss.
    """
    err = loss_fn(pred_ctx, tgt_ctx, reduction="none")               # (B, Nctx)
    w = context_distance_weights(ctx_idx, tgt_idx, grid_size, gamma)  # (B, Nctx)
    return alpha * (w * err).mean()
