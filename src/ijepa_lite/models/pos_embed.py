"""
Positional embedding factory — shared by encoder, predictor, and maskers.

Supports:
  ``"learned"``  — standard learnable embeddings (trunc_normal_ init, std=0.02).
  ``"sincos"``   — fixed 2D sine-cosine embeddings (MAE / ViT style, not learnable).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def _sincos_2d(grid_size: int, dim: int) -> torch.Tensor:
    """Generate 2D sine-cosine positional embeddings.

    Returns (N, dim) where N = grid_size^2.
    First half of channels encode row, second half encode column.
    Within each half: even indices = sin, odd indices = cos.
    """
    assert dim % 4 == 0, f"sincos pos embed requires dim divisible by 4, got {dim}"
    half = dim // 2
    omega = 1.0 / (10000.0 ** (torch.arange(0, half, 2).float() / half))  # (half/2,)

    grid = torch.arange(grid_size, dtype=torch.float32)
    gy, gx = torch.meshgrid(grid, grid, indexing="ij")
    gy = gy.reshape(-1)  # (N,)
    gx = gx.reshape(-1)  # (N,)

    # Row embeddings: (N, half)
    out_y = torch.zeros(gy.shape[0], half)
    angles_y = gy.unsqueeze(1) * omega.unsqueeze(0)  # (N, half/2)
    out_y[:, 0::2] = angles_y.sin()
    out_y[:, 1::2] = angles_y.cos()

    # Column embeddings: (N, half)
    out_x = torch.zeros(gx.shape[0], half)
    angles_x = gx.unsqueeze(1) * omega.unsqueeze(0)  # (N, half/2)
    out_x[:, 0::2] = angles_x.sin()
    out_x[:, 1::2] = angles_x.cos()

    return torch.cat([out_y, out_x], dim=1)  # (N, dim)


def build_pos_embed_2d(
    kind: str,
    grid_size: int,
    dim: int,
) -> nn.Parameter:
    """Build positional embeddings for predictor / maskers.

    Returns ``nn.Parameter`` of shape ``(1, N, dim)`` where ``N = grid_size^2``.
    For ``"sincos"``, ``requires_grad`` is False (frozen).
    """
    num_patches = grid_size * grid_size

    if kind == "learned":
        pe = nn.Parameter(torch.zeros(1, num_patches, dim))
        nn.init.trunc_normal_(pe, std=0.02)
        return pe

    if kind == "sincos":
        pe = _sincos_2d(grid_size, dim).unsqueeze(0)  # (1, N, dim)
        return nn.Parameter(pe, requires_grad=False)

    raise ValueError(f"Unknown pos_embed_kind={kind!r}. Use 'learned' or 'sincos'.")


def build_pos_embed_2d_with_cls(
    kind: str,
    grid_size: int,
    dim: int,
) -> nn.Parameter:
    """Build positional embeddings for the encoder (includes CLS position).

    Returns ``nn.Parameter`` of shape ``(1, 1+N, dim)``.
    CLS position is always zeros (standard ViT convention).
    """
    num_patches = grid_size * grid_size

    if kind == "learned":
        pe = nn.Parameter(torch.zeros(1, 1 + num_patches, dim))
        nn.init.trunc_normal_(pe, std=0.02)
        return pe

    if kind == "sincos":
        patch_pe = _sincos_2d(grid_size, dim)                  # (N, dim)
        cls_pe = torch.zeros(1, dim)                            # (1, dim)
        pe = torch.cat([cls_pe, patch_pe], dim=0).unsqueeze(0)  # (1, 1+N, dim)
        return nn.Parameter(pe, requires_grad=False)

    raise ValueError(f"Unknown pos_embed_kind={kind!r}. Use 'learned' or 'sincos'.")
