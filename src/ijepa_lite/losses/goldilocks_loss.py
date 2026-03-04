"""
Goldilocks curriculum loss for the target-scoring masker.

Computes MSE between the masker's soft scores at selected target positions
and a Gaussian target signal derived from z-scored per-patch reconstruction
errors. Target patches with median error get target=1; tails (too easy or
too hard) get target→0.

patch_loss must be pre-detached by the caller (enforces gradient isolation).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GoldilocksLoss(nn.Module):
    """
    Goldilocks curriculum loss for the target-scoring masker.

    Args
    ----
    z_score_eps : float
        Floor for the z-score denominator. Prevents division by zero on
        batches where all patch losses are identical.
    """

    def __init__(self, z_score_eps: float = 1e-6) -> None:
        super().__init__()
        self.z_score_eps = float(z_score_eps)

    def forward(
        self,
        p_tgt:      torch.Tensor,  # (B, N)  — full soft scores from masker
        tgt_idx:    torch.Tensor,  # (B, K)  — hard-selected target indices
        patch_loss: torch.Tensor,  # (B, K) or (B, K, D) — detached recon errors
    ) -> torch.Tensor:             # scalar
        if patch_loss.dim() == 3:
            patch_loss = patch_loss.mean(-1)          # (B, K)

        # Soft masker scores at hard-selected positions
        q_i = p_tgt.gather(1, tgt_idx)               # (B, K)

        # Z-score per sample over the K selected patches
        mu    = patch_loss.mean(dim=1, keepdim=True)
        sigma = patch_loss.std(dim=1, keepdim=True)
        e_hat = (patch_loss - mu) / (sigma + self.z_score_eps)  # (B, K)

        # Gaussian Goldilocks target: 1 at z=0 (median), 0 at tails
        target_i = torch.exp(-0.5 * e_hat.pow(2))    # (B, K) ∈ (0, 1]

        return (q_i - target_i).pow(2).mean()
