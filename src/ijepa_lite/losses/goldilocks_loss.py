"""
Goldilocks curriculum loss for the target-scoring masker.

Computes MSE between the masker's soft scores at selected target positions
and a Gaussian target signal derived from z-scored per-patch reconstruction
errors. Target patches with median error get target=1; tails (too easy or
too hard) get target→0.

The caller controls whether patch_loss is detached. The current pretraining
runtime supplies the live reconstruction tensor; changing that boundary alters
the training objective and must be handled as an explicit scientific change.
"""
from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


class GoldilocksLoss(nn.Module):
    """
    Goldilocks curriculum loss for the target-scoring masker.

    Args
    ----
    z_score_eps : float
        Floor for the z-score denominator. Prevents division by zero on
        batches where all patch losses are identical.
    global_z_score : bool
        If False (default), z-score over only the K selected patches per
        sample. This has a known degenerate fixed point: once the masker
        concentrates on a near-fixed subset of K patches, σ(patch_loss)→0,
        all ê_i→0, all targets→1, and the masker receives no gradient
        differentiation (observed as positional collapse ~halfway through
        training).

        If True, z-score against a running EMA of the global training
        distribution (mean and variance over all patch losses seen during
        training). A consistent positional selection no longer automatically
        produces uniform z-scores, breaking the degenerate fixed point.

        NOTE: the two buffers _running_mean and _running_var are persistent
        and will be saved / restored with checkpoints.
    running_momentum : float
        EMA momentum for the running statistics (only used when
        global_z_score=True). A value of 0.99 gives a half-life of ~100
        updates; 0.999 gives ~700 updates.
    """

    def __init__(
        self,
        z_score_eps: float = 1e-6,
        global_z_score: bool = False,
        running_momentum: float = 0.99,
        log_transform: bool = False,
        correlation_loss: bool = False,
    ) -> None:
        super().__init__()
        self.z_score_eps = float(z_score_eps)
        self.global_z_score = bool(global_z_score)
        self.running_momentum = float(running_momentum)
        self.log_transform = bool(log_transform)
        self.correlation_loss = bool(correlation_loss)

        if not self.global_z_score:
            warnings.warn(
                "GoldilocksLoss: global_z_score=False (default). Z-scoring over only the K "
                "selected patches creates a degenerate fixed point once the masker concentrates "
                "on a fixed subset (sigma->0 -> all targets->1 -> positional collapse). "
                "Set global_z_score=True to z-score against the running training distribution.",
                UserWarning,
                stacklevel=2,
            )

        self.register_buffer("_running_mean", torch.tensor(0.0), persistent=True)
        self.register_buffer("_running_var",  torch.tensor(1.0), persistent=True)

    def forward(
        self,
        p_tgt:      torch.Tensor,  # (B, N)  — full soft scores from masker
        tgt_idx:    torch.Tensor,  # (B, K)  — hard-selected target indices
        patch_loss: torch.Tensor,  # (B, K) or (B, K, D) reconstruction errors
    ) -> torch.Tensor:             # scalar
        if patch_loss.dim() == 3:
            patch_loss = patch_loss.mean(-1)          # (B, K)

        if self.log_transform:
            patch_loss = torch.log(patch_loss.clamp(min=1e-8))

        # Soft masker scores at hard-selected positions
        q_i = p_tgt.gather(1, tgt_idx)               # (B, K)

        if self.global_z_score:
            # Branch B: z-score against the running training distribution
            if self.training:
                batch_mean = patch_loss.mean().detach()
                batch_var  = patch_loss.var().detach()
                m = self.running_momentum
                self._running_mean.mul_(m).add_(batch_mean * (1.0 - m))
                self._running_var .mul_(m).add_(batch_var  * (1.0 - m))

            sigma_global = (self._running_var + self.z_score_eps).sqrt()
            e_hat = (patch_loss - self._running_mean) / sigma_global   # (B, K)
        else:
            # Branch A: z-score per sample over the K selected patches (original behaviour)
            mu    = patch_loss.mean(dim=1, keepdim=True)
            sigma = patch_loss.std(dim=1, keepdim=True)
            e_hat = (patch_loss - mu) / (sigma + self.z_score_eps)  # (B, K)

        # Gaussian Goldilocks target: 1 at z=0 (median), 0 at tails
        target_i = torch.exp(-0.5 * e_hat.pow(2))    # (B, K) ∈ (0, 1]

        if self.correlation_loss:
            # Correlation loss: maximize Pearson correlation between scores
            # and targets. Invariant to mean/scale — only the ranking matters.
            # Avoids the score-suppression fixed point that MSE hits when
            # t̄ < q̄_sel (systematic with symmetric z-scores / log_transform).
            q_c = q_i - q_i.mean(dim=1, keepdim=True)
            t_c = target_i - target_i.mean(dim=1, keepdim=True)
            return -F.cosine_similarity(q_c, t_c, dim=1).mean()

        return (q_i - target_i).pow(2).mean()
