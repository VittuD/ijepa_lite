"""
Example LatentMasker implementation: Gumbel-TopK.

This is a scaffold that demonstrates the full pattern — how to implement
the forward pass, the soft/hard split, and the gradient routing via aux_loss.
Use it as a starting point and replace the score network and loss with your
own design.

To activate this masker in your config:
    masking:
      latent:
        name: gumbel_topk
        temperature: 1.0
        entropy_coeff: 0.01
      compressor:
        mode: full   # or any other mode

Register it by importing this module once before build_latent_masker is called.
The simplest way is to add the import to your run.py or build.py:
    import ijepa_lite.masking.example_latent_masker  # noqa: F401  registers the class
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ijepa_lite.masking.base import (
    LatentMasker,
    MaskOutput,
    MaskPartition,
    TwoWayAssignment,
)
from ijepa_lite.masking.registry import register


@register("gumbel_topk")
class GumbelTopKMasker(LatentMasker):
    """
    Selects target and context patches via Gumbel-TopK over a learned score distribution.

    Architecture
    ------------
    1.  Mean-pool the M compressed tokens to a single (B, D) vector.
    2.  A linear head produces N logits — one per patch position.
    3.  During training: add Gumbel noise and apply softmax (differentiable approximation
        of the selection distribution).
        During eval: plain softmax over logits.
    4.  Hard indices: top-ntgt patches for targets, then top-nctx from the remainder.
    5.  Soft scores are stored in MaskOutput for aux_loss.

    Gradient flow
    -------------
    context_idx / target_idx are LongTensors — gradients do NOT flow through them.
    The gradient path from the reconstruction loss back to masker parameters goes
    through aux_loss(), which applies entropy regularisation over the soft scores.

    For straight-through or REINFORCE, override aux_loss() in a subclass.

    Args
    ----
    dim            : token embedding dim (must match compressor output dim)
    num_patches    : total number of patch positions N
    target_ratio   : fraction of patches to select as targets
    context_ratio  : fraction of patches to select as context
    temperature    : Gumbel-softmax temperature (default 1.0)
    entropy_coeff  : weight on entropy regularisation in aux_loss (default 0.01)
                     Positive = encourage higher entropy = more uniform selection.
                     Negative = encourage lower entropy = more decisive selection.
    """

    def __init__(
        self,
        dim: int,
        num_patches: int,
        target_ratio: float,
        context_ratio: float,
        temperature: float = 1.0,
        entropy_coeff: float = 0.01,
    ) -> None:
        super().__init__()

        self.num_patches = int(num_patches)
        self.ntgt = max(1, int(round(self.num_patches * float(target_ratio))))
        self.nctx = max(1, int(round(self.num_patches * float(context_ratio))))
        self.temperature = float(temperature)
        self.entropy_coeff = float(entropy_coeff)

        # Score network: compressed token pool → N patch logits
        # Input is mean-pooled over the M sequence dimension → (B, D)
        self.score_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, self.num_patches),
        )
        # Initialise near-zero so early training starts close to uniform sampling
        nn.init.zeros_(self.score_head[-1].weight)
        nn.init.zeros_(self.score_head[-1].bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, tokens: torch.Tensor) -> MaskOutput:
        """
        Args:
            tokens: (B, M, D) compressed EMA encoder tokens.

        Returns:
            MaskOutput with hard indices and soft scores.
        """
        B = tokens.shape[0]

        # Pool compressed tokens → single vector per image
        pooled = tokens.mean(dim=1)  # (B, D)

        # Score each of the N patch positions
        logits = self.score_head(pooled)  # (B, N)

        # Gumbel-softmax: differentiable relaxation of the sampling distribution
        if self.training:
            gumbel_noise = _sample_gumbel(logits)
            soft_scores = F.softmax((logits + gumbel_noise) / self.temperature, dim=-1)
        else:
            soft_scores = F.softmax(logits / self.temperature, dim=-1)

        # ----------------------------------------------------------------
        # Hard selection: top-ntgt for target, top-nctx from the remainder
        # ----------------------------------------------------------------
        _, tgt_idx = torch.topk(soft_scores, self.ntgt, dim=-1, sorted=False)  # (B, ntgt)

        # Zero out target positions so context selection ignores them
        ctx_scores = soft_scores.clone()
        ctx_scores.scatter_(1, tgt_idx, 0.0)
        # Re-normalise (avoids zero-sum edge cases on tiny grids)
        ctx_scores = ctx_scores / (ctx_scores.sum(dim=-1, keepdim=True) + 1e-10)

        _, ctx_idx = torch.topk(ctx_scores, self.nctx, dim=-1, sorted=False)  # (B, nctx)

        return MaskOutput(
            partition=MaskPartition(context_idx=ctx_idx, target_idx=tgt_idx),
            assignment=TwoWayAssignment(
                context=ctx_scores,
                target=soft_scores,
            ),
            diagnostics={"logits": logits.detach()},
        )

    # ------------------------------------------------------------------
    # Auxiliary loss — gradient routing back to masker parameters
    # ------------------------------------------------------------------

    def aux_loss(
        self,
        mask_output: MaskOutput,
        reconstruction_loss: torch.Tensor,
        patch_loss=None,  # accepted but not used; kept for interface consistency
    ) -> torch.Tensor:
        """
        Entropy regularisation over the target selection distribution.

        Positive entropy_coeff → penalise low-entropy distributions (encourage
            the masker to spread its selections, avoiding degenerate solutions
            where only a few patches are ever selected).
        Negative entropy_coeff → reward sharper, more decisive selections.

        The reconstruction_loss is available here if you want to define a
        reward signal relative to it (e.g. REINFORCE baseline).

        Override this method in a subclass to implement different gradient
        routing strategies without changing forward().
        """
        if not isinstance(mask_output.assignment, TwoWayAssignment):
            return reconstruction_loss.new_zeros(())

        soft = mask_output.assignment.target
        # Shannon entropy over patch-selection distribution, averaged over batch
        entropy = -(soft * (soft + 1e-10).log()).sum(dim=-1).mean()

        return -self.entropy_coeff * entropy


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------

def _sample_gumbel(like: torch.Tensor) -> torch.Tensor:
    """Sample Gumbel noise with the same shape and device as `like`."""
    u = torch.zeros_like(like).uniform_().clamp_(1e-10, 1.0 - 1e-10)
    return -(-u.log()).log()
