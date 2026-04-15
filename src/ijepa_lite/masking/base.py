from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class MaskOutput:
    """
    Unified output contract for all maskers, deterministic or learned.

    Hard indices
    ------------
    context_idx / target_idx are LongTensors used directly for token
    gathering in the JEPA forward pass.  No gradients flow through them.

    Soft scores  (learned maskers only)
    ------------
    context_soft / target_soft are floating-point score tensors over all N
    patch positions.  For 2-way maskers (GumbelTopK, PredictorBased) these
    are normalised selection probabilities.  For the 3-way RD masker these
    are the categorical probabilities p_ctx and p_tgt respectively, with
    p_ign stored in aux["p_ign"].

    aux
    ---
    Open dict for diagnostics and intermediate values.
    Additional masker-specific semantic metadata may also live here when a
    rollout needs to preserve richer internal state without changing the main
    tensor contract yet (for example, winners-first metadata alongside legacy
    dense target tensors).
    Keys set by RateDist3WayMasker:
      "lambda"  : (B,) tensor — the λ sample used this step
      "p_ign"   : (B, N) tensor — ignore-class probability (detached)
      "logits"  : (B, N, 3) tensor — raw 3-way logits (detached)
    Keys set by aux_loss (written in-place for metrics):
      "D_soft"  : float — soft-weighted distortion
      "R"       : float — expected context fraction
    """

    context_idx: torch.Tensor                        # (B, Nctx)            always
    target_idx: torch.Tensor                         # (B, Ntgt)|(B, M, K)  always
    context_soft: Optional[torch.Tensor] = None      # (B, N)               learned only
    target_soft: Optional[torch.Tensor] = None       # (B, N)               learned only
    aux: dict = field(default_factory=dict)


class CollateMasker(ABC):
    """
    ABC for CPU-side, DataLoader-worker maskers.

    These run inside IJEPACollate, before any GPU work, in DataLoader worker
    processes.  They must be picklable (no GPU tensors as instance state).
    They have no access to encoder outputs.
    """

    @abstractmethod
    def __call__(self, batch_size: int) -> MaskOutput: ...


class LatentMasker(ABC, nn.Module):
    """
    ABC for GPU-side, learned maskers that run inside IJEPAModel.forward.

    owns_loss
    ---------
    When False (default), aux_loss() returns an *additive* term on top of
    the standard reconstruction loss:
        total = reconstruction_loss + aux_loss(...)

    When True, aux_loss() returns the *complete* training objective and the
    standard reconstruction_loss is used for monitoring only:
        total = aux_loss(...)     # e.g. D_soft + λ·R

    Set owns_loss = True on maskers that define their own rate-distortion
    objective (e.g. RateDist3WayMasker).

    needs_full_tokens
    -----------------
    When True, ijepa.py passes ema_full=(B, N, D) full (uncompressed) EMA
    encoder tokens as a keyword argument to forward().  Required by maskers
    that run geometry cross-attention over all N positions (e.g.
    RateDist3WayMasker with Bayesian surprise scoring).
    When False (default), forward() receives only the compressed (B, M, D)
    tokens.

    aux_loss signature
    ------------------
    All subclasses receive patch_loss as a keyword argument.  Maskers that
    do not need it can ignore it; the default implementation ignores it.
    """

    owns_loss: bool = False
    needs_full_tokens: bool = False

    def __init__(self) -> None:
        nn.Module.__init__(self)

    @abstractmethod
    def forward(
        self,
        tokens: torch.Tensor,
        ema_full: Optional[torch.Tensor] = None,
    ) -> MaskOutput:
        """
        Args:
            tokens   : (B, M, D) compressed EMA encoder tokens.
            ema_full : (B, N, D) full EMA encoder tokens — only provided when
                       needs_full_tokens=True.  Ignored by most maskers.

        Returns:
            MaskOutput with hard indices always populated.
            Soft scores populated when the masker needs gradients.
        """
        ...

    def aux_loss(
        self,
        mask_output: MaskOutput,
        reconstruction_loss: torch.Tensor,
        patch_loss: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            mask_output        : the MaskOutput produced by this masker's forward().
            reconstruction_loss: scalar mean reconstruction loss from ijepa.py.
                                 Used as the full loss when owns_loss=False and
                                 aux_loss returns zero; passed for reference to
                                 maskers that want to define reward signals relative
                                 to it (e.g. REINFORCE baselines).
            patch_loss         : (B, K) per-patch reconstruction loss, or None if
                                 the model is not in compute_patch_loss mode.
                                 Required by RateDist3WayMasker; ignored by simpler
                                 maskers.

        Returns:
            Scalar tensor.
            owns_loss=False → additive term (0.0 by default).
            owns_loss=True  → full training objective replacing reconstruction_loss.
        """
        return reconstruction_loss.new_zeros(())
