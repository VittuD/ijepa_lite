from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, Optional, TypeVar

import torch
import torch.nn as nn


@dataclass(frozen=True)
class MaskPartition:
    """Hard token partition consumed by the JEPA encoder and predictor."""

    context_idx: torch.Tensor
    target_idx: torch.Tensor
    target_block_counts: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class TwoWayAssignment:
    """Per-position context and target selection probabilities."""

    context: torch.Tensor
    target: torch.Tensor


@dataclass(frozen=True)
class ThreeWayAssignment:
    """Categorical probabilities for context, target, and ignore roles."""

    context: torch.Tensor
    target: torch.Tensor
    ignore: torch.Tensor


@dataclass(frozen=True)
class NWayAssignment:
    """Categorical probabilities for context, target blocks, and ignore."""

    probabilities: torch.Tensor

    @property
    def context(self) -> torch.Tensor:
        return self.probabilities[..., 0]

    @property
    def targets(self) -> torch.Tensor:
        return self.probabilities[..., 1:-1]

    @property
    def target(self) -> torch.Tensor:
        return self.targets.sum(dim=-1)

    @property
    def ignore(self) -> torch.Tensor:
        return self.probabilities[..., -1]


@dataclass(frozen=True)
class TargetScoreAssignment:
    """Independent target scores, not a categorical role distribution."""

    target: torch.Tensor


MaskAssignment = (
    TwoWayAssignment
    | ThreeWayAssignment
    | NWayAssignment
    | TargetScoreAssignment
)
ObjectiveStateT = TypeVar("ObjectiveStateT")


@dataclass
class MaskOutput(Generic[ObjectiveStateT]):
    """Typed output shared by deterministic and learned maskers.

    ``partition`` and ``assignment`` describe mask semantics. Objective inputs
    needed by a learned masker live in its typed ``objective_state``. The open
    ``diagnostics`` mapping is reserved for logging and visualization and must
    not be required to compute the training objective.
    """

    partition: MaskPartition
    assignment: Optional[MaskAssignment] = None
    objective_state: Optional[ObjectiveStateT] = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def context_idx(self) -> torch.Tensor:
        return self.partition.context_idx

    @property
    def target_idx(self) -> torch.Tensor:
        return self.partition.target_idx

    @property
    def context_soft(self) -> Optional[torch.Tensor]:
        assignment = self.assignment
        if isinstance(assignment, (TwoWayAssignment, ThreeWayAssignment, NWayAssignment)):
            return assignment.context
        return None

    @property
    def target_soft(self) -> Optional[torch.Tensor]:
        assignment = self.assignment
        if assignment is None:
            return None
        return assignment.target


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
