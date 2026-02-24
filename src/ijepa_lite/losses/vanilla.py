# FILE: src/ijepa_lite/losses/vanilla.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VanillaTokenLoss(nn.Module):
    """
    Token-level regression loss for i-JEPA.

    normalize=False (default) matches the original i-JEPA paper, which applies
    plain MSE directly on the target encoder's output features.

    normalize=True applies L2 normalization to both pred and target before MSE,
    making the loss sensitive only to angular distance (magnitude-invariant).

    reduction
    ---------
    "mean"  (default) : scalar loss averaged over all tokens and embedding dims.
                        This is the standard training objective.
    "none"            : returns (B, K) — one scalar per patch token, obtained by
                        computing the per-element loss and averaging over the
                        embedding dimension D.  Used by RateDist3WayLoss and for
                        per-patch diagnostic logging.
    """

    VALID_REDUCTIONS = frozenset({"mean", "none"})

    def __init__(
        self,
        normalize: bool = False,
        kind: str = "mse",
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in self.VALID_REDUCTIONS:
            raise ValueError(
                f"reduction={reduction!r} is invalid. Choose from: {sorted(self.VALID_REDUCTIONS)}"
            )
        self.normalize = bool(normalize)
        self.kind = str(kind)
        self.reduction = str(reduction)

    def forward(
        self,
        pred: torch.Tensor,    # (B, K, D)  or  (B, M, K, D) for multi-block
        target: torch.Tensor,  # same shape as pred
        reduction: str | None = None,  # per-call override; None → use self.reduction
    ) -> torch.Tensor:
        """
        Args:
            pred      : predicted token embeddings.
            target    : target token embeddings.
            reduction : optional per-call override of the instance reduction.
                        Useful when ijepa.py needs the (B, K) form for aux_loss
                        while keeping the instance default as "mean" for eval.

        Returns:
            scalar   if effective reduction is "mean"
            (B, K)   if effective reduction is "none"
                     (multi-block inputs are flattened to (B, M*K) before reduction)
        """
        r = reduction if reduction is not None else self.reduction

        if self.normalize:
            pred = F.normalize(pred, dim=-1)
            target = F.normalize(target, dim=-1)

        if self.kind == "smooth_l1":
            elem = F.smooth_l1_loss(pred, target, reduction="none")  # (..., D)
        elif self.kind == "mse":
            elem = F.mse_loss(pred, target, reduction="none")        # (..., D)
        elif self.kind == "cosine_mse":
            p = F.normalize(pred, dim=-1)
            t = F.normalize(target, dim=-1)
            cos = (p * t).sum(dim=-1)                                # (...) without D
            # cosine_mse has no D dimension — handle reduction directly
            if r == "none":
                # Flatten all leading dims except last spatial one to (B, K)
                return cos.reshape(cos.shape[0], -1)                 # (B, K) or (B, M*K)
            return F.mse_loss(cos, torch.ones_like(cos))
        else:
            raise ValueError(f"Unknown loss kind={self.kind!r}")

        # elem: (..., D) — average over embedding dim to get per-patch scalars
        per_patch = elem.mean(dim=-1)   # (B, K) or (B, M, K)

        if r == "none":
            return per_patch.reshape(per_patch.shape[0], -1)  # (B, K) or (B, M*K)

        return per_patch.mean()  # scalar