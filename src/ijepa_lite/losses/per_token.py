from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PerTokenLoss(nn.Module):
    """
    Token-level regression loss that supports both scalar and per-token output.

    This is a companion to VanillaTokenLoss used exclusively by the learnable
    masking path.

    The key difference is the reduction argument on forward:

      "mean"    scalar, equivalent to VanillaTokenLoss (used as fallback /
                sanity check).
      "none"    returns per-token errors of shape (B, k) by averaging over
                the feature dimension F.  Required by IJEPAModel to form L_W
                before reducing to a scalar.

    Supported kinds: "mse", "smooth_l1", "cosine_mse" — same semantics as
    VanillaTokenLoss so the two are interchangeable for the JEPA loss itself.
    """

    def __init__(self, normalize: bool = False, kind: str = "smooth_l1"):
        super().__init__()
        self.normalize = bool(normalize)
        self.kind = str(kind)

    def forward(
        self,
        pred: torch.Tensor,    # (B, k, F)  or any (..., F)
        target: torch.Tensor,  # same shape as pred
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Args:
            pred:      Predictor output  (..., F).
            target:    EMA target tokens (..., F).
            reduction: "mean" -> scalar loss.
                       "none" -> per-token scalar (...,) by averaging over F.

        Returns:
            Scalar tensor when reduction="mean".
            (...,) tensor  when reduction="none".
        """
        if reduction not in ("mean", "none"):
            raise ValueError(
                f"Unknown reduction={reduction!r}, expected 'mean' or 'none'."
            )

        if self.normalize:
            pred = F.normalize(pred, dim=-1)
            target = F.normalize(target, dim=-1)

        if self.kind == "smooth_l1":
            e = F.smooth_l1_loss(pred, target, reduction="none")   # (..., F)
        elif self.kind == "mse":
            e = F.mse_loss(pred, target, reduction="none")          # (..., F)
        elif self.kind == "cosine_mse":
            p = F.normalize(pred, dim=-1)
            t = F.normalize(target, dim=-1)
            cos = (p * t).sum(dim=-1)                               # (...,)
            e = F.mse_loss(cos, torch.ones_like(cos), reduction="none")
            # cosine_mse already collapsed the feature dim
            if reduction == "mean":
                return e.mean()
            return e   # (...,)
        else:
            raise ValueError(f"Unknown loss kind={self.kind!r}.")

        # e: (..., F)
        if reduction == "none":
            return e.mean(dim=-1)   # (...,)  per-token scalar
        return e.mean()             # scalar