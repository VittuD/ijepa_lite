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
    This can stabilize early training but is a deviation from the paper — set
    it explicitly in your config if you want it.
    """

    def __init__(self, normalize: bool = False, kind: str = "mse"):
        super().__init__()
        self.normalize = bool(normalize)
        self.kind = str(kind)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            pred = F.normalize(pred, dim=-1)
            target = F.normalize(target, dim=-1)

        if self.kind == "smooth_l1":
            return F.smooth_l1_loss(pred, target)
        if self.kind == "mse":
            return F.mse_loss(pred, target)
        if self.kind == "cosine_mse":
            p = F.normalize(pred, dim=-1)
            t = F.normalize(target, dim=-1)
            cos = (p * t).sum(dim=-1)
            return F.mse_loss(cos, torch.ones_like(cos))
        raise ValueError(f"Unknown loss kind={self.kind}")
