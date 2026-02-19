from __future__ import annotations

import torch
import torch.nn as nn


@torch.no_grad()
def ema_update(target: nn.Module, source: nn.Module, m: float) -> None:
    for p_t, p_s in zip(target.parameters(), source.parameters()):
        p_t.data.mul_(m).add_(p_s.data, alpha=1.0 - m)
