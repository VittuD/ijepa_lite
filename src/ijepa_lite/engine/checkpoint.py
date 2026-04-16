from __future__ import annotations

import os
from typing import Optional

import torch

from ijepa_lite.utils.dist import unwrap_model


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: Optional[torch.amp.GradScaler],
    state: dict,
    ema_start: Optional[float] = None,
    masker_optimizer: Optional[torch.optim.Optimizer] = None,
    masker_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    core = unwrap_model(model)
    payload = {
        "model": core.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "state": state,
        "ema_start": ema_start,
        "masker_optimizer": masker_optimizer.state_dict() if masker_optimizer is not None else None,
        "masker_scheduler": masker_scheduler.state_dict() if masker_scheduler is not None else None,
    }
    torch.save(payload, path)


def load_checkpoint_if_available(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    masker_optimizer: Optional[torch.optim.Optimizer] = None,
    masker_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    strict: bool = True,
    load_training_state: bool = True,
) -> dict:
    """
    Returns the saved state dict (contains global_step, epoch, etc.)
    plus ema_start if present, so the caller can resume the EMA schedule
    from the correct position.

    Returned dict keys:
      - everything from payload["state"]  (epoch, global_step, best, …)
      - "ema_start"  float | None
    """
    if not path or not os.path.exists(path):
        return {}

    payload = torch.load(path, map_location="cpu", weights_only=True)
    core = unwrap_model(model)
    model_state = payload["model"]
    if strict:
        core.load_state_dict(model_state, strict=True)
    else:
        current_state = core.state_dict()
        compatible_state = {
            k: v
            for k, v in model_state.items()
            if k in current_state and tuple(current_state[k].shape) == tuple(v.shape)
        }
        core.load_state_dict(compatible_state, strict=False)

    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    if masker_optimizer is not None and payload.get("masker_optimizer") is not None:
        masker_optimizer.load_state_dict(payload["masker_optimizer"])
    if masker_scheduler is not None and payload.get("masker_scheduler") is not None:
        masker_scheduler.load_state_dict(payload["masker_scheduler"])

    if not load_training_state:
        return {}

    restored = dict(payload.get("state", {}) or {})
    restored["ema_start"] = payload.get("ema_start", None)
    return restored
