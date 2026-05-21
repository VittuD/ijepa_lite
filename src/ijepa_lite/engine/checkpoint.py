from __future__ import annotations

import math
import os
from typing import Optional

import torch
import torch.nn.functional as F

from ijepa_lite.utils.dist import is_rank0, unwrap_model


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
    core.load_state_dict(payload["model"], strict=True)

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

    restored = dict(payload.get("state", {}) or {})
    restored["ema_start"] = payload.get("ema_start", None)
    return restored


def load_model_weights(
    path: str,
    model: torch.nn.Module,
    *,
    strict: bool = False,
) -> None:
    """
    Load model weights from a checkpoint or raw state dict without restoring
    optimizer, scheduler, scaler, or training state.

    This is intended for continued pretraining runs that should start from
    checkpointed weights but run with fresh optimization state and a fresh
    epoch budget.
    """
    if not path:
        raise ValueError("load_model_weights requires a non-empty path.")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    payload = torch.load(path, map_location="cpu", weights_only=True)
    state_dict = (
        payload["model"]
        if isinstance(payload, dict) and "model" in payload
        else payload
    )

    core = unwrap_model(model)
    state_dict, resized_keys = _adapt_weight_init_state_dict(state_dict, core.state_dict())
    incompatible = core.load_state_dict(state_dict, strict=strict)

    if is_rank0():
        mode = "strict" if strict else "non-strict"
        print(f"[init_weights] Loaded model weights from {path} ({mode}).")
        if resized_keys:
            print(f"[init_weights] Resized positional embeddings: {resized_keys}")
        if incompatible.missing_keys:
            print(f"[init_weights] Missing keys: {incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            print(f"[init_weights] Unexpected keys: {incompatible.unexpected_keys}")


def _adapt_weight_init_state_dict(
    state_dict: dict[str, torch.Tensor],
    target_state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[str]]:
    adapted = dict(state_dict)
    resized_keys: list[str] = []

    for key, value in list(adapted.items()):
        target = target_state_dict.get(key)
        if target is None or not isinstance(value, torch.Tensor):
            continue
        if value.shape == target.shape:
            continue
        if _is_pos_embedding_key(key):
            resized = _resize_square_grid_pos_embedding(value, target)
            if resized is not None:
                adapted[key] = resized
                resized_keys.append(key)
                continue
        raise ValueError(
            "Cannot initialize model weights because a checkpoint tensor shape "
            f"does not match the current model: {key} "
            f"ckpt{tuple(value.shape)} != model{tuple(target.shape)}"
        )

    return adapted, resized_keys


def _is_pos_embedding_key(key: str) -> bool:
    return "pos_embed" in key or "pos_embedding" in key


def _square_grid_size(tokens: int) -> int | None:
    grid = int(math.isqrt(tokens))
    return grid if grid * grid == tokens else None


def _resize_square_grid_pos_embedding(
    src: torch.Tensor,
    dst: torch.Tensor,
) -> torch.Tensor | None:
    if src.ndim != 3 or dst.ndim != 3:
        return None
    if src.shape[0] != 1 or dst.shape[0] != 1:
        return None
    if src.shape[2] != dst.shape[2]:
        return None

    src_tokens = int(src.shape[1])
    dst_tokens = int(dst.shape[1])
    src_grid = _square_grid_size(src_tokens)
    dst_grid = _square_grid_size(dst_tokens)
    has_cls = False

    if src_grid is None or dst_grid is None:
        src_grid = _square_grid_size(src_tokens - 1)
        dst_grid = _square_grid_size(dst_tokens - 1)
        has_cls = src_grid is not None and dst_grid is not None

    if src_grid is None or dst_grid is None:
        return None

    cls_pos = src[:, :1, :] if has_cls else None
    patch_pos = src[:, 1:, :] if has_cls else src
    patch_pos = patch_pos.reshape(1, src_grid, src_grid, src.shape[2]).permute(0, 3, 1, 2)
    patch_pos = F.interpolate(
        patch_pos,
        size=(dst_grid, dst_grid),
        mode="bicubic",
        align_corners=False,
    )
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, dst_grid * dst_grid, src.shape[2])
    if cls_pos is not None:
        return torch.cat([cls_pos, patch_pos], dim=1)
    return patch_pos
