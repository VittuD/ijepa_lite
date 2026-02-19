from __future__ import annotations

import datetime
import os
from typing import Optional

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def get_local_rank() -> int:
    # torchrun sets LOCAL_RANK
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_rank0() -> bool:
    return get_rank() == 0


def maybe_init_distributed(cfg) -> None:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world <= 1:
        return

    backend = str(cfg.distributed.backend)
    if backend == "nccl" and not torch.cuda.is_available():
        backend = "gloo"

    timeout = datetime.timedelta(
        minutes=int(getattr(cfg.distributed, "timeout_minutes", 30))
    )

    # Ensure device is set before init/barrier on NCCL to avoid "devices used by this process are unknown".
    if backend == "nccl" and torch.cuda.is_available():
        local_rank = get_local_rank()
        torch.cuda.set_device(local_rank)

    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://", timeout=timeout)


def setup_device(cfg) -> torch.device:
    want_cuda = str(getattr(cfg, "device", "cuda")).startswith("cuda")
    if torch.cuda.is_available() and want_cuda:
        local_rank = get_local_rank()
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def barrier(device: Optional[torch.device] = None) -> None:
    """
    NCCL barrier should specify device_ids to avoid warnings/hangs when device mapping is not yet known.
    """
    if not is_distributed():
        return

    backend = dist.get_backend()
    if backend == "nccl" and torch.cuda.is_available():
        if device is None:
            # use current cuda device
            dev_idx = torch.cuda.current_device()
        else:
            # accept torch.device("cuda:X") or plain cuda device
            dev_idx = (
                device.index
                if device.type == "cuda" and device.index is not None
                else torch.cuda.current_device()
            )
        dist.barrier(device_ids=[dev_idx])
    else:
        dist.barrier()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


@torch.no_grad()
def all_reduce_sum(t: torch.Tensor) -> torch.Tensor:
    if is_distributed():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


@torch.no_grad()
def all_reduce_mean(t: torch.Tensor) -> torch.Tensor:
    if is_distributed():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t.div_(float(get_world_size()))
    return t


def cleanup_distributed(device: Optional[torch.device] = None) -> None:
    """
    Best-effort distributed cleanup.
    """
    if not is_distributed():
        return

    try:
        barrier(device)
    except Exception:
        # Barrier can fail during teardown if another rank already died/exited.
        pass

    try:
        dist.destroy_process_group()
    except Exception:
        pass
