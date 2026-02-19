from __future__ import annotations

import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from ijepa_lite.utils.dist import (
    all_reduce_sum,
    is_distributed,
    is_rank0,
    unwrap_model,
)
from ijepa_lite.utils.meters import AverageMeter


class LinearProbeModel(nn.Module):
    """Frozen encoder + trainable linear head."""

    def __init__(self, encoder: nn.Module, head: nn.Module, pool: str = "mean"):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.pool = str(pool)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder(x)  # (B, N, D) patch tokens
        if self.pool == "mean":
            return tokens.mean(dim=1)  # (B, D)
        raise ValueError(f"Unsupported pool={self.pool}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder params have requires_grad=False (set in build.py), so no
        # context manager is needed here — head gradients flow normally.
        feat = self._features(x)
        return self.head(feat)


@torch.no_grad()
def _acc_top1(
    logits: torch.Tensor, y: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    correct = (logits.argmax(dim=1) == y).sum()
    total = torch.tensor(y.numel(), device=y.device, dtype=torch.long)
    return correct, total


def _build_scheduler(optimizer, cfg_sched, epochs: int):
    """
    Build a LR scheduler for the linear head, or return None for constant LR.

    Reads from task.sched in the experiment config:
      name:      "step" | "none"   (default: "none")
      step_size: int               (epochs between decays, default: epochs // 3)
      gamma:     float             (multiplicative decay factor, default: 0.1)

    Example config fragment:
      task:
        sched:
          name: step
          step_size: 150
          gamma: 0.1
    """
    if cfg_sched is None:
        return None

    name = str(getattr(cfg_sched, "name", "none")).lower()

    if name == "none":
        return None

    if name == "step":
        step_size = int(getattr(cfg_sched, "step_size", max(1, epochs // 3)))
        gamma = float(getattr(cfg_sched, "gamma", 0.1))
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )

    raise ValueError(
        f"Unknown linear probe scheduler name='{name}'. Supported: 'step', 'none'."
    )


def linear_probe_eval(
    cfg,
    encoder,
    train_loader,
    val_loader,
    num_classes: int,
    callbacks,
    device,
):
    amp = bool(getattr(cfg.task, "amp", True)) and (device.type == "cuda")
    epochs = int(cfg.train.epochs)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    head = nn.Linear(int(cfg.model.embed_dim), int(num_classes)).to(device)
    model = LinearProbeModel(
        encoder=encoder,
        head=head,
        pool=str(getattr(cfg.task, "pool", "mean")),
    ).to(device)

    if bool(getattr(cfg, "compile", False)) and hasattr(torch, "compile"):
        # Compile the full (encoder+head) graph for linear-probe experiments.
        # Keep it before DDP wrapping, matching the pretrain codepath.
        model = torch.compile(model, dynamic=True)

    if is_distributed():
        from torch.nn.parallel import DistributedDataParallel as DDP

        kwargs = dict(broadcast_buffers=False)
        if device.type == "cuda":
            kwargs.update(device_ids=[device.index], output_device=device.index)
        model = DDP(model, **kwargs)

    # ------------------------------------------------------------------
    # Optimizer + optional scheduler (head params only)
    # ------------------------------------------------------------------
    opt = torch.optim.SGD(
        unwrap_model(model).head.parameters(),
        lr=float(cfg.train.lr),
        momentum=0.9,
        weight_decay=float(cfg.train.weight_decay),
    )
    sched = _build_scheduler(opt, getattr(cfg.task, "sched", None), epochs)
    scaler = GradScaler("cuda", enabled=amp)

    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------
    state = {"epoch": 0, "global_step": 0, "best_acc1": 0.0}
    train_sampler = getattr(train_loader, "sampler", None)
    log_every = int(getattr(cfg.train, "log_every", 50))

    callbacks.on_run_start(cfg=cfg, state=state, model=unwrap_model(model))

    for epoch in range(epochs):
        state["epoch"] = epoch
        callbacks.on_epoch_start(cfg=cfg, state=state)

        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        # --------------------------------------------------------------
        # Train — encoder stays frozen in eval mode
        # --------------------------------------------------------------
        unwrap_model(model).encoder.eval()
        unwrap_model(model).head.train()

        loss_meter = AverageMeter()
        correct_sum = torch.zeros((), device=device, dtype=torch.long)
        total_sum = torch.zeros((), device=device, dtype=torch.long)

        for batch in train_loader:
            x = batch["images"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logits = model(x)
                loss = F.cross_entropy(logits, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            loss_meter.update(float(loss.item()), n=x.size(0))
            state["global_step"] += 1

            with torch.no_grad():
                c, t = _acc_top1(logits, y)
                correct_sum += c
                total_sum += t

            if state["global_step"] % log_every == 0 and is_rank0():
                callbacks.on_step_end(
                    cfg=cfg,
                    state=state,
                    metrics={
                        "probe/train_loss": float(loss_meter.avg),
                        "probe/lr": float(opt.param_groups[0]["lr"]),
                        "probe/epoch": float(epoch),
                    },
                )

        if sched is not None:
            sched.step()

        correct_sum = all_reduce_sum(correct_sum.float())
        total_sum = all_reduce_sum(total_sum.float())
        train_acc1 = (correct_sum / total_sum.clamp(min=1.0)).item()

        # --------------------------------------------------------------
        # Validate
        # --------------------------------------------------------------
        unwrap_model(model).eval()

        val_loss_sum = torch.zeros((), device=device)
        val_correct = torch.zeros((), device=device, dtype=torch.long)
        val_total = torch.zeros((), device=device, dtype=torch.long)

        with torch.no_grad():
            for batch in val_loader:
                x = batch["images"].to(device, non_blocking=True)
                y = batch["labels"].to(device, non_blocking=True)

                with autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    logits = model(x)
                    loss = F.cross_entropy(logits, y)

                val_loss_sum += loss.detach() * x.size(0)
                val_correct += _acc_top1(logits, y)[0]
                val_total += y.numel()

        val_loss_sum = all_reduce_sum(val_loss_sum)
        val_correct = all_reduce_sum(val_correct.float())
        val_total = all_reduce_sum(val_total.float())

        val_loss = (val_loss_sum / val_total.clamp(min=1.0)).item()
        val_acc1 = (val_correct / val_total.clamp(min=1.0)).item()

        if is_rank0():
            callbacks.on_epoch_end(
                cfg=cfg,
                state=state,
                metrics={
                    "probe/train_epoch_loss": float(loss_meter.avg),
                    "probe/train_acc1": float(train_acc1),
                    "probe/val_loss": float(val_loss),
                    "probe/val_acc1": float(val_acc1),
                    "probe/lr": float(opt.param_groups[0]["lr"]),
                    "probe/epoch": float(epoch),
                },
            )

            ckpt_dir = str(getattr(cfg.train, "ckpt_dir", "checkpoints"))
            os.makedirs(ckpt_dir, exist_ok=True)

            payload = {
                "head": unwrap_model(model).head.state_dict(),
                "val_acc1": val_acc1,
                "epoch": epoch,
            }

            if val_acc1 > float(state["best_acc1"]):
                state["best_acc1"] = float(val_acc1)
                torch.save(payload, os.path.join(ckpt_dir, "linear_probe_best.pt"))

            torch.save(payload, os.path.join(ckpt_dir, "linear_probe_last.pt"))

    if is_rank0():
        callbacks.on_run_end(cfg=cfg, state=state)
