# FILE: engine/train_loop.py
from __future__ import annotations

import math

import torch
from torch.amp import GradScaler, autocast

from ijepa_lite.utils.dist import (
    all_reduce_sum,
    barrier,
    is_rank0,
    unwrap_model,
)
from ijepa_lite.utils.meters import AverageMeter
from ijepa_lite.utils.metrics import (
    ema_param_metrics,
    encoder_agreement,
    grad_norm,
    token_metrics,
)


def _linear_ema_momentum(
    start_m: float, end_m: float, step: int, total_steps: int
) -> float:
    """Linear schedule from start_m → end_m over total_steps. Matches original i-JEPA."""
    return start_m + (end_m - start_m) * (step / max(1, total_steps))


def _cosine_wd(wd_start: float, wd_end: float, step: int, total_steps: int) -> float:
    """Cosine schedule from wd_start → wd_end over total_steps. Matches original i-JEPA."""
    progress = step / max(1, total_steps)
    wd = wd_end + (wd_start - wd_end) * 0.5 * (1.0 + math.cos(math.pi * progress))
    if wd_end <= wd_start:
        return max(wd_end, wd)
    return min(wd_end, wd)


def train(
    cfg,
    model,
    loader,
    optimizer,
    scheduler,
    callbacks,
    device,
    resumed_state: dict | None = None,
    wd_start: float | None = None,
    wd_end: float | None = None,
):
    amp = bool(cfg.train.amp) and (device.type == "cuda")
    scaler = GradScaler("cuda", enabled=amp)

    # ------------------------------------------------------------------
    # State — restored from checkpoint when resuming, fresh otherwise
    # ------------------------------------------------------------------
    state = {"epoch": 0, "global_step": 0, "best": None}
    if resumed_state:
        state.update({k: v for k, v in resumed_state.items() if k != "ema_start"})

    # ------------------------------------------------------------------
    # EMA momentum schedule
    # ------------------------------------------------------------------
    if resumed_state and resumed_state.get("ema_start") is not None:
        ema_start = float(resumed_state["ema_start"])
    else:
        ema_start = float(cfg.model.ema_momentum[0])

    ema_end = float(cfg.model.ema_momentum[1])

    # ------------------------------------------------------------------
    # Provide runtime objects to callbacks via PRIVATE state keys.
    # These keys are intentionally filtered out by CheckpointCallback when saving.
    # ------------------------------------------------------------------
    state["_ema_start"] = float(ema_start)
    state["_ckpt_bundle"] = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "scaler": scaler,
    }

    core = unwrap_model(model)
    callbacks.on_run_start(cfg=cfg, state=state, model=core)

    sampler = getattr(loader, "sampler", None)
    log_every = int(cfg.train.log_every)
    clip_norm = float(getattr(cfg.train, "grad_clip_norm", 0.0))

    # ------------------------------------------------------------------
    # WD schedule (no-op when wd_start == wd_end or wd_start is None)
    # ------------------------------------------------------------------
    _wd_start = wd_start if wd_start is not None else 0.0
    _wd_end = wd_end if wd_end is not None else _wd_start
    _do_wd_sched = wd_start is not None and (_wd_start != _wd_end)

    total_steps = int(cfg.train.epochs) * len(loader)

    # ------------------------------------------------------------------
    # Resume fix: checkpoint "epoch" is the last completed epoch, so we resume
    # from epoch+1 (do NOT change saving logic).
    # ------------------------------------------------------------------
    start_epoch = int(state.get("epoch", 0))
    if resumed_state:
        start_epoch += 1

    for epoch in range(start_epoch, int(cfg.train.epochs)):
        state["epoch"] = epoch
        callbacks.on_epoch_start(cfg=cfg, state=state)

        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

        model.train()
        loss_meter = AverageMeter()

        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)

            # ----------------------------------------------------------
            # Masks arrive pre-computed from IJEPACollate (CPU tensors).
            # IJEPACollate returns "context_idx"/"target_idx" directly,
            # not nested under batch["masks"].
            # ----------------------------------------------------------
            masks = None
            if "context_idx" in batch:
                masks = {
                    "context_idx": batch["context_idx"],
                    "target_idx": batch["target_idx"],
                }

            optimizer.zero_grad(set_to_none=True)

            next_step = state["global_step"] + 1
            do_log = next_step % log_every == 0

            with autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                out = model(images, masks=masks, compute_agreement=do_log)
                loss = out["loss"]

            scaler.scale(loss).backward()
            state["global_step"] = next_step

            gnorm = None
            if clip_norm > 0 or do_log:
                if amp:
                    scaler.unscale_(optimizer)
                if clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                if do_log:
                    gnorm = float(grad_norm(model.parameters()))

            scaler.step(optimizer)
            scaler.update()

            # ----------------------------------------------------------
            # WD cosine schedule
            # ----------------------------------------------------------
            if _do_wd_sched:
                new_wd = _cosine_wd(
                    _wd_start, _wd_end, state["global_step"], total_steps
                )
                for pg in optimizer.param_groups:
                    if pg.get("weight_decay", 0.0) > 0.0:
                        pg["weight_decay"] = new_wd

            # ----------------------------------------------------------
            # EMA linear schedule + target encoder update
            # ----------------------------------------------------------
            core = unwrap_model(model)
            core.ema_momentum = _linear_ema_momentum(
                ema_start, ema_end, state["global_step"], total_steps
            )
            core.update_target()

            loss_meter.update(float(loss.item()), n=images.size(0))

            if do_log:
                sum_t = torch.tensor(loss_meter.sum, device=device)
                cnt_t = torch.tensor(loss_meter.count, device=device, dtype=torch.long)
                sum_t = all_reduce_sum(sum_t)
                cnt_t = all_reduce_sum(cnt_t.to(torch.float32))

                global_loss = (sum_t / cnt_t.clamp(min=1.0)).item()
                lr = float(optimizer.param_groups[0]["lr"])

                extra = token_metrics(out["pred"], out["target"])

                if out.get("ctx_tokens_all") is not None:
                    extra.update(
                        encoder_agreement(
                            out["ctx_tokens_all"],
                            out["tgt_tokens_all"],
                        )
                    )

                extra.update(
                    {k: float(v) for k, v in out.get("mask_stats", {}).items()}
                )

                if gnorm is not None:
                    extra["train/grad_norm"] = gnorm

                if is_rank0():
                    extra.update(
                        ema_param_metrics(core.target_encoder, core.context_encoder)
                    )
                    extra["ema/momentum"] = float(core.ema_momentum)

                    if _do_wd_sched:
                        extra["train/weight_decay"] = float(
                            next(
                                pg["weight_decay"]
                                for pg in optimizer.param_groups
                                if pg.get("weight_decay", 0.0) > 0.0
                            )
                        )

                    callbacks.on_step_end(
                        cfg=cfg,
                        state=state,
                        metrics={
                            "train/loss": float(global_loss),
                            "train/lr": lr,
                            "train/epoch": float(epoch),
                            **extra,
                        },
                    )

        if scheduler is not None:
            scheduler.step()

        sum_t = torch.tensor(loss_meter.sum, device=device)
        cnt_t = torch.tensor(loss_meter.count, device=device, dtype=torch.long)
        sum_t = all_reduce_sum(sum_t)
        cnt_t = all_reduce_sum(cnt_t.to(torch.float32))
        epoch_loss = (sum_t / cnt_t.clamp(min=1.0)).item()

        if is_rank0():
            callbacks.on_epoch_end(
                cfg=cfg,
                state=state,
                metrics={
                    "train/epoch_loss": float(epoch_loss),
                    "train/epoch": float(epoch),
                },
            )

            ckpt_path = state.pop("_checkpoint_path", None)
            if ckpt_path:
                callbacks.on_checkpoint_saved(cfg=cfg, state=state, path=str(ckpt_path))

        # Synchronise all ranks at the end of every epoch
        barrier(device)

    if is_rank0():
        callbacks.on_run_end(cfg=cfg, state=state)
