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


def _grad_norm_from_grads(grads, norm_type: float = 2.0) -> float:
    grads = [g for g in grads if g is not None]
    if not grads:
        return 0.0
    device = grads[0].device
    total = torch.zeros((), device=device)
    for g in grads:
        total += g.detach().pow(norm_type).sum()
    return float(total.pow(1.0 / norm_type).item())


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
    masker_optimizer=None,
    masker_scheduler=None,
    masker_wd_start: float | None = None,
    masker_wd_end: float | None = None,
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
        ema_start = float(getattr(cfg.model, "ema_momentum", [0.0, 0.0])[0])

    ema_end = float(getattr(cfg.model, "ema_momentum", [ema_start, ema_start])[1])

    # ------------------------------------------------------------------
    # Provide runtime objects to callbacks via PRIVATE state keys.
    # These keys are intentionally filtered out by CheckpointCallback when saving.
    # ------------------------------------------------------------------
    state["_ema_start"] = float(ema_start) if getattr(core := unwrap_model(model), "has_ema_target", False) else None
    state["_ckpt_bundle"] = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "scaler": scaler,
        "masker_optimizer": masker_optimizer,
        "masker_scheduler": masker_scheduler,
    }

    callbacks.on_run_start(cfg=cfg, state=state, model=core)
    # Rank-0 does more work in on_run_start (wandb.init, dataset loads, etc.).
    # Without this barrier the other ranks can reach loss.backward() → DDP
    # all_reduce before rank 0 has left on_run_start, causing a collective hang.
    barrier(device)

    # Unwrap once here; DDP wrapping doesn't change between epochs.
    # (The variable is reused inside the loop for EMA/metrics without re-wrapping.)

    sampler = getattr(loader, "sampler", None)
    log_every = int(cfg.train.log_every)
    clip_norm = float(getattr(cfg.train, "grad_clip_norm", 0.0))
    masker_step_every = int(getattr(cfg.train, "masker_step_every", 1))
    masker_reset_every = int(getattr(cfg.train, "masker_reset_every", -1))
    _prog_kl_meter = AverageMeter()  # per-epoch progressive KL average (for adaptive mode)

    # ------------------------------------------------------------------
    # WD schedule (no-op when wd_start == wd_end or wd_start is None)
    # ------------------------------------------------------------------
    _wd_start = wd_start if wd_start is not None else 0.0
    _wd_end = wd_end if wd_end is not None else _wd_start
    _do_wd_sched = wd_start is not None and (_wd_start != _wd_end)

    _masker_wd_start = masker_wd_start if masker_wd_start is not None else _wd_start
    _masker_wd_end = masker_wd_end if masker_wd_end is not None else _masker_wd_start
    _do_masker_wd_sched = masker_optimizer is not None and (_masker_wd_start != _masker_wd_end)

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

        # Grow λ sampling range according to warmup schedule.
        # warmup=0 means disabled: _progress stays at its init value (1.0 = full range).
        _masker = getattr(core, "latent_masker", None)
        if _masker is not None and hasattr(_masker, "set_progress"):
            warmup = getattr(_masker, "warmup_epochs", 0)
            if warmup > 0:
                _masker.set_progress(epoch / warmup)

        # True only during vanilla-multiblock warmup epochs where the masker is
        # NOT trained on its own soft assignments.  In that regime the masker
        # backward produces exactly-zero gradients (0 * anchor.sum()), so we must
        # not call scaler.step(masker_optimizer): Adam would increment its step
        # counter without accumulating m/v, causing a ~3.16× LR overshoot on the
        # first real gradient step after warmup ends.
        _masker_in_warmup_no_train = (
            _masker is not None
            and getattr(_masker, "warmup_use_vanilla_multiblock", False)
            and not getattr(_masker, "warmup_train_masker_on_soft_assignments", False)
            and epoch < getattr(_masker, "warmup_epochs", 0)
        )

        # ----------------------------------------------------------
        # Masker reset trigger: full wipe — weights, Adam state, LR
        # ----------------------------------------------------------
        if masker_reset_every > 0 and epoch > 0 and epoch % masker_reset_every == 0 and _masker is not None:
            # 1. Re-init masker weights (and phase state if progressive KL is active)
            if hasattr(_masker, "reset_parameters"):
                _masker.reset_parameters()

            # 2. Wipe Adam m/v/step buffers and restore initial LR
            if masker_optimizer is not None:
                masker_optimizer.state.clear()
                for pg in masker_optimizer.param_groups:
                    if "initial_lr" in pg:
                        pg["lr"] = pg["initial_lr"]
            elif len(optimizer.param_groups) > 1:
                for p in optimizer.param_groups[-1]["params"]:
                    optimizer.state.pop(p, None)
                pg = optimizer.param_groups[-1]
                if "initial_lr" in pg:
                    pg["lr"] = pg["initial_lr"]

            # 3. Restart the LR schedule from epoch 0
            #    (overwrites the manual lr set above with lambda(0) * initial_lr)
            if masker_scheduler is not None:
                masker_scheduler.last_epoch = -1
                masker_scheduler.step()

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
            if masker_optimizer is not None:
                masker_optimizer.zero_grad(set_to_none=True)

            next_step = state["global_step"] + 1
            do_log = next_step % log_every == 0

            with autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                out = model(images, masks=masks, compute_agreement=do_log, compute_mask_metrics=do_log, epoch=epoch)
                loss = out["loss"]

            rec_encoder_gnorm = None
            sigreg_encoder_gnorm = None
            if do_log:
                encoder_params = [
                    p for p in core.context_encoder.parameters() if p.requires_grad
                ]
                rec_grads = torch.autograd.grad(
                    out["reconstruction_loss"],
                    encoder_params,
                    retain_graph=True,
                    allow_unused=True,
                )
                rec_encoder_gnorm = _grad_norm_from_grads(rec_grads)

                sigreg_term = out.get("sigreg_loss_weighted")
                if sigreg_term is not None:
                    sigreg_grads = torch.autograd.grad(
                        sigreg_term,
                        encoder_params,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    sigreg_encoder_gnorm = _grad_norm_from_grads(sigreg_grads)

            scaler.scale(loss).backward()
            state["global_step"] = next_step

            gnorm = None
            base_gnorm = None
            masker_gnorm = None
            if clip_norm > 0 or do_log:
                if amp:
                    scaler.unscale_(optimizer)
                    if masker_optimizer is not None:
                        scaler.unscale_(masker_optimizer)
                if clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                if do_log:
                    gnorm = float(grad_norm(model.parameters()))
                    masker_params = []
                    if _masker is not None:
                        masker_params.extend(_masker.parameters())
                    _compressor = getattr(core, "token_compressor", None)
                    if _compressor is not None:
                        masker_params.extend(_compressor.parameters())
                    if masker_params:
                        masker_gnorm = float(grad_norm(masker_params))
                        masker_param_ids = {id(p) for p in masker_params}
                        base_gnorm = float(
                            grad_norm(
                                p for p in model.parameters()
                                if id(p) not in masker_param_ids
                            )
                        )

            is_masker_step = masker_step_every == 1 or next_step % masker_step_every == 0

            if masker_optimizer is not None:
                # Separate optimizer path: skip masker step on non-masker steps,
                # and also skip during warmup-no-train to prevent Adam phantom steps.
                scaler.step(optimizer)
                if is_masker_step and not _masker_in_warmup_no_train:
                    scaler.step(masker_optimizer)
            else:
                # Shared optimizer path: zero masker grads on non-masker steps
                if masker_step_every > 1 and not is_masker_step:
                    if len(optimizer.param_groups) > 1:
                        for p in optimizer.param_groups[-1]["params"]:
                            p.grad = None
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
            if _do_masker_wd_sched:
                new_masker_wd = _cosine_wd(
                    _masker_wd_start, _masker_wd_end, state["global_step"], total_steps
                )
                for pg in masker_optimizer.param_groups:
                    if pg.get("weight_decay", 0.0) > 0.0:
                        pg["weight_decay"] = new_masker_wd

            # ----------------------------------------------------------
            # EMA linear schedule + target encoder update
            # ----------------------------------------------------------
            if core.has_ema_target:
                core.ema_momentum = _linear_ema_momentum(
                    ema_start, ema_end, state["global_step"], total_steps
                )
                core.update_target()
                if _masker is not None and hasattr(_masker, "set_ema_decay"):
                    _masker.set_ema_decay(core.ema_momentum)
            if _masker is not None and hasattr(_masker, "set_step"):
                _masker.set_step(state["global_step"], total_steps)

            loss_meter.update(float(loss.item()), n=images.size(0))

            # Track progressive KL every step for adaptive phase switching.
            _pkv = out.get("mask_stats", {}).get("mask/prog_kl/loss")
            if _pkv is not None:
                _prog_kl_meter.update(float(_pkv))

            if do_log:
                sum_t = torch.tensor(loss_meter.sum, device=device)
                cnt_t = torch.tensor(loss_meter.count, device=device, dtype=torch.long)
                sum_t = all_reduce_sum(sum_t)
                cnt_t = all_reduce_sum(cnt_t.to(torch.float32))

                global_loss = (sum_t / cnt_t.clamp(min=1.0)).item()
                lr = float(optimizer.param_groups[0]["lr"])
                if masker_optimizer is not None:
                    masker_lr = float(masker_optimizer.param_groups[0]["lr"])
                elif len(optimizer.param_groups) > 1:
                    masker_lr = float(optimizer.param_groups[-1]["lr"])
                else:
                    masker_lr = lr

                extra = token_metrics(out["pred"], out["target"])

                if out.get("ctx_tokens_all") is not None:
                    extra.update(
                        encoder_agreement(
                            out["ctx_tokens_all"],
                            out["tgt_tokens_all"],
                        )
                    )

                for k, v in out.get("mask_stats", {}).items():
                    if str(k).startswith("_hist/"):
                        extra[k] = v  # numpy array — passed through to WandbCallback
                    else:
                        extra[k] = float(v)
                for k, v in out.get("model_stats", {}).items():
                    extra[k] = float(v)

                if gnorm is not None:
                    extra["train/grad_norm"] = gnorm
                if base_gnorm is not None:
                    extra["train/base_grad_norm"] = base_gnorm
                if masker_gnorm is not None:
                    extra["train/masker_grad_norm"] = masker_gnorm
                if rec_encoder_gnorm is not None:
                    extra["train/rec_encoder_grad_norm"] = rec_encoder_gnorm
                if sigreg_encoder_gnorm is not None:
                    extra["train/sigreg_encoder_grad_norm"] = sigreg_encoder_gnorm

                if is_rank0():
                    if core.has_ema_target:
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

                    metrics = {
                        "train/loss": float(global_loss),
                        "train/reconstruction_loss": float(out["reconstruction_loss"].item()),
                        "train/lr": lr,
                        "train/epoch": float(epoch),
                        **extra,
                    }
                    if "sigreg/loss_weighted" in extra:
                        metrics["train/sigreg_loss_weighted"] = float(
                            extra["sigreg/loss_weighted"]
                        )
                    if out.get("ctx_loss") is not None:
                        metrics["train/ctx_loss"] = out["ctx_loss"]
                    if masker_lr != lr:
                        metrics["train/masker_lr"] = masker_lr

                    callbacks.on_step_end(
                        cfg=cfg,
                        state=state,
                        metrics=metrics,
                    )

        if scheduler is not None:
            scheduler.step()
        if masker_scheduler is not None:
            masker_scheduler.step()

        # Adaptive phase switching: notify masker of per-epoch KL average.
        if _masker is not None and hasattr(_masker, "on_epoch_kl"):
            avg_kl = _prog_kl_meter.avg if _prog_kl_meter.count > 0 else float("inf")
            _masker.on_epoch_kl(avg_kl)
        _prog_kl_meter = AverageMeter()  # reset for next epoch

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
