from __future__ import annotations

import math
import os
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from ijepa_lite.engine.eval_linear import _build_head, _build_optimizer, _build_scheduler
from ijepa_lite.utils.dist import (
    all_reduce_sum,
    get_world_size,
    is_distributed,
    is_rank0,
    unwrap_model,
)
from ijepa_lite.utils.meters import AverageMeter


class BoxHeatmapProbeModel(nn.Module):
    def __init__(self, encoder: nn.Module, head: nn.Module, pool: str = "mean") -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.pool = str(pool)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if self.pool == "mean":
                feat = self.encoder(x)
            elif self.pool == "last4_mean":
                if not hasattr(self.encoder, "forward_last_n"):
                    raise ValueError(
                        "pool=last4_mean requires an encoder with forward_last_n support."
                    )
                layer_tokens = self.encoder.forward_last_n(x, last_n=4)
                feat = torch.cat(layer_tokens, dim=-1)
            else:
                raise ValueError(f"Unsupported box heatmap pool={self.pool}")
        return F.layer_norm(feat, (feat.shape[-1],))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self._features(x)
        bsz, n_patches, dim = feat.shape
        grid = int(math.isqrt(n_patches))
        if grid * grid != n_patches:
            raise ValueError(
                f"Box heatmap probe expects a square patch grid, got n_patches={n_patches}."
            )
        logits = self.head(feat.reshape(bsz * n_patches, dim)).view(bsz, n_patches)
        return logits


def _rasterize_box_targets(
    targets: list[dict[str, Any]],
    *,
    grid: int,
    device: torch.device,
    positive_radius: int,
) -> torch.Tensor:
    heatmaps = torch.zeros((len(targets), grid, grid), device=device)
    radius = max(0, int(positive_radius))
    for bidx, target in enumerate(targets):
        boxes = target["boxes"].to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
        if boxes.numel() == 0:
            continue
        for box in boxes:
            x0 = max(0, int(torch.floor(box[0] * grid).item()) - radius)
            y0 = max(0, int(torch.floor(box[1] * grid).item()) - radius)
            x1 = min(grid, int(torch.ceil(box[2] * grid).item()) + radius)
            y1 = min(grid, int(torch.ceil(box[3] * grid).item()) + radius)
            if x1 <= x0 or y1 <= y0:
                continue
            heatmaps[bidx, y0:y1, x0:x1] = 1.0
    return heatmaps.reshape(len(targets), grid * grid)


def _sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float,
    gamma: float,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    prob = torch.sigmoid(logits)
    p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
    alpha_t = float(alpha) * targets + (1.0 - float(alpha)) * (1.0 - targets)
    return (alpha_t * (1.0 - p_t).pow(float(gamma)) * bce).mean()


def _soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits.float())
    intersection = (probs * targets).sum(dim=1)
    denom = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()


def _box_heatmap_loss(
    logits: torch.Tensor,
    heatmaps: torch.Tensor,
    *,
    loss_kind: str,
    focal_alpha: float,
    focal_gamma: float,
    bce_weight: float,
    dice_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if loss_kind == "focal":
        bce = _sigmoid_focal_loss(
            logits,
            heatmaps,
            alpha=focal_alpha,
            gamma=focal_gamma,
        )
    elif loss_kind in {"bce", "weighted_bce"}:
        pos_weight = None
        if loss_kind == "weighted_bce":
            n_pos = heatmaps.sum()
            n_neg = heatmaps.numel() - n_pos
            pos_weight = (n_neg / n_pos.clamp(min=1.0)).clamp(min=1.0)
        bce = F.binary_cross_entropy_with_logits(logits, heatmaps, pos_weight=pos_weight)
    else:
        raise ValueError(
            f"Unknown box heatmap loss='{loss_kind}'. Supported: focal|bce|weighted_bce."
        )
    dice = _soft_dice_loss(logits, heatmaps)
    loss = float(bce_weight) * bce + float(dice_weight) * dice
    return loss, {
        "bce_loss": bce.detach(),
        "dice_loss": dice.detach(),
    }


def _gather_tensors(parts: list[torch.Tensor]) -> torch.Tensor:
    local = torch.cat([part.detach().cpu() for part in parts], dim=0) if parts else torch.empty(0)
    if not is_distributed():
        return local
    gathered: list[torch.Tensor | None] = [None for _ in range(get_world_size())]
    dist.all_gather_object(gathered, local)
    valid = [part for part in gathered if part is not None and part.numel() > 0]
    return torch.cat(valid, dim=0) if valid else torch.empty(0)


def _binary_auc(scores: torch.Tensor, targets: torch.Tensor) -> float:
    scores = scores.reshape(-1).double()
    targets = targets.reshape(-1).bool()
    n_pos = int(targets.sum().item())
    n_total = int(targets.numel())
    n_neg = n_total - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0
    order = torch.argsort(scores)
    sorted_scores = scores[order]
    ranks = torch.empty(n_total, dtype=torch.float64)
    start = 0
    while start < n_total:
        end = start + 1
        while end < n_total and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    pos_rank_sum = ranks[targets].sum().item()
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _binary_ap(scores: torch.Tensor, targets: torch.Tensor) -> float:
    scores = scores.reshape(-1).double()
    targets = targets.reshape(-1).bool()
    n_pos = int(targets.sum().item())
    if n_pos == 0 or scores.numel() == 0:
        return 0.0
    order = torch.argsort(scores, descending=True)
    sorted_targets = targets[order].double()
    tp = sorted_targets.cumsum(0)
    precision = tp / torch.arange(1, sorted_targets.numel() + 1, dtype=torch.float64)
    return float((precision * sorted_targets).sum().item() / float(n_pos))


def _dice_at_threshold(
    scores: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
    eps: float = 1e-6,
) -> float:
    pred = scores.reshape(-1) >= float(threshold)
    target = targets.reshape(-1).bool()
    intersection = (pred & target).sum().float()
    denom = pred.sum().float() + target.sum().float()
    return float(((2.0 * intersection + eps) / (denom + eps)).item())


def _metrics_from_parts(
    patch_score_parts: list[torch.Tensor],
    patch_target_parts: list[torch.Tensor],
    image_score_parts: list[torch.Tensor],
    image_target_parts: list[torch.Tensor],
    *,
    threshold: float,
) -> dict[str, float]:
    patch_scores = _gather_tensors(patch_score_parts)
    patch_targets = _gather_tensors(patch_target_parts)
    image_scores = _gather_tensors(image_score_parts)
    image_targets = _gather_tensors(image_target_parts)
    return {
        "patch_ap": _binary_ap(patch_scores, patch_targets),
        "patch_auroc": _binary_auc(patch_scores, patch_targets),
        "heatmap_dice": _dice_at_threshold(patch_scores, patch_targets, threshold),
        "image_auroc": _binary_auc(image_scores, image_targets),
        "num_images": float(image_targets.numel()),
        "num_positive_patches": float(patch_targets.sum().item()),
    }


def box_heatmap_probe_eval(
    cfg,
    encoder,
    train_loader,
    val_loader,
    num_classes: int,
    callbacks,
    device,
):
    del num_classes
    amp = bool(getattr(cfg.task, "amp", True)) and (device.type == "cuda")
    epochs = int(cfg.train.epochs)
    pool = str(getattr(cfg.task, "pool", "mean"))
    head_in_dim = int(cfg.model.embed_dim) * (4 if pool == "last4_mean" else 1)
    grid = int(cfg.model.image_size) // int(cfg.model.patch_size)
    primary_metric = str(getattr(cfg.task, "metric", "patch_ap")).lower()
    positive_radius = int(getattr(cfg.task, "positive_radius", 0))
    loss_kind = str(getattr(cfg.task, "loss", "focal")).lower()
    focal_alpha = float(getattr(cfg.task, "focal_alpha", 0.25))
    focal_gamma = float(getattr(cfg.task, "focal_gamma", 2.0))
    bce_weight = float(getattr(cfg.task, "bce_weight", 1.0))
    dice_weight = float(getattr(cfg.task, "dice_weight", 1.0))
    threshold = float(getattr(cfg.task, "threshold", 0.5))

    head = _build_head(head_in_dim, 1, getattr(cfg.task, "head", None)).to(device)
    model = BoxHeatmapProbeModel(encoder=encoder, head=head, pool=pool).to(device)

    if bool(getattr(cfg, "compile", False)) and hasattr(torch, "compile"):
        model = torch.compile(model, dynamic=True)

    unwrap_model(model).encoder.requires_grad_(False)
    unwrap_model(model).encoder.eval()

    if is_distributed():
        from torch.nn.parallel import DistributedDataParallel as DDP

        kwargs = dict(
            broadcast_buffers=False,
            find_unused_parameters=bool(
                getattr(getattr(cfg, "distributed", None), "find_unused_parameters", False)
            ),
        )
        if device.type == "cuda":
            kwargs.update(device_ids=[device.index], output_device=device.index)
        model = DDP(model, **kwargs)

    opt = _build_optimizer(model, cfg)
    sched = _build_scheduler(opt, getattr(cfg.task, "sched", None), epochs)
    scaler = GradScaler("cuda", enabled=amp)

    state = {"epoch": 0, "global_step": 0, "best_metric": 0.0}
    log_every = int(getattr(cfg.train, "log_every", 50))
    early_stop_patience = int(getattr(cfg.train, "early_stop_patience", 0))
    early_stop_min_epochs = int(getattr(cfg.train, "early_stop_min_epochs", 0))
    early_stop_min_delta = float(getattr(cfg.train, "early_stop_min_delta", 0.0))
    no_improve_epochs = 0
    best_epoch = -1
    best_metrics: dict[str, float] = {}
    last_metrics: dict[str, float] = {}
    last_train_metric = 0.0
    last_val_loss = 0.0

    callbacks.on_run_start(cfg=cfg, state=state, model=unwrap_model(model))

    for epoch in range(epochs):
        state["epoch"] = epoch
        callbacks.on_epoch_start(cfg=cfg, state=state)
        unwrap_model(model).head.train()

        loss_meter = AverageMeter()
        bce_loss_meter = AverageMeter()
        dice_loss_meter = AverageMeter()
        train_score_parts: list[torch.Tensor] = []
        train_target_parts: list[torch.Tensor] = []
        train_image_score_parts: list[torch.Tensor] = []
        train_image_target_parts: list[torch.Tensor] = []

        for batch in train_loader:
            x = batch["images"].to(device, non_blocking=True)
            targets = batch["targets"]
            heatmaps = _rasterize_box_targets(
                targets,
                grid=grid,
                device=device,
                positive_radius=positive_radius,
            )
            opt.zero_grad(set_to_none=True)
            with autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logits = model(x)
                loss, loss_parts = _box_heatmap_loss(
                    logits,
                    heatmaps,
                    loss_kind=loss_kind,
                    focal_alpha=focal_alpha,
                    focal_gamma=focal_gamma,
                    bce_weight=bce_weight,
                    dice_weight=dice_weight,
                )

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            loss_meter.update(float(loss.item()), n=x.size(0))
            bce_loss_meter.update(float(loss_parts["bce_loss"].item()), n=x.size(0))
            dice_loss_meter.update(float(loss_parts["dice_loss"].item()), n=x.size(0))
            state["global_step"] += 1

            with torch.no_grad():
                probs = torch.sigmoid(logits.detach()).float().cpu()
                heat_cpu = heatmaps.detach().float().cpu()
                train_score_parts.append(probs.reshape(-1))
                train_target_parts.append(heat_cpu.reshape(-1))
                train_image_score_parts.append(probs.max(dim=1).values)
                train_image_target_parts.append((heat_cpu.sum(dim=1) > 0).float())

            if state["global_step"] % log_every == 0 and is_rank0():
                callbacks.on_step_end(
                    cfg=cfg,
                    state=state,
                    metrics={
                        "probe/train_loss": float(loss_meter.avg),
                        "probe/train_bce_loss": float(bce_loss_meter.avg),
                        "probe/train_dice_loss": float(dice_loss_meter.avg),
                        "probe/lr": float(opt.param_groups[0]["lr"]),
                        "probe/epoch": float(epoch),
                    },
                )

        if sched is not None:
            sched.step()

        train_metrics = _metrics_from_parts(
            train_score_parts,
            train_target_parts,
            train_image_score_parts,
            train_image_target_parts,
            threshold=threshold,
        )
        train_primary = float(train_metrics.get(primary_metric, train_metrics["patch_ap"]))

        unwrap_model(model).eval()
        val_loss_sum = torch.zeros((), device=device)
        val_count = torch.zeros((), device=device)
        val_score_parts: list[torch.Tensor] = []
        val_target_parts: list[torch.Tensor] = []
        val_image_score_parts: list[torch.Tensor] = []
        val_image_target_parts: list[torch.Tensor] = []

        with torch.no_grad():
            for batch in val_loader:
                x = batch["images"].to(device, non_blocking=True)
                targets = batch["targets"]
                heatmaps = _rasterize_box_targets(
                    targets,
                    grid=grid,
                    device=device,
                    positive_radius=positive_radius,
                )
                with autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    logits = model(x)
                    loss, _ = _box_heatmap_loss(
                        logits,
                        heatmaps,
                        loss_kind=loss_kind,
                        focal_alpha=focal_alpha,
                        focal_gamma=focal_gamma,
                        bce_weight=bce_weight,
                        dice_weight=dice_weight,
                    )
                val_loss_sum += loss.detach() * x.size(0)
                val_count += x.size(0)
                probs = torch.sigmoid(logits.detach()).float().cpu()
                heat_cpu = heatmaps.detach().float().cpu()
                val_score_parts.append(probs.reshape(-1))
                val_target_parts.append(heat_cpu.reshape(-1))
                val_image_score_parts.append(probs.max(dim=1).values)
                val_image_target_parts.append((heat_cpu.sum(dim=1) > 0).float())

        val_loss_sum = all_reduce_sum(val_loss_sum)
        val_count = all_reduce_sum(val_count)
        val_loss = (val_loss_sum / val_count.clamp(min=1)).item()
        metric_values = _metrics_from_parts(
            val_score_parts,
            val_target_parts,
            val_image_score_parts,
            val_image_target_parts,
            threshold=threshold,
        )
        val_primary = float(metric_values.get(primary_metric, metric_values["patch_ap"]))
        improved = val_primary > float(state["best_metric"]) + early_stop_min_delta

        last_train_metric = float(train_primary)
        last_val_loss = float(val_loss)
        last_metrics = dict(metric_values)

        if is_rank0():
            callbacks.on_epoch_end(
                cfg=cfg,
                state=state,
                metrics={
                    "probe/train_epoch_loss": float(loss_meter.avg),
                    f"probe/train_{primary_metric}": float(train_primary),
                    "probe/val_loss": float(val_loss),
                    "probe/val_patch_ap": float(metric_values["patch_ap"]),
                    "probe/val_patch_auroc": float(metric_values["patch_auroc"]),
                    "probe/val_heatmap_dice": float(metric_values["heatmap_dice"]),
                    "probe/val_image_auroc": float(metric_values["image_auroc"]),
                    "probe/lr": float(opt.param_groups[0]["lr"]),
                    "probe/epoch": float(epoch),
                },
            )

            payload = {
                "head": unwrap_model(model).head.state_dict(),
                "metric_name": primary_metric,
                "val_metric": val_primary,
                "epoch": epoch,
                **{f"val_{key}": value for key, value in metric_values.items()},
            }
            if improved:
                state["best_metric"] = float(val_primary)
                best_metrics = dict(metric_values)
                best_epoch = int(epoch)
                no_improve_epochs = 0
                if bool(getattr(cfg.train, "save_probe_checkpoints", True)):
                    ckpt_dir = str(getattr(cfg.train, "ckpt_dir", "checkpoints"))
                    os.makedirs(ckpt_dir, exist_ok=True)
                    torch.save(payload, os.path.join(ckpt_dir, "box_heatmap_probe_best.pt"))
            else:
                no_improve_epochs += 1

            if bool(getattr(cfg.train, "save_probe_checkpoints", True)):
                ckpt_dir = str(getattr(cfg.train, "ckpt_dir", "checkpoints"))
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save(payload, os.path.join(ckpt_dir, "box_heatmap_probe_last.pt"))

        should_stop = (
            early_stop_patience > 0
            and (epoch + 1) >= early_stop_min_epochs
            and no_improve_epochs >= early_stop_patience
        )
        stop_tensor = torch.tensor(
            1 if should_stop else 0,
            device=device,
            dtype=torch.long,
        )
        stop_tensor = all_reduce_sum(stop_tensor)
        if stop_tensor.item() > 0:
            if is_rank0():
                print(
                    "[box_heatmap_probe_eval] early stopping triggered at "
                    f"epoch={epoch} after {no_improve_epochs} non-improving epochs."
                )
            break

    if is_rank0():
        callbacks.on_run_end(cfg=cfg, state=state)
    if best_epoch < 0:
        best_epoch = int(state["epoch"])
        best_metrics = dict(last_metrics)

    best_primary = float(
        best_metrics.get(primary_metric, best_metrics.get("patch_ap", 0.0))
    )
    last_primary = float(
        last_metrics.get(primary_metric, last_metrics.get("patch_ap", 0.0))
    )
    return {
        "task_kind": "box_heatmap",
        "metric_name": primary_metric,
        "train_metric": float(last_train_metric),
        "val_metric": float(last_primary),
        "best_val_metric": float(best_primary),
        "train_acc": float(last_train_metric),
        "val_acc": float(last_primary),
        "best_val_acc": float(best_primary),
        "best_epoch": int(best_epoch),
        "last_epoch": int(state["epoch"]),
        "val_loss": float(last_val_loss),
        "val_patch_ap": float(last_metrics.get("patch_ap", 0.0)),
        "best_val_patch_ap": float(best_metrics.get("patch_ap", 0.0)),
        "val_patch_auroc": float(last_metrics.get("patch_auroc", 0.0)),
        "best_val_patch_auroc": float(best_metrics.get("patch_auroc", 0.0)),
        "val_heatmap_dice": float(last_metrics.get("heatmap_dice", 0.0)),
        "best_val_heatmap_dice": float(best_metrics.get("heatmap_dice", 0.0)),
        "val_image_auroc": float(last_metrics.get("image_auroc", 0.0)),
        "best_val_image_auroc": float(best_metrics.get("image_auroc", 0.0)),
        "num_images": float(last_metrics.get("num_images", 0.0)),
        "num_positive_patches": float(last_metrics.get("num_positive_patches", 0.0)),
    }
