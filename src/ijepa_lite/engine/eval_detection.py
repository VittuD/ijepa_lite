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


class DetectionProbeModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        head: nn.Module,
        pool: str = "mean",
        max_box_size: float = 1.0,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.pool = str(pool)
        self.max_box_size = float(max_box_size)

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
                raise ValueError(f"Unsupported detection pool={self.pool}")
        return F.layer_norm(feat, (feat.shape[-1],))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self._features(x)
        bsz, n_patches, dim = feat.shape
        grid = int(math.isqrt(n_patches))
        if grid * grid != n_patches:
            raise ValueError(
                f"Detection probe expects a square patch grid, got n_patches={n_patches}."
            )

        pred = self.head(feat.reshape(bsz * n_patches, dim)).view(bsz, n_patches, 5)
        obj_logits = pred[..., 0]
        raw_boxes = pred[..., 1:5]

        ys, xs = torch.meshgrid(
            torch.arange(grid, device=feat.device, dtype=feat.dtype),
            torch.arange(grid, device=feat.device, dtype=feat.dtype),
            indexing="ij",
        )
        cell_xy = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)
        offset = raw_boxes[..., 0:2].sigmoid()
        center = (cell_xy.unsqueeze(0) + offset) / float(grid)
        wh = raw_boxes[..., 2:4].sigmoid() * self.max_box_size
        boxes = torch.cat([center - 0.5 * wh, center + 0.5 * wh], dim=-1)
        boxes = boxes.clamp(0.0, 1.0)
        return {"obj_logits": obj_logits, "boxes": boxes}


def _box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) *
             (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0))
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) *
             (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0))
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-12)


def _nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        idx = order[0]
        keep.append(idx)
        if order.numel() == 1:
            break
        ious = _box_iou(boxes[idx].unsqueeze(0), boxes[order[1:]]).squeeze(0)
        order = order[1:][ious <= float(iou_threshold)]
    return torch.stack(keep) if keep else torch.empty((0,), dtype=torch.long, device=boxes.device)


def _build_targets(
    targets: list[dict[str, Any]],
    *,
    grid: int,
    device: torch.device,
    positive_radius: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz = len(targets)
    n_patches = grid * grid
    obj_targets = torch.zeros((bsz, n_patches), device=device)
    box_targets = torch.zeros((bsz, n_patches, 4), device=device)
    pos_mask = torch.zeros((bsz, n_patches), device=device, dtype=torch.bool)
    assigned_area = torch.full((bsz, n_patches), -1.0, device=device)

    radius = max(0, int(positive_radius))
    for bidx, target in enumerate(targets):
        boxes = target["boxes"].to(device=device, dtype=torch.float32)
        boxes = boxes.clamp(0.0, 1.0)
        if boxes.numel() == 0:
            continue
        centers = 0.5 * (boxes[:, :2] + boxes[:, 2:])
        areas = ((boxes[:, 2] - boxes[:, 0]).clamp(min=0) *
                 (boxes[:, 3] - boxes[:, 1]).clamp(min=0))
        for box, center, area in zip(boxes, centers, areas):
            cx = int(torch.clamp(center[0] * grid, 0, grid - 1).item())
            cy = int(torch.clamp(center[1] * grid, 0, grid - 1).item())
            for yy in range(max(0, cy - radius), min(grid, cy + radius + 1)):
                for xx in range(max(0, cx - radius), min(grid, cx + radius + 1)):
                    patch_idx = yy * grid + xx
                    if float(area.item()) < float(assigned_area[bidx, patch_idx].item()):
                        continue
                    assigned_area[bidx, patch_idx] = area
                    obj_targets[bidx, patch_idx] = 1.0
                    box_targets[bidx, patch_idx] = box
                    pos_mask[bidx, patch_idx] = True
    return obj_targets, box_targets, pos_mask


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
    modulating = (1.0 - p_t).pow(float(gamma))
    alpha_t = float(alpha) * targets + (1.0 - float(alpha)) * (1.0 - targets)
    return (alpha_t * modulating * bce).mean()


def _detection_loss(
    outputs: dict[str, torch.Tensor],
    targets: list[dict[str, Any]],
    *,
    grid: int,
    positive_radius: int,
    objectness_loss: str,
    focal_alpha: float,
    focal_gamma: float,
    obj_weight: float,
    box_weight: float,
    objectness_recall_threshold: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    obj_targets, box_targets, pos_mask = _build_targets(
        targets,
        grid=grid,
        device=outputs["obj_logits"].device,
        positive_radius=positive_radius,
    )

    if objectness_loss == "focal":
        obj_loss = _sigmoid_focal_loss(
            outputs["obj_logits"],
            obj_targets,
            alpha=focal_alpha,
            gamma=focal_gamma,
        )
    elif objectness_loss in {"bce", "weighted_bce"}:
        pos_weight = None
        if objectness_loss == "weighted_bce":
            n_pos = obj_targets.sum()
            n_neg = obj_targets.numel() - n_pos
            pos_weight = (n_neg / n_pos.clamp(min=1.0)).clamp(min=1.0)
        obj_loss = F.binary_cross_entropy_with_logits(
            outputs["obj_logits"],
            obj_targets,
            pos_weight=pos_weight,
        )
    else:
        raise ValueError(
            f"Unknown detection objectness_loss='{objectness_loss}'. "
            "Supported: focal|bce|weighted_bce."
        )

    if pos_mask.any():
        box_loss = F.smooth_l1_loss(outputs["boxes"][pos_mask], box_targets[pos_mask])
    else:
        box_loss = outputs["boxes"].sum() * 0.0

    loss = float(obj_weight) * obj_loss + float(box_weight) * box_loss
    return loss, {
        "obj_loss": obj_loss.detach(),
        "box_loss": box_loss.detach(),
        "num_pos": pos_mask.sum().detach(),
        "pos_recall": (
            (
                (outputs["obj_logits"].sigmoid() > float(objectness_recall_threshold))
                & pos_mask
            ).sum().float()
            / pos_mask.sum().clamp(min=1).float()
        ).detach(),
    }


def _records_from_batch(
    outputs: dict[str, torch.Tensor],
    targets: list[dict[str, Any]],
    *,
    score_threshold: float,
    nms_iou_threshold: float,
    max_detections: int,
) -> list[dict[str, Any]]:
    scores_all = outputs["obj_logits"].sigmoid().detach().cpu()
    boxes_all = outputs["boxes"].detach().cpu()
    records = []
    for idx, target in enumerate(targets):
        scores = scores_all[idx]
        boxes = boxes_all[idx]
        keep = scores >= float(score_threshold)
        scores = scores[keep]
        boxes = boxes[keep]
        if scores.numel() > int(max_detections):
            topk = scores.topk(int(max_detections)).indices
            scores = scores[topk]
            boxes = boxes[topk]
        if scores.numel() > 0:
            keep_idx = _nms(boxes, scores, iou_threshold=float(nms_iou_threshold))
            scores = scores[keep_idx]
            boxes = boxes[keep_idx]
        records.append(
            {
                "image_id": str(target["image_id"]),
                "scores": scores.float(),
                "boxes": boxes.float(),
                "gt_boxes": target["boxes"].detach().cpu().float(),
            }
        )
    return records


def _gather_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not is_distributed():
        return records
    gathered: list[list[dict[str, Any]] | None] = [None for _ in range(get_world_size())]
    dist.all_gather_object(gathered, records)
    out: list[dict[str, Any]] = []
    for part in gathered:
        if part:
            out.extend(part)
    by_id = {str(record["image_id"]): record for record in out}
    return list(by_id.values())


def _binary_auc(scores: list[float], targets: list[int]) -> float:
    if not scores:
        return 0.0
    score_t = torch.as_tensor(scores, dtype=torch.float64)
    target_t = torch.as_tensor(targets, dtype=torch.bool)
    n_pos = int(target_t.sum().item())
    n_total = int(target_t.numel())
    n_neg = n_total - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0
    order = torch.argsort(score_t)
    sorted_scores = score_t[order]
    ranks = torch.empty(n_total, dtype=torch.float64)
    start = 0
    while start < n_total:
        end = start + 1
        while end < n_total and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    pos_rank_sum = ranks[target_t].sum().item()
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _detection_metrics(
    records: list[dict[str, Any]],
    *,
    iou_threshold: float,
    froc_fp_per_image: list[float],
) -> dict[str, float]:
    records = _gather_records(records)
    num_images = max(1, len(records))
    gt_by_image = {record["image_id"]: record["gt_boxes"] for record in records}
    matched = {
        image_id: torch.zeros((gt.shape[0],), dtype=torch.bool)
        for image_id, gt in gt_by_image.items()
    }
    total_gt = int(sum(gt.shape[0] for gt in gt_by_image.values()))

    preds = []
    image_scores = []
    image_targets = []
    for record in records:
        image_id = record["image_id"]
        scores = record["scores"]
        boxes = record["boxes"]
        for score, box in zip(scores, boxes):
            preds.append((float(score.item()), image_id, box))
        image_scores.append(float(scores.max().item()) if scores.numel() else 0.0)
        image_targets.append(1 if record["gt_boxes"].numel() > 0 else 0)

    preds.sort(key=lambda item: item[0], reverse=True)
    tp, fp = [], []
    for _, image_id, box in preds:
        gt = gt_by_image[image_id]
        if gt.numel() == 0:
            tp.append(0.0)
            fp.append(1.0)
            continue
        ious = _box_iou(box.unsqueeze(0), gt).squeeze(0)
        best_iou, best_idx = ious.max(dim=0)
        if float(best_iou.item()) >= float(iou_threshold) and not matched[image_id][best_idx]:
            matched[image_id][best_idx] = True
            tp.append(1.0)
            fp.append(0.0)
        else:
            tp.append(0.0)
            fp.append(1.0)

    if total_gt <= 0 or not preds:
        ap50 = 0.0
        froc_mean = 0.0
        froc_values = {float(thr): 0.0 for thr in froc_fp_per_image}
    else:
        tp_t = torch.as_tensor(tp, dtype=torch.float64).cumsum(0)
        fp_t = torch.as_tensor(fp, dtype=torch.float64).cumsum(0)
        recall = tp_t / float(total_gt)
        precision = tp_t / (tp_t + fp_t).clamp(min=1e-12)
        mrec = torch.cat([torch.tensor([0.0]), recall, torch.tensor([1.0])])
        mpre = torch.cat([torch.tensor([0.0]), precision, torch.tensor([0.0])])
        for idx in range(mpre.numel() - 2, -1, -1):
            mpre[idx] = torch.maximum(mpre[idx], mpre[idx + 1])
        changing = torch.where(mrec[1:] != mrec[:-1])[0]
        ap50 = float(((mrec[changing + 1] - mrec[changing]) * mpre[changing + 1]).sum().item())

        froc_values = {}
        fp_per_image = fp_t / float(num_images)
        for threshold in froc_fp_per_image:
            valid = fp_per_image <= float(threshold)
            froc_values[float(threshold)] = (
                float(recall[valid].max().item()) if valid.any() else 0.0
            )
        froc_mean = float(sum(froc_values.values()) / max(1, len(froc_values)))

    out = {
        "ap50": ap50,
        "froc_mean": froc_mean,
        "image_auroc": _binary_auc(image_scores, image_targets),
        "num_images": float(num_images),
        "num_gt": float(total_gt),
    }
    for threshold, value in froc_values.items():
        key = str(threshold).replace(".", "p")
        out[f"sens_at_{key}_fp"] = float(value)
    return out


def detection_probe_eval(
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
    primary_metric = str(getattr(cfg.task, "metric", "ap50")).lower()
    positive_radius = int(getattr(cfg.task, "positive_radius", 0))
    objectness_loss = str(getattr(cfg.task, "objectness_loss", "focal")).lower()
    focal_alpha = float(getattr(cfg.task, "focal_alpha", 0.25))
    focal_gamma = float(getattr(cfg.task, "focal_gamma", 2.0))
    obj_weight = float(getattr(cfg.task, "obj_weight", 1.0))
    box_weight = float(getattr(cfg.task, "box_weight", 5.0))
    max_box_size = float(getattr(cfg.task, "max_box_size", 1.0))
    score_threshold = float(getattr(cfg.task, "score_threshold", 0.05))
    nms_iou_threshold = float(getattr(cfg.task, "nms_iou_threshold", 0.5))
    max_detections = int(getattr(cfg.task, "max_detections", 100))
    ap_iou_threshold = float(getattr(cfg.task, "ap_iou_threshold", 0.5))
    objectness_recall_threshold = float(
        getattr(cfg.task, "objectness_recall_threshold", score_threshold)
    )
    froc_fp_per_image = [
        float(x) for x in getattr(cfg.task, "froc_fp_per_image", [0.5, 1.0, 2.0, 4.0])
    ]

    head = _build_head(
        head_in_dim,
        5,
        getattr(cfg.task, "head", None),
    ).to(device)
    model = DetectionProbeModel(
        encoder=encoder,
        head=head,
        pool=pool,
        max_box_size=max_box_size,
    ).to(device)

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
    last_train_pos_recall = 0.0
    last_val_loss = 0.0

    callbacks.on_run_start(cfg=cfg, state=state, model=unwrap_model(model))

    for epoch in range(epochs):
        state["epoch"] = epoch
        callbacks.on_epoch_start(cfg=cfg, state=state)
        unwrap_model(model).head.train()

        loss_meter = AverageMeter()
        obj_loss_meter = AverageMeter()
        box_loss_meter = AverageMeter()
        pos_recall_sum = torch.zeros((), device=device)
        pos_recall_count = torch.zeros((), device=device)

        for batch in train_loader:
            x = batch["images"].to(device, non_blocking=True)
            targets = batch["targets"]
            opt.zero_grad(set_to_none=True)
            with autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                outputs = model(x)
                loss, loss_parts = _detection_loss(
                    outputs,
                    targets,
                    grid=grid,
                    positive_radius=positive_radius,
                    objectness_loss=objectness_loss,
                    focal_alpha=focal_alpha,
                    focal_gamma=focal_gamma,
                    obj_weight=obj_weight,
                    box_weight=box_weight,
                    objectness_recall_threshold=objectness_recall_threshold,
                )

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            loss_meter.update(float(loss.item()), n=x.size(0))
            obj_loss_meter.update(float(loss_parts["obj_loss"].item()), n=x.size(0))
            box_loss_meter.update(float(loss_parts["box_loss"].item()), n=x.size(0))
            pos_recall_sum += loss_parts["pos_recall"] * loss_parts["num_pos"].clamp(min=1)
            pos_recall_count += loss_parts["num_pos"]
            state["global_step"] += 1

            if state["global_step"] % log_every == 0 and is_rank0():
                callbacks.on_step_end(
                    cfg=cfg,
                    state=state,
                    metrics={
                        "probe/train_loss": float(loss_meter.avg),
                        "probe/train_obj_loss": float(obj_loss_meter.avg),
                        "probe/train_box_loss": float(box_loss_meter.avg),
                        "probe/lr": float(opt.param_groups[0]["lr"]),
                        "probe/epoch": float(epoch),
                    },
                )

        if sched is not None:
            sched.step()

        pos_recall_sum = all_reduce_sum(pos_recall_sum)
        pos_recall_count = all_reduce_sum(pos_recall_count)
        train_pos_recall = (pos_recall_sum / pos_recall_count.clamp(min=1)).item()

        unwrap_model(model).eval()
        val_loss_sum = torch.zeros((), device=device)
        val_count = torch.zeros((), device=device)
        records: list[dict[str, Any]] = []

        with torch.no_grad():
            for batch in val_loader:
                x = batch["images"].to(device, non_blocking=True)
                targets = batch["targets"]
                with autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    outputs = model(x)
                    loss, _ = _detection_loss(
                        outputs,
                        targets,
                        grid=grid,
                        positive_radius=positive_radius,
                        objectness_loss=objectness_loss,
                        focal_alpha=focal_alpha,
                        focal_gamma=focal_gamma,
                        obj_weight=obj_weight,
                        box_weight=box_weight,
                        objectness_recall_threshold=objectness_recall_threshold,
                    )
                val_loss_sum += loss.detach() * x.size(0)
                val_count += x.size(0)
                records.extend(
                    _records_from_batch(
                        outputs,
                        targets,
                        score_threshold=score_threshold,
                        nms_iou_threshold=nms_iou_threshold,
                        max_detections=max_detections,
                    )
                )

        val_loss_sum = all_reduce_sum(val_loss_sum)
        val_count = all_reduce_sum(val_count)
        val_loss = (val_loss_sum / val_count.clamp(min=1)).item()
        metric_values = _detection_metrics(
            records,
            iou_threshold=ap_iou_threshold,
            froc_fp_per_image=froc_fp_per_image,
        )
        val_primary = float(metric_values.get(primary_metric, metric_values["ap50"]))
        improved = val_primary > float(state["best_metric"]) + early_stop_min_delta

        last_train_pos_recall = float(train_pos_recall)
        last_val_loss = float(val_loss)
        last_metrics = dict(metric_values)

        if is_rank0():
            log_metrics = {
                "probe/train_epoch_loss": float(loss_meter.avg),
                "probe/train_pos_recall": float(train_pos_recall),
                "probe/val_loss": float(val_loss),
                "probe/val_ap50": float(metric_values["ap50"]),
                "probe/val_froc_mean": float(metric_values["froc_mean"]),
                "probe/val_image_auroc": float(metric_values["image_auroc"]),
                "probe/lr": float(opt.param_groups[0]["lr"]),
                "probe/epoch": float(epoch),
            }
            callbacks.on_epoch_end(cfg=cfg, state=state, metrics=log_metrics)

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
                    torch.save(payload, os.path.join(ckpt_dir, "detection_probe_best.pt"))
            else:
                no_improve_epochs += 1

            if bool(getattr(cfg.train, "save_probe_checkpoints", True)):
                ckpt_dir = str(getattr(cfg.train, "ckpt_dir", "checkpoints"))
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save(payload, os.path.join(ckpt_dir, "detection_probe_last.pt"))

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
                    "[detection_probe_eval] early stopping triggered at "
                    f"epoch={epoch} after {no_improve_epochs} non-improving epochs."
                )
            break

    if is_rank0():
        callbacks.on_run_end(cfg=cfg, state=state)
    if best_epoch < 0:
        best_epoch = int(state["epoch"])
        best_metrics = dict(last_metrics)

    best_primary = float(best_metrics.get(primary_metric, best_metrics.get("ap50", 0.0)))
    last_primary = float(last_metrics.get(primary_metric, last_metrics.get("ap50", 0.0)))
    return {
        "task_kind": "detection",
        "metric_name": primary_metric,
        "train_metric": float(last_train_pos_recall),
        "val_metric": float(last_primary),
        "best_val_metric": float(best_primary),
        "train_acc": float(last_train_pos_recall),
        "val_acc": float(last_primary),
        "best_val_acc": float(best_primary),
        "best_epoch": int(best_epoch),
        "last_epoch": int(state["epoch"]),
        "val_loss": float(last_val_loss),
        "val_ap50": float(last_metrics.get("ap50", 0.0)),
        "best_val_ap50": float(best_metrics.get("ap50", 0.0)),
        "val_froc_mean": float(last_metrics.get("froc_mean", 0.0)),
        "best_val_froc_mean": float(best_metrics.get("froc_mean", 0.0)),
        "val_image_auroc": float(last_metrics.get("image_auroc", 0.0)),
        "best_val_image_auroc": float(best_metrics.get("image_auroc", 0.0)),
        "num_images": float(last_metrics.get("num_images", 0.0)),
        "num_gt": float(last_metrics.get("num_gt", 0.0)),
    }
