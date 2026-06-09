from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader as _TDL, TensorDataset

from ijepa_lite.engine.eval_linear import _build_head, _build_scheduler
from ijepa_lite.utils.dist import all_reduce_sum, is_distributed, is_rank0, unwrap_model
from ijepa_lite.utils.meters import AverageMeter


@torch.no_grad()
def _extract_dense_features(
    encode_fn,
    loader,
    device: torch.device,
    amp: bool,
):
    all_feats, all_masks = [], []
    for batch in loader:
        x = batch["images"].to(device, non_blocking=True)
        masks = batch["masks"].to(device, non_blocking=True)
        with autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            feat = encode_fn(x)
        all_feats.append(feat.detach())
        all_masks.append(masks)
    return torch.cat(all_feats, 0), torch.cat(all_masks, 0)


def _confusion_matrix(
    logits: torch.Tensor,
    masks: torch.Tensor,
    num_classes: int,
    ignore_index: int,
) -> torch.Tensor:
    pred = logits.argmax(dim=1)
    valid = masks != ignore_index
    if not valid.any():
        return torch.zeros((num_classes, num_classes), device=logits.device, dtype=torch.long)
    target = masks[valid]
    pred = pred[valid]
    hist = torch.bincount(
        target * num_classes + pred,
        minlength=num_classes * num_classes,
    )
    return hist.view(num_classes, num_classes)


def _miou_and_pixel_acc(confusion: torch.Tensor) -> tuple[float, float]:
    confusion = confusion.float()
    tp = confusion.diag()
    denom = confusion.sum(dim=1) + confusion.sum(dim=0) - tp
    valid = denom > 0
    miou = (tp[valid] / denom[valid]).mean().item() if valid.any() else 0.0
    pixel_acc = (tp.sum() / confusion.sum().clamp(min=1.0)).item()
    return miou, pixel_acc


def _foreground_iou_and_dice(
    confusion: torch.Tensor,
    foreground_class: int = 1,
) -> tuple[float, float]:
    confusion = confusion.float()
    cls = int(foreground_class)
    if cls < 0 or cls >= confusion.shape[0]:
        return 0.0, 0.0
    tp = confusion[cls, cls]
    fp = confusion[:, cls].sum() - tp
    fn = confusion[cls, :].sum() - tp
    iou = (tp / (tp + fp + fn).clamp(min=1.0)).item()
    dice = ((2.0 * tp) / (2.0 * tp + fp + fn).clamp(min=1.0)).item()
    return iou, dice


def _class_weights(
    num_classes: int,
    foreground_class: int,
    foreground_weight: float,
    device: torch.device,
) -> torch.Tensor | None:
    if foreground_weight <= 1.0:
        return None
    if foreground_class < 0 or foreground_class >= num_classes:
        return None
    weights = torch.ones(num_classes, device=device)
    weights[int(foreground_class)] = float(foreground_weight)
    return weights


def _foreground_dice_loss(
    logits: torch.Tensor,
    masks: torch.Tensor,
    *,
    foreground_class: int,
    ignore_index: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    valid = masks != ignore_index
    if not valid.any():
        return logits.sum() * 0.0

    probs = torch.softmax(logits.float(), dim=1)[:, int(foreground_class)]
    target = (masks == int(foreground_class)).float()
    probs = probs[valid]
    target = target[valid]
    intersection = (probs * target).sum()
    denom = probs.sum() + target.sum()
    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice


def _segmentation_loss(
    logits: torch.Tensor,
    masks: torch.Tensor,
    *,
    num_classes: int,
    ignore_index: int,
    loss_kind: str,
    foreground_class: int,
    foreground_weight: float,
    ce_weight: float,
    dice_weight: float,
) -> torch.Tensor:
    loss_kind = str(loss_kind).lower()
    supported = {"ce", "weighted_ce", "ce_dice", "weighted_ce_dice", "dice"}
    if loss_kind not in supported:
        raise ValueError(
            f"Unsupported segmentation loss={loss_kind}. "
            f"Expected one of {sorted(supported)}."
        )

    weights = None
    if loss_kind.startswith("weighted") or foreground_weight > 1.0:
        weights = _class_weights(
            num_classes=num_classes,
            foreground_class=foreground_class,
            foreground_weight=foreground_weight,
            device=logits.device,
        )

    if loss_kind == "dice":
        effective_dice_weight = dice_weight if dice_weight > 0.0 else 1.0
        return effective_dice_weight * _foreground_dice_loss(
            logits,
            masks,
            foreground_class=foreground_class,
            ignore_index=ignore_index,
        )

    ce = F.cross_entropy(
        logits,
        masks,
        ignore_index=ignore_index,
        weight=weights,
    )
    if "dice" not in loss_kind and dice_weight <= 0.0:
        return ce_weight * ce

    effective_dice_weight = dice_weight if dice_weight > 0.0 else 1.0
    dice = _foreground_dice_loss(
        logits,
        masks,
        foreground_class=foreground_class,
        ignore_index=ignore_index,
    )
    return ce_weight * ce + effective_dice_weight * dice


class SegmentationProbeModel(nn.Module):
    def __init__(self, encoder: nn.Module, head: nn.Module, pool: str = "mean") -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.pool = str(pool)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        if self.pool == "mean":
            return self.encoder(x)
        if self.pool == "last4_mean":
            if not hasattr(self.encoder, "forward_last_n"):
                raise ValueError(
                    "pool=last4_mean requires an encoder with forward_last_n support."
                )
            layer_tokens = self.encoder.forward_last_n(x, last_n=4)
            return torch.cat(layer_tokens, dim=-1)
        raise ValueError(f"Unsupported segmentation pool={self.pool}")

    def forward(self, x: torch.Tensor | None, mask_hw: tuple[int, int], feat: torch.Tensor | None = None):
        if feat is None:
            feat = self._features(x)

        bsz, n_patches, dim = feat.shape
        grid = int(math.isqrt(n_patches))
        if grid * grid != n_patches:
            raise ValueError(
                f"Segmentation probe expects a square patch grid, got n_patches={n_patches}."
            )

        logits = self.head(feat.reshape(bsz * n_patches, dim))
        logits = logits.view(bsz, grid, grid, -1).permute(0, 3, 1, 2).contiguous()
        if logits.shape[-2:] != mask_hw:
            logits = F.interpolate(
                logits,
                size=mask_hw,
                mode="bilinear",
                align_corners=False,
            )
        return logits


def segmentation_probe_eval(
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
    ignore_index = int(getattr(cfg.task, "ignore_index", 255))
    primary_metric = str(getattr(cfg.task, "metric", "miou")).lower()
    foreground_class = int(getattr(cfg.task, "foreground_class", 1))
    loss_kind = str(getattr(cfg.task, "loss", "ce")).lower()
    foreground_weight = float(getattr(cfg.task, "foreground_weight", 1.0))
    ce_weight = float(getattr(cfg.task, "ce_weight", 1.0))
    dice_weight = float(getattr(cfg.task, "dice_weight", 0.0))
    pool = str(getattr(cfg.task, "pool", "mean"))
    head_in_dim = int(cfg.model.embed_dim) * (4 if pool == "last4_mean" else 1)

    head = _build_head(
        head_in_dim,
        int(num_classes),
        getattr(cfg.task, "head", None),
    ).to(device)
    model = SegmentationProbeModel(
        encoder=encoder,
        head=head,
        pool=pool,
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

    opt = torch.optim.SGD(
        unwrap_model(model).head.parameters(),
        lr=float(cfg.train.lr),
        momentum=0.9,
        weight_decay=float(cfg.train.weight_decay),
    )
    sched = _build_scheduler(opt, getattr(cfg.task, "sched", None), epochs)
    scaler = GradScaler("cuda", enabled=amp)

    def encode_fn(x):
        t = unwrap_model(model)._features(x)
        return F.layer_norm(t, (t.shape[-1],))

    feats_tr, masks_tr = _extract_dense_features(encode_fn, train_loader, device, amp)
    feats_val, masks_val = _extract_dense_features(encode_fn, val_loader, device, amp)

    bsz = train_loader.batch_size
    train_cache = _TDL(TensorDataset(feats_tr, masks_tr), batch_size=bsz, shuffle=True, drop_last=True)
    val_cache = _TDL(TensorDataset(feats_val, masks_val), batch_size=bsz, shuffle=False)

    state = {"epoch": 0, "global_step": 0, "best_miou": 0.0}
    log_every = int(getattr(cfg.train, "log_every", 50))
    early_stop_patience = int(getattr(cfg.train, "early_stop_patience", 0))
    early_stop_min_epochs = int(getattr(cfg.train, "early_stop_min_epochs", 0))
    early_stop_min_delta = float(getattr(cfg.train, "early_stop_min_delta", 0.0))
    no_improve_epochs = 0
    best_epoch = -1
    best_val_miou = 0.0
    best_val_pixel_acc = 0.0
    best_val_fg_iou = 0.0
    best_val_fg_dice = 0.0
    last_train_miou = 0.0
    last_train_pixel_acc = 0.0
    last_train_fg_iou = 0.0
    last_train_fg_dice = 0.0
    last_val_miou = 0.0
    last_val_pixel_acc = 0.0
    last_val_fg_iou = 0.0
    last_val_fg_dice = 0.0
    last_val_loss = 0.0

    callbacks.on_run_start(cfg=cfg, state=state, model=unwrap_model(model))

    for epoch in range(epochs):
        state["epoch"] = epoch
        callbacks.on_epoch_start(cfg=cfg, state=state)

        unwrap_model(model).head.train()

        loss_meter = AverageMeter()
        train_conf = torch.zeros((num_classes, num_classes), device=device, dtype=torch.long)

        for feat, masks in train_cache:
            opt.zero_grad(set_to_none=True)
            with autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logits = model(None, mask_hw=tuple(masks.shape[-2:]), feat=feat)
                loss = _segmentation_loss(
                    logits,
                    masks,
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    loss_kind=loss_kind,
                    foreground_class=foreground_class,
                    foreground_weight=foreground_weight,
                    ce_weight=ce_weight,
                    dice_weight=dice_weight,
                )

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            loss_meter.update(float(loss.item()), n=feat.size(0))
            state["global_step"] += 1

            with torch.no_grad():
                train_conf += _confusion_matrix(
                    logits, masks, num_classes=num_classes, ignore_index=ignore_index
                )

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

        train_conf = all_reduce_sum(train_conf)
        train_miou, train_pixel_acc = _miou_and_pixel_acc(train_conf)
        train_fg_iou, train_fg_dice = _foreground_iou_and_dice(
            train_conf,
            foreground_class=foreground_class,
        )
        train_metric_values = {
            "miou": train_miou,
            "pixel_acc": train_pixel_acc,
            "fg_iou": train_fg_iou,
            "fg_dice": train_fg_dice,
        }
        train_primary = float(train_metric_values.get(primary_metric, train_miou))

        unwrap_model(model).eval()
        val_loss_sum = torch.zeros((), device=device)
        val_items = torch.zeros((), device=device)
        val_conf = torch.zeros((num_classes, num_classes), device=device, dtype=torch.long)

        with torch.no_grad():
            for feat, masks in val_cache:
                with autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    logits = model(None, mask_hw=tuple(masks.shape[-2:]), feat=feat)
                    loss = _segmentation_loss(
                        logits,
                        masks,
                        num_classes=num_classes,
                        ignore_index=ignore_index,
                        loss_kind=loss_kind,
                        foreground_class=foreground_class,
                        foreground_weight=foreground_weight,
                        ce_weight=ce_weight,
                        dice_weight=dice_weight,
                    )

                valid = (masks != ignore_index).sum()
                val_loss_sum += loss.detach() * valid
                val_items += valid
                val_conf += _confusion_matrix(
                    logits, masks, num_classes=num_classes, ignore_index=ignore_index
                )

        val_loss_sum = all_reduce_sum(val_loss_sum)
        val_items = all_reduce_sum(val_items)
        val_conf = all_reduce_sum(val_conf)

        val_loss = (val_loss_sum / val_items.clamp(min=1)).item()
        val_miou, val_pixel_acc = _miou_and_pixel_acc(val_conf)
        val_fg_iou, val_fg_dice = _foreground_iou_and_dice(
            val_conf,
            foreground_class=foreground_class,
        )
        metric_values = {
            "miou": val_miou,
            "pixel_acc": val_pixel_acc,
            "fg_iou": val_fg_iou,
            "fg_dice": val_fg_dice,
        }
        val_primary = float(metric_values.get(primary_metric, val_miou))
        last_train_miou = float(train_miou)
        last_train_pixel_acc = float(train_pixel_acc)
        last_train_fg_iou = float(train_fg_iou)
        last_train_fg_dice = float(train_fg_dice)
        last_val_miou = float(val_miou)
        last_val_pixel_acc = float(val_pixel_acc)
        last_val_fg_iou = float(val_fg_iou)
        last_val_fg_dice = float(val_fg_dice)
        last_val_loss = float(val_loss)
        improved = val_primary > float(state["best_miou"]) + early_stop_min_delta

        if is_rank0():
            callbacks.on_epoch_end(
                cfg=cfg,
                state=state,
                metrics={
                    "probe/train_epoch_loss": float(loss_meter.avg),
                    "probe/train_miou": float(train_miou),
                    "probe/train_pixel_acc": float(train_pixel_acc),
                    "probe/train_fg_iou": float(train_fg_iou),
                    "probe/train_fg_dice": float(train_fg_dice),
                    "probe/val_loss": float(val_loss),
                    "probe/val_miou": float(val_miou),
                    "probe/val_pixel_acc": float(val_pixel_acc),
                    "probe/val_fg_iou": float(val_fg_iou),
                    "probe/val_fg_dice": float(val_fg_dice),
                    "probe/lr": float(opt.param_groups[0]["lr"]),
                    "probe/epoch": float(epoch),
                },
            )

            payload = {
                "head": unwrap_model(model).head.state_dict(),
                "val_miou": val_miou,
                "val_fg_iou": val_fg_iou,
                "val_fg_dice": val_fg_dice,
                "metric_name": primary_metric,
                "train_metric": train_primary,
                "val_metric": val_primary,
                "epoch": epoch,
            }

            if improved:
                state["best_miou"] = float(val_primary)
                best_val_miou = float(val_miou)
                best_val_pixel_acc = float(val_pixel_acc)
                best_val_fg_iou = float(val_fg_iou)
                best_val_fg_dice = float(val_fg_dice)
                best_epoch = int(epoch)
                no_improve_epochs = 0
                if bool(getattr(cfg.train, "save_probe_checkpoints", True)):
                    ckpt_dir = str(getattr(cfg.train, "ckpt_dir", "checkpoints"))
                    os.makedirs(ckpt_dir, exist_ok=True)
                    torch.save(payload, os.path.join(ckpt_dir, "segmentation_probe_best.pt"))
            else:
                no_improve_epochs += 1

            if bool(getattr(cfg.train, "save_probe_checkpoints", True)):
                ckpt_dir = str(getattr(cfg.train, "ckpt_dir", "checkpoints"))
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save(payload, os.path.join(ckpt_dir, "segmentation_probe_last.pt"))

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
                    "[segmentation_probe_eval] early stopping triggered at "
                    f"epoch={epoch} after {no_improve_epochs} non-improving epochs."
                )
            break

    if is_rank0():
        callbacks.on_run_end(cfg=cfg, state=state)
    if best_epoch < 0:
        best_epoch = int(state["epoch"])
        best_val_miou = float(last_val_miou)
        best_val_pixel_acc = float(last_val_pixel_acc)
        best_val_fg_iou = float(last_val_fg_iou)
        best_val_fg_dice = float(last_val_fg_dice)
    primary_last = {
        "miou": last_val_miou,
        "pixel_acc": last_val_pixel_acc,
        "fg_iou": last_val_fg_iou,
        "fg_dice": last_val_fg_dice,
    }.get(primary_metric, last_val_miou)
    primary_train_last = {
        "miou": last_train_miou,
        "pixel_acc": last_train_pixel_acc,
        "fg_iou": last_train_fg_iou,
        "fg_dice": last_train_fg_dice,
    }.get(primary_metric, last_train_miou)
    primary_best = {
        "miou": best_val_miou,
        "pixel_acc": best_val_pixel_acc,
        "fg_iou": best_val_fg_iou,
        "fg_dice": best_val_fg_dice,
    }.get(primary_metric, best_val_miou)
    return {
        "task_kind": "segmentation",
        "metric_name": primary_metric,
        "train_metric": float(primary_train_last),
        "val_metric": float(primary_last),
        "best_val_metric": float(primary_best),
        "train_acc": float(last_train_pixel_acc),
        "val_acc": float(last_val_pixel_acc),
        "best_val_acc": float(best_val_pixel_acc),
        "best_epoch": int(best_epoch),
        "last_epoch": int(state["epoch"]),
        "val_loss": float(last_val_loss),
        "train_miou": float(last_train_miou),
        "val_miou": float(last_val_miou),
        "best_val_miou": float(best_val_miou),
        "train_fg_iou": float(last_train_fg_iou),
        "val_fg_iou": float(last_val_fg_iou),
        "best_val_fg_iou": float(best_val_fg_iou),
        "train_fg_dice": float(last_train_fg_dice),
        "val_fg_dice": float(last_val_fg_dice),
        "best_val_fg_dice": float(best_val_fg_dice),
    }
