from __future__ import annotations

import os
from typing import Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader as _TDL, TensorDataset

from ijepa_lite.utils.dist import (
    all_reduce_sum,
    is_distributed,
    is_rank0,
    unwrap_model,
)
from ijepa_lite.utils.meters import AverageMeter


class LARS(torch.optim.Optimizer):
    """Minimal LARS optimizer for linear probing.

    Credit:
        Update semantics aligned to `kakaobrain/torchlars`
        https://github.com/kakaobrain/torchlars

    Matches the core SGD/LARS step used there:
    - adaptive LR uses `||w|| / (||g|| + wd * ||w|| + eps)`
    - weight decay is applied before momentum/update
    - falls back to adaptive_lr=1 when param or grad norm is zero
    """

    def __init__(
        self,
        params,
        lr: float,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        eta: float = 1e-3,
        eps: float = 1e-8,
    ) -> None:
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            eta=eta,
            eps=eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            momentum = float(group["momentum"])
            weight_decay = float(group["weight_decay"])
            eta = float(group["eta"])
            eps = float(group["eps"])

            for p in group["params"]:
                if p.grad is None:
                    continue
                param_norm = torch.norm(p)
                grad = p.grad
                grad_norm = torch.norm(grad)
                adaptive_lr = 1.0
                if param_norm > 0 and grad_norm > 0:
                    divisor = grad_norm + weight_decay * param_norm + eps
                    adaptive_lr = float(eta * param_norm / divisor)
                if weight_decay != 0.0:
                    grad = grad.add(p, alpha=weight_decay)

                state = self.state[p]
                if "mu" not in state:
                    state["mu"] = torch.zeros_like(p)
                mu = state["mu"]
                mu.mul_(momentum).add_(grad, alpha=adaptive_lr)
                p.add_(mu, alpha=-lr)

        return loss


@torch.no_grad()
def _extract_features(
    encode_fn,
    loader,
    device: torch.device,
    amp: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-pass feature extraction from a frozen encoder.

    Returns (feats [N, D], labels [N]) on ``device``.
    encode_fn: callable (B, C, H, W) → (B, D) — must not require grad.
    """
    all_feats, all_labels = [], []
    for batch in loader:
        x = batch["images"].to(device, non_blocking=True)
        y = batch["labels"].to(device, non_blocking=True)
        with autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            feat = encode_fn(x)
        all_feats.append(feat.detach())
        all_labels.append(y)
    return torch.cat(all_feats, 0), torch.cat(all_labels, 0)


def _build_head(embed_dim: int, num_classes: int, cfg_head) -> nn.Module:
    """Build a linear or MLP probe head from config.

    cfg_head fields (all optional):
        type:       "linear" | "mlp"   (default: "linear")
        hidden_dim: int                (default: embed_dim, only for mlp)
        num_layers: int                (default: 1, only for mlp — number of hidden layers)
        dropout:    float              (default: 0.0, only for mlp)
    """
    head_type = str(getattr(cfg_head, "type", "linear")).lower() if cfg_head else "linear"

    if head_type == "linear":
        return nn.Linear(embed_dim, num_classes)

    if head_type == "mlp":
        hidden_dim = int(getattr(cfg_head, "hidden_dim", embed_dim))
        num_layers = int(getattr(cfg_head, "num_layers", 1))
        dropout = float(getattr(cfg_head, "dropout", 0.0))

        layers: list[nn.Module] = []
        in_dim = embed_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
            ])
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_classes))
        return nn.Sequential(*layers)

    raise ValueError(f"Unknown head type='{head_type}'. Supported: 'linear', 'mlp'.")


class LinearProbeModel(nn.Module):
    """Frozen encoder + trainable linear/MLP head."""

    def __init__(self, encoder: nn.Module, head: nn.Module, pool: str = "mean"):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.pool = str(pool)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        if self.pool == "mean":
            tokens = self.encoder(x)  # (B, N, D) patch tokens
            return tokens.mean(dim=1)  # (B, D)
        if self.pool == "last4_mean":
            if not hasattr(self.encoder, "forward_last_n"):
                raise ValueError("pool=last4_mean requires an encoder with forward_last_n support.")
            layer_tokens = self.encoder.forward_last_n(x, last_n=4)
            pooled = [tokens.mean(dim=1) for tokens in layer_tokens]
            return torch.stack(pooled, dim=0).mean(dim=0)
        raise ValueError(f"Unsupported pool={self.pool}")

    def forward(self, x: torch.Tensor | None, feat: torch.Tensor | None = None) -> torch.Tensor:
        # Encoder params have requires_grad=False (set in build.py), so no
        # context manager is needed here — head gradients flow normally.
        # When `feat` is pre-extracted (feature caching), pass x=None, feat=cached.
        if feat is None:
            feat = self._features(x)
        return self.head(feat)


@torch.no_grad()
def _acc_top1(
    logits: torch.Tensor, y: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    correct = (logits.argmax(dim=1) == y).sum()
    total = torch.tensor(y.numel(), device=y.device, dtype=torch.long)
    return correct, total


def _is_multilabel_probe(cfg) -> bool:
    data_cfg = getattr(cfg, "data", None)
    task_cfg = getattr(cfg, "task", None)
    target_type = str(
        getattr(
            data_cfg,
            "task_type",
            getattr(task_cfg, "target_type", ""),
        )
    ).lower()
    return target_type in {"multilabel", "multi-label", "multi_label"}


def _is_binary_auroc_probe(cfg, num_classes: int) -> bool:
    data_cfg = getattr(cfg, "data", None)
    task_cfg = getattr(cfg, "task", None)
    target_type = str(
        getattr(
            data_cfg,
            "task_type",
            getattr(task_cfg, "target_type", ""),
        )
    ).lower()
    data_name = str(getattr(data_cfg, "name", "")).lower()
    return (
        target_type in {"binary", "binary_auroc", "binary-classification"}
        or data_name in {"pneumoniamnist"}
    ) and int(num_classes) == 2


def _primary_metric_name(cfg, num_classes: int, multilabel: bool) -> str:
    if multilabel or _is_binary_auroc_probe(cfg, num_classes):
        return "auroc"
    return "acc1"


def _probe_loss(logits: torch.Tensor, y: torch.Tensor, multilabel: bool) -> torch.Tensor:
    if multilabel:
        return F.binary_cross_entropy_with_logits(logits, y.float())
    return F.cross_entropy(logits, y.long())


@torch.no_grad()
def _probe_correct_total(
    logits: torch.Tensor,
    y: torch.Tensor,
    multilabel: bool,
    threshold: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if multilabel:
        pred = torch.sigmoid(logits) >= threshold
        target = y.bool()
        correct = (pred == target).sum()
        total = torch.tensor(y.numel(), device=y.device, dtype=torch.long)
        return correct, total
    return _acc_top1(logits, y.long())


def _gather_metric_tensors(
    logits: torch.Tensor,
    y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not is_distributed():
        return logits.detach().float().cpu(), y.detach().cpu()

    logits = logits.detach().contiguous()
    y = y.detach().contiguous()
    gathered_logits = [torch.empty_like(logits) for _ in range(dist.get_world_size())]
    gathered_y = [torch.empty_like(y) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_logits, logits)
    dist.all_gather(gathered_y, y)
    return torch.cat(gathered_logits, dim=0).float().cpu(), torch.cat(gathered_y, dim=0).cpu()


def _auroc_from_parts(
    logits_parts: list[torch.Tensor],
    y_parts: list[torch.Tensor],
    *,
    multilabel: bool,
) -> float:
    logits = torch.cat(logits_parts, dim=0)
    y = torch.cat(y_parts, dim=0)
    logits, y = _gather_metric_tensors(logits, y)
    if multilabel:
        return _multilabel_mean_auroc(logits, y.float())
    return _binary_classification_auroc(logits, y.long())


def _multilabel_mean_auroc(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Macro AUROC over labels, skipping labels without both classes present."""
    aucs = []
    for class_idx in range(logits.shape[1]):
        auc = _binary_auroc(logits[:, class_idx], y[:, class_idx])
        if auc is not None:
            aucs.append(auc)
    if not aucs:
        return float("nan")
    return float(sum(aucs) / len(aucs))


def _binary_classification_auroc(logits: torch.Tensor, y: torch.Tensor) -> float:
    if logits.ndim == 2 and logits.shape[1] == 2:
        scores = logits[:, 1] - logits[:, 0]
    else:
        scores = logits.reshape(-1)
    auc = _binary_auroc(scores, y.reshape(-1))
    return float("nan") if auc is None else float(auc)


def _binary_auroc(scores: torch.Tensor, y: torch.Tensor) -> float | None:
    target = y.bool().flatten()
    scores = scores.float().flatten()
    n_pos = int(target.sum().item())
    n_total = int(target.numel())
    n_neg = n_total - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    order = torch.argsort(scores)
    sorted_scores = scores[order]
    ranks = torch.empty(n_total, dtype=torch.float64)

    start = 0
    while start < n_total:
        end = start + 1
        while end < n_total and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end

    pos_rank_sum = ranks[target].sum().item()
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


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


def _build_optimizer(model: nn.Module, cfg) -> torch.optim.Optimizer:
    cfg_optim = getattr(getattr(cfg, "task", None), "optim", None)
    name = str(getattr(cfg_optim, "name", "sgd")).lower() if cfg_optim else "sgd"
    head_params = unwrap_model(model).head.parameters()
    lr = float(cfg.train.lr)
    weight_decay = float(cfg.train.weight_decay)

    if name == "sgd":
        momentum = float(getattr(cfg_optim, "momentum", 0.9)) if cfg_optim else 0.9
        return torch.optim.SGD(
            head_params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

    if name == "lars":
        momentum = float(getattr(cfg_optim, "momentum", 0.9)) if cfg_optim else 0.9
        eta = float(getattr(cfg_optim, "eta", 1e-3)) if cfg_optim else 1e-3
        eps = float(getattr(cfg_optim, "eps", 1e-8)) if cfg_optim else 1e-8
        return LARS(
            head_params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            eta=eta,
            eps=eps,
        )

    raise ValueError(f"Unknown probe optimizer='{name}'. Supported: 'sgd', 'lars'.")


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
    multilabel = _is_multilabel_probe(cfg)
    metric_name = _primary_metric_name(cfg, num_classes, multilabel)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    head = _build_head(
        int(cfg.model.embed_dim),
        int(num_classes),
        getattr(cfg.task, "head", None),
    ).to(device)
    model = LinearProbeModel(
        encoder=encoder,
        head=head,
        pool=str(getattr(cfg.task, "pool", "mean")),
    ).to(device)

    if bool(getattr(cfg, "compile", False)) and hasattr(torch, "compile"):
        # Compile the full (encoder+head) graph for linear-probe experiments.
        # Keep it before DDP wrapping, matching the pretrain codepath.
        model = torch.compile(model, dynamic=True)

    # Freeze encoder before DDP wrapping: with feature caching the encoder is never
    # called inside the DDP-wrapped forward, so unfrozen params would cause a
    # DDP allreduce deadlock (find_unused_parameters=False).
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

    # ------------------------------------------------------------------
    # Optimizer + optional scheduler (head params only)
    # ------------------------------------------------------------------
    opt = _build_optimizer(model, cfg)
    sched = _build_scheduler(opt, getattr(cfg.task, "sched", None), epochs)
    scaler = GradScaler("cuda", enabled=amp)

    # ------------------------------------------------------------------
    # Pre-extract features — encoder is frozen, no need to re-run it
    # each batch of each epoch.  Each DDP rank caches its own data shard;
    # DDP gradient sync for the head still fires normally via the model wrapper.
    # ------------------------------------------------------------------
    def encode_fn(x):
        t = unwrap_model(model)._features(x)
        return F.layer_norm(t, (t.shape[-1],))

    feats_tr, labs_tr   = _extract_features(encode_fn, train_loader, device, amp)
    feats_val, labs_val = _extract_features(encode_fn, val_loader,   device, amp)

    bsz = train_loader.batch_size
    train_cache = _TDL(TensorDataset(feats_tr, labs_tr),     batch_size=bsz, shuffle=True,  drop_last=True)
    val_cache   = _TDL(TensorDataset(feats_val, labs_val),   batch_size=bsz, shuffle=False)

    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------
    state = {"epoch": 0, "global_step": 0, "best_metric": 0.0}
    log_every = int(getattr(cfg.train, "log_every", 50))
    early_stop_patience = int(getattr(cfg.train, "early_stop_patience", 0))
    early_stop_min_epochs = int(getattr(cfg.train, "early_stop_min_epochs", 0))
    early_stop_min_delta = float(getattr(cfg.train, "early_stop_min_delta", 0.0))
    no_improve_epochs = 0
    best_epoch = -1
    best_val_metric = 0.0
    last_train_metric = 0.0
    last_val_metric = 0.0
    last_val_loss = 0.0

    callbacks.on_run_start(cfg=cfg, state=state, model=unwrap_model(model))

    for epoch in range(epochs):
        state["epoch"] = epoch
        callbacks.on_epoch_start(cfg=cfg, state=state)

        # --------------------------------------------------------------
        # Train — encoder stays frozen; DDP syncs head grads via model wrapper
        # --------------------------------------------------------------
        unwrap_model(model).head.train()

        loss_meter = AverageMeter()
        correct_sum = torch.zeros((), device=device, dtype=torch.long)
        total_sum = torch.zeros((), device=device, dtype=torch.long)
        train_logits_parts: list[torch.Tensor] = []
        train_target_parts: list[torch.Tensor] = []

        for feat, y in train_cache:
            opt.zero_grad(set_to_none=True)
            with autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logits = model(None, feat=feat)
                loss = _probe_loss(logits, y, multilabel)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            loss_meter.update(float(loss.item()), n=feat.size(0))
            state["global_step"] += 1

            with torch.no_grad():
                if metric_name == "auroc":
                    train_logits_parts.append(logits.detach())
                    train_target_parts.append(y.detach())
                else:
                    c, t = _probe_correct_total(logits, y, multilabel)
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

        if metric_name == "auroc":
            train_metric = _auroc_from_parts(
                train_logits_parts,
                train_target_parts,
                multilabel=multilabel,
            )
        else:
            correct_sum = all_reduce_sum(correct_sum.float())
            total_sum = all_reduce_sum(total_sum.float())
            train_metric = (correct_sum / total_sum.clamp(min=1.0)).item()

        # --------------------------------------------------------------
        # Validate
        # --------------------------------------------------------------
        unwrap_model(model).eval()

        val_loss_sum = torch.zeros((), device=device)
        val_count = torch.zeros((), device=device)
        val_correct = torch.zeros((), device=device, dtype=torch.long)
        val_total = torch.zeros((), device=device, dtype=torch.long)
        val_logits_parts: list[torch.Tensor] = []
        val_target_parts: list[torch.Tensor] = []

        with torch.no_grad():
            for feat, y in val_cache:
                with autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    logits = model(None, feat=feat)
                    loss = _probe_loss(logits, y, multilabel)

                val_loss_sum += loss.detach() * feat.size(0)
                val_count += feat.size(0)
                if metric_name == "auroc":
                    val_logits_parts.append(logits.detach())
                    val_target_parts.append(y.detach())
                else:
                    c, t = _probe_correct_total(logits, y, multilabel)
                    val_correct += c
                    val_total += t

        val_loss_sum = all_reduce_sum(val_loss_sum)
        val_count = all_reduce_sum(val_count)

        val_loss = (val_loss_sum / val_count.clamp(min=1.0)).item()
        if metric_name == "auroc":
            val_metric = _auroc_from_parts(
                val_logits_parts,
                val_target_parts,
                multilabel=multilabel,
            )
        else:
            val_correct = all_reduce_sum(val_correct.float())
            val_total = all_reduce_sum(val_total.float())
            val_metric = (val_correct / val_total.clamp(min=1.0)).item()
        last_train_metric = float(train_metric)
        last_val_metric = float(val_metric)
        last_val_loss = float(val_loss)
        improved = val_metric > float(state["best_metric"]) + early_stop_min_delta

        if is_rank0():
            callbacks.on_epoch_end(
                cfg=cfg,
                state=state,
                metrics={
                    "probe/train_epoch_loss": float(loss_meter.avg),
                    f"probe/train_{metric_name}": float(train_metric),
                    "probe/val_loss": float(val_loss),
                    f"probe/val_{metric_name}": float(val_metric),
                    "probe/lr": float(opt.param_groups[0]["lr"]),
                    "probe/epoch": float(epoch),
                },
            )

            ckpt_dir = str(getattr(cfg.train, "ckpt_dir", "checkpoints"))
            os.makedirs(ckpt_dir, exist_ok=True)

            payload = {
                "head": unwrap_model(model).head.state_dict(),
                "metric_name": metric_name,
                "val_metric": val_metric,
                "epoch": epoch,
            }
            if metric_name == "acc1":
                payload["val_acc1"] = val_metric
            elif metric_name == "auroc":
                payload["val_auroc"] = val_metric

            if improved:
                state["best_metric"] = float(val_metric)
                best_val_metric = float(val_metric)
                best_epoch = int(epoch)
                no_improve_epochs = 0
                torch.save(payload, os.path.join(ckpt_dir, "linear_probe_best.pt"))
            else:
                no_improve_epochs += 1

            torch.save(payload, os.path.join(ckpt_dir, "linear_probe_last.pt"))

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
                    "[linear_probe_eval] early stopping triggered at "
                    f"epoch={epoch} after {no_improve_epochs} non-improving epochs."
                )
            break

    if is_rank0():
        callbacks.on_run_end(cfg=cfg, state=state)
    if best_epoch < 0:
        best_epoch = int(state["epoch"])
        best_val_metric = float(last_val_metric)
    return {
        "task_kind": "multilabel" if multilabel else "classification",
        "metric_name": metric_name,
        "train_metric": float(last_train_metric),
        "val_metric": float(last_val_metric),
        "best_val_metric": float(best_val_metric),
        "train_acc": float(last_train_metric),
        "val_acc": float(last_val_metric),
        "best_val_acc": float(best_val_metric),
        "best_epoch": int(best_epoch),
        "last_epoch": int(state["epoch"]),
        "val_loss": float(last_val_loss),
    }
