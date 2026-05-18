from __future__ import annotations

import os
from typing import Tuple

import torch
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
        tokens = self.encoder(x)  # (B, N, D) patch tokens
        if self.pool == "mean":
            return tokens.mean(dim=1)  # (B, D)
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
    state = {"epoch": 0, "global_step": 0, "best_acc1": 0.0}
    log_every = int(getattr(cfg.train, "log_every", 50))
    early_stop_patience = int(getattr(cfg.train, "early_stop_patience", 0))
    early_stop_min_epochs = int(getattr(cfg.train, "early_stop_min_epochs", 0))
    early_stop_min_delta = float(getattr(cfg.train, "early_stop_min_delta", 0.0))
    no_improve_epochs = 0
    best_epoch = -1
    best_val_acc1 = 0.0
    last_train_acc1 = 0.0
    last_val_acc1 = 0.0
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

        for feat, y in train_cache:
            opt.zero_grad(set_to_none=True)
            with autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logits = model(None, feat=feat)
                loss = F.cross_entropy(logits, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            loss_meter.update(float(loss.item()), n=feat.size(0))
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
            for feat, y in val_cache:
                with autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    logits = model(None, feat=feat)
                    loss = F.cross_entropy(logits, y)

                val_loss_sum += loss.detach() * feat.size(0)
                val_correct += _acc_top1(logits, y)[0]
                val_total += y.numel()

        val_loss_sum = all_reduce_sum(val_loss_sum)
        val_correct = all_reduce_sum(val_correct.float())
        val_total = all_reduce_sum(val_total.float())

        val_loss = (val_loss_sum / val_total.clamp(min=1.0)).item()
        val_acc1 = (val_correct / val_total.clamp(min=1.0)).item()
        last_train_acc1 = float(train_acc1)
        last_val_acc1 = float(val_acc1)
        last_val_loss = float(val_loss)
        improved = val_acc1 > float(state["best_acc1"]) + early_stop_min_delta

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

            if improved:
                state["best_acc1"] = float(val_acc1)
                best_val_acc1 = float(val_acc1)
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
        best_val_acc1 = float(last_val_acc1)
    return {
        "task_kind": "classification",
        "train_acc": float(last_train_acc1),
        "val_acc": float(last_val_acc1),
        "best_val_acc": float(best_val_acc1),
        "best_epoch": int(best_epoch),
        "last_epoch": int(state["epoch"]),
        "val_loss": float(last_val_loss),
    }
