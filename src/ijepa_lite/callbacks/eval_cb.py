from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from ijepa_lite.callbacks.base import Callback
from ijepa_lite.utils.dist import is_rank0, unwrap_model


def _build_dataset_local(name: str, root: str, split: str, transform):
    """Build a dataset directly — no barrier(), safe for rank-0-only use."""
    from torchvision import datasets as tv_datasets

    if name == "stl10":
        return tv_datasets.STL10(root=root, split=split, download=False, transform=transform)
    if name == "cifar10":
        return tv_datasets.CIFAR10(root=root, train=(split == "train"), download=False, transform=transform)
    if name == "cifar100":
        return tv_datasets.CIFAR100(root=root, train=(split == "train"), download=False, transform=transform)
    if name == "food101":
        return tv_datasets.Food101(root=root, split=split, download=False, transform=transform)
    raise ValueError(f"InlineEvalCallback: unsupported dataset '{name}'")


class InlineEvalCallback(Callback):
    """
    Opt-in inline downstream evaluation during pretraining.

    Runs a linear/MLP probe on a supervised dataset (e.g. STL-10) at a fixed
    epoch cadence.  Mutates the ``metrics`` dict in-place so that downstream
    loggers (e.g. WandbCallback) pick up the results automatically.

    Activated when ``cfg.train.eval_every_epochs > 0``.  Rank 0 only.
    """

    def __init__(self) -> None:
        self._eval_every: int = 0
        self._train_loader: DataLoader | None = None
        self._val_loader: DataLoader | None = None
        self._num_classes: int = 10
        self._embed_dim: int = 384
        self._device: torch.device = torch.device("cpu")
        self._cfg_inline: Any = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_run_start(self, cfg: Any, state: dict, model: Any) -> None:
        self._eval_every = int(getattr(cfg.train, "eval_every_epochs", 0))
        if self._eval_every <= 0 or not is_rank0():
            return

        self._embed_dim = int(cfg.model.embed_dim)
        self._device = next(model.parameters()).device

        icfg = getattr(cfg.train, "inline_eval", None)
        self._cfg_inline = icfg

        # Build eval dataloaders once and reuse across epochs.
        # NOTE: we build datasets directly (not via build_dataset) because
        # build_dataset calls barrier() for DDP coordination, but this callback
        # only runs on rank 0 — using build_dataset would deadlock.
        data_root = str(getattr(icfg, "data_root", "/scratch/datasets/"))
        dataset_name = str(getattr(icfg, "dataset", "stl10"))
        train_split = str(getattr(icfg, "train_split", "train"))
        val_split = str(getattr(icfg, "val_split", "test"))
        self._num_classes = int(getattr(icfg, "num_classes", 10))
        batch_size = int(getattr(icfg, "batch_size", 256))
        num_workers = int(getattr(icfg, "num_workers", 4))

        from ijepa_lite.data.transforms import build_linear_probe_transforms

        train_tfm, val_tfm = build_linear_probe_transforms(cfg)

        from ijepa_lite.data.collate import SupervisedCollate

        ds_train = _build_dataset_local(dataset_name, data_root, train_split, train_tfm)
        ds_val = _build_dataset_local(dataset_name, data_root, val_split, val_tfm)

        collate = SupervisedCollate()

        self._train_loader = DataLoader(
            ds_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate,
            drop_last=True,
        )
        self._val_loader = DataLoader(
            ds_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate,
            drop_last=False,
        )

        print(f"[InlineEval] Enabled: every {self._eval_every} epochs, "
              f"dataset={dataset_name}, num_classes={self._num_classes}")

    # ------------------------------------------------------------------
    # Epoch end — run probe
    # ------------------------------------------------------------------

    def on_epoch_end(self, cfg: Any, state: dict, metrics: Dict[str, float]) -> None:
        if self._eval_every <= 0 or not is_rank0():
            return

        epoch = int(state.get("epoch", 0))
        if (epoch + 1) % self._eval_every != 0:
            return

        icfg = self._cfg_inline

        # Extract target encoder from the live model (EMA, already frozen).
        bundle = state.get("_ckpt_bundle")
        if bundle is None:
            return
        model = bundle["model"]
        encoder = unwrap_model(model).target_encoder

        # Build a fresh head
        from ijepa_lite.engine.eval_linear import _build_head

        head_cfg_dict = {
            "type": str(getattr(icfg, "head_type", "mlp")),
            "hidden_dim": int(getattr(icfg, "head_hidden_dim", 384)),
            "num_layers": int(getattr(icfg, "head_num_layers", 1)),
        }
        from omegaconf import OmegaConf
        head_cfg = OmegaConf.create(head_cfg_dict)
        head = _build_head(self._embed_dim, self._num_classes, head_cfg).to(self._device)

        # Run the probe
        train_acc, val_acc = self._run_probe(encoder, head, icfg)

        metrics["inline_eval/train_acc1"] = train_acc
        metrics["inline_eval/val_acc1"] = val_acc

        print(f"[InlineEval] epoch={epoch}  train_acc1={train_acc:.4f}  val_acc1={val_acc:.4f}")

        # Restore training mode
        model.train()

    # ------------------------------------------------------------------
    # Self-contained probe loop
    # ------------------------------------------------------------------

    def _run_probe(
        self,
        encoder: nn.Module,
        head: nn.Module,
        icfg: Any,
    ) -> tuple[float, float]:
        probe_epochs = int(getattr(icfg, "probe_epochs", 100))
        probe_lr = float(getattr(icfg, "probe_lr", 0.1))
        probe_wd = float(getattr(icfg, "probe_weight_decay", 0.0))
        sched_name = str(getattr(icfg, "probe_sched", "step")).lower()

        amp = self._device.type == "cuda"

        opt = torch.optim.SGD(
            head.parameters(), lr=probe_lr, momentum=0.9, weight_decay=probe_wd,
        )

        sched = None
        if sched_name == "step":
            step_size = int(getattr(icfg, "probe_step_size", 30))
            gamma = float(getattr(icfg, "probe_gamma", 0.1))
            sched = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)

        scaler = GradScaler("cuda", enabled=amp)

        encoder.eval()

        for _ep in range(probe_epochs):
            head.train()
            for batch in self._train_loader:
                x = batch["images"].to(self._device, non_blocking=True)
                y = batch["labels"].to(self._device, non_blocking=True)

                opt.zero_grad(set_to_none=True)
                with torch.no_grad():
                    with autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                        tokens = encoder(x)
                        feat = tokens.mean(dim=1)
                with autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    logits = head(feat)
                    loss = F.cross_entropy(logits, y)

                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

            if sched is not None:
                sched.step()

        # Final train accuracy
        train_acc = self._evaluate(encoder, head, self._train_loader, amp)
        val_acc = self._evaluate(encoder, head, self._val_loader, amp)
        return train_acc, val_acc

    @torch.no_grad()
    def _evaluate(
        self,
        encoder: nn.Module,
        head: nn.Module,
        loader: DataLoader,
        amp: bool,
    ) -> float:
        encoder.eval()
        head.eval()
        correct = 0
        total = 0
        for batch in loader:
            x = batch["images"].to(self._device, non_blocking=True)
            y = batch["labels"].to(self._device, non_blocking=True)
            with autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                tokens = encoder(x)
                feat = tokens.mean(dim=1)
                logits = head(feat)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.numel()
        return correct / max(total, 1)
