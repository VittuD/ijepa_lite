from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, DataLoader as _TDL, TensorDataset

from ijepa_lite.callbacks.base import Callback
from ijepa_lite.data.datasets import _build_dataset_local
from ijepa_lite.engine.eval_linear import _extract_features
from ijepa_lite.utils.dist import is_rank0, unwrap_model


def _infer_num_classes_local(dataset, configured: Any) -> int:
    if configured is not None:
        return int(configured)
    if hasattr(dataset, "classes"):
        return len(dataset.classes)
    if hasattr(dataset, "class_to_idx"):
        return len(dataset.class_to_idx)
    raise ValueError(
        "InlineEvalCallback could not infer num_classes from the dataset. "
        "Set train.inline_eval.num_classes explicitly."
    )


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
        batch_size = int(getattr(icfg, "batch_size", 256))
        num_workers = int(getattr(icfg, "num_workers", 4))

        from ijepa_lite.data.transforms import build_linear_probe_transforms

        train_tfm, val_tfm = build_linear_probe_transforms(cfg)

        from ijepa_lite.data.collate import SupervisedCollate

        dataset_cfg = SimpleNamespace(
            name=dataset_name,
            root=data_root,
            size=getattr(icfg, "size", 128),
            as_rgb=getattr(icfg, "as_rgb", True),
            download=getattr(icfg, "download", False),
            mmap_mode=getattr(icfg, "mmap_mode", None),
            task_type=getattr(icfg, "task_type", "classification"),
            class_names=list(getattr(icfg, "class_names", [])),
            partition=getattr(icfg, "partition", 1),
            train_ratio=getattr(icfg, "train_ratio", 0.8),
            val_ratio=getattr(icfg, "val_ratio", 0.1),
            split_seed=getattr(icfg, "split_seed", 0),
            target_attr=getattr(icfg, "target_attr", "race"),
            image_dirname=getattr(
                icfg, "image_dirname", "fairface-img-margin025-trainval"
            ),
            train_csv=getattr(icfg, "train_csv", "fairface_label_train.csv"),
            val_csv=getattr(icfg, "val_csv", "fairface_label_val.csv"),
            csv_dir=getattr(icfg, "csv_dir", None),
            image_root=getattr(icfg, "image_root", None),
        )

        ds_train = _build_dataset_local(dataset_cfg, train_split, train_tfm)
        ds_val = _build_dataset_local(dataset_cfg, val_split, val_tfm)
        self._num_classes = _infer_num_classes_local(
            ds_train, getattr(icfg, "num_classes", None)
        )

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

        # Extract the representation encoder from the live model.
        bundle = state.get("_ckpt_bundle")
        if bundle is None:
            return
        model = bundle["model"]
        encoder = unwrap_model(model).get_eval_encoder()

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

        metric_name = (
            "label_acc"
            if _is_multilabel_inline_eval(icfg)
            else "acc1"
        )
        metrics[f"inline_eval/train_{metric_name}"] = train_acc
        metrics[f"inline_eval/val_{metric_name}"] = val_acc

        print(
            f"[InlineEval] epoch={epoch}  "
            f"train_{metric_name}={train_acc:.4f}  "
            f"val_{metric_name}={val_acc:.4f}"
        )

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
        multilabel = _is_multilabel_inline_eval(icfg)

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

        # Pre-extract features once — encoder is frozen so features are epoch-invariant.
        # Apply the same F.layer_norm used on target tokens during pretraining (ijepa.py).
        def encode_fn(x):
            t = encoder(x)
            return F.layer_norm(t, (t.shape[-1],)).mean(dim=1)
        feats_tr, labs_tr   = _extract_features(encode_fn, self._train_loader, self._device, amp)
        feats_val, labs_val = _extract_features(encode_fn, self._val_loader,   self._device, amp)

        bsz = self._train_loader.batch_size
        train_cache = _TDL(TensorDataset(feats_tr, labs_tr),   batch_size=bsz, shuffle=True,  drop_last=True)
        val_cache   = _TDL(TensorDataset(feats_val, labs_val), batch_size=bsz, shuffle=False)

        for _ep in range(probe_epochs):
            head.train()
            for feat, y in train_cache:
                opt.zero_grad(set_to_none=True)
                with autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    logits = head(feat)
                    loss = _inline_eval_loss(logits, y, multilabel)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

            if sched is not None:
                sched.step()

        # Final accuracy over cached features
        train_acc = self._evaluate(head, train_cache, amp, multilabel)
        val_acc   = self._evaluate(head, val_cache,   amp, multilabel)
        return train_acc, val_acc

    @torch.no_grad()
    def _evaluate(
        self,
        head: nn.Module,
        cache_loader: _TDL,
        amp: bool,
        multilabel: bool = False,
    ) -> float:
        head.eval()
        correct = 0
        total = 0
        for feat, y in cache_loader:
            with autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logits = head(feat)
            c, t = _inline_eval_correct_total(logits, y, multilabel)
            correct += c
            total += t
        return correct / max(total, 1)


def _is_multilabel_inline_eval(icfg: Any) -> bool:
    target_type = str(getattr(icfg, "task_type", "")).lower()
    return target_type in {"multilabel", "multi-label", "multi_label"}


def _inline_eval_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    multilabel: bool,
) -> torch.Tensor:
    if multilabel:
        return F.binary_cross_entropy_with_logits(logits, y.float())
    return F.cross_entropy(logits, y.long())


@torch.no_grad()
def _inline_eval_correct_total(
    logits: torch.Tensor,
    y: torch.Tensor,
    multilabel: bool,
    threshold: float = 0.5,
) -> tuple[int, int]:
    if multilabel:
        pred = torch.sigmoid(logits) >= threshold
        target = y.bool()
        return int((pred == target).sum().item()), int(y.numel())
    return int((logits.argmax(dim=1) == y.long()).sum().item()), int(y.numel())
