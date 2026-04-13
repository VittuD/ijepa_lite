"""
Probe benchmarking script: compare all three probing strategies on STL-10
for a given checkpoint.

Probes evaluated
----------------
  logreg_mean  : mean-pool patch tokens  → StandardScaler + LogisticRegression
  logreg_cls   : CLS token               → StandardScaler + LogisticRegression
  mlp_sgd      : mean-pool patch tokens  → MLP head (Linear-BN-ReLU-Linear)
                                            trained with SGD (shows overfitting)

Metrics reported per probe
--------------------------
  train_acc    : training-set accuracy (gap vs val_acc reveals overfitting)
  val_acc      : validation accuracy (primary quality metric)
  time_s       : wall-clock time for extraction + fitting/training
  peak_gpu_mb  : peak GPU memory during the probe run

Usage
-----
    python scripts/probe_benchmark.py ckpt.pt \\
        --data-root /scratch/datasets \\
        --image-size 96 --embed-dim 384 --depth 12 --num-heads 6 --patch-size 8
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets as tv_datasets, transforms
from torchvision.models.vision_transformer import VisionTransformer


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ProbeResult:
    name: str
    train_acc: float
    val_acc: float
    time_s: float
    peak_gpu_mb: float


# ---------------------------------------------------------------------------
# Encoder build + load
# ---------------------------------------------------------------------------

def _build_vit_tokens(image_size: int, patch_size: int, embed_dim: int,
                      depth: int, num_heads: int) -> nn.Module:
    from ijepa_lite.models.vit_tokens import ViTTokens

    vit = VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=depth,
        num_heads=num_heads,
        hidden_dim=embed_dim,
        mlp_dim=embed_dim * 4,
        num_classes=1000,
    )
    if hasattr(vit, "heads"):
        vit.heads = nn.Identity()
    return ViTTokens(vit)


def _load_target_encoder(ckpt_path: str, image_size: int, patch_size: int,
                          embed_dim: int, depth: int, num_heads: int,
                          device: torch.device) -> nn.Module:
    print(f"Loading: {ckpt_path}")
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    prefix  = "target_encoder."
    enc_sd  = {k[len(prefix):]: v
               for k, v in payload["model"].items() if k.startswith(prefix)}

    encoder = _build_vit_tokens(image_size, patch_size, embed_dim, depth, num_heads)
    missing, unexpected = encoder.load_state_dict(enc_sd, strict=False)
    if missing:
        print(f"  [warn] missing keys ({len(missing)}): {missing[:5]}")
    if unexpected:
        print(f"  [warn] unexpected keys ({len(unexpected)}): {unexpected[:5]}")

    encoder = encoder.to(device).eval()
    encoder.requires_grad_(False)
    epoch = payload.get("state", {}).get("epoch", "?")
    print(f"  epoch: {epoch}\n")
    return encoder


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _extract_mean_pool(encoder: nn.Module, loader: DataLoader,
                       device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Mean-pool layer-normed patch tokens → (N, D)."""
    amp = device.type == "cuda"
    feats, labels = [], []
    for images, y in loader:
        images = images.to(device, non_blocking=True)
        with autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            t = encoder(images)
            t = F.layer_norm(t, (t.shape[-1],))
            f = t.mean(dim=1)
        feats.append(f.float().cpu())
        labels.append(y.cpu())
    return torch.cat(feats).numpy(), torch.cat(labels).numpy()


@torch.no_grad()
def _extract_cls(encoder: nn.Module, loader: DataLoader,
                 device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Layer-normed CLS token → (N, D)."""
    amp = device.type == "cuda"
    feats, labels = [], []
    for images, y in loader:
        images = images.to(device, non_blocking=True)
        with autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            cls, _ = encoder(images, return_cls=True)
            cls = F.layer_norm(cls, (cls.shape[-1],))
        feats.append(cls.float().cpu())
        labels.append(y.cpu())
    return torch.cat(feats).numpy(), torch.cat(labels).numpy()


# ---------------------------------------------------------------------------
# Probe 1: logreg_mean
# ---------------------------------------------------------------------------

def probe_logreg_mean(encoder: nn.Module,
                      train_loader: DataLoader, val_loader: DataLoader,
                      device: torch.device,
                      max_iter: int, C: float) -> ProbeResult:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.perf_counter()

    X_tr, y_tr = _extract_mean_pool(encoder, train_loader, device)
    X_va, y_va = _extract_mean_pool(encoder, val_loader,   device)

    peak_mb = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else 0.0

    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=max_iter, C=C, solver="lbfgs"))
    clf.fit(X_tr, y_tr)

    elapsed = time.perf_counter() - t0

    return ProbeResult(
        name        = "logreg_mean",
        train_acc   = float((clf.predict(X_tr) == y_tr).mean()),
        val_acc     = float((clf.predict(X_va) == y_va).mean()),
        time_s      = elapsed,
        peak_gpu_mb = peak_mb,
    )


# ---------------------------------------------------------------------------
# Probe 2: logreg_cls
# ---------------------------------------------------------------------------

def probe_logreg_cls(encoder: nn.Module,
                     train_loader: DataLoader, val_loader: DataLoader,
                     device: torch.device,
                     max_iter: int, C: float) -> ProbeResult:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.perf_counter()

    X_tr, y_tr = _extract_cls(encoder, train_loader, device)
    X_va, y_va = _extract_cls(encoder, val_loader,   device)

    peak_mb = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else 0.0

    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=max_iter, C=C, solver="lbfgs"))
    clf.fit(X_tr, y_tr)

    elapsed = time.perf_counter() - t0

    return ProbeResult(
        name        = "logreg_cls",
        train_acc   = float((clf.predict(X_tr) == y_tr).mean()),
        val_acc     = float((clf.predict(X_va) == y_va).mean()),
        time_s      = elapsed,
        peak_gpu_mb = peak_mb,
    )


# ---------------------------------------------------------------------------
# Probe 3: mlp_sgd  (kept to show overfitting; matches InlineEvalCallback)
# ---------------------------------------------------------------------------

def probe_mlp_sgd(encoder: nn.Module, embed_dim: int, num_classes: int,
                  train_loader: DataLoader, val_loader: DataLoader,
                  device: torch.device,
                  probe_epochs: int, probe_lr: float, probe_wd: float,
                  step_size: int, gamma: float,
                  head_hidden_dim: int, head_num_layers: int) -> ProbeResult:
    from ijepa_lite.engine.eval_linear import _build_head
    from omegaconf import OmegaConf

    amp = device.type == "cuda"

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.perf_counter()

    # --- feature extraction (mean-pool, same as logreg_mean) ---
    X_tr_t, y_tr_t = [], []
    X_va_t, y_va_t = [], []
    with torch.no_grad():
        for images, y in train_loader:
            images = images.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                t = encoder(images)
                t = F.layer_norm(t, (t.shape[-1],))
                f = t.mean(dim=1)
            X_tr_t.append(f.detach().float()); y_tr_t.append(y)
        for images, y in val_loader:
            images = images.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                t = encoder(images)
                t = F.layer_norm(t, (t.shape[-1],))
                f = t.mean(dim=1)
            X_va_t.append(f.detach().float()); y_va_t.append(y)

    feats_tr = torch.cat(X_tr_t); labs_tr = torch.cat(y_tr_t)
    feats_va = torch.cat(X_va_t); labs_va = torch.cat(y_va_t)

    bsz = train_loader.batch_size
    train_cache = DataLoader(TensorDataset(feats_tr, labs_tr),
                             batch_size=bsz, shuffle=True, drop_last=True)
    val_cache   = DataLoader(TensorDataset(feats_va, labs_va),
                             batch_size=bsz, shuffle=False)

    # --- build head ---
    head_cfg = OmegaConf.create({"type": "mlp",
                                  "hidden_dim": head_hidden_dim,
                                  "num_layers": head_num_layers})
    head = _build_head(embed_dim, num_classes, head_cfg).to(device)

    opt    = torch.optim.SGD(head.parameters(), lr=probe_lr,
                             momentum=0.9, weight_decay=probe_wd)
    sched  = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)
    scaler = GradScaler("cuda", enabled=amp)

    # --- training loop ---
    for _ in range(probe_epochs):
        head.train()
        for feat, y in train_cache:
            opt.zero_grad(set_to_none=True)
            with autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                loss = F.cross_entropy(head(feat), y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
        sched.step()

    peak_mb = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else 0.0
    elapsed = time.perf_counter() - t0

    @torch.no_grad()
    def _acc(cache):
        head.eval()
        correct = total = 0
        for feat, y in cache:
            with autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logits = head(feat)
            correct += (logits.argmax(1) == y).sum().item()
            total   += y.numel()
        return correct / max(total, 1)

    return ProbeResult(
        name        = "mlp_sgd",
        train_acc   = _acc(train_cache),
        val_acc     = _acc(val_cache),
        time_s      = elapsed,
        peak_gpu_mb = peak_mb,
    )


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _build_loaders(data_root: str, image_size: int,
                   batch_size: int, num_workers: int) -> tuple[DataLoader, DataLoader]:
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)
    train_tfm = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.3, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tfm = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    ds_tr = tv_datasets.STL10(data_root, split="train", download=False, transform=train_tfm)
    ds_va = tv_datasets.STL10(data_root, split="test",  download=False, transform=val_tfm)
    kw = dict(num_workers=num_workers, pin_memory=True)
    return (DataLoader(ds_tr, batch_size=batch_size, shuffle=False, **kw),
            DataLoader(ds_va, batch_size=batch_size, shuffle=False, **kw))


# ---------------------------------------------------------------------------
# Pretty-print table
# ---------------------------------------------------------------------------

def _print_table(results: list[ProbeResult]) -> None:
    hdr = f"{'probe':<14}  {'train_acc':>9}  {'val_acc':>8}  {'time_s':>8}  {'peak_gpu_MB':>12}"
    sep = "-" * len(hdr)
    print(sep)
    print(hdr)
    print(sep)
    for r in results:
        print(f"{r.name:<14}  {r.train_acc:>9.4f}  {r.val_acc:>8.4f}"
              f"  {r.time_s:>8.1f}  {r.peak_gpu_mb:>12.1f}")
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark three probe strategies on STL-10 from a checkpoint"
    )
    parser.add_argument("checkpoint", help=".pt checkpoint path")
    # Data
    parser.add_argument("--data-root",   default="/scratch/datasets")
    parser.add_argument("--batch-size",  type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    # Model (defaults: stl10_vits_ps8)
    parser.add_argument("--image-size",  type=int, default=96)
    parser.add_argument("--patch-size",  type=int, default=8)
    parser.add_argument("--embed-dim",   type=int, default=384)
    parser.add_argument("--depth",       type=int, default=12)
    parser.add_argument("--num-heads",   type=int, default=6)
    parser.add_argument("--num-classes", type=int, default=10)
    # sklearn probe args
    parser.add_argument("--max-iter",    type=int,   default=2000)
    parser.add_argument("--C",           type=float, default=1.0)
    # MLP probe args (intentionally match InlineEvalCallback defaults)
    parser.add_argument("--probe-epochs",    type=int,   default=100)
    parser.add_argument("--probe-lr",        type=float, default=0.1)
    parser.add_argument("--probe-wd",        type=float, default=0.0)
    parser.add_argument("--step-size",       type=int,   default=30)
    parser.add_argument("--gamma",           type=float, default=0.1)
    parser.add_argument("--head-hidden-dim", type=int,   default=384)
    parser.add_argument("--head-num-layers", type=int,   default=1)
    # Misc
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-mlp", action="store_true",
                        help="Skip the slow mlp_sgd probe")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}\n")

    train_loader, val_loader = _build_loaders(
        args.data_root, args.image_size, args.batch_size, args.num_workers,
    )
    print(f"STL-10  train={len(train_loader.dataset)}  val={len(val_loader.dataset)}\n")

    encoder = _load_target_encoder(
        args.checkpoint, args.image_size, args.patch_size,
        args.embed_dim, args.depth, args.num_heads, device,
    )

    results: list[ProbeResult] = []

    print("[ 1/3 ]  logreg_mean …")
    results.append(probe_logreg_mean(
        encoder, train_loader, val_loader, device, args.max_iter, args.C,
    ))

    print("[ 2/3 ]  logreg_cls …")
    results.append(probe_logreg_cls(
        encoder, train_loader, val_loader, device, args.max_iter, args.C,
    ))

    if not args.skip_mlp:
        print("[ 3/3 ]  mlp_sgd …")
        results.append(probe_mlp_sgd(
            encoder, args.embed_dim, args.num_classes,
            train_loader, val_loader, device,
            args.probe_epochs, args.probe_lr, args.probe_wd,
            args.step_size, args.gamma,
            args.head_hidden_dim, args.head_num_layers,
        ))
    else:
        print("[ 3/3 ]  mlp_sgd  (skipped)")

    print()
    _print_table(results)


if __name__ == "__main__":
    main()
