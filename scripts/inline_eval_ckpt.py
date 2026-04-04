"""
Hacky sanity-check script: run inline eval (linear probe on STL-10)
directly from one or two checkpoint files, matching exactly what
InlineEvalCallback does during pretraining.

Usage:
    python scripts/inline_eval_ckpt.py ckpt1.pt [ckpt2.pt] \
        --data-root /scratch/datasets \
        --probe-epochs 100 \
        --image-size 96 \
        --embed-dim 384 \
        --depth 12 \
        --num-heads 6 \
        --patch-size 8
"""
from __future__ import annotations

import argparse
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets as tv_datasets, transforms
from torchvision.models.vision_transformer import VisionTransformer


# ---------------------------------------------------------------------------
# Build + load target encoder (ViTTokens)
# ---------------------------------------------------------------------------

def _build_vit_tokens(image_size: int, patch_size: int, embed_dim: int,
                      depth: int, num_heads: int) -> nn.Module:
    """Build a ViTTokens instance without going through the full build system."""
    # Import the wrapper from the repo (must be on PYTHONPATH)
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
    # Remove classifier head — same as build_torchvision_vit_tokens
    if hasattr(vit, "heads"):
        vit.heads = nn.Identity()

    return ViTTokens(vit)


def _load_target_encoder(ckpt_path: str, image_size: int, patch_size: int,
                          embed_dim: int, depth: int, num_heads: int,
                          device: torch.device) -> nn.Module:
    """Extract target_encoder weights from an IJEPAModel checkpoint."""
    print(f"  Loading: {ckpt_path}")
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    full_sd = payload["model"]

    # Strip "target_encoder." prefix → ViTTokens state dict
    prefix = "target_encoder."
    enc_sd = {k[len(prefix):]: v for k, v in full_sd.items() if k.startswith(prefix)}

    encoder = _build_vit_tokens(image_size, patch_size, embed_dim, depth, num_heads)
    missing, unexpected = encoder.load_state_dict(enc_sd, strict=False)
    if missing:
        print(f"  [warn] missing keys ({len(missing)}): {missing[:5]}")
    if unexpected:
        print(f"  [warn] unexpected keys ({len(unexpected)}): {unexpected[:5]}")

    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    epoch = payload.get("state", {}).get("epoch", "?")
    print(f"  Checkpoint epoch: {epoch}")
    return encoder


# ---------------------------------------------------------------------------
# Feature extraction — mirrors InlineEvalCallback exactly
# ---------------------------------------------------------------------------

@torch.no_grad()
def _extract_features(encoder: nn.Module, loader: DataLoader,
                       device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (feats [N, D], labels [N]).

    encode_fn: encoder(x) returns (B, num_patches, D) patch tokens.
    Mean-pools to (B, D) — same as InlineEvalCallback._run_probe.
    """
    amp = device.type == "cuda"
    all_feats, all_labels = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            tokens = encoder(images)       # (B, N, D)
            feat   = tokens.mean(dim=1)    # (B, D)
        all_feats.append(feat.detach().float())
        all_labels.append(labels)
    return torch.cat(all_feats), torch.cat(all_labels)


# ---------------------------------------------------------------------------
# Probe loop — mirrors InlineEvalCallback._run_probe exactly
# ---------------------------------------------------------------------------

def _run_probe(encoder: nn.Module, embed_dim: int, num_classes: int,
               train_loader: DataLoader, val_loader: DataLoader,
               device: torch.device, probe_epochs: int, probe_lr: float,
               probe_wd: float, step_size: int, gamma: float) -> tuple[float, float]:
    amp = device.type == "cuda"

    print("  Extracting train features …")
    feats_tr, labs_tr = _extract_features(encoder, train_loader, device)
    print(f"  train feats: {tuple(feats_tr.shape)}")
    print("  Extracting val features …")
    feats_val, labs_val = _extract_features(encoder, val_loader, device)
    print(f"  val feats:   {tuple(feats_val.shape)}")

    bsz = train_loader.batch_size
    train_cache = DataLoader(TensorDataset(feats_tr, labs_tr),
                             batch_size=bsz, shuffle=True, drop_last=True)
    val_cache   = DataLoader(TensorDataset(feats_val, labs_val),
                             batch_size=bsz, shuffle=False)

    head = nn.Linear(embed_dim, num_classes).to(device)
    opt  = torch.optim.SGD(head.parameters(), lr=probe_lr,
                           momentum=0.9, weight_decay=probe_wd)
    sched  = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)
    scaler = GradScaler("cuda", enabled=amp)

    for ep in range(probe_epochs):
        head.train()
        for feat, y in train_cache:
            opt.zero_grad(set_to_none=True)
            with autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                loss = F.cross_entropy(head(feat), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
        if (ep + 1) % 20 == 0:
            print(f"    probe epoch {ep+1}/{probe_epochs}  "
                  f"lr={opt.param_groups[0]['lr']:.4g}")

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

    return _acc(train_cache), _acc(val_cache)


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

    ds_train = tv_datasets.STL10(data_root, split="train", download=False, transform=train_tfm)
    ds_val   = tv_datasets.STL10(data_root, split="test",  download=False, transform=val_tfm)

    kw = dict(num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(ds_train, batch_size=batch_size, shuffle=True,  drop_last=True, **kw),
        DataLoader(ds_val,   batch_size=batch_size, shuffle=False, **kw),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Inline eval sanity check from checkpoint(s)"
    )
    parser.add_argument("checkpoints", nargs="+", help="One or two .pt checkpoint paths")
    # Data
    parser.add_argument("--data-root",   default="/scratch/datasets")
    parser.add_argument("--batch-size",  type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    # Model (defaults match stl10_vits_ps8 experiment)
    parser.add_argument("--image-size",  type=int, default=96)
    parser.add_argument("--patch-size",  type=int, default=8)
    parser.add_argument("--embed-dim",   type=int, default=384)
    parser.add_argument("--depth",       type=int, default=12)
    parser.add_argument("--num-heads",   type=int, default=6)
    parser.add_argument("--num-classes", type=int, default=10)
    # Probe (defaults match InlineEvalCallback defaults)
    parser.add_argument("--probe-epochs", type=int,   default=100)
    parser.add_argument("--probe-lr",     type=float, default=0.1)
    parser.add_argument("--probe-wd",     type=float, default=0.0)
    parser.add_argument("--step-size",    type=int,   default=30)
    parser.add_argument("--gamma",        type=float, default=0.1)
    # Misc
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if len(args.checkpoints) > 2:
        print("Error: provide at most 2 checkpoints", file=sys.stderr)
        sys.exit(1)

    device = torch.device(args.device)
    print(f"Device: {device}\n")

    train_loader, val_loader = _build_loaders(
        args.data_root, args.image_size, args.batch_size, args.num_workers
    )
    print(f"STL-10  train={len(train_loader.dataset)}  val={len(val_loader.dataset)}\n")

    results = []
    for ckpt in args.checkpoints:
        print(f"{'='*60}")
        print(f"Checkpoint: {ckpt}")
        encoder = _load_target_encoder(
            ckpt, args.image_size, args.patch_size,
            args.embed_dim, args.depth, args.num_heads, device,
        )
        train_acc, val_acc = _run_probe(
            encoder, args.embed_dim, args.num_classes,
            train_loader, val_loader, device,
            args.probe_epochs, args.probe_lr, args.probe_wd,
            args.step_size, args.gamma,
        )
        results.append((ckpt, train_acc, val_acc))
        print(f"  → train_acc1={train_acc:.4f}  val_acc1={val_acc:.4f}\n")

    print(f"{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for ckpt, tr, val in results:
        print(f"  {ckpt}")
        print(f"    train_acc1 = {tr:.4f}   val_acc1 = {val:.4f}")


if __name__ == "__main__":
    main()
