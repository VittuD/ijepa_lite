"""
Attentive probing via CLS token + sklearn LogisticRegression.

Instead of mean-pooling patch tokens (as in inline_eval_ckpt.py), this script
uses the ViT's own CLS token as the image representation.  The CLS token is
already an attention-weighted aggregation of all patch tokens (via the
transformer's self-attention layers) — no additional learnable pooler needed.

The classifier is a sklearn StandardScaler + LogisticRegression (L-BFGS),
matching the logreg eval in eval_suite.py.  This avoids the overfitting that
occurs when training an MLP probe with SGD on the small STL-10 training set.

Usage:
    python scripts/attentive_probe_ckpt.py ckpt1.pt [ckpt2.pt] \\
        --data-root /scratch/datasets \\
        --image-size 96 \\
        --embed-dim 384 \\
        --depth 12 \\
        --num-heads 6 \\
        --patch-size 8
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader
from torchvision import datasets as tv_datasets, transforms
from torchvision.models.vision_transformer import VisionTransformer


# ---------------------------------------------------------------------------
# Build + load target encoder (ViTTokens)
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
    print(f"  Loading: {ckpt_path}")
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    full_sd = payload["model"]

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
# Feature extraction — CLS token (attention-pooled representation)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _extract_cls(encoder: nn.Module, loader: DataLoader,
                 device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Returns (feats [N, D], labels [N]) as numpy arrays.

    Uses the ViT CLS token as the image embedding.  The CLS token is an
    attention-weighted aggregation of all patch tokens via the transformer's
    own self-attention — no additional pooling module required.

    F.layer_norm is applied on the last dim to match the normalisation that
    ijepa.py applies to target tokens during pretraining.
    """
    amp = device.type == "cuda"
    all_feats, all_labels = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        with autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            cls_token, _ = encoder(images, return_cls=True)  # (B, D)
            cls_token = F.layer_norm(cls_token, (cls_token.shape[-1],))
        all_feats.append(cls_token.float().cpu())
        all_labels.append(labels.cpu())
    X = torch.cat(all_feats).numpy()
    y = torch.cat(all_labels).numpy()
    return X, y


# ---------------------------------------------------------------------------
# Probe — StandardScaler + LogisticRegression (mirrors logreg eval in eval_suite.py)
# ---------------------------------------------------------------------------

def _run_probe(encoder: nn.Module,
               train_loader: DataLoader, val_loader: DataLoader,
               device: torch.device,
               max_iter: int, C: float) -> tuple[float, float]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    print("  Extracting train CLS tokens …")
    X_train, y_train = _extract_cls(encoder, train_loader, device)
    print(f"  train feats: {X_train.shape}")
    print("  Extracting val CLS tokens …")
    X_val, y_val = _extract_cls(encoder, val_loader, device)
    print(f"  val feats:   {X_val.shape}")

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=max_iter, C=C, solver="lbfgs",
                           multi_class="multinomial"),
    )
    print(f"  Fitting LogisticRegression  C={C}  max_iter={max_iter} …")
    clf.fit(X_train, y_train)

    train_acc = float((clf.predict(X_train) == y_train).mean())
    val_acc   = float((clf.predict(X_val)   == y_val).mean())
    return train_acc, val_acc


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
        DataLoader(ds_train, batch_size=batch_size, shuffle=False, **kw),
        DataLoader(ds_val,   batch_size=batch_size, shuffle=False, **kw),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Attentive probe (CLS token + sklearn logreg) from checkpoint(s)"
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
    # Probe (mirrors eval_suite.py logreg defaults)
    parser.add_argument("--max-iter", type=int,   default=2000)
    parser.add_argument("--C",        type=float, default=1.0,
                        help="Inverse regularisation strength for LogisticRegression")
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
            encoder, train_loader, val_loader, device,
            args.max_iter, args.C,
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
