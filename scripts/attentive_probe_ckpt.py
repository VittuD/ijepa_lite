"""
Attentive probing sanity-check: evaluate one or two checkpoints on STL-10
using a learnable cross-attention pooler instead of mean-pooling.

A single learnable query attends to the frozen patch tokens via multi-head
attention; the attended vector goes to a linear classifier.  Only the probe
(attention pooler + head) is trained.

Usage:
    python scripts/attentive_probe_ckpt.py ckpt1.pt [ckpt2.pt] \\
        --data-root /scratch/datasets \\
        --probe-epochs 100 \\
        --image-size 96 \\
        --embed-dim 384 \\
        --depth 12 \\
        --num-heads 6 \\
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
# Attentive probe head
# ---------------------------------------------------------------------------

class AttentiveProbe(nn.Module):
    """
    Cross-attention pooler + linear classifier.

    A single learnable query attends to the N patch tokens via nn.MultiheadAttention.
    The attended vector is layer-normed and passed to a linear head.

    Args:
        embed_dim:   encoder hidden dimension D
        num_classes: number of output classes
        num_heads:   attention heads (must divide embed_dim evenly)
    """

    def __init__(self, embed_dim: int, num_classes: int, num_heads: int = 6) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (B, N, D) patch token sequence
        Returns:
            logits: (B, num_classes)
        """
        B = tokens.shape[0]
        q = self.query.expand(B, -1, -1)            # (B, 1, D)
        out, _ = self.attn(q, tokens, tokens)        # (B, 1, D)
        out = self.norm(out.squeeze(1))              # (B, D)
        return self.head(out)                        # (B, C)


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
# Feature extraction — caches full (B, N, D) patch tokens
# ---------------------------------------------------------------------------

@torch.no_grad()
def _extract_tokens(encoder: nn.Module, loader: DataLoader,
                    device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (tokens [N_total, num_patches, D], labels [N_total]).

    Applies F.layer_norm on the last dimension (same as ijepa.py) but does
    NOT mean-pool — the attentive probe needs the full token sequence.
    """
    amp = device.type == "cuda"
    all_tokens, all_labels = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            tokens = encoder(images)                             # (B, N, D)
            tokens = F.layer_norm(tokens, (tokens.shape[-1],))  # same as ijepa.py
        all_tokens.append(tokens.detach().float())
        all_labels.append(labels)
    return torch.cat(all_tokens), torch.cat(all_labels)


# ---------------------------------------------------------------------------
# Probe loop
# ---------------------------------------------------------------------------

def _run_probe(encoder: nn.Module, embed_dim: int, num_classes: int,
               train_loader: DataLoader, val_loader: DataLoader,
               device: torch.device, probe_epochs: int, probe_lr: float,
               probe_wd: float, step_size: int, gamma: float,
               num_heads: int) -> tuple[float, float]:
    amp = device.type == "cuda"

    print("  Extracting train tokens …")
    tokens_tr, labs_tr = _extract_tokens(encoder, train_loader, device)
    print(f"  train tokens: {tuple(tokens_tr.shape)}")
    print("  Extracting val tokens …")
    tokens_val, labs_val = _extract_tokens(encoder, val_loader, device)
    print(f"  val tokens:   {tuple(tokens_val.shape)}")

    bsz = train_loader.batch_size
    train_cache = DataLoader(TensorDataset(tokens_tr, labs_tr),
                             batch_size=bsz, shuffle=True, drop_last=True)
    val_cache   = DataLoader(TensorDataset(tokens_val, labs_val),
                             batch_size=bsz, shuffle=False)

    probe = AttentiveProbe(embed_dim, num_classes, num_heads=num_heads).to(device)
    print(f"  probe: AttentiveProbe  embed_dim={embed_dim}  num_heads={num_heads}")

    opt    = torch.optim.SGD(probe.parameters(), lr=probe_lr,
                             momentum=0.9, weight_decay=probe_wd)
    sched  = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)
    scaler = GradScaler("cuda", enabled=amp)

    for ep in range(probe_epochs):
        probe.train()
        for toks, y in train_cache:
            opt.zero_grad(set_to_none=True)
            with autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                loss = F.cross_entropy(probe(toks), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
        if (ep + 1) % 20 == 0:
            print(f"    probe epoch {ep+1}/{probe_epochs}  "
                  f"lr={opt.param_groups[0]['lr']:.4g}")

    @torch.no_grad()
    def _acc(cache):
        probe.eval()
        correct = total = 0
        for toks, y in cache:
            with autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logits = probe(toks)
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
        description="Attentive probe eval from checkpoint(s)"
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
    # Probe
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
            args.num_heads,
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
