#!/usr/bin/env python3
"""
hacky_cross_ckpt_probe.py

Pool-bin ablation using encoder and masker loaded from SEPARATE checkpoints.

Phase-2 diagnostic: take the EMA encoder from a vanilla JEPA run (Phase 1)
and pair it with the learned masker from the MI-coupled run. If the masker
produces content-adaptive assignments when given a better encoder, the
co-adaptation-trap hypothesis is confirmed.

All pools share the same frozen encoder; the masker is only invoked for
masker-based pools (ctx / tgt / ign / ctx+tgt). The "center" pool needs no
masker and serves as the positional-prior baseline.

Usage:
  python scripts/hacky/hacky_cross_ckpt_probe.py \\
      --encoder-ckpt /path/to/vanilla_last.pt \\
      --masker-ckpt  /path/to/mi_coupled_last.pt \\
      --data-root    /path/to/datasets/ \\
      --dataset      stl10 \\
      --pools        ign center \\
      --seeds        0 42 \\
      --epochs       100 \\
      --device       cuda

  # masker-ckpt optional — omit when only running center / all pools
  python scripts/hacky/hacky_cross_ckpt_probe.py \\
      --encoder-ckpt /path/to/vanilla_last.pt \\
      --data-root    /path/to/datasets/ \\
      --dataset      stl10 \\
      --pools        center all \\
      --seeds        0 42
"""
import argparse
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets as tv_datasets, transforms
from torchvision.models.vision_transformer import VisionTransformer

import ijepa_lite.masking.mi_masker  # noqa: F401 — populates registry
from ijepa_lite.masking.mi_masker import MIRateMasker
from ijepa_lite.models.vit_tokens import ViTTokens, _remove_classifier_head

# ---------------------------------------------------------------------------
# Architecture constants (stl10 ViT-S/8)
# ---------------------------------------------------------------------------
IMG_SIZE   = 96
PATCH_SIZE = 8
EMBED_DIM  = 384
DEPTH      = 12
NUM_HEADS  = 6
PRED_DIM   = 192
PRED_DEPTH = 2
PRED_HEADS = 6
GRID_H = GRID_W = IMG_SIZE // PATCH_SIZE   # 12

LAM   = 1.0
ALPHA = 0.01

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _load_sd(path: str) -> dict:
    sd = torch.load(path, map_location="cpu", weights_only=True)
    return sd.get("model", sd)


def build_encoder(ckpt_path: str, device: torch.device) -> ViTTokens:
    vit = VisionTransformer(
        image_size=IMG_SIZE, patch_size=PATCH_SIZE, num_layers=DEPTH,
        num_heads=NUM_HEADS, hidden_dim=EMBED_DIM, mlp_dim=EMBED_DIM * 4,
        num_classes=1000,
    )
    _remove_classifier_head(vit)
    enc = ViTTokens(vit)

    sd = _load_sd(ckpt_path)
    prefix = "target_encoder."
    enc_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    missing, _ = enc.load_state_dict(enc_sd, strict=False)
    if missing:
        print(f"  [encoder] missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    enc.requires_grad_(False)
    return enc.to(device).eval()


def build_masker(ckpt_path: str, device: torch.device) -> MIRateMasker:
    masker = MIRateMasker(
        dim=EMBED_DIM, predictor_dim=PRED_DIM, depth=PRED_DEPTH,
        num_heads=PRED_HEADS, mlp_ratio=4.0, dropout=0.0,
        num_patches=GRID_H * GRID_W,
        lam_min=0.01, lam_max=1.0, alpha_min=0.01, alpha_max=0.5,
        coupled_scalarization=True, ratio_logit_std=1.0,
        h_floor=0.1, floor_weight=5.0, lam_warmup_epochs=0,
    )

    sd = _load_sd(ckpt_path)
    prefix = "latent_masker."
    m_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    known_optional = {"_ema_mi_rate", "_ema_surprise"}
    missing, _ = masker.load_state_dict(m_sd, strict=False)
    real_missing = [k for k in missing if k not in known_optional]
    if real_missing:
        print(f"  [masker] missing keys: {real_missing[:5]}{'...' if len(real_missing) > 5 else ''}")

    masker.requires_grad_(False)
    return masker.to(device).eval()

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_loaders(dataset: str, data_root: str, batch_size: int):
    tfm = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    if dataset == "stl10":
        ds_train = tv_datasets.STL10(data_root, split="train", download=False, transform=tfm)
        ds_val   = tv_datasets.STL10(data_root, split="test",  download=False, transform=tfm)
        num_classes = 10
    elif dataset == "food101":
        ds_train = tv_datasets.Food101(data_root, split="train", download=False, transform=tfm)
        ds_val   = tv_datasets.Food101(data_root, split="test",  download=False, transform=tfm)
        num_classes = 101
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")

    kw = dict(num_workers=8, pin_memory=True)
    train_loader = torch.utils.data.DataLoader(
        ds_train, batch_size=batch_size, shuffle=True,  drop_last=True,  **kw
    )
    val_loader = torch.utils.data.DataLoader(
        ds_val,   batch_size=batch_size * 2, shuffle=False, drop_last=False, **kw
    )
    return train_loader, val_loader, num_classes

# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------

_MASKER_POOLS = {"ctx", "tgt", "ign", "ctx+tgt", "ctx+tgt+ign"}

def make_pool_fn(pool: str, encoder: ViTTokens, masker, device: torch.device):
    """Returns a no-grad function (B,3,H,W) → (B,D)."""
    rates = torch.tensor([[LAM, ALPHA]], device=device)

    row_start = GRID_H // 3        # 4
    row_end   = 2 * GRID_H // 3   # 8
    center_idx = torch.tensor(
        [r * GRID_W + c for r in range(row_start, row_end) for c in range(GRID_W)],
        device=device, dtype=torch.long,
    )  # 48 patches

    @torch.no_grad()
    def _pool(x: torch.Tensor) -> torch.Tensor:
        tokens = encoder(x)              # (B, N, D)
        B, N, D = tokens.shape

        if pool == "center":
            return tokens[:, center_idx, :].mean(1)

        if pool == "all":
            return tokens.mean(1)

        # masker-based selection
        mask_out = masker(tokens, ema_full=None, rates=rates.expand(B, -1))
        ctx_idx  = mask_out.context_idx  # (B, nctx)
        tgt_idx  = mask_out.target_idx   # (B, ntgt)

        ctx_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        tgt_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        ctx_mask.scatter_(1, ctx_idx, True)
        tgt_mask.scatter_(1, tgt_idx, True)
        ign_mask = ~(ctx_mask | tgt_mask)

        sel = torch.zeros(B, N, dtype=torch.bool, device=device)
        if "ctx" in pool: sel |= ctx_mask
        if "tgt" in pool: sel |= tgt_mask
        if "ign" in pool: sel |= ign_mask

        n = sel.sum(1, keepdim=True).float().clamp(min=1)
        return (tokens * sel.unsqueeze(-1)).sum(1) / n

    return _pool

# ---------------------------------------------------------------------------
# Linear probe
# ---------------------------------------------------------------------------

def run_probe(
    pool_fn,
    train_loader,
    val_loader,
    num_classes: int,
    epochs: int,
    lr: float,
    device: torch.device,
    log_every: int = 20,
) -> float:
    head = nn.Linear(EMBED_DIM, num_classes).to(device)
    opt  = torch.optim.SGD(head.parameters(), lr=lr, momentum=0.9, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_acc = 0.0
    for epoch in range(epochs):
        head.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            feat = pool_fn(x)
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(head(feat), y).backward()
            opt.step()
        sched.step()

        if (epoch + 1) % log_every == 0 or epoch == epochs - 1:
            head.eval()
            correct = total = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    correct += (head(pool_fn(x)).argmax(1) == y).sum().item()
                    total   += y.size(0)
            acc = correct / total
            best_acc = max(best_acc, acc)
            print(f"    epoch {epoch+1:3d}/{epochs}: val_acc={acc:.4f}")

    return best_acc

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder-ckpt", required=True,
                        help="Checkpoint whose target_encoder.* weights are used")
    parser.add_argument("--masker-ckpt",  default=None,
                        help="Checkpoint whose latent_masker.* weights are used "
                             "(required for masker-based pools: ctx/tgt/ign/ctx+tgt)")
    parser.add_argument("--data-root",    default=os.environ.get("FAST", "/scratch") + "/datasets/")
    parser.add_argument("--dataset",      default="stl10", choices=["stl10", "food101"])
    parser.add_argument("--pools",        nargs="+", default=["ign", "center"],
                        metavar="POOL",
                        help="Pools to evaluate. Any of: ctx tgt ign ctx+tgt center all")
    parser.add_argument("--seeds",        nargs="+", type=int, default=[0, 42])
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--lr",           type=float, default=0.1)
    parser.add_argument("--batch-size",   type=int,   default=256)
    parser.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Validate: masker-ckpt required for masker-based pools
    needs_masker = any(p in _MASKER_POOLS for p in args.pools)
    if needs_masker and args.masker_ckpt is None:
        parser.error(f"--masker-ckpt required for pools: {[p for p in args.pools if p in _MASKER_POOLS]}")

    device = torch.device(args.device)

    print(f"Loading encoder from : {args.encoder_ckpt}")
    encoder = build_encoder(args.encoder_ckpt, device)

    masker = None
    if needs_masker:
        print(f"Loading masker from  : {args.masker_ckpt}")
        masker = build_masker(args.masker_ckpt, device)

    train_loader, val_loader, num_classes = build_loaders(
        args.dataset, args.data_root, args.batch_size
    )
    print(f"\nDataset : {args.dataset}  ({num_classes} classes)")
    print(f"Pools   : {args.pools}")
    print(f"Seeds   : {args.seeds}  Epochs: {args.epochs}  LR: {args.lr}\n")

    results: dict[str, list[float]] = {}
    for pool in args.pools:
        pool_fn = make_pool_fn(pool, encoder, masker, device)
        accs = []
        for seed in args.seeds:
            torch.manual_seed(seed)
            print(f"  pool={pool:<10}  seed={seed}")
            acc = run_probe(
                pool_fn, train_loader, val_loader, num_classes,
                args.epochs, args.lr, device,
            )
            accs.append(acc)
        results[pool] = accs

    print("\n=== Results ===")
    print(f"  {'pool':<12}  {'mean':>6}  {'std':>6}   per-seed")
    print("  " + "-" * 50)
    for pool, accs in results.items():
        mean = sum(accs) / len(accs)
        var  = sum((a - mean) ** 2 for a in accs) / max(len(accs) - 1, 1)
        std  = math.sqrt(var)
        vals = "  ".join(f"{a:.4f}" for a in accs)
        print(f"  {pool:<12}  {mean:.4f}  {std:.4f}   {vals}")


if __name__ == "__main__":
    main()
