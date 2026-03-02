#!/usr/bin/env python3
"""
hacky_stl10_masker_probe.py

Evaluates a pretrained IJEPA-Lite checkpoint by treating (λ, α) as two
learnable scalars.  The frozen pretrained masker is conditioned on them, and
the representation is a soft-weighted pool of encoder tokens weighted by the
masker's p_ctx.  Gradients from a linear probe's cross-entropy loss flow back
through the soft pool into p_ctx, through the frozen masker transformer, and
into log_lam / log_alpha via rates_proj.

This answers: "at which (λ, α) operating point does the pretrained masker
produce the best downstream representations?"

Baseline: mean-pool sklearn LogisticRegression (same as hacky_stl10_logreg.py)
printed at the end for direct comparison.
"""

from __future__ import annotations

import argparse
import math
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

from ijepa_lite.masking.mi_masker import MIRateMasker


# ---------------------------------------------------------------------------
# Helpers: checkpoint loading  (shared with hacky_stl10_logreg.py)
# ---------------------------------------------------------------------------

def torch_load_safely(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_state_dict(payload) -> Dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for k in ("state_dict", "model", "model_state_dict", "net", "encoder"):
            v = payload.get(k, None)
            if isinstance(v, dict) and any(isinstance(x, torch.Tensor) for x in v.values()):
                return v
        if any(isinstance(x, torch.Tensor) for x in payload.values()):
            return payload
    raise ValueError("Could not find a state_dict-like mapping in checkpoint payload.")


def strip_prefix(sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    plen = len(prefix)
    for k, v in sd.items():
        if k.startswith(prefix):
            out[k[plen:]] = v
    return out


def has_any_prefix(sd: Dict[str, torch.Tensor], prefix: str) -> bool:
    return any(k.startswith(prefix) for k in sd.keys())


def pick_encoder_prefix(sd: Dict[str, torch.Tensor], prefer: str) -> str:
    if prefer == "ema":
        order = ["ema_encoder.vit.", "target_encoder.vit.", "context_encoder.vit.", "encoder.vit.", "vit."]
    elif prefer == "target":
        order = ["target_encoder.vit.", "ema_encoder.vit.", "context_encoder.vit.", "encoder.vit.", "vit."]
    else:
        order = ["context_encoder.vit.", "target_encoder.vit.", "ema_encoder.vit.", "encoder.vit.", "vit."]

    for p in order:
        if has_any_prefix(sd, p):
            return p

    torchvision_markers = ("class_token", "conv_proj.weight", "encoder.pos_embedding")
    if any(k in sd for k in torchvision_markers):
        return ""

    raise ValueError(
        "Could not find any known encoder prefix in checkpoint keys.\n"
        "Tip: print a few keys from the checkpoint and update the prefix list."
    )


# ---------------------------------------------------------------------------
# Helpers: masker loading
# ---------------------------------------------------------------------------

def load_masker_weights(masker: MIRateMasker, sd_full: Dict[str, torch.Tensor]) -> None:
    sd_masker = strip_prefix(sd_full, "latent_masker.")
    if not sd_masker:
        raise ValueError("No 'latent_masker.*' keys in checkpoint.")
    masker.load_state_dict(sd_masker, strict=True)


# ---------------------------------------------------------------------------
# Torchvision ViT feature extraction
# ---------------------------------------------------------------------------

def vit_tokens(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Return token embeddings [B, N+1, D] from torchvision VisionTransformer."""
    x = model._process_input(images)
    n = x.shape[1]
    cls = model.class_token.expand(images.shape[0], -1, -1)
    x = torch.cat([cls, x], dim=1)
    x = x + model.encoder.pos_embedding[:, : n + 1, :]
    x = model.encoder.dropout(x)
    x = model.encoder.layers(x)
    x = model.encoder.ln(x)
    return x


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def stl10_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def build_stl10_loaders(
    data_root: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, int, int]:
    tfm = stl10_transform(image_size)
    train_ds = torchvision.datasets.STL10(root=data_root, split="train", transform=tfm, download=True)
    test_ds  = torchvision.datasets.STL10(root=data_root, split="test",  transform=tfm, download=True)
    kw = dict(num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **kw)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **kw)
    return train_loader, test_loader, len(train_ds), len(test_ds)


# ---------------------------------------------------------------------------
# Soft-pool forward
# ---------------------------------------------------------------------------

def soft_pool_forward(
    vit: nn.Module,
    masker: MIRateMasker,
    images: torch.Tensor,
    log_lam: nn.Parameter,
    log_alpha: nn.Parameter,
) -> torch.Tensor:
    """
    Gradient path:
        log_lam, log_alpha
        → rates (B, 2)
        → masker.rates_proj (frozen, but grad flows through input)
        → pos_embed_rates → transformer → logits → softmax
        → p_ctx (B, N)
        → soft-weighted pool of detached ema_tokens
        → repr (B, D)
    """
    B = images.size(0)
    device = images.device

    with torch.no_grad():
        ema_tokens = vit_tokens(vit, images)[:, 1:]   # (B, N, D)  — detached

    lam_val   = log_lam.exp().clamp(min=1e-6)         # scalar, requires_grad
    alpha_val = log_alpha.exp().clamp(min=1e-6)        # scalar, requires_grad
    rates = torch.stack([lam_val.expand(B), alpha_val.expand(B)], dim=-1)  # (B, 2)

    mask_out = masker(ema_tokens, ema_full=ema_tokens, rates=rates)
    p_ctx = mask_out.context_soft                      # (B, N)

    w = p_ctx / (p_ctx.sum(1, keepdim=True) + 1e-6)   # (B, N) normalised weights
    return (w.unsqueeze(-1) * ema_tokens).sum(1)        # (B, D)


# ---------------------------------------------------------------------------
# Baseline: mean-pool sklearn LogisticRegression
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_mean_pool_embs(
    vit: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    vit.eval()
    embs, labels = [], []
    for images, targets in tqdm(loader, desc="extract (baseline)", leave=False):
        toks = vit_tokens(vit, images.to(device))
        embs.append(toks[:, 1:].mean(dim=1).cpu())
        labels.append(targets.cpu())
    X = torch.cat(embs, dim=0).numpy()
    y = torch.cat(labels, dim=0).numpy()
    return X, y


def run_logreg_baseline(
    vit: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    max_iter: int,
    C: float,
) -> float:
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    print("\n--- Baseline: mean-pool LogisticRegression ---")
    X_train, y_train = collect_mean_pool_embs(vit, train_loader, device)
    X_test,  y_test  = collect_mean_pool_embs(vit, test_loader,  device)

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=max_iter, C=C, solver="lbfgs"),
    )
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"baseline mean-pool logreg  val_acc={acc:.4f}")
    return acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="STL-10 linear probe that optimises (λ, α) through the frozen masker."
    )
    # --- checkpoint / data ---
    ap.add_argument("--ckpt",       required=True, help="Path to last.pt")
    ap.add_argument("--data_root",  required=True, help="Root folder containing stl10_binary/")
    ap.add_argument("--prefer",     default="ema", choices=["ema", "target", "context"])

    # --- ViT architecture ---
    ap.add_argument("--image_size",  type=int,   default=96)
    ap.add_argument("--patch_size",  type=int,   default=8)
    ap.add_argument("--embed_dim",   type=int,   default=384)
    ap.add_argument("--depth",       type=int,   default=12)
    ap.add_argument("--num_heads",   type=int,   default=6)
    ap.add_argument("--mlp_ratio",   type=float, default=4.0)

    # --- masker architecture ---
    ap.add_argument("--predictor_dim",  type=int,   default=192)
    ap.add_argument("--masker_depth",   type=int,   default=2)
    ap.add_argument("--masker_heads",   type=int,   default=6)
    ap.add_argument("--lam_min",        type=float, default=0.01)
    ap.add_argument("--lam_max",        type=float, default=1.0)
    ap.add_argument("--alpha_min",      type=float, default=0.01)
    ap.add_argument("--alpha_max",      type=float, default=0.5)
    ap.add_argument("--ntgt_min",       type=int,   default=4)
    ap.add_argument("--nctx_min",       type=int,   default=1)
    ap.add_argument("--h_floor",        type=float, default=0.1)
    ap.add_argument("--floor_weight",   type=float, default=5.0)

    # --- training ---
    ap.add_argument("--lr",         type=float, default=1e-3,  help="LR for linear head")
    ap.add_argument("--rates_lr",   type=float, default=1e-2,  help="LR for log_lam / log_alpha")
    ap.add_argument("--epochs",     type=int,   default=20)
    ap.add_argument("--batch_size", type=int,   default=256)
    ap.add_argument("--num_workers",type=int,   default=4)

    # --- baseline ---
    ap.add_argument("--max_iter",   type=int,   default=2000)
    ap.add_argument("--C",          type=float, default=1.0)

    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    train_loader, test_loader, n_train, n_test = build_stl10_loaders(
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"train={n_train}  test={n_test}")

    num_patches = (args.image_size // args.patch_size) ** 2
    num_classes = 10
    print(f"num_patches={num_patches}")

    # -----------------------------------------------------------------------
    # Checkpoint
    # -----------------------------------------------------------------------
    payload = torch_load_safely(args.ckpt)
    sd_full  = extract_state_dict(payload)

    # -----------------------------------------------------------------------
    # Encoder (frozen)
    # -----------------------------------------------------------------------
    mlp_dim = int(args.embed_dim * args.mlp_ratio)
    vit = torchvision.models.vision_transformer.VisionTransformer(
        image_size=args.image_size,
        patch_size=args.patch_size,
        num_layers=args.depth,
        num_heads=args.num_heads,
        hidden_dim=args.embed_dim,
        mlp_dim=mlp_dim,
        num_classes=1000,
        dropout=0.0,
        attention_dropout=0.0,
    )
    prefix = pick_encoder_prefix(sd_full, args.prefer)
    sd_vit = strip_prefix(sd_full, prefix) if prefix else sd_full
    missing, unexpected = vit.load_state_dict(sd_vit, strict=False)
    print(f"Loaded encoder prefix='{prefix}'  missing={len(missing)}  unexpected={len(unexpected)}")
    if missing:
        print("  missing (first 10):", missing[:10])

    vit.eval().to(device)
    for p in vit.parameters():
        p.requires_grad_(False)

    # -----------------------------------------------------------------------
    # Masker (frozen weights, but computation graph is live)
    # -----------------------------------------------------------------------
    masker = MIRateMasker(
        dim=args.embed_dim,
        predictor_dim=args.predictor_dim,
        depth=args.masker_depth,
        num_heads=args.masker_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=0.0,
        num_patches=num_patches,
        lam_min=args.lam_min,
        lam_max=args.lam_max,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        ntgt_min=args.ntgt_min,
        nctx_min=args.nctx_min,
        h_floor=args.h_floor,
        floor_weight=args.floor_weight,
        lam_warmup_epochs=0,
    )
    load_masker_weights(masker, sd_full)
    masker.eval().to(device)
    for p in masker.parameters():
        p.requires_grad_(False)

    # -----------------------------------------------------------------------
    # Learnable rates + linear head
    # -----------------------------------------------------------------------
    lam_init   = math.exp(0.5 * (math.log(args.lam_min)   + math.log(args.lam_max)))
    alpha_init = math.exp(0.5 * (math.log(args.alpha_min) + math.log(args.alpha_max)))
    log_lam   = nn.Parameter(torch.tensor(math.log(lam_init),   device=device))
    log_alpha = nn.Parameter(torch.tensor(math.log(alpha_init), device=device))

    head = nn.Linear(args.embed_dim, num_classes).to(device)

    optimizer = torch.optim.AdamW([
        {"params": head.parameters(),    "lr": args.lr},
        {"params": [log_lam, log_alpha], "lr": args.rates_lr},
    ])

    print(f"\nInitial  λ={log_lam.exp().item():.4f}  α={log_alpha.exp().item():.4f}")
    print(f"Training for {args.epochs} epochs...\n")

    # -----------------------------------------------------------------------
    # Training + eval loop
    # -----------------------------------------------------------------------
    for epoch in range(args.epochs):
        # --- train ---
        masker.train()
        head.train()
        running_loss = 0.0
        n_batches = 0
        for images, labels in tqdm(train_loader, desc=f"epoch {epoch:3d} train", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            repr_vec = soft_pool_forward(vit, masker, images, log_lam, log_alpha)
            loss = F.cross_entropy(head(repr_vec), labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        avg_loss = running_loss / max(n_batches, 1)

        # --- eval ---
        masker.eval()
        head.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc=f"epoch {epoch:3d} eval", leave=False):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                repr_vec = soft_pool_forward(vit, masker, images, log_lam, log_alpha)
                correct += (head(repr_vec).argmax(1) == labels).sum().item()
                total   += labels.size(0)

        val_acc = correct / total
        print(
            f"epoch {epoch:3d}  "
            f"λ={log_lam.exp().item():.4f}  "
            f"α={log_alpha.exp().item():.4f}  "
            f"loss={avg_loss:.4f}  "
            f"val_acc={val_acc:.4f}"
        )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    final_lam   = log_lam.exp().item()
    final_alpha = log_alpha.exp().item()
    print(f"\n=== Masker-probe results ===")
    print(f"  Optimised λ={final_lam:.4f}  α={final_alpha:.4f}")
    print(f"  Final val_acc={val_acc:.4f}")

    # -----------------------------------------------------------------------
    # Baseline: mean-pool LogisticRegression
    # -----------------------------------------------------------------------
    baseline_acc = run_logreg_baseline(
        vit, train_loader, test_loader, device,
        max_iter=args.max_iter, C=args.C,
    )

    print(f"\n=== Comparison ===")
    print(f"  Masker soft-pool probe : val_acc={val_acc:.4f}  (λ={final_lam:.4f}, α={final_alpha:.4f})")
    print(f"  Mean-pool logreg       : val_acc={baseline_acc:.4f}")


if __name__ == "__main__":
    main()
