#!/usr/bin/env python3
"""
hacky_stl10_logreg.py

Quick-and-dirty: load an IJEPA-Lite checkpoint (last.pt), extract ViT features on STL10,
then train a scikit-learn LogisticRegression on top.
"""

from __future__ import annotations

import argparse
from typing import Dict, Tuple

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm


# ----------------------------
# Helpers: checkpoint loading
# ----------------------------

def torch_load_safely(path: str):
    """
    Prefer weights_only=True to avoid torch's pickle warning.
    Falls back if older torch doesn't support weights_only.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_state_dict(payload) -> Dict[str, torch.Tensor]:
    """
    Try common checkpoint layouts.
    """
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
    """
    Choose which encoder weights to use.
    """
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


# -----------------------------------
# Torchvision ViT feature extraction
# -----------------------------------

@torch.no_grad()
def vit_tokens(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    """
    Return token embeddings [B, N+1, D] from torchvision VisionTransformer,
    without using the classifier head.
    """
    x = model._process_input(images)
    n = x.shape[1]

    cls = model.class_token.expand(images.shape[0], -1, -1)
    x = torch.cat([cls, x], dim=1)

    x = x + model.encoder.pos_embedding[:, : n + 1, :]
    x = model.encoder.dropout(x)

    x = model.encoder.layers(x)
    x = model.encoder.ln(x)
    return x


@torch.no_grad()
def get_embs_labels(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    pool: str = "mean",
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    embs = []
    labels = []

    for images, targets in tqdm(loader, desc="extract", leave=False):
        images = images.to(device, non_blocking=True)
        toks = vit_tokens(model, images)  # [B, N+1, D]

        if pool == "cls":
            feat = toks[:, 0]  # [B, D]
        else:
            # mean over PATCH tokens only (exclude CLS)
            feat = toks[:, 1:].mean(dim=1)  # [B, D]

        embs.append(feat.cpu())
        labels.append(targets.cpu())

    X = torch.cat(embs, dim=0).numpy()
    y = torch.cat(labels, dim=0).numpy()
    return X, y


# ----------------------------
# Data
# ----------------------------

def stl10_transform(image_size: int) -> transforms.Compose:
    # Normalization must match the pretrain transforms so the encoder's features
    # are on the same input distribution as during pretraining.
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

    train_ds = torchvision.datasets.STL10(
        root=data_root,
        split="train",
        transform=tfm,
        download=True,
    )
    test_ds = torchvision.datasets.STL10(
        root=data_root,
        split="test",
        transform=tfm,
        download=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    return train_loader, test_loader, len(train_ds), len(test_ds)


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Path to last.pt")
    ap.add_argument("--data_root", type=str, required=True, help="Root folder that contains stl10_binary/")
    ap.add_argument("--prefer", type=str, default="ema", choices=["ema", "target", "context"])
    ap.add_argument("--pool", type=str, default="mean", choices=["mean", "cls"])

    ap.add_argument("--image_size", type=int, default=96)
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--embed_dim", type=int, default=384)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--num_heads", type=int, default=6)
    ap.add_argument("--mlp_ratio", type=float, default=4.0)

    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--max_iter", type=int, default=2000)
    ap.add_argument("--C", type=float, default=1.0)

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader, n_train, n_test = build_stl10_loaders(
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"train={n_train} test={n_test}")

    mlp_dim = int(args.embed_dim * args.mlp_ratio)
    model = torchvision.models.vision_transformer.VisionTransformer(
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

    payload = torch_load_safely(args.ckpt)
    sd_full = extract_state_dict(payload)

    prefix = pick_encoder_prefix(sd_full, args.prefer)
    if prefix == "":
        sd_vit = sd_full
        print("No prefix stripping needed (state_dict already looks like torchvision ViT).")
    else:
        sd_vit = strip_prefix(sd_full, prefix)
        print(f"Using encoder prefix='{prefix}' (stripped)")

    missing, unexpected = model.load_state_dict(sd_vit, strict=False)

    print(f"Loaded ckpt={args.ckpt}")
    print(f"missing_keys={len(missing)} unexpected_keys={len(unexpected)}")
    if missing:
        print("missing (first 20):", missing[:20])
    if unexpected:
        print("unexpected (first 20):", unexpected[:20])

    model = model.to(device).eval()

    X_train, y_train = get_embs_labels(model, train_loader, device, pool=args.pool)
    X_test, y_test = get_embs_labels(model, test_loader, device, pool=args.pool)

    print("X_train", X_train.shape, "y_train", y_train.shape)
    print("X_test ", X_test.shape, "y_test ", y_test.shape)

    if X_train.shape[1] == 0:
        raise RuntimeError("Extracted 0-dim features. Something is still wrong with feature extraction.")

    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=args.max_iter,
            C=args.C,
            solver="lbfgs",
        ),
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification report (test):\n", classification_report(y_test, y_pred))

    y_pred_train = clf.predict(X_train)
    print("Classification report (train):\n", classification_report(y_train, y_pred_train))


if __name__ == "__main__":
    main()
