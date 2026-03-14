#!/usr/bin/env python3
"""
hacky_visualize_attention.py

Visualize attention maps from the EMA (target) encoder of a trained I-JEPA model.

Produces per-image panels:
  [original | CLS attn (last layer, head avg) | CLS attn per-head grid]

Also produces aggregate heatmaps (avg across dataset, per-class).

CLS note: I-JEPA doesn't explicitly train CLS, but it participates in
self-attention across all layers, so it develops attention patterns.
They're noisier than DINO's but still informative.

Usage:
  python hacky_visualize_attention.py \
      --ckpt /path/to/last.pt \
      --experiment stl10_vits_ps8_multiblock \
      [--data-root /path/to/datasets] [--out-dir attn_viz] [--n 200] \
      [--layer -1] [--mode cls]
"""
import argparse
import math
import os
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets as tv_datasets, transforms

# ---------------------------------------------------------------------------
# Attention capture via monkey-patching
# ---------------------------------------------------------------------------

@contextmanager
def capture_attention(vit_encoder_layers, layer_indices=None):
    """
    Context manager that monkey-patches torchvision EncoderBlock.forward
    to capture attention weights from nn.MultiheadAttention.

    Yields a dict mapping layer_index -> (B, H, S, S) attention weights
    (one entry per forward pass through that layer).
    """
    attn_maps: dict = {}
    originals = {}

    num_layers = len(vit_encoder_layers)
    if layer_indices is None:
        layer_indices = list(range(num_layers))

    for idx in layer_indices:
        block = vit_encoder_layers[idx]
        orig_self_attn = block.self_attention
        originals[idx] = orig_self_attn

        # Wrap MHA to force need_weights=True
        class _AttnWrapper(torch.nn.Module):
            def __init__(self, mha, store_key):
                super().__init__()
                self._mha = mha
                self._key = store_key

            def forward(self, query, key, value, **kwargs):
                kwargs["need_weights"] = True
                kwargs["average_attn_weights"] = False
                out, weights = self._mha(query, key, value, **kwargs)
                # weights: (B, H, S, S)
                attn_maps[self._key] = weights.detach()
                return out, weights

        block.self_attention = _AttnWrapper(orig_self_attn, idx)

    try:
        yield attn_maps
    finally:
        # Restore originals
        for idx, orig in originals.items():
            vit_encoder_layers[idx].self_attention = orig


# ---------------------------------------------------------------------------
# Model loading (same pattern as hacky_visualize_goldilocks.py)
# ---------------------------------------------------------------------------

def _load_model(ckpt_path: str, experiment: str, device: torch.device):
    from hydra import compose, initialize_config_dir
    from ijepa_lite.build import build_pretrain_model

    config_dir = str(Path(__file__).resolve().parent / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(config_name="config", overrides=[f"experiment={experiment}"])

    model = build_pretrain_model(cfg).to(device)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = sd.get("model", sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [model] missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    model.requires_grad_(False)
    model.eval()

    image_size = int(cfg.model.image_size)
    patch_size = int(cfg.model.patch_size)
    num_heads = int(cfg.model.num_heads)

    return model, image_size, patch_size, num_heads


def _make_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def _denorm(img_tensor):
    """Undo ImageNet normalization for display."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


# ---------------------------------------------------------------------------
# Attention extraction
# ---------------------------------------------------------------------------

def get_attention_maps(model, images, layer_indices):
    """
    Run EMA encoder on images and capture attention weights.

    Returns dict: layer_idx -> (B, H, S, S) where S = 1 + num_patches (CLS + patches).
    """
    encoder = model.target_encoder
    vit = encoder.vit
    layers = vit.encoder.layers

    with torch.no_grad(), capture_attention(layers, layer_indices) as attn_maps:
        encoder(images)  # full image, no masking

    return attn_maps


def cls_attention_map(attn, grid_h, grid_w):
    """
    Extract CLS -> patch attention from (B, H, S, S) attention weights.

    Returns:
        head_avg: (B, grid_h, grid_w) — averaged across heads
        per_head: (B, H, grid_h, grid_w)
    """
    # attn[:, :, 0, 1:] = CLS query attending to all patch keys
    cls_attn = attn[:, :, 0, 1:]  # (B, H, N)
    b, h, n = cls_attn.shape
    per_head = cls_attn.reshape(b, h, grid_h, grid_w)
    head_avg = cls_attn.mean(dim=1).reshape(b, grid_h, grid_w)
    return head_avg, per_head


def mean_patch_attention(attn, grid_h, grid_w):
    """
    Mean attention received by each patch (from all other tokens).

    Returns: (B, grid_h, grid_w)
    """
    # Sum attention each patch key receives from all queries
    # attn: (B, H, S, S), column-wise sum over queries for patch keys
    patch_attn = attn[:, :, :, 1:]  # (B, H, S, N) — all queries -> patch keys
    received = patch_attn.sum(dim=2).mean(dim=1)  # (B, N) — avg over heads
    b, n = received.shape
    return received.reshape(b, grid_h, grid_w)


def attention_rollout(attn_maps, grid_h, grid_w):
    """
    Attention rollout: multiply attention matrices across layers (with residual).

    Returns: (B, grid_h, grid_w) — CLS row of the rolled-out attention.
    """
    sorted_layers = sorted(attn_maps.keys())
    rollout = None
    for idx in sorted_layers:
        attn = attn_maps[idx]  # (B, H, S, S)
        # Average over heads
        attn_avg = attn.mean(dim=1)  # (B, S, S)
        # Add residual connection (identity)
        eye = torch.eye(attn_avg.shape[-1], device=attn_avg.device).unsqueeze(0)
        attn_res = 0.5 * attn_avg + 0.5 * eye
        # Re-normalize rows
        attn_res = attn_res / attn_res.sum(dim=-1, keepdim=True)
        if rollout is None:
            rollout = attn_res
        else:
            rollout = torch.bmm(rollout, attn_res)

    # CLS -> patches
    cls_rollout = rollout[:, 0, 1:]  # (B, N)
    b, n = cls_rollout.shape
    return cls_rollout.reshape(b, grid_h, grid_w)


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def _heatmap_image(heatmap_np, cmap="viridis"):
    """Convert a [0,1]-normalized heatmap to an RGB image via colormap."""
    cm = plt.get_cmap(cmap)
    return cm(heatmap_np)[..., :3]  # (H, W, 3)


def save_grid(images_panels, out_path, grid_cols=10):
    """
    Save a grid of image rows.
    images_panels: list of lists, each inner list = panels for one sample.
    """
    n_samples = len(images_panels)
    n_panels = len(images_panels[0])
    n_rows = math.ceil(n_samples / grid_cols)

    fig, axes = plt.subplots(
        n_rows, grid_cols * n_panels,
        figsize=(grid_cols * n_panels * 1.2, n_rows * 1.2),
        squeeze=False,
    )
    for ax_row in axes:
        for ax in ax_row:
            ax.axis("off")

    for i, panels in enumerate(images_panels):
        row = i // grid_cols
        col_base = (i % grid_cols) * n_panels
        for j, panel in enumerate(panels):
            axes[row, col_base + j].imshow(panel)

    plt.tight_layout(pad=0.3)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def save_per_head_grid(img_np, per_head_maps, out_path, grid_h, grid_w):
    """
    Save a grid showing CLS attention for each head on a single image.
    per_head_maps: (H, grid_h, grid_w)
    """
    n_heads = per_head_maps.shape[0]
    cols = min(n_heads, 6)
    rows = math.ceil((n_heads + 1) / cols)  # +1 for original

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2), squeeze=False)
    for ax_row in axes:
        for ax in ax_row:
            ax.axis("off")

    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title("original", fontsize=8)

    for h in range(n_heads):
        idx = h + 1
        r, c = idx // cols, idx % cols
        hmap = per_head_maps[h].numpy()
        hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-8)
        axes[r, c].imshow(hmap, cmap="viridis", interpolation="nearest")
        axes[r, c].set_title(f"head {h}", fontsize=8)

    plt.tight_layout(pad=0.3)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_avg_heatmap(score_sums, count, out_path, title="Average attention"):
    avg = score_sums / max(count, 1)
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    im = ax.imshow(avg, cmap="viridis", interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def save_per_class_heatmap(score_sums_global, count_global,
                           class_score_sums, class_counts,
                           class_names, out_path, title="Per-class attention"):
    n_classes = len(class_score_sums)
    if n_classes == 0:
        return
    cols = min(n_classes, 5)
    rows = math.ceil((n_classes + 1) / cols)  # +1 for global avg

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5), squeeze=False)
    for ax_row in axes:
        for ax in ax_row:
            ax.axis("off")

    # Global average
    avg_global = score_sums_global / max(count_global, 1)
    axes[0, 0].imshow(avg_global, cmap="viridis", interpolation="nearest")
    axes[0, 0].set_title("all", fontsize=8)

    sorted_classes = sorted(class_score_sums.keys())
    for i, c in enumerate(sorted_classes):
        idx = i + 1
        r, c_idx = idx // cols, idx % cols
        avg = class_score_sums[c] / max(class_counts[c], 1)
        axes[r, c_idx].imshow(avg, cmap="viridis", interpolation="nearest")
        name = class_names[c] if c < len(class_names) else str(c)
        axes[r, c_idx].set_title(name, fontsize=7)

    fig.suptitle(title, fontsize=11)
    plt.tight_layout(pad=0.3)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_dataset(model, dataset, device, image_size, patch_size, num_heads,
                    layer_idx, mode, out_dir, dataset_name, split, n, grid_cols):
    grid_h = grid_w = image_size // patch_size
    tfm = _make_transform(image_size)

    # Accumulators for aggregate heatmaps
    score_sums = np.zeros((grid_h, grid_w), dtype=np.float64)
    class_score_sums = {}
    class_counts = {}
    total = 0
    all_panels = []

    # Resolve layer index
    num_layers = len(model.target_encoder.vit.encoder.layers)
    if layer_idx < 0:
        layer_idx = num_layers + layer_idx

    # For rollout we need all layers
    capture_layers = list(range(num_layers)) if mode == "rollout" else [layer_idx]

    batch_size = 32
    n = min(n, len(dataset))
    indices = list(range(n))

    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        batch_indices = indices[batch_start:batch_end]

        images = []
        labels = []
        raw_images = []
        for idx in batch_indices:
            img, label = dataset[idx]
            if not isinstance(img, torch.Tensor):
                raw_images.append(img)
                img = tfm(img)
            else:
                raw_images.append(None)
                img = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
                ])(img) if img.shape[-1] != image_size else img
            images.append(img)
            labels.append(label)

        batch = torch.stack(images).to(device)
        attn_maps = get_attention_maps(model, batch, capture_layers)

        for i in range(len(batch_indices)):
            img_np = _denorm(batch[i])
            label = labels[i]

            if mode == "cls":
                attn = attn_maps[layer_idx]
                head_avg, per_head = cls_attention_map(
                    attn[i:i+1], grid_h, grid_w
                )
                hmap = head_avg[0].cpu().numpy()
            elif mode == "mean":
                attn = attn_maps[layer_idx]
                hmap = mean_patch_attention(
                    attn[i:i+1], grid_h, grid_w
                )[0].cpu().numpy()
            elif mode == "rollout":
                single_maps = {k: v[i:i+1] for k, v in attn_maps.items()}
                hmap = attention_rollout(single_maps, grid_h, grid_w)[0].cpu().numpy()
            else:
                raise ValueError(f"Unknown mode: {mode}")

            # Normalize to [0, 1]
            hmap_norm = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-8)

            # Accumulate
            score_sums += hmap_norm
            total += 1
            class_score_sums[label] = class_score_sums.get(
                label, np.zeros((grid_h, grid_w), dtype=np.float64)
            ) + hmap_norm
            class_counts[label] = class_counts.get(label, 0) + 1

            hmap_rgb = _heatmap_image(hmap_norm)
            all_panels.append([img_np, hmap_rgb])

        print(f"\r  Processed {min(batch_end, n)}/{n}", end="", flush=True)

    print()

    # Save grid
    if all_panels:
        grid_path = out_dir / f"{dataset_name}_{split}_attn_{mode}.png"
        save_grid(all_panels[:min(len(all_panels), grid_cols * 20)],
                  grid_path, grid_cols=grid_cols)

    # Save per-head detail for first 4 images
    if mode == "cls" and n >= 1:
        attn_first = get_attention_maps(
            model,
            torch.stack([tfm(dataset[i][0]) if not isinstance(dataset[i][0], torch.Tensor)
                         else dataset[i][0]
                         for i in range(min(4, n))]).to(device),
            [layer_idx],
        )
        attn = attn_first[layer_idx]
        for i in range(min(4, n)):
            img_item = dataset[i][0]
            if not isinstance(img_item, torch.Tensor):
                img_np = _denorm(tfm(img_item))
            else:
                img_np = _denorm(img_item)
            _, per_head = cls_attention_map(attn[i:i+1], grid_h, grid_w)
            head_path = out_dir / f"{dataset_name}_{split}_heads_img{i}.png"
            save_per_head_grid(img_np, per_head[0].cpu(), head_path, grid_h, grid_w)

    return score_sums, class_score_sums, class_counts, total


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize EMA encoder attention maps from a trained I-JEPA model."
    )
    parser.add_argument("--ckpt", default=os.environ.get("PRETRAIN_CKPT", ""),
                        help="Path to checkpoint file")
    parser.add_argument("--experiment", required=True,
                        help="Hydra experiment config name")
    parser.add_argument("--data-root",
                        default=os.environ.get("FAST", "/scratch") + "/datasets/")
    parser.add_argument("--out-dir", default="attn_viz")
    parser.add_argument("--n", type=int, default=200,
                        help="Number of images to process per split")
    parser.add_argument("--grid-cols", type=int, default=10)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--layer", type=int, default=-1,
                        help="Transformer layer index (-1 = last layer)")
    parser.add_argument("--mode", choices=["cls", "mean", "rollout"], default="cls",
                        help="cls: CLS->patch attention, mean: avg received attention, "
                             "rollout: attention rollout across all layers")
    args = parser.parse_args()

    if not args.ckpt:
        raise SystemExit("Provide --ckpt or set PRETRAIN_CKPT env")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"Loading checkpoint: {args.ckpt}")
    print(f"Experiment config:  {args.experiment}")
    model, image_size, patch_size, num_heads = _load_model(
        args.ckpt, args.experiment, device
    )
    num_layers = len(model.target_encoder.vit.encoder.layers)
    resolved_layer = args.layer if args.layer >= 0 else num_layers + args.layer
    print(f"  image_size={image_size}  patch_size={patch_size}  "
          f"num_patches={(image_size // patch_size) ** 2}  "
          f"num_heads={num_heads}  num_layers={num_layers}  "
          f"layer={resolved_layer}  mode={args.mode}  device={device}")

    datasets_cfg = [
        ("stl10", [
            ("train", lambda r: tv_datasets.STL10(r, split="train", download=False)),
            ("test",  lambda r: tv_datasets.STL10(r, split="test",  download=False)),
        ]),
        ("food101", [
            ("train", lambda r: tv_datasets.Food101(r, split="train", download=False)),
            ("test",  lambda r: tv_datasets.Food101(r, split="test",  download=False)),
        ]),
    ]

    for name, splits in datasets_cfg:
        gh = gw = image_size // patch_size
        total_score_sums = np.zeros((gh, gw), dtype=np.float64)
        total_class_sums = {}
        total_class_counts = {}
        total_images = 0
        class_names = []

        for split, loader_fn in splits:
            print(f"\n{name}/{split}")
            try:
                ds = loader_fn(args.data_root)
            except Exception as e:
                print(f"  skipped ({e})")
                continue

            if not class_names and hasattr(ds, "classes"):
                class_names = list(ds.classes)

            sums, cls_sums, cls_counts, count = process_dataset(
                model, ds, device, image_size, patch_size, num_heads,
                args.layer, args.mode, out_dir, name, split,
                args.n, args.grid_cols,
            )
            total_score_sums += sums
            total_images += count
            for c, arr in cls_sums.items():
                total_class_sums[c] = total_class_sums.get(
                    c, np.zeros((gh, gw), dtype=np.float64)) + arr
                total_class_counts[c] = total_class_counts.get(c, 0) + cls_counts[c]

        if total_images > 0:
            save_avg_heatmap(
                total_score_sums, total_images,
                out_dir / f"{name}_avg_attn_{args.mode}.png",
                title=f"Avg {args.mode} attention (layer {resolved_layer})",
            )
            if total_class_sums:
                save_per_class_heatmap(
                    total_score_sums, total_images,
                    total_class_sums, total_class_counts,
                    class_names,
                    out_dir / f"{name}_per_class_attn_{args.mode}.png",
                    title=f"Per-class {args.mode} attention (layer {resolved_layer})",
                )

    print(f"\nDone. Output in ./{out_dir}/")
    print(f"Mode: {args.mode} | Layer: {resolved_layer}")
    if args.mode == "cls":
        print("Note: CLS is not explicitly trained in I-JEPA — patterns may be "
              "less semantic than DINO. Try --mode rollout or --mode mean for comparison.")


if __name__ == "__main__":
    main()
