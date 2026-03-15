#!/usr/bin/env python3
"""
hacky_visualize_attention_comparison.py

Compare attention maps from our trained EMA encoder vs the original I-JEPA
ViT-H/14 (300ep, ImageNet-1K) on the same STL-10 images.

Each model runs at its native resolution (ours: 96×96, original: 224×224).
Modes: mean and rollout (CLS only for ours — original I-JEPA has no CLS token).

Produces per-mode:
  <out_dir>/<mode>/comparison_grid.png   — [image | ours | original]
  <out_dir>/<mode>/ours_avg.png          — aggregate heatmap (ours)
  <out_dir>/<mode>/original_avg.png      — aggregate heatmap (original)

Usage:
  python hacky_visualize_attention_comparison.py \
      --ckpt-ours /path/to/our/last.pt \
      --ckpt-orig /path/to/ijepa_vith14_ep300.pth.tar \
      --experiment stl10_vits_ps8_multiblock \
      [--data-root /path/to/datasets] [--out-dir attn_comparison] [--n 100]
"""
import argparse
import math
import os
import sys
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets as tv_datasets, transforms

# ---------------------------------------------------------------------------
# Attention capture: our model (torchvision ViT, monkey-patch MHA)
# ---------------------------------------------------------------------------

@contextmanager
def capture_attention_torchvision(vit_encoder_layers, layer_indices=None):
    """Monkey-patch torchvision EncoderBlock to capture attention weights."""
    attn_maps: dict = {}
    originals = {}

    num_layers = len(vit_encoder_layers)
    if layer_indices is None:
        layer_indices = list(range(num_layers))

    for idx in layer_indices:
        block = vit_encoder_layers[idx]
        orig_self_attn = block.self_attention
        originals[idx] = orig_self_attn

        class _AttnWrapper(nn.Module):
            def __init__(self, mha, store_key):
                super().__init__()
                self._mha = mha
                self._key = store_key

            def forward(self, query, key, value, **kwargs):
                kwargs["need_weights"] = True
                kwargs["average_attn_weights"] = False
                out, weights = self._mha(query, key, value, **kwargs)
                attn_maps[self._key] = weights.detach()
                return out, weights

        block.self_attention = _AttnWrapper(orig_self_attn, idx)

    try:
        yield attn_maps
    finally:
        for idx, orig in originals.items():
            vit_encoder_layers[idx].self_attention = orig


# ---------------------------------------------------------------------------
# Attention capture: original I-JEPA (custom ViT, Attention returns (x, attn))
# ---------------------------------------------------------------------------

@contextmanager
def capture_attention_ijepa(blocks, layer_indices=None):
    """Hook into original I-JEPA Block.attn to capture attention weights."""
    attn_maps: dict = {}
    handles = []

    num_layers = len(blocks)
    if layer_indices is None:
        layer_indices = list(range(num_layers))

    for idx in layer_indices:
        def hook_fn(store_key):
            def hook(module, input, output):
                # Attention.forward returns (x, attn)
                _, attn = output
                attn_maps[store_key] = attn.detach()
            return hook
        handle = blocks[idx].attn.register_forward_hook(hook_fn(idx))
        handles.append(handle)

    try:
        yield attn_maps
    finally:
        for h in handles:
            h.remove()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_our_model(ckpt_path: str, experiment: str, device: torch.device):
    from hydra import compose, initialize_config_dir
    from ijepa_lite.build import build_pretrain_model

    config_dir = str(Path(__file__).resolve().parent / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(config_name="config", overrides=[f"experiment={experiment}"])

    model = build_pretrain_model(cfg).to(device)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = sd.get("model", sd)
    missing, _ = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [ours] missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    model.requires_grad_(False)
    model.eval()

    image_size = int(cfg.model.image_size)
    patch_size = int(cfg.model.patch_size)
    return model, image_size, patch_size


def _import_ijepa_vit():
    """Import vit_huge from the original I-JEPA repo without requiring package install."""
    import importlib.util

    ijepa_root = Path(__file__).resolve().parent / "ijepa"

    # We need to load the dependency modules first (src.utils.tensors, src.masks.utils)
    # because vision_transformer.py imports them at module level.
    # Load them as fake `src.*` packages so the imports resolve.
    def _load_module(mod_name, file_path):
        spec = importlib.util.spec_from_file_location(mod_name, file_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        return mod

    # Create the package stubs so nested imports work
    import types
    for pkg in ["src", "src.utils", "src.masks", "src.models"]:
        if pkg not in sys.modules:
            sys.modules[pkg] = types.ModuleType(pkg)

    _load_module("src.utils.tensors",
                 str(ijepa_root / "src" / "utils" / "tensors.py"))
    _load_module("src.masks.utils",
                 str(ijepa_root / "src" / "masks" / "utils.py"))
    vit_mod = _load_module("src.models.vision_transformer",
                           str(ijepa_root / "src" / "models" / "vision_transformer.py"))
    return vit_mod.vit_huge


def _load_original_ijepa(ckpt_path: str, device: torch.device):
    """Load original I-JEPA ViT-H/14 target encoder."""
    vit_huge = _import_ijepa_vit()

    model = vit_huge(patch_size=14)
    model.to(device)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = ckpt.get("target_encoder", ckpt)

    # Strip "module." prefix if present (DDP)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    # Strip "backbone." prefix if present
    sd = {k.replace("backbone.", ""): v for k, v in sd.items()}

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [original] missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  [original] unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    model.requires_grad_(False)
    model.eval()

    image_size = 224
    patch_size = 14
    return model, image_size, patch_size


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def _make_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def _denorm(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


# ---------------------------------------------------------------------------
# Attention extraction
# ---------------------------------------------------------------------------

def get_attn_our_model(model, images, layer_indices):
    """Returns dict: layer_idx -> (B, H, S, S), S = 1 + N (CLS + patches)."""
    layers = model.target_encoder.vit.encoder.layers
    with torch.no_grad(), capture_attention_torchvision(layers, layer_indices) as attn_maps:
        model.target_encoder(images)
    return attn_maps


def get_attn_original(model, images, layer_indices):
    """Returns dict: layer_idx -> (B, H, N, N), N = num_patches (no CLS)."""
    with torch.no_grad(), capture_attention_ijepa(model.blocks, layer_indices) as attn_maps:
        model(images)
    return attn_maps


# ---------------------------------------------------------------------------
# Attention map computation
# ---------------------------------------------------------------------------

def mean_attention(attn, grid_h, grid_w, has_cls=False):
    """Mean attention received by each patch from all queries."""
    if has_cls:
        patch_attn = attn[:, :, :, 1:]  # (B, H, S, N) — queries -> patch keys
    else:
        patch_attn = attn  # (B, H, N, N)
    received = patch_attn.sum(dim=2).mean(dim=1)  # (B, N)
    b, n = received.shape
    return received.reshape(b, grid_h, grid_w)


def attention_rollout(attn_maps, grid_h, grid_w, has_cls=False):
    """Attention rollout across layers."""
    sorted_layers = sorted(attn_maps.keys())
    rollout = None
    for idx in sorted_layers:
        attn = attn_maps[idx]  # (B, H, S, S) or (B, H, N, N)
        attn_avg = attn.mean(dim=1)  # (B, S, S)
        eye = torch.eye(attn_avg.shape[-1], device=attn_avg.device).unsqueeze(0)
        attn_res = 0.5 * attn_avg + 0.5 * eye
        attn_res = attn_res / attn_res.sum(dim=-1, keepdim=True)
        if rollout is None:
            rollout = attn_res
        else:
            rollout = torch.bmm(rollout, attn_res)

    if has_cls:
        # CLS -> patches
        result = rollout[:, 0, 1:]
    else:
        # No CLS: average across all query positions
        result = rollout.mean(dim=1)  # (B, N)

    b, n = result.shape
    return result.reshape(b, grid_h, grid_w)


def compute_heatmap(attn_maps, mode, grid_h, grid_w, has_cls, layer_idx):
    """Compute a single (B, grid_h, grid_w) heatmap from attention maps."""
    if mode == "mean":
        return mean_attention(attn_maps[layer_idx], grid_h, grid_w, has_cls)
    elif mode == "rollout":
        return attention_rollout(attn_maps, grid_h, grid_w, has_cls)
    else:
        raise ValueError(f"Unknown mode: {mode}")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _heatmap_image(heatmap_np, cmap="viridis"):
    cm = plt.get_cmap(cmap)
    return cm(heatmap_np)[..., :3]


def save_comparison_grid(panels_list, out_path, grid_cols=10):
    """
    panels_list: list of (img_np, hmap_ours_np, hmap_orig_np) tuples.
    Each is an RGB array.
    """
    n = len(panels_list)
    n_panels = 3  # [image | ours | original]
    n_rows = math.ceil(n / grid_cols)

    fig, axes = plt.subplots(
        n_rows, grid_cols * n_panels,
        figsize=(grid_cols * n_panels * 1.3, n_rows * 1.3),
        squeeze=False,
    )
    for ax_row in axes:
        for ax in ax_row:
            ax.axis("off")

    # Column headers on first row
    for col in range(grid_cols):
        base = col * n_panels
        if col == 0:
            axes[0, base].set_title("image", fontsize=7, pad=2)
            axes[0, base + 1].set_title("ours", fontsize=7, pad=2)
            axes[0, base + 2].set_title("I-JEPA", fontsize=7, pad=2)

    for i, (img, hm_ours, hm_orig) in enumerate(panels_list):
        row = i // grid_cols
        base = (i % grid_cols) * n_panels
        axes[row, base].imshow(img)
        axes[row, base + 1].imshow(hm_ours)
        axes[row, base + 2].imshow(hm_orig)

    plt.tight_layout(pad=0.2)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


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


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process(our_model, orig_model, dataset, device,
            our_img_size, our_patch_size, orig_img_size, orig_patch_size,
            mode, out_dir, n, grid_cols):
    our_gh = our_gw = our_img_size // our_patch_size      # 12×12
    orig_gh = orig_gw = orig_img_size // orig_patch_size   # 16×16

    tfm_ours = _make_transform(our_img_size)
    tfm_orig = _make_transform(orig_img_size)

    our_num_layers = len(our_model.target_encoder.vit.encoder.layers)
    orig_num_layers = len(orig_model.blocks)

    # For rollout: all layers; for mean: last layer only
    our_layer = our_num_layers - 1
    orig_layer = orig_num_layers - 1
    our_capture = list(range(our_num_layers)) if mode == "rollout" else [our_layer]
    orig_capture = list(range(orig_num_layers)) if mode == "rollout" else [orig_layer]

    n = min(n, len(dataset))
    batch_size = 16  # ViT-H is large

    # Accumulators
    our_sums = np.zeros((our_gh, our_gw), dtype=np.float64)
    orig_sums = np.zeros((orig_gh, orig_gw), dtype=np.float64)
    total = 0
    all_panels = []

    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)

        imgs_ours = []
        imgs_orig = []
        display_imgs = []
        for idx in range(batch_start, batch_end):
            img, _ = dataset[idx]
            imgs_ours.append(tfm_ours(img))
            imgs_orig.append(tfm_orig(img))

        batch_ours = torch.stack(imgs_ours).to(device)
        batch_orig = torch.stack(imgs_orig).to(device)

        attn_ours = get_attn_our_model(our_model, batch_ours, our_capture)
        attn_orig = get_attn_original(orig_model, batch_orig, orig_capture)

        bs = batch_ours.shape[0]
        hmaps_ours = compute_heatmap(attn_ours, mode, our_gh, our_gw,
                                     has_cls=True, layer_idx=our_layer)
        hmaps_orig = compute_heatmap(attn_orig, mode, orig_gh, orig_gw,
                                     has_cls=False, layer_idx=orig_layer)

        for i in range(bs):
            img_np = _denorm(batch_ours[i])

            hm_ours = hmaps_ours[i].cpu().numpy()
            hm_orig = hmaps_orig[i].cpu().numpy()

            hm_ours_norm = (hm_ours - hm_ours.min()) / (hm_ours.max() - hm_ours.min() + 1e-8)
            hm_orig_norm = (hm_orig - hm_orig.min()) / (hm_orig.max() - hm_orig.min() + 1e-8)

            our_sums += hm_ours_norm
            orig_sums += hm_orig_norm
            total += 1

            all_panels.append((
                img_np,
                _heatmap_image(hm_ours_norm),
                _heatmap_image(hm_orig_norm),
            ))

        print(f"\r  Processed {min(batch_end, n)}/{n}", end="", flush=True)

    print()

    # Save comparison grid
    if all_panels:
        save_comparison_grid(
            all_panels[:min(len(all_panels), grid_cols * 20)],
            out_dir / f"comparison_{mode}.png",
            grid_cols=grid_cols,
        )

    # Save per-model average heatmaps
    if total > 0:
        save_avg_heatmap(our_sums, total,
                         out_dir / f"ours_avg_{mode}.png",
                         title=f"Ours avg {mode} (layer {our_layer})")
        save_avg_heatmap(orig_sums, total,
                         out_dir / f"original_avg_{mode}.png",
                         title=f"I-JEPA ViT-H avg {mode} (layer {orig_layer})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare attention maps: our EMA encoder vs original I-JEPA ViT-H/14."
    )
    parser.add_argument("--ckpt-ours", required=True,
                        help="Our model checkpoint")
    parser.add_argument("--ckpt-orig", required=True,
                        help="Original I-JEPA ViT-H/14 checkpoint (.pth.tar)")
    parser.add_argument("--experiment", required=True,
                        help="Hydra experiment config name for our model")
    parser.add_argument("--data-root",
                        default=os.environ.get("FAST", "/scratch") + "/datasets/")
    parser.add_argument("--out-dir", default="attn_comparison")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--grid-cols", type=int, default=10)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Load models
    print(f"Loading our model: {args.ckpt_ours}")
    our_model, our_img_size, our_patch_size = _load_our_model(
        args.ckpt_ours, args.experiment, device
    )
    our_grid = our_img_size // our_patch_size
    print(f"  image_size={our_img_size}  patch_size={our_patch_size}  "
          f"grid={our_grid}x{our_grid}  "
          f"layers={len(our_model.target_encoder.vit.encoder.layers)}")

    print(f"Loading original I-JEPA: {args.ckpt_orig}")
    orig_model, orig_img_size, orig_patch_size = _load_original_ijepa(
        args.ckpt_orig, device
    )
    orig_grid = orig_img_size // orig_patch_size
    print(f"  image_size={orig_img_size}  patch_size={orig_patch_size}  "
          f"grid={orig_grid}x{orig_grid}  "
          f"layers={len(orig_model.blocks)}")

    # Load STL-10
    print(f"\nLoading STL-10 from {args.data_root}")
    try:
        ds = tv_datasets.STL10(args.data_root, split="test", download=False)
    except Exception as e:
        raise SystemExit(f"Failed to load STL-10: {e}")

    # Run all modes
    for mode in ["mean", "rollout"]:
        mode_dir = out_root / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"  Mode: {mode}")
        print(f"  Output: {mode_dir}")
        print(f"{'='*60}")

        process(
            our_model, orig_model, ds, device,
            our_img_size, our_patch_size,
            orig_img_size, orig_patch_size,
            mode, mode_dir, args.n, args.grid_cols,
        )

    print(f"\nDone. Output in ./{out_root}/  (subdirs: mean, rollout)")


if __name__ == "__main__":
    main()
