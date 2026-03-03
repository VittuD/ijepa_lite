#!/usr/bin/env python3
"""
hacky_visualize_ign_patches.py

Saves side-by-side grids of [original | ign-only | color-coded] for 500 train
+ 500 test images from STL-10 and Food101.

Color coding:
  ctx = blue  (locally predictable patches)
  tgt = red   (worth predicting patches)
  ign = green (ignored — the interesting ones)

In the ign-only panel, ctx+tgt patches are replaced with grey.

Also saves per-dataset average bin-assignment heatmaps (12x12 patch grid,
averaged over all N images) to check for positional bias:
  ign_viz/stl10_avg_bins.png
  ign_viz/food101_avg_bins.png

Output (4 grid PNGs + 2 heatmap PNGs):
  ign_viz/stl10_train.png
  ign_viz/stl10_test.png
  ign_viz/food101_train.png
  ign_viz/food101_test.png
  ign_viz/stl10_avg_bins.png
  ign_viz/food101_avg_bins.png

Each grid is 25 columns x 20 rows = 500 images.
Each cell is a 3-panel strip: [original | ign-only | color-coded] = 288x96 px.

Usage:
  PRETRAIN_CKPT=/path/to/last.pt python hacky_visualize_ign_patches.py \\
      [--data-root /path/to/datasets] [--out-dir ign_viz] [--n 500] [--device cuda]
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import datasets as tv_datasets
from torchvision import transforms
from torchvision.models.vision_transformer import VisionTransformer

import ijepa_lite.masking.mi_masker  # noqa: F401 — populates registry
from ijepa_lite.masking.mi_masker import MIRateMasker
from ijepa_lite.models.vit_tokens import ViTTokens, _remove_classifier_head

# ---------------------------------------------------------------------------
# Architecture constants (stl10_masker_hard_probe_sweep experiment)
# ---------------------------------------------------------------------------
IMG_SIZE   = 96
PATCH_SIZE = 8
EMBED_DIM  = 384
DEPTH      = 12
NUM_HEADS  = 6
PRED_DIM   = 192
PRED_DEPTH = 2
PRED_HEADS = 6

# Fixed masker operating point used in the ablation
LAM   = 1.0
ALPHA = 0.01

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

GRID_COLS      = 25
BATCH_SIZE     = 32
ALPHA_OVERLAY  = 0.55   # colour blend weight for the colored panel

# Patch-bin overlay colours (RGB)
CTX_RGB = (70,  130, 180)   # steel blue
TGT_RGB = (220,  60,  60)   # red
IGN_RGB = ( 60, 200,  60)   # green
GREY    = (128, 128, 128)


# ---------------------------------------------------------------------------
# Model builders (no Hydra — construct directly from known params)
# ---------------------------------------------------------------------------

def _build_encoder(sd_full: dict, device: torch.device) -> ViTTokens:
    vit = VisionTransformer(
        image_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        num_layers=DEPTH,
        num_heads=NUM_HEADS,
        hidden_dim=EMBED_DIM,
        mlp_dim=EMBED_DIM * 4,
        num_classes=1000,
    )
    _remove_classifier_head(vit)
    enc = ViTTokens(vit)

    prefix = "target_encoder."
    enc_sd = {k[len(prefix):]: v for k, v in sd_full.items() if k.startswith(prefix)}
    missing, unexpected = enc.load_state_dict(enc_sd, strict=False)
    if missing:
        print(f"  [encoder] missing keys: {missing[:5]}{'...' if len(missing)>5 else ''}")

    enc.requires_grad_(False)
    return enc.to(device).eval()


def _build_masker(sd_full: dict, device: torch.device) -> MIRateMasker:
    num_patches = (IMG_SIZE // PATCH_SIZE) ** 2
    masker = MIRateMasker(
        dim=EMBED_DIM,
        predictor_dim=PRED_DIM,
        depth=PRED_DEPTH,
        num_heads=PRED_HEADS,
        mlp_ratio=4.0,
        dropout=0.0,
        num_patches=num_patches,
        lam_min=0.01,
        lam_max=1.0,
        alpha_min=0.01,
        alpha_max=0.5,
        coupled_scalarization=True,
        ratio_logit_std=1.0,
        h_floor=0.1,
        floor_weight=5.0,
        lam_warmup_epochs=0,
    )

    prefix = "latent_masker."
    known_optional = {"_ema_mi_rate", "_ema_surprise"}
    m_sd = {k[len(prefix):]: v for k, v in sd_full.items() if k.startswith(prefix)}
    missing, _ = masker.load_state_dict(m_sd, strict=False)
    real_missing = [k for k in missing if k not in known_optional]
    if real_missing:
        print(f"  [masker] missing keys: {real_missing[:5]}{'...' if len(real_missing)>5 else ''}")

    masker.requires_grad_(False)
    return masker.to(device).eval()


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

_to_tensor_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

_resize_crop = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(IMG_SIZE),
])


def _pil_to_display_and_tensor(img_raw):
    """
    Accept either:
      - PIL Image (Food101)
      - numpy (C, H, W) uint8 (STL-10 default)
    Returns:
      display_np : (H, W, 3) uint8, already at IMG_SIZE x IMG_SIZE
      tensor     : (3, H, W) float32, normalized
    """
    if isinstance(img_raw, np.ndarray):
        # STL-10: (C, H, W) → (H, W, C) PIL
        if img_raw.ndim == 3 and img_raw.shape[0] == 3:
            img_raw = np.transpose(img_raw, (1, 2, 0))
        pil = Image.fromarray(img_raw.astype(np.uint8))
    else:
        pil = img_raw

    pil_rgb  = pil.convert("RGB")
    pil_crop = _resize_crop(pil_rgb)                 # (96, 96, 3) PIL
    display  = np.array(pil_crop, dtype=np.uint8)    # (H, W, 3)
    tensor   = _to_tensor_norm(pil_crop)             # (3, H, W)
    return display, tensor


def _apply_bin_overlay(
    orig_np: np.ndarray,   # (H, W, 3) uint8
    bin_map: np.ndarray,   # (grid_h, grid_w) int: 0=ctx, 1=tgt, 2=ign
    mode: str,             # "ign_only" | "colored"
) -> np.ndarray:
    """Apply per-patch overlay to orig_np according to bin_map."""
    out = orig_np.copy()
    P = PATCH_SIZE
    gh, gw = bin_map.shape
    colors = [CTX_RGB, TGT_RGB, IGN_RGB]

    for i in range(gh):
        for j in range(gw):
            b = bin_map[i, j]
            y0, y1 = i * P, (i + 1) * P
            x0, x1 = j * P, (j + 1) * P

            if mode == "ign_only":
                if b != 2:   # ctx or tgt → grey
                    out[y0:y1, x0:x1] = GREY
            elif mode == "colored":
                col = np.array(colors[b], dtype=float)
                orig_patch = orig_np[y0:y1, x0:x1].astype(float)
                blended = (1.0 - ALPHA_OVERLAY) * orig_patch + ALPHA_OVERLAY * col
                out[y0:y1, x0:x1] = blended.clip(0, 255).astype(np.uint8)

    return out


def _make_cell(orig_np, ign_np, col_np) -> Image.Image:
    """3-panel horizontal strip: [original | ign-only | color-coded]."""
    h, w = orig_np.shape[:2]
    strip = Image.new("RGB", (3 * w, h))
    strip.paste(Image.fromarray(orig_np), (0,     0))
    strip.paste(Image.fromarray(ign_np),  (w,     0))
    strip.paste(Image.fromarray(col_np),  (2 * w, 0))
    return strip


def save_avg_bin_heatmap(
    bin_counts: np.ndarray,   # (3, gh, gw) float — accumulated counts per bin
    n_images: int,
    out_path: Path,
    patch_px: int = 40,       # display pixels per patch cell
) -> None:
    """
    Save a heatmap showing the average bin assignment per patch position.

    Each of the 3 panels (ctx / tgt / ign) shows a 12x12 grid where
    brightness = fraction of images that assigned that position to that bin.
    A perfectly positional masker produces near-binary (white/black) panels.
    """
    gh, gw = bin_counts.shape[1], bin_counts.shape[2]
    frac   = bin_counts / max(n_images, 1)   # (3, gh, gw) in [0, 1]

    bin_names  = ["ctx (blue)", "tgt (red)", "ign (green)"]
    bin_colors = [CTX_RGB, TGT_RGB, IGN_RGB]

    cell_w = gw * patch_px
    cell_h = gh * patch_px
    label_h = 24
    panel_w = cell_w
    panel_h = cell_h + label_h

    img = Image.new("RGB", (3 * panel_w, panel_h), (20, 20, 20))

    try:
        from PIL import ImageFont
        font = ImageFont.load_default()
    except Exception:
        font = None

    for b, (name, color) in enumerate(zip(bin_names, bin_colors)):
        panel = Image.new("RGB", (panel_w, panel_h), (20, 20, 20))
        draw  = ImageDraw.Draw(panel)

        # Draw label
        draw.text((4, 4), name, fill=color, font=font)

        # Draw heatmap: each patch cell coloured by its average assignment freq
        for i in range(gh):
            for j in range(gw):
                v   = float(frac[b, i, j])          # 0..1
                col = tuple(int(c * v) for c in color)
                x0, y0 = j * patch_px, label_h + i * patch_px
                x1, y1 = x0 + patch_px, y0 + patch_px
                draw.rectangle([x0, y0, x1, y1], fill=col)
                # grid lines
                draw.rectangle([x0, y0, x1, y1], outline=(40, 40, 40))

        img.paste(panel, (b * panel_w, 0))

    img.save(out_path)
    print(f"  Saved heatmap → {out_path}  (n={n_images})")


# ---------------------------------------------------------------------------
# Main visualization loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def visualize_split(
    dataset_name: str,
    split: str,
    dataset,
    encoder: ViTTokens,
    masker: MIRateMasker,
    device: torch.device,
    out_dir: Path,
    n: int,
    grid_cols: int,
) -> np.ndarray:
    """Returns bin_counts (3, gh, gw) accumulated over all processed images."""
    n = min(n, len(dataset))
    rates = torch.tensor([[LAM, ALPHA]], device=device)   # (1, 2) fixed
    gh = gw = IMG_SIZE // PATCH_SIZE                      # 12x12 patch grid

    cells      = []
    bin_counts = np.zeros((3, gh, gw), dtype=np.float32)  # ctx/tgt/ign

    for start in range(0, n, BATCH_SIZE):
        end  = min(start + BATCH_SIZE, n)
        idxs = range(start, end)

        displays, tensors = [], []
        for i in idxs:
            img_raw, _ = dataset[i]
            disp, ten  = _pil_to_display_and_tensor(img_raw)
            displays.append(disp)
            tensors.append(ten)

        x        = torch.stack(tensors).to(device)         # (B, 3, H, W)
        tokens   = encoder(x)                               # (B, N, D)
        B, N, _  = tokens.shape
        mask_out = masker(tokens, ema_full=None, rates=rates.expand(B, -1))

        ctx_idx  = mask_out.context_idx                    # (B, nctx)
        tgt_idx  = mask_out.target_idx                     # (B, ntgt)

        ctx_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        tgt_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        ctx_mask.scatter_(1, ctx_idx, True)
        tgt_mask.scatter_(1, tgt_idx, True)

        for bi in range(B):
            # 0=ctx, 1=tgt, 2=ign  (flat, then reshape to patch grid)
            bin_flat = torch.full((N,), 2, dtype=torch.long)
            bin_flat[ctx_mask[bi].cpu()] = 0
            bin_flat[tgt_mask[bi].cpu()] = 1
            bin_map = bin_flat.reshape(gh, gw).numpy()     # (12, 12)

            # Accumulate per-position bin counts
            for b in range(3):
                bin_counts[b] += (bin_map == b).astype(np.float32)

            orig_np = displays[bi]
            ign_np  = _apply_bin_overlay(orig_np, bin_map, "ign_only")
            col_np  = _apply_bin_overlay(orig_np, bin_map, "colored")
            cells.append(_make_cell(orig_np, ign_np, col_np))

        print(f"  {dataset_name}/{split}: {end}/{n}", end="\r")

    print()

    # Assemble grid
    cell_w, cell_h = cells[0].size
    n_rows = (len(cells) + grid_cols - 1) // grid_cols
    grid   = Image.new("RGB", (grid_cols * cell_w, n_rows * cell_h), (30, 30, 30))
    for k, cell in enumerate(cells):
        r, c = divmod(k, grid_cols)
        grid.paste(cell, (c * cell_w, r * cell_h))

    fname = out_dir / f"{dataset_name}_{split}.png"
    grid.save(fname)
    nctx = int(mask_out.context_idx.shape[1])
    ntgt = int(mask_out.target_idx.shape[1])
    nign = N - nctx - ntgt
    print(f"  Saved → {fname}   [{len(cells)} images, nctx={nctx} ntgt={ntgt} nign={nign}]")

    return bin_counts


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",      default=os.environ.get("PRETRAIN_CKPT", ""),
                        help="Path to pretrained checkpoint (or set PRETRAIN_CKPT env)")
    parser.add_argument("--data-root", default=os.environ.get("FAST", "/scratch") + "/datasets/")
    parser.add_argument("--out-dir",   default="ign_viz")
    parser.add_argument("--n",         type=int, default=500, help="Images per split")
    parser.add_argument("--grid-cols", type=int, default=GRID_COLS)
    parser.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not args.ckpt:
        raise SystemExit("Set PRETRAIN_CKPT or pass --ckpt")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device  = torch.device(args.device)

    print(f"Loading checkpoint: {args.ckpt}")
    sd_full = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    sd_full = sd_full.get("model", sd_full)

    print("Building encoder and masker ...")
    encoder = _build_encoder(sd_full, device)
    masker  = _build_masker(sd_full, device)
    print(f"  λ={LAM}  α={ALPHA}  device={device}")

    datasets_cfg = [
        ("stl10",   [
            ("train", lambda r: tv_datasets.STL10(r,  split="train", download=False)),
            ("test",  lambda r: tv_datasets.STL10(r,  split="test",  download=False)),
        ]),
        ("food101", [
            ("train", lambda r: tv_datasets.Food101(r, split="train", download=False)),
            ("test",  lambda r: tv_datasets.Food101(r, split="test",  download=False)),
        ]),
    ]

    for name, splits in datasets_cfg:
        gh = gw = IMG_SIZE // PATCH_SIZE
        total_counts  = np.zeros((3, gh, gw), dtype=np.float32)
        total_images  = 0

        for split, loader_fn in splits:
            print(f"\n{name}/{split}")
            try:
                ds = loader_fn(args.data_root)
            except Exception as e:
                print(f"  skipped ({e})")
                continue
            counts = visualize_split(
                name, split, ds, encoder, masker, device,
                out_dir, args.n, args.grid_cols,
            )
            total_counts += counts
            total_images += min(args.n, len(ds))

        if total_images > 0:
            save_avg_bin_heatmap(
                total_counts, total_images,
                out_dir / f"{name}_avg_bins.png",
            )

    print(f"\nDone. Output in ./{out_dir}/")
    print("Legend: [original | ign-only (ctx+tgt greyed) | colored (ctx=blue tgt=red ign=green)]")
    print("Heatmaps: brightness = fraction of images assigning that position to that bin.")


if __name__ == "__main__":
    main()
