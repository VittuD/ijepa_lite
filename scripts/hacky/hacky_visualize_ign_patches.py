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

Usage (single checkpoint — encoder and masker from same file):
  python scripts/hacky/hacky_visualize_ign_patches.py \\
      --ckpt /path/to/mi_coupled_last.pt \\
      [--data-root /path/to/datasets] [--out-dir ign_viz] [--n 500] [--device cuda]

Usage (split checkpoints — Phase-2 diagnostic):
  python scripts/hacky/hacky_visualize_ign_patches.py \\
      --encoder-ckpt /path/to/vanilla_last.pt \\
      --masker-ckpt  /path/to/mi_coupled_last.pt \\
      [--data-root /path/to/datasets] [--out-dir ign_viz_phase2]
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


def save_class_bin_heatmaps(
    overall_counts: np.ndarray,   # (3, gh, gw)
    overall_n: int,
    class_counts: dict,           # label → (3, gh, gw)
    class_n: dict,                # label → int
    class_names: list,            # list of str indexed by label
    out_path: Path,
    patch_px: int = 32,
    label_w: int = 140,
    row_gap: int = 6,             # dark separator pixels between rows
) -> None:
    """
    Save a vertically stacked heatmap: overall avg on top, one row per class below.

    Each row shows the 3 bin panels (ctx/tgt/ign) for that class, separated by a
    dark gap. Brightness = fraction of images in that class assigning that position
    to that bin.

    Also prints inter-class std per bin to stdout:
      For each bin and each spatial position, std across the per-class avg fractions.
      High std → different classes receive systematically different masking patterns.
      Near 0  → all classes get the same pattern (purely positional).
    """
    gh, gw = overall_counts.shape[1], overall_counts.shape[2]
    panel_w    = gw * patch_px
    row_h      = gh * patch_px
    row_stride = row_h + row_gap
    header_h   = 22   # space for "ctx / tgt / ign" column headers
    total_w    = label_w + 3 * panel_w

    sorted_cls = sorted(class_counts.keys())
    rows = [("overall", overall_counts, overall_n)] + [
        (class_names[c] if class_names and c < len(class_names) else str(c),
         class_counts[c], class_n[c])
        for c in sorted_cls
    ]

    # --- Inter-class std (only over the class rows, not overall) -------------
    bin_names  = ["ctx", "tgt", "ign"]
    bin_colors = [CTX_RGB, TGT_RGB, IGN_RGB]
    interclass_std_mean = [float("nan")] * 3
    if len(sorted_cls) >= 2:
        class_fracs = np.stack(
            [class_counts[c] / max(class_n[c], 1) for c in sorted_cls]
        )  # (n_classes, 3, gh, gw)
        per_pos_std = class_fracs.std(axis=0)          # (3, gh, gw)
        interclass_std_mean = per_pos_std.mean(axis=(1, 2)).tolist()  # (3,)
    print(f"  Inter-class spatial std  —  "
          + "  ".join(f"{n}={v:.4f}" for n, v in zip(bin_names, interclass_std_mean)))

    # -------------------------------------------------------------------------
    total_h = header_h + len(rows) * row_stride - row_gap  # no trailing gap
    img  = Image.new("RGB", (total_w, total_h), (20, 20, 20))
    draw = ImageDraw.Draw(img)

    try:
        from PIL import ImageFont
        font = ImageFont.load_default()
    except Exception:
        font = None

    # Column headers with inter-class std
    for b, (color, bname) in enumerate(zip(bin_colors, bin_names)):
        std_str = f"  σ={interclass_std_mean[b]:.4f}" if not float("nan") == interclass_std_mean[b] else ""
        draw.text((label_w + b * panel_w + 4, 4),
                  f"{bname} ({['blue','red','green'][b]}){std_str}",
                  fill=color, font=font)

    # Data rows
    for row_idx, (name, counts, n_img) in enumerate(rows):
        y0   = header_h + row_idx * row_stride
        frac = counts / max(n_img, 1)

        # Separator line at the top of every row except the first
        if row_idx > 0:
            draw.rectangle([0, y0 - row_gap, total_w, y0 - 1], fill=(50, 50, 50))

        # Class label on the left
        draw.text((4, y0 + row_h // 2 - 5),
                  f"{name}\n(n={n_img})", fill=(200, 200, 200), font=font)

        # 3 bin panels
        for b, color in enumerate(bin_colors):
            x0_panel = label_w + b * panel_w
            for i in range(gh):
                for j in range(gw):
                    v   = float(frac[b, i, j])
                    col = tuple(int(c * v) for c in color)
                    x0  = x0_panel + j * patch_px
                    y1  = y0 + i * patch_px
                    draw.rectangle([x0, y1, x0 + patch_px - 1, y1 + patch_px - 1],
                                   fill=col, outline=(40, 40, 40))

    img.save(out_path)
    print(f"  Saved per-class heatmap → {out_path}  "
          f"({len(sorted_cls)} classes, overall n={overall_n})")


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
) -> tuple:
    """Returns (bin_counts, class_bin_counts, class_img_n) accumulated over all processed images."""
    n = min(n, len(dataset))
    rates = torch.tensor([[LAM, ALPHA]], device=device)   # (1, 2) fixed
    gh = gw = IMG_SIZE // PATCH_SIZE                      # 12x12 patch grid

    cells            = []
    bin_counts       = np.zeros((3, gh, gw), dtype=np.float32)
    class_bin_counts: dict = {}   # label → (3, gh, gw)
    class_img_n:      dict = {}   # label → int

    for start in range(0, n, BATCH_SIZE):
        end  = min(start + BATCH_SIZE, n)
        idxs = range(start, end)

        displays, tensors, labels_batch = [], [], []
        for i in idxs:
            img_raw, lbl   = dataset[i]
            disp, ten      = _pil_to_display_and_tensor(img_raw)
            displays.append(disp)
            tensors.append(ten)
            labels_batch.append(int(lbl))

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

            # Accumulate overall per-position bin counts
            for b in range(3):
                bin_counts[b] += (bin_map == b).astype(np.float32)

            # Accumulate per-class bin counts
            lbl = labels_batch[bi]
            if lbl not in class_bin_counts:
                class_bin_counts[lbl] = np.zeros((3, gh, gw), dtype=np.float32)
                class_img_n[lbl] = 0
            for b in range(3):
                class_bin_counts[lbl][b] += (bin_map == b).astype(np.float32)
            class_img_n[lbl] += 1

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

    return bin_counts, class_bin_counts, class_img_n


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",         default=os.environ.get("PRETRAIN_CKPT", ""),
                        help="Checkpoint for BOTH encoder and masker (single-ckpt mode). "
                             "Can also be set via PRETRAIN_CKPT env var.")
    parser.add_argument("--encoder-ckpt", default=None,
                        help="Separate checkpoint for the encoder (target_encoder.*). "
                             "If set, --ckpt / MASKER_CKPT is used only for the masker.")
    parser.add_argument("--masker-ckpt",  default=os.environ.get("MASKER_CKPT", None),
                        help="Separate checkpoint for the masker (latent_masker.*). "
                             "Defaults to --ckpt when not provided.")
    parser.add_argument("--data-root", default=os.environ.get("FAST", "/scratch") + "/datasets/")
    parser.add_argument("--out-dir",   default="ign_viz")
    parser.add_argument("--n",         type=int, default=500, help="Images per split")
    parser.add_argument("--grid-cols", type=int, default=GRID_COLS)
    parser.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    encoder_ckpt = args.encoder_ckpt or args.ckpt
    masker_ckpt  = args.masker_ckpt  or args.ckpt

    if not encoder_ckpt:
        raise SystemExit("Provide --ckpt, --encoder-ckpt, or set PRETRAIN_CKPT env")
    if not masker_ckpt:
        raise SystemExit("Provide --ckpt, --masker-ckpt, or set PRETRAIN_CKPT / MASKER_CKPT env")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device  = torch.device(args.device)

    print(f"Loading encoder from : {encoder_ckpt}")
    sd_enc = torch.load(encoder_ckpt, map_location="cpu", weights_only=True)
    sd_enc = sd_enc.get("model", sd_enc)

    if masker_ckpt == encoder_ckpt:
        sd_msk = sd_enc
    else:
        print(f"Loading masker from  : {masker_ckpt}")
        sd_msk = torch.load(masker_ckpt, map_location="cpu", weights_only=True)
        sd_msk = sd_msk.get("model", sd_msk)

    print("Building encoder and masker ...")
    encoder = _build_encoder(sd_enc, device)
    masker  = _build_masker(sd_msk, device)
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
        total_counts       = np.zeros((3, gh, gw), dtype=np.float32)
        total_images       = 0
        total_class_counts: dict = {}
        total_class_n:      dict = {}
        class_names:        list = []

        for split, loader_fn in splits:
            print(f"\n{name}/{split}")
            try:
                ds = loader_fn(args.data_root)
            except Exception as e:
                print(f"  skipped ({e})")
                continue
            counts, cls_counts, cls_n = visualize_split(
                name, split, ds, encoder, masker, device,
                out_dir, args.n, args.grid_cols,
            )
            total_counts  += counts
            total_images  += min(args.n, len(ds))
            for c, arr in cls_counts.items():
                total_class_counts[c] = total_class_counts.get(
                    c, np.zeros_like(arr)) + arr
                total_class_n[c] = total_class_n.get(c, 0) + cls_n[c]
            if not class_names and hasattr(ds, "classes"):
                class_names = list(ds.classes)

        if total_images > 0:
            save_avg_bin_heatmap(
                total_counts, total_images,
                out_dir / f"{name}_avg_bins.png",
            )
            if total_class_counts:
                save_class_bin_heatmaps(
                    total_counts, total_images,
                    total_class_counts, total_class_n,
                    class_names,
                    out_dir / f"{name}_avg_bins_per_class.png",
                )

    print(f"\nDone. Output in ./{out_dir}/")
    print("Legend: [original | ign-only (ctx+tgt greyed) | colored (ctx=blue tgt=red ign=green)]")
    print("Heatmaps: brightness = fraction of images assigning that position to that bin.")


if __name__ == "__main__":
    main()
