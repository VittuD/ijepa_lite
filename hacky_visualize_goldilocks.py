#!/usr/bin/env python3
"""
hacky_visualize_goldilocks.py

Visualizes the GoldilocksTeacherMasker's ctx/tgt patch assignments for
STL-10 (and optionally Food101) images.

Color coding:
  ctx = blue   (context block — given to predictor)
  tgt = red    (target patches — selected by masker)
  rest = grey  (unselected)

Panels per image:
  [original | tgt-highlighted (ctx+rest greyed) | colored]

Also saves:
  - Per-position average assignment heatmaps (ctx and tgt)
  - Per-class average tgt heatmap (inter-class std = content-adaptivity signal)
  - Numeric report: marginal_score_std (positional bias detector)

Usage (single checkpoint — encoder and masker from same file):
  python hacky_visualize_goldilocks.py \\
      --ckpt /path/to/goldilocks_last.pt \\
      [--data-root /path/to/datasets] [--out-dir goldilocks_viz] [--n 500]

Usage (split checkpoints):
  python hacky_visualize_goldilocks.py \\
      --encoder-ckpt /path/to/encoder.pt \\
      --masker-ckpt  /path/to/masker.pt \\
      [--data-root /path/to/datasets] [--out-dir goldilocks_viz_split]
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

import ijepa_lite.masking.goldilocks_masker  # noqa: F401 — populates registry
from ijepa_lite.masking.goldilocks_masker import GoldilocksTeacherMasker
from ijepa_lite.models.vit_tokens import ViTTokens, _remove_classifier_head

# ---------------------------------------------------------------------------
# Architecture constants — stl10_vits_ps8_goldilocks experiment
# ---------------------------------------------------------------------------
IMG_SIZE   = 96
PATCH_SIZE = 8
EMBED_DIM  = 384
DEPTH      = 12
NUM_HEADS  = 6
PRED_DIM   = 192
PRED_DEPTH = 2
PRED_HEADS = 6

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

GRID_COLS     = 25
BATCH_SIZE    = 32
ALPHA_OVERLAY = 0.55

# 2-way colour scheme
CTX_RGB  = (70,  130, 180)   # steel blue
TGT_RGB  = (220,  60,  60)   # red
GREY     = (128, 128, 128)


# ---------------------------------------------------------------------------
# Model builders
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
    missing, _ = enc.load_state_dict(enc_sd, strict=False)
    if missing:
        print(f"  [encoder] missing keys: {missing[:5]}{'...' if len(missing)>5 else ''}")

    enc.requires_grad_(False)
    return enc.to(device).eval()


def _build_masker(sd_full: dict, device: torch.device) -> GoldilocksTeacherMasker:
    num_patches = (IMG_SIZE // PATCH_SIZE) ** 2
    masker = GoldilocksTeacherMasker(
        dim=EMBED_DIM,
        predictor_dim=PRED_DIM,
        depth=PRED_DEPTH,
        num_heads=PRED_HEADS,
        mlp_ratio=4.0,
        dropout=0.0,
        num_patches=num_patches,
    )

    prefix = "latent_masker."
    m_sd = {k[len(prefix):]: v for k, v in sd_full.items() if k.startswith(prefix)}
    missing, unexpected = masker.load_state_dict(m_sd, strict=False)
    if missing:
        print(f"  [masker] missing keys: {missing[:5]}{'...' if len(missing)>5 else ''}")

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
    if isinstance(img_raw, np.ndarray):
        if img_raw.ndim == 3 and img_raw.shape[0] == 3:
            img_raw = np.transpose(img_raw, (1, 2, 0))
        pil = Image.fromarray(img_raw.astype(np.uint8))
    else:
        pil = img_raw
    pil_rgb  = pil.convert("RGB")
    pil_crop = _resize_crop(pil_rgb)
    display  = np.array(pil_crop, dtype=np.uint8)
    tensor   = _to_tensor_norm(pil_crop)
    return display, tensor


def _apply_overlay(
    orig_np: np.ndarray,   # (H, W, 3)
    bin_map: np.ndarray,   # (gh, gw) int: 0=ctx, 1=tgt, 2=rest
    mode: str,             # "tgt_only" | "colored"
) -> np.ndarray:
    out = orig_np.copy()
    P = PATCH_SIZE
    gh, gw = bin_map.shape

    for i in range(gh):
        for j in range(gw):
            b = bin_map[i, j]
            y0, y1 = i * P, (i + 1) * P
            x0, x1 = j * P, (j + 1) * P

            if mode == "tgt_only":
                # Show only target patches; grey out everything else
                if b != 1:
                    out[y0:y1, x0:x1] = GREY
            elif mode == "colored":
                if b == 0:
                    col = np.array(CTX_RGB, dtype=float)
                elif b == 1:
                    col = np.array(TGT_RGB, dtype=float)
                else:
                    continue   # rest = unmodified original
                orig_patch = orig_np[y0:y1, x0:x1].astype(float)
                blended = (1.0 - ALPHA_OVERLAY) * orig_patch + ALPHA_OVERLAY * col
                out[y0:y1, x0:x1] = blended.clip(0, 255).astype(np.uint8)

    return out


def _make_cell(orig_np, tgt_np, col_np) -> Image.Image:
    h, w = orig_np.shape[:2]
    strip = Image.new("RGB", (3 * w, h))
    strip.paste(Image.fromarray(orig_np), (0,     0))
    strip.paste(Image.fromarray(tgt_np),  (w,     0))
    strip.paste(Image.fromarray(col_np),  (2 * w, 0))
    return strip


def _save_avg_heatmap(
    bin_counts: np.ndarray,   # (2, gh, gw) — [ctx, tgt]
    n_images: int,
    out_path: Path,
    patch_px: int = 40,
) -> None:
    """2-panel heatmap: ctx (blue) and tgt (red). Brightness = selection frequency."""
    gh, gw = bin_counts.shape[1], bin_counts.shape[2]
    frac   = bin_counts / max(n_images, 1)

    bin_names  = ["ctx (blue)", "tgt (red)"]
    bin_colors = [CTX_RGB, TGT_RGB]

    cell_w  = gw * patch_px
    cell_h  = gh * patch_px
    label_h = 24
    panel_h = cell_h + label_h

    img = Image.new("RGB", (2 * cell_w, panel_h), (20, 20, 20))

    try:
        from PIL import ImageFont
        font = ImageFont.load_default()
    except Exception:
        font = None

    for b, (name, color) in enumerate(zip(bin_names, bin_colors)):
        panel = Image.new("RGB", (cell_w, panel_h), (20, 20, 20))
        draw  = ImageDraw.Draw(panel)
        draw.text((4, 4), name, fill=color, font=font)

        for i in range(gh):
            for j in range(gw):
                v   = float(frac[b, i, j])
                col = tuple(int(c * v) for c in color)
                x0, y0 = j * patch_px, label_h + i * patch_px
                x1, y1 = x0 + patch_px, y0 + patch_px
                draw.rectangle([x0, y0, x1, y1], fill=col)
                draw.rectangle([x0, y0, x1, y1], outline=(40, 40, 40))

        img.paste(panel, (b * cell_w, 0))

    img.save(out_path)
    print(f"  Saved heatmap → {out_path}  (n={n_images})")


def _save_class_tgt_heatmap(
    overall_counts: np.ndarray,   # (2, gh, gw)
    overall_n: int,
    class_counts: dict,           # label → (2, gh, gw)
    class_n: dict,
    class_names: list,
    out_path: Path,
    patch_px: int = 32,
    label_w: int = 140,
    row_gap: int = 6,
) -> None:
    """
    Per-class tgt-assignment heatmap. Rows: overall + one per class.
    Also prints inter-class spatial std of tgt frequency:
      High → different classes get different targets (content-adaptive).
      Near 0 → all classes get the same targets (positional).
    """
    gh, gw = overall_counts.shape[1], overall_counts.shape[2]
    panel_w    = gw * patch_px
    row_h      = gh * patch_px
    row_stride = row_h + row_gap
    header_h   = 22
    total_w    = label_w + 2 * panel_w   # ctx + tgt panels

    sorted_cls = sorted(class_counts.keys())
    rows = [("overall", overall_counts, overall_n)] + [
        (class_names[c] if class_names and c < len(class_names) else str(c),
         class_counts[c], class_n[c])
        for c in sorted_cls
    ]

    # Inter-class std on the tgt panel (bin=1) — the content-adaptivity signal
    bin_names  = ["ctx", "tgt"]
    bin_colors = [CTX_RGB, TGT_RGB]
    interclass_std = [float("nan"), float("nan")]
    if len(sorted_cls) >= 2:
        class_fracs = np.stack(
            [class_counts[c] / max(class_n[c], 1) for c in sorted_cls]
        )  # (n_classes, 2, gh, gw)
        per_pos_std = class_fracs.std(axis=0)          # (2, gh, gw)
        interclass_std = per_pos_std.mean(axis=(1, 2)).tolist()
    print(f"  Inter-class spatial std  —  "
          + "  ".join(f"{n}={v:.4f}" for n, v in zip(bin_names, interclass_std)))
    print(f"  (tgt inter-class std > 0.05 suggests content-adaptive target selection)")

    total_h = header_h + len(rows) * row_stride - row_gap
    img  = Image.new("RGB", (total_w, total_h), (20, 20, 20))
    draw = ImageDraw.Draw(img)

    try:
        from PIL import ImageFont
        font = ImageFont.load_default()
    except Exception:
        font = None

    for b, (color, bname) in enumerate(zip(bin_colors, bin_names)):
        std_str = f"  σ={interclass_std[b]:.4f}"
        draw.text((label_w + b * panel_w + 4, 4),
                  f"{bname} ({'blue' if b==0 else 'red'}){std_str}",
                  fill=color, font=font)

    for row_idx, (name, counts, n_img) in enumerate(rows):
        y0   = header_h + row_idx * row_stride
        frac = counts / max(n_img, 1)

        if row_idx > 0:
            draw.rectangle([0, y0 - row_gap, total_w, y0 - 1], fill=(50, 50, 50))

        draw.text((4, y0 + row_h // 2 - 5),
                  f"{name}\n(n={n_img})", fill=(200, 200, 200), font=font)

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
    masker: GoldilocksTeacherMasker,
    device: torch.device,
    out_dir: Path,
    n: int,
    grid_cols: int,
) -> tuple:
    n = min(n, len(dataset))
    gh = gw = IMG_SIZE // PATCH_SIZE

    cells            = []
    bin_counts       = np.zeros((2, gh, gw), dtype=np.float32)   # [ctx, tgt]
    class_bin_counts: dict = {}
    class_img_n:      dict = {}

    # For marginal_score_std computation
    all_p_tgt = []

    for start in range(0, n, BATCH_SIZE):
        end  = min(start + BATCH_SIZE, n)
        idxs = range(start, end)

        displays, tensors, labels_batch = [], [], []
        for i in idxs:
            img_raw, lbl = dataset[i]
            disp, ten    = _pil_to_display_and_tensor(img_raw)
            displays.append(disp)
            tensors.append(ten)
            labels_batch.append(int(lbl))

        x      = torch.stack(tensors).to(device)   # (B, 3, H, W)
        tokens = encoder(x)                         # (B, N, D)
        B, N, _ = tokens.shape

        # Goldilocks masker forward: tokens + ema_full (same for compressor=full)
        mask_out = masker(tokens, ema_full=tokens)

        ctx_idx = mask_out.context_idx   # (B, K_ctx)
        tgt_idx = mask_out.target_idx    # (B, K_tgt)

        if mask_out.target_soft is not None:
            all_p_tgt.append(mask_out.target_soft.cpu())

        ctx_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        tgt_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        ctx_mask.scatter_(1, ctx_idx, True)
        tgt_mask.scatter_(1, tgt_idx, True)

        for bi in range(B):
            # 0=ctx, 1=tgt, 2=rest
            bin_flat = torch.full((N,), 2, dtype=torch.long)
            bin_flat[ctx_mask[bi].cpu()] = 0
            bin_flat[tgt_mask[bi].cpu()] = 1
            bin_map = bin_flat.reshape(gh, gw).numpy()

            for b in range(2):
                bin_counts[b] += (bin_map == b).astype(np.float32)

            lbl = labels_batch[bi]
            if lbl not in class_bin_counts:
                class_bin_counts[lbl] = np.zeros((2, gh, gw), dtype=np.float32)
                class_img_n[lbl] = 0
            for b in range(2):
                class_bin_counts[lbl][b] += (bin_map == b).astype(np.float32)
            class_img_n[lbl] += 1

            orig_np = displays[bi]
            tgt_np  = _apply_overlay(orig_np, bin_map, "tgt_only")
            col_np  = _apply_overlay(orig_np, bin_map, "colored")
            cells.append(_make_cell(orig_np, tgt_np, col_np))

        print(f"  {dataset_name}/{split}: {end}/{n}", end="\r")

    print()

    # Numeric content-adaptivity report
    if all_p_tgt:
        p_cat = torch.cat(all_p_tgt, dim=0)           # (total_B, N)
        marginal = p_cat.mean(dim=0)                   # (N,)
        marginal_score_std = marginal.std().item()
        p_tgt_score_std    = p_cat.std(dim=0).mean().item()
        print(f"  marginal_score_std  = {marginal_score_std:.4f}  "
              f"(↓ toward 0 = uniform marginal = no positional bias)")
        print(f"  p_tgt_score_std     = {p_tgt_score_std:.4f}  "
              f"(↑ = scores vary more across images = content-adaptive)")

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
    print(f"  Saved → {fname}   [{len(cells)} images, nctx={nctx} ntgt={ntgt}]")

    return bin_counts, class_bin_counts, class_img_n


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",         default=os.environ.get("PRETRAIN_CKPT", ""))
    parser.add_argument("--encoder-ckpt", default=None)
    parser.add_argument("--masker-ckpt",  default=os.environ.get("MASKER_CKPT", None))
    parser.add_argument("--data-root", default=os.environ.get("FAST", "/scratch") + "/datasets/")
    parser.add_argument("--out-dir",   default="goldilocks_viz")
    parser.add_argument("--n",         type=int, default=500)
    parser.add_argument("--grid-cols", type=int, default=GRID_COLS)
    parser.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    encoder_ckpt = args.encoder_ckpt or args.ckpt
    masker_ckpt  = args.masker_ckpt  or args.ckpt

    if not encoder_ckpt:
        raise SystemExit("Provide --ckpt, --encoder-ckpt, or set PRETRAIN_CKPT env")

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
    print(f"  k_tgt_eval={masker.k_tgt_eval}  k_ctx_eval={masker.k_ctx_eval}  device={device}")

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
        gh = gw = IMG_SIZE // PATCH_SIZE
        total_counts       = np.zeros((2, gh, gw), dtype=np.float32)
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
            _save_avg_heatmap(
                total_counts, total_images,
                out_dir / f"{name}_avg_bins.png",
            )
            if total_class_counts:
                _save_class_tgt_heatmap(
                    total_counts, total_images,
                    total_class_counts, total_class_n,
                    class_names,
                    out_dir / f"{name}_avg_bins_per_class.png",
                )

    print(f"\nDone. Output in ./{out_dir}/")
    print("Legend: [original | tgt-highlighted (ctx+rest greyed) | colored (ctx=blue, tgt=red)]")
    print("Heatmaps: brightness = fraction of images assigning that position to ctx/tgt.")
    print("Inter-class tgt std > 0.05 → content-adaptive target selection.")


if __name__ == "__main__":
    main()
