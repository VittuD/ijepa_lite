#!/usr/bin/env python3
"""
hacky_visualize_goldilocks.py  (v2)

Visualizes the GoldilocksTeacherMasker's scoring and ctx/tgt patch assignments.

Panels per image (left to right):
  [original | soft score heatmap | hard assignment (ctx=blue, tgt=red, ignored=grey)]

The soft score heatmap is the primary diagnostic: it shows the raw p_tgt scores
(cold=low, hot=high) overlaid on a desaturated image. With high K/N (e.g. K=94/N=144
≈ 65%), the hard assignment is near-trivial, so the score heatmap carries the signal.

Aggregate outputs:
  <out_dir>/<dataset>_avg_score.png          — mean p_tgt per position across all images
  <out_dir>/<dataset>_per_class_score.png    — mean p_tgt per position per class

Numeric output (stdout):
  marginal_score_std  — near 0 = uniform marginal = no positional bias
  p_tgt_score_std     — high = scores vary more across images = content-adaptive
  inter-class score std — > 0.05 = different classes scored differently

Usage (single checkpoint — encoder and masker from same file):
  python hacky_visualize_goldilocks.py \\
      --ckpt /path/to/last.pt \\
      [--data-root /path/to/datasets] [--out-dir goldilocks_viz] [--n 500]

  Override eval k_tgt (e.g. to avoid 17-tgt display from variable-k checkpoint):
      --k-tgt 94

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

GRID_COLS    = 25
BATCH_SIZE   = 32
ALPHA_SCORE  = 0.70   # blend weight for score heatmap overlay
ALPHA_ASSIGN = 0.55   # blend weight for hard assignment overlay

CTX_RGB = (70,  130, 180)   # steel blue
TGT_RGB = (220,  60,  60)   # red
GREY    = (128, 128, 128)


# ---------------------------------------------------------------------------
# Colormap: cold (blue=0) → yellow (0.5) → hot (red=1), no matplotlib
# ---------------------------------------------------------------------------

def _score_color(s: float) -> tuple:
    """Map score ∈ [0, 1] to an RGB color without matplotlib."""
    s = max(0.0, min(1.0, float(s)))
    if s <= 0.5:
        t = s * 2.0
        r = int(30  + t * (240 - 30))
        g = int(80  + t * (240 - 80))
        b = int(220 + t * (100 - 220))
    else:
        t = (s - 0.5) * 2.0
        r = int(240 + t * (220 - 240))
        g = int(240 - t * (240 - 40))
        b = int(100 - t * (100 - 40))
    return (r, g, b)


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
        print(f"  [encoder] missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
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
        print(f"  [masker] missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
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


def _apply_score_heatmap(
    orig_np: np.ndarray,   # (H, W, 3) uint8
    scores:  np.ndarray,   # (gh, gw) float in [0, 1]
) -> np.ndarray:
    """Score heatmap: desaturate original, overlay each patch with cold→hot color."""
    grey = np.dot(orig_np.astype(float), [0.299, 0.587, 0.114])
    grey3 = np.stack([grey, grey, grey], axis=-1)   # (H, W, 3)
    out = grey3.copy()
    P = PATCH_SIZE
    gh, gw = scores.shape
    for i in range(gh):
        for j in range(gw):
            col  = np.array(_score_color(scores[i, j]), dtype=float)
            y0, y1 = i * P, (i + 1) * P
            x0, x1 = j * P, (j + 1) * P
            patch = grey3[y0:y1, x0:x1]
            out[y0:y1, x0:x1] = ((1 - ALPHA_SCORE) * patch + ALPHA_SCORE * col).clip(0, 255)
    return out.astype(np.uint8)


def _apply_assignment_overlay(
    orig_np: np.ndarray,   # (H, W, 3) uint8
    bin_map: np.ndarray,   # (gh, gw) int: 0=ctx, 1=tgt, 2=ignored
) -> np.ndarray:
    """Hard assignment: ctx=blue, tgt=red, ignored=grey."""
    out = orig_np.copy().astype(float)
    P = PATCH_SIZE
    gh, gw = bin_map.shape
    for i in range(gh):
        for j in range(gw):
            b = bin_map[i, j]
            y0, y1 = i * P, (i + 1) * P
            x0, x1 = j * P, (j + 1) * P
            if b == 0:
                col = np.array(CTX_RGB, dtype=float)
            elif b == 1:
                col = np.array(TGT_RGB, dtype=float)
            else:
                col = np.array(GREY, dtype=float)
            orig_patch = orig_np[y0:y1, x0:x1].astype(float)
            blended = (1.0 - ALPHA_ASSIGN) * orig_patch + ALPHA_ASSIGN * col
            out[y0:y1, x0:x1] = blended
    return out.clip(0, 255).astype(np.uint8)


def _make_cell(orig_np, score_np, assign_np) -> Image.Image:
    h, w = orig_np.shape[:2]
    strip = Image.new("RGB", (3 * w, h))
    strip.paste(Image.fromarray(orig_np),  (0,     0))
    strip.paste(Image.fromarray(score_np), (w,     0))
    strip.paste(Image.fromarray(assign_np),(2 * w, 0))
    return strip


# ---------------------------------------------------------------------------
# Aggregate heatmap outputs (soft scores, not binary counts)
# ---------------------------------------------------------------------------

def _save_avg_score_heatmap(
    score_sums: np.ndarray,   # (gh, gw) cumulative p_tgt
    n_images:   int,
    out_path:   Path,
    patch_px:   int = 40,
) -> None:
    """Single-panel heatmap: mean p_tgt per position (cold→hot)."""
    gh, gw = score_sums.shape
    mean_scores = score_sums / max(n_images, 1)   # (gh, gw) ∈ [0, 1]

    cell_h  = gh * patch_px
    cell_w  = gw * patch_px
    label_h = 24
    img_h   = cell_h + label_h

    img  = Image.new("RGB", (cell_w, img_h), (20, 20, 20))
    draw = ImageDraw.Draw(img)

    try:
        from PIL import ImageFont
        font = ImageFont.load_default()
    except Exception:
        font = None

    draw.text((4, 4), "mean p_tgt per position  (cold=low, hot=high)", fill=(200, 200, 200), font=font)

    for i in range(gh):
        for j in range(gw):
            col = _score_color(float(mean_scores[i, j]))
            x0, y0 = j * patch_px, label_h + i * patch_px
            x1, y1 = x0 + patch_px, y0 + patch_px
            draw.rectangle([x0, y0, x1, y1], fill=col)
            draw.rectangle([x0, y0, x1, y1], outline=(40, 40, 40))

    img.save(out_path)
    score_min = float(mean_scores.min())
    score_max = float(mean_scores.max())
    score_std = float(mean_scores.std())
    print(f"  Saved score heatmap → {out_path}  "
          f"(n={n_images}, min={score_min:.3f}, max={score_max:.3f}, std={score_std:.4f})")


def _save_class_score_heatmap(
    overall_sums:  np.ndarray,   # (gh, gw)
    overall_n:     int,
    class_sums:    dict,         # label → (gh, gw)
    class_n:       dict,
    class_names:   list,
    out_path:      Path,
    patch_px:      int = 32,
    label_w:       int = 140,
    row_gap:       int = 6,
) -> None:
    """
    Per-class soft score heatmap.  Rows: overall + one per class.
    Prints inter-class spatial std of mean scores:
      High (> 0.05) → different classes scored differently = content-adaptive.
      Near 0        → same score map for all classes = positional / content-blind.
    """
    gh, gw = overall_sums.shape
    panel_w    = gw * patch_px
    row_h      = gh * patch_px
    row_stride = row_h + row_gap
    header_h   = 22

    sorted_cls = sorted(class_sums.keys())
    rows = [("overall", overall_sums, overall_n)] + [
        (class_names[c] if class_names and c < len(class_names) else str(c),
         class_sums[c], class_n[c])
        for c in sorted_cls
    ]

    # Inter-class std of per-position mean score (the content-adaptivity signal)
    interclass_std = float("nan")
    if len(sorted_cls) >= 2:
        class_means = np.stack(
            [class_sums[c] / max(class_n[c], 1) for c in sorted_cls]
        )  # (n_classes, gh, gw)
        interclass_std = float(class_means.std(axis=0).mean())
    print(f"  Inter-class score std = {interclass_std:.4f}  "
          f"(> 0.05 → content-adaptive scoring)")

    total_w = label_w + panel_w
    total_h = header_h + len(rows) * row_stride - row_gap
    img  = Image.new("RGB", (total_w, total_h), (20, 20, 20))
    draw = ImageDraw.Draw(img)

    try:
        from PIL import ImageFont
        font = ImageFont.load_default()
    except Exception:
        font = None

    draw.text((label_w + 4, 4),
              f"mean p_tgt  (inter-class std={interclass_std:.4f})",
              fill=(200, 200, 200), font=font)

    for row_idx, (name, sums, n_img) in enumerate(rows):
        y0   = header_h + row_idx * row_stride
        mean = sums / max(n_img, 1)

        if row_idx > 0:
            draw.rectangle([0, y0 - row_gap, total_w, y0 - 1], fill=(50, 50, 50))

        draw.text((4, y0 + row_h // 2 - 5),
                  f"{name}\n(n={n_img})", fill=(200, 200, 200), font=font)

        for i in range(gh):
            for j in range(gw):
                col = _score_color(float(mean[i, j]))
                x0  = label_w + j * patch_px
                y1  = y0 + i * patch_px
                draw.rectangle([x0, y1, x0 + patch_px - 1, y1 + patch_px - 1],
                                fill=col, outline=(40, 40, 40))

    img.save(out_path)
    print(f"  Saved per-class score heatmap → {out_path}  "
          f"({len(sorted_cls)} classes, overall n={overall_n})")


# ---------------------------------------------------------------------------
# Main visualization loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def visualize_split(
    dataset_name: str,
    split:        str,
    dataset,
    encoder:  ViTTokens,
    masker:   GoldilocksTeacherMasker,
    device:   torch.device,
    out_dir:  Path,
    n:        int,
    grid_cols: int,
) -> tuple:
    n = min(n, len(dataset))
    gh = gw = IMG_SIZE // PATCH_SIZE

    cells             = []
    score_sums        = np.zeros((gh, gw), dtype=np.float64)
    class_score_sums: dict = {}
    class_img_n:      dict = {}
    all_p_tgt               = []

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

        x      = torch.stack(tensors).to(device)
        tokens = encoder(x)                          # (B, N, D)
        B, N, _ = tokens.shape

        mask_out = masker(tokens, ema_full=tokens)

        ctx_idx = mask_out.context_idx              # (B, K_ctx)
        tgt_idx = mask_out.target_idx               # (B, K_tgt)
        p_tgt   = mask_out.target_soft              # (B, N) soft scores

        if p_tgt is not None:
            all_p_tgt.append(p_tgt.cpu())

        ctx_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        tgt_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        ctx_mask.scatter_(1, ctx_idx, True)
        tgt_mask.scatter_(1, tgt_idx, True)

        for bi in range(B):
            # Hard assignment map (gh, gw): 0=ctx, 1=tgt, 2=ignored
            bin_flat = torch.full((N,), 2, dtype=torch.long)
            bin_flat[ctx_mask[bi].cpu()] = 0
            bin_flat[tgt_mask[bi].cpu()] = 1
            bin_map = bin_flat.reshape(gh, gw).numpy()

            # Soft score map (gh, gw)
            if p_tgt is not None:
                score_map = p_tgt[bi].cpu().reshape(gh, gw).numpy()
            else:
                # fallback: binary tgt as proxy
                score_map = (bin_map == 1).astype(np.float32)

            score_sums += score_map.astype(np.float64)

            lbl = labels_batch[bi]
            if lbl not in class_score_sums:
                class_score_sums[lbl] = np.zeros((gh, gw), dtype=np.float64)
                class_img_n[lbl] = 0
            class_score_sums[lbl] += score_map.astype(np.float64)
            class_img_n[lbl] += 1

            orig_np   = displays[bi]
            score_np  = _apply_score_heatmap(orig_np, score_map)
            assign_np = _apply_assignment_overlay(orig_np, bin_map)
            cells.append(_make_cell(orig_np, score_np, assign_np))

        print(f"  {dataset_name}/{split}: {end}/{n}", end="\r")

    print()

    # Numeric content-adaptivity report
    if all_p_tgt:
        p_cat = torch.cat(all_p_tgt, dim=0)           # (total, N)
        marginal_score_std = p_cat.mean(dim=0).std().item()
        p_tgt_score_std    = p_cat.std(dim=0).mean().item()
        print(f"  marginal_score_std = {marginal_score_std:.4f}  "
              f"(↓ 0 = uniform marginal = no positional bias)")
        print(f"  p_tgt_score_std    = {p_tgt_score_std:.4f}  "
              f"(↑ = scores vary across images = content-adaptive)")

    # Assemble image grid
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
    print(f"  Saved → {fname}   [{len(cells)} images, nctx={nctx} ntgt={ntgt} "
          f"K/N={ntgt}/{N}={ntgt/N:.2f}]")

    return score_sums, class_score_sums, class_img_n


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
    parser.add_argument("--k-tgt",     type=int, default=None,
                        help="Override masker k_tgt_eval (e.g. 94 for fixed-k checkpoints). "
                             "Default: use masker's own k_tgt_eval.")
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

    if args.k_tgt is not None:
        print(f"  Overriding k_tgt_eval: {masker.k_tgt_eval} → {args.k_tgt}")
        masker.k_tgt_eval = args.k_tgt

    print(f"  k_tgt_eval={masker.k_tgt_eval}  k_ctx_eval={masker.k_ctx_eval}  "
          f"K/N={masker.k_tgt_eval}/{masker.num_patches}={masker.k_tgt_eval/masker.num_patches:.2f}  "
          f"device={device}")

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
        total_score_sums        = np.zeros((gh, gw), dtype=np.float64)
        total_images            = 0
        total_class_score_sums: dict = {}
        total_class_n:          dict = {}
        class_names:            list = []

        for split, loader_fn in splits:
            print(f"\n{name}/{split}")
            try:
                ds = loader_fn(args.data_root)
            except Exception as e:
                print(f"  skipped ({e})")
                continue

            score_sums, cls_sums, cls_n = visualize_split(
                name, split, ds, encoder, masker, device,
                out_dir, args.n, args.grid_cols,
            )
            total_score_sums += score_sums
            total_images     += min(args.n, len(ds))
            for c, arr in cls_sums.items():
                total_class_score_sums[c] = total_class_score_sums.get(
                    c, np.zeros_like(arr)) + arr
                total_class_n[c] = total_class_n.get(c, 0) + cls_n[c]
            if not class_names and hasattr(ds, "classes"):
                class_names = list(ds.classes)

        if total_images > 0:
            _save_avg_score_heatmap(
                total_score_sums, total_images,
                out_dir / f"{name}_avg_score.png",
            )
            if total_class_score_sums:
                _save_class_score_heatmap(
                    total_score_sums, total_images,
                    total_class_score_sums, total_class_n,
                    class_names,
                    out_dir / f"{name}_per_class_score.png",
                )

    print(f"\nDone. Output in ./{out_dir}/")
    print("Legend: [original | soft score (cold=low, hot=high) | assignment (ctx=blue tgt=red ignored=grey)]")


if __name__ == "__main__":
    main()
