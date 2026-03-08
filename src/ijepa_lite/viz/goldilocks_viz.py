"""
Reusable rendering utilities for Goldilocks masker visualization.

Extracted from hacky_visualize_goldilocks.py so that both the standalone
script and the VizCallback can share the same logic.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
ALPHA_SCORE  = 0.70
ALPHA_ASSIGN = 0.55
CTX_RGB  = (70,  130, 180)
TGT_RGB  = (220,  60,  60)
GREY     = (128, 128, 128)
BATCH_SIZE = 32


# ---------------------------------------------------------------------------
# Colormap: cold (blue=0) → yellow (0.5) → hot (red=1)
# ---------------------------------------------------------------------------

def score_color(s: float) -> tuple:
    """Map score in [0, 1] to an RGB color without matplotlib."""
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
# Per-image overlays
# ---------------------------------------------------------------------------

def apply_score_heatmap(
    orig_np: np.ndarray,
    scores: np.ndarray,
    patch_size: int,
    alpha: float = ALPHA_SCORE,
) -> np.ndarray:
    """Score heatmap: desaturate original, overlay each patch with cold→hot color."""
    grey = np.dot(orig_np.astype(float), [0.299, 0.587, 0.114])
    grey3 = np.stack([grey, grey, grey], axis=-1)
    out = grey3.copy()
    gh, gw = scores.shape
    for i in range(gh):
        for j in range(gw):
            col = np.array(score_color(scores[i, j]), dtype=float)
            y0, y1 = i * patch_size, (i + 1) * patch_size
            x0, x1 = j * patch_size, (j + 1) * patch_size
            patch = grey3[y0:y1, x0:x1]
            out[y0:y1, x0:x1] = ((1 - alpha) * patch + alpha * col).clip(0, 255)
    return out.astype(np.uint8)


def apply_assignment_overlay(
    orig_np: np.ndarray,
    bin_map: np.ndarray,
    patch_size: int,
    alpha: float = ALPHA_ASSIGN,
) -> np.ndarray:
    """Hard assignment: ctx=blue, tgt=red, ignored=grey."""
    out = orig_np.copy().astype(float)
    gh, gw = bin_map.shape
    for i in range(gh):
        for j in range(gw):
            b = bin_map[i, j]
            y0, y1 = i * patch_size, (i + 1) * patch_size
            x0, x1 = j * patch_size, (j + 1) * patch_size
            if b == 0:
                col = np.array(CTX_RGB, dtype=float)
            elif b == 1:
                col = np.array(TGT_RGB, dtype=float)
            else:
                col = np.array(GREY, dtype=float)
            orig_patch = orig_np[y0:y1, x0:x1].astype(float)
            blended = (1.0 - alpha) * orig_patch + alpha * col
            out[y0:y1, x0:x1] = blended
    return out.clip(0, 255).astype(np.uint8)


def make_cell(orig_np: np.ndarray, score_np: np.ndarray, assign_np: np.ndarray) -> Image.Image:
    h, w = orig_np.shape[:2]
    strip = Image.new("RGB", (3 * w, h))
    strip.paste(Image.fromarray(orig_np),   (0,     0))
    strip.paste(Image.fromarray(score_np),  (w,     0))
    strip.paste(Image.fromarray(assign_np), (2 * w, 0))
    return strip


# ---------------------------------------------------------------------------
# Aggregate heatmaps
# ---------------------------------------------------------------------------

def save_avg_score_heatmap(
    score_sums: np.ndarray,
    n_images: int,
    out_path: Path,
    patch_px: int = 40,
) -> None:
    """Single-panel heatmap: mean p_tgt per position."""
    gh, gw = score_sums.shape
    mean_scores = score_sums / max(n_images, 1)

    cell_h = gh * patch_px
    cell_w = gw * patch_px
    label_h = 24
    img_h = cell_h + label_h

    img  = Image.new("RGB", (cell_w, img_h), (20, 20, 20))
    draw = ImageDraw.Draw(img)

    try:
        from PIL import ImageFont
        font = ImageFont.load_default()
    except Exception:
        font = None

    draw.text((4, 4), "mean p_tgt per position  (cold=low, hot=high)",
              fill=(200, 200, 200), font=font)

    for i in range(gh):
        for j in range(gw):
            col = score_color(float(mean_scores[i, j]))
            x0, y0_ = j * patch_px, label_h + i * patch_px
            x1, y1_ = x0 + patch_px, y0_ + patch_px
            draw.rectangle([x0, y0_, x1, y1_], fill=col)
            draw.rectangle([x0, y0_, x1, y1_], outline=(40, 40, 40))

    img.save(out_path)
    print(f"  Saved score heatmap -> {out_path}  "
          f"(n={n_images}, min={float(mean_scores.min()):.3f}, "
          f"max={float(mean_scores.max()):.3f}, std={float(mean_scores.std()):.4f})")


def save_class_score_heatmap(
    overall_sums: np.ndarray,
    overall_n: int,
    class_sums: dict,
    class_n: dict,
    class_names: list,
    out_path: Path,
    patch_px: int = 32,
    label_w: int = 140,
    row_gap: int = 6,
) -> None:
    """Per-class soft score heatmap."""
    gh, gw = overall_sums.shape
    panel_w = gw * patch_px
    row_h = gh * patch_px
    row_stride = row_h + row_gap
    header_h = 22

    sorted_cls = sorted(class_sums.keys())
    rows = [("overall", overall_sums, overall_n)] + [
        (class_names[c] if class_names and c < len(class_names) else str(c),
         class_sums[c], class_n[c])
        for c in sorted_cls
    ]

    interclass_std = float("nan")
    if len(sorted_cls) >= 2:
        class_means = np.stack(
            [class_sums[c] / max(class_n[c], 1) for c in sorted_cls]
        )
        interclass_std = float(class_means.std(axis=0).mean())
    print(f"  Inter-class score std = {interclass_std:.4f}  "
          f"(> 0.05 -> content-adaptive scoring)")

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
        y0 = header_h + row_idx * row_stride
        mean = sums / max(n_img, 1)

        if row_idx > 0:
            draw.rectangle([0, y0 - row_gap, total_w, y0 - 1], fill=(50, 50, 50))

        draw.text((4, y0 + row_h // 2 - 5),
                  f"{name}\n(n={n_img})", fill=(200, 200, 200), font=font)

        for i in range(gh):
            for j in range(gw):
                col = score_color(float(mean[i, j]))
                x0 = label_w + j * patch_px
                y1 = y0 + i * patch_px
                draw.rectangle([x0, y1, x0 + patch_px - 1, y1 + patch_px - 1],
                               fill=col, outline=(40, 40, 40))

    img.save(out_path)
    print(f"  Saved per-class score heatmap -> {out_path}  "
          f"({len(sorted_cls)} classes, overall n={overall_n})")


# ---------------------------------------------------------------------------
# Main visualization driver
# ---------------------------------------------------------------------------

@torch.no_grad()
def visualize_split(
    dataset_name: str,
    split: str,
    dataset,
    encoder,
    masker,
    device: torch.device,
    out_dir: Path,
    n: int,
    grid_cols: int,
    patch_size: int,
    image_size: int,
    batch_size: int = BATCH_SIZE,
    k_tgt: Optional[int] = None,
) -> tuple:
    """
    Run masker on ``n`` images from ``dataset`` and produce visualization grids.

    Parameters
    ----------
    encoder : nn.Module
        Frozen encoder (target_encoder or similar).
    masker : LatentMasker
        Must return ``target_soft`` in MaskOutput.
    patch_size, image_size : int
        Used to compute grid dimensions.
    k_tgt : int or None
        If given, override masker.k_tgt_eval before running.

    Returns (score_sums, class_score_sums, class_img_n) for aggregation.
    """
    from torchvision import transforms

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    _to_tensor_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    _resize_crop = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
    ])

    if k_tgt is not None and hasattr(masker, "k_tgt_eval"):
        masker.k_tgt_eval = k_tgt

    n = min(n, len(dataset))
    gh = gw = image_size // patch_size

    cells = []
    score_sums = np.zeros((gh, gw), dtype=np.float64)
    class_score_sums: dict = {}
    class_img_n: dict = {}
    all_p_tgt = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        idxs = range(start, end)

        displays, tensors, labels_batch = [], [], []
        for i in idxs:
            img_raw, lbl = dataset[i]
            # Handle numpy arrays (STL-10 raw)
            if isinstance(img_raw, np.ndarray):
                if img_raw.ndim == 3 and img_raw.shape[0] == 3:
                    img_raw = np.transpose(img_raw, (1, 2, 0))
                pil = Image.fromarray(img_raw.astype(np.uint8))
            else:
                pil = img_raw
            pil_rgb = pil.convert("RGB")
            pil_crop = _resize_crop(pil_rgb)
            display = np.array(pil_crop, dtype=np.uint8)
            tensor = _to_tensor_norm(pil_crop)

            displays.append(display)
            tensors.append(tensor)
            labels_batch.append(int(lbl))

        x = torch.stack(tensors).to(device)
        tokens = encoder(x)
        B, N, _ = tokens.shape

        mask_out = masker(tokens, ema_full=tokens)

        ctx_idx = mask_out.context_idx
        tgt_idx = mask_out.target_idx
        p_tgt = mask_out.target_soft

        if p_tgt is not None:
            all_p_tgt.append(p_tgt.cpu())

        ctx_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        tgt_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        ctx_mask.scatter_(1, ctx_idx, True)
        tgt_mask.scatter_(1, tgt_idx, True)

        for bi in range(B):
            bin_flat = torch.full((N,), 2, dtype=torch.long)
            bin_flat[ctx_mask[bi].cpu()] = 0
            bin_flat[tgt_mask[bi].cpu()] = 1
            bin_map = bin_flat.reshape(gh, gw).numpy()

            if p_tgt is not None:
                score_map = p_tgt[bi].cpu().reshape(gh, gw).numpy()
            else:
                score_map = (bin_map == 1).astype(np.float32)

            score_sums += score_map.astype(np.float64)

            lbl = labels_batch[bi]
            if lbl not in class_score_sums:
                class_score_sums[lbl] = np.zeros((gh, gw), dtype=np.float64)
                class_img_n[lbl] = 0
            class_score_sums[lbl] += score_map.astype(np.float64)
            class_img_n[lbl] += 1

            orig_np = displays[bi]
            score_np = apply_score_heatmap(orig_np, score_map, patch_size)
            assign_np = apply_assignment_overlay(orig_np, bin_map, patch_size)
            cells.append(make_cell(orig_np, score_np, assign_np))

        print(f"  {dataset_name}/{split}: {end}/{n}", end="\r")

    print()

    # Numeric content-adaptivity report
    if all_p_tgt:
        p_cat = torch.cat(all_p_tgt, dim=0)
        marginal_score_std = p_cat.mean(dim=0).std().item()
        p_tgt_score_std = p_cat.std(dim=0).mean().item()
        print(f"  marginal_score_std = {marginal_score_std:.4f}  "
              f"(-> 0 = uniform marginal = no positional bias)")
        print(f"  p_tgt_score_std    = {p_tgt_score_std:.4f}  "
              f"(high = scores vary across images = content-adaptive)")

    # Assemble image grid
    if cells:
        cell_w, cell_h = cells[0].size
        n_rows = (len(cells) + grid_cols - 1) // grid_cols
        grid = Image.new("RGB", (grid_cols * cell_w, n_rows * cell_h), (30, 30, 30))
        for k, cell in enumerate(cells):
            r, c = divmod(k, grid_cols)
            grid.paste(cell, (c * cell_w, r * cell_h))

        fname = out_dir / f"{dataset_name}_{split}.png"
        grid.save(fname)
        nctx = int(mask_out.context_idx.shape[1])
        ntgt = int(mask_out.target_idx.shape[1])
        print(f"  Saved -> {fname}   [{len(cells)} images, nctx={nctx} ntgt={ntgt} "
              f"K/N={ntgt}/{N}={ntgt/N:.2f}]")

    return score_sums, class_score_sums, class_img_n
