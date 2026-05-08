"""
Reusable rendering utilities for masker visualization.

Supports two masker styles:
- **Goldilocks / 2-way**: middle panel = cold→hot heatmap of p_tgt.
- **MI / 3-way**: middle panel = soft 3-way overlay (ctx=blue, tgt=red, ign=grey
  blended by soft probabilities).  Detected via ``p_ign`` in ``mask_out.aux``.

Extracted from hacky_visualize_goldilocks.py so that both the standalone
script and the VizCallback can share the same logic.
"""
from __future__ import annotations

import inspect
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


def _null_interclass_std(p: float, n_min: int) -> float:
    """Expected inter-class std under uniform random assignment.

    Each patch has probability *p* of being assigned to a role. With *n_min*
    images per class, the per-position sample mean has std √(p(1-p)/n_min).
    Averaging across positions doesn't change the expectation (positions are
    identically distributed), so this is the null baseline for inter-class std.

    Values well above this indicate content-adaptive scoring.
    """
    if n_min < 1:
        return 0.0
    import math
    return math.sqrt(p * (1.0 - p) / n_min)


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
    ctx_seed_idx: Optional[int] = None,
    tgt_seed_idx: Optional[int] = None,
    seed_alpha: float = 1.0,
) -> np.ndarray:
    """Hard assignment: ctx=blue, tgt=red, ignored=grey.

    When ctx_seed_idx / tgt_seed_idx are provided, those patches are repainted
    with a stronger alpha so RandomRegionGrowth seeds are easy to spot.
    """
    out = orig_np.copy().astype(float)
    gh, gw = bin_map.shape
    for i in range(gh):
        for j in range(gw):
            b = bin_map[i, j]
            flat_idx = i * gw + j
            y0, y1 = i * patch_size, (i + 1) * patch_size
            x0, x1 = j * patch_size, (j + 1) * patch_size
            alpha_ij = alpha
            if ctx_seed_idx is not None and flat_idx == int(ctx_seed_idx):
                col = np.array(CTX_RGB, dtype=float)
                alpha_ij = seed_alpha
            elif tgt_seed_idx is not None and flat_idx == int(tgt_seed_idx):
                col = np.array(TGT_RGB, dtype=float)
                alpha_ij = seed_alpha
            elif b == 0:
                col = np.array(CTX_RGB, dtype=float)
            elif b == 1:
                col = np.array(TGT_RGB, dtype=float)
            else:
                col = np.array(GREY, dtype=float)
            orig_patch = orig_np[y0:y1, x0:x1].astype(float)
            blended = (1.0 - alpha_ij) * orig_patch + alpha_ij * col
            out[y0:y1, x0:x1] = blended
    return out.clip(0, 255).astype(np.uint8)


def apply_soft_3way_overlay(
    orig_np: np.ndarray,
    p_ctx_map: np.ndarray,
    p_tgt_map: np.ndarray,
    p_ign_map: np.ndarray,
    patch_size: int,
    alpha_max: float = ALPHA_SCORE,
) -> np.ndarray:
    """Soft 3-way overlay: winner's color at intensity proportional to its prob.

    Each patch is colored by the most probable role (ctx=blue, tgt=red, ign=grey).
    The blend alpha scales with the winning probability: 100% tgt = fully red,
    34% ctx = faintly blue (near the desaturated original).
    """
    grey = np.dot(orig_np.astype(float), [0.299, 0.587, 0.114])
    grey3 = np.stack([grey, grey, grey], axis=-1)
    out = grey3.copy()
    colors = [np.array(CTX_RGB, dtype=float),
              np.array(TGT_RGB, dtype=float),
              np.array(GREY, dtype=float)]
    gh, gw = p_ctx_map.shape
    for i in range(gh):
        for j in range(gw):
            probs = [float(p_ctx_map[i, j]),
                     float(p_tgt_map[i, j]),
                     float(p_ign_map[i, j])]
            winner = int(np.argmax(probs))
            col = colors[winner]
            intensity = probs[winner] * alpha_max
            y0, y1 = i * patch_size, (i + 1) * patch_size
            x0, x1 = j * patch_size, (j + 1) * patch_size
            patch = grey3[y0:y1, x0:x1]
            out[y0:y1, x0:x1] = ((1 - intensity) * patch + intensity * col).clip(0, 255)
    return out.astype(np.uint8)


def apply_soft_nway_overlay(
    orig_np: np.ndarray,
    soft_map: np.ndarray,
    patch_size: int,
    num_tgt_blocks: int,
    alpha_max: float = ALPHA_SCORE,
) -> np.ndarray:
    """Soft N-way overlay: winner's color at intensity proportional to its prob.

    Parameters
    ----------
    soft_map  : (gh, gw, M+2) array of per-patch probabilities.
    """
    grey_img = np.dot(orig_np.astype(float), [0.299, 0.587, 0.114])
    grey3 = np.stack([grey_img, grey_img, grey_img], axis=-1)
    out = grey3.copy()

    M = num_tgt_blocks
    colors = [np.array(CTX_RGB, dtype=float)]
    for k in range(M):
        colors.append(np.array(_BLOCK_COLORS[k % len(_BLOCK_COLORS)], dtype=float))
    colors.append(np.array(GREY, dtype=float))

    gh, gw = soft_map.shape[:2]
    for i in range(gh):
        for j in range(gw):
            probs = soft_map[i, j]
            winner = int(np.argmax(probs))
            col = colors[winner]
            intensity = float(probs[winner]) * alpha_max
            y0, y1 = i * patch_size, (i + 1) * patch_size
            x0, x1 = j * patch_size, (j + 1) * patch_size
            patch = grey3[y0:y1, x0:x1]
            out[y0:y1, x0:x1] = ((1 - intensity) * patch + intensity * col).clip(0, 255)
    return out.astype(np.uint8)


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
    verbose: bool = True,
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
    if verbose:
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
    verbose: bool = True,
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
    # Null baseline: expected std under uniform scoring with min per-class n
    n_min = min(class_n[c] for c in sorted_cls) if sorted_cls else 1
    p_expected = float(overall_sums.sum()) / max(overall_n, 1) / max(gh * gw, 1)
    null_std = _null_interclass_std(p_expected, n_min)
    if verbose:
        print(f"  Inter-class score std = {interclass_std:.4f}  "
              f"(null={null_std:.4f}; well above -> content-adaptive scoring)")

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
    if verbose:
        print(f"  Saved per-class score heatmap -> {out_path}  "
              f"({len(sorted_cls)} classes, overall n={overall_n})")


# ---------------------------------------------------------------------------
# 3-way aggregate heatmaps (MI masker)
# ---------------------------------------------------------------------------

def save_avg_3way_heatmap(
    role_sums: dict[str, np.ndarray],
    n_images: int,
    out_path: Path,
    patch_px: int = 40,
    verbose: bool = True,
) -> None:
    """Four-panel heatmap: mean p_ctx / p_tgt / p_ign / blended per position."""
    gh, gw = role_sums["ctx"].shape
    means = {k: v / max(n_images, 1) for k, v in role_sums.items()}

    cell_h = gh * patch_px
    cell_w = gw * patch_px
    label_h = 24
    gap = 8
    n_panels = 4
    total_w = n_panels * cell_w + (n_panels - 1) * gap
    total_h = cell_h + label_h

    img = Image.new("RGB", (total_w, total_h), (20, 20, 20))
    draw = ImageDraw.Draw(img)

    try:
        from PIL import ImageFont
        font = ImageFont.load_default()
    except Exception:
        font = None

    panels = [
        ("p_ctx (blue=high)", means["ctx"]),
        ("p_tgt (blue=high)", means["tgt"]),
        ("p_ign (blue=high)", means["ign"]),
    ]

    ctx_col = np.array(CTX_RGB, dtype=float)
    tgt_col = np.array(TGT_RGB, dtype=float)
    ign_col = np.array(GREY, dtype=float)

    for pi, (title, mean_map) in enumerate(panels):
        x_off = pi * (cell_w + gap)
        draw.text((x_off + 4, 4), title, fill=(200, 200, 200), font=font)
        for i in range(gh):
            for j in range(gw):
                col = score_color(float(mean_map[i, j]))
                x0 = x_off + j * patch_px
                y0 = label_h + i * patch_px
                draw.rectangle([x0, y0, x0 + patch_px - 1, y0 + patch_px - 1],
                               fill=col, outline=(40, 40, 40))

    # Winner panel: color of most probable role, intensity = winning prob
    colors = [ctx_col, tgt_col, ign_col]
    bg = np.array((20, 20, 20), dtype=float)
    x_off = 3 * (cell_w + gap)
    draw.text((x_off + 4, 4), "winner (ctx/tgt/ign)", fill=(200, 200, 200), font=font)
    for i in range(gh):
        for j in range(gw):
            probs = [means["ctx"][i, j], means["tgt"][i, j], means["ign"][i, j]]
            winner = int(np.argmax(probs))
            intensity = probs[winner]
            col = tuple(int(c) for c in np.clip(
                (1 - intensity) * bg + intensity * colors[winner], 0, 255))
            x0 = x_off + j * patch_px
            y0 = label_h + i * patch_px
            draw.rectangle([x0, y0, x0 + patch_px - 1, y0 + patch_px - 1],
                           fill=col, outline=(40, 40, 40))

    img.save(out_path)
    if verbose:
        print(f"  Saved 3-way heatmap -> {out_path}  (n={n_images})")


def save_class_3way_heatmap(
    overall_sums: dict[str, np.ndarray],
    overall_n: int,
    class_sums: dict[int, dict[str, np.ndarray]],
    class_n: dict[int, int],
    class_names: list,
    out_path: Path,
    patch_px: int = 32,
    label_w: int = 140,
    row_gap: int = 6,
    verbose: bool = True,
) -> None:
    """Per-class blended 3-way heatmap (ctx=blue, tgt=red, ign=grey)."""
    gh, gw = overall_sums["ctx"].shape
    panel_w = gw * patch_px
    row_h = gh * patch_px
    row_stride = row_h + row_gap
    header_h = 22

    ctx_col = np.array(CTX_RGB, dtype=float)
    tgt_col = np.array(TGT_RGB, dtype=float)
    ign_col = np.array(GREY, dtype=float)
    colors = [ctx_col, tgt_col, ign_col]
    bg = np.array((20, 20, 20), dtype=float)

    sorted_cls = sorted(class_sums.keys())

    def _blend(sums_dict, n_img):
        n_img = max(n_img, 1)
        mc = sums_dict["ctx"] / n_img
        mt = sums_dict["tgt"] / n_img
        mi = sums_dict["ign"] / n_img
        return mc, mt, mi

    rows = [("overall", *_blend(overall_sums, overall_n))] + [
        (class_names[c] if class_names and c < len(class_names) else str(c),
         *_blend(class_sums[c], class_n[c]))
        for c in sorted_cls
    ]

    # Inter-class std on p_tgt
    interclass_std = float("nan")
    if len(sorted_cls) >= 2:
        class_means = np.stack(
            [class_sums[c]["tgt"] / max(class_n[c], 1) for c in sorted_cls]
        )
        interclass_std = float(class_means.std(axis=0).mean())
    # Null baseline: p_tgt ≈ 1/3 for 3-way uniform
    n_min = min(class_n[c] for c in sorted_cls) if sorted_cls else 1
    null_std = _null_interclass_std(1.0 / 3.0, n_min)
    if verbose:
        print(f"  Inter-class p_tgt std = {interclass_std:.4f}  "
              f"(null={null_std:.4f}; well above -> content-adaptive scoring)")

    total_w = label_w + panel_w
    total_h = header_h + len(rows) * row_stride - row_gap
    img = Image.new("RGB", (total_w, total_h), (20, 20, 20))
    draw = ImageDraw.Draw(img)

    try:
        from PIL import ImageFont
        font = ImageFont.load_default()
    except Exception:
        font = None

    draw.text((label_w + 4, 4),
              f"blended 3-way  (inter-class p_tgt std={interclass_std:.4f})",
              fill=(200, 200, 200), font=font)

    for row_idx, (name, mc, mt, mi) in enumerate(rows):
        y0 = header_h + row_idx * row_stride

        if row_idx > 0:
            draw.rectangle([0, y0 - row_gap, total_w, y0 - 1], fill=(50, 50, 50))

        n_img = overall_n if row_idx == 0 else class_n[sorted_cls[row_idx - 1]]
        draw.text((4, y0 + row_h // 2 - 5),
                  f"{name}\n(n={n_img})", fill=(200, 200, 200), font=font)

        for i in range(gh):
            for j in range(gw):
                probs = [mc[i, j], mt[i, j], mi[i, j]]
                winner = int(np.argmax(probs))
                intensity = probs[winner]
                col = tuple(int(c) for c in np.clip(
                    (1 - intensity) * bg + intensity * colors[winner], 0, 255))
                x0 = label_w + j * patch_px
                y1 = y0 + i * patch_px
                draw.rectangle([x0, y1, x0 + patch_px - 1, y1 + patch_px - 1],
                               fill=col, outline=(40, 40, 40))

    img.save(out_path)
    if verbose:
        print(f"  Saved per-class 3-way heatmap -> {out_path}  "
              f"({len(sorted_cls)} classes, overall n={overall_n})")


# ---------------------------------------------------------------------------
# Multiblock-specific overlays and visualization
# ---------------------------------------------------------------------------

# Fixed per-block color palette (M up to 8)
_BLOCK_COLORS = [
    (220,  60,  60),   # red
    (255, 165,   0),   # orange
    ( 60, 160,  60),   # green
    (140,  60, 200),   # purple
    (  0, 200, 200),   # cyan
    (200, 200,   0),   # yellow
    (255, 105, 180),   # pink
    (100, 100, 220),   # blue-violet
]


def apply_multiblock_overlay(
    orig_np: np.ndarray,
    context_idx: np.ndarray,
    target_idx_3d: np.ndarray,
    patch_size: int,
    image_size: int,
    alpha: float = ALPHA_ASSIGN,
    ctx_seed_idx: Optional[int] = None,
    tgt_seed_idx_blocks: Optional[np.ndarray] = None,
    seed_alpha: float = 1.0,
) -> np.ndarray:
    """Per-block colored overlay: ctx=blue, each target block k→distinct color, ignored=grey.

    Parameters
    ----------
    context_idx  : (Nctx,) int array of context patch flat indices
    target_idx_3d: (M, K) int array of target block patch flat indices
    """
    N = (image_size // patch_size) ** 2
    gh = gw = image_size // patch_size

    # Build role map: -1=ignored, 0=ctx, 1..M=target block k
    role = np.full(N, -1, dtype=np.int32)
    role[context_idx] = 0
    M = target_idx_3d.shape[0]
    for k in range(M):
        role[target_idx_3d[k]] = k + 1

    role_map = role.reshape(gh, gw)
    out = orig_np.copy().astype(float)
    tgt_seed_to_block: dict[int, int] = {}
    if tgt_seed_idx_blocks is not None:
        for k, seed_idx in enumerate(np.asarray(tgt_seed_idx_blocks).reshape(-1).tolist()):
            tgt_seed_to_block[int(seed_idx)] = k

    for i in range(gh):
        for j in range(gw):
            r = role_map[i, j]
            flat_idx = i * gw + j
            alpha_ij = alpha
            if ctx_seed_idx is not None and flat_idx == int(ctx_seed_idx):
                col = np.array(CTX_RGB, dtype=float)
                alpha_ij = seed_alpha
            elif flat_idx in tgt_seed_to_block:
                block_idx = tgt_seed_to_block[flat_idx]
                col = np.array(_BLOCK_COLORS[block_idx % len(_BLOCK_COLORS)], dtype=float)
                alpha_ij = seed_alpha
            elif r == -1:
                col = np.array(GREY, dtype=float)
            elif r == 0:
                col = np.array(CTX_RGB, dtype=float)
            else:
                col = np.array(_BLOCK_COLORS[(r - 1) % len(_BLOCK_COLORS)], dtype=float)
            y0, y1 = i * patch_size, (i + 1) * patch_size
            x0, x1 = j * patch_size, (j + 1) * patch_size
            orig_patch = orig_np[y0:y1, x0:x1].astype(float)
            out[y0:y1, x0:x1] = (1.0 - alpha_ij) * orig_patch + alpha_ij * col

    return out.clip(0, 255).astype(np.uint8)


@torch.no_grad()
def visualize_split_multiblock(
    dataset_name: str,
    split: str,
    dataset,
    masker,
    out_dir: Path,
    n: int,
    grid_cols: int,
    patch_size: int,
    image_size: int,
    batch_size: int = BATCH_SIZE,
    verbose: bool = True,
) -> tuple:
    """Visualize multiblock masker (content-independent).

    Does NOT require an encoder — calls ``masker(B)`` directly.

    Returns
    -------
    (sums, cls_sums, cls_n)
    sums     = {"ctx": (gh,gw) float64, "tgt": (gh,gw) float64}
    cls_sums = {label: {"ctx":..., "tgt":...}}
    cls_n    = {label: int}
    """
    from torchvision import transforms

    _resize_crop = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
    ])

    n = min(n, len(dataset))
    gh = gw = image_size // patch_size
    N = gh * gw

    cells = []
    _zero = lambda: np.zeros((gh, gw), dtype=np.float64)
    sums = {"ctx": _zero(), "tgt": _zero()}
    cls_sums: dict = {}
    cls_n: dict = {}

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        idxs = range(start, end)
        B = end - start

        displays, labels_batch = [], []
        for i in idxs:
            img_raw, lbl = dataset[i]
            if isinstance(img_raw, np.ndarray):
                if img_raw.ndim == 3 and img_raw.shape[0] == 3:
                    img_raw = np.transpose(img_raw, (1, 2, 0))
                pil = Image.fromarray(img_raw.astype(np.uint8))
            else:
                pil = img_raw
            pil_crop = _resize_crop(pil.convert("RGB"))
            displays.append(np.array(pil_crop, dtype=np.uint8))
            labels_batch.append(int(lbl))

        mask_out = masker(B)

        ctx_idx  = mask_out.context_idx.cpu().numpy()   # (B, Nctx)
        tgt_idx  = mask_out.target_idx.cpu().numpy()    # (B, M, K)

        for bi in range(B):
            ci = ctx_idx[bi]           # (Nctx,)
            ti = tgt_idx[bi]           # (M, K)
            lbl = labels_batch[bi]

            # Accumulate frequency maps
            ctx_freq = np.zeros(N, dtype=np.float64)
            ctx_freq[ci] = 1.0
            tgt_freq = np.zeros(N, dtype=np.float64)
            tgt_freq[ti.ravel()] = 1.0

            sums["ctx"] += ctx_freq.reshape(gh, gw)
            sums["tgt"] += tgt_freq.reshape(gh, gw)

            if lbl not in cls_sums:
                cls_sums[lbl] = {"ctx": _zero(), "tgt": _zero()}
                cls_n[lbl] = 0
            cls_sums[lbl]["ctx"] += ctx_freq.reshape(gh, gw)
            cls_sums[lbl]["tgt"] += tgt_freq.reshape(gh, gw)
            cls_n[lbl] += 1

            orig_np = displays[bi]

            # Panel 2: binary assignment (ctx=blue, all tgt=red, ign=grey)
            bin_flat = np.full(N, 2, dtype=np.int64)
            bin_flat[ci] = 0
            bin_flat[ti.ravel()] = 1
            bin_map = bin_flat.reshape(gh, gw)
            assign_np = apply_assignment_overlay(orig_np, bin_map, patch_size)

            # Panel 3: per-block colored overlay
            multiblock_np = apply_multiblock_overlay(
                orig_np, ci, ti, patch_size, image_size
            )

            cells.append(make_cell(orig_np, assign_np, multiblock_np))

        if verbose:
            print(f"  {dataset_name}/{split}: {end}/{n}", end="\r")

    if verbose:
        print()

    if cells:
        cell_w, cell_h = cells[0].size
        n_rows = (len(cells) + grid_cols - 1) // grid_cols
        grid = Image.new("RGB", (grid_cols * cell_w, n_rows * cell_h), (30, 30, 30))
        for k, cell in enumerate(cells):
            r, c = divmod(k, grid_cols)
            grid.paste(cell, (c * cell_w, r * cell_h))
        fname = out_dir / f"{dataset_name}_{split}_multiblock.png"
        grid.save(fname)
        if verbose:
            print(f"  Saved -> {fname}  [{len(cells)} images]")

    return sums, cls_sums, cls_n


def save_avg_coverage_heatmap(
    sums: dict,
    n: int,
    path: Path,
    patch_px: int = 40,
    verbose: bool = True,
) -> None:
    """Two-panel heatmap: ctx frequency | tgt frequency (cold→hot)."""
    gh, gw = sums["ctx"].shape
    means = {k: v / max(n, 1) for k, v in sums.items()}

    cell_h = gh * patch_px
    cell_w = gw * patch_px
    label_h = 24
    gap = 8
    total_w = 2 * cell_w + gap
    total_h = cell_h + label_h

    img = Image.new("RGB", (total_w, total_h), (20, 20, 20))
    draw = ImageDraw.Draw(img)

    try:
        from PIL import ImageFont
        font = ImageFont.load_default()
    except Exception:
        font = None

    panels = [("ctx frequency", means["ctx"], 0),
              ("tgt frequency", means["tgt"], cell_w + gap)]

    for title, mean_map, x_off in panels:
        draw.text((x_off + 4, 4), title, fill=(200, 200, 200), font=font)
        for i in range(gh):
            for j in range(gw):
                col = score_color(float(mean_map[i, j]))
                x0 = x_off + j * patch_px
                y0 = label_h + i * patch_px
                draw.rectangle([x0, y0, x0 + patch_px - 1, y0 + patch_px - 1],
                               fill=col, outline=(40, 40, 40))

    img.save(path)
    if verbose:
        print(f"  Saved coverage heatmap -> {path}  (n={n})")


def save_class_coverage_heatmap(
    sums: dict,
    n: int,
    cls_sums: dict,
    cls_n: dict,
    class_names: list,
    path: Path,
    patch_px: int = 32,
    label_w: int = 140,
    row_gap: int = 6,
    verbose: bool = True,
) -> None:
    """Per-class tgt coverage heatmap (should be uniform for geometric masker)."""
    gh, gw = sums["tgt"].shape
    panel_w = gw * patch_px
    row_h = gh * patch_px
    row_stride = row_h + row_gap
    header_h = 22

    sorted_cls = sorted(cls_sums.keys())

    interclass_std = float("nan")
    if len(sorted_cls) >= 2:
        class_means = np.stack(
            [cls_sums[c]["tgt"] / max(cls_n[c], 1) for c in sorted_cls]
        )
        interclass_std = float(class_means.std(axis=0).mean())
    if verbose:
        print(f"  Inter-class tgt std = {interclass_std:.4f}  "
              f"(~0 expected for geometric/content-independent masker)")

    rows = [("overall", sums["tgt"] / max(n, 1), n)] + [
        (class_names[c] if class_names and c < len(class_names) else str(c),
         cls_sums[c]["tgt"] / max(cls_n[c], 1), cls_n[c])
        for c in sorted_cls
    ]

    total_w = label_w + panel_w
    total_h = header_h + len(rows) * row_stride - row_gap
    img = Image.new("RGB", (total_w, total_h), (20, 20, 20))
    draw = ImageDraw.Draw(img)

    try:
        from PIL import ImageFont
        font = ImageFont.load_default()
    except Exception:
        font = None

    draw.text((label_w + 4, 4),
              f"tgt coverage  (inter-class std={interclass_std:.4f})",
              fill=(200, 200, 200), font=font)

    for row_idx, (name, mean_map, n_img) in enumerate(rows):
        y0 = header_h + row_idx * row_stride
        if row_idx > 0:
            draw.rectangle([0, y0 - row_gap, total_w, y0 - 1], fill=(50, 50, 50))
        draw.text((4, y0 + row_h // 2 - 5),
                  f"{name}\n(n={n_img})", fill=(200, 200, 200), font=font)
        for i in range(gh):
            for j in range(gw):
                col = score_color(float(mean_map[i, j]))
                x0 = label_w + j * patch_px
                y1 = y0 + i * patch_px
                draw.rectangle([x0, y1, x0 + patch_px - 1, y1 + patch_px - 1],
                               fill=col, outline=(40, 40, 40))

    img.save(path)
    if verbose:
        print(f"  Saved per-class coverage heatmap -> {path}  "
              f"({len(sorted_cls)} classes, overall n={n})")


# ---------------------------------------------------------------------------
# N-way aggregate heatmaps
# ---------------------------------------------------------------------------

def save_avg_nway_heatmap(
    role_sums: dict[str, np.ndarray],
    n_images: int,
    out_path: Path,
    num_tgt_blocks: int = 4,
    patch_px: int = 40,
    verbose: bool = True,
) -> None:
    """(M+3)-panel heatmap: p_ctx, p_tgt_0..M-1, p_ign, winner."""
    M = num_tgt_blocks
    gh, gw = role_sums["ctx"].shape
    means = {k: v / max(n_images, 1) for k, v in role_sums.items()}

    cell_h = gh * patch_px
    cell_w = gw * patch_px
    label_h = 24
    gap = 8
    n_panels = M + 3  # ctx + M tgts + ign + winner
    total_w = n_panels * cell_w + (n_panels - 1) * gap
    total_h = cell_h + label_h

    img = Image.new("RGB", (total_w, total_h), (20, 20, 20))
    draw = ImageDraw.Draw(img)

    try:
        from PIL import ImageFont
        font = ImageFont.load_default()
    except Exception:
        font = None

    panels = [("p_ctx", means["ctx"])]
    for k in range(M):
        panels.append((f"p_tgt_{k}", means[f"tgt_{k}"]))
    panels.append(("p_ign", means["ign"]))

    for pi, (title, mean_map) in enumerate(panels):
        x_off = pi * (cell_w + gap)
        draw.text((x_off + 4, 4), title, fill=(200, 200, 200), font=font)
        for i in range(gh):
            for j in range(gw):
                col = score_color(float(mean_map[i, j]))
                x0 = x_off + j * patch_px
                y0 = label_h + i * patch_px
                draw.rectangle([x0, y0, x0 + patch_px - 1, y0 + patch_px - 1],
                               fill=col, outline=(40, 40, 40))

    # Winner panel
    ctx_col = np.array(CTX_RGB, dtype=float)
    ign_col = np.array(GREY, dtype=float)
    colors = [ctx_col]
    for k in range(M):
        colors.append(np.array(_BLOCK_COLORS[k % len(_BLOCK_COLORS)], dtype=float))
    colors.append(ign_col)
    bg = np.array((20, 20, 20), dtype=float)

    x_off = (M + 2) * (cell_w + gap)
    draw.text((x_off + 4, 4), "winner", fill=(200, 200, 200), font=font)
    all_keys = ["ctx"] + [f"tgt_{k}" for k in range(M)] + ["ign"]
    for i in range(gh):
        for j in range(gw):
            probs = [means[key][i, j] for key in all_keys]
            winner = int(np.argmax(probs))
            intensity = probs[winner]
            col = tuple(int(c) for c in np.clip(
                (1 - intensity) * bg + intensity * colors[winner], 0, 255))
            x0 = x_off + j * patch_px
            y0 = label_h + i * patch_px
            draw.rectangle([x0, y0, x0 + patch_px - 1, y0 + patch_px - 1],
                           fill=col, outline=(40, 40, 40))

    img.save(out_path)
    if verbose:
        print(f"  Saved N-way heatmap -> {out_path}  (n={n_images})")


def save_class_nway_heatmap(
    overall_sums: dict[str, np.ndarray],
    overall_n: int,
    class_sums: dict[int, dict[str, np.ndarray]],
    class_n: dict[int, int],
    class_names: list,
    out_path: Path,
    num_tgt_blocks: int = 4,
    patch_px: int = 32,
    label_w: int = 140,
    row_gap: int = 6,
    verbose: bool = True,
) -> None:
    """Per-class blended N-way heatmap (winner color at winning prob intensity)."""
    M = num_tgt_blocks
    all_keys = ["ctx"] + [f"tgt_{k}" for k in range(M)] + ["ign"]
    gh, gw = overall_sums["ctx"].shape
    panel_w = gw * patch_px
    row_h = gh * patch_px
    row_stride = row_h + row_gap
    header_h = 22

    ctx_col = np.array(CTX_RGB, dtype=float)
    ign_col = np.array(GREY, dtype=float)
    colors = [ctx_col]
    for k in range(M):
        colors.append(np.array(_BLOCK_COLORS[k % len(_BLOCK_COLORS)], dtype=float))
    colors.append(ign_col)
    bg = np.array((20, 20, 20), dtype=float)

    sorted_cls = sorted(class_sums.keys())

    def _means(sums_dict, n_img):
        n_img = max(n_img, 1)
        return {key: sums_dict[key] / n_img for key in all_keys}

    rows = [("overall", _means(overall_sums, overall_n))] + [
        (class_names[c] if class_names and c < len(class_names) else str(c),
         _means(class_sums[c], class_n[c]))
        for c in sorted_cls
    ]

    # Inter-class std on total target mass (sum of M block probs)
    interclass_std = float("nan")
    if len(sorted_cls) >= 2:
        tgt_keys = [f"tgt_{k}" for k in range(M)]
        class_tgt_means = np.stack([
            sum(class_sums[c][tk] / max(class_n[c], 1) for tk in tgt_keys)
            for c in sorted_cls
        ])
        interclass_std = float(class_tgt_means.std(axis=0).mean())
    # Null baseline: total tgt mass = M/(M+2) under uniform (M+2)-way
    n_min = min(class_n[c] for c in sorted_cls) if sorted_cls else 1
    p_tgt_total = M / (M + 2)
    null_std = _null_interclass_std(p_tgt_total, n_min)
    if verbose:
        print(f"  Inter-class p_tgt std = {interclass_std:.4f}  "
              f"(null={null_std:.4f}; well above -> content-adaptive scoring)")

    total_w = label_w + panel_w
    total_h = header_h + len(rows) * row_stride - row_gap
    img = Image.new("RGB", (total_w, total_h), (20, 20, 20))
    draw = ImageDraw.Draw(img)

    try:
        from PIL import ImageFont
        font = ImageFont.load_default()
    except Exception:
        font = None

    draw.text((label_w + 4, 4),
              f"blended N-way  (inter-class p_tgt std={interclass_std:.4f})",
              fill=(200, 200, 200), font=font)

    for row_idx, (name, m) in enumerate(rows):
        y0 = header_h + row_idx * row_stride
        if row_idx > 0:
            draw.rectangle([0, y0 - row_gap, total_w, y0 - 1], fill=(50, 50, 50))

        n_img = overall_n if row_idx == 0 else class_n[sorted_cls[row_idx - 1]]
        draw.text((4, y0 + row_h // 2 - 5),
                  f"{name}\n(n={n_img})", fill=(200, 200, 200), font=font)

        for i in range(gh):
            for j in range(gw):
                probs = [m[key][i, j] for key in all_keys]
                winner = int(np.argmax(probs))
                intensity = probs[winner]
                col = tuple(int(c) for c in np.clip(
                    (1 - intensity) * bg + intensity * colors[winner], 0, 255))
                x0 = label_w + j * patch_px
                y1 = y0 + i * patch_px
                draw.rectangle([x0, y1, x0 + patch_px - 1, y1 + patch_px - 1],
                               fill=col, outline=(40, 40, 40))

    img.save(out_path)
    if verbose:
        print(f"  Saved per-class N-way heatmap -> {out_path}  "
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
    epoch: Optional[int] = None,
    verbose: bool = True,
) -> tuple:
    """
    Run masker on ``n`` images from ``dataset`` and produce visualization grids.

    Automatically detects masker type:
    - N-way: ``soft`` tensor in aux with shape[-1] > 3
    - 3-way: ``p_ign`` in aux (but no N-way soft)
    - 2-way: everything else

    Returns
    -------
    For 2-way (Goldilocks):
        (score_sums, class_score_sums, class_img_n, "2way")
    For 3-way (MI):
        (role_sums, class_role_sums, class_img_n, "3way")
    For N-way (MI N-way):
        (role_sums, class_role_sums, class_img_n, "nway")

    where role_sums keys depend on type.
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
    all_p_tgt = []
    class_img_n: dict = {}

    # Accumulators — initialised lazily after first batch reveals masker type
    masker_type: Optional[str] = None  # "2way", "3way", "nway"
    nway_M: int = 0
    # 2-way accumulators
    score_sums = np.zeros((gh, gw), dtype=np.float64)
    class_score_sums: dict = {}
    # 3-way / N-way accumulators
    _zero = lambda: np.zeros((gh, gw), dtype=np.float64)
    role_sums: dict[str, np.ndarray] = {}
    class_role_sums: dict[int, dict[str, np.ndarray]] = {}

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

        forward_sig = inspect.signature(masker.forward)
        latent_kwargs = {}
        if "epoch" in forward_sig.parameters:
            latent_kwargs["epoch"] = epoch
        mask_out = masker(tokens, ema_full=tokens, **latent_kwargs)

        ctx_idx = mask_out.context_idx
        tgt_idx = mask_out.target_idx
        p_tgt = mask_out.target_soft
        p_ctx = mask_out.context_soft
        p_ign = mask_out.aux.get("p_ign")
        soft_nway = mask_out.aux.get("soft")  # (B, N, M+2) for N-way
        rrg_ctx_seed_idx = mask_out.aux.get("rrg_ctx_seed_idx")
        rrg_tgt_seed_idx = mask_out.aux.get("rrg_tgt_seed_idx")
        rrg_tgt_seed_idx_blocks = mask_out.aux.get("rrg_tgt_seed_idx_blocks")

        # Detect masker type on first batch
        if masker_type is None:
            is_nway = (soft_nway is not None
                       and torch.is_tensor(soft_nway)
                       and soft_nway.shape[-1] > 3)
            if is_nway:
                masker_type = "nway"
                nway_M = soft_nway.shape[-1] - 2
                role_keys = ["ctx"] + [f"tgt_{k}" for k in range(nway_M)] + ["ign"]
                role_sums = {k: _zero() for k in role_keys}
            elif p_ign is not None and p_ctx is not None:
                masker_type = "3way"
                role_sums = {"ctx": _zero(), "tgt": _zero(), "ign": _zero()}
            else:
                masker_type = "2way"

        if p_tgt is not None:
            all_p_tgt.append(p_tgt.cpu())

        ctx_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        tgt_flat = tgt_idx.reshape(B, -1) if tgt_idx.dim() == 3 else tgt_idx
        tgt_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        ctx_mask.scatter_(1, ctx_idx, True)
        tgt_mask.scatter_(1, tgt_flat, True)

        for bi in range(B):
            lbl = labels_batch[bi]
            if lbl not in class_img_n:
                class_img_n[lbl] = 0
            class_img_n[lbl] += 1

            orig_np = displays[bi]

            if masker_type == "nway":
                # N-way: middle panel = soft N-way overlay
                soft_bi = soft_nway[bi].detach().cpu().reshape(gh, gw, -1).numpy()
                mid_np = apply_soft_nway_overlay(
                    orig_np, soft_bi, patch_size, nway_M)

                # Hard assignment panel: per-block colored overlay
                assign_np = apply_multiblock_overlay(
                    orig_np,
                    ctx_idx[bi].cpu().numpy(),
                    tgt_idx[bi].cpu().numpy(),  # (M, K)
                    patch_size, image_size,
                )

                # Accumulate per-role sums
                role_keys = ["ctx"] + [f"tgt_{k}" for k in range(nway_M)] + ["ign"]
                for ci_k, key in enumerate(role_keys):
                    ch = soft_bi[..., ci_k].astype(np.float64)
                    role_sums[key] += ch
                    if lbl not in class_role_sums:
                        class_role_sums[lbl] = {k: _zero() for k in role_keys}
                    class_role_sums[lbl][key] += ch

            elif masker_type == "3way":
                # 3-way: middle panel = soft 3-way overlay
                bin_flat = torch.full((N,), 2, dtype=torch.long)
                bin_flat[ctx_mask[bi].cpu()] = 0
                bin_flat[tgt_mask[bi].cpu()] = 1
                bin_map = bin_flat.reshape(gh, gw).numpy()

                pc = p_ctx[bi].cpu().reshape(gh, gw).numpy()
                pt = p_tgt[bi].cpu().reshape(gh, gw).numpy()
                pi = p_ign[bi].detach().cpu().reshape(gh, gw).numpy()
                mid_np = apply_soft_3way_overlay(orig_np, pc, pt, pi, patch_size)
                if tgt_idx.dim() == 3:
                    ctx_seed = None if rrg_ctx_seed_idx is None else int(rrg_ctx_seed_idx[bi].item())
                    tgt_seed_blocks = None if rrg_tgt_seed_idx_blocks is None else (
                        rrg_tgt_seed_idx_blocks[bi].detach().cpu().numpy()
                    )
                    assign_np = apply_multiblock_overlay(
                        orig_np,
                        ctx_idx[bi].cpu().numpy(),
                        tgt_idx[bi].cpu().numpy(),
                        patch_size,
                        image_size,
                        ctx_seed_idx=ctx_seed,
                        tgt_seed_idx_blocks=tgt_seed_blocks,
                    )
                else:
                    ctx_seed = None if rrg_ctx_seed_idx is None else int(rrg_ctx_seed_idx[bi].item())
                    tgt_seed = None if rrg_tgt_seed_idx is None else int(rrg_tgt_seed_idx[bi].item())
                    assign_np = apply_assignment_overlay(
                        orig_np,
                        bin_map,
                        patch_size,
                        ctx_seed_idx=ctx_seed,
                        tgt_seed_idx=tgt_seed,
                    )

                role_sums["ctx"] += pc.astype(np.float64)
                role_sums["tgt"] += pt.astype(np.float64)
                role_sums["ign"] += pi.astype(np.float64)
                if lbl not in class_role_sums:
                    class_role_sums[lbl] = {
                        "ctx": _zero(), "tgt": _zero(), "ign": _zero(),
                    }
                class_role_sums[lbl]["ctx"] += pc.astype(np.float64)
                class_role_sums[lbl]["tgt"] += pt.astype(np.float64)
                class_role_sums[lbl]["ign"] += pi.astype(np.float64)
            else:
                # 2-way: middle panel = cold→hot p_tgt heatmap
                bin_flat = torch.full((N,), 2, dtype=torch.long)
                bin_flat[ctx_mask[bi].cpu()] = 0
                bin_flat[tgt_mask[bi].cpu()] = 1
                bin_map = bin_flat.reshape(gh, gw).numpy()

                if p_tgt is not None:
                    score_map = p_tgt[bi].cpu().reshape(gh, gw).numpy()
                else:
                    score_map = (bin_map == 1).astype(np.float32)
                mid_np = apply_score_heatmap(orig_np, score_map, patch_size)
                assign_np = apply_assignment_overlay(orig_np, bin_map, patch_size)

                score_sums += score_map.astype(np.float64)
                if lbl not in class_score_sums:
                    class_score_sums[lbl] = np.zeros((gh, gw), dtype=np.float64)
                class_score_sums[lbl] += score_map.astype(np.float64)

            cells.append(make_cell(orig_np, mid_np, assign_np))

        if verbose:
            print(f"  {dataset_name}/{split}: {end}/{n}", end="\r")

    if verbose:
        print()

    # Numeric content-adaptivity report
    if all_p_tgt:
        p_cat = torch.cat(all_p_tgt, dim=0)  # (n_total, N)
        n_total = p_cat.shape[0]
        marginal_score_std = p_cat.mean(dim=0).std().item()
        p_tgt_score_std = p_cat.std(dim=0).mean().item()
        # Null baseline for p_tgt_score_std: per-position std of Bernoulli(p) samples
        # For soft assignments p is the mean target prob; with n_total images
        p_mean = float(p_cat.mean().item())
        null_score_std = _null_interclass_std(p_mean, n_total)
        if verbose:
            print(f"  marginal_score_std = {marginal_score_std:.4f}  "
                  f"(-> 0 = uniform marginal = no positional bias)")
            print(f"  p_tgt_score_std    = {p_tgt_score_std:.4f}  "
                  f"(null={null_score_std:.4f}; well above -> content-adaptive)")

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
        if verbose:
            print(f"  Saved -> {fname}   [{len(cells)} images, nctx={nctx} ntgt={ntgt} "
                  f"K/N={ntgt}/{N}={ntgt/N:.2f}]")

    if masker_type == "nway":
        return role_sums, class_role_sums, class_img_n, "nway"
    if masker_type == "3way":
        return role_sums, class_role_sums, class_img_n, "3way"
    return score_sums, class_score_sums, class_img_n, "2way"
