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
from PIL import Image
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
# Rendering utilities — imported from the reusable viz module
# ---------------------------------------------------------------------------
from ijepa_lite.viz.goldilocks_viz import (
    apply_assignment_overlay as _apply_assignment_overlay,
    apply_score_heatmap as _apply_score_heatmap,
    make_cell as _make_cell,
    save_avg_score_heatmap as _save_avg_score_heatmap_impl,
    save_class_score_heatmap as _save_class_score_heatmap_impl,
    score_color as _score_color,
)


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
# Wrappers delegating to ijepa_lite.viz.goldilocks_viz
# ---------------------------------------------------------------------------
from ijepa_lite.viz.goldilocks_viz import visualize_split as _visualize_split_impl


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


def _save_avg_score_heatmap(score_sums, n_images, out_path, patch_px=40):
    return _save_avg_score_heatmap_impl(score_sums, n_images, out_path, patch_px)


def _save_class_score_heatmap(overall_sums, overall_n, class_sums, class_n,
                               class_names, out_path, patch_px=32,
                               label_w=140, row_gap=6):
    return _save_class_score_heatmap_impl(
        overall_sums, overall_n, class_sums, class_n, class_names,
        out_path, patch_px, label_w, row_gap,
    )


@torch.no_grad()
def visualize_split(
    dataset_name, split, dataset, encoder, masker, device, out_dir, n, grid_cols,
):
    """Delegate to the reusable module, filling in script-level constants."""
    return _visualize_split_impl(
        dataset_name=dataset_name,
        split=split,
        dataset=dataset,
        encoder=encoder,
        masker=masker,
        device=device,
        out_dir=out_dir,
        n=n,
        grid_cols=grid_cols,
        patch_size=PATCH_SIZE,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )


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
