#!/usr/bin/env python3
"""
hacky_visualize_masker.py

Visualizes any LatentMasker's scoring and ctx/tgt patch assignments by
loading a checkpoint through the standard build pipeline.

Panels per image (left to right):
  [original | middle panel | hard assignment (ctx=blue, tgt=red, ignored=grey)]

Middle panel depends on masker type:
  - 2-way (Goldilocks): cold->hot heatmap of p_tgt
  - 3-way (MI):         soft 3-way overlay (ctx=blue, tgt=red, ign=grey)

Aggregate outputs:
  2-way: <out_dir>/<dataset>_avg_score.png, <dataset>_per_class_score.png
  3-way: <out_dir>/<dataset>_avg_3way.png,  <dataset>_per_class_3way.png

Usage:
  python hacky_visualize_goldilocks.py \\
      --ckpt /path/to/last.pt \\
      --experiment stl10_vits_ps8_mi \\
      [--data-root /path/to/datasets] [--out-dir masker_viz] [--n 500]

  Override eval k_tgt (Goldilocks only):
      --k-tgt 94
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets as tv_datasets

from ijepa_lite.viz.goldilocks_viz import (
    save_avg_3way_heatmap,
    save_avg_score_heatmap,
    save_class_3way_heatmap,
    save_class_score_heatmap,
    visualize_split,
)

GRID_COLS = 25


# ---------------------------------------------------------------------------
# Model loading via build pipeline
# ---------------------------------------------------------------------------

def _load_model(ckpt_path: str, experiment: str, device: torch.device):
    """Load encoder + masker from checkpoint using the Hydra config pipeline."""
    from omegaconf import OmegaConf
    from hydra import compose, initialize_config_dir

    # Resolve config dir
    config_dir = str(Path(__file__).resolve().parent / "configs")

    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(config_name="config", overrides=[f"experiment={experiment}"])

    from ijepa_lite.build import build_pretrain_model

    model = build_pretrain_model(cfg).to(device)

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = sd.get("model", sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [model] missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    model.requires_grad_(False)
    model.eval()

    encoder = model.target_encoder
    masker = model.latent_masker

    image_size = int(cfg.model.image_size)
    patch_size = int(cfg.model.patch_size)

    return encoder, masker, image_size, patch_size


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize any LatentMasker checkpoint."
    )
    parser.add_argument("--ckpt", default=os.environ.get("PRETRAIN_CKPT", ""),
                        help="Path to checkpoint file")
    parser.add_argument("--experiment", required=True,
                        help="Hydra experiment config name (e.g. stl10_vits_ps8_mi)")
    parser.add_argument("--data-root", default=os.environ.get("FAST", "/scratch") + "/datasets/")
    parser.add_argument("--out-dir", default="masker_viz")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--grid-cols", type=int, default=GRID_COLS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--k-tgt", type=int, default=None,
                        help="Override masker k_tgt_eval (Goldilocks only).")
    args = parser.parse_args()

    if not args.ckpt:
        raise SystemExit("Provide --ckpt or set PRETRAIN_CKPT env")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"Loading checkpoint: {args.ckpt}")
    print(f"Experiment config:  {args.experiment}")
    encoder, masker, image_size, patch_size = _load_model(
        args.ckpt, args.experiment, device,
    )

    if masker is None:
        raise SystemExit("No latent_masker found in the model — nothing to visualize.")

    if args.k_tgt is not None and hasattr(masker, "k_tgt_eval"):
        print(f"  Overriding k_tgt_eval: {masker.k_tgt_eval} -> {args.k_tgt}")
        masker.k_tgt_eval = args.k_tgt

    print(f"  image_size={image_size}  patch_size={patch_size}  "
          f"num_patches={(image_size // patch_size) ** 2}  device={device}")

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
        # Lazy accumulators — set after first split reveals masker type
        is_3way = None
        total_images = 0
        total_class_n: dict = {}
        class_names: list = []
        # 2-way
        total_score_sums = np.zeros((gh, gw), dtype=np.float64)
        total_class_score_sums: dict = {}
        # 3-way
        _zero = lambda: np.zeros((gh, gw), dtype=np.float64)
        total_role_sums = {"ctx": _zero(), "tgt": _zero(), "ign": _zero()}
        total_class_role_sums: dict = {}

        for split, loader_fn in splits:
            print(f"\n{name}/{split}")
            try:
                ds = loader_fn(args.data_root)
            except Exception as e:
                print(f"  skipped ({e})")
                continue

            sums, cls_sums, cls_n, three_way = visualize_split(
                dataset_name=name,
                split=split,
                dataset=ds,
                encoder=encoder,
                masker=masker,
                device=device,
                out_dir=out_dir,
                n=args.n,
                grid_cols=args.grid_cols,
                patch_size=patch_size,
                image_size=image_size,
            )

            if is_3way is None:
                is_3way = three_way

            total_images += min(args.n, len(ds))

            if is_3way:
                for role in ("ctx", "tgt", "ign"):
                    total_role_sums[role] += sums[role]
                for c, role_dict in cls_sums.items():
                    if c not in total_class_role_sums:
                        total_class_role_sums[c] = {
                            "ctx": _zero(), "tgt": _zero(), "ign": _zero(),
                        }
                    for role in ("ctx", "tgt", "ign"):
                        total_class_role_sums[c][role] += role_dict[role]
                    total_class_n[c] = total_class_n.get(c, 0) + cls_n[c]
            else:
                total_score_sums += sums
                for c, arr in cls_sums.items():
                    total_class_score_sums[c] = total_class_score_sums.get(
                        c, np.zeros_like(arr)) + arr
                    total_class_n[c] = total_class_n.get(c, 0) + cls_n[c]

            if not class_names and hasattr(ds, "classes"):
                class_names = list(ds.classes)

        if total_images > 0:
            if is_3way:
                save_avg_3way_heatmap(
                    total_role_sums, total_images,
                    out_dir / f"{name}_avg_3way.png",
                )
                if total_class_role_sums:
                    save_class_3way_heatmap(
                        total_role_sums, total_images,
                        total_class_role_sums, total_class_n,
                        class_names,
                        out_dir / f"{name}_per_class_3way.png",
                    )
            else:
                save_avg_score_heatmap(
                    total_score_sums, total_images,
                    out_dir / f"{name}_avg_score.png",
                )
                if total_class_score_sums:
                    save_class_score_heatmap(
                        total_score_sums, total_images,
                        total_class_score_sums, total_class_n,
                        class_names,
                        out_dir / f"{name}_per_class_score.png",
                    )

    print(f"\nDone. Output in ./{out_dir}/")
    if is_3way:
        print("Legend: [original | soft 3-way (ctx=blue tgt=red ign=grey) | hard assignment]")
    else:
        print("Legend: [original | soft score (cold=low, hot=high) | assignment (ctx=blue tgt=red ignored=grey)]")


if __name__ == "__main__":
    main()
