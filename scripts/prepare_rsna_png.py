#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image


def _convert_one(args: tuple[str, str, bool]) -> tuple[str, str]:
    src_path, dst_path, overwrite = args
    src = Path(src_path)
    dst = Path(dst_path)
    if dst.exists() and not overwrite:
        return "skip", src.name

    import pydicom

    ds = pydicom.dcmread(str(src))
    arr = ds.pixel_array.astype("float32")
    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        arr = arr.max() - arr
    arr -= arr.min()
    denom = arr.max()
    if denom > 0:
        arr /= denom
    out = (arr * 255.0).clip(0, 255).astype("uint8")
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(dst)
    return "write", src.name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert RSNA Pneumonia Detection DICOM images to PNG."
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("DATA_ROOT", os.environ.get("FAST", "/scratch") + "/datasets"),
        help="Dataset root containing rsna-pneumonia-detection.",
    )
    parser.add_argument(
        "--data-dir",
        default="rsna-pneumonia-detection",
        help="RSNA dataset directory under data-root.",
    )
    parser.add_argument(
        "--src-dir",
        default="stage_2_train_images",
        help="DICOM image directory under data-dir.",
    )
    parser.add_argument(
        "--dst-dir",
        default="png_images",
        help="PNG output directory under data-dir.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
        help="Number of conversion workers.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing PNGs instead of resuming.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=500,
        help="Print progress every N completed DICOM files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.data_root) / str(args.data_dir)
    src_dir = root / str(args.src_dir)
    dst_dir = root / str(args.dst_dir)

    if not src_dir.is_dir():
        raise FileNotFoundError(f"RSNA DICOM source directory not found: {src_dir}")
    dst_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(src_dir.glob("*.dcm"))
    if not paths:
        raise RuntimeError(f"No DICOM files found under {src_dir}")

    jobs = [
        (str(path), str(dst_dir / f"{path.stem}.png"), bool(args.overwrite))
        for path in paths
    ]
    workers = max(1, int(args.workers))
    print(
        f"converting {len(jobs)} DICOMs from {src_dir} to {dst_dir} "
        f"with workers={workers} overwrite={bool(args.overwrite)}",
        flush=True,
    )

    counts = {"write": 0, "skip": 0}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_convert_one, job) for job in jobs]
        for idx, future in enumerate(as_completed(futures), 1):
            status, name = future.result()
            counts[status] = counts.get(status, 0) + 1
            if idx % int(args.log_every) == 0:
                print(
                    f"done={idx}/{len(jobs)} written={counts.get('write', 0)} "
                    f"skipped={counts.get('skip', 0)} last={name}",
                    flush=True,
                )

    print(
        f"finished: written={counts.get('write', 0)} skipped={counts.get('skip', 0)} "
        f"output={dst_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
