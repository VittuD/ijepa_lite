from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Materialize a Hugging Face dataset into save_to_disk format."
    )
    p.add_argument("--repo-id", required=True, help="HF dataset repo id.")
    p.add_argument(
        "--source-dir",
        default=None,
        help="Optional local HF dataset snapshot to load instead of the repo id.",
    )
    p.add_argument(
        "--dest-dir",
        required=True,
        help="Target directory for DatasetDict.save_to_disk(...).",
    )
    p.add_argument(
        "--cache-dir",
        default=None,
        help="Optional HF datasets cache dir for dataset preparation.",
    )
    p.add_argument(
        "--subset",
        default=None,
        help="Optional dataset config/subset name.",
    )
    p.add_argument(
        "--splits",
        nargs="+",
        required=True,
        help="Splits to materialize, e.g. train test.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    source_dir = Path(args.source_dir) if args.source_dir is not None else None
    source = str(source_dir) if source_dir is not None and source_dir.exists() else args.repo_id

    load_kwargs = {}
    if args.subset:
        load_kwargs["name"] = args.subset
    if args.cache_dir:
        load_kwargs["cache_dir"] = args.cache_dir

    ds_dict = {}
    for split in args.splits:
        print(f"[save_hf_dataset] loading split={split} from source={source}")
        ds_dict[split] = load_dataset(source, split=split, **load_kwargs)

    dest_dir = Path(args.dest_dir)
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"[save_hf_dataset] saving DatasetDict to {dest_dir}")
    from datasets import DatasetDict

    DatasetDict(ds_dict).save_to_disk(str(dest_dir))
    print("[save_hf_dataset] done")


if __name__ == "__main__":
    main()
