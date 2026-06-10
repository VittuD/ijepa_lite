#!/usr/bin/env python3
"""Run grouped token-viz statistics for task-relevance visualization outputs.

Given a directory like:

    token_embedding_viz/task_relevance_vith14_downstream_last_448/

with one subdirectory per checkpoint, each containing ``summary.csv``, this
wrapper writes three analysis bundles:

    <out-dir>/in1k/
    <out-dir>/chest/
    <out-dir>/all/

The ``all`` bundle includes both within-family vanilla-vs-variant comparisons
and same-variant chest-vs-IN1K comparisons.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_token_embedding_viz_stats import (
    PRIMARY_METRICS,
    aggregate_rows,
    automatic_pairs,
    compare_pair,
    infer_variant,
    read_rows,
    write_csv,
    write_report,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Create in1k, chest, and all-together reports from task-relevance "
            "token embedding visualization summary.csv files."
        )
    )
    p.add_argument(
        "viz_root",
        help="Root containing checkpoint output folders with summary.csv files.",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Output directory. Defaults to <viz_root>/stats_by_family."
        ),
    )
    p.add_argument(
        "--group-by",
        nargs="+",
        default=("checkpoint", "dataset", "method", "params"),
        choices=("checkpoint", "family", "variant", "dataset", "split", "method", "params"),
        help="Columns used for aggregate rows.",
    )
    p.add_argument(
        "--metrics",
        nargs="+",
        default=PRIMARY_METRICS,
        help="Numeric metrics to aggregate.",
    )
    p.add_argument(
        "--no-cross-family",
        action="store_true",
        help="Do not add same-variant chest-vs-IN1K comparisons in the all report.",
    )
    return p.parse_args()


def find_grouped_summary_csvs(viz_root: Path) -> dict[str, list[Path]]:
    groups = {"chest": [], "in1k": [], "all": []}
    for path in sorted(viz_root.rglob("summary.csv")):
        checkpoint_dir = path.parent.name
        if checkpoint_dir.startswith("chest_"):
            groups["chest"].append(path)
            groups["all"].append(path)
        elif checkpoint_dir.startswith("in1k_"):
            groups["in1k"].append(path)
            groups["all"].append(path)
    return groups


def checkpoint_by_variant(rows: list[dict], family: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in rows:
        checkpoint = str(row.get("checkpoint", ""))
        if not checkpoint.startswith(f"{family}_"):
            continue
        out[infer_variant(checkpoint)] = checkpoint
    return out


def cross_family_pairs(rows: list[dict]) -> list[tuple[str, str]]:
    chest = checkpoint_by_variant(rows, "chest")
    in1k = checkpoint_by_variant(rows, "in1k")
    pairs: list[tuple[str, str]] = []
    for variant in sorted(set(chest) & set(in1k)):
        pairs.append((in1k[variant], chest[variant]))
    return pairs


def write_bundle(
    *,
    name: str,
    paths: list[Path],
    out_dir: Path,
    group_by: tuple[str, ...],
    metrics: tuple[str, ...],
    include_cross_family: bool,
) -> dict[str, object]:
    bundle_dir = out_dir / name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(paths)
    aggregates = aggregate_rows(rows, group_by=group_by, metrics=metrics)

    pairs = automatic_pairs(rows)
    if include_cross_family:
        pairs.extend(cross_family_pairs(rows))

    comparisons: list[dict] = []
    for baseline, other in pairs:
        comparisons.extend(
            compare_pair(rows, baseline=baseline, other=other, metrics=metrics)
        )

    report = write_report(
        rows=rows,
        aggregates=aggregates,
        comparisons=comparisons,
        metrics=metrics,
    )

    write_csv(bundle_dir / "aggregates.csv", aggregates)
    write_csv(bundle_dir / "comparisons.csv", comparisons)
    (bundle_dir / "aggregates.json").write_text(json.dumps(aggregates, indent=2) + "\n")
    (bundle_dir / "comparisons.json").write_text(json.dumps(comparisons, indent=2) + "\n")
    (bundle_dir / "report.md").write_text(report)

    return {
        "name": name,
        "summary_csvs": [str(path) for path in paths],
        "n_summary_csvs": len(paths),
        "n_rows": len(rows),
        "n_aggregates": len(aggregates),
        "n_comparisons": len(comparisons),
        "report": str(bundle_dir / "report.md"),
    }


def main() -> None:
    args = parse_args()
    viz_root = Path(args.viz_root).resolve()
    if not viz_root.is_dir():
        raise SystemExit(f"viz_root is not a directory: {viz_root}")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else viz_root / "stats_by_family"
    groups = find_grouped_summary_csvs(viz_root)

    missing = [name for name in ("in1k", "chest") if not groups[name]]
    if missing:
        raise SystemExit(
            "Missing expected summary groups: "
            + ", ".join(missing)
            + f". Looked under {viz_root}"
        )

    manifest = {
        "viz_root": str(viz_root),
        "out_dir": str(out_dir),
        "bundles": [],
    }

    for name in ("in1k", "chest", "all"):
        include_cross = name == "all" and not args.no_cross_family
        bundle = write_bundle(
            name=name,
            paths=groups[name],
            out_dir=out_dir,
            group_by=tuple(args.group_by),
            metrics=tuple(args.metrics),
            include_cross_family=include_cross,
        )
        manifest["bundles"].append(bundle)
        print(
            f"[{name}] csvs={bundle['n_summary_csvs']} rows={bundle['n_rows']} "
            f"aggregates={bundle['n_aggregates']} comparisons={bundle['n_comparisons']} "
            f"report={bundle['report']}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote grouped token-viz stats to {out_dir}")


if __name__ == "__main__":
    main()
