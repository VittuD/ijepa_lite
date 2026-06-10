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


RANKING_METRICS = (
    "boundary_fraction",
    "extra_connected_components",
    "total_connected_components",
    "max_connected_components",
    "silhouette",
    "pca_top3_sum",
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


def mean_finite(rows: list[dict], metric: str) -> float | None:
    values = []
    for row in rows:
        value = row.get(metric)
        if isinstance(value, (int, float)):
            numeric = float(value)
        else:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
        if numeric == numeric:
            values.append(numeric)
    if not values:
        return None
    return sum(values) / len(values)


def compact_rows(rows: list[dict], group_by: tuple[str, ...]) -> list[dict]:
    grouped: dict[tuple[str, ...], list[dict]] = {}
    for row in rows:
        key = tuple(str(row.get(col, "")) for col in group_by)
        grouped.setdefault(key, []).append(row)

    out: list[dict] = []
    for key, group_rows in sorted(grouped.items()):
        compact = {col: value for col, value in zip(group_by, key)}
        compact["n_rows"] = len(group_rows)
        compact["n_samples"] = len(
            {
                (
                    str(row.get("dataset", "")),
                    str(row.get("split", "")),
                    str(row.get("sample_id", "")),
                )
                for row in group_rows
            }
        )
        for metric in RANKING_METRICS:
            value = mean_finite(group_rows, metric)
            if value is not None:
                compact[f"{metric}_mean"] = value
        out.append(compact)

    add_smoothness_score(out)
    return sorted(
        out,
        key=lambda row: (
            float(row.get("smoothness_score", float("inf"))),
            str(row.get("family", "")),
            str(row.get("variant", "")),
            str(row.get("checkpoint", "")),
        ),
    )


def compact_rows_by_method(rows: list[dict], group_by: tuple[str, ...]) -> dict[str, list[dict]]:
    methods = sorted({str(row.get("method", "")) for row in rows if row.get("method", "")})
    return {
        method: compact_rows(
            [row for row in rows if str(row.get("method", "")) == method],
            group_by,
        )
        for method in methods
    }


def add_smoothness_score(rows: list[dict]) -> None:
    """Add a lower-is-smoother z-score over boundary and fragmentation metrics."""
    score_metrics = (
        "boundary_fraction_mean",
        "extra_connected_components_mean",
        "total_connected_components_mean",
    )
    means: dict[str, float] = {}
    sds: dict[str, float] = {}
    for metric in score_metrics:
        values = [
            float(row[metric])
            for row in rows
            if metric in row and float(row[metric]) == float(row[metric])
        ]
        if not values:
            continue
        mu = sum(values) / len(values)
        var = sum((value - mu) ** 2 for value in values) / max(len(values) - 1, 1)
        means[metric] = mu
        sds[metric] = var ** 0.5

    for row in rows:
        z_values = []
        for metric in score_metrics:
            if metric not in row or metric not in means:
                continue
            sd = sds.get(metric, 0.0)
            if sd <= 1e-12:
                continue
            z_values.append((float(row[metric]) - means[metric]) / sd)
        if z_values:
            row["smoothness_score"] = sum(z_values) / len(z_values)


def write_compact_markdown(path: Path, title: str, rows: list[dict]) -> None:
    columns = [
        "smoothness_score",
        "checkpoint",
        "family",
        "variant",
        "method",
        "n_samples",
        "boundary_fraction_mean",
        "extra_connected_components_mean",
        "total_connected_components_mean",
        "silhouette_mean",
        "pca_top3_sum_mean",
    ]
    columns = [col for col in columns if any(col in row for row in rows)]
    lines = [f"# {title}", "", "Lower `smoothness_score` is smoother.", ""]
    if rows:
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join("---" for _ in columns) + " |")
        for row in rows:
            cells = []
            for col in columns:
                value = row.get(col, "")
                cells.append(f"{value:.4g}" if isinstance(value, float) else str(value))
            lines.append("| " + " | ".join(cells) + " |")
    else:
        lines.append("_No rows._")
    path.write_text("\n".join(lines) + "\n")


def write_rankings_by_method(bundle_dir: Path, rows: list[dict]) -> dict[str, dict[str, str]]:
    method_dir = bundle_dir / "rankings_by_method"
    method_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, dict[str, str]] = {}
    checkpoint_tables = compact_rows_by_method(
        rows,
        ("method", "checkpoint", "family", "variant"),
    )

    for method in sorted(checkpoint_tables):
        checkpoint_rows = checkpoint_tables.get(method, [])
        checkpoint_csv = method_dir / f"{method}_ranking_by_checkpoint.csv"
        checkpoint_md = method_dir / f"{method}_ranking_by_checkpoint.md"

        write_csv(checkpoint_csv, checkpoint_rows)
        write_compact_markdown(
            checkpoint_md,
            f"{method} Ranking By Checkpoint",
            checkpoint_rows,
        )
        out[method] = {
            "ranking_by_checkpoint": str(checkpoint_csv),
        }
    return out


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
    ranking_by_checkpoint = compact_rows(rows, ("checkpoint", "family", "variant"))
    rankings_by_method = write_rankings_by_method(bundle_dir, rows)

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
    write_csv(bundle_dir / "ranking_by_checkpoint.csv", ranking_by_checkpoint)
    (bundle_dir / "aggregates.json").write_text(json.dumps(aggregates, indent=2) + "\n")
    (bundle_dir / "comparisons.json").write_text(json.dumps(comparisons, indent=2) + "\n")
    (bundle_dir / "ranking_by_checkpoint.json").write_text(
        json.dumps(ranking_by_checkpoint, indent=2) + "\n"
    )
    (bundle_dir / "report.md").write_text(report)
    write_compact_markdown(
        bundle_dir / "ranking_by_checkpoint.md",
        f"{name} Pooled Ranking By Checkpoint",
        ranking_by_checkpoint,
    )

    return {
        "name": name,
        "summary_csvs": [str(path) for path in paths],
        "n_summary_csvs": len(paths),
        "n_rows": len(rows),
        "n_aggregates": len(aggregates),
        "n_comparisons": len(comparisons),
        "ranking_by_checkpoint": str(bundle_dir / "ranking_by_checkpoint.csv"),
        "rankings_by_method": rankings_by_method,
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
