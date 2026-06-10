#!/usr/bin/env python3
"""Aggregate token-embedding visualization statistics.

The visualization script writes one ``summary.csv`` per checkpoint output
directory. This script reads those CSVs and reports means/stds for spatial
fragmentation and cluster-separability metrics.

Examples:

    .venv/bin/python scripts/analyze_token_embedding_viz_stats.py \
        token_embedding_viz/task_relevance_vith14_downstream_last_448

    .venv/bin/python scripts/analyze_token_embedding_viz_stats.py \
        /path/to/viz/root \
        --compare chest_vanilla_multiblock_last chest_affinity_novelty_last \
        --out-dir /tmp/token_viz_stats
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable


PRIMARY_METRICS = (
    "boundary_fraction",
    "total_connected_components",
    "extra_connected_components",
    "max_connected_components",
    "cluster_entropy",
    "silhouette",
    "calinski_harabasz",
    "davies_bouldin",
    "pca_pc1",
    "pca_pc2",
    "pca_pc3",
    "pca_top3_sum",
)

VARIANT_ORDER = {
    "vanilla_multiblock": 0,
    "affinity_novelty": 1,
    "hmargbs_floor": 2,
    "hmargcosbs_floor": 3,
    "hmarg_sketchorthcosbs_floor": 4,
    "semantic_pca": 5,
}


@dataclass(frozen=True)
class Stat:
    n: int
    mean: float
    sd: float
    sem: float
    min: float
    max: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Aggregate summary.csv files from token embedding visualizations and "
            "compare checkpoint smoothness statistics."
        )
    )
    p.add_argument(
        "inputs",
        nargs="+",
        help="summary.csv files or directories. Directories are searched recursively.",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Optional directory for aggregate CSV/JSON/Markdown outputs.",
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
        "--compare",
        nargs=2,
        action="append",
        metavar=("BASELINE_CHECKPOINT", "OTHER_CHECKPOINT"),
        help=(
            "Compare two checkpoints on matched dataset/sample/method/params rows. "
            "Repeat for multiple pairs."
        ),
    )
    p.add_argument(
        "--auto-compare-variants",
        action="store_true",
        help=(
            "Automatically compare variants within each family against "
            "vanilla_multiblock when checkpoint names follow chest_* or in1k_*."
        ),
    )
    return p.parse_args()


def iter_summary_csvs(inputs: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_file():
            paths.append(path)
        elif path.is_dir():
            paths.extend(sorted(path.rglob("summary.csv")))
        else:
            print(f"warning: missing input skipped: {path}", file=sys.stderr)
    return sorted(set(p.resolve() for p in paths))


def parse_json(value: str, default: Any) -> Any:
    if value is None or value == "":
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def parse_float(value: Any) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def infer_family(checkpoint: str) -> str:
    if checkpoint.startswith("chest_"):
        return "chest"
    if checkpoint.startswith("in1k_"):
        return "in1k"
    return "unknown"


def infer_variant(checkpoint: str) -> str:
    core = checkpoint
    for prefix in ("chest_", "in1k_"):
        if core.startswith(prefix):
            core = core[len(prefix):]
    if core.endswith("_last"):
        core = core[:-len("_last")]
    return core


def enrich_row(row: dict[str, str], source_csv: Path) -> dict[str, Any]:
    out: dict[str, Any] = dict(row)
    out["source_csv"] = str(source_csv)
    checkpoint = str(row.get("checkpoint", ""))
    out["family"] = infer_family(checkpoint)
    out["variant"] = infer_variant(checkpoint)

    components = parse_json(str(row.get("connected_components", "")), {})
    component_values = [int(v) for v in components.values()] if isinstance(components, dict) else []
    out["total_connected_components"] = float(sum(component_values))
    out["extra_connected_components"] = float(sum(max(v - 1, 0) for v in component_values))
    out["max_connected_components"] = float(max(component_values) if component_values else 0)

    pca = parse_json(str(row.get("pca_explained_variance", "")), [])
    if not isinstance(pca, list):
        pca = []
    for idx in range(3):
        out[f"pca_pc{idx + 1}"] = float(pca[idx]) if idx < len(pca) else math.nan
    out["pca_top3_sum"] = float(sum(float(v) for v in pca[:3]))

    for key in (
        "n_clusters",
        "noise_count",
        "cluster_entropy",
        "boundary_fraction",
        "silhouette",
        "calinski_harabasz",
        "davies_bouldin",
    ):
        out[key] = parse_float(row.get(key))
    return out


def read_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(enrich_row(row, path))
    return rows


def finite_values(rows: list[dict[str, Any]], metric: str) -> list[float]:
    values = [parse_float(row.get(metric)) for row in rows]
    return [v for v in values if math.isfinite(v)]


def summarize(values: list[float]) -> Stat | None:
    if not values:
        return None
    sd = stdev(values) if len(values) >= 2 else 0.0
    return Stat(
        n=len(values),
        mean=mean(values),
        sd=sd,
        sem=sd / math.sqrt(len(values)) if values else math.nan,
        min=min(values),
        max=max(values),
    )


def grouped(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(key, "")) for key in keys)].append(row)
    return groups


def aggregate_rows(
    rows: list[dict[str, Any]],
    *,
    group_by: tuple[str, ...],
    metrics: tuple[str, ...],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key, group_rows in sorted(grouped(rows, group_by).items()):
        row = {name: value for name, value in zip(group_by, key)}
        row["n_rows"] = len(group_rows)
        for metric in metrics:
            stat = summarize(finite_values(group_rows, metric))
            if stat is None:
                continue
            row[f"{metric}_n"] = stat.n
            row[f"{metric}_mean"] = stat.mean
            row[f"{metric}_sd"] = stat.sd
            row[f"{metric}_sem"] = stat.sem
            row[f"{metric}_min"] = stat.min
            row[f"{metric}_max"] = stat.max
        out.append(row)
    return out


def match_key(row: dict[str, Any]) -> tuple[str, ...]:
    return (
        str(row.get("dataset", "")),
        str(row.get("split", "")),
        str(row.get("sample_id", "")),
        str(row.get("method", "")),
        str(row.get("params", "")),
    )


def compare_pair(
    rows: list[dict[str, Any]],
    *,
    baseline: str,
    other: str,
    metrics: tuple[str, ...],
) -> list[dict[str, Any]]:
    baseline_rows = [row for row in rows if row.get("checkpoint") == baseline]
    other_rows = [row for row in rows if row.get("checkpoint") == other]
    baseline_by_key = {match_key(row): row for row in baseline_rows}
    other_by_key = {match_key(row): row for row in other_rows}
    common_keys = sorted(set(baseline_by_key) & set(other_by_key))

    pair_rows: list[dict[str, Any]] = []
    by_dataset_method: dict[tuple[str, str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for key in common_keys:
        b = baseline_by_key[key]
        o = other_by_key[key]
        by_dataset_method[(str(b["dataset"]), str(b["method"]), str(b["params"]))].append((b, o))

    for (dataset, method, params), pairs in sorted(by_dataset_method.items()):
        row: dict[str, Any] = {
            "baseline": baseline,
            "other": other,
            "baseline_family": infer_family(baseline),
            "other_family": infer_family(other),
            "baseline_variant": infer_variant(baseline),
            "other_variant": infer_variant(other),
            "dataset": dataset,
            "method": method,
            "params": params,
            "n_matched": len(pairs),
        }
        for metric in metrics:
            deltas: list[float] = []
            baseline_values: list[float] = []
            other_values: list[float] = []
            for b, o in pairs:
                bv = parse_float(b.get(metric))
                ov = parse_float(o.get(metric))
                if not (math.isfinite(bv) and math.isfinite(ov)):
                    continue
                baseline_values.append(bv)
                other_values.append(ov)
                deltas.append(ov - bv)
            bstat = summarize(baseline_values)
            ostat = summarize(other_values)
            dstat = summarize(deltas)
            if bstat is None or ostat is None or dstat is None:
                continue
            row[f"{metric}_baseline_mean"] = bstat.mean
            row[f"{metric}_other_mean"] = ostat.mean
            row[f"{metric}_delta_mean"] = dstat.mean
            row[f"{metric}_delta_sd"] = dstat.sd
            row[f"{metric}_delta_sem"] = dstat.sem
            if abs(bstat.mean) > 1e-12:
                row[f"{metric}_relative_delta_pct"] = 100.0 * dstat.mean / abs(bstat.mean)
        pair_rows.append(row)
    return pair_rows


def automatic_pairs(rows: list[dict[str, Any]]) -> list[tuple[str, str]]:
    by_family_variant: dict[tuple[str, str], str] = {}
    for row in rows:
        checkpoint = str(row.get("checkpoint", ""))
        family = infer_family(checkpoint)
        variant = infer_variant(checkpoint)
        if family != "unknown":
            by_family_variant[(family, variant)] = checkpoint

    pairs: list[tuple[str, str]] = []
    for family in sorted({family for family, _ in by_family_variant}):
        baseline = by_family_variant.get((family, "vanilla_multiblock"))
        if baseline is None:
            continue
        variants = sorted(
            variant for fam, variant in by_family_variant if fam == family and variant != "vanilla_multiblock"
        )
        for variant in variants:
            pairs.append((baseline, by_family_variant[(family, variant)]))
    return pairs


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.4g}"
    return str(value)


def markdown_table(rows: list[dict[str, Any]], columns: list[str], limit: int | None = None) -> str:
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        return "_No rows._\n"
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines) + "\n"


def write_report(
    *,
    rows: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    metrics: tuple[str, ...],
) -> str:
    report: list[str] = []
    report.append("# Token Embedding Visualization Statistics\n")
    report.append(f"Input rows: {len(rows)}\n")
    report.append(f"Metrics: {', '.join(metrics)}\n")

    compact_cols = [
        "checkpoint",
        "dataset",
        "method",
        "params",
        "n_rows",
        "boundary_fraction_mean",
        "boundary_fraction_sd",
        "total_connected_components_mean",
        "extra_connected_components_mean",
        "silhouette_mean",
        "pca_top3_sum_mean",
    ]
    existing_cols = [col for col in compact_cols if any(col in row for row in aggregates)]
    report.append("## Aggregates\n")
    report.append(markdown_table(aggregates, existing_cols, limit=80))

    if comparisons:
        compare_cols = [
            "baseline",
            "other",
            "dataset",
            "method",
            "params",
            "n_matched",
            "boundary_fraction_baseline_mean",
            "boundary_fraction_other_mean",
            "boundary_fraction_delta_mean",
            "boundary_fraction_relative_delta_pct",
            "total_connected_components_baseline_mean",
            "total_connected_components_other_mean",
            "total_connected_components_delta_mean",
            "total_connected_components_relative_delta_pct",
            "pca_top3_sum_delta_mean",
        ]
        existing_compare_cols = [
            col for col in compare_cols if any(col in row for row in comparisons)
        ]
        report.append("## Matched Comparisons\n")
        report.append(markdown_table(comparisons, existing_compare_cols, limit=120))

    report.append("## Reading Guide\n")
    report.append(
        "- Lower `boundary_fraction` means fewer neighboring patch-label changes.\n"
        "- Lower `total_connected_components` / `extra_connected_components` means fewer spatial islands.\n"
        "- Similar silhouette/Calinski/Davies-Bouldin with lower fragmentation suggests smoother spatial organization, not merely easier feature-space clustering.\n"
        "- Similar `pca_top3_sum` means smoothness is not explained by a large change in top-3 PCA variance alone.\n"
    )
    return "\n".join(report)


def main() -> None:
    args = parse_args()
    paths = iter_summary_csvs(args.inputs)
    if not paths:
        raise SystemExit("No summary.csv files found.")

    rows = read_rows(paths)
    metrics = tuple(args.metrics)
    aggregates = aggregate_rows(
        rows,
        group_by=tuple(args.group_by),
        metrics=metrics,
    )

    pairs = list(args.compare or [])
    if args.auto_compare_variants:
        pairs.extend(automatic_pairs(rows))

    comparisons: list[dict[str, Any]] = []
    for baseline, other in pairs:
        comparisons.extend(compare_pair(rows, baseline=baseline, other=other, metrics=metrics))

    report = write_report(
        rows=rows,
        aggregates=aggregates,
        comparisons=comparisons,
        metrics=metrics,
    )
    print(report)

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_csv(out_dir / "aggregates.csv", aggregates)
        write_csv(out_dir / "comparisons.csv", comparisons)
        (out_dir / "report.md").write_text(report)
        (out_dir / "aggregates.json").write_text(json.dumps(aggregates, indent=2) + "\n")
        (out_dir / "comparisons.json").write_text(json.dumps(comparisons, indent=2) + "\n")


if __name__ == "__main__":
    main()
