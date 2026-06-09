#!/usr/bin/env python3
"""Summarize task-relevance masking array logs.

This script is intentionally offline and dependency-free. Copy the remote
``.out`` files from the ChestMNIST and IN1K task-relevance arrays to this
machine, then run:

    .venv/bin/python scripts/analyze_task_relevance_logs.py path/to/copied/logs

It parses ``[InlineEval]`` lines, builds one row per data condition and masking
variant, then computes the paired task-relevance delta:

    delta_val_chestmnist - delta_val_in1k
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


EXPECTED_FINAL_STEP = 1170

VARIANT_ORDER = {
    "vanilla_multiblock": 0,
    "mi_affinity_novelty": 1,
    "mi_hmargbs_floor": 2,
    "mi_hmargcosbs_floor": 3,
    "mi_hmarg_sketchorthcosbs_floor": 4,
    "semantic_pca": 5,
}

INLINE_RE = re.compile(
    r"^\[InlineEval\]\s+"
    r"(?P<label>\S+)\s+"
    r"train_(?P<metric>auroc|acc1)=(?P<train>[-+0-9.eE]+)\s+"
    r"val_(?P=metric)=(?P<val>[-+0-9.eE]+)"
)
ARRAY_RE = re.compile(
    r"Array task\s+(?P<task_id>\d+):\s+"
    r"variant=(?P<variant>\S+)\s+exp=(?P<exp>\S+)"
)


@dataclass(frozen=True)
class EvalPoint:
    label: str
    step: int
    train: float
    val: float
    metric: str


@dataclass
class RunSummary:
    path: Path
    pretrain_data: str
    variant: str
    exp_name: str
    task_id: int | None
    points: list[EvalPoint]
    error_summary: str = ""

    @property
    def start(self) -> EvalPoint | None:
        for point in self.points:
            if point.label == "start":
                return point
        return self.points[0] if self.points else None

    @property
    def final(self) -> EvalPoint | None:
        return self.points[-1] if self.points else None

    @property
    def best(self) -> EvalPoint | None:
        if not self.points:
            return None
        return max(self.points, key=lambda point: point.val)

    @property
    def delta_val(self) -> float:
        if self.start is None or self.final is None:
            return math.nan
        return self.final.val - self.start.val

    @property
    def notes(self) -> str:
        notes: list[str] = []
        if self.start is None:
            notes.append("missing_start")
        if self.final is None:
            notes.append("missing_final")
        elif self.final.step != EXPECTED_FINAL_STEP:
            notes.append(f"final_step={self.final.step}")
        if self.best is not None and self.final is not None and self.best.step != self.final.step:
            notes.append(f"best_before_final={self.best.step}")
        if self.error_summary:
            notes.append(self.error_summary)
        return ";".join(notes)


def iter_log_paths(inputs: Iterable[Path]) -> list[Path]:
    paths: list[Path] = []
    for input_path in inputs:
        if input_path.is_dir():
            for child in input_path.rglob("*"):
                if child.is_file() and child.suffix in {".out", ".log", ".txt", ""}:
                    paths.append(child)
        elif input_path.is_file():
            paths.append(input_path)
        else:
            print(f"warning: skipping missing path: {input_path}", file=sys.stderr)
    return sorted(set(paths))


def parse_step(label: str) -> int:
    if label == "start":
        return 0
    if label.startswith("step="):
        return int(label.split("=", 1)[1])
    if label.startswith("epoch="):
        return -1
    return -1


def infer_pretrain_data(path: Path, exp_name: str, text: str) -> str:
    haystack = f"{path} {exp_name} {text[:2000]}".lower()
    if "claim_in1k" in haystack or "in1k224" in haystack or "data=imagenet" in haystack:
        return "in1k"
    if "claim_chestmnist" in haystack or "chestmnist224" in haystack or "data=chestmnist" in haystack:
        return "chestmnist"
    return "unknown"


def parse_log(path: Path) -> RunSummary | None:
    text = path.read_text(errors="replace")
    task_id: int | None = None
    variant = ""
    exp_name = ""
    points: list[EvalPoint] = []

    for line in text.splitlines():
        array_match = ARRAY_RE.search(line)
        if array_match:
            task_id = int(array_match.group("task_id"))
            variant = array_match.group("variant")
            exp_name = array_match.group("exp")
            continue

        inline_match = INLINE_RE.search(line)
        if inline_match:
            label = inline_match.group("label")
            points.append(
                EvalPoint(
                    label=label,
                    step=parse_step(label),
                    train=float(inline_match.group("train")),
                    val=float(inline_match.group("val")),
                    metric=inline_match.group("metric"),
                )
            )

    if not points and not variant:
        return None

    if not variant and exp_name:
        variant = infer_variant(exp_name)
    if not variant:
        variant = f"task_{task_id}" if task_id is not None else path.stem

    pretrain_data = infer_pretrain_data(path, exp_name, text)
    return RunSummary(
        path=path,
        pretrain_data=pretrain_data,
        variant=variant,
        exp_name=exp_name,
        task_id=task_id,
        points=points,
        error_summary=summarize_error_file(path.with_suffix(".err")),
    )


def summarize_error_file(path: Path) -> str:
    if not path.is_file():
        return ""

    text = path.read_text(errors="replace")
    if not text.strip():
        return ""

    notes: list[str] = []
    if "DUE TO TIME LIMIT" in text:
        notes.append("time_limit")
    if "ProcessGroupNCCL" in text or "DistBackendError" in text:
        notes.append("nccl_timeout")
    if "ChildFailedError" in text:
        notes.append("child_failed")
    if "out of memory" in text.lower() or "cuda oom" in text.lower():
        notes.append("oom")
    if "srun: error" in text:
        notes.append("srun_error")
    if "CANCELLED" in text and "time_limit" not in notes:
        notes.append("cancelled")

    if notes:
        return "err=" + "+".join(dict.fromkeys(notes))
    return "err_present"


def infer_variant(exp_name: str) -> str:
    for variant in sorted(VARIANT_ORDER, key=len, reverse=True):
        if variant in exp_name:
            return variant
    return ""


def fmt_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.4f}"


def row_for_run(run: RunSummary) -> dict[str, str]:
    start = run.start
    final = run.final
    best = run.best
    return {
        "pretrain_data": run.pretrain_data,
        "variant": run.variant,
        "start_train_auroc": fmt_float(start.train) if start else "nan",
        "start_val_auroc": fmt_float(start.val) if start else "nan",
        "final_train_auroc": fmt_float(final.train) if final else "nan",
        "final_val_auroc": fmt_float(final.val) if final else "nan",
        "best_val_auroc": fmt_float(best.val) if best else "nan",
        "step_of_best": str(best.step) if best else "",
        "delta_val": fmt_float(run.delta_val),
        "notes": run.notes,
        "path": str(run.path),
    }


def print_markdown(rows: list[dict[str, str]]) -> None:
    columns = [
        "pretrain_data",
        "variant",
        "start_val_auroc",
        "final_val_auroc",
        "best_val_auroc",
        "step_of_best",
        "delta_val",
        "notes",
    ]
    print("| " + " | ".join(columns) + " |")
    print("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        print("| " + " | ".join(row[column] for column in columns) + " |")


def print_csv(rows: list[dict[str, str]]) -> None:
    columns = list(rows[0].keys()) if rows else [
        "pretrain_data",
        "variant",
        "start_train_auroc",
        "start_val_auroc",
        "final_train_auroc",
        "final_val_auroc",
        "best_val_auroc",
        "step_of_best",
        "delta_val",
        "notes",
        "path",
    ]
    writer = csv.DictWriter(sys.stdout, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)


def print_task_relevance(runs: list[RunSummary]) -> None:
    by_pair = {(run.pretrain_data, run.variant): run for run in runs}
    print()
    print("task_relevance_delta by variant")
    print("| variant | chest_delta_val | in1k_delta_val | task_relevance_delta | notes |")
    print("| --- | --- | --- | --- | --- |")
    variants = sorted(
        {run.variant for run in runs},
        key=lambda variant: (VARIANT_ORDER.get(variant, 999), variant),
    )
    for variant in variants:
        chest = by_pair.get(("chestmnist", variant))
        in1k = by_pair.get(("in1k", variant))
        if chest is None or in1k is None:
            missing = []
            if chest is None:
                missing.append("missing_chestmnist")
            if in1k is None:
                missing.append("missing_in1k")
            print(f"| {variant} | nan | nan | nan | {';'.join(missing)} |")
            continue
        task_delta = chest.delta_val - in1k.delta_val
        notes = ";".join(note for note in [chest.notes, in1k.notes] if note)
        print(
            f"| {variant} | {fmt_float(chest.delta_val)} | "
            f"{fmt_float(in1k.delta_val)} | {fmt_float(task_delta)} | {notes} |"
        )


def print_trajectories(runs: list[RunSummary]) -> None:
    print()
    print("inline_eval trajectories")
    print("| pretrain_data | variant | train_auroc_by_step | val_auroc_by_step |")
    print("| --- | --- | --- | --- |")
    for run in runs:
        train_items = []
        val_items = []
        for point in run.points:
            train_items.append(f"{point.step}:{fmt_float(point.train)}")
            val_items.append(f"{point.step}:{fmt_float(point.val)}")
        print(
            f"| {run.pretrain_data} | {run.variant} | "
            f"{' '.join(train_items)} | {' '.join(val_items)} |"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Copied log files or directories")
    parser.add_argument(
        "--format",
        choices=("markdown", "csv"),
        default="markdown",
        help="Output format for per-run rows",
    )
    args = parser.parse_args()

    paths = iter_log_paths(args.paths)
    runs = [run for path in paths if (run := parse_log(path)) is not None]
    if not runs:
        print("no parseable logs found", file=sys.stderr)
        return 1

    runs.sort(
        key=lambda run: (
            run.pretrain_data,
            VARIANT_ORDER.get(run.variant, 999),
            run.variant,
            str(run.path),
        )
    )
    rows = [row_for_run(run) for run in runs]
    if args.format == "csv":
        print_csv(rows)
    else:
        print_markdown(rows)
        print_task_relevance(runs)
        print_trajectories(runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
