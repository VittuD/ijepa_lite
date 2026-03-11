#!/usr/bin/env python3
"""Fetch summary of recent wandb runs for comparison.

Prints config diffs and key metrics at step 0, 1k, 2k, 3k, and last step
for the most recent N runs in the project.

Usage:
    python scripts/fetch_wandb_runs.py [--n 6] [--project ijepa-lite] [--entity vitturini-davide]
"""
import argparse
import json
import sys

import wandb


# Metrics to extract at each checkpoint step
METRICS = [
    "val/acc",
    "val/loss",
    "train/loss",
    "mask/masker_loss",
    "mask/expected_ntgt",
    "goldilocks/error_skewness",
    "goldilocks/error_kurtosis",
    "mask/batch_iou",
    "mask/tgt_pos_std_norm",
]

# Config keys that are always the same — skip when showing diffs
SKIP_CONFIG_KEYS = {
    "wandb_version", "_wandb",
}

CHECKPOINT_STEPS = [0, 1000, 2000, 3000]  # plus "last"


def get_config_diffs(runs):
    """Find config keys that differ across runs."""
    all_configs = []
    all_keys = set()
    for r in runs:
        cfg = {k: v for k, v in r.config.items() if k not in SKIP_CONFIG_KEYS}
        all_configs.append(cfg)
        all_keys.update(cfg.keys())

    diff_keys = []
    for k in sorted(all_keys):
        vals = [cfg.get(k, "<missing>") for cfg in all_configs]
        # Flatten dicts/lists to string for comparison
        str_vals = [json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else str(v) for v in vals]
        if len(set(str_vals)) > 1:
            diff_keys.append(k)

    return diff_keys, all_configs


def get_metrics_at_steps(run, steps, metrics):
    """Sample metrics from run history at specific steps."""
    results = {}

    # Fetch full history for the metrics we care about (plus _step)
    keys = ["_step"] + metrics
    history = list(run.scan_history(keys=keys, min_step=0, max_step=run.lastHistoryStep + 1))

    if not history:
        return results

    # Build a step -> row lookup
    by_step = {}
    for row in history:
        s = row.get("_step")
        if s is not None:
            by_step[s] = row

    all_steps = sorted(by_step.keys())

    for target in steps:
        # Find closest step
        closest = min(all_steps, key=lambda s: abs(s - target)) if all_steps else None
        if closest is not None and abs(closest - target) < 200:
            results[target] = {m: by_step[closest].get(m) for m in metrics}
            results[target]["_actual_step"] = closest

    # Always include last step
    if all_steps:
        last = all_steps[-1]
        results["last"] = {m: by_step[last].get(m) for m in metrics}
        results["last"]["_actual_step"] = last

    return results


def fmt_val(v):
    if v is None:
        return "—"
    if isinstance(v, float):
        if abs(v) < 0.001 and v != 0:
            return f"{v:.2e}"
        return f"{v:.4f}"
    return str(v)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=6, help="Number of recent runs")
    parser.add_argument("--project", default="ijepa-lite")
    parser.add_argument("--entity", default="vitturini-davide")
    args = parser.parse_args()

    api = wandb.Api()
    runs = list(api.runs(
        f"{args.entity}/{args.project}",
        order="-created_at",
        per_page=args.n,
    ))[:args.n]

    if not runs:
        print("No runs found.")
        sys.exit(1)

    print(f"Found {len(runs)} runs\n")

    # --- Config diffs ---
    diff_keys, all_configs = get_config_diffs(runs)

    print("=" * 80)
    print("CONFIG DIFFS (only keys that vary across runs)")
    print("=" * 80)

    # Header
    header = f"{'key':<40s}"
    for i, r in enumerate(runs):
        header += f" | R{i} ({r.name[:15]})"
    print(header)
    print("-" * len(header))

    for k in diff_keys:
        row = f"{k:<40s}"
        for cfg in all_configs:
            v = cfg.get(k, "—")
            if isinstance(v, (dict, list)):
                v = json.dumps(v, sort_keys=True)
            row += f" | {str(v)[:20]:<20s}"
        print(row)

    # --- Metrics at checkpoints ---
    print()
    print("=" * 80)
    print("METRICS AT CHECKPOINTS")
    print("=" * 80)

    for i, r in enumerate(runs):
        print(f"\n--- R{i}: {r.name} (state={r.state}, steps={r.lastHistoryStep}) ---")
        # Show differing config
        diff_cfg = {k: all_configs[i].get(k, "—") for k in diff_keys}
        print(f"    Config: {json.dumps(diff_cfg, default=str)}")

        data = get_metrics_at_steps(r, CHECKPOINT_STEPS, METRICS)
        if not data:
            print("    (no history)")
            continue

        # Table header
        step_labels = [str(s) for s in CHECKPOINT_STEPS] + ["last"]
        hdr = f"  {'metric':<35s}"
        for sl in step_labels:
            hdr += f" | {sl:>10s}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))

        # Actual steps row
        actual_row = f"  {'(_actual_step)':<35s}"
        for sl in step_labels:
            key = int(sl) if sl != "last" else "last"
            if key in data:
                actual_row += f" | {data[key]['_actual_step']:>10}"
            else:
                actual_row += f" | {'—':>10s}"
        print(actual_row)

        for m in METRICS:
            row = f"  {m:<35s}"
            for sl in step_labels:
                key = int(sl) if sl != "last" else "last"
                if key in data:
                    row += f" | {fmt_val(data[key].get(m)):>10s}"
                else:
                    row += f" | {'—':>10s}"
            print(row)


if __name__ == "__main__":
    main()
