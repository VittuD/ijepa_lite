#!/usr/bin/env python3
"""Fetch summary of recent wandb runs for comparison.

Prints config diffs (flattened leaf keys) and key metrics at step 0, 1k, 2k,
3k, and last step for the most recent N runs in the project.

Usage:
    python scripts/fetch_wandb_runs.py [--n 6] [--project ijepa-lite] [--entity vitturini-davide]
"""
import argparse
import json
import sys

import wandb


# Metrics to extract at each checkpoint step
METRICS = [
    "inline_eval/val_acc1",
    "inline_eval/train_acc1",
    "train/loss",
    "train/reconstruction_loss",
    "mask/masker_loss",
    "mask/expected_ntgt",
    "goldilocks/error_skewness",
    "goldilocks/error_kurtosis",
    "goldilocks/score_error_corr",
    "goldilocks/z_std_mean",
    "mask/batch_iou",
    "mask/tgt_pos_std_norm",
    "mask/marginal_score_std",
]

SKIP_CONFIG_KEYS = {"wandb_version", "_wandb"}

CHECKPOINT_STEPS = [0, 1000, 2000, 3000]  # plus "last"


def flatten_dict(d, prefix=""):
    """Flatten nested dict to dot-separated leaf keys."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(flatten_dict(v, key))
        else:
            out[key] = v
    return out


def get_config_diffs(runs):
    """Find flattened config keys that differ across runs."""
    all_configs = []
    all_keys = set()
    for r in runs:
        raw = {k: v for k, v in r.config.items() if k not in SKIP_CONFIG_KEYS}
        flat = flatten_dict(raw)
        all_configs.append(flat)
        all_keys.update(flat.keys())

    diff_keys = []
    for k in sorted(all_keys):
        vals = [str(cfg.get(k, "<missing>")) for cfg in all_configs]
        if len(set(vals)) > 1:
            diff_keys.append(k)

    return diff_keys, all_configs


def get_metrics_at_steps(run, steps, metrics):
    """Sample metrics from run history at specific steps."""
    results = {}

    keys = ["_step"] + metrics
    try:
        history = list(run.scan_history(keys=keys))
    except Exception:
        # Fallback: try .history() which works for some offline-synced runs
        try:
            history = list(run.history(keys=keys, pandas=False))
        except Exception:
            return results

    if not history:
        return results

    # Build step -> row lookup
    by_step = {}
    for row in history:
        s = row.get("_step")
        if s is not None:
            by_step[s] = row

    all_steps = sorted(by_step.keys())
    if not all_steps:
        return results

    for target in steps:
        closest = min(all_steps, key=lambda s: abs(s - target))
        if abs(closest - target) < 200:
            results[target] = {m: by_step[closest].get(m) for m in metrics}
            results[target]["_actual_step"] = closest

    # Always include last step
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
    print("CONFIG DIFFS (only leaf keys that vary across runs)")
    print("=" * 80)

    # Compute column widths
    col_width = 22
    key_width = max(len(k) for k in diff_keys) + 2 if diff_keys else 40

    header = f"{'key':<{key_width}s}"
    for i, r in enumerate(runs):
        header += f" | R{i:<{col_width - 4}}"
    print(header)
    print("-" * len(header))

    for k in diff_keys:
        row = f"{k:<{key_width}s}"
        for cfg in all_configs:
            v = cfg.get(k, "—")
            row += f" | {str(v):<{col_width - 3}s}"
        print(row)

    # --- Metrics at checkpoints ---
    print()
    print("=" * 80)
    print("METRICS AT CHECKPOINTS")
    print("=" * 80)

    for i, r in enumerate(runs):
        last_step = r.lastHistoryStep if r.lastHistoryStep else "?"
        print(f"\n--- R{i}: {r.name} (state={r.state}, steps={last_step}) ---")

        # Show only differing config as compact summary
        diff_cfg = {k.split(".")[-1]: all_configs[i].get(k, "—") for k in diff_keys}
        print(f"    Config: {diff_cfg}")

        data = get_metrics_at_steps(r, CHECKPOINT_STEPS, METRICS)
        if not data:
            print("    (no history)")
            continue

        step_labels = [str(s) for s in CHECKPOINT_STEPS] + ["last"]
        hdr = f"  {'metric':<35s}"
        for sl in step_labels:
            hdr += f" | {sl:>10s}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))

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

    # --- Summary table: val/acc at last step ---
    print()
    print("=" * 80)
    print("SUMMARY: val/acc at last step")
    print("=" * 80)
    print(f"{'Run':<5s} {'k_tgt':<7s} {'log_t':<7s} {'corr':<7s} {'val/acc':>10s} {'step':>6s}")
    print("-" * 50)
    for i, r in enumerate(runs):
        cfg = all_configs[i]
        k_tgt = cfg.get("masking.latent.k_tgt_min", "?")
        logt = cfg.get("masking.latent.log_transform", False)
        corr = cfg.get("masking.latent.correlation_loss", False)
        data = get_metrics_at_steps(r, [], METRICS)
        acc = fmt_val(data.get("last", {}).get("inline_eval/val_acc1"))
        step = data.get("last", {}).get("_actual_step", "?")
        print(f"R{i:<4} {str(k_tgt):<7s} {str(logt):<7s} {str(corr):<7s} {acc:>10s} {str(step):>6s}")


if __name__ == "__main__":
    main()
