#!/usr/bin/env python3
"""List all available history keys from a wandb run.

Usage:
    python scripts/list_wandb_keys.py [--project ijepa-lite] [--entity vitturini-davide] [--run-index 0]
"""
import argparse

import wandb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="ijepa-lite")
    parser.add_argument("--entity", default="vitturini-davide")
    parser.add_argument("--run-index", type=int, default=0,
                        help="Index of run (0 = most recent)")
    args = parser.parse_args()

    api = wandb.Api()
    runs = list(api.runs(
        f"{args.entity}/{args.project}",
        order="-created_at",
        per_page=args.run_index + 1,
    ))

    if len(runs) <= args.run_index:
        print(f"Only {len(runs)} runs found, index {args.run_index} out of range.")
        return

    run = runs[args.run_index]
    print(f"Run: {run.name} (id={run.id}, state={run.state})")
    print(f"lastHistoryStep: {run.lastHistoryStep}")
    print()

    # --- Summary keys ---
    print("=" * 60)
    print("SUMMARY KEYS (run.summary)")
    print("=" * 60)
    for k in sorted(run.summary.keys()):
        v = run.summary[k]
        vtype = type(v).__name__
        preview = str(v)[:80]
        print(f"  {k:<45s} ({vtype}) = {preview}")

    # --- History keys (from first few rows) ---
    print()
    print("=" * 60)
    print("HISTORY KEYS (from scan_history, first 5 rows)")
    print("=" * 60)
    all_keys = set()
    try:
        for i, row in enumerate(run.scan_history()):
            all_keys.update(row.keys())
            if i >= 4:
                break
    except Exception as e:
        print(f"  scan_history failed: {e}")
        print("  Trying run.history() fallback...")
        try:
            for i, row in enumerate(run.history(pandas=False)):
                all_keys.update(row.keys())
                if i >= 4:
                    break
        except Exception as e2:
            print(f"  history() also failed: {e2}")

    if all_keys:
        for k in sorted(all_keys):
            print(f"  {k}")
    else:
        print("  (no keys found)")

    # --- Config keys (flattened) ---
    print()
    print("=" * 60)
    print("CONFIG KEYS (flattened)")
    print("=" * 60)

    def flatten(d, prefix=""):
        out = {}
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                out.update(flatten(v, key))
            else:
                out[key] = v
        return out

    flat_cfg = flatten(run.config)
    for k in sorted(flat_cfg.keys()):
        print(f"  {k:<55s} = {flat_cfg[k]}")


if __name__ == "__main__":
    main()
