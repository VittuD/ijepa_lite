#!/usr/bin/env python3
"""Fetch last-step metrics for the most recent N wandb runs.

Output is plain key=value, LLM-ingestible. No pretty formatting.

Usage:
    python scripts/fetch_wandb_runs.py [--n 4] [--project ijepa-lite] [--entity vitturini-davide]
"""
import argparse
import sys

import wandb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=4)
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
        print("no runs found")
        sys.exit(1)

    for i, r in enumerate(runs):
        print(f"run={i} name={r.name} state={r.state} steps={r.lastHistoryStep}")
        try:
            history = list(r.scan_history())
        except Exception:
            try:
                history = list(r.history(pandas=False))
            except Exception:
                print("  error: could not fetch history")
                continue

        if not history:
            print("  no history")
            continue

        last = history[-1]
        for k, v in sorted(last.items()):
            print(f"  {k}={v}")
        print()


if __name__ == "__main__":
    main()
