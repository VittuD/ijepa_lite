#!/usr/bin/env python3
"""
Analyze Goldilocks error distribution from a wandb run.

Checks whether log(patch_loss) is approximately symmetric — the key
assumption needed for the log-transform z-score fix (Option A).

Usage:
  # 1. Sync the offline run first:
  wandb sync path/to/wandb/offline-run-XXXXXXXX-YYYYYYYY

  # 2. Then analyze (use the run path printed after sync):
  python scripts/hacky/hacky_analyze_goldilocks_hist.py --run entity/project/run_id

  # Or by local wandb dir (attempts direct file read):
  python scripts/hacky/hacky_analyze_goldilocks_hist.py --run-dir path/to/wandb/offline-run-...

  # Options:
  --steps 10          # how many steps to sample (default: all, can be slow)
  --out-dir plots/    # where to save PNGs (default: goldilocks_hist_analysis/)
"""
import argparse
import math
import os
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Reconstruct approximate samples from wandb histogram bins/counts
# ---------------------------------------------------------------------------

def bins_to_samples(hist_dict: dict) -> np.ndarray:
    """
    Convert wandb histogram dict {bins: [...], values: [...]} to
    approximate samples using bin midpoints weighted by counts.
    """
    bins = np.array(hist_dict["bins"], dtype=np.float64)
    counts = np.array(hist_dict["values"], dtype=np.float64)

    # bins has len(counts)+1 edges
    midpoints = (bins[:-1] + bins[1:]) / 2.0

    # Expand: repeat each midpoint by its count
    int_counts = np.round(counts).astype(int)
    samples = np.repeat(midpoints, int_counts)
    return samples


# ---------------------------------------------------------------------------
# Distribution statistics
# ---------------------------------------------------------------------------

def dist_stats(samples: np.ndarray, name: str = "") -> dict:
    """Compute mean, std, skewness, kurtosis, median, IQR."""
    n = len(samples)
    if n < 4:
        return {}

    mu = np.mean(samples)
    sigma = np.std(samples, ddof=1)
    median = np.median(samples)

    centered = samples - mu
    if sigma > 1e-12:
        m3 = np.mean(centered ** 3)
        m4 = np.mean(centered ** 4)
        skew = m3 / sigma ** 3
        kurt = m4 / sigma ** 4 - 3.0
    else:
        skew = kurt = 0.0

    q25, q75 = np.percentile(samples, [25, 75])

    return {
        f"{name}mean": mu,
        f"{name}std": sigma,
        f"{name}median": median,
        f"{name}skewness": skew,
        f"{name}kurtosis": kurt,
        f"{name}q25": q25,
        f"{name}q75": q75,
        f"{name}iqr": q75 - q25,
        f"{name}n": n,
    }


def print_stats(stats: dict, header: str = ""):
    if header:
        print(f"\n{'=' * 60}")
        print(f"  {header}")
        print(f"{'=' * 60}")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k:30s} = {v:.6f}")
        else:
            print(f"  {k:30s} = {v}")


# ---------------------------------------------------------------------------
# Symmetry tests
# ---------------------------------------------------------------------------

def symmetry_report(samples: np.ndarray, name: str = ""):
    """Quantitative symmetry assessment."""
    stats = dist_stats(samples, name)
    print_stats(stats, f"{name}Distribution")

    skew = stats.get(f"{name}skewness", 0.0)
    n = len(samples)

    # Standard error of skewness (for approximate z-test)
    se_skew = math.sqrt(6.0 / n) if n > 6 else 1.0
    z_skew = skew / se_skew if se_skew > 0 else 0.0

    print(f"\n  Skewness z-test: z = {z_skew:.2f}  "
          f"(|z| < 2 => consistent with symmetric)")
    if abs(z_skew) < 2:
        print(f"  => PASS: {name}distribution is consistent with symmetry")
    else:
        print(f"  => FAIL: {name}distribution is significantly skewed "
              f"({'right' if skew > 0 else 'left'})")

    # Nonparametric symmetry: compare distances of quantiles from median
    median = np.median(samples)
    for p in (10, 25):
        q_lo = np.percentile(samples, p)
        q_hi = np.percentile(samples, 100 - p)
        d_lo = median - q_lo
        d_hi = q_hi - median
        ratio = d_hi / d_lo if d_lo > 1e-12 else float("inf")
        sym = "symmetric" if 0.7 < ratio < 1.43 else "asymmetric"
        print(f"  p{p}/p{100-p} distance ratio from median: "
              f"{ratio:.3f}  ({sym})")

    return stats


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(
    raw_samples: np.ndarray,
    log_samples: np.ndarray,
    step: int,
    out_dir: Path,
):
    """Side-by-side histograms + QQ plots for raw vs log errors."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy import stats as sp_stats
    except ImportError:
        print("  (matplotlib/scipy not available — skipping plots)")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Goldilocks Error Distribution — step {step}", fontsize=14)

    # Top left: raw histogram
    ax = axes[0, 0]
    ax.hist(raw_samples, bins=80, density=True, alpha=0.7, color="steelblue",
            edgecolor="none")
    ax.set_title("Raw patch_loss")
    ax.set_xlabel("error")
    ax.set_ylabel("density")
    skew_raw = float(np.mean((raw_samples - raw_samples.mean()) ** 3)
                     / (raw_samples.std() ** 3 + 1e-12))
    ax.text(0.97, 0.95, f"skew={skew_raw:.2f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    # Top right: log histogram
    ax = axes[0, 1]
    ax.hist(log_samples, bins=80, density=True, alpha=0.7, color="darkorange",
            edgecolor="none")
    ax.set_title("log(patch_loss)")
    ax.set_xlabel("log(error)")
    ax.set_ylabel("density")
    skew_log = float(np.mean((log_samples - log_samples.mean()) ** 3)
                     / (log_samples.std() ** 3 + 1e-12))
    ax.text(0.97, 0.95, f"skew={skew_log:.2f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    # Overlay Gaussian fit
    x = np.linspace(log_samples.min(), log_samples.max(), 200)
    mu, sigma = log_samples.mean(), log_samples.std()
    if sigma > 1e-12:
        gaussian = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        ax.plot(x, gaussian, "k--", linewidth=1.5, label="Gaussian fit")
        ax.legend(fontsize=9)

    # Bottom left: QQ plot raw
    ax = axes[1, 0]
    sp_stats.probplot(raw_samples, dist="norm", plot=ax)
    ax.set_title("QQ plot — raw errors vs Normal")
    ax.get_lines()[0].set_markersize(2)

    # Bottom right: QQ plot log
    ax = axes[1, 1]
    sp_stats.probplot(log_samples, dist="norm", plot=ax)
    ax.set_title("QQ plot — log(errors) vs Normal")
    ax.get_lines()[0].set_markersize(2)

    plt.tight_layout()
    fname = out_dir / f"step_{step:06d}.png"
    plt.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved -> {fname}")


def plot_skewness_over_time(steps, raw_skews, log_skews, out_dir: Path):
    """Skewness trajectory: raw vs log over training."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, raw_skews, "o-", color="steelblue", markersize=3,
            label="raw errors")
    ax.plot(steps, log_skews, "s-", color="darkorange", markersize=3,
            label="log(errors)")
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.axhspan(-0.5, 0.5, color="green", alpha=0.08, label="symmetric zone")
    ax.set_xlabel("step")
    ax.set_ylabel("skewness")
    ax.set_title("Error Distribution Skewness Over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = out_dir / "skewness_trajectory.png"
    plt.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"\nSaved skewness trajectory -> {fname}")


def plot_kurtosis_over_time(steps, raw_kurts, log_kurts, out_dir: Path):
    """Kurtosis trajectory: raw vs log over training."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, raw_kurts, "o-", color="steelblue", markersize=3,
            label="raw errors")
    ax.plot(steps, log_kurts, "s-", color="darkorange", markersize=3,
            label="log(errors)")
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8,
               label="Gaussian (kurt=0)")
    ax.set_xlabel("step")
    ax.set_ylabel("excess kurtosis")
    ax.set_title("Error Distribution Kurtosis Over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = out_dir / "kurtosis_trajectory.png"
    plt.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved kurtosis trajectory -> {fname}")


# ---------------------------------------------------------------------------
# wandb data loading
# ---------------------------------------------------------------------------

def load_histograms_from_api(run_path: str):
    """
    Load goldilocks/error histograms from a synced wandb run.

    Returns list of (step, hist_dict) tuples — every available step.
    """
    import wandb
    api = wandb.Api()
    run = api.run(run_path)

    hist_key = "goldilocks/error"
    results = []

    print(f"Scanning run {run_path} for '{hist_key}' ...")
    for row in run.scan_history(keys=[hist_key, "_step"]):
        h = row.get(hist_key)
        if h is None:
            continue
        step = int(row.get("_step", len(results)))
        results.append((step, h))

    print(f"  Found {len(results)} histogram entries")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze Goldilocks error distribution symmetry"
    )
    parser.add_argument("--run", default="vitturini-davide/ijepa-lite/wmthmty0",
                        help="wandb run path: entity/project/run_id")
    parser.add_argument("--out-dir", default="goldilocks_hist_analysis")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plot generation")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = load_histograms_from_api(args.run)

    if not entries:
        print("No histogram data found. Did you sync the offline run?")
        print("  wandb sync path/to/wandb/offline-run-...")
        return

    # Collect trajectories
    all_steps = []
    raw_skews = []
    log_skews = []
    raw_kurts = []
    log_kurts = []

    for i, (step, hist_dict) in enumerate(entries):
        samples = bins_to_samples(hist_dict)
        if len(samples) < 10:
            print(f"  step {step}: too few samples ({len(samples)}), skipping")
            continue

        # Filter out zeros/negatives before log
        positive = samples[samples > 0]
        if len(positive) < 10:
            print(f"  step {step}: too few positive samples, skipping")
            continue

        log_samples = np.log(positive)

        print(f"\n{'─' * 60}")
        print(f"  Step {step}  ({i + 1}/{len(entries)})")
        print(f"{'─' * 60}")

        raw_stats = symmetry_report(samples, "raw/")
        log_stats = symmetry_report(log_samples, "log/")

        all_steps.append(step)
        raw_skews.append(raw_stats.get("raw/skewness", 0.0))
        log_skews.append(log_stats.get("log/skewness", 0.0))
        raw_kurts.append(raw_stats.get("raw/kurtosis", 0.0))
        log_kurts.append(log_stats.get("log/kurtosis", 0.0))

        if not args.no_plots:
            plot_comparison(samples, log_samples, step, out_dir)

    # Summary
    if all_steps:
        print(f"\n{'=' * 60}")
        print(f"  SUMMARY ACROSS TRAINING")
        print(f"{'=' * 60}")
        print(f"  Steps analyzed: {len(all_steps)}")
        print(f"")
        print(f"  Raw error skewness:  "
              f"mean={np.mean(raw_skews):.3f}  "
              f"std={np.std(raw_skews):.3f}  "
              f"range=[{np.min(raw_skews):.3f}, {np.max(raw_skews):.3f}]")
        print(f"  Log error skewness:  "
              f"mean={np.mean(log_skews):.3f}  "
              f"std={np.std(log_skews):.3f}  "
              f"range=[{np.min(log_skews):.3f}, {np.max(log_skews):.3f}]")
        print(f"")
        print(f"  Raw error kurtosis:  "
              f"mean={np.mean(raw_kurts):.3f}  "
              f"std={np.std(raw_kurts):.3f}  "
              f"range=[{np.min(raw_kurts):.3f}, {np.max(raw_kurts):.3f}]")
        print(f"  Log error kurtosis:  "
              f"mean={np.mean(log_kurts):.3f}  "
              f"std={np.std(log_kurts):.3f}  "
              f"range=[{np.min(log_kurts):.3f}, {np.max(log_kurts):.3f}]")
        print(f"")

        log_sym = np.mean(np.abs(log_skews)) < 0.5
        print(f"  VERDICT: log(error) is "
              f"{'approximately symmetric' if log_sym else 'NOT symmetric'} "
              f"(mean |skew| = {np.mean(np.abs(log_skews)):.3f}, "
              f"threshold = 0.5)")
        if log_sym:
            print(f"  => Log-transform z-score (Option A) is well-motivated.")
        else:
            print(f"  => Log-transform may not be sufficient.")
            print(f"     Consider quantile normalization or a different target.")

        if not args.no_plots:
            plot_skewness_over_time(all_steps, raw_skews, log_skews, out_dir)
            plot_kurtosis_over_time(all_steps, raw_kurts, log_kurts, out_dir)

    print(f"\nOutput in ./{out_dir}/")


if __name__ == "__main__":
    main()
