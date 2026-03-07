#!/bin/bash
#SBATCH --job-name=viz-goldilocks
#SBATCH --account=IscrC_TBoneAI
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=sbatch_logs/viz-goldilocks/%x_%j.out
#SBATCH --error=sbatch_logs/viz-goldilocks/%x_%j.err

# ---------------------------------------------------------------------------
# Visualize GoldilocksTeacherMasker scoring and ctx/tgt assignments.
#
# Panels per image (left to right):
#   [original | soft score heatmap (cold=low, hot=high) | hard assignment (ctx=blue, tgt=red)]
#
# The soft score heatmap is the primary diagnostic. With high K/N (e.g. K=94/N=144),
# the hard assignment is near-trivial — the score heatmap carries the signal.
#
# Output per dataset (STL-10, Food101):
#   <out_dir>/<dataset>_train.png           — image grid
#   <out_dir>/<dataset>_test.png            — image grid
#   <out_dir>/<dataset>_avg_score.png       — mean p_tgt per position (positional bias check)
#   <out_dir>/<dataset>_per_class_score.png — mean p_tgt per class (content-adaptivity check)
#
# Numeric output (stdout):
#   marginal_score_std   — near 0 = uniform marginals = no positional bias
#   p_tgt_score_std      — high = scores vary across images = content-adaptive
#   inter-class score std — > 0.05 = content-adaptive scoring across classes
#
# Checkpoint notes:
#   - Fixed-k checkpoints (goldilocks_fixed_k): k_tgt_eval is already 94, no --k-tgt needed.
#   - Variable-k checkpoints (goldilocks.yaml, k∈[4,72]): k_tgt_eval=17 by default.
#     Use --k-tgt 94 to visualize with a larger budget, or omit to see the 17-tgt eval mode.
#
# Run:
#   sbatch slurm/goldilocks/visualize_goldilocks.sbatch
# ---------------------------------------------------------------------------

set -euo pipefail

module purge
module load profile/deeplrn
module load python/3.11.6--gcc--8.5.0

source $WORK/vitturini/ijepa_venv/bin/activate

nvidia-smi -L || true

mkdir -p sbatch_logs/viz-goldilocks

# --- Single checkpoint (default) ---
export PRETRAIN_CKPT="$WORK/vitturini/outputs/2026-03-06/15-52-28_stl10_vits_ps8_goldilocks/rank0/checkpoints/last.pt"

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python ijepa_lite/hacky_visualize_goldilocks.py \
    --ckpt      "$PRETRAIN_CKPT" \
    --k-tgt     36 \
    --data-root "$FAST/datasets/" \
    --out-dir   goldilocks_viz_4blks_gz_5000ep_36p \
    --n         500 \
    --device    cuda
