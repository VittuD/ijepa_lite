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
# Visualize GoldilocksTeacherMasker ctx/tgt assignments.
# Works for both goldilocks and goldilocks_ema checkpoints — same masker class.
#
# Output per dataset (STL-10, Food101):
#   <out_dir>/<dataset>_train.png          — 500-image grid
#   <out_dir>/<dataset>_test.png           — 500-image grid
#   <out_dir>/<dataset>_avg_bins.png       — avg ctx/tgt heatmap (positional bias check)
#   <out_dir>/<dataset>_avg_bins_per_class.png — per-class tgt heatmap (content-adaptivity)
#
# Numeric output (stdout):
#   marginal_score_std  — near 0 = uniform = good; high = positional bias
#   p_tgt_score_std     — high = scores vary by image = content-adaptive
#   Inter-class tgt std — > 0.05 = different classes get different targets
#
# Modes:
#   Single checkpoint (default):
#     Set PRETRAIN_CKPT and run. Encoder and masker loaded from same file.
#
#   Split checkpoints (test masker on a different encoder):
#     Set ENCODER_CKPT and MASKER_CKPT separately.
#     Useful to test goldilocks_ema masker on vanilla JEPA encoder or vice-versa.
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
export PRETRAIN_CKPT="$WORK/vitturini/outputs/2026-03-05/15-56-38_stl10_vits_ps8_goldilocks/rank0/checkpoints/last.pt"

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python ijepa_lite/hacky_visualize_goldilocks.py \
    --ckpt      "$PRETRAIN_CKPT" \
    --data-root "$FAST/datasets/" \
    --out-dir   goldilocks_viz_4blks_500ep \
    --n         500 \
    --device    cuda

# --- Split checkpoints: test goldilocks_ema masker on vanilla encoder ---
# Uncomment after goldilocks_ema training completes.
#
# export ENCODER_CKPT="$WORK/vitturini/outputs/<vanilla_run>/rank0/checkpoints/last.pt"
# export MASKER_CKPT="$WORK/vitturini/outputs/<goldilocks_ema_run>/rank0/checkpoints/last.pt"
#
# srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
#   python ijepa_lite/hacky_visualize_goldilocks.py \
#     --encoder-ckpt "$ENCODER_CKPT" \
#     --masker-ckpt  "$MASKER_CKPT" \
#     --data-root    "$FAST/datasets/" \
#     --out-dir      goldilocks_viz_ema_split \
#     --n            500 \
#     --device       cuda
