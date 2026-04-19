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
# Visualize masker scoring and ctx/tgt assignments.
#
# Uses the build pipeline to load the correct masker type from the experiment
# config. Automatically detects 2-way (Goldilocks) vs 3-way (MI) maskers
# and adjusts the middle panel accordingly.
#
# Panels per image (left to right):
#   2-way: [original | soft score heatmap (cold=low, hot=high) | hard assignment]
#   3-way: [original | soft 3-way (ctx=blue, tgt=red, ign=grey) | hard assignment]
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

export PRETRAIN_CKPT="$WORK/vitturini/outputs/2026-03-07/19-39-54_stl10_vits_ps8_goldilocks/rank0/checkpoints/last.pt"

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python ijepa_lite/scripts/hacky/hacky_visualize_goldilocks.py \
    --ckpt       "$PRETRAIN_CKPT" \
    --experiment stl10_vits_ps8_goldilocks \
    --k-tgt      36 \
    --data-root  "$FAST/datasets/" \
    --out-dir    goldilocks_viz \
    --n          500 \
    --device     cuda
