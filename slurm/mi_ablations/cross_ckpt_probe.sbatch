#!/bin/bash
#SBATCH --job-name=cross-ckpt-probe
#SBATCH --account=IscrC_TBoneAI
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=sbatch_logs/cross-ckpt-probe/%x_%A_%a.out
#SBATCH --error=sbatch_logs/cross-ckpt-probe/%x_%A_%a.err

# ---------------------------------------------------------------------------
# Phase-2 diagnostic: vanilla JEPA encoder + MI-coupled masker.
#
# If the masker produces content-adaptive assignments when paired with a
# better (vanilla-trained) encoder, the co-adaptation-trap hypothesis is
# confirmed (ign > center). If ign ≈ center again, the fixed-point is
# globally stable on these datasets regardless of encoder quality.
#
# Array layout (dataset × seed):
#   idx  dataset  seed
#    0   stl10      0
#    1   stl10     42
#    2   food101    0
#    3   food101   42
#
# Pools evaluated per job: ign + center (the key Phase-2 comparison).
# Add ctx / tgt / ctx+tgt to --pools if the full ablation is needed.
#
# Submit:
#   sbatch slurm/mi_ablations/cross_ckpt_probe.sbatch
# ---------------------------------------------------------------------------

set -euo pipefail

module purge
module load profile/deeplrn
module load python/3.11.6--gcc--8.5.0

source $WORK/vitturini/ijepa_venv/bin/activate

nvidia-smi -L || true

# Vanilla JEPA checkpoint (Phase 1 encoder — no MI masker)
export ENCODER_CKPT="$WORK/vitturini/outputs/<vanilla_run_dir>/rank0/checkpoints/last.pt"

# MI-coupled checkpoint (masker weights — same run used for pool ablation)
export MASKER_CKPT="$WORK/vitturini/outputs/2026-03-02/23-23-53_stl10_vits_ps8_mi_coupled_sr/rank0/checkpoints/last.pt"

DATASETS=(stl10 food101)
SEEDS=(0 42)

DATASET_IDX=$((SLURM_ARRAY_TASK_ID / 2))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % 2))

DATASET=${DATASETS[$DATASET_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "Array task ${SLURM_ARRAY_TASK_ID}: dataset=${DATASET}  seed=${SEED}"
echo "  encoder : $ENCODER_CKPT"
echo "  masker  : $MASKER_CKPT"

mkdir -p sbatch_logs/cross-ckpt-probe

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python ijepa_lite/scripts/hacky/hacky_cross_ckpt_probe.py \
    --encoder-ckpt "$ENCODER_CKPT" \
    --masker-ckpt  "$MASKER_CKPT" \
    --data-root    "$FAST/datasets/" \
    --dataset      "${DATASET}" \
    --pools        ign center \
    --seeds        "${SEED}" \
    --epochs       100 \
    --device       cuda
