#!/bin/bash
#SBATCH --job-name=masker-only-train
#SBATCH --account=IscrC_TBoneAI
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --array=0-1
#SBATCH --output=sbatch_logs/masker-only-train/%x_%A_%a.out
#SBATCH --error=sbatch_logs/masker-only-train/%x_%A_%a.err

# ---------------------------------------------------------------------------
# Phase-2 diagnostic: train masker + predictor with a FROZEN encoder.
#
# Array layout:
#   idx  encoder                     out_dir
#    0   MI-coupled (joint)          masker_only_joint_enc   ← main diagnostic
#    1   vanilla JEPA                masker_only_vanilla_enc ← control
#
# Key metric: spatial_pos_std
#   Near 0 → purely positional masker (no content adaptation).
#   Higher  → content-adaptive masking re-emerging from frozen encoder.
#
# If idx=0 converges to high spatial_pos_std:
#   the jointly-trained encoder IS providing content-informative features
#   that a fresh masker can learn to exploit → Phase-2 hypothesis supported.
# If both converge near 0:
#   the horizontal-stripes fixed point is globally stable regardless of
#   encoder quality → co-adaptation artifact, not a real signal.
#
# Submit:
#   sbatch slurm/mi_ablations/masker_only_train.sbatch
# ---------------------------------------------------------------------------

set -euo pipefail

module purge
module load profile/deeplrn
module load python/3.11.6--gcc--8.5.0

source $WORK/vitturini/ijepa_venv/bin/activate

nvidia-smi -L || true

# MI-coupled checkpoint (encoder + masker jointly trained)
export MI_COUPLED_CKPT="$WORK/vitturini/outputs/2026-03-02/23-23-53_stl10_vits_ps8_mi_coupled_sr/rank0/checkpoints/last.pt"

# Vanilla JEPA checkpoint (fill in before submitting)
export VANILLA_CKPT="$WORK/vitturini/outputs/<vanilla_run_dir>/rank0/checkpoints/last.pt"

# Each condition warm-starts the predictor from the same checkpoint as its encoder.
# idx=0: MI-coupled predictor  (was co-trained with MI-coupled encoder)
# idx=1: vanilla predictor     (broad ctx/tgt coverage from random multiblock masking)
ENCODER_CKPTS=("$MI_COUPLED_CKPT" "$VANILLA_CKPT")
PREDICTOR_CKPTS=("$MI_COUPLED_CKPT" "$VANILLA_CKPT")
OUT_DIRS=(masker_only_joint_enc masker_only_vanilla_enc)

ENCODER_CKPT=${ENCODER_CKPTS[$SLURM_ARRAY_TASK_ID]}
PREDICTOR_CKPT=${PREDICTOR_CKPTS[$SLURM_ARRAY_TASK_ID]}
OUT_DIR=${OUT_DIRS[$SLURM_ARRAY_TASK_ID]}

echo "Array task ${SLURM_ARRAY_TASK_ID}: encoder=${ENCODER_CKPT}  predictor=${PREDICTOR_CKPT}  out=${OUT_DIR}"

mkdir -p sbatch_logs/masker-only-train

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python hacky_masker_only_train.py \
    --encoder-ckpt   "$ENCODER_CKPT" \
    --predictor-ckpt "$PREDICTOR_CKPT" \
    --data-root      "$FAST/datasets/" \
    --out-dir        "$OUT_DIR" \
    --epochs         400 \
    --batch-size     512 \
    --lr             3e-4 \
    --weight-decay   1e-5 \
    --log-every      50 \
    --save-every     50 \
    --device         cuda
