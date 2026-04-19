#!/bin/bash
#SBATCH --job-name=masker-only-amin
#SBATCH --account=IscrC_TBoneAI
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#SBATCH --output=sbatch_logs/masker-only-amin/%x_%A_%a.out
#SBATCH --error=sbatch_logs/masker-only-amin/%x_%A_%a.err

# ---------------------------------------------------------------------------
# Sweep alpha_min to find a floor that prevents the nCtx=1 MI-rate collapse
# while keeping both encoder conditions comparable.
#
# Array layout (encoder × alpha_min):
#   idx  encoder       alpha_min   min_nCtx   out_dir
#    0   joint         0.15        ~22        masker_only_joint_amin015
#    1   joint         0.25        ~36        masker_only_joint_amin025
#    2   vanilla       0.15        ~22        masker_only_vanilla_amin015
#    3   vanilla       0.25        ~36        masker_only_vanilla_amin025
#
# Both conditions use the vanilla predictor as a warm start (broader ctx/tgt
# coverage from random multiblock masking).
#
# Submit:
#   sbatch slurm/mi_ablations/masker_only_alpha_min_sweep.sbatch
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
export VANILLA_CKPT="$WORK/vitturini/outputs/2026-02-21/18-28-32_stl10_vits_ps8_comp4/rank0/checkpoints/last.pt"

ENCODER_CKPTS=("$MI_COUPLED_CKPT" "$MI_COUPLED_CKPT" "$VANILLA_CKPT" "$VANILLA_CKPT")
ALPHA_MINS=(0.15 0.25 0.15 0.25)
OUT_DIRS=(
  masker_only_joint_amin015
  masker_only_joint_amin025
  masker_only_vanilla_amin015
  masker_only_vanilla_amin025
)

ENCODER_CKPT=${ENCODER_CKPTS[$SLURM_ARRAY_TASK_ID]}
ALPHA_MIN=${ALPHA_MINS[$SLURM_ARRAY_TASK_ID]}
OUT_DIR=${OUT_DIRS[$SLURM_ARRAY_TASK_ID]}

echo "Array task ${SLURM_ARRAY_TASK_ID}: encoder=${ENCODER_CKPT}"
echo "  alpha_min=${ALPHA_MIN}  out=${OUT_DIR}"

mkdir -p sbatch_logs/masker-only-amin

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python ijepa_lite/scripts/hacky/hacky_masker_only_train.py \
    --encoder-ckpt   "$ENCODER_CKPT" \
    --predictor-ckpt "$VANILLA_CKPT" \
    --data-root      "$FAST/datasets/" \
    --out-dir        "$OUT_DIR" \
    --alpha-min      "$ALPHA_MIN" \
    --epochs         400 \
    --batch-size     4096 \
    --lr             3e-4 \
    --weight-decay   1e-5 \
    --log-every      50 \
    --save-every     50 \
    --device         cuda
