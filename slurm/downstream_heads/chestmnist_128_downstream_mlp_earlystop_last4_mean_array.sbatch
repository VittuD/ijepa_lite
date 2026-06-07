#!/bin/bash
#SBATCH --job-name=ijepa-chest128-heads-l4m
#SBATCH --account=IscrC_PRIMAL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --array=0-9
#SBATCH --output=sbatch_logs/downstream_heads/chestmnist_128/%x_%A_%a.out
#SBATCH --error=sbatch_logs/downstream_heads/chestmnist_128/%x_%A_%a.err

# ---------------------------------------------------------------------------
# Early-stopped MLP downstream probes for ChestMNIST-continued 128x128 / ps8
# checkpoints, using task.pool=last4_mean.
#
# Each array item evaluates one checkpoint on:
#   - chestmnist
#   - pneumoniamnist
#   - siimacr_pneumothorax
#
# Checkpoints:
#   0  array_0_vanilla_multiblock            epoch_00169
#   1  array_0_vanilla_multiblock            last
#   2  array_1_mi_affinity_novelty           epoch_00169
#   3  array_1_mi_affinity_novelty           last
#   4  array_2_mi_hmargbs_floor              epoch_00169
#   5  array_2_mi_hmargbs_floor              last
#   6  array_3_mi_hmargcosbs_floor           epoch_00169
#   7  array_3_mi_hmargcosbs_floor           last
#   8  array_4_mi_hmarg_sketchorthcosbs_floor epoch_00169
#   9  array_4_mi_hmarg_sketchorthcosbs_floor last
#
# Submit:
#   mkdir -p sbatch_logs/downstream_heads/chestmnist_128
#   sbatch slurm/downstream_heads/chestmnist_128_downstream_mlp_earlystop_last4_mean_array.sbatch
# ---------------------------------------------------------------------------

set -euo pipefail

module purge
module load profile/deeplrn
module load python/3.11.6--gcc--8.5.0

source $WORK/vitturini/ijepa_venv/bin/activate

export WANDB_MODE=offline

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NPROC_PER_NODE=4

REPO_ROOT="${WORK}/vitturini/ijepa_lite"
DATA_ROOT="${FAST:-/scratch}/datasets"
LOG_ROOT="${WORK}/vitturini/sbatch_logs/downstream_heads/chestmnist_128"

CLASSIFICATION_BATCH_SIZE=512
CLASSIFICATION_NUM_WORKERS=1
SEGMENTATION_BATCH_SIZE=128
SEGMENTATION_NUM_WORKERS=1

DATASETS=(
  chestmnist
  pneumoniamnist
  siimacr_pneumothorax
)

METHODS=(
  array_0_vanilla_multiblock
  array_0_vanilla_multiblock
  array_1_mi_affinity_novelty
  array_1_mi_affinity_novelty
  array_2_mi_hmargbs_floor
  array_2_mi_hmargbs_floor
  array_3_mi_hmargcosbs_floor
  array_3_mi_hmargcosbs_floor
  array_4_mi_hmarg_sketchorthcosbs_floor
  array_4_mi_hmarg_sketchorthcosbs_floor
)

CKPT_TAGS=(
  epoch169
  last
  epoch169
  last
  epoch169
  last
  epoch169
  last
  epoch169
  last
)

CKPT_FILES=(
  epoch_00169.pt
  last.pt
  epoch_00169.pt
  last.pt
  epoch_00169.pt
  last.pt
  epoch_00169.pt
  last.pt
  epoch_00169.pt
  last.pt
)

RUN_ROOTS=(
  outputs/2026-06-06/20-13-24_chestmnist_128_vits_ps8_vanilla_multiblock_from_in1k96_e247_origlr_ema99925_nowarmup
  outputs/2026-06-06/20-13-24_chestmnist_128_vits_ps8_vanilla_multiblock_from_in1k96_e247_origlr_ema99925_nowarmup
  outputs/2026-06-06/20-13-24_chestmnist_128_vits_ps8_mi_affinity_novelty_from_in1k96_e247_origlr_ema99925_nowarmup
  outputs/2026-06-06/20-13-24_chestmnist_128_vits_ps8_mi_affinity_novelty_from_in1k96_e247_origlr_ema99925_nowarmup
  outputs/2026-06-06/20-13-24_chestmnist_128_vits_ps8_mi_hmargbs_floor_from_in1k96_e247_origlr_ema99925_nowarmup
  outputs/2026-06-06/20-13-24_chestmnist_128_vits_ps8_mi_hmargbs_floor_from_in1k96_e247_origlr_ema99925_nowarmup
  outputs/2026-06-06/20-13-24_chestmnist_128_vits_ps8_mi_hmargcosbs_floor_from_in1k96_e247_origlr_ema99925_nowarmup
  outputs/2026-06-06/20-13-24_chestmnist_128_vits_ps8_mi_hmargcosbs_floor_from_in1k96_e247_origlr_ema99925_nowarmup
  outputs/2026-06-06/20-13-24_chestmnist_128_vits_ps8_mi_hmarg_sketchorthcosbs_floor_from_in1k96_e247_origlr_ema99925_nowarmup
  outputs/2026-06-06/20-13-24_chestmnist_128_vits_ps8_mi_hmarg_sketchorthcosbs_floor_from_in1k96_e247_origlr_ema99925_nowarmup
)

TASK_ID="${SLURM_ARRAY_TASK_ID}"
METHOD="${METHODS[$TASK_ID]}"
CKPT_TAG="${CKPT_TAGS[$TASK_ID]}"
CKPT_FILE="${CKPT_FILES[$TASK_ID]}"
RUN_ROOT="${RUN_ROOTS[$TASK_ID]}"

CKPT_LABEL="${METHOD}_${CKPT_TAG}"
CKPT_PATH="${WORK}/vitturini/${RUN_ROOT}/rank0/checkpoints/${CKPT_FILE}"
LOG_DIR="${LOG_ROOT}/${METHOD}"
SUMMARY_PATH="${LOG_DIR}/${CKPT_LABEL}_last4_mean_results_summary.txt"

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/${CKPT_LABEL}_${SLURM_JOB_ID}_${TASK_ID}.out")
exec 2> >(tee -a "${LOG_DIR}/${CKPT_LABEL}_${SLURM_JOB_ID}_${TASK_ID}.err" >&2)

if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "Checkpoint not found: ${CKPT_PATH}"
  exit 1
fi

CMD=(
  python -m torch.distributed.run --standalone --nproc_per_node "${NPROC_PER_NODE}" -m ijepa_lite.scripts.downstream_sweep
  --repo-root "$REPO_ROOT"
  --data-root "$DATA_ROOT"
  --summary-path "$SUMMARY_PATH"
  --logger-mode offline
  --continue-on-error
  --classification-experiment in1k_96_vits_ps8_downstream_mlp_earlystop
  --segmentation-experiment in1k_96_vits_ps8_vocseg_probe_earlystop
  --run-suffix chestmnist128_es_probe_last4_mean
  --task-pool last4_mean
  --model-arch vit_small_16
  --model-image-size 128
  --model-patch-size 8
  --model-embed-dim 384
  --model-num-heads 6
  --model-depth 12
  --classification-batch-size "${CLASSIFICATION_BATCH_SIZE}"
  --classification-num-workers "${CLASSIFICATION_NUM_WORKERS}"
  --segmentation-batch-size "${SEGMENTATION_BATCH_SIZE}"
  --segmentation-num-workers "${SEGMENTATION_NUM_WORKERS}"
  --checkpoint "${CKPT_LABEL}=${CKPT_PATH}"
)

for dataset in "${DATASETS[@]}"; do
  CMD+=(--dataset "$dataset")
done

echo "TASK_ID=${TASK_ID}"
echo "METHOD=${METHOD}"
echo "CKPT_TAG=${CKPT_TAG}"
echo "REPO_ROOT=${REPO_ROOT}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "LOG_DIR=${LOG_DIR}"
echo "SUMMARY_PATH=${SUMMARY_PATH}"
echo "CKPT_PATH=${CKPT_PATH}"
echo "CLASSIFICATION_BATCH_SIZE=${CLASSIFICATION_BATCH_SIZE}"
echo "CLASSIFICATION_NUM_WORKERS=${CLASSIFICATION_NUM_WORKERS}"
echo "SEGMENTATION_BATCH_SIZE=${SEGMENTATION_BATCH_SIZE}"
echo "SEGMENTATION_NUM_WORKERS=${SEGMENTATION_NUM_WORKERS}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
nvidia-smi -L

"${CMD[@]}"
