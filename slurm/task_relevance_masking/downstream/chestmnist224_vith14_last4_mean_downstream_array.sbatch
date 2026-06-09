#!/bin/bash
#SBATCH --job-name=ijepa-trm-chest-ds-l4m
#SBATCH --account=IscrC_PRIMAL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=03:00:00
#SBATCH --array=0-4
#SBATCH --output=sbatch_logs/task_relevance_masking/downstream/chestmnist224_vith14_last4_mean/%x_%A_%a.out
#SBATCH --error=sbatch_logs/task_relevance_masking/downstream/chestmnist224_vith14_last4_mean/%x_%A_%a.err

# ---------------------------------------------------------------------------
# Downstream claim-check array for ChestMNIST-continued ViT-H/14 task-relevance
# runs.
#
# Each array item evaluates one last.pt checkpoint on:
#   - chestmnist
#   - pneumoniamnist
#   - siimacr_pneumothorax
#
# Uses early-stopped MLP probes with task.pool=last4_mean. SIIM is evaluated
# with foreground Dice as the primary metric via downstream_sweep overrides.
#
# Submit:
#   mkdir -p sbatch_logs/task_relevance_masking/downstream/chestmnist224_vith14_last4_mean
#   sbatch slurm/task_relevance_masking/downstream/chestmnist224_vith14_last4_mean_downstream_array.sbatch
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
DATA_ROOT="${DATA_ROOT:-${CHESTMNIST_ROOT:-${FAST:-/scratch}/datasets}}"
LOG_ROOT="${WORK}/vitturini/sbatch_logs/task_relevance_masking/downstream/chestmnist224_vith14_last4_mean"

CLASSIFICATION_BATCH_SIZE=512
CLASSIFICATION_NUM_WORKERS=1
SEGMENTATION_BATCH_SIZE=128
SEGMENTATION_NUM_WORKERS=1

DATASETS=(
  chestmnist
  pneumoniamnist
  siimacr_pneumothorax
)

CKPT_LABELS=(
  chest_semantic_pca_last
  chest_hmargcosbs_floor_last
  chest_hmarg_sketchorthcosbs_floor_last
  chest_hmargbs_floor_last
  chest_affinity_novelty_last
)

CKPT_RELS=(
  outputs/2026-06-08/16-36-21_claim_chestmnist_224_vith14_semantic_pca_from_meta_step1170_b2048/rank0/checkpoints/last.pt
  outputs/2026-06-08/13-37-50_claim_chestmnist_224_vith14_mi_hmargcosbs_floor_from_meta_step1170_b2048/rank0/checkpoints/last.pt
  outputs/2026-06-08/13-37-50_claim_chestmnist_224_vith14_mi_hmarg_sketchorthcosbs_floor_from_meta_step1170_b2048/rank0/checkpoints/last.pt
  outputs/2026-06-08/13-36-37_claim_chestmnist_224_vith14_mi_hmargbs_floor_from_meta_step1170_b2048/rank0/checkpoints/last.pt
  outputs/2026-06-08/13-35-55_claim_chestmnist_224_vith14_mi_affinity_novelty_from_meta_step1170_b2048/rank0/checkpoints/last.pt
)

TASK_ID="${SLURM_ARRAY_TASK_ID}"
CKPT_LABEL="${CKPT_LABELS[$TASK_ID]}"
CKPT_REL="${CKPT_RELS[$TASK_ID]}"
CKPT_PATH="${WORK}/vitturini/${CKPT_REL}"
LOG_DIR="${LOG_ROOT}/${CKPT_LABEL}"
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
  --classification-experiment downstream_mlp_earlystop
  --segmentation-experiment downstream_vocseg_mlp_earlystop
  --run-suffix task_relevance_chestmnist224_vith14_last4_mean
  --task-pool last4_mean
  --model-arch vit_huge
  --model-image-size 224
  --model-patch-size 14
  --model-embed-dim 1280
  --model-num-heads 16
  --model-depth 32
  --model-use-cls-token false
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
echo "CKPT_LABEL=${CKPT_LABEL}"
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
