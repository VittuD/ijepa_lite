#!/bin/bash
#SBATCH --job-name=ijepa-trm-in1k-ds-l4m
#SBATCH --account=IscrC_TBoneAI
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=sbatch_logs/task_relevance_masking/downstream/in1k224_vith14_last4_mean/%x_%A_%a.out
#SBATCH --error=sbatch_logs/task_relevance_masking/downstream/in1k224_vith14_last4_mean/%x_%A_%a.err

# ---------------------------------------------------------------------------
# Downstream claim-check array for IN1K-control ViT-H/14 task-relevance runs.
#
# Each array item evaluates a chunk of three last.pt checkpoints on:
#   - chestmnist
#   - rsna_pneumonia_detection
#   - siimacr_pneumothorax
#
# Uses early-stopped MLP probes with task.pool=last4_mean. RSNA is evaluated
# with the box-heatmap patch-AP probe; SIIM is evaluated with foreground Dice
# as the primary metric via downstream_sweep overrides.
#
# Submit:
#   mkdir -p sbatch_logs/task_relevance_masking/downstream/in1k224_vith14_last4_mean
#   sbatch slurm/task_relevance_masking/downstream/in1k224_vith14_last4_mean_downstream_array.sbatch
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
OUTPUT_ROOT="${WORK}/vitturini/outputs"
LOG_ROOT="${WORK}/vitturini/sbatch_logs/task_relevance_masking/downstream/in1k224_vith14_last4_mean"

CLASSIFICATION_BATCH_SIZE=512
CLASSIFICATION_NUM_WORKERS=1
HEATMAP_BATCH_SIZE=128
HEATMAP_NUM_WORKERS=2
SEGMENTATION_BATCH_SIZE=128
SEGMENTATION_NUM_WORKERS=2

DATASETS=(
  chestmnist
  rsna_pneumonia_detection
  siimacr_pneumothorax
)

CKPT_LABELS=(
  in1k_vanilla_multiblock_last
  in1k_affinity_novelty_last
  in1k_hmargbs_floor_last
  in1k_hmargcosbs_floor_last
  in1k_hmarg_sketchorthcosbs_floor_last
  in1k_semantic_pca_last
)

CKPT_RELS=(
  outputs/2026-06-08/17-48-36_claim_in1k_224_vith14_vanilla_multiblock_from_meta_step1170_b2048_chest_eval/rank0/checkpoints/last.pt
  outputs/2026-06-08/17-48-36_claim_in1k_224_vith14_mi_affinity_novelty_from_meta_step1170_b2048_chest_eval/rank0/checkpoints/last.pt
  outputs/2026-06-08/17-49-20_claim_in1k_224_vith14_mi_hmargbs_floor_from_meta_step1170_b2048_chest_eval/rank0/checkpoints/last.pt
  outputs/2026-06-08/17-49-45_claim_in1k_224_vith14_mi_hmargcosbs_floor_from_meta_step1170_b2048_chest_eval/rank0/checkpoints/last.pt
  outputs/2026-06-08/17-50-26_claim_in1k_224_vith14_mi_hmarg_sketchorthcosbs_floor_from_meta_step1170_b2048_chest_eval/rank0/checkpoints/last.pt
  outputs/2026-06-08/17-50-26_claim_in1k_224_vith14_semantic_pca_from_meta_step1170_b2048_chest_eval/rank0/checkpoints/last.pt
)

TASK_ID="${SLURM_ARRAY_TASK_ID}"
CHUNK_SIZE=3
START_IDX=$((TASK_ID * CHUNK_SIZE))
END_IDX=$((START_IDX + CHUNK_SIZE - 1))
TOTAL_CKPTS="${#CKPT_LABELS[@]}"
if (( END_IDX >= TOTAL_CKPTS )); then
  END_IDX=$((TOTAL_CKPTS - 1))
fi

LOG_DIR="${LOG_ROOT}/chunk_${TASK_ID}"
SUMMARY_PATH="${LOG_DIR}/chunk_${TASK_ID}_last4_mean_results_summary.txt"

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/chunk_${TASK_ID}_${SLURM_JOB_ID}.out")
exec 2> >(tee -a "${LOG_DIR}/chunk_${TASK_ID}_${SLURM_JOB_ID}.err" >&2)

CMD=(
  python -m torch.distributed.run --standalone --nproc_per_node "${NPROC_PER_NODE}" -m ijepa_lite.scripts.downstream_sweep
  --repo-root "$REPO_ROOT"
  --data-root "$DATA_ROOT"
  --output-root "$OUTPUT_ROOT"
  --summary-path "$SUMMARY_PATH"
  --logger-mode offline
  --continue-on-error
  --no-save-probe-checkpoints
  --classification-experiment downstream_mlp_earlystop
  --heatmap-experiment downstream_rsna_box_heatmap_mlp_fast
  --segmentation-experiment downstream_vocseg_mlp_earlystop
  --run-suffix task_relevance_in1k224_vith14_last4_mean
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
  --heatmap-batch-size "${HEATMAP_BATCH_SIZE}"
  --heatmap-num-workers "${HEATMAP_NUM_WORKERS}"
  --segmentation-batch-size "${SEGMENTATION_BATCH_SIZE}"
  --segmentation-num-workers "${SEGMENTATION_NUM_WORKERS}"
)

for idx in $(seq "${START_IDX}" "${END_IDX}"); do
  CKPT_LABEL="${CKPT_LABELS[$idx]}"
  CKPT_PATH="${WORK}/vitturini/${CKPT_RELS[$idx]}"
  if [[ ! -f "${CKPT_PATH}" ]]; then
    echo "Checkpoint not found: ${CKPT_PATH}"
    exit 1
  fi
  CMD+=(--checkpoint "${CKPT_LABEL}=${CKPT_PATH}")
done

for dataset in "${DATASETS[@]}"; do
  CMD+=(--dataset "$dataset")
done

echo "TASK_ID=${TASK_ID}"
echo "CHUNK_SIZE=${CHUNK_SIZE}"
echo "START_IDX=${START_IDX}"
echo "END_IDX=${END_IDX}"
echo "REPO_ROOT=${REPO_ROOT}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "LOG_DIR=${LOG_DIR}"
echo "SUMMARY_PATH=${SUMMARY_PATH}"
echo "CHECKPOINTS:"
for idx in $(seq "${START_IDX}" "${END_IDX}"); do
  echo "  ${CKPT_LABELS[$idx]}=${WORK}/vitturini/${CKPT_RELS[$idx]}"
done
echo "CLASSIFICATION_BATCH_SIZE=${CLASSIFICATION_BATCH_SIZE}"
echo "CLASSIFICATION_NUM_WORKERS=${CLASSIFICATION_NUM_WORKERS}"
echo "HEATMAP_BATCH_SIZE=${HEATMAP_BATCH_SIZE}"
echo "HEATMAP_NUM_WORKERS=${HEATMAP_NUM_WORKERS}"
echo "SEGMENTATION_BATCH_SIZE=${SEGMENTATION_BATCH_SIZE}"
echo "SEGMENTATION_NUM_WORKERS=${SEGMENTATION_NUM_WORKERS}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
nvidia-smi -L

"${CMD[@]}"
