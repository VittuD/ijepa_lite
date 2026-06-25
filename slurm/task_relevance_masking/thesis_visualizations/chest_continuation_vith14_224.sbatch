#!/bin/bash
#SBATCH --job-name=viz-trm-thesis-224
#SBATCH --account=IscrC_TBoneAI
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=sbatch_logs/task_relevance_masking/thesis_visualizations/%x_%j.out
#SBATCH --error=sbatch_logs/task_relevance_masking/thesis_visualizations/%x_%j.err

# ---------------------------------------------------------------------------
# Thesis token-embedding PCA visualizations for the ChestMNIST ViT-H/14
# task-relevance continuation checkpoints at the native 224x224 resolution:
#
#   0  chest_vanilla_multiblock_last
#   1  chest_affinity_novelty_last
#   2  chest_random_split_50_last
#   3  chest_upper_half_random_split_50_last
#
# The single job runs all checkpoints sequentially and writes each checkpoint
# under:
#
#   ${WORK}/vitturini/token_embedding_viz/task_relevance_thesis_visualizations_224
#
# Submit:
#   mkdir -p sbatch_logs/task_relevance_masking/thesis_visualizations
#   sbatch slurm/task_relevance_masking/thesis_visualizations/chest_continuation_vith14_224.sbatch
#
# Useful overrides:
#   SAMPLE_MODE=first sbatch ...
#   N_SAMPLES=8 sbatch ...
#   FEATURE_MODE=last4_mean sbatch ...
#   OUT_ROOT=/path/to/output CKPT_START=1 CKPT_END=2 sbatch ...
# ---------------------------------------------------------------------------

set -euo pipefail

module purge
module load profile/deeplrn
module load python/3.11.6--gcc--8.5.0
module load imagenet

source "${WORK}/vitturini/ijepa_venv/bin/activate"

export WANDB_MODE=offline
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

nvidia-smi -L || true

REPO_ROOT="${REPO_ROOT:-${WORK}/vitturini/ijepa_lite}"
IN1K_ROOT="${IN1K_ROOT:-${IMAGENET2012_HOME}}"
DATA_ROOT="${DATA_ROOT:-${FAST:-/scratch}/datasets}"
OUT_ROOT="${OUT_ROOT:-${WORK}/vitturini/token_embedding_viz/task_relevance_thesis_visualizations_224}"

N_SAMPLES="${N_SAMPLES:-12}"
SAMPLE_MODE="${SAMPLE_MODE:-random}"
SEED="${SEED:-0}"
FEATURE_MODE="${FEATURE_MODE:-final}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-2}"

CKPT_LABELS=(
  chest_vanilla_multiblock_last
  chest_affinity_novelty_last
  chest_random_split_50_last
  chest_upper_half_random_split_50_last
)

CKPT_RELS=(
  outputs/2026-06-08/13-35-55_claim_chestmnist_224_vith14_vanilla_multiblock_from_meta_step1170_b2048/rank0/checkpoints/last.pt
  outputs/2026-06-08/13-35-55_claim_chestmnist_224_vith14_mi_affinity_novelty_from_meta_step1170_b2048/rank0/checkpoints/last.pt
  outputs/2026-06-10/19-51-24_claim_chestmnist_224_vith14_random_split_50_from_meta_step1170_b2048/rank0/checkpoints/last.pt
  outputs/2026-06-10/20-14-29_claim_chestmnist_224_vith14_upper_half_random_split_50_from_meta_step1170_b2048/rank0/checkpoints/last.pt
)

DATASETS=(
  imagenet:val
  chestmnist:val
  rsna_pneumonia_detection:val
  siimacr_pneumothorax:val
)

METHODS=(
  kmeans
  gmm
)

CLUSTER_K=(
  3
  4
  6
)

echo "REPO_ROOT=${REPO_ROOT}"
echo "IN1K_ROOT=${IN1K_ROOT}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "N_SAMPLES=${N_SAMPLES}"
echo "SAMPLE_MODE=${SAMPLE_MODE}"
echo "SEED=${SEED}"
echo "FEATURE_MODE=${FEATURE_MODE}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "NUM_WORKERS=${NUM_WORKERS}"
echo "CKPT_START=${CKPT_START:-0}"
echo "CKPT_END=${CKPT_END:-$(( ${#CKPT_LABELS[@]} - 1 ))}"

cd "${REPO_ROOT}"

START_IDX="${CKPT_START:-0}"
END_IDX="${CKPT_END:-$(( ${#CKPT_LABELS[@]} - 1 ))}"
TOTAL_CKPTS="${#CKPT_LABELS[@]}"

if (( START_IDX < 0 || START_IDX >= TOTAL_CKPTS )); then
  echo "Invalid CKPT_START=${START_IDX}; expected 0..$((TOTAL_CKPTS - 1))"
  exit 1
fi
if (( END_IDX < START_IDX || END_IDX >= TOTAL_CKPTS )); then
  echo "Invalid CKPT_END=${END_IDX}; expected ${START_IDX}..$((TOTAL_CKPTS - 1))"
  exit 1
fi

for idx in $(seq "${START_IDX}" "${END_IDX}"); do
  CKPT_LABEL="${CKPT_LABELS[$idx]}"
  CKPT_PATH="${WORK}/vitturini/${CKPT_RELS[$idx]}"
  OUT_DIR="${OUT_ROOT}/${CKPT_LABEL}_${SLURM_JOB_ID}_${idx}"

  if [[ ! -f "${CKPT_PATH}" ]]; then
    echo "Checkpoint not found: ${CKPT_PATH}"
    exit 1
  fi

  mkdir -p "${OUT_DIR}"

  CMD=(
    python -m ijepa_lite.scripts.visualize_token_embeddings
    --repo-root "${REPO_ROOT}"
    --data-root "${DATA_ROOT}"
    --dataset-root "imagenet=${IN1K_ROOT}"
    --dataset-root "chestmnist=${DATA_ROOT}"
    --dataset-root "rsna_pneumonia_detection=${DATA_ROOT}"
    --dataset-root "siimacr_pneumothorax=${DATA_ROOT}"
    --checkpoint "${CKPT_LABEL}=${CKPT_PATH}"
    --out-dir "${OUT_DIR}"
    --experiment downstream_mlp_earlystop
    --feature-mode "${FEATURE_MODE}"
    --n-samples "${N_SAMPLES}"
    --sample-mode "${SAMPLE_MODE}"
    --seed "${SEED}"
    --batch-size "${BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --model-arch vit_huge
    --model-image-size 224
    --model-patch-size 14
    --model-embed-dim 1280
    --model-num-heads 16
    --model-depth 32
    --model-use-cls-token false
    --methods "${METHODS[@]}"
    --cluster-k "${CLUSTER_K[@]}"
  )

  for dataset in "${DATASETS[@]}"; do
    CMD+=(--dataset "${dataset}")
  done

  echo "---- checkpoint ${idx}/${TOTAL_CKPTS}: ${CKPT_LABEL} ----"
  echo "CKPT_PATH=${CKPT_PATH}"
  echo "OUT_DIR=${OUT_DIR}"
  echo "CMD=${CMD[*]}"

  srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" "${CMD[@]}"
done
