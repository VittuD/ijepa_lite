#!/bin/bash
#SBATCH --job-name=viz-trm-vith14-last
#SBATCH --account=IscrC_TBoneAI
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --array=0-10
#SBATCH --output=sbatch_logs/viz_token_emb/task_relevance_vith14_downstream_last/%x_%A_%a.out
#SBATCH --error=sbatch_logs/viz_token_emb/task_relevance_vith14_downstream_last/%x_%A_%a.err

# ---------------------------------------------------------------------------
# Token-embedding PCA and clustering visualization for the ViT-H/14
# task-relevance checkpoints used by the downstream claim-check arrays.
#
# Each array item runs one last.pt checkpoint:
#   0   chest_vanilla_multiblock_last
#   1   chest_affinity_novelty_last
#   2   chest_hmargcosbs_floor_last
#   3   chest_hmarg_sketchorthcosbs_floor_last
#   4   chest_semantic_pca_last
#   5   in1k_vanilla_multiblock_last
#   6   in1k_affinity_novelty_last
#   7   in1k_hmargbs_floor_last
#   8   in1k_hmargcosbs_floor_last
#   9   in1k_hmarg_sketchorthcosbs_floor_last
#   10  in1k_semantic_pca_last
#
# Default visualization datasets mirror the downstream medical readouts:
#   - chestmnist:val
#   - rsna_pneumonia_detection:val
#   - siimacr_pneumothorax:val
#
# Submit:
#   mkdir -p sbatch_logs/viz_token_emb/task_relevance_vith14_downstream_last
#   sbatch slurm/viz_token_emb/task_relevance_vith14_downstream_last_array.sbatch
#
# Useful overrides:
#   N_SAMPLES=24 SAMPLE_MODE=first sbatch ...
#   FEATURE_MODE=last4_mean sbatch ...
#   OUT_ROOT=/path/to/output sbatch ...
# ---------------------------------------------------------------------------

set -euo pipefail

module purge
module load profile/deeplrn
module load python/3.11.6--gcc--8.5.0

source "${WORK}/vitturini/ijepa_venv/bin/activate"

export WANDB_MODE=offline
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

nvidia-smi -L || true

REPO_ROOT="${REPO_ROOT:-${WORK}/vitturini/ijepa_lite}"
DATA_ROOT="${DATA_ROOT:-${FAST:-/scratch}/datasets}"
OUT_ROOT="${OUT_ROOT:-${WORK}/vitturini/token_embedding_viz/task_relevance_vith14_downstream_last}"

N_SAMPLES="${N_SAMPLES:-12}"
SAMPLE_MODE="${SAMPLE_MODE:-random}"
SEED="${SEED:-0}"
FEATURE_MODE="${FEATURE_MODE:-final}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-2}"

CKPT_LABELS=(
  chest_vanilla_multiblock_last
  chest_affinity_novelty_last
  chest_hmargcosbs_floor_last
  chest_hmarg_sketchorthcosbs_floor_last
  chest_semantic_pca_last
  in1k_vanilla_multiblock_last
  in1k_affinity_novelty_last
  in1k_hmargbs_floor_last
  in1k_hmargcosbs_floor_last
  in1k_hmarg_sketchorthcosbs_floor_last
  in1k_semantic_pca_last
)

CKPT_RELS=(
  outputs/2026-06-08/13-35-55_claim_chestmnist_224_vith14_vanilla_multiblock_from_meta_step1170_b2048/rank0/checkpoints/last.pt
  outputs/2026-06-08/13-35-55_claim_chestmnist_224_vith14_mi_affinity_novelty_from_meta_step1170_b2048/rank0/checkpoints/last.pt
  outputs/2026-06-08/13-37-50_claim_chestmnist_224_vith14_mi_hmargcosbs_floor_from_meta_step1170_b2048/rank0/checkpoints/last.pt
  outputs/2026-06-08/13-37-50_claim_chestmnist_224_vith14_mi_hmarg_sketchorthcosbs_floor_from_meta_step1170_b2048/rank0/checkpoints/last.pt
  outputs/2026-06-08/16-36-21_claim_chestmnist_224_vith14_semantic_pca_from_meta_step1170_b2048/rank0/checkpoints/last.pt
  outputs/2026-06-08/17-48-36_claim_in1k_224_vith14_vanilla_multiblock_from_meta_step1170_b2048_chest_eval/rank0/checkpoints/last.pt
  outputs/2026-06-08/17-48-36_claim_in1k_224_vith14_mi_affinity_novelty_from_meta_step1170_b2048_chest_eval/rank0/checkpoints/last.pt
  outputs/2026-06-08/17-49-20_claim_in1k_224_vith14_mi_hmargbs_floor_from_meta_step1170_b2048_chest_eval/rank0/checkpoints/last.pt
  outputs/2026-06-08/17-49-45_claim_in1k_224_vith14_mi_hmargcosbs_floor_from_meta_step1170_b2048_chest_eval/rank0/checkpoints/last.pt
  outputs/2026-06-08/17-50-26_claim_in1k_224_vith14_mi_hmarg_sketchorthcosbs_floor_from_meta_step1170_b2048_chest_eval/rank0/checkpoints/last.pt
  outputs/2026-06-08/17-50-26_claim_in1k_224_vith14_semantic_pca_from_meta_step1170_b2048_chest_eval/rank0/checkpoints/last.pt
)

DATASETS=(
  chestmnist:val
  rsna_pneumonia_detection:val
  siimacr_pneumothorax:val
)

METHODS=(
  kmeans
  agglomerative
  spectral
  gmm
)

CLUSTER_K=(
  3
  4
  6
)

DBSCAN_EPS=(
  1.5
  2.5
  4.0
)

TASK_ID="${SLURM_ARRAY_TASK_ID}"
CKPT_LABEL="${CKPT_LABELS[$TASK_ID]}"
CKPT_PATH="${WORK}/vitturini/${CKPT_RELS[$TASK_ID]}"
OUT_DIR="${OUT_ROOT}/${CKPT_LABEL}_${SLURM_JOB_ID}_${TASK_ID}"

if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "Checkpoint not found: ${CKPT_PATH}"
  exit 1
fi

mkdir -p "${OUT_DIR}"

CMD=(
  python -m ijepa_lite.scripts.visualize_token_embeddings
  --repo-root "${REPO_ROOT}"
  --data-root "${DATA_ROOT}"
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
  --dbscan-eps "${DBSCAN_EPS[@]}"
)

for dataset in "${DATASETS[@]}"; do
  CMD+=(--dataset "${dataset}")
done

echo "TASK_ID=${TASK_ID}"
echo "REPO_ROOT=${REPO_ROOT}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "CKPT_LABEL=${CKPT_LABEL}"
echo "CKPT_PATH=${CKPT_PATH}"
echo "OUT_DIR=${OUT_DIR}"
echo "N_SAMPLES=${N_SAMPLES}"
echo "SAMPLE_MODE=${SAMPLE_MODE}"
echo "FEATURE_MODE=${FEATURE_MODE}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "NUM_WORKERS=${NUM_WORKERS}"
echo "CMD=${CMD[*]}"

cd "${REPO_ROOT}"
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" "${CMD[@]}"
