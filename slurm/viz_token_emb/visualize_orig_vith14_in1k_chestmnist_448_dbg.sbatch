#!/bin/bash
#SBATCH --job-name=viz-orig-vith14-448
#SBATCH --account=IscrC_PRIMAL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=sbatch_logs/viz_token_emb/%x_%j.out
#SBATCH --error=sbatch_logs/viz_token_emb/%x_%j.err

# ---------------------------------------------------------------------------
# Token-embedding PCA and clustering visualization for the converted native
# ijepa_lite original I-JEPA ImageNet-1K ViT-H/14 checkpoint at 448x448.
#
# The checkpoint was trained at 224x224, so this relies on patch-positional
# embedding interpolation from a 16x16 token grid to a 32x32 token grid.
#
# Uses:
#   - imagenet:val from the CINECA imagenet module (IMAGENET2012_HOME)
#   - chestmnist:test from FAST/datasets
#
# Submit:
#   mkdir -p sbatch_logs/viz_token_emb
#   sbatch slurm/viz_token_emb/visualize_orig_vith14_in1k_chestmnist_448_dbg.sbatch
#
# Useful overrides:
#   N_SAMPLES=16 SAMPLE_MODE=first sbatch ...
#   FEATURE_MODE=last4_concat sbatch ...
#   BATCH_SIZE=1 sbatch ...
#   OUT_DIR=/path/to/output sbatch ...
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
MED_ROOT="${MED_ROOT:-${FAST:-/scratch}/datasets}"

CKPT_LABEL="${CKPT_LABEL:-orig_ijepa_vith14_1k_ijepalite}"
CKPT_PATH="${CKPT_PATH:-${WORK}/vitturini/original_jepa_weights/IN1K-vit.h.14-300e.ijepa_lite.pt}"
OUT_DIR="${OUT_DIR:-${WORK}/vitturini/token_embedding_viz/${CKPT_LABEL}_448_in1k_chestmnist_${SLURM_JOB_ID}}"

N_SAMPLES="${N_SAMPLES:-8}"
SAMPLE_MODE="${SAMPLE_MODE:-random}"
SEED="${SEED:-0}"
FEATURE_MODE="${FEATURE_MODE:-final}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"

DATASETS=(
  imagenet:val
  chestmnist:test
)

METHODS=(
  kmeans
  agglomerative
  spectral
  gmm
  dbscan
)

CLUSTER_K=(
  2
  3
  4
  6
  8
)

DBSCAN_EPS=(
  1.5
  2.5
  4.0
)

if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "Checkpoint not found: ${CKPT_PATH}"
  exit 1
fi

echo "IN1K_ROOT=${IN1K_ROOT}"
ls -ld "${IN1K_ROOT}" "${IN1K_ROOT}/train" "${IN1K_ROOT}/val"

mkdir -p "${OUT_DIR}"

CMD=(
  python -m ijepa_lite.scripts.visualize_token_embeddings
  --repo-root "${REPO_ROOT}"
  --data-root "${MED_ROOT}"
  --dataset-root "imagenet=${IN1K_ROOT}"
  --dataset-root "chestmnist=${MED_ROOT}"
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
  --model-image-size 448
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

echo "REPO_ROOT=${REPO_ROOT}"
echo "MED_ROOT=${MED_ROOT}"
echo "CKPT_LABEL=${CKPT_LABEL}"
echo "CKPT_PATH=${CKPT_PATH}"
echo "OUT_DIR=${OUT_DIR}"
echo "N_SAMPLES=${N_SAMPLES}"
echo "SAMPLE_MODE=${SAMPLE_MODE}"
echo "FEATURE_MODE=${FEATURE_MODE}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "CMD=${CMD[*]}"

cd "${REPO_ROOT}"
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" "${CMD[@]}"
