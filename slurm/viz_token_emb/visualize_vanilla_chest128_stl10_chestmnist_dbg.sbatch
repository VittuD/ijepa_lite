#!/bin/bash
#SBATCH --job-name=viz-token-emb
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
# Token-embedding PCA and clustering visualization for the ChestMNIST-128
# vanilla checkpoint.
#
# Submit:
#   mkdir -p sbatch_logs/viz_token_emb
#   sbatch slurm/viz_token_emb/visualize_vanilla_chest128_stl10_chestmnist_dbg.sbatch
#
# Useful overrides:
#   N_SAMPLES=16 SAMPLE_MODE=first sbatch ...
#   FEATURE_MODE=last4_concat sbatch ...
#   CKPT_PATH=/path/to/last.pt CKPT_LABEL=my_ckpt sbatch ...
#   OUT_DIR=/path/to/output sbatch ...
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

CKPT_LABEL="${CKPT_LABEL:-array_0_vanilla_multiblock_last}"
CKPT_PATH="${CKPT_PATH:-${WORK}/vitturini/outputs/2026-06-06/20-13-24_chestmnist_128_vits_ps8_vanilla_multiblock_from_in1k96_e247_origlr_ema99925_nowarmup/rank0/checkpoints/last.pt}"
OUT_DIR="${OUT_DIR:-${WORK}/vitturini/token_embedding_viz/${CKPT_LABEL}_${SLURM_JOB_ID}}"

N_SAMPLES="${N_SAMPLES:-8}"
SAMPLE_MODE="${SAMPLE_MODE:-random}"
SEED="${SEED:-0}"
FEATURE_MODE="${FEATURE_MODE:-final}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"

DATASETS=(
  stl10:test
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

mkdir -p "${OUT_DIR}"

CMD=(
  python -m ijepa_lite.scripts.visualize_token_embeddings
  --repo-root "${REPO_ROOT}"
  --data-root "${DATA_ROOT}"
  --checkpoint "${CKPT_LABEL}=${CKPT_PATH}"
  --out-dir "${OUT_DIR}"
  --experiment downstream_mlp_earlystop
  --feature-mode "${FEATURE_MODE}"
  --n-samples "${N_SAMPLES}"
  --sample-mode "${SAMPLE_MODE}"
  --seed "${SEED}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --model-arch vit_small_16
  --model-image-size 128
  --model-patch-size 8
  --model-embed-dim 384
  --model-num-heads 6
  --model-depth 12
  --methods "${METHODS[@]}"
  --cluster-k "${CLUSTER_K[@]}"
  --dbscan-eps "${DBSCAN_EPS[@]}"
)

for dataset in "${DATASETS[@]}"; do
  CMD+=(--dataset "${dataset}")
done

echo "REPO_ROOT=${REPO_ROOT}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "CKPT_LABEL=${CKPT_LABEL}"
echo "CKPT_PATH=${CKPT_PATH}"
echo "OUT_DIR=${OUT_DIR}"
echo "N_SAMPLES=${N_SAMPLES}"
echo "SAMPLE_MODE=${SAMPLE_MODE}"
echo "FEATURE_MODE=${FEATURE_MODE}"
echo "CMD=${CMD[*]}"

cd "${REPO_ROOT}"
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" "${CMD[@]}"
