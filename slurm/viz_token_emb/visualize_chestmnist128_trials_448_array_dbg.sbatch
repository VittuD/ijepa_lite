#!/bin/bash
#SBATCH --job-name=viz-chest128-trials-448
#SBATCH --account=IscrC_TBoneAI
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --array=0-5
#SBATCH --output=sbatch_logs/viz_token_emb/chestmnist128_trials_448/%x_%A_%a.out
#SBATCH --error=sbatch_logs/viz_token_emb/chestmnist128_trials_448/%x_%A_%a.err

# ---------------------------------------------------------------------------
# Token-embedding PCA and clustering visualization for the ChestMNIST-128
# continued-pretraining trial checkpoints plus their starting IN1K-96
# checkpoint, using last.pt at 448x448.
#
# These checkpoints were trained at 128x128 / patch_size=8, so this relies on
# learned positional embedding interpolation from a 16x16 token grid to a
# 56x56 token grid.
#
# Each array task runs one checkpoint:
#   0  array_0_vanilla_multiblock
#   1  array_1_mi_affinity_novelty
#   2  array_2_mi_hmargbs_floor
#   3  array_3_mi_hmargcosbs_floor
#   4  array_4_mi_hmarg_sketchorthcosbs_floor
#   5  start_in1k96_vanilla_multiblock
#
# Uses:
#   - imagenet:val from the CINECA imagenet module (IMAGENET2012_HOME)
#   - chestmnist:test from FAST/datasets
#
# Submit:
#   mkdir -p sbatch_logs/viz_token_emb/chestmnist128_trials_448
#   sbatch slurm/viz_token_emb/visualize_chestmnist128_trials_448_array_dbg.sbatch
#
# Useful overrides:
#   N_SAMPLES=16 SAMPLE_MODE=first sbatch ...
#   FEATURE_MODE=last4_concat sbatch ...
#   BATCH_SIZE=2 sbatch ...
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
OUT_ROOT="${OUT_ROOT:-${WORK}/vitturini/token_embedding_viz/chestmnist128_trials_448}"

N_SAMPLES="${N_SAMPLES:-8}"
SAMPLE_MODE="${SAMPLE_MODE:-random}"
SEED="${SEED:-0}"
FEATURE_MODE="${FEATURE_MODE:-final}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"

METHODS=(
  array_0_vanilla_multiblock
  array_1_mi_affinity_novelty
  array_2_mi_hmargbs_floor
  array_3_mi_hmargcosbs_floor
  array_4_mi_hmarg_sketchorthcosbs_floor
  start_in1k96_vanilla_multiblock
)

RUN_ROOTS=(
  outputs/2026-06-06/20-13-24_chestmnist_128_vits_ps8_vanilla_multiblock_from_in1k96_e247_origlr_ema99925_nowarmup
  outputs/2026-06-06/20-13-24_chestmnist_128_vits_ps8_mi_affinity_novelty_from_in1k96_e247_origlr_ema99925_nowarmup
  outputs/2026-06-06/20-13-24_chestmnist_128_vits_ps8_mi_hmargbs_floor_from_in1k96_e247_origlr_ema99925_nowarmup
  outputs/2026-06-06/20-13-24_chestmnist_128_vits_ps8_mi_hmargcosbs_floor_from_in1k96_e247_origlr_ema99925_nowarmup
  outputs/2026-06-06/20-13-24_chestmnist_128_vits_ps8_mi_hmarg_sketchorthcosbs_floor_from_in1k96_e247_origlr_ema99925_nowarmup
  outputs/2026-05-04/19-30-19_in1k_96_vits_ps8_vanilla_multiblock_300ep
)

DATASETS=(
  imagenet:val
  chestmnist:test
)

CLUSTER_METHODS=(
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

TASK_ID="${SLURM_ARRAY_TASK_ID}"
METHOD="${METHODS[$TASK_ID]}"
RUN_ROOT="${RUN_ROOTS[$TASK_ID]}"

CKPT_LABEL="${METHOD}_last"
CKPT_PATH="${WORK}/vitturini/${RUN_ROOT}/rank0/checkpoints/last.pt"
OUT_DIR="${OUT_ROOT}/${CKPT_LABEL}_${SLURM_JOB_ID}_${TASK_ID}"

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
  --model-arch vit_small_16
  --model-image-size 448
  --model-patch-size 8
  --model-embed-dim 384
  --model-num-heads 6
  --model-depth 12
  --methods "${CLUSTER_METHODS[@]}"
  --cluster-k "${CLUSTER_K[@]}"
  --dbscan-eps "${DBSCAN_EPS[@]}"
)

for dataset in "${DATASETS[@]}"; do
  CMD+=(--dataset "${dataset}")
done

echo "TASK_ID=${TASK_ID}"
echo "METHOD=${METHOD}"
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
