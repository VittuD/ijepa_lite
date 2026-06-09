#!/bin/bash
#SBATCH --job-name=ijepa-rsna-det-dbg
#SBATCH --account=IscrC_TBoneAI
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --output=sbatch_logs/downstream/rsna_detection_orig_vith14_dbg/%x_%j.out
#SBATCH --error=sbatch_logs/downstream/rsna_detection_orig_vith14_dbg/%x_%j.err

# ---------------------------------------------------------------------------
# Debug RSNA pneumonia detection probe for original I-JEPA ImageNet-1K ViT-H/14.
#
# Expected data layout:
#   ${DATA_ROOT}/rsna-pneumonia-detection/stage_2_train_labels.csv
#   ${DATA_ROOT}/rsna-pneumonia-detection/png_images/<patientId>.png
#
# Submit:
#   mkdir -p sbatch_logs/downstream/rsna_detection_orig_vith14_dbg
#   sbatch slurm/downstream/orig_vith14_rsna_detection_dbg.sbatch
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
DATA_ROOT="${DATA_ROOT:-${FAST:-/scratch}/datasets}"
LOG_DIR="${WORK}/vitturini/sbatch_logs/downstream/rsna_detection_orig_vith14_dbg"
SUMMARY_PATH="${LOG_DIR}/orig_vith14_rsna_detection_dbg_summary.txt"
CKPT_LABEL="orig_ijepa_vith14_1k_ijepalite"
CKPT_PATH="${WORK}/vitturini/original_jepa_weights/IN1K-vit.h.14-300e.ijepa_lite.pt"

DETECTION_BATCH_SIZE=32
DETECTION_NUM_WORKERS=1

mkdir -p "${LOG_DIR}"

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
  --no-save-probe-checkpoints
  --train-epochs 2
  --train-early-stop-patience 0
  --train-early-stop-min-epochs 0
  --detection-experiment downstream_rsna_detection_mlp_earlystop
  --run-suffix rsna_detection_dbg
  --task-pool last4_mean
  --model-arch vit_huge
  --model-image-size 224
  --model-patch-size 14
  --model-embed-dim 1280
  --model-num-heads 16
  --model-depth 32
  --model-use-cls-token false
  --detection-batch-size "${DETECTION_BATCH_SIZE}"
  --detection-num-workers "${DETECTION_NUM_WORKERS}"
  --checkpoint "${CKPT_LABEL}=${CKPT_PATH}"
  --dataset rsna_pneumonia_detection
)

echo "REPO_ROOT=${REPO_ROOT}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "LOG_DIR=${LOG_DIR}"
echo "SUMMARY_PATH=${SUMMARY_PATH}"
echo "CKPT_PATH=${CKPT_PATH}"
echo "DETECTION_BATCH_SIZE=${DETECTION_BATCH_SIZE}"
echo "DETECTION_NUM_WORKERS=${DETECTION_NUM_WORKERS}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
nvidia-smi -L

"${CMD[@]}"
