#!/bin/bash
#SBATCH --job-name=ijepa-orig-vith14-lite-heads
#SBATCH --account=IscrC_PRIMAL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=sbatch_logs/downstream_heads/%x_%j.out
#SBATCH --error=sbatch_logs/downstream_heads/%x_%j.err

# ---------------------------------------------------------------------------
# Single-job early-stopped MLP downstream sweep for the converted native
# ijepa_lite version of the original I-JEPA ImageNet-1K ViT-H/14 checkpoint.
#
# The downstream_sweep script loads the checkpoint once, then runs all
# requested datasets sequentially. Geometry overrides below must stay aligned
# with the converted checkpoint:
#   arch=vit_huge, image_size=224, patch_size=14,
#   embed_dim=1280, num_heads=16, depth=32, use_cls_token=false
#
# Edit the DATASETS array below to add or remove downstream tasks.
#
# Submit:
#   sbatch slurm/downstream_heads/orig_ijepa_vith14_ijepalite_mlp_earlystop_single_job.sbatch
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

mkdir -p sbatch_logs/downstream_heads

REPO_ROOT="${WORK}/vitturini/ijepa_lite"
DATA_ROOT="${FAST:-/scratch}/datasets"
SUMMARY_PATH="${REPO_ROOT}/sbatch_logs/downstream_heads/orig_ijepa_vith14_ijepalite_results_summary.txt"
CKPT_LABEL="orig_ijepa_vith14_1k_ijepalite"
CKPT_PATH="${WORK}/vitturini/original_jepa_weights/IN1K-vit.h.14-300e.ijepa_lite.pt"
CLASSIFICATION_BATCH_SIZE=384
SEGMENTATION_BATCH_SIZE=192

DATASETS=(
  stl10
  cifar100
  minc2500
  sun397
  food101
  fairface
  clevr_count
  organmnist
  tissuemnist
  vocseg
)

CMD=(
  python -m ijepa_lite.scripts.downstream_sweep
  --repo-root "$REPO_ROOT"
  --data-root "$DATA_ROOT"
  --summary-path "$SUMMARY_PATH"
  --logger-mode offline
  --fairface-target-attr race
  --continue-on-error
  --model-arch vit_huge
  --model-image-size 224
  --model-patch-size 14
  --model-embed-dim 1280
  --model-num-heads 16
  --model-depth 32
  --model-use-cls-token false
  --classification-batch-size "${CLASSIFICATION_BATCH_SIZE}"
  --segmentation-batch-size "${SEGMENTATION_BATCH_SIZE}"
  --checkpoint "${CKPT_LABEL}=${CKPT_PATH}"
)

for dataset in "${DATASETS[@]}"; do
  CMD+=(--dataset "$dataset")
done

echo "REPO_ROOT=${REPO_ROOT}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "SUMMARY_PATH=${SUMMARY_PATH}"
echo "CKPT_PATH=${CKPT_PATH}"
echo "CLASSIFICATION_BATCH_SIZE=${CLASSIFICATION_BATCH_SIZE}"
echo "SEGMENTATION_BATCH_SIZE=${SEGMENTATION_BATCH_SIZE}"
nvidia-smi -L

"${CMD[@]}"
