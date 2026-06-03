#!/bin/bash
#SBATCH --job-name=ijepa-downstream-es-dbg-1job
#SBATCH --account=IscrC_PRIMAL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=sbatch_logs/downstream_heads/%x_%j.out
#SBATCH --error=sbatch_logs/downstream_heads/%x_%j.err

# ---------------------------------------------------------------------------
# Single-job early-stopped MLP downstream sweep for the IN1K 96x96 / ps8
# vanilla-multiblock debug checkpoint used by the array launcher.
#
# This mirrors:
#   slurm/downstream_heads/dbg_in1k_96_downstream_mlp_earlystop_array.sbatch
#
# but loads the checkpoint once and runs all downstream datasets sequentially.
#
# Submit:
#   sbatch slurm/downstream_heads/dbg_in1k_96_downstream_mlp_earlystop_single_job.sbatch
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

mkdir -p sbatch_logs/downstream_heads

REPO_ROOT="${WORK}/vitturini/ijepa_lite"
DATA_ROOT="${FAST:-/scratch}/datasets"
SUMMARY_PATH="${REPO_ROOT}/sbatch_logs/downstream_heads/dbg_in1k_96_results_summary.txt"
CKPT_LABEL="in1k_96_vits_ps8_vanilla_multiblock_dbg"
CKPT_PATH="${REPO_ROOT}/outputs/2026-05-04/19-30-19_in1k_96_vits_ps8_vanilla_multiblock_300ep/rank0/checkpoints/epoch_00099.pt"
CLASSIFICATION_BATCH_SIZE=512
SEGMENTATION_BATCH_SIZE=192
CLASSIFICATION_NUM_WORKERS=1
SEGMENTATION_NUM_WORKERS=1

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
  python -m torch.distributed.run --standalone --nproc_per_node "${NPROC_PER_NODE}" -m ijepa_lite.scripts.downstream_sweep
  --repo-root "$REPO_ROOT"
  --data-root "$DATA_ROOT"
  --summary-path "$SUMMARY_PATH"
  --logger-mode offline
  --fairface-target-attr race
  --continue-on-error
  --classification-experiment in1k_96_vits_ps8_downstream_mlp_earlystop
  --segmentation-experiment in1k_96_vits_ps8_vocseg_probe_earlystop
  --classification-batch-size "${CLASSIFICATION_BATCH_SIZE}"
  --segmentation-batch-size "${SEGMENTATION_BATCH_SIZE}"
  --classification-num-workers "${CLASSIFICATION_NUM_WORKERS}"
  --segmentation-num-workers "${SEGMENTATION_NUM_WORKERS}"
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
echo "CLASSIFICATION_NUM_WORKERS=${CLASSIFICATION_NUM_WORKERS}"
echo "SEGMENTATION_NUM_WORKERS=${SEGMENTATION_NUM_WORKERS}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
nvidia-smi -L

"${CMD[@]}"
