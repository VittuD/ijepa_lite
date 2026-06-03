#!/bin/bash
#SBATCH --job-name=ijepa-ctp-heads-es-1job
#SBATCH --account=IscrC_PRIMAL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=sbatch_logs/downstream_heads/%x_%j.out
#SBATCH --error=sbatch_logs/downstream_heads/%x_%j.err

# ---------------------------------------------------------------------------
# Single-job early-stopped MLP downstream sweep over selected continued-
# pretraining checkpoints and downstream datasets.
#
# Edit the CKPT_LABELS / CKPT_REL_PATHS / DATASETS arrays below to add or
# remove sweep items. The script loads each checkpoint once, then runs all
# requested datasets sequentially before moving to the next checkpoint.
#
# Submit:
#   sbatch slurm/downstream_heads/continued_pretrain_mlp_earlystop_single_job.sbatch
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
SUMMARY_PATH="${REPO_ROOT}/sbatch_logs/downstream_heads/results_summary.txt"

CKPT_LABELS=(
  mi_affinity_novelty
  mi_hmarg_sketchorthcosbs_floor
  mi_hmargbs_floor
  mi_hmargcosbs_floor
  vanilla_multiblock
)

CKPT_REL_PATHS=(
  outputs/2026-05-08/03-48-11_stl10_mi_affinity_novelty_cont_from_e999_200ep/rank0/checkpoints/epoch_00099.pt
  outputs/2026-05-08/03-48-27_stl10_mi_hmarg_sketchorthcosbs_floor_cont_from_e999_200ep/rank0/checkpoints/epoch_00099.pt
  outputs/2026-05-08/03-48-27_stl10_mi_hmargbs_floor_cont_from_e999_200ep/rank0/checkpoints/epoch_00099.pt
  outputs/2026-05-08/03-48-27_stl10_mi_hmargcosbs_floor_cont_from_e999_200ep/rank0/checkpoints/epoch_00099.pt
  outputs/2026-05-08/03-48-27_stl10_vanilla_multiblock_cont_from_e999_200ep/rank0/checkpoints/epoch_00099.pt
)

DATASETS=(
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

if [[ ${#CKPT_LABELS[@]} -ne ${#CKPT_REL_PATHS[@]} ]]; then
  echo "CKPT_LABELS and CKPT_REL_PATHS must have the same length"
  exit 1
fi

CMD=(
  python -m ijepa_lite.scripts.downstream_sweep
  --repo-root "$REPO_ROOT"
  --data-root "$DATA_ROOT"
  --summary-path "$SUMMARY_PATH"
  --logger-mode offline
  --fairface-target-attr race
  --continue-on-error
)

for idx in "${!CKPT_LABELS[@]}"; do
  CMD+=(--checkpoint "${CKPT_LABELS[$idx]}=${REPO_ROOT}/${CKPT_REL_PATHS[$idx]}")
done

for dataset in "${DATASETS[@]}"; do
  CMD+=(--dataset "$dataset")
done

echo "REPO_ROOT=${REPO_ROOT}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "SUMMARY_PATH=${SUMMARY_PATH}"
nvidia-smi -L

"${CMD[@]}"
