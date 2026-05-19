#!/bin/bash
#SBATCH --job-name=ijepa-downstream-es-dbg
#SBATCH --account=IscrC_DEEPECHO
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --array=0-9
#SBATCH --output=sbatch_logs/downstream_heads/%x_%A_%a.out
#SBATCH --error=sbatch_logs/downstream_heads/%x_%A_%a.err

# ---------------------------------------------------------------------------
# Debug downstream early-stopped MLP probes for the IN1K 96x96 / ps8
# vanilla-multiblock checkpoint.
#
# Array mapping:
#   0 -> stl10
#   1 -> cifar100
#   2 -> minc2500
#   3 -> sun397
#   4 -> food101
#   5 -> fairface (race)
#   6 -> clevr_count
#   7 -> organmnist
#   8 -> tissuemnist
#   9 -> vocseg
#
# Submit:
#   sbatch slurm/downstream_heads/dbg_in1k_96_downstream_mlp_earlystop_array.sbatch
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

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
export PRETRAIN_CKPT="$REPO_ROOT/outputs/2026-05-04/19-30-19_in1k_96_vits_ps8_vanilla_multiblock_300ep/rank0/checkpoints/epoch_00099.pt"
export DATA_ROOT="${FAST:-/scratch}/datasets/"

DATASETS=(stl10 cifar100 minc2500 sun397 food101 fairface clevr_count organmnist tissuemnist vocseg)
DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"

EXTRA_ARGS=()
if [[ "$DATASET" == "fairface" ]]; then
  EXTRA_ARGS+=(data.target_attr=race)
fi

EXPERIMENT="in1k_96_vits_ps8_downstream_mlp_earlystop"
if [[ "$DATASET" == "vocseg" ]]; then
  EXPERIMENT="in1k_96_vits_ps8_vocseg_probe_earlystop"
fi

echo "dataset=${DATASET}"
echo "PRETRAIN_CKPT=${PRETRAIN_CKPT}"
echo "DATA_ROOT=${DATA_ROOT}"
nvidia-smi -L

EXP_NAME="in1k_96_vits_ps8_${DATASET}_es_probe_dbg"

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python -m ijepa_lite.run \
    experiment="${EXPERIMENT}" \
    data="${DATASET}" \
    data.root="${DATA_ROOT}" \
    exp_name="${EXP_NAME}" \
    logger.mode=offline \
    "${EXTRA_ARGS[@]}"
