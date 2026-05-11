#!/bin/bash
#SBATCH --job-name=ijepa-downstream-heads-dbg
#SBATCH --account=IscrC_PRIMAL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --array=0-4
#SBATCH --output=sbatch_logs/downstream_heads/%x_%A_%a.out
#SBATCH --error=sbatch_logs/downstream_heads/%x_%A_%a.err

# ---------------------------------------------------------------------------
# Debug downstream linear probes for the IN1K 96x96 / ps8 vanilla-multiblock
# checkpoint:
#   outputs/2026-05-04/19-30-19_in1k_96_vits_ps8_vanilla_multiblock_300ep/
#     rank0/checkpoints/last.pt
#
# Array mapping:
#   0 -> cifar100
#   1 -> dtd
#   2 -> sun397
#   3 -> food101
#   4 -> fairface (race)
#
# Submit:
#   sbatch slurm/downstream_heads/dbg_in1k_96_downstream_linear_probe_array.sbatch
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
export PRETRAIN_CKPT="$REPO_ROOT/outputs/2026-05-04/19-30-19_in1k_96_vits_ps8_vanilla_multiblock_300ep/rank0/checkpoints/last.pt"
export DATA_ROOT="${FAST:-/scratch}/datasets/"

DATASETS=(cifar100 dtd sun397 food101 fairface)
DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"

EXTRA_ARGS=()
if [[ "$DATASET" == "fairface" ]]; then
  EXTRA_ARGS+=(data.target_attr=race)
fi

echo "dataset=${DATASET}"
echo "PRETRAIN_CKPT=${PRETRAIN_CKPT}"
echo "DATA_ROOT=${DATA_ROOT}"
nvidia-smi -L

EXP_NAME="in1k_96_vits_ps8_${DATASET}_linear_probe_dbg"

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python -m ijepa_lite.run \
    experiment=in1k_96_vits_ps8_downstream_linear_probe \
    data="${DATASET}" \
    data.root="${DATA_ROOT}" \
    exp_name="${EXP_NAME}" \
    logger.mode=offline \
    "${EXTRA_ARGS[@]}"
