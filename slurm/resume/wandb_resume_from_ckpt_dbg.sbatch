#!/bin/bash
#SBATCH --job-name=ijepa-wandb-resume-dbg
#SBATCH --account=IscrC_PRIMAL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --output=sbatch_logs/resume/%x_%j.out
#SBATCH --error=sbatch_logs/resume/%x_%j.err

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

nvidia-smi -L

mkdir -p sbatch_logs/resume

# ---------------------------------------------------------------------------
# Fill at least CHECKPOINT_PATH before launching.
# For old runs created before the W&B resume patch, RUN_ID must be provided.
# In offline mode, the run id is typically the suffix in:
#   .../wandb/offline-run-YYYYMMDD_HHMMSS-<run_id>
# ---------------------------------------------------------------------------
CHECKPOINT_PATH="/leonardo_work/IscrC_TBoneAI/vitturini/outputs/2026-04-29/19-13-50_mi_HmargCosBS_floor_learned_rrg50_4blk_warmup_vanilla_dbg/rank0/checkpoints/last.pt"
RUN_ID="kvci0d5f"
RUN_NAME="mi_HmargCosBS_floor_learned_rrg50_4blk_warmup_vanilla_dbg_0429_191433_r0"
RESUME_MODE="must"

if [[ -z "${CHECKPOINT_PATH}" ]]; then
  echo "CHECKPOINT_PATH is empty."
  exit 1
fi

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT_PATH}"
  exit 1
fi

EXP_NAME="$(basename "$(dirname "$(dirname "${CHECKPOINT_PATH}")")")"

EXTRA_ARGS=()
if [[ -n "${RUN_ID}" ]]; then
  EXTRA_ARGS+=(logger.run_id="$RUN_ID")
else
  echo "RUN_ID is empty; relying on wandb_run_id stored in the checkpoint state."
fi

if [[ -n "${RUN_NAME}" ]]; then
  EXTRA_ARGS+=(logger.run_name="$RUN_NAME")
fi

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python -m torch.distributed.run --standalone --nproc_per_node=4 \
    -m ijepa_lite.run \
      experiment=stl10_vits_ps8_mi \
      exp_name="$EXP_NAME" \
      resume="$CHECKPOINT_PATH" \
      data.root=$FAST/datasets/ \
      logger.mode=offline \
      logger.resume="$RESUME_MODE" \
      "${EXTRA_ARGS[@]}"
