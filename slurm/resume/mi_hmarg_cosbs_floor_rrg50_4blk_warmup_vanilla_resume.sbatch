#!/bin/bash
#SBATCH --job-name=ijepa-resume-rrg50-4blk-wup-vanilla
#SBATCH --account=IscrC_PRIMAL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00
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

CHECKPOINT_PATH="/leonardo_work/IscrC_TBoneAI/vitturini/outputs/2026-04-29/19-13-50_mi_HmargCosBS_floor_learned_rrg50_4blk_warmup_vanilla_dbg/rank0/checkpoints/last.pt"
RUN_ID="kvci0d5f"
RUN_NAME="mi_HmargCosBS_floor_learned_rrg50_4blk_warmup_vanilla_dbg_0429_191433_r0"
RESUME_MODE="must"
EXP_NAME="$(basename "$(dirname "$(dirname "${CHECKPOINT_PATH}")")")"

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python -m torch.distributed.run --standalone --nproc_per_node=4 \
    -m ijepa_lite.run \
      experiment=stl10_vits_ps8_mi \
      exp_name="$EXP_NAME" \
      resume="$CHECKPOINT_PATH" \
      data.root=$FAST/datasets/ \
      train.eval_every_epochs=5 \
      train.viz_enabled=true \
      +train.viz_every_epochs=5 \
      train.viz.out_dir="viz_output_${EXP_NAME}" \
      logger.mode=offline \
      logger.run_id="$RUN_ID" \
      logger.run_name="$RUN_NAME" \
      logger.resume="$RESUME_MODE"
