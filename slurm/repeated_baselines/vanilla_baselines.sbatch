#!/bin/bash
#SBATCH --job-name=ijepa-vanilla-baseline
#SBATCH --account=IscrC_PRIMAL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=06:30:00
#SBATCH --array=0-2
#SBATCH --output=sbatch_logs/repeated_baselines/%x_%A_%a.out
#SBATCH --error=sbatch_logs/repeated_baselines/%x_%A_%a.err

# ---------------------------------------------------------------------------
# Vanilla baselines (evalv2 re-run)
#   0 — vanilla_baseline_multiblock_stl10        (stl10_vits_ps8_multiblock)
#   1 — vanilla_baseline_multiblock_separate_pred (stl10_vits_ps8_multiblock_separate_pred)
#   2 — vanilla_single_blk_stl10                 (stl10_vits_ps8_single_blk)
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

nvidia-smi -L

mkdir -p sbatch_logs/repeated_baselines

EXPERIMENTS=(
  stl10_vits_ps8_multiblock
  stl10_vits_ps8_multiblock_separate_pred
  stl10_vits_ps8_single_blk
)
EXP_NAMES=(
  vanilla_baseline_multiblock_stl10
  vanilla_baseline_multiblock_separate_pred_stl10
  vanilla_single_blk_stl10
)

EXP=${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}
EXP_NAME=${EXP_NAMES[$SLURM_ARRAY_TASK_ID]}

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python -m torch.distributed.run --standalone --nproc_per_node=4 \
    -m ijepa_lite.run \
      experiment="$EXP" \
      exp_name="$EXP_NAME" \
      data.root=$FAST/datasets/ \
      data.batch_size=2048 \
      train.viz.out_dir="viz_output_${EXP_NAME}" \
      logger.mode=offline
