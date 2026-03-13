#!/bin/bash
#SBATCH --job-name=ijepa-unclaimed-ablation
#SBATCH --account=IscrC_PRIMAL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
#SBATCH --array=0-1
#SBATCH --output=sbatch_logs/%x_%A_%a.out
#SBATCH --error=sbatch_logs/%x_%A_%a.err

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

EXPERIMENTS=(
  stl10_vits_ps8_multiblock_unclaimed_ctx
  stl10_vits_ps8_multiblock_unclaimed_tgt
)
EXP_NAMES=(
  unclaimed_to_ctx
  unclaimed_to_tgt
)

EXP=${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}
NAME=${EXP_NAMES[$SLURM_ARRAY_TASK_ID]}

nvidia-smi -L

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python -m torch.distributed.run --standalone --nproc_per_node=4 \
    -m ijepa_lite.run \
      experiment=$EXP \
      exp_name=$NAME \
      data.root=$FAST/datasets/ \
      data.batch_size=2048 \
      train.viz.out_dir="viz_output_${NAME}" \
      logger.mode=offline
