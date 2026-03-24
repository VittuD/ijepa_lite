#!/bin/bash
#SBATCH --job-name=ijepa-vanilla-multiblock-stl10-ctxloss
#SBATCH --account=IscrC_PRIMAL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
#SBATCH --output=sbatch_logs/vanilla_baseline/%x_%j.out
#SBATCH --error=sbatch_logs/vanilla_baseline/%x_%j.err

# ---------------------------------------------------------------------------
# Vanilla multiblock I-JEPA + V-JEPA 2.1 context loss.
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

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python -m torch.distributed.run --standalone --nproc_per_node=4 \
    -m ijepa_lite.run \
      experiment=stl10_vits_ps8_multiblock \
      exp_name=vanilla_baseline_multiblock_stl10_ctxloss \
      +ctx_loss.weight=1.0 \
      +ctx_loss.gamma=0.7 \
      +ctx_loss.warmup_start=0 \
      +ctx_loss.warmup_end=0 \
      data.root=$FAST/datasets/ \
      data.batch_size=2048 \
      train.viz.out_dir="viz_output_vanilla_baseline_multiblock_stl10_ctxloss" \
      logger.mode=offline
