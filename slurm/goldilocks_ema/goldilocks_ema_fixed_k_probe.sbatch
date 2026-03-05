#!/bin/bash
#SBATCH --job-name=ijepa-gldk-ema-fixk
#SBATCH --account=IscrC_PRIMAL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=sbatch_logs/goldilocks_ema_fixed_k/%x_%j.out
#SBATCH --error=sbatch_logs/goldilocks_ema_fixed_k/%x_%j.err

# ---------------------------------------------------------------------------
# Path B diagnostic probe — fixed k_tgt=36, k_ctx=64, 50 epochs
#
# Run this ALONGSIDE goldilocks_fixed_k_probe.sbatch (same budget, same epochs)
# to isolate the effect of the EMA signal from all other variables.
#
# Direct comparison metrics (both runs, same W&B project):
#   mask/batch_iou          — should converge toward 0.143 faster with EMA signal
#   mask/tgt_pos_std_norm   — should be lower with EMA signal, indicating more consistent masker outputs across images
#   mask/marginal_score_std — KEY: should DECREASE with EMA signal
#                             If it stays high → add diversity term
#   mask/masker_loss        — should be smoother and lower with stable EMA signal
#
# Run:
#   sbatch slurm/goldilocks_ema/goldilocks_ema_fixed_k_probe.sbatch
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

mkdir -p sbatch_logs/goldilocks_ema_fixed_k

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python -m torch.distributed.run --standalone --nproc_per_node=4 \
    -m ijepa_lite.run \
      experiment=stl10_vits_ps8_goldilocks_ema \
      masking/latent=goldilocks_ema_fixed_k \
      data.root=$FAST/datasets/ \
      train.epochs=500 \
      train.save_every_epochs=50 \
      data.batch_size=1024 \
      logger.mode=offline
