#!/bin/bash
#SBATCH --job-name=ijepa-gldk-beta
#SBATCH --account=IscrC_TBoneAI
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --time=02:30:00
#SBATCH --array=0-2
#SBATCH --output=sbatch_logs/gldk-beta/%x_%A_%a.out
#SBATCH --error=sbatch_logs/gldk-beta/%x_%A_%a.err

# ---------------------------------------------------------------------------
# P3 beta sweep — Goldilocks loss weight sensitivity, 50 epochs
#
# Tests whether beta=1.0 (default) is well-calibrated vs {0.1, 5.0}.
# 50 epochs is enough to see if L_goldilocks is decreasing and tgt_pos_std
# is rising. Full 250-epoch training is only warranted for the winner.
#
# Array layout:
#   idx   beta   interpretation
#    0    0.1    masker loss barely influences masker ψ; mostly learns from position
#    1    1.0    default — Goldilocks term ≈ reconstruction term in magnitude
#    2    5.0    masker strongly pushed toward Goldilocks target; risk of instability
#
# Decision rule after sweep:
#   If tgt_pos_std and reconstruction_loss are similar across all three →
#     beta is not critical, keep default 1.0.
#   If one beta clearly outperforms →
#     Run full 250-epoch training with that beta.
#
# Run:
#   sbatch slurm/goldilocks/goldilocks_beta_sweep.sbatch
# ---------------------------------------------------------------------------

set -euo pipefail

module purge
module load profile/deeplrn
module load python/3.11.6--gcc--8.5.0

source $WORK/vitturini/ijepa_venv/bin/activate

export WANDB_MODE=offline

NPROC=2
THREADS_PER_PROC=$((SLURM_CPUS_PER_TASK / NPROC))   # 16/2 = 8
export OMP_NUM_THREADS=$THREADS_PER_PROC
export MKL_NUM_THREADS=$THREADS_PER_PROC
export OPENBLAS_NUM_THREADS=$THREADS_PER_PROC
export NUMEXPR_NUM_THREADS=$THREADS_PER_PROC

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

nvidia-smi -L || true

BETAS=(0.1 1.0 5.0)
BETA=${BETAS[$SLURM_ARRAY_TASK_ID]}

echo "Array task ${SLURM_ARRAY_TASK_ID}: beta=${BETA}"

mkdir -p sbatch_logs/gldk-beta

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python -m torch.distributed.run --standalone --nproc_per_node=2 \
    -m ijepa_lite.run \
      experiment=stl10_vits_ps8_goldilocks \
      data.root=$FAST/datasets/ \
      masking.latent.beta=${BETA} \
      train.epochs=50 \
      train.save_every_epochs=50 \
      data.batch_size=1024 \
      logger.mode=offline
