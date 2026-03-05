#!/bin/bash
#SBATCH --job-name=ijepa-gldk-fixk
#SBATCH --account=IscrC_TBoneAI
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=sbatch_logs/%x_%j.out
#SBATCH --error=sbatch_logs/%x_%j.err

# ---------------------------------------------------------------------------
# Diagnostic run — fixed k_tgt=36, k_ctx=64, 50 epochs
#
# Purpose: isolate masker learning signal from per-step budget noise.
# With constant k, tgt_pos_std_norm, batch_iou, and L_goldilocks reflect
# only masker behaviour — no k_tgt-induced variance.
#
# Includes the ctx_embed trunc_normal init fix.
#
# Go/no-go after this run:
#   tgt_pos_std_norm trending toward >1.0  → ctx_embed fix worked, run full training
#   batch_iou trending toward batch_iou_random (≈0.20 at k=36, N=144) → same
#   both still flat/wrong direction → patch_loss signal too weak, deeper fix needed
#
# Run:
#   sbatch slurm/goldilocks/goldilocks_fixed_k_probe.sbatch
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

mkdir -p sbatch_logs

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python -m torch.distributed.run --standalone --nproc_per_node=4 \
    -m ijepa_lite.run \
      experiment=stl10_vits_ps8_goldilocks \
      masking/latent=goldilocks_fixed_k \
      data.root=$FAST/datasets/ \
      train.epochs=500 \
      train.save_every_epochs=50 \
      data.batch_size=1024 \
      logger.mode=offline
