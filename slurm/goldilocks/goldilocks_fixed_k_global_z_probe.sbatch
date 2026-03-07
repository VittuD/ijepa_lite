#!/bin/bash
#SBATCH --job-name=ijepa-gldk-fixk-gz
#SBATCH --account=IscrC_PRIMAL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=20:00:00
#SBATCH --output=sbatch_logs/%x_%j.out
#SBATCH --error=sbatch_logs/%x_%j.err

# ---------------------------------------------------------------------------
# global_z_score fix probe — fixed k_tgt=94, k_ctx=[22,43], 500 epochs
#
# Run ALONGSIDE goldilocks_fixed_k_probe.sbatch to isolate the effect of
# global_z_score=true from all other variables (same budget, same epochs).
#
# Expected behaviour vs degenerate run (global_z_score=false):
#   mask/marginal_score_std   — should stay LOW (no positional concentration)
#   mask/batch_iou            — should stay near random baseline (~0.485)
#   mask/tgt_pos_std_norm     — should trend above 1.0 (more diverse than random)
#   No two-phase collapse around epoch 245/500
#
# Run:
#   sbatch slurm/goldilocks/goldilocks_fixed_k_global_z_probe.sbatch
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
      masking.latent.global_z_score=true \
      data.root=$FAST/datasets/ \
      train.epochs=400 \
      train.save_every_epochs=50 \
      data.batch_size=1024 \
      logger.mode=offline
