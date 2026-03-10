#!/bin/bash
#SBATCH --job-name=ijepa-gldk-corrloss
#SBATCH --account=IscrC_TBoneAI
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=20:00:00
#SBATCH --array=0-1
#SBATCH --output=sbatch_logs/corrloss_4n/%x_%A_%a.out
#SBATCH --error=sbatch_logs/corrloss_4n/%x_%A_%a.err

# ---------------------------------------------------------------------------
# Correlation loss probe — fixed k_tgt=36, global_z_score=true, 5000 epochs
#
# Tests whether replacing MSE with Pearson correlation loss fixes the
# score-suppression problem observed with log_transform=true + MSE.
#
# Array layout:
#   idx   log_transform   correlation_loss   notes
#    0    false            true               correlation without log (baseline comparison)
#    1    true             true               correlation + log (the full fix)
#
# Already tested (MSE loss, not repeated here):
#   k=36  logt=false  corr=false  → 60% val acc
#   k=36  logt=true   corr=false  → 38% val acc (score suppression)
#
# Run:
#   sbatch slurm/goldilocks/goldilocks_corr_loss_probe_multi_node.sbatch
# ---------------------------------------------------------------------------

set -euo pipefail

module purge
module load profile/deeplrn
module load python/3.11.6--gcc--8.5.0

source $WORK/vitturini/ijepa_venv/bin/activate

# Rendezvous
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500
export MASTER_ADDR MASTER_PORT

echo "MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT"
echo "SLURM_NNODES=$SLURM_NNODES"

export WANDB_MODE=offline

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

nvidia-smi -L

LOG_TRANSFORMS=(false true)
LOGT=${LOG_TRANSFORMS[$SLURM_ARRAY_TASK_ID]}

echo "Array task ${SLURM_ARRAY_TASK_ID}: log_transform=${LOGT}, correlation_loss=true"

mkdir -p sbatch_logs/corrloss_4n

srun bash -c '
  torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=4 \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    -m ijepa_lite.run \
      experiment=stl10_vits_ps8_goldilocks \
      masking/latent=goldilocks_fixed_k \
      masking.latent.global_z_score=true \
      masking.latent.log_transform='"$LOGT"' \
      masking.latent.correlation_loss=true \
      data.root=$FAST/datasets/ \
      train.epochs=5000 \
      train.save_every_epochs=50 \
      train.log_every=10 \
      data.batch_size=4096 \
      logger.mode=offline
'
