#!/bin/bash
#SBATCH --job-name=ijepa-gldk-fixk-gz-4n
#SBATCH --account=IscrC_PRIMAL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=20:00:00
#SBATCH --output=sbatch_logs/%x_%j.out
#SBATCH --error=sbatch_logs/%x_%j.err

# ---------------------------------------------------------------------------
# Multi-node (4 nodes, 16 GPUs) version of goldilocks_fixed_k_global_z_probe
#
# global_z_score fix probe — fixed k_tgt=36, k_ctx=43, 5000 epochs
#
# Run:
#   sbatch slurm/goldilocks/goldilocks_fixed_k_global_z_probe_multi_node.sbatch
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

mkdir -p sbatch_logs

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
      data.root=$FAST/datasets/ \
      train.epochs=20 \
      train.save_every_epochs=50 \
      train.log_every=10 \
      data.batch_size=4096 \
      logger.mode=offline
'
