#!/bin/bash
#SBATCH --job-name=ijepa-lite-4nodes-env
#SBATCH --account=IscrC_TBoneAI
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --output=sbatch_logs/%x_%j.out
#SBATCH --error=sbatch_logs/%x_%j.err

set -euo pipefail

module purge
module load profile/deeplrn
module load python/3.11.6--gcc--8.5.0

source $WORK/vitturini/ijepa_venv/bin/activate

echo "python:   $(which python)"
echo "torchrun: $(which torchrun)"
echo "torch:    $(python -c 'import torch; print(torch.__version__)')"
python -c "import sys; print('sys.executable:', sys.executable)"
python -c "import ijepa_lite; print('ijepa_lite:', ijepa_lite.__file__)"

# Rendezvous
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500
export MASTER_ADDR MASTER_PORT

echo "MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT"
echo "SLURM_NNODES=$SLURM_NNODES"

# Environment
export WANDB_MODE=offline

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

nvidia-smi -L

# Launch
# Single quotes around the bash -c argument are critical: they prevent the head
# node's shell from expanding $SLURM_NODEID at submission time. Instead, each
# srun task expands it locally, where it correctly holds that node's index (0-3).
srun bash -c '
  torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=4 \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    -m ijepa_lite.run \
      experiment=stl10_vits_ps8_comp4 \
      data.root=$FAST/datasets/ \
      logger.mode=offline \
      train.epochs=4000
'