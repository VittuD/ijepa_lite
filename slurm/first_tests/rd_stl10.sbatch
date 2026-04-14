#!/bin/bash
#SBATCH --job-name=ijepa-rd-stl10
#SBATCH --account=IscrC_TBoneAI
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --output=sbatch_logs/%x_%j.out
#SBATCH --error=sbatch_logs/%x_%j.err

# Time budget vs baseline
# -----------------------
# stl10_vits_ps8_comp4 (deterministic masking): ~01:30:00 for 400 epochs
# stl10_vits_ps8_rd    (RD masker):             ~02:30:00 for 400 epochs
#
# Extra cost vs the deterministic baseline:
#   +1 masker transformer forward (depth=2, sequence M+N=288 with compressor=full)
#   +1 RateDist3WayLoss backward (cheap — flows through soft scores only)
# The EMA encoder runs once and its output is reused for both masking and targets.
# To reduce the masker sequence to 180 tokens:
#   python -m torch.distributed.run ... masking/compressor=spatial_stride

set -euo pipefail

module purge
module load profile/deeplrn
module load python/3.11.6--gcc--8.5.0

source $WORK/vitturini/ijepa_venv/bin/activate

# echo "python:   $(which python)"
# echo "torchrun: $(which torchrun)"
# echo "torch:    $(python -c 'import torch; print(torch.__version__)')"
# python -c "import sys; print('sys.executable:', sys.executable)"
# python -c "import ijepa_lite; print('ijepa_lite:', ijepa_lite.__file__)"

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
    -m ijepa_lite.run experiment=stl10_vits_ps8_rd data.root=$FAST/datasets/ logger.mode=offline