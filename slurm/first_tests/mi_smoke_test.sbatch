#!/bin/bash
#SBATCH --job-name=mi-smoke-test-colors
#SBATCH --account=IscrC_TBoneAI
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=sbatch_logs/mi-smoke-test/%x_%j.out
#SBATCH --error=sbatch_logs/mi-smoke-test/%x_%j.err

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

nvidia-smi -L

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python -m torch.distributed.run --standalone --nproc_per_node=1 \
    -m ijepa_lite.run experiment=stl10_vits_ps8_mi \
    data.root=$FAST/datasets/ \
    train.epochs=5 \
    train.eval_every_epochs=2 \
    train.save_every_epochs=2 \
    train.log_every=10 \
    train.viz_enabled=true \
    data.batch_size=2048 \
    logger=wandb \
    logger.mode=offline
