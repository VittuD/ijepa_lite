#!/bin/bash
#SBATCH --job-name=ijepa-gldk-probe
#SBATCH --account=IscrC_TBoneAI
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=sbatch_logs/%x_%j.out
#SBATCH --error=sbatch_logs/%x_%j.err

# ---------------------------------------------------------------------------
# Linear probe on features from goldilocks_stl10 pretraining.
#
# Set PRETRAIN_CKPT to the last.pt from the pretraining run before submitting.
# Results are directly comparable to the MI-rate and vanilla baselines.
#
# Run:
#   export PRETRAIN_CKPT=/path/to/goldilocks_run/rank0/checkpoints/last.pt
#   sbatch slurm/goldilocks/goldilocks_linear_probe.sbatch
# ---------------------------------------------------------------------------

set -euo pipefail

module purge
module load profile/deeplrn
module load python/3.11.6--gcc--8.5.0

source $WORK/vitturini/ijepa_venv/bin/activate

nvidia-smi -L || true

: "${PRETRAIN_CKPT:?ERROR: set PRETRAIN_CKPT to the pretraining checkpoint path}"

mkdir -p sbatch_logs

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python -m ijepa_lite.run \
    experiment=stl10_linear_probe \
    data.root=$FAST/datasets/ \
    task.pretrained_ckpt="${PRETRAIN_CKPT}" \
    logger.mode=offline
