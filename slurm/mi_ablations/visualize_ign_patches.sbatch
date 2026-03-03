#!/bin/bash
#SBATCH --job-name=viz-ign-patches
#SBATCH --account=IscrC_TBoneAI
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=sbatch_logs/viz-ign-patches/%x_%j.out
#SBATCH --error=sbatch_logs/viz-ign-patches/%x_%j.err

# ---------------------------------------------------------------------------
# Visualize the masker's 3-way token partition (ctx/tgt/ign) for 500 train
# + 500 test images from STL-10 and Food101.
#
# Output: ign_viz/ directory in the job's working directory (Hydra chdir
# is NOT used here — outputs land relative to the repo root via srun).
#
# Submit:
#   sbatch slurm/mi_ablations/visualize_ign_patches.sbatch
# ---------------------------------------------------------------------------

set -euo pipefail

module purge
module load profile/deeplrn
module load python/3.11.6--gcc--8.5.0

source $WORK/vitturini/ijepa_venv/bin/activate

nvidia-smi -L || true

export PRETRAIN_CKPT="$WORK/vitturini/outputs/2026-03-02/23-23-53_stl10_vits_ps8_mi_coupled_sr/rank0/checkpoints/last.pt"

mkdir -p sbatch_logs/viz-ign-patches

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python hacky_visualize_ign_patches.py \
    --ckpt     "$PRETRAIN_CKPT" \
    --data-root "$FAST/datasets/" \
    --out-dir  ign_viz \
    --n        500 \
    --device   cuda
