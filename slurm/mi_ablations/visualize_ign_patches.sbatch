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
# Two modes:
#
#   Single checkpoint (default):
#     Both encoder and masker are loaded from PRETRAIN_CKPT.
#     Output: ign_viz/
#
#   Split checkpoints (Phase-2 diagnostic):
#     Encoder from ENCODER_CKPT (vanilla JEPA), masker from MASKER_CKPT.
#     Uncomment the split-checkpoint srun block and set ENCODER_CKPT below.
#     Output: ign_viz_phase2/
#
# Output lands relative to the repo root (Hydra chdir is NOT used).
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

# MI-coupled checkpoint (encoder + masker in same file)
export PRETRAIN_CKPT="$WORK/vitturini/outputs/2026-03-02/23-23-53_stl10_vits_ps8_mi_coupled_sr/rank0/checkpoints/last.pt"

# Vanilla JEPA checkpoint (encoder only — for split-checkpoint mode)
# export ENCODER_CKPT="$WORK/vitturini/outputs/<vanilla_run_dir>/rank0/checkpoints/last.pt"

mkdir -p sbatch_logs/viz-ign-patches

# --- Single checkpoint (default) ---
srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python ijepa_lite/hacky_visualize_ign_patches.py \
    --ckpt      "$PRETRAIN_CKPT" \
    --data-root "$FAST/datasets/" \
    --out-dir   ign_viz \
    --n         500 \
    --device    cuda

# --- Split checkpoints: vanilla encoder + MI masker (Phase-2 diagnostic) ---
# Uncomment and fill ENCODER_CKPT above before running.
#
# srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
#   python ijepa_lite/hacky_visualize_ign_patches.py \
#     --encoder-ckpt "$ENCODER_CKPT" \
#     --masker-ckpt  "$PRETRAIN_CKPT" \
#     --data-root    "$FAST/datasets/" \
#     --out-dir      ign_viz_phase2 \
#     --n            500 \
#     --device       cuda
