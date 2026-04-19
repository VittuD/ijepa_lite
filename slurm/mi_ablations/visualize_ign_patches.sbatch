#!/bin/bash
#SBATCH --job-name=viz-ign-patches
#SBATCH --account=IscrC_TBoneAI
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=sbatch_logs/viz-ign-patches/%x_%j.out
#SBATCH --error=sbatch_logs/viz-ign-patches/%x_%j.err

# ---------------------------------------------------------------------------
# Visualize the MI masker's 3-way token partition (ctx/tgt/ign).
#
# Uses the unified viz script which auto-detects 3-way maskers and shows
# soft 3-way overlays (ctx=blue, tgt=red, ign=grey) in the middle panel.
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

export PRETRAIN_CKPT="$WORK/vitturini/masker_only_joint_enc_pretrain_class/epoch_0299.pt"

mkdir -p sbatch_logs/viz-ign-patches

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python ijepa_lite/scripts/hacky/hacky_visualize_goldilocks.py \
    --ckpt       "$PRETRAIN_CKPT" \
    --experiment stl10_vits_ps8_mi \
    --data-root  "$FAST/datasets/" \
    --out-dir    mi_viz \
    --n          500 \
    --device     cuda
