#!/bin/bash
#SBATCH --job-name=stl10-center-crop
#SBATCH --account=IscrC_TBoneAI
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=sbatch_logs/stl10-center-crop/%x_%A_%a.out
#SBATCH --error=sbatch_logs/stl10-center-crop/%x_%A_%a.err

# ---------------------------------------------------------------------------
# Fixed spatial center-crop baseline on STL-10 — Occam's razor check.
#
# Selects the middle vertical third of the 12×12 patch grid (rows 4–7,
# 48 patches) with NO masker, and trains an SGD linear head on top.
#
# If this matches masker_hard_probe(pool=ign, λ=1.0, α=0.01 → nign≈48),
# the masker has learned nothing beyond a center-crop positional prior.
#
# Four seeds matching the pool ablation multi-seed run:
#   idx  seed
#    0     0
#    1     1
#    2     2
#    3    42
#
# Reference results (masker_hard_probe, same ckpt):
#   pool=ign      : 0.689 ± 0.005  (4 seeds: 0,1,2,42)
#   pool=baseline : 0.688 ± 0.003  (linear_probe, all tokens)
#
# Submit:
#   sbatch slurm/mi_ablations/stl10_center_crop_baseline.sbatch
# ---------------------------------------------------------------------------

set -euo pipefail

module purge
module load profile/deeplrn
module load python/3.11.6--gcc--8.5.0

source $WORK/vitturini/ijepa_venv/bin/activate

nvidia-smi -L || true

export PRETRAIN_CKPT="$WORK/vitturini/outputs/2026-03-02/23-23-53_stl10_vits_ps8_mi_coupled_sr/rank0/checkpoints/last.pt"

SEEDS=(0 1 2 42)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "Array task ${SLURM_ARRAY_TASK_ID}: seed=${SEED}  pool=center  n_patches=48  dataset=stl10"

mkdir -p sbatch_logs/stl10-center-crop

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python -m ijepa_lite.run \
    experiment=stl10_masker_hard_probe_sweep \
    seed=${SEED} \
    task.eval_modes.masker_hard_probe=false \
    task.eval_modes.spatial_probe=true \
    task.spatial_probe.pool=center
