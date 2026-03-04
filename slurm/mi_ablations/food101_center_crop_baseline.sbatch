#!/bin/bash
#SBATCH --job-name=food101-center-crop
#SBATCH --account=IscrC_TBoneAI
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=sbatch_logs/food101-center-crop/%x_%A_%a.out
#SBATCH --error=sbatch_logs/food101-center-crop/%x_%A_%a.err

# ---------------------------------------------------------------------------
# Fixed spatial center-crop baseline on Food101 — Occam's razor check.
#
# Same falsification test as the STL-10 version, on the out-of-domain dataset.
# Two seeds matching the Food101 pool ablation run (0 and 42).
#
#   idx  seed
#    0     0
#    1    42
#
# Reference results (masker_hard_probe, same ckpt, Food101):
#   pool=ign      : 0.341 ± 0.000  (seeds 0, 42)
#   pool=baseline : 0.339 ± 0.003  (linear_probe, all tokens)
#
# Submit:
#   sbatch slurm/mi_ablations/food101_center_crop_baseline.sbatch
# ---------------------------------------------------------------------------

set -euo pipefail

module purge
module load profile/deeplrn
module load python/3.11.6--gcc--8.5.0

source $WORK/vitturini/ijepa_venv/bin/activate

nvidia-smi -L || true

export PRETRAIN_CKPT="$WORK/vitturini/outputs/2026-03-02/23-23-53_stl10_vits_ps8_mi_coupled_sr/rank0/checkpoints/last.pt"

SEEDS=(0 42)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "Array task ${SLURM_ARRAY_TASK_ID}: seed=${SEED}  pool=center  n_patches=48  dataset=food101"

mkdir -p sbatch_logs/food101-center-crop

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python -m ijepa_lite.run \
    experiment=food101_masker_pool_ablation \
    seed=${SEED} \
    task.eval_modes.masker_hard_probe=false \
    task.eval_modes.spatial_probe=true \
    task.spatial_probe.pool=center \
    data.root=$FAST/datasets/
