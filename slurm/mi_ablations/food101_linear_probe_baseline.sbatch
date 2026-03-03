#!/bin/bash
#SBATCH --job-name=food101-lp-baseline
#SBATCH --account=IscrC_TBoneAI
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=sbatch_logs/food101-lp-baseline/%x_%A_%a.out
#SBATCH --error=sbatch_logs/food101-lp-baseline/%x_%A_%a.err

# ---------------------------------------------------------------------------
# Food101 mean-pool-all baseline (linear_probe on all 144 tokens).
# Matches the STL-10 baseline used in the pool-bin ablation, giving a
# reference point for "ign with 1/3 tokens vs mean-pool all tokens".
#
# Four seeds matching the STL-10 multi-seed run (0, 1, 2, 42):
#   idx  seed
#    0     0
#    1     1
#    2     2
#    3    42
#
# Submit:
#   sbatch slurm/mi_ablations/food101_linear_probe_baseline.sbatch
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

echo "Array task ${SLURM_ARRAY_TASK_ID}: seed=${SEED}  pool=mean (all tokens)  dataset=food101"

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python -m ijepa_lite.run \
    experiment=food101_masker_pool_ablation \
    data.root=$FAST/datasets/ \
    seed=${SEED} \
    task.eval_modes.masker_hard_probe=false \
    task.eval_modes.linear_probe=true \
    masker_probe.epochs=20
