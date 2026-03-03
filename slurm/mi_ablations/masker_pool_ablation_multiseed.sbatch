#!/bin/bash
#SBATCH --job-name=pool-ablation-seed
#SBATCH --account=IscrC_TBoneAI
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --array=0-7
#SBATCH --output=sbatch_logs/pool-ablation-seed/%x_%A_%a.out
#SBATCH --error=sbatch_logs/pool-ablation-seed/%x_%A_%a.err

# ---------------------------------------------------------------------------
# Multi-seed pool-bin ablation at the best operating point (λ=1.0, α=0.01).
# Seeds 0 and 42 are already available; this script runs seeds 1 and 2,
# giving a 4-seed table for reporting.
#
# Existing results (for reference):
#   seed  pool     val_acc1
#    0    ctx      0.6696
#    0    tgt      0.6610
#    0    ign      0.6891
#    0    ctx+tgt  0.6821
#   42    ctx      0.6701
#   42    tgt      0.6631
#   42    ign      0.6969
#   42    ctx+tgt  0.6839
#
# Layout for this array (row=seed, col=pool):
#   idx  seed  pool
#    0     1   ctx
#    1     1   tgt
#    2     1   ign
#    3     1   ctx+tgt
#    4     2   ctx
#    5     2   tgt
#    6     2   ign
#    7     2   ctx+tgt
#
# Submit:
#   sbatch slurm/mi_ablations/masker_pool_ablation_multiseed.sbatch
# ---------------------------------------------------------------------------

set -euo pipefail

module purge
module load profile/deeplrn
module load python/3.11.6--gcc--8.5.0

source $WORK/vitturini/ijepa_venv/bin/activate

nvidia-smi -L || true

export PRETRAIN_CKPT="$WORK/vitturini/outputs/2026-03-02/23-23-53_stl10_vits_ps8_mi_coupled_sr/rank0/checkpoints/last.pt"

POOLS=(ctx tgt ign ctx+tgt)
SEEDS=(1 2)

POOL_IDX=$((SLURM_ARRAY_TASK_ID % 4))
SEED_IDX=$((SLURM_ARRAY_TASK_ID / 4))

POOL=${POOLS[$POOL_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "Array task ${SLURM_ARRAY_TASK_ID}: seed=${SEED}  lam=1.0  alpha=0.01  pool=${POOL}"

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python -m ijepa_lite.run \
    experiment=stl10_masker_hard_probe_sweep \
    data.root=$FAST/datasets/ \
    seed=${SEED} \
    task.masker_hard_probe.lam=1.0 \
    task.masker_hard_probe.alpha=0.01 \
    "task.masker_hard_probe.pool=${POOL}"
