#!/bin/bash
#SBATCH --job-name=pool-ablation
#SBATCH --account=IscrC_TBoneAI
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --array=0-3
#SBATCH --output=sbatch_logs/pool-ablation_seed42/%x_%A_%a.out
#SBATCH --error=sbatch_logs/pool-ablation_seed42/%x_%A_%a.err

# ---------------------------------------------------------------------------
# Pool-bin ablation at the best operating point from the ctx sweep
# (λ=1.0, α=0.01 → nctx=47).
#
# idx  pool
#  0   ctx        (context only — reference, matches sweep idx 6)
#  1   tgt        (target only)
#  2   ign        (ignored only — sanity check)
#  3   ctx+tgt    (non-ignored)
#
# Baseline: mean-pool all tokens (linear_probe) = 0.6880
#
# Submit:
#   sbatch slurm/mi_ablations/masker_pool_ablation.sbatch
# ---------------------------------------------------------------------------

set -euo pipefail

module purge
module load profile/deeplrn
module load python/3.11.6--gcc--8.5.0

source $WORK/vitturini/ijepa_venv/bin/activate

nvidia-smi -L || true

export PRETRAIN_CKPT="$WORK/vitturini/outputs/2026-03-02/23-23-53_stl10_vits_ps8_mi_coupled_sr/rank0/checkpoints/last.pt"

POOLS=(ctx tgt ign ctx+tgt)
POOL=${POOLS[$SLURM_ARRAY_TASK_ID]}

echo "Array task ${SLURM_ARRAY_TASK_ID}: lam=1.0  alpha=0.01  pool=${POOL}"

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python -m ijepa_lite.run \
    experiment=stl10_masker_hard_probe_sweep \
    data.root=$FAST/datasets/ \
    task.masker_hard_probe.lam=1.0 \
    task.masker_hard_probe.alpha=0.01 \
    "task.masker_hard_probe.pool=${POOL}" \
    seed=42
