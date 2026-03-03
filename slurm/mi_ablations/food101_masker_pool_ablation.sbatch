#!/bin/bash
#SBATCH --job-name=food101-pool-ablation
#SBATCH --account=IscrC_TBoneAI
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-7
#SBATCH --output=sbatch_logs/food101-pool-ablation/%x_%A_%a.out
#SBATCH --error=sbatch_logs/food101-pool-ablation/%x_%A_%a.err

# ---------------------------------------------------------------------------
# Pool-bin ablation on Food101 — transfer test for the STL-10 coupled-SR ckpt.
#
# Same (λ, α) operating point as the STL-10 ablation (λ=1.0, α=0.01 → nctx≈47).
# Food101 has 101 classes; images resized to 96×96 (pretraining resolution).
# Two seeds (0, 42) matching the first STL-10 pair, for comparable error bars.
#
# Layout (row=seed, col=pool):
#   idx  seed  pool
#    0     0   ctx
#    1     0   tgt
#    2     0   ign
#    3     0   ctx+tgt
#    4    42   ctx
#    5    42   tgt
#    6    42   ign
#    7    42   ctx+tgt
#
# NOTE: if seed-0 jobs (0-3) are already running, submit only the new ones:
#   sbatch --array=4-7 slurm/mi_ablations/food101_masker_pool_ablation.sbatch
#
# Full submit:
#   sbatch slurm/mi_ablations/food101_masker_pool_ablation.sbatch
# ---------------------------------------------------------------------------

set -euo pipefail

module purge
module load profile/deeplrn
module load python/3.11.6--gcc--8.5.0

source $WORK/vitturini/ijepa_venv/bin/activate

nvidia-smi -L || true

export PRETRAIN_CKPT="$WORK/vitturini/outputs/2026-03-02/23-23-53_stl10_vits_ps8_mi_coupled_sr/rank0/checkpoints/last.pt"

POOLS=(ctx tgt ign ctx+tgt)
SEEDS=(0 42)

POOL_IDX=$((SLURM_ARRAY_TASK_ID % 4))
SEED_IDX=$((SLURM_ARRAY_TASK_ID / 4))

POOL=${POOLS[$POOL_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "Array task ${SLURM_ARRAY_TASK_ID}: seed=${SEED}  lam=1.0  alpha=0.01  pool=${POOL}  dataset=food101"

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python -m ijepa_lite.run \
    experiment=food101_masker_pool_ablation \
    data.root=$FAST/datasets/ \
    seed=${SEED} \
    task.masker_hard_probe.lam=1.0 \
    task.masker_hard_probe.alpha=0.01 \
    "task.masker_hard_probe.pool=${POOL}" \
    masker_probe.epochs=20
