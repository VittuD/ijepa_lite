#!/bin/bash
#SBATCH --job-name=food101-pool-ablation
#SBATCH --account=IscrC_TBoneAI
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --array=0-3
#SBATCH --output=sbatch_logs/food101-pool-ablation/%x_%A_%a.out
#SBATCH --error=sbatch_logs/food101-pool-ablation/%x_%A_%a.err

# ---------------------------------------------------------------------------
# Pool-bin ablation on Food101 — transfer test for the STL-10 coupled-SR ckpt.
#
# Same (λ, α) operating point as the STL-10 ablation (λ=1.0, α=0.01 → nctx≈47).
# Food101 has 101 classes; images resized to 96×96 (pretraining resolution).
#
# idx  pool
#  0   ctx        (context only)
#  1   tgt        (target only)
#  2   ign        (ignored only)
#  3   ctx+tgt    (non-ignored)
#
# PRE-REQUISITE: Food101 must be pre-downloaded into $FAST/datasets/food101/.
#   python -c "import torchvision; torchvision.datasets.Food101('$FAST/datasets/', download=True)"
#
# Submit:
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
POOL=${POOLS[$SLURM_ARRAY_TASK_ID]}

echo "Array task ${SLURM_ARRAY_TASK_ID}: lam=1.0  alpha=0.01  pool=${POOL}  dataset=food101"

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python -m ijepa_lite.run \
    experiment=food101_masker_pool_ablation \
    data.root=$FAST/datasets/ \
    task.masker_hard_probe.lam=1.0 \
    task.masker_hard_probe.alpha=0.01 \
    "task.masker_hard_probe.pool=${POOL}"
