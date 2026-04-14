#!/bin/bash
#SBATCH --job-name=ijepa-probe-bench-dbg
#SBATCH --account=IscrC_TBoneAI
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=sbatch_logs/progressive_kl/%x_%j.out
#SBATCH --error=sbatch_logs/progressive_kl/%x_%j.err

# ---------------------------------------------------------------------------
# Probe benchmark: epoch-count sweep on a weak checkpoint to find where
# train_acc drops below 1.
#
# Run 1 — normal head (hidden_dim=384), 100 epochs
# Run 2 — normal head (hidden_dim=384),  20 epochs
# Run 3 — normal head (hidden_dim=384),  10 epochs
# Run 4 — normal head (hidden_dim=384),   5 epochs
# Run 5 — small head  (hidden_dim=128), 100 epochs
# ---------------------------------------------------------------------------

set -euo pipefail

module purge
module load profile/deeplrn
module load python/3.11.6--gcc--8.5.0

source $WORK/vitturini/ijepa_venv/bin/activate

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

nvidia-smi -L

mkdir -p sbatch_logs/progressive_kl

CKPT="outputs/2026-03-14/15-43-45_vanilla_baseline_multiblock_separate_pred_stl10/rank0/checkpoints/last.pt"

COMMON_ARGS=(
  "$CKPT"
  --data-root $FAST/datasets
  --image-size 96
  --patch-size 8
  --embed-dim 384
  --depth 12
  --num-heads 6
  --num-classes 10
  --batch-size 256
  --num-workers 4
  --max-iter 2000
  --C 1.0
  --probe-lr 0.1
  --step-size 30
  --gamma 0.1
  --head-num-layers 1
  --device cuda
)

echo "=== Run 1: normal head (hidden_dim=384), 100 epochs ==="
srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python ijepa_lite/scripts/probe_benchmark.py \
    "${COMMON_ARGS[@]}" \
    --probe-epochs 100 \
    --head-hidden-dim 384

echo "=== Run 2: normal head (hidden_dim=384), 20 epochs ==="
srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python ijepa_lite/scripts/probe_benchmark.py \
    "${COMMON_ARGS[@]}" \
    --probe-epochs 20 \
    --head-hidden-dim 384

echo "=== Run 3: normal head (hidden_dim=384), 10 epochs ==="
srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python ijepa_lite/scripts/probe_benchmark.py \
    "${COMMON_ARGS[@]}" \
    --probe-epochs 10 \
    --head-hidden-dim 384

echo "=== Run 4: normal head (hidden_dim=384), 5 epochs ==="
srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python ijepa_lite/scripts/probe_benchmark.py \
    "${COMMON_ARGS[@]}" \
    --probe-epochs 5 \
    --head-hidden-dim 384

echo "=== Run 5: small head (hidden_dim=128), 100 epochs ==="
srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python ijepa_lite/scripts/probe_benchmark.py \
    "${COMMON_ARGS[@]}" \
    --probe-epochs 100 \
    --head-hidden-dim 128
