#!/bin/bash
#SBATCH --job-name=ijepa-probe-bench-dbg
#SBATCH --account=IscrC_PRIMAL
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
# Probe benchmark: runs all three probing strategies (logreg_mean, logreg_cls,
# mlp_sgd) on the same checkpoint and reports accuracy, speed, and peak GPU
# memory in a comparison table.
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

CKPT="outputs/2026-04-02/20-33-35_prog_kl_det_5blk_klw1.0_fwd/rank0/checkpoints/last.pt"

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python scripts/probe_benchmark.py \
    "$CKPT" \
    --data-root $FAST/datasets \
    --image-size 96 \
    --patch-size 8 \
    --embed-dim 384 \
    --depth 12 \
    --num-heads 6 \
    --num-classes 10 \
    --batch-size 256 \
    --num-workers 4 \
    --max-iter 2000 \
    --C 1.0 \
    --probe-epochs 100 \
    --probe-lr 0.1 \
    --step-size 30 \
    --gamma 0.1 \
    --head-hidden-dim 384 \
    --head-num-layers 1 \
    --device cuda
