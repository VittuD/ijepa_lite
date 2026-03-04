#!/bin/bash
#SBATCH --job-name=ijepa-gldk-dbg
#SBATCH --account=IscrC_TBoneAI
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=sbatch_logs/%x_%j.out
#SBATCH --error=sbatch_logs/%x_%j.err

# ---------------------------------------------------------------------------
# P1 smoke test — GoldilocksTeacherMasker integration check
#
# Runs 1 epoch of the full goldilocks config (single GPU, small batch).
# Pass criteria:
#   - No Python / shape / NaN errors
#   - L_goldilocks is logged and is a finite scalar
#   - tgt_pos_std is logged and > 0 (masker not identically zeroed)
#   - tgt_assignment_entropy is logged
#
# Run:
#   sbatch slurm/goldilocks/goldilocks_debug.sbatch
# ---------------------------------------------------------------------------

set -euo pipefail

module purge
module load profile/deeplrn
module load python/3.11.6--gcc--8.5.0

source $WORK/vitturini/ijepa_venv/bin/activate

nvidia-smi -L || true

mkdir -p sbatch_logs

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python -m ijepa_lite.run \
    experiment=stl10_vits_ps8_goldilocks \
    data.root=$FAST/datasets/ \
    data.batch_size=64 \
    data.num_workers=4 \
    train.epochs=1 \
    train.log_every=5 \
    train.save_every_epochs=1 \
    logger.mode=offline
