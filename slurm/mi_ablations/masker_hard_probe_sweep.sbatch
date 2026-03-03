#!/bin/bash
#SBATCH --job-name=hard-probe-sweep
#SBATCH --account=IscrC_TBoneAI
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --array=0-8
#SBATCH --output=sbatch_logs/%x_%A_%a.out
#SBATCH --error=sbatch_logs/%x_%A_%a.err

# ---------------------------------------------------------------------------
# 3×3 grid sweep over (λ, α) operating points of the frozen coupled-SR masker.
#
# Each array element trains a linear head on the hard-selected context tokens
# at one fixed (λ, α) point. All 9 jobs run in parallel, one GPU each.
#
# Baseline for comparison (from previous eval_suite run):
#   linear_probe val_acc1 = 0.6880  (mean-pool ALL tokens, SGD head)
#
# Submit from the repo root:
#   sbatch slurm/mi_ablations/masker_hard_probe_sweep.sbatch
# ---------------------------------------------------------------------------

set -euo pipefail

module purge
module load profile/deeplrn
module load python/3.11.6--gcc--8.5.0

source $WORK/vitturini/ijepa_venv/bin/activate

nvidia-smi -L || true

export PRETRAIN_CKPT="$WORK/vitturini/outputs/stl10_vits_ps8_mi_coupled_sr/rank0/checkpoints/last.pt"

# ---------------------------------------------------------------------------
# (λ, α) grid — row-major order matching --array=0-8
#   idx  lam    alpha
#    0   0.01   0.01
#    1   0.01   0.10
#    2   0.01   0.50
#    3   0.10   0.01
#    4   0.10   0.10
#    5   0.10   0.50
#    6   1.00   0.01
#    7   1.00   0.10
#    8   1.00   0.50
# ---------------------------------------------------------------------------
LAMS=(  0.01  0.01  0.01  0.1  0.1  0.1  1.0  1.0  1.0)
ALPHAS=(0.01  0.1   0.5   0.01 0.1  0.5  0.01 0.1  0.5)

LAM=${LAMS[$SLURM_ARRAY_TASK_ID]}
ALPHA=${ALPHAS[$SLURM_ARRAY_TASK_ID]}

echo "Array task ${SLURM_ARRAY_TASK_ID}: lam=${LAM}  alpha=${ALPHA}"

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python -m ijepa_lite.run \
    experiment=stl10_masker_hard_probe_sweep \
    data.root=$FAST/datasets/ \
    task.masker_hard_probe.lam="${LAM}" \
    task.masker_hard_probe.alpha="${ALPHA}"
