#!/bin/bash
#SBATCH --job-name=hard-probe-sweep
#SBATCH --account=IscrC_TBoneAI
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --array=0-8
#SBATCH --output=sbatch_logs/hard-probe-sweep/%x_%A_%a.out
#SBATCH --error=sbatch_logs/hard-probe-sweep/%x_%A_%a.err

# ---------------------------------------------------------------------------
# 3×3 grid sweep over (λ, α) at pool=ctx (context tokens only).
# Change pool= below or override on the sbatch command line for other bins:
#   pool=tgt / pool=ign / pool=ctx+tgt / pool=ctx+tgt+ign
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

export PRETRAIN_CKPT="$WORK/vitturini/outputs/2026-03-02/23-23-53_stl10_vits_ps8_mi_coupled_sr/rank0/checkpoints/last.pt"

# Pool spec — override by setting POOL env var before sbatch, e.g.:
#   POOL=tgt sbatch slurm/mi_ablations/masker_hard_probe_sweep.sbatch
POOL="${POOL:-ctx}"

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

echo "Array task ${SLURM_ARRAY_TASK_ID}: lam=${LAM}  alpha=${ALPHA}  pool=${POOL}"

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python -m ijepa_lite.run \
    experiment=stl10_masker_hard_probe_sweep \
    data.root=$FAST/datasets/ \
    task.masker_hard_probe.lam="${LAM}" \
    task.masker_hard_probe.alpha="${ALPHA}" \
    "task.masker_hard_probe.pool=${POOL}"
