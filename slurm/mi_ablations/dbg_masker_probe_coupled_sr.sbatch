#!/bin/bash
#SBATCH --job-name=eval-suite-coupled-sr
#SBATCH --account=IscrC_TBoneAI
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=sbatch_logs/%x_%j.out
#SBATCH --error=sbatch_logs/%x_%j.err

set -euo pipefail

module purge
module load profile/deeplrn
module load python/3.11.6--gcc--8.5.0

source $WORK/vitturini/ijepa_venv/bin/activate

nvidia-smi -L || true

# TODO: update with the actual coupled-sr run timestamp after training completes
export PRETRAIN_CKPT="$WORK/vitturini/outputs/stl10_vits_ps8_mi_coupled_sr/rank0/checkpoints/last.pt"

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python -m ijepa_lite.run \
    experiment=stl10_eval_suite_coupled_sr \
    data.root=$FAST/datasets/ \
    task.eval_modes.linear_probe=true \
    task.eval_modes.masker_probe=true \
    task.eval_modes.logreg=true
