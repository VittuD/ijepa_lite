#!/bin/bash
#SBATCH --job-name=ijepa-refactor-tests
#SBATCH --account=IscrC_TBoneAI
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=sbatch_logs/refactor/%x_%j.out
#SBATCH --error=sbatch_logs/refactor/%x_%j.err

set -euo pipefail

module purge
module load profile/deeplrn
module load python/3.11.6--gcc--8.5.0

source $WORK/vitturini/ijepa_venv/bin/activate

mkdir -p sbatch_logs/refactor

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python -m pytest -q \
    tests/test_masker_registry.py \
    tests/test_mask_semantics.py \
    tests/test_pretrain_runtime.py
