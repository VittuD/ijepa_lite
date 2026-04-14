#!/bin/bash
#SBATCH --job-name=mi-stl10-logreg
#SBATCH --account=IscrC_TBoneAI
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=sbatch_logs/%x_%j.out
#SBATCH --error=sbatch_logs/%x_%j.err

set -euo pipefail

module purge
module load profile/deeplrn
module load python/3.11.6--gcc--8.5.0

source $WORK/vitturini/ijepa_venv/bin/activate

# echo "HOST:   $(hostname)"
# echo "python: $(which python)"
# python -c "import sys; print('sys.executable:', sys.executable)"

nvidia-smi -L || true

CKPT="$WORK/vitturini/outputs/2026-02-27/01-16-13_stl10_vits_ps8_mi/rank0/checkpoints/last.pt"
DATA_ROOT="$FAST/datasets"

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python $WORK/vitturini/ijepa_lite/hacky_stl10_masker_probe.py \
    --ckpt "$CKPT" \
    --data_root "$DATA_ROOT" \
    --prefer target \
    --epochs 100 \
    --image_size 96 \
    --patch_size 8 \
    --embed_dim 384 \
    --depth 12 \
    --num_heads 6 \
    --mlp_ratio 4.0 \
    --batch_size 256 \
    --num_workers 4