#!/bin/bash
#SBATCH --job-name=masker-probe-coupled-sr
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

nvidia-smi -L || true

# TODO: update with the actual coupled-sr run timestamp after training completes
CKPT="$WORK/vitturini/outputs/stl10_vits_ps8_mi_coupled_sr/rank0/checkpoints/last.pt"
DATA_ROOT="$FAST/datasets"

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python $WORK/vitturini/ijepa_lite/hacky_stl10_masker_probe.py \
    --ckpt "$CKPT" \
    --data_root "$DATA_ROOT" \
    --prefer ema \
    --image_size 96 \
    --patch_size 8 \
    --embed_dim 384 \
    --depth 12 \
    --num_heads 6 \
    --mlp_ratio 4.0 \
    --predictor_dim 192 \
    --masker_depth 2 \
    --masker_heads 6 \
    --lam_min 0.01 \
    --lam_max 1.0 \
    --alpha_min 0.01 \
    --alpha_max 0.5 \
    --ntgt_min 4 \
    --nctx_min 1 \
    --h_floor 0.1 \
    --floor_weight 5.0 \
    --lr 1e-3 \
    --rates_lr 1e-2 \
    --epochs 20 \
    --batch_size 256 \
    --num_workers 4 \
    --max_iter 2000 \
    --C 1.0
