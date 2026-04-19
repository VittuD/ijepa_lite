#!/bin/bash
#SBATCH --job-name=attn-orig-in1k
#SBATCH --account=IscrC_PRIMAL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=sbatch_logs/attn_viz/%x_%j.out
#SBATCH --error=sbatch_logs/attn_viz/%x_%j.err

set -euo pipefail

module purge
module load profile/deeplrn
module load python/3.11.6--gcc--8.5.0
module load imagenet

source $WORK/vitturini/ijepa_venv/bin/activate

nvidia-smi -L || true

export CKPT_ORIG="$WORK/vitturini/original_jepa_weights/IN1K-vit.h.14-300e.pth.tar"
export DATA_ROOT="$IMAGENET2012_HOME"
export N=100
export OUT_DIR="$WORK/vitturini/attn_orig_in1k_output/${SLURM_JOB_ID}"

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python $WORK/vitturini/ijepa_lite/scripts/hacky/hacky_visualize_attention_comparison.py \
    --ckpt-orig "$CKPT_ORIG" \
    --data-root "$DATA_ROOT" \
    --dataset imagenet \
    --out-dir "$OUT_DIR" \
    --n "$N"
