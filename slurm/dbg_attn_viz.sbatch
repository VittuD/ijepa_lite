#!/bin/bash
#SBATCH --job-name=attn-viz
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

source $WORK/vitturini/ijepa_venv/bin/activate

nvidia-smi -L || true

export CKPT="$WORK/vitturini/outputs/2026-03-13/10-30-44_vanilla_baseline_multiblock_stl10/rank0/checkpoints/last.pt"
export EXPERIMENT="stl10_vits_ps8_multiblock"
export DATA_ROOT="$FAST/datasets/"
export N=100
export LAYER=-1
export OUT_DIR="$WORK/vitturini/attn_viz_output/${SLURM_JOB_ID}"

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python $WORK/vitturini/ijepa_lite/hacky_visualize_attention.py \
    --ckpt "$CKPT" \
    --experiment "$EXPERIMENT" \
    --data-root "$DATA_ROOT" \
    --out-dir "$OUT_DIR" \
    --n "$N" \
    --layer "$LAYER"
