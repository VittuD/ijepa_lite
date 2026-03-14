#!/bin/bash
#SBATCH --job-name=attn-viz
#SBATCH --account=IscrC_PRIMAL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=sbatch_logs/attn_viz/%x_%j.out
#SBATCH --error=sbatch_logs/attn_viz/%x_%j.err

set -euo pipefail

module purge
module load profile/deeplrn
module load python/3.11.6--gcc--8.5.0

source $WORK/vitturini/ijepa_venv/bin/activate

nvidia-smi -L || true

CKPT="${1:?Usage: sbatch dbg_attn_viz.sbatch <ckpt_path> [experiment]}"
EXPERIMENT="${2:-stl10_vits_ps8_multiblock}"

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python $WORK/vitturini/ijepa_lite/hacky_visualize_attention.py \
    --ckpt "$CKPT" \
    --experiment "$EXPERIMENT" \
    --data-root "$FAST/datasets/" \
    --out-dir "$WORK/vitturini/attn_viz_output" \
    --n 200 \
    --mode cls \
    --layer -1
