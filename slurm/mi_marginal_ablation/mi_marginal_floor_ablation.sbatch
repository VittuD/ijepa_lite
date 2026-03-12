#!/bin/bash
#SBATCH --job-name=ijepa-mi-floor
#SBATCH --account=IscrC_PRIMAL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
#SBATCH --array=0-3
#SBATCH --output=sbatch_logs/mi_marginal/%x_%A_%a.out
#SBATCH --error=sbatch_logs/mi_marginal/%x_%A_%a.err

# ---------------------------------------------------------------------------
# Complement to mi_marginal_ablation — adds floor_penalty=ON to fill the 2x2x2
# design (H_cond x floor_penalty x pos_embed).
#
# Already done (floor=OFF, see mi_marginal_ablation.sbatch):
#   H_cond=OFF, floor=OFF, learned   -> 0.706
#   H_cond=OFF, floor=OFF, sincos    -> 0.682
#   H_cond=ON,  floor=OFF, learned   -> 0.546 (collapsed: batch_iou=1.0)
#   H_cond=ON,  floor=OFF, sincos    -> 0.614 (collapsed: batch_iou=1.0)
#
# This script (floor=ON):
#   idx   H_cond   floor   pos_embed   exp_name
#    0    ON       ON      learned     mi_HcondHmargBS_floor_learned
#    1    OFF      ON      learned     mi_HmargBS_floor_learned
#    2    ON       ON      sincos      mi_HcondHmargBS_floor_sincos
#    3    OFF      ON      sincos      mi_HmargBS_floor_sincos
#
# Why each cell:
#   idx 0,2 — critical: does floor_penalty rescue H_cond from collapse?
#             does H_cond add value over baseline once properly guarded?
#   idx 1,3 — control: floor_penalty should be near-neutral when H_cond is
#             absent (nothing drives H(Y|n) down, so ReLU(floor - H(Y|n))
#             stays at ~0). Confirms floor_penalty only activates under pressure.
#
# All term weights fixed (no LogUniform sampling).
# neg_H_marg=[1,1], neg_surprise=[0.5,0.5], floor_penalty=[5,5], h_floor=0.1.
#
# Run:
#   sbatch slurm/mi_marginal_ablation/mi_marginal_floor_ablation.sbatch
# ---------------------------------------------------------------------------

set -euo pipefail

module purge
module load profile/deeplrn
module load python/3.11.6--gcc--8.5.0

source $WORK/vitturini/ijepa_venv/bin/activate

export WANDB_MODE=offline

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

nvidia-smi -L

# ---- Per-task configuration ----

# H_cond weight: ON = [1.0,1.0] (fixed), OFF = [0,0]
H_COND_WEIGHTS=("[1.0,1.0]" "[0,0]" "[1.0,1.0]" "[0,0]")
POS_EMBED_KINDS=(learned learned sincos sincos)
EXP_NAMES=(mi_HcondHmargBS_floor_learned mi_HmargBS_floor_learned mi_HcondHmargBS_floor_sincos mi_HmargBS_floor_sincos)

IDX=$SLURM_ARRAY_TASK_ID
H_COND=${H_COND_WEIGHTS[$IDX]}
POS_EMBED=${POS_EMBED_KINDS[$IDX]}
EXP_NAME=${EXP_NAMES[$IDX]}

echo "Array task ${IDX}: H_cond=${H_COND}, floor=ON, pos_embed=${POS_EMBED}, exp_name=${EXP_NAME}"

mkdir -p sbatch_logs/mi_marginal

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python -m torch.distributed.run --standalone --nproc_per_node=4 \
    -m ijepa_lite.run \
      experiment=stl10_vits_ps8_mi \
      exp_name="$EXP_NAME" \
      masking.latent.terms.H_cond.weight="$H_COND" \
      masking.latent.terms.neg_H_marg.weight='[1.0,1.0]' \
      masking.latent.terms.neg_surprise.weight='[0.5,0.5]' \
      masking.latent.terms.floor_penalty.weight='[5.0,5.0]' \
      model.pos_embed_kind="$POS_EMBED" \
      data.root=$FAST/datasets/ \
      data.batch_size=2048 \
      train.viz.out_dir="viz_output_${EXP_NAME}" \
      logger.mode=offline
