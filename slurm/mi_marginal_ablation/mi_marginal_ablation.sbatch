#!/bin/bash
#SBATCH --job-name=ijepa-mi-marginal
#SBATCH --account=IscrC_TBoneAI
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
# Marginal H(Y|X) ablation — does the conditional entropy term matter?
#
# Array layout:
#   idx   H_cond   neg_H_marg   neg_surprise   pos_embed   exp_name
#    0    ON       ON           ON             learned     mi_HcondHmargBS_learned
#    1    OFF      ON           ON             learned     mi_HmargBS_learned
#    2    ON       ON           ON             sincos      mi_HcondHmargBS_sincos
#    3    OFF      ON           ON             sincos      mi_HmargBS_sincos
#
# All runs keep floor_penalty=[5,5] and h_floor=0.1.
# ignore_tax, context_rate, target_rate remain disabled [0,0].
#
# Run:
#   sbatch slurm/mi_marginal_ablation/mi_marginal_ablation.sbatch
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

# H_cond weight: ON = [0.01,1.0], OFF = [0,0]
H_COND_WEIGHTS=("[0.01,1.0]" "[0,0]" "[0.01,1.0]" "[0,0]")
POS_EMBED_KINDS=(learned learned sincos sincos)
EXP_NAMES=(mi_HcondHmargBS_learned mi_HmargBS_learned mi_HcondHmargBS_sincos mi_HmargBS_sincos)

IDX=$SLURM_ARRAY_TASK_ID
H_COND=${H_COND_WEIGHTS[$IDX]}
POS_EMBED=${POS_EMBED_KINDS[$IDX]}
EXP_NAME=${EXP_NAMES[$IDX]}

echo "Array task ${IDX}: H_cond=${H_COND}, pos_embed=${POS_EMBED}, exp_name=${EXP_NAME}"

mkdir -p sbatch_logs/mi_marginal

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python -m torch.distributed.run --standalone --nproc_per_node=4 \
    -m ijepa_lite.run \
      experiment=stl10_vits_ps8_mi \
      exp_name="$EXP_NAME" \
      masking.latent.terms.H_cond.weight="$H_COND" \
      model.pos_embed_kind="$POS_EMBED" \
      data.root=$FAST/datasets/ \
      train.viz.out_dir="viz_output_${EXP_NAME}" \
      logger.mode=offline
