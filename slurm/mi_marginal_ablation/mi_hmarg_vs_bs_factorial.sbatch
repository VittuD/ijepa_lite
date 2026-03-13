#!/bin/bash
#SBATCH --job-name=ijepa-mi-hmarg-vs-bs
#SBATCH --account=IscrC_PRIMAL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
#SBATCH --array=0-1
#SBATCH --output=sbatch_logs/mi_marginal/%x_%A_%a.out
#SBATCH --error=sbatch_logs/mi_marginal/%x_%A_%a.err

# ---------------------------------------------------------------------------
# H(Y) vs Bayesian-surprise factorial with floor ON.
#
# Isolates whether both terms are load-bearing or one carries the other.
# H_cond disabled. Floor penalty fixed at [5,5], h_floor=0.1. Learned pos_embed.
#
# The reference run (both ON + floor) already exists — only the two
# single-term ablations are needed.
#
# Array layout:
#   idx   neg_H_marg   neg_surprise   exp_name
#    0    ON [1,1]     OFF [0,0]      mi_Hmarg_only_floor     (diversity only)
#    1    OFF [0,0]    ON  [0.5,0.5]  mi_BS_only_floor        (informativeness only)
#
# Compare with existing results:
#   mi_HmargBS_floor (both ON, floor ON) — already run
#   mi_HmargBS_learned = 0.706 (both ON, floor OFF)
#
# Run:
#   sbatch slurm/mi_marginal_ablation/mi_hmarg_vs_bs_factorial.sbatch
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

H_MARG_WEIGHTS=("[1.0,1.0]" "[0,0]")
BS_WEIGHTS=("[0,0]"      "[0.5,0.5]")
EXP_NAMES=(mi_Hmarg_only_floor mi_BS_only_floor)

IDX=$SLURM_ARRAY_TASK_ID
H_MARG=${H_MARG_WEIGHTS[$IDX]}
BS=${BS_WEIGHTS[$IDX]}
EXP_NAME=${EXP_NAMES[$IDX]}

echo "Array task ${IDX}: neg_H_marg=${H_MARG}, neg_surprise=${BS}, exp_name=${EXP_NAME}"

mkdir -p sbatch_logs/mi_marginal

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python -m torch.distributed.run --standalone --nproc_per_node=4 \
    -m ijepa_lite.run \
      experiment=stl10_vits_ps8_mi \
      exp_name="$EXP_NAME" \
      masking.latent.terms.H_cond.weight='[0,0]' \
      masking.latent.terms.neg_H_marg.weight="$H_MARG" \
      masking.latent.terms.neg_surprise.weight="$BS" \
      masking.latent.terms.floor_penalty.weight='[5.0,5.0]' \
      masking.latent.terms.ignore_tax.weight='[0,0]' \
      masking.latent.terms.context_rate.weight='[0,0]' \
      masking.latent.terms.target_rate.weight='[0,0]' \
      data.root=$FAST/datasets/ \
      data.batch_size=2048 \
      train.viz.out_dir="viz_output_${EXP_NAME}" \
      logger.mode=offline
