#!/bin/bash
#SBATCH --job-name=ijepa-mi-lambda
#SBATCH --account=IscrC_PRIMAL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
#SBATCH --array=0-6
#SBATCH --output=sbatch_logs/mi_lambda/%x_%A_%a.out
#SBATCH --error=sbatch_logs/mi_lambda/%x_%A_%a.err

# ---------------------------------------------------------------------------
# Log-ratio sweep for H(Y) vs Bayesian-surprise trade-off.
#
# Masker maximises:  α · H(Y) + β · BS + floor_penalty
#
# We fix total scale α+β = 1 and sweep the ratio r = α/β in log-space:
#   α = r/(1+r),   β = 1/(1+r)
#
# This covers 3 orders of magnitude and guarantees we cross the balance
# point regardless of the terms' natural scales.
#
# H_cond disabled. Floor penalty fixed at [5,5], h_floor=0.1. Learned pos_embed.
#
# NOTE: Run mi_hmarg_vs_bs_factorial.sbatch first to confirm both terms are
#       load-bearing before spending compute on this sweep.
#
# Array layout (7 tasks):
#   idx   r=α/β   α (H_marg)   β (surprise)   exp_name
#    0    0.03    0.03         0.97            mi_ratio_003
#    1    0.10    0.09         0.91            mi_ratio_010
#    2    0.30    0.23         0.77            mi_ratio_030
#    3    1.00    0.50         0.50            mi_ratio_100
#    4    3.00    0.75         0.25            mi_ratio_300
#    5    10.0    0.91         0.09            mi_ratio_1000
#    6    30.0    0.97         0.03            mi_ratio_3000
#
# Run:
#   sbatch slurm/mi_marginal_ablation/mi_lambda_sweep.sbatch
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

# Log-spaced ratios: r = α/β
RATIOS=(0.03 0.10 0.30 1.00 3.00 10.00 30.00)
RATIO_TAGS=(003 010 030 100 300 1000 3000)

IDX=$SLURM_ARRAY_TASK_ID
R=${RATIOS[$IDX]}
TAG=${RATIO_TAGS[$IDX]}

# α = r/(1+r), β = 1/(1+r)
ALPHA=$(awk "BEGIN {printf \"%.4f\", $R / (1.0 + $R)}")
BETA=$(awk  "BEGIN {printf \"%.4f\", 1.0 / (1.0 + $R)}")

EXP_NAME="mi_ratio_${TAG}"

echo "Array task ${IDX}: ratio=${R}, alpha(H_marg)=${ALPHA}, beta(BS)=${BETA}, exp_name=${EXP_NAME}"

mkdir -p sbatch_logs/mi_lambda

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python -m torch.distributed.run --standalone --nproc_per_node=4 \
    -m ijepa_lite.run \
      experiment=stl10_vits_ps8_mi \
      exp_name="$EXP_NAME" \
      masking.latent.terms.H_cond.weight='[0,0]' \
      masking.latent.terms.neg_H_marg.weight="[${ALPHA},${ALPHA}]" \
      masking.latent.terms.neg_surprise.weight="[${BETA},${BETA}]" \
      masking.latent.terms.floor_penalty.weight='[5.0,5.0]' \
      masking.latent.terms.ignore_tax.weight='[0,0]' \
      masking.latent.terms.context_rate.weight='[0,0]' \
      masking.latent.terms.target_rate.weight='[0,0]' \
      data.root=$FAST/datasets/ \
      data.batch_size=2048 \
      train.viz.out_dir="viz_output_${EXP_NAME}" \
      logger.mode=offline
