#!/bin/bash
#SBATCH --job-name=ijepa-ctp-in1k96-affnov-candtest
#SBATCH --account=IscrC_DEEPECHO
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --output=sbatch_logs/continued_pretrain/in1k96_affnov_cand_test/%x_%j.out
#SBATCH --error=sbatch_logs/continued_pretrain/in1k96_affnov_cand_test/%x_%j.err

# ---------------------------------------------------------------------------
# Validate the masker-FREE candidate test (mask-interchangeability v2) and read
# the decisive number. Single arm, gamma=0.1 + cos_surprise w1, no stabilizers,
# IN1K, 1 epoch (~625 steps -> ~12 readings at the 50-step throttle).
#
# Why this supersedes the swap test
# ---------------------------------
# The roll-based swap test read ~1e-7 (vs a ~1.5 loss scale) because it used the
# MASKER'S own masks, which are never both fitted-per-image AND diverse: random
# early (not fitted), static late (not diverse) -> own ~= foreign trivially in
# both regimes, consistent with EITHER hypothesis. Uninformative.
#
# The candidate test bypasses the masker. It evaluates K=16 random partitions
# (shared across the batch, activity matched to the masker's current occupancy)
# on every image's own affinity A/Z, and asks whether one mask is best for
# everyone or each image prefers its own.
#
# Prior result (raw affinity): cand_gap ~3e-4 vs |loss| ~2.2 (~0.02%),
# cand_argmin_frac ~0.25, cand_global_loss flat at -2.196 => the RAW objective is
# image-invariant and barely mask-sensitive (best-of-16-random ~ trained masker).
# This run adds the spread + residual-A probes to localize WHY and whether it's
# recoverable.
#
# Read-out (affinity/cand_*):
#   cand_gap = mean_i[ L(k_global*, i) - min_k L(k, i) ]; ~0 => one mask is best
#              for everyone => objective image-invariant.
#   cand_spread = mean_i[max_k L - min_k L]: how much the objective differentiates
#              masks at all. spread~gap~0 => mask-insensitive (gap~0 trivial).
#              spread>>gap => masks differ but ranking is image-invariant.
#   cand_gap_resid / cand_spread_resid -> SAME test after removing the position-
#              conditional (stationary) feature mode. gap_resid >> gap => per-image
#              structure EXISTS but the stationary mode hid it (=> residualize A
#              before training). gap_resid still ~0 => even that can't extract it
#              => need a cross-image contrastive objective, not coverage.
#   cand_argmin_frac(_resid) ~1 => invariant; low + gap~0 => tied/flat.
#   cand_nctx / cand_ntgt -> activity the candidates were matched to (sanity).
#
# Interpret all gaps/spreads relative to |cand_global_loss| (~2.2).
#
# Submit:
#   sbatch slurm/continued_pretrain/mi_affnov_cand_test_in1k96.sbatch
# ---------------------------------------------------------------------------

set -euo pipefail

module purge
module load profile/deeplrn
module load python/3.11.6--gcc--8.5.0
module load imagenet

source $WORK/vitturini/ijepa_venv/bin/activate

export DATA_ROOT="$IMAGENET2012_HOME"
export WANDB_MODE=offline

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

mkdir -p sbatch_logs/continued_pretrain/in1k96_affnov_cand_test

CHECKPOINT_PATH="${WORK}/vitturini/outputs/2026-05-04/19-30-19_in1k_96_vits_ps8_vanilla_multiblock_300ep/rank0/checkpoints/last.pt"

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT_PATH}"
  exit 1
fi

echo "DATA_ROOT=${DATA_ROOT}"
ls -ld "$DATA_ROOT" "$DATA_ROOT/train" "$DATA_ROOT/val"

EXP_NAME="in1k_96_vits_ps8_mi_affnov_rho0p25_g0p1_cos1p0_candtest"

nvidia-smi -L

echo "Candidate-test validation run: gamma=0.1 + cos_surprise w1 (no stabilizers) exp=${EXP_NAME}"

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python -m torch.distributed.run --standalone --nproc_per_node=4 \
    -m ijepa_lite.run \
      experiment=stl10_vits_ps8_mi \
      exp_name="$EXP_NAME" \
      init_weights="$CHECKPOINT_PATH" \
      loss.base_kind=smooth_l1 \
      data.name=imagenet \
      data.root="${DATA_ROOT}" \
      data.pretrain_split=train \
      data.train_split=train \
      data.val_split=val \
      data.download=false \
      data.batch_size=2048 \
      model.ema_momentum="[0.99925,0.99925]" \
      optim.lr=6e-4 \
      sched.warmup_epochs=0 \
      masking.latent.terms.H_cond.weight="[0.0,0.0]" \
      masking.latent.terms.neg_H_marg.weight="[0.0,0.0]" \
      masking.latent.terms.neg_surprise.weight="[0.0,0.0]" \
      masking.latent.terms.floor_penalty.weight="[0.0,0.0]" \
      masking.latent.terms.neg_affinity_novelty.weight="[1.0,1.0]" \
      masking.latent.terms.neg_affinity_novelty.alpha=4.0 \
      masking.latent.terms.neg_affinity_novelty.w_tgt=1.0 \
      masking.latent.terms.neg_affinity_novelty.gamma=0.1 \
      masking.latent.terms.neg_affinity_novelty.sigma_rho="[0.25,0.25]" \
      masking.latent.terms.neg_affinity_novelty.sketch_dim=64 \
      +masking.latent.terms.neg_cos_surprise.weight="[1.0,1.0]" \
      masking.latent.hard_assignment=gumbel \
      masking.latent.gumbel_tau=1.0 \
      masking.latent.warmup_epochs=0 \
      masking.latent.nctx_min=1 \
      masking.latent.ntgt_min=4 \
      train.epochs=1 \
      train.eval_every_epochs=1 \
      train.viz_enabled=true \
      +train.viz_every_epochs=1 \
      train.viz.out_dir="viz_output_${EXP_NAME}" \
      logger.mode=offline
