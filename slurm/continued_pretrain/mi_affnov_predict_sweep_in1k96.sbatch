#!/bin/bash
#SBATCH --job-name=ijepa-ctp-in1k96-affnov-predict
#SBATCH --account=IscrC_PRIMAL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --array=0-3
#SBATCH --output=sbatch_logs/continued_pretrain/in1k96_affnov_predict/%x_%A_%a.out
#SBATCH --error=sbatch_logs/continued_pretrain/in1k96_affnov_predict/%x_%A_%a.err

# ---------------------------------------------------------------------------
# Affinity-novelty (COVERAGE only, w_tgt=0) + NegPredictabilityTerm (ROLE),
# continued from the IN1K 96x96 / ps8 vanilla-multiblock checkpoint. NO
# stabilizers (H_cond / H_marg / neg_surprise / floor all zeroed).
#
# Why this design
# ---------------
# The candidate diagnostic proved the COVERAGE objective is rank-invariant (no
# mask x image interaction): cand_gap ~3e-4, spread/|loss| ~0.18%, so its per-
# image signal (~0.03% of loss) was pure noise -> the masker never felt a
# gradient off the static solution and collapsed. The PREDICTABILITY cost (reuse
# A as an NW regressor; targets = hard to predict from context) instead
# differentiates masks ~62x more (spread/|loss| ~11%) and its per-image signal
# is ~94x stronger (~3.2% of loss) -- a gradient the masker can actually feel.
#
# Composition (the thesis): affinity owns the active-set / coverage axis (run
# with w_tgt=0 -> context coverage + activity tax only); predictability owns the
# ctx-vs-tgt axis. residualize subtracts each position's batch-mean difficulty:
# isolates "hard IN THIS image" and self-guards the ctx-collapse mode (emptying
# context makes difficulty uniform -> per-position-centered reward -> 0).
#
# Matrix (residualize x predict_weight):
#   0  residualize=true   pred_w=1.0
#   1  residualize=true   pred_w=2.0
#   2  residualize=false  pred_w=1.0
#   3  residualize=false  pred_w=2.0
#
# Read-out (debug budget: 1 epoch ~625 steps; first look at the de-static TREND,
# full 5ep run on a normal-QOS account if promising):
#   mask/batch_iou      -> must DROP toward ~0.24 (per-image masks at last!)
#   mask/entropy_3way   -> commits below log 3 without going fully decisive
#   predict/reward      -> rises (targets becoming harder to predict)
#   mask/nctx           -> must HOLD (no ctx collapse; watch residualize=false)
#   train/mse_raw, train/encoder_agreement -> must NOT tank (encoder health;
#                          the masker gets no recon gradient, so this is the
#                          safety check that a degenerate mask isn't starving it)
#   affinity/cand_gap_diff -> diagnostic: per-image difficulty structure present
#
# Cooldown knobs (ema=0.99925, lr=6e-4, no warmup) match the in1k96 array.
# Kernel: sigma_rho=0.25 / sketch_dim=64 (shared by affinity and predictability).
#
# Submit:
#   sbatch slurm/continued_pretrain/mi_affnov_predict_sweep_in1k96.sbatch
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

mkdir -p sbatch_logs/continued_pretrain/in1k96_affnov_predict

CHECKPOINT_PATH="${WORK}/vitturini/outputs/2026-05-04/19-30-19_in1k_96_vits_ps8_vanilla_multiblock_300ep/rank0/checkpoints/last.pt"

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT_PATH}"
  exit 1
fi

echo "DATA_ROOT=${DATA_ROOT}"
ls -ld "$DATA_ROOT" "$DATA_ROOT/train" "$DATA_ROOT/val"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    RESID="true"
    RLABEL="res"
    PRED_W="1.0"
    PLABEL="1p0"
    ;;
  1)
    RESID="true"
    RLABEL="res"
    PRED_W="2.0"
    PLABEL="2p0"
    ;;
  2)
    RESID="false"
    RLABEL="raw"
    PRED_W="1.0"
    PLABEL="1p0"
    ;;
  3)
    RESID="false"
    RLABEL="raw"
    PRED_W="2.0"
    PLABEL="2p0"
    ;;
  *)
    echo "Unsupported SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
    exit 1
    ;;
esac

EXP_NAME="in1k_96_vits_ps8_mi_affnovcov_predict_${RLABEL}_w${PLABEL}_in1k1ep"

nvidia-smi -L

echo "Array task ${SLURM_ARRAY_TASK_ID}: residualize=${RESID} predict_w=${PRED_W} exp=${EXP_NAME}"

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
      masking.latent.terms.neg_affinity_novelty.w_tgt=0.0 \
      masking.latent.terms.neg_affinity_novelty.gamma=0.1 \
      masking.latent.terms.neg_affinity_novelty.sigma_rho="[0.25,0.25]" \
      masking.latent.terms.neg_affinity_novelty.sketch_dim=64 \
      masking.latent.terms.neg_predictability.weight="[${PRED_W},${PRED_W}]" \
      masking.latent.terms.neg_predictability.sigma_rho=0.25 \
      masking.latent.terms.neg_predictability.sketch_dim=64 \
      masking.latent.terms.neg_predictability.residualize=${RESID} \
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
