#!/bin/bash
#SBATCH --job-name=ijepa-ctp-in1k96-predict-regionfix
#SBATCH --account=IscrC_PRIMAL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=sbatch_logs/continued_pretrain/in1k96_predict_regionfix/%x_%A_%a.out
#SBATCH --error=sbatch_logs/continued_pretrain/in1k96_predict_regionfix/%x_%A_%a.err

# ---------------------------------------------------------------------------
# Fix the single-target degeneracy of the predictability role term.
#
# What worked (residualize=true, w=1): batch_iou crashed 0.9->0.018, masks went
# genuinely per-image and CONTOUR the subject (coverage spreads context, the
# role term finds the subject). Encoder healthy. THE THESIS WORKS.
#
# The degeneracy: reward was a MEAN over targets (Σ p_tgt·diff / Σ p_tgt) and a
# mean is maximized by a SINGLETON -> the masker put all target mass on the one
# hardest patch (ntgt_min=4 just clamps 3 filler). Spatial structure was right,
# target COUNT was broken.
#
# The fix: reward_norm="all" -> Σ p_tgt·diff / N (mean over ALL patches). The
# per-patch gradient is diff_j/N, so every above-baseline-difficulty patch wants
# to be a target -> targets become the hard REGION; size emerges from the SIGN of
# the (residualized) difficulty, i.e. content -- no count injected.
#
# Emergence knob: standardize=true per-image RMS-normalizes the difficulty, so
# the loss weight is a dimensionless O(1) threshold in std-units (transferable)
# instead of chasing the raw magnitude (which is why the ÷N-raw version needed
# ~×N weights). The target set/count stay emergent (sign preserved).
#
# Matrix (all on residualize=true + standardize=true, the validated winner):
#   0  reward_norm=all      w=1.0
#   1  reward_norm=all      w=2.0
#   2  reward_norm=all      w=4.0
#   3  reward_norm=targets  w=1.0   (CONTROL: reproduces the singleton)
#
# Read-out (watch the TREND across the run):
#   mask/ntgt          -> THE metric: must grow from ~1 to a stable REGION
#                         (the subject), not 1 and not all-active.
#   mask/batch_iou     -> stay LOW (per-image); should hold the ~0.02-0.05 win.
#   mask/nctx          -> hold (no all-ctx / all-ign swing as at w=2 before).
#   predict/diff_tgt_mean -> per-target difficulty (interpretable across norms);
#                         expect it to drop a little vs the singleton (a region is
#                         less extreme than the single max) but stay clearly > 0.
#   predict/reward     -> rises; train/mse_raw + encoder_agreement must NOT tank.
#   affinity/cand_gap_diff -> stays positive (per-image structure used).
#
# Cooldown knobs match the validated run: ema=0.99925, lr=6e-4, no warmup,
# bs=1024, affinity coverage-only (w_tgt=0, gamma=0.1, rho=0.25), no stabilizers.
#
# Submit:
#   sbatch slurm/continued_pretrain/mi_affnov_predict_regionfix_in1k96.sbatch
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

mkdir -p sbatch_logs/continued_pretrain/in1k96_predict_regionfix

CHECKPOINT_PATH="${WORK}/vitturini/outputs/2026-05-04/19-30-19_in1k_96_vits_ps8_vanilla_multiblock_300ep/rank0/checkpoints/last.pt"

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT_PATH}"
  exit 1
fi

echo "DATA_ROOT=${DATA_ROOT}"
ls -ld "$DATA_ROOT" "$DATA_ROOT/train" "$DATA_ROOT/val"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    RNORM="all"
    PRED_W="1.0"
    LABEL="all_w1"
    ;;
  1)
    RNORM="all"
    PRED_W="2.0"
    LABEL="all_w2"
    ;;
  2)
    RNORM="all"
    PRED_W="4.0"
    LABEL="all_w4"
    ;;
  3)
    RNORM="targets"
    PRED_W="1.0"
    LABEL="tgt_w1"
    ;;
  *)
    echo "Unsupported SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
    exit 1
    ;;
esac

EXP_NAME="in1k_96_vits_ps8_mi_predict_res_${LABEL}_in1k10ep"

nvidia-smi -L

echo "Array task ${SLURM_ARRAY_TASK_ID}: reward_norm=${RNORM} predict_w=${PRED_W} exp=${EXP_NAME}"

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
      data.batch_size=1024 \
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
      masking.latent.terms.neg_predictability.residualize=true \
      masking.latent.terms.neg_predictability.standardize=true \
      masking.latent.terms.neg_predictability.reward_norm=${RNORM} \
      masking.latent.hard_assignment=gumbel \
      masking.latent.gumbel_tau=1.0 \
      masking.latent.warmup_epochs=0 \
      masking.latent.nctx_min=1 \
      masking.latent.ntgt_min=4 \
      train.epochs=10 \
      train.eval_every_epochs=1 \
      train.viz_enabled=true \
      +train.viz_every_epochs=1 \
      train.viz.out_dir="viz_output_${EXP_NAME}" \
      logger.mode=offline
