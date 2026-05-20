#!/bin/bash
#SBATCH --job-name=ijepa-ctp-in1k96-affnov-cos-gamma
#SBATCH --account=IscrC_DEEPECHO
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --array=0-2
#SBATCH --output=sbatch_logs/continued_pretrain/in1k96_affnov_cos_gamma/%x_%A_%a.out
#SBATCH --error=sbatch_logs/continued_pretrain/in1k96_affnov_cos_gamma/%x_%A_%a.err

# ---------------------------------------------------------------------------
# affinity-novelty + real-centroid cos-surprise, NO stabilizers, sweeping the
# affinity ACTIVITY TAX (gamma). From the IN1K 96 vanilla checkpoint.
#
# Why gamma
# ---------
# The cos_w=1.0 run collapsed the WHOLE active set (nctx 43->1, nign 48->132)
# and tanked the encoder (mse_raw 0.29->0.59, encoder_agreement 0.82->0.37).
# The smoking gun: affinity/loss barely moved (-1.416 -> -1.477) across that
# catastrophe -- affinity is indifferent to (slightly prefers) collapse. Decomp:
#   lose coverage  -dU_ctx = +0.757
#   gain novelty   -dU_tgt = -0.348
#   save tax  coupling*dC_act = 0.805*(0.080-0.665) = -0.471
#   net = -0.062  => collapse is FREE for affinity.
# The activity tax C_act (loss = -U_ctx - w_tgt*U_tgt + gamma*log(1+alpha)*C_act)
# rewards ignoring uniformly with no floor, so the masker rides it to zero and
# cos (+0.329) tips it over. Lowering gamma makes the U_ctx coverage reward
# dominate, so shrinking the active set genuinely hurts affinity and it anchors
# both the active set AND ctx size against cos. (At gamma=0, step-50 affinity
# -1.951 vs collapsed -1.542 => +0.41 resistance, swamping cos's +0.33 pull.)
# This is an affinity-INTERNAL lever, not a stabilizer (no floor/H_marg/ceil).
#
# Sweep: neg_affinity_novelty.gamma in {0.0, 0.1, 0.25}  (current default 0.5)
#   0  gamma=0.0
#   1  gamma=0.1
#   2  gamma=0.25
#
# Read-out (step ~750): mask/nctx & affinity/active_coverage_mean must HOLD
# (no collapse); resid=(t-c)-(1-a)/alpha should break content-adaptively;
# mask/batch_iou should drop; and train/mse_raw + train/encoder_agreement must
# NOT collapse (masker gets no recon gradient, so this is the safety check).
#
# Cooldown knobs (ema=0.99925, lr=6e-4, no warmup) match the in1k96 array.
#
# Submit:
#   sbatch slurm/continued_pretrain/mi_affnov_cos_gamma_sweep_in1k96.sbatch
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

mkdir -p sbatch_logs/continued_pretrain/in1k96_affnov_cos_gamma

CHECKPOINT_PATH="${WORK}/vitturini/outputs/2026-05-04/19-30-19_in1k_96_vits_ps8_vanilla_multiblock_300ep/rank0/checkpoints/last.pt"

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT_PATH}"
  exit 1
fi

echo "DATA_ROOT=${DATA_ROOT}"
ls -ld "$DATA_ROOT" "$DATA_ROOT/train" "$DATA_ROOT/val"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    GAMMA="0.0"
    GLABEL="0p0"
    ;;
  1)
    GAMMA="0.1"
    GLABEL="0p1"
    ;;
  2)
    GAMMA="0.25"
    GLABEL="0p25"
    ;;
  *)
    echo "Unsupported SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
    exit 1
    ;;
esac

EXP_NAME="in1k_96_vits_ps8_mi_affnov_rho0p25_g${GLABEL}_cos1p0_in1k5ep"

nvidia-smi -L

echo "Array task ${SLURM_ARRAY_TASK_ID}: affinity gamma=${GAMMA} + cos_surprise w1 (no stabilizers) exp=${EXP_NAME}"

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
      masking.latent.terms.neg_affinity_novelty.gamma=${GAMMA} \
      masking.latent.terms.neg_affinity_novelty.sigma_rho="[0.25,0.25]" \
      masking.latent.terms.neg_affinity_novelty.sketch_dim=64 \
      +masking.latent.terms.neg_cos_surprise.weight="[1.0,1.0]" \
      masking.latent.hard_assignment=gumbel \
      masking.latent.gumbel_tau=1.0 \
      masking.latent.warmup_epochs=0 \
      masking.latent.nctx_min=1 \
      masking.latent.ntgt_min=4 \
      train.epochs=5 \
      train.eval_every_epochs=1 \
      train.viz_enabled=true \
      +train.viz_every_epochs=1 \
      train.viz.out_dir="viz_output_${EXP_NAME}" \
      logger.mode=offline
