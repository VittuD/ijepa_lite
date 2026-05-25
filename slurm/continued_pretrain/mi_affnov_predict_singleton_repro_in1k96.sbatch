#!/bin/bash
#SBATCH --job-name=ijepa-ctp-in1k96-predict-singleton-repro
#SBATCH --account=IscrC_PRIMAL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --output=sbatch_logs/continued_pretrain/in1k96_predict_singleton_repro/%x_%j.out
#SBATCH --error=sbatch_logs/continued_pretrain/in1k96_predict_singleton_repro/%x_%j.err

# ---------------------------------------------------------------------------
# REGRESSION TEST: reproduce the validated singleton.
#
# The regionfix sweep (reward_norm=all + standardize=true) collapsed every arm,
# INCLUDING the reward_norm=targets control, which only started as a singleton
# then drifted to all-ctx by ep10. The single config delta from the validated
# winner was standardize=true, so before doing anything else we re-pin the
# validated winner EXACTLY and confirm it still holds the stable singleton.
#
# Validated winner = residualize=true, standardize=FALSE, reward_norm=targets,
# predict_w=1.0. Behavior to reproduce: masks go genuinely per-image, CONTOUR
# the subject, batch_iou crashes ~0.9 -> ~0.02, encoder stays healthy. The
# target COUNT degenerates to ~1 (the singleton) -- THAT is expected here; this
# run is the baseline, not the fix.
#
# Why standardize=false: standardize forces unit per-image RMS on the residual
# difficulty, which amplifies noise to unit variance once the genuine per-image
# structure weakens -> destabilizes the masker into the affinity all-ctx
# attractor. It does NOT change the argmax (positive per-image constant), which
# is why the regionfix control still STARTED as a singleton before drifting.
#
# Read-out (the singleton must be STABLE across the whole run, not just ep0):
#   mask/ntgt          -> ~1 (clamped to ntgt_min filler); must NOT climb to
#                         all-active nor decay into all-ctx.
#   mask/batch_iou     -> crash to ~0.02-0.05 and HOLD (per-image masks).
#   mask/nctx          -> hold; no all-ctx / all-ign swing late in the run.
#   predict/diff_tgt_mean -> per-target difficulty, clearly > 0 and stable.
#   predict/reward     -> positive, stable (NOT manufactured-noise drift).
#   affinity/C_act, affinity/c_mean -> must NOT climb toward all-ctx.
#   train/mse_raw + encoder_agreement -> healthy, no tank.
#
# Cooldown knobs match the validated run: ema=0.99925, lr=6e-4, no warmup,
# bs=1024, affinity coverage-only (w_tgt=0, gamma=0.1, rho=0.25), no stabilizers.
#
# Submit:
#   sbatch slurm/continued_pretrain/mi_affnov_predict_singleton_repro_in1k96.sbatch
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

mkdir -p sbatch_logs/continued_pretrain/in1k96_predict_singleton_repro

CHECKPOINT_PATH="${WORK}/vitturini/outputs/2026-05-04/19-30-19_in1k_96_vits_ps8_vanilla_multiblock_300ep/rank0/checkpoints/last.pt"

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT_PATH}"
  exit 1
fi

echo "DATA_ROOT=${DATA_ROOT}"
ls -ld "$DATA_ROOT" "$DATA_ROOT/train" "$DATA_ROOT/val"

EXP_NAME="in1k_96_vits_ps8_mi_predict_res_tgt_w1_nostd_in1k10ep"

nvidia-smi -L

echo "Singleton repro: reward_norm=targets predict_w=1.0 standardize=false exp=${EXP_NAME}"

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
      masking.latent.terms.neg_predictability.weight="[1.0,1.0]" \
      masking.latent.terms.neg_predictability.sigma_rho=0.25 \
      masking.latent.terms.neg_predictability.sketch_dim=64 \
      masking.latent.terms.neg_predictability.residualize=true \
      masking.latent.terms.neg_predictability.standardize=false \
      masking.latent.terms.neg_predictability.reward_norm=targets \
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
