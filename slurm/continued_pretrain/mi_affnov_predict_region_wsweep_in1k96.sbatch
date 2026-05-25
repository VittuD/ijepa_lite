#!/bin/bash
#SBATCH --job-name=ijepa-ctp-in1k96-predict-region-wsweep
#SBATCH --account=IscrC_PRIMAL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=sbatch_logs/continued_pretrain/in1k96_predict_region_wsweep/%x_%A_%a.out
#SBATCH --error=sbatch_logs/continued_pretrain/in1k96_predict_region_wsweep/%x_%A_%a.err

# ---------------------------------------------------------------------------
# REGION EXPERIMENT: grow the singleton into a target REGION via reward_norm=all,
# at the correct weight scale.
#
# Established so far:
#   * Validated winner (reward_norm=targets, w=1, standardize=FALSE) reproduces a
#     STABLE singleton: ctx contours the subject (coverage), tgt = the hard core,
#     ign = subject body. Plateaus -- no all-ctx drift. (singleton_repro run.)
#   * standardize=TRUE was the regionfix destabilizer (forces unit per-image RMS
#     -> amplifies noise once structure weakens -> all-ctx). Kept FALSE here.
#   * The first regionfix sweep (reward_norm=all) used w in {1,2,4} and formed NO
#     targets -> all-ctx. reward_norm=all divides the reward by a FIXED N (=144 at
#     96px/ps8), so at the singleton operating point (Sum p_tgt ~ 1) it needs
#     w ~ N just to MATCH the force that produced the singleton, and more to grow
#     it. w<=4 was ~36-144x too weak.
#
# The mechanism we want: with a fixed denominator the per-patch gradient is
# diff_j/N for EVERY patch, so every above-baseline-difficulty patch (the ign
# subject body in the repro) wants to be a target -> the singleton core expands
# into the hard REGION. Size emerges from balance vs affinity coverage; no count
# is injected (sign of the residualized difficulty decides membership).
#
# Sweep (all on residualize=true, standardize=FALSE, reward_norm=all):
#   0  predict_w=16
#   1  predict_w=32
#   2  predict_w=64
#   3  predict_w=128
# (~N=144 is the scale where all-norm matches targets-norm w=1; bracket around it.)
#
# Read-out (watch the TREND across the run):
#   mask/ntgt          -> THE metric: must grow from ~1 to a stable REGION (the
#                         subject body), not stay 1 and not run to all-active.
#                         Expect it to scale UP with predict_w.
#   mask/batch_iou     -> stay LOW (per-image); hold the ~0.02-0.05 win.
#   mask/nctx          -> hold; no all-ctx swing (would mean predict too weak) and
#                         no all-tgt swing (predict too strong, ctx starved).
#   predict/diff_tgt_mean -> per-target difficulty; a region is less extreme than
#                         the single max, so expect a small drop but clearly > 0.
#   predict/reward     -> rises with w; train/mse_raw + encoder_agreement must NOT
#                         tank (target region must stay learnable, not impossible).
#   affinity/C_act, affinity/c_mean -> must NOT climb toward all-ctx.
#
# Failure reads: still all-ctx at w=128 -> denominator/scale wrong, not weight
# (revisit reward_norm). all-tgt / encoder tank at low w -> predict too strong
# vs coverage (lower the bracket / raise affinity gamma).
#
# Cooldown knobs match the validated run: ema=0.99925, lr=6e-4, no warmup,
# bs=1024, affinity coverage-only (w_tgt=0, gamma=0.1, rho=0.25), no stabilizers.
#
# Submit:
#   sbatch slurm/continued_pretrain/mi_affnov_predict_region_wsweep_in1k96.sbatch
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

mkdir -p sbatch_logs/continued_pretrain/in1k96_predict_region_wsweep

CHECKPOINT_PATH="${WORK}/vitturini/outputs/2026-05-04/19-30-19_in1k_96_vits_ps8_vanilla_multiblock_300ep/rank0/checkpoints/last.pt"

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT_PATH}"
  exit 1
fi

echo "DATA_ROOT=${DATA_ROOT}"
ls -ld "$DATA_ROOT" "$DATA_ROOT/train" "$DATA_ROOT/val"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    PRED_W="16.0"
    LABEL="all_w16"
    ;;
  1)
    PRED_W="32.0"
    LABEL="all_w32"
    ;;
  2)
    PRED_W="64.0"
    LABEL="all_w64"
    ;;
  3)
    PRED_W="128.0"
    LABEL="all_w128"
    ;;
  *)
    echo "Unsupported SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
    exit 1
    ;;
esac

EXP_NAME="in1k_96_vits_ps8_mi_predict_res_${LABEL}_nostd_in1k10ep"

nvidia-smi -L

echo "Array task ${SLURM_ARRAY_TASK_ID}: reward_norm=all predict_w=${PRED_W} standardize=false exp=${EXP_NAME}"

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
      masking.latent.terms.neg_predictability.standardize=false \
      masking.latent.terms.neg_predictability.reward_norm=all \
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
