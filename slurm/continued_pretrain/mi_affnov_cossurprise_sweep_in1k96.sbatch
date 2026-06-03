#!/bin/bash
#SBATCH --job-name=ijepa-ctp-in1k96-affnov-cos
#SBATCH --account=IscrC_PRIMAL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=03:00:00
#SBATCH --array=0-2
#SBATCH --output=sbatch_logs/continued_pretrain/in1k96_affnov_cossurprise/%x_%A_%a.out
#SBATCH --error=sbatch_logs/continued_pretrain/in1k96_affnov_cossurprise/%x_%A_%a.err

# ---------------------------------------------------------------------------
# Affinity-novelty (active set / coverage) + REAL-centroid cos-surprise (roles),
# continued from the IN1K 96x96 / ps8 vanilla-multiblock checkpoint.
# NO stabilizers: floor_penalty / H_marg / neg_surprise are all zeroed.
#
# Hypothesis being tested
# -----------------------
# floor_penalty + H_marg are pure entropy stabilizers that manufacture diversity
# regardless of content; relying on them confounds the thesis. Pure affinity-
# novelty already self-regulates role OCCUPANCY (it held ~25/35/40 ctx/tgt/ign,
# never role-collapsed), so floor's actual job is covered. What it lacked was
# CONTENT-ADAPTIVITY: static active set (batch_iou rose to ~0.4) and dead roles
# (resid = (t-c)-(1-a)/alpha ~ 0, pinned on the indifference manifold). Both are
# plausibly the same defect -- a flat role direction with no content-rich
# gradient, so SGD took the lazy static solution.
#
# Bet: a REAL-centroid cos-surprise role term (tgt = far from the ctx centroid,
# 1 - cos(z, ctx_centroid)) supplies the missing content signal. It should (a)
# break the role manifold AND (b) de-static the mask, because a fixed target set
# cannot be "far from the context centroid" for every image -> content-varying
# masks become strictly better. Diversity should EMERGE, not be imposed.
#
# Sweep axis: cos-surprise weight (may need >= affinity weight to overcome the
# flat direction; sketch-orth at 0.5 never bit).
#   0  cos_w=0.5
#   1  cos_w=1.0
#   2  cos_w=2.0
#
# Read-out (step ~750, ~3k masker steps):
#   resid = (t_mean - c_mean) - (1 - active_coverage_mean)/alpha
#       -> must peel OFF zero == cos finally bites the role axis.
#   mask/batch_iou
#       -> must DROP (toward the ~0.24 of the stabilized reference) == emergent
#          content diversity with NO crutch. If it stays ~0.4 while resid breaks,
#          roles got fixed but the ACTIVE SET is still static -> affinity's
#          active-set selection itself needs to be made content-adaptive (a
#          separate fix), NOT stabilizers.
#   mask/nctx -> guard against degenerate (tiny) ctx gaming cheap surprise.
#   verify padding-avoidance is retained (affinity gates the active pool).
#
# Cooldown knobs (ema=0.99925 fixed, lr=6e-4, no warmup) match the in1k96 array.
# affinity kernel uses the sweep-winner setting rho=0.25 / sketch_dim=64.
#
# Submit:
#   sbatch slurm/continued_pretrain/mi_affnov_cossurprise_sweep_in1k96.sbatch
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

mkdir -p sbatch_logs/continued_pretrain/in1k96_affnov_cossurprise

CHECKPOINT_PATH="${WORK}/vitturini/outputs/2026-05-04/19-30-19_in1k_96_vits_ps8_vanilla_multiblock_300ep/rank0/checkpoints/last.pt"

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT_PATH}"
  exit 1
fi

echo "DATA_ROOT=${DATA_ROOT}"
ls -ld "$DATA_ROOT" "$DATA_ROOT/train" "$DATA_ROOT/val"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    COS_W="0.5"
    COS_LABEL="0p5"
    ;;
  1)
    COS_W="1.0"
    COS_LABEL="1p0"
    ;;
  2)
    COS_W="2.0"
    COS_LABEL="2p0"
    ;;
  *)
    echo "Unsupported SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
    exit 1
    ;;
esac

EXP_NAME="in1k_96_vits_ps8_mi_affnov_rho0p25_coscw${COS_LABEL}_in1k30ep"

nvidia-smi -L

echo "Array task ${SLURM_ARRAY_TASK_ID}: cos_surprise_weight=${COS_W} exp=${EXP_NAME}"

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
      masking.latent.terms.neg_affinity_novelty.gamma=0.5 \
      masking.latent.terms.neg_affinity_novelty.sigma_rho="[0.25,0.25]" \
      masking.latent.terms.neg_affinity_novelty.sketch_dim=64 \
      +masking.latent.terms.neg_cos_surprise.weight="[${COS_W},${COS_W}]" \
      masking.latent.hard_assignment=gumbel \
      masking.latent.gumbel_tau=1.0 \
      masking.latent.warmup_epochs=0 \
      masking.latent.nctx_min=1 \
      masking.latent.ntgt_min=4 \
      train.epochs=30 \
      train.eval_every_epochs=1 \
      train.viz_enabled=true \
      +train.viz_every_epochs=1 \
      train.viz.out_dir="viz_output_${EXP_NAME}" \
      logger.mode=offline
