#!/bin/bash
#SBATCH --job-name=ijepa-ctp-in1k96-affnov-socb
#SBATCH --account=IscrC_PRIMAL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
#SBATCH --array=0-1
#SBATCH --output=sbatch_logs/continued_pretrain/in1k96_affnov_sketchorthcosbs/%x_%A_%a.out
#SBATCH --error=sbatch_logs/continued_pretrain/in1k96_affnov_sketchorthcosbs/%x_%A_%a.err

# ---------------------------------------------------------------------------
# Two-run sweep over the CosBS assignment temperature, continued-pretraining
# from the IN1K 96x96 / ps8 vanilla-multiblock checkpoint. Two orthogonal
# masker objectives are combined:
#
#   neg_affinity_novelty            -> owns the active-vs-ignore / coverage axis
#   neg_sketched_orthogonal_cos_..  -> owns the ctx-vs-tgt role-contrast axis
#
# Sweep axis: assign_tau (sketched-orthogonal CosBS)
#   0  tau=0.25  baseline; matches array variant 4 / the single-run launcher
#   1  tau=0.05  sharpened; ~ the natural cosine scale 1/sqrt(D) for D=384, so
#                the K=8 random-anchor softmax actually separates buckets
#                instead of collapsing toward a near-uniform assignment (which
#                makes all K prototypes weak perturbations of one ctx centroid).
#
# Rationale
# ---------
# affinity-novelty alone parks the masker on the role-swap indifference
# manifold  1 - a = alpha * (t - c), so the ctx/tgt split is a flat direction
# and collapses to ~50/50. CosBS supplies the missing role curvature: its
# gradient on the ctx/tgt axis does NOT vanish at coverage saturation because
# it scores feature-space separation, not coverage. K=8 random prototypes are
# kept (random anchors need a margin above the true ctx mode count to separate
# them); assign_tau is the lever that decides whether those prototypes are
# actually distinct.
#
# Isolation: floor_penalty / H_cond / H_marg / surprise are all zeroed so the
# only forces are affinity (coverage) + CosBS (roles). Falsifiable signature of
# success: t_mean - c_mean peels away from (1 - active_coverage_mean) / alpha
# once the CosBS term bites; expect the tau=0.05 arm to peel further.
#
# Cooldown knobs (ema=0.99925 fixed, lr=6e-4, no warmup) match the in1k96 array.
#
# Submit:
#   sbatch slurm/continued_pretrain/mi_affnov_sketchorthcosbs_in1k96.sbatch
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

mkdir -p sbatch_logs/continued_pretrain/in1k96_affnov_sketchorthcosbs

CHECKPOINT_PATH="${WORK}/vitturini/outputs/2026-05-04/19-30-19_in1k_96_vits_ps8_vanilla_multiblock_300ep/rank0/checkpoints/last.pt"

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT_PATH}"
  exit 1
fi

echo "DATA_ROOT=${DATA_ROOT}"
ls -ld "$DATA_ROOT" "$DATA_ROOT/train" "$DATA_ROOT/val"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    ASSIGN_TAU="0.25"
    TAU_LABEL="tau025"
    ;;
  1)
    ASSIGN_TAU="0.05"
    TAU_LABEL="tau005"
    ;;
  *)
    echo "Unsupported SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
    exit 1
    ;;
esac

EXP_NAME="in1k_96_vits_ps8_mi_affnov_sketchorthcosbs_${TAU_LABEL}_in1k30_origlr_ema99925_nowarmup"

RUN_ARGS=(
  "masking.latent.terms.H_cond.weight=[0.0,0.0]"
  "masking.latent.terms.neg_H_marg.weight=[0.0,0.0]"
  "masking.latent.terms.neg_surprise.weight=[0.0,0.0]"
  "masking.latent.terms.floor_penalty.weight=[0.0,0.0]"
  "masking.latent.terms.neg_affinity_novelty.weight=[1.0,1.0]"
  "masking.latent.terms.neg_affinity_novelty.alpha=4.0"
  "masking.latent.terms.neg_affinity_novelty.w_tgt=1.0"
  "masking.latent.terms.neg_affinity_novelty.gamma=0.5"
  "masking.latent.terms.neg_sketched_orthogonal_cos_surprise.weight=[0.5,0.5]"
  "masking.latent.terms.neg_sketched_orthogonal_cos_surprise.num_sketches=8"
  "masking.latent.terms.neg_sketched_orthogonal_cos_surprise.assign_tau=${ASSIGN_TAU}"
  "masking.latent.hard_assignment=gumbel"
  "masking.latent.gumbel_tau=1.0"
  "masking.latent.warmup_epochs=0"
  "masking.latent.nctx_min=1"
  "masking.latent.ntgt_min=4"
)

COMMON_ARGS=(
  "experiment=stl10_vits_ps8_mi"
  "exp_name=${EXP_NAME}"
  "init_weights=${CHECKPOINT_PATH}"
  "loss.base_kind=smooth_l1"
  "data.name=imagenet"
  "data.root=${DATA_ROOT}"
  "data.pretrain_split=train"
  "data.train_split=train"
  "data.val_split=val"
  "data.download=false"
  "data.batch_size=2048"
  "model.ema_momentum=[0.99925,0.99925]"
  "optim.lr=6e-4"
  "sched.warmup_epochs=0"
  "train.epochs=30"
  "train.save_every_epochs=1"
  "train.eval_every_epochs=1"
  "train.viz_enabled=true"
  "+train.viz_every_epochs=1"
  "train.viz.out_dir=viz_output_${EXP_NAME}"
  "logger.mode=offline"
)

nvidia-smi -L

echo "Array task ${SLURM_ARRAY_TASK_ID}: affinity_novelty + sketched_orthogonal_cosbs assign_tau=${ASSIGN_TAU} exp=${EXP_NAME}"

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python -m torch.distributed.run --standalone --nproc_per_node=4 \
    -m ijepa_lite.run \
      "${COMMON_ARGS[@]}" \
      "${RUN_ARGS[@]}"
