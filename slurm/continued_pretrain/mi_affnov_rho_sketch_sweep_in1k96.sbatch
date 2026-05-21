#!/bin/bash
#SBATCH --job-name=ijepa-ctp-in1k96-affnov-rhosweep
#SBATCH --account=IscrC_PRIMAL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --array=0-3
#SBATCH --output=sbatch_logs/continued_pretrain/in1k96_affnov_rho_sketch_sweep/%x_%A_%a.out
#SBATCH --error=sbatch_logs/continued_pretrain/in1k96_affnov_rho_sketch_sweep/%x_%A_%a.err

# ---------------------------------------------------------------------------
# Diagnostic 2x2 sweep on PURE affinity-novelty (no CosBS, no floor, no H_marg)
# continued from the IN1K 96x96 / ps8 vanilla-multiblock checkpoint.
#
# Why IN1K and only 5 epochs
# --------------------------
# The STL10/e999 version of this sweep settled the KERNEL question (sigma_rho is
# the eff_rank lever: 1.07 -> 1.27 -> 2.2 as rho 1 -> 0.5 -> 0.25; sketch_dim is
# irrelevant, so the recovered rank is encoder structure, not JL noise). But at
# 5 STL10 epochs the masker never committed (entropy_3way stuck at log 3, LR
# already decayed) so the MASK question -- does a higher-rank kernel actually
# yield adaptive / diverse masks -- was unanswerable. IN1K has ~625 steps/epoch
# vs ~50 on STL10, so 5 epochs here is ~3k masker steps: enough to commit while
# staying cheap. Geometry is identical (96px, ps8, 144 patches).
#
# Matrix (sigma_rho x sketch_dim):
#   0  rho=0.25  S=64
#   1  rho=0.25  S=256
#   2  rho=0.50  S=64
#   3  rho=0.50  S=256
# (S is already shown irrelevant on STL10; kept here only to re-confirm on IN1K.)
#
# Read-out:
#   affinity/eff_rank_pr_mean  -> confirm the rho lift reproduces on IN1K
#   mask/entropy_3way          -> does the masker COMMIT (fall below log 3)?
#   mask/batch_iou             -> with a committed masker + higher-rank kernel,
#                                 does cross-image diversity finally drop below
#                                 the ~0.24 baseline? If it stays ~0.24, the
#                                 static-mask problem is in the masker/training
#                                 dynamics, not the affinity kernel.
#
# Cooldown knobs (ema=0.99925 fixed, lr=6e-4, no warmup) match the in1k96 array.
#
# Submit:
#   sbatch slurm/continued_pretrain/mi_affnov_rho_sketch_sweep_in1k96.sbatch
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

mkdir -p sbatch_logs/continued_pretrain/in1k96_affnov_rho_sketch_sweep

CHECKPOINT_PATH="${WORK}/vitturini/outputs/2026-05-04/19-30-19_in1k_96_vits_ps8_vanilla_multiblock_300ep/rank0/checkpoints/last.pt"

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT_PATH}"
  exit 1
fi

echo "DATA_ROOT=${DATA_ROOT}"
ls -ld "$DATA_ROOT" "$DATA_ROOT/train" "$DATA_ROOT/val"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    SIGMA_RHO="0.25"
    SKETCH_DIM="64"
    ;;
  1)
    SIGMA_RHO="0.25"
    SKETCH_DIM="256"
    ;;
  2)
    SIGMA_RHO="0.5"
    SKETCH_DIM="64"
    ;;
  3)
    SIGMA_RHO="0.5"
    SKETCH_DIM="256"
    ;;
  *)
    echo "Unsupported SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
    exit 1
    ;;
esac

RHO_LABEL="${SIGMA_RHO/./p}"
EXP_NAME="in1k_96_vits_ps8_mi_affnov_rho${RHO_LABEL}_s${SKETCH_DIM}_in1k5ep"

nvidia-smi -L

echo "Array task ${SLURM_ARRAY_TASK_ID}: sigma_rho=${SIGMA_RHO} sketch_dim=${SKETCH_DIM} exp=${EXP_NAME}"

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
      masking.latent.terms.neg_affinity_novelty.sigma_rho="[${SIGMA_RHO},${SIGMA_RHO}]" \
      masking.latent.terms.neg_affinity_novelty.sketch_dim=${SKETCH_DIM} \
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
