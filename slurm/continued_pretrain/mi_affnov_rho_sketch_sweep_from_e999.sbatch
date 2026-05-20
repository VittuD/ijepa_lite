#!/bin/bash
#SBATCH --job-name=ijepa-ctp-affnov-rhosweep
#SBATCH --account=IscrC_PRIMAL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --array=0-3
#SBATCH --output=sbatch_logs/continued_pretrain/affnov_rho_sketch_sweep/%x_%A_%a.out
#SBATCH --error=sbatch_logs/continued_pretrain/affnov_rho_sketch_sweep/%x_%A_%a.err

# ---------------------------------------------------------------------------
# Diagnostic 2x2 sweep on PURE affinity-novelty (no CosBS, no floor, no H_marg)
# continued from the STL10 vanilla-multiblock epoch_00999 checkpoint -- the
# original deterministic-collapse case.
#
# Question being tested
# ---------------------
# The cosine-RBF affinity kernel A is rank-1 on the pretrained encoder
# (eff_rank(A) ~ 1.07), so c/t are position-independent -> static masks. This
# is driven by the median bandwidth self-normalizing away the (tight) spread of
# patch distances, NOT by the JL sketch being too small. Two coupled levers:
#
#   sigma_rho < 1  -> sharpen the RBF, decouple from the degenerate median,
#                     amplify whatever real patch spread exists.
#   sketch_dim up  -> only matters once sharpened: sharpening also amplifies the
#                     ~sqrt(2/S) per-pair JL noise (17.7% at S=64, 8.8% at S=256),
#                     so a bigger sketch is needed to keep recovered spread =
#                     signal, not noise.
#
# Matrix (sigma_rho x sketch_dim):
#   0  rho=0.25  S=64
#   1  rho=0.25  S=256
#   2  rho=0.50  S=64
#   3  rho=0.50  S=256
#
# Read-out (per task, watch from step 50 onward):
#   affinity/eff_rank_pr_mean  -> must lift well above 1 for any spatial signal
#   mask/batch_iou             -> must stay LOW (diverse across images) and stable
#   compare S=64 vs S=256 at fixed rho: if masks track sketch_seed noise (flicker,
#   high run-to-run variance) the lift is JL garbage, not encoder structure.
#
# Positive control already on record: from-SCRATCH STL10 yields adaptive masks,
# so STL10 content + a non-collapsed encoder supports eff_rank > 1. This sweep
# asks whether a sharpened kernel can recover that from the COLLAPSED e999
# encoder. If even rho=0.25/S=256 stays rank-1 with static masks, the e999
# encoder's last-layer token geometry is the wall and affinity-novelty is the
# wrong tool on this checkpoint.
#
# Submit:
#   sbatch slurm/continued_pretrain/mi_affnov_rho_sketch_sweep_from_e999.sbatch
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

mkdir -p sbatch_logs/continued_pretrain/affnov_rho_sketch_sweep

CHECKPOINT_PATH="${WORK}/vitturini/outputs/2026-05-04/19-42-25_vanilla_baseline_multiblock_stl10/rank0/checkpoints/epoch_00999.pt"

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT_PATH}"
  exit 1
fi

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
EXP_NAME="stl10_mi_affnov_rho${RHO_LABEL}_s${SKETCH_DIM}_from_e999"

nvidia-smi -L

echo "Array task ${SLURM_ARRAY_TASK_ID}: sigma_rho=${SIGMA_RHO} sketch_dim=${SKETCH_DIM} exp=${EXP_NAME}"

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python -m torch.distributed.run --standalone --nproc_per_node=4 \
    -m ijepa_lite.run \
      experiment=stl10_vits_ps8_mi \
      exp_name="$EXP_NAME" \
      init_weights="$CHECKPOINT_PATH" \
      loss.base_kind=smooth_l1 \
      data.root=$FAST/datasets/ \
      data.batch_size=2048 \
      optim.lr=3e-4 \
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
