#!/bin/bash
#SBATCH --job-name=ijepa-ctp-in1k96-predict-region-w37to43
#SBATCH --account=IscrC_PRIMAL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --array=0-6
#SBATCH --output=sbatch_logs/continued_pretrain/in1k96_predict_region_w37to43/%x_%A_%a.out
#SBATCH --error=sbatch_logs/continued_pretrain/in1k96_predict_region_w37to43/%x_%A_%a.err

# ---------------------------------------------------------------------------
# REGION EXPERIMENT: grow the singleton into a target REGION via reward_norm=all,
# at the correct weight scale.
#
# Sweep (all on residualize=true, standardize=FALSE, reward_norm=all):
#   0  predict_w=37
#   1  predict_w=38
#   2  predict_w=39
#   3  predict_w=40
#   4  predict_w=41
#   5  predict_w=42
#   6  predict_w=43
#
# Narrow integer sweep around the lower edge of the previous 36-60 bracket.
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

mkdir -p sbatch_logs/continued_pretrain/in1k96_predict_region_w37to43

CHECKPOINT_PATH="${WORK}/vitturini/outputs/2026-05-04/19-30-19_in1k_96_vits_ps8_vanilla_multiblock_300ep/rank0/checkpoints/last.pt"

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT_PATH}"
  exit 1
fi

echo "DATA_ROOT=${DATA_ROOT}"
ls -ld "$DATA_ROOT" "$DATA_ROOT/train" "$DATA_ROOT/val"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    PRED_W="37"
    LABEL="all_w37"
    ;;
  1)
    PRED_W="38"
    LABEL="all_w38"
    ;;
  2)
    PRED_W="39"
    LABEL="all_w39"
    ;;
  3)
    PRED_W="40"
    LABEL="all_w40"
    ;;
  4)
    PRED_W="41"
    LABEL="all_w41"
    ;;
  5)
    PRED_W="42"
    LABEL="all_w42"
    ;;
  6)
    PRED_W="43"
    LABEL="all_w43"
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
