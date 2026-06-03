#!/bin/bash
#SBATCH --job-name=ijepa-ctp-chest128-all
#SBATCH --account=IscrC_PRIMAL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=12:30:00
#SBATCH --array=0-4
#SBATCH --output=sbatch_logs/continued_pretrain/chestmnist224to128_all_from_in1k96_e170_origlr_ema99925_nowarmup/%x_%A_%a.out
#SBATCH --error=sbatch_logs/continued_pretrain/chestmnist224to128_all_from_in1k96_e170_origlr_ema99925_nowarmup/%x_%A_%a.err

# ---------------------------------------------------------------------------
# Submit all five 128x128 / ps8 continued-pretraining variants from one command,
# starting from the IN1K 96x96 / ps8 vanilla-multiblock checkpoint and switching
# the pretraining data to ChestMNIST 224px source images over all splits.
#
# ChestMNIST-all has 112,120 samples. With data.batch_size=1024 per process and
# 4 processes, 170 epochs is roughly the same optimizer-step budget as the
# original 15-epoch ImageNet-1K 128px run.
#
# Matrix:
#   0  vanilla_multiblock
#   1  mi_affinity_novelty
#   2  mi_hmargbs_floor
#   3  mi_hmargcosbs_floor
#   4  mi_hmarg_sketchorthcosbs_floor
#
# Submit:
#   sbatch slurm/continued_pretrain/continued_pretrain_chestmnist128_all_array.sbatch
# ---------------------------------------------------------------------------

set -euo pipefail

module purge
module load profile/deeplrn
module load python/3.11.6--gcc--8.5.0

source $WORK/vitturini/ijepa_venv/bin/activate

export DATA_ROOT="${CHESTMNIST_ROOT:-/leonardo_scratch/fast/IscrC_TBoneAI/datasets}"
export WANDB_MODE=offline

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

mkdir -p sbatch_logs/continued_pretrain/chestmnist224to128_all_from_in1k96_e170_origlr_ema99925_nowarmup

CHECKPOINT_PATH="${WORK}/vitturini/outputs/2026-05-04/19-30-19_in1k_96_vits_ps8_vanilla_multiblock_300ep/rank0/checkpoints/last.pt"

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT_PATH}"
  exit 1
fi

echo "DATA_ROOT=${DATA_ROOT}"
ls -ld "$DATA_ROOT"
ls -lh "$DATA_ROOT"/chestmnist_224.npz

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    RUN_LABEL="vanilla_multiblock"
    EXPERIMENT="stl10_vits_ps8_multiblock"
    EXP_NAME="chestmnist_128_vits_ps8_vanilla_multiblock_from_in1k96_e170_origlr_ema99925_nowarmup"
    LOSS_ARG="loss.kind=smooth_l1"
    RUN_ARGS=()
    ;;
  1)
    RUN_LABEL="mi_affinity_novelty"
    EXPERIMENT="stl10_vits_ps8_mi"
    EXP_NAME="chestmnist_128_vits_ps8_mi_affinity_novelty_from_in1k96_e170_origlr_ema99925_nowarmup"
    LOSS_ARG="loss.base_kind=smooth_l1"
    RUN_ARGS=(
      "masking.latent.terms.H_cond.weight=[0.0,0.0]"
      "masking.latent.terms.neg_H_marg.weight=[0.0,0.0]"
      "masking.latent.terms.neg_surprise.weight=[0.0,0.0]"
      "masking.latent.terms.floor_penalty.weight=[0.0,0.0]"
      "masking.latent.terms.neg_affinity_novelty.weight=[1.0,1.0]"
      "masking.latent.terms.neg_affinity_novelty.alpha=4.0"
      "masking.latent.terms.neg_affinity_novelty.w_tgt=1.0"
      "masking.latent.terms.neg_affinity_novelty.gamma=0.5"
      "masking.latent.hard_assignment=gumbel"
      "masking.latent.gumbel_tau=1.0"
      "masking.latent.warmup_epochs=0"
      "masking.latent.nctx_min=1"
      "masking.latent.ntgt_min=4"
    )
    ;;
  2)
    RUN_LABEL="mi_hmargbs_floor"
    EXPERIMENT="stl10_vits_ps8_mi"
    EXP_NAME="chestmnist_128_vits_ps8_mi_hmargbs_floor_from_in1k96_e170_origlr_ema99925_nowarmup"
    LOSS_ARG="loss.base_kind=smooth_l1"
    RUN_ARGS=(
      "masking.latent.terms.H_cond.weight=[0.0,0.0]"
      "masking.latent.terms.neg_H_marg.weight=[1.0,1.0]"
      "masking.latent.terms.neg_surprise.weight=[0.5,0.5]"
      "masking.latent.terms.floor_penalty.weight=[5.0,5.0]"
      "masking.latent.hard_assignment=gumbel"
      "masking.latent.gumbel_tau=1.0"
      "masking.latent.warmup_epochs=0"
      "masking.latent.nctx_min=1"
      "masking.latent.ntgt_min=4"
    )
    ;;
  3)
    RUN_LABEL="mi_hmargcosbs_floor"
    EXPERIMENT="stl10_vits_ps8_mi"
    EXP_NAME="chestmnist_128_vits_ps8_mi_hmargcosbs_floor_from_in1k96_e170_origlr_ema99925_nowarmup"
    LOSS_ARG="loss.base_kind=smooth_l1"
    RUN_ARGS=(
      "masking.latent.terms.H_cond.weight=[0.0,0.0]"
      "masking.latent.terms.neg_H_marg.weight=[1.0,1.0]"
      "masking.latent.terms.neg_surprise.weight=[0.0,0.0]"
      "+masking.latent.terms.neg_cos_surprise.weight=[0.5,0.5]"
      "masking.latent.terms.floor_penalty.weight=[5.0,5.0]"
      "masking.latent.hard_assignment=gumbel"
      "masking.latent.gumbel_tau=1.0"
      "masking.latent.warmup_epochs=0"
      "masking.latent.nctx_min=1"
      "masking.latent.ntgt_min=4"
    )
    ;;
  4)
    RUN_LABEL="mi_hmarg_sketchorthcosbs_floor"
    EXPERIMENT="stl10_vits_ps8_mi"
    EXP_NAME="chestmnist_128_vits_ps8_mi_hmarg_sketchorthcosbs_floor_from_in1k96_e170_origlr_ema99925_nowarmup"
    LOSS_ARG="loss.base_kind=smooth_l1"
    RUN_ARGS=(
      "masking.latent.terms.H_cond.weight=[0.0,0.0]"
      "masking.latent.terms.neg_H_marg.weight=[1.0,1.0]"
      "masking.latent.terms.neg_surprise.weight=[0.0,0.0]"
      "masking.latent.terms.neg_sketched_orthogonal_cos_surprise.weight=[0.5,0.5]"
      "masking.latent.terms.floor_penalty.weight=[5.0,5.0]"
      "masking.latent.hard_assignment=gumbel"
      "masking.latent.gumbel_tau=1.0"
      "masking.latent.warmup_epochs=0"
      "masking.latent.nctx_min=1"
      "masking.latent.ntgt_min=4"
    )
    ;;
  *)
    echo "Unsupported SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
    exit 1
    ;;
esac

COMMON_ARGS=(
  "experiment=${EXPERIMENT}"
  "data=chestmnist"
  "exp_name=${EXP_NAME}"
  "init_weights=${CHECKPOINT_PATH}"
  "${LOSS_ARG}"
  "data.root=${DATA_ROOT}"
  "data.pretrain_split=all"
  "data.train_split=train"
  "data.val_split=val"
  "data.download=false"
  "data.size=224"
  "data.as_rgb=true"
  "data.batch_size=1024"
  "model.image_size=128"
  "model.ema_momentum=[0.99925,0.99925]"
  "optim.lr=6e-4"
  "sched.warmup_epochs=0"
  "train.epochs=170"
  "train.save_every_epochs=10"
  "train.eval_every_epochs=0"
  "train.viz_enabled=false"
  "train.viz.out_dir=viz_output_${EXP_NAME}"
  "logger.mode=offline"
)

nvidia-smi -L

echo "Array task ${SLURM_ARRAY_TASK_ID}: variant=${RUN_LABEL} exp=${EXP_NAME}"

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python -m torch.distributed.run --standalone --nproc_per_node=4 \
    -m ijepa_lite.run \
      "${COMMON_ARGS[@]}" \
      "${RUN_ARGS[@]}"
