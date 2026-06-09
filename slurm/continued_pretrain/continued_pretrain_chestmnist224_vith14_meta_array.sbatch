#!/bin/bash
#SBATCH --job-name=ijepa-ctp-chest224-vith14
#SBATCH --account=IscrC_PRIMAL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=02:30:00
#SBATCH --array=0-5
#SBATCH --output=sbatch_logs/continued_pretrain/chestmnist224_vith14_meta_e4_b16_ema99925_nowarmup/%x_%A_%a.out
#SBATCH --error=sbatch_logs/continued_pretrain/chestmnist224_vith14_meta_e4_b16_ema99925_nowarmup/%x_%A_%a.err

# ---------------------------------------------------------------------------
# ChestMNIST continued-pretraining array starting from the converted native
# ijepa_lite version of Meta's original ImageNet-1K ViT-H/14 I-JEPA checkpoint.
#
# Geometry stays native to the checkpoint:
#   image_size=224, patch_size=14, grid=16x16=256 patches,
#   embed_dim=1280, num_heads=16, depth=32, use_cls_token=false.
#
# Batch/epoch budget:
#   ViT-S/8 Chest array: batch=1024 per rank * 4 ranks, 247 epochs
#     -> ceil(78468 / 4096) * 247 ~= 4940 optimizer steps.
#   This ViT-H/14 array: batch=16 per rank * 4 ranks, 4 epochs
#     -> ceil(78468 / 64) * 4 ~= 4908 optimizer steps.
#
# Matrix:
#   0  vanilla_multiblock
#   1  mi_affinity_novelty
#   2  mi_hmargbs_floor
#   3  mi_hmargcosbs_floor
#   4  mi_hmarg_sketchorthcosbs_floor
#   5  semantic_pca
#
# Submit:
#   sbatch slurm/continued_pretrain/continued_pretrain_chestmnist224_vith14_meta_array.sbatch
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

LOG_DIR="sbatch_logs/continued_pretrain/chestmnist224_vith14_meta_e4_b16_ema99925_nowarmup"
mkdir -p "${LOG_DIR}"

CHECKPOINT_PATH="${WORK}/vitturini/original_jepa_weights/IN1K-vit.h.14-300e.ijepa_lite.pt"

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
    EXP_NAME="chestmnist_224_vith14_vanilla_multiblock_from_meta_e4_b16_origlr_ema99925_nowarmup"
    LOSS_ARG="loss.kind=smooth_l1"
    RUN_ARGS=()
    ;;
  1)
    RUN_LABEL="mi_affinity_novelty"
    EXPERIMENT="stl10_vits_ps8_mi"
    EXP_NAME="chestmnist_224_vith14_mi_affinity_novelty_from_meta_e4_b16_origlr_ema99925_nowarmup"
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
    EXP_NAME="chestmnist_224_vith14_mi_hmargbs_floor_from_meta_e4_b16_origlr_ema99925_nowarmup"
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
    EXP_NAME="chestmnist_224_vith14_mi_hmargcosbs_floor_from_meta_e4_b16_origlr_ema99925_nowarmup"
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
    EXP_NAME="chestmnist_224_vith14_mi_hmarg_sketchorthcosbs_floor_from_meta_e4_b16_origlr_ema99925_nowarmup"
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
  5)
    RUN_LABEL="semantic_pca"
    EXPERIMENT="stl10_vits_ps8_multiblock"
    EXP_NAME="chestmnist_224_vith14_semantic_pca_from_meta_e4_b16_origlr_ema99925_nowarmup"
    LOSS_ARG="loss.kind=smooth_l1"
    RUN_ARGS=(
      "+masking/compressor=full"
      "+masking/latent=semantic_pca"
      "masking.latent.pca_dim=1"
      "masking.latent.power_iterations=4"
      "masking.latent.split_mode=band"
      "masking.latent.context_mode=complement"
      "masking.latent.normalize_tokens=true"
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
  "data.pretrain_split=train"
  "data.train_split=train"
  "data.val_split=val"
  "data.download=false"
  "data.size=224"
  "data.as_rgb=true"
  "data.batch_size=16"
  "model.arch=vit_huge"
  "model.image_size=224"
  "model.patch_size=14"
  "model.embed_dim=1280"
  "model.num_heads=16"
  "model.depth=32"
  "model.use_cls_token=false"
  "model.ema_momentum=[0.99925,0.99925]"
  "predictor.predictor_dim=384"
  "predictor.depth=12"
  "predictor.num_heads=12"
  "predictor.mlp_ratio=4.0"
  "optim.lr=6e-4"
  "sched.warmup_epochs=0"
  "train.epochs=4"
  "train.save_every_epochs=1"
  "train.eval_every_epochs=1"
  "train.inline_eval.dataset=chestmnist"
  "train.inline_eval.data_root=${DATA_ROOT}"
  "train.inline_eval.train_split=val"
  "train.inline_eval.val_split=test"
  "train.inline_eval.num_classes=14"
  "+train.inline_eval.task_type=multilabel"
  "+train.inline_eval.size=224"
  "+train.inline_eval.as_rgb=true"
  "+train.inline_eval.download=false"
  "train.inline_eval.batch_size=64"
  "train.inline_eval.num_workers=4"
  "train.inline_eval.eval_at_start=true"
  "train.inline_eval.head_type=mlp"
  "train.inline_eval.head_hidden_dim=1280"
  "train.inline_eval.head_num_layers=1"
  "train.inline_eval.probe_epochs=10"
  "train.inline_eval.probe_lr=0.1"
  "train.inline_eval.probe_weight_decay=0.0"
  "train.inline_eval.probe_sched=step"
  "train.inline_eval.probe_step_size=5"
  "train.inline_eval.probe_gamma=0.1"
  "train.viz_enabled=true"
  "+train.viz_every_epochs=1"
  "train.viz.eval_at_start=true"
  "train.viz.dataset=chestmnist"
  "train.viz.data_root=${DATA_ROOT}"
  "+train.viz.split=test"
  "+train.viz.size=224"
  "train.viz.out_dir=viz_output_${EXP_NAME}"
  "logger.mode=offline"
)

nvidia-smi -L

echo "Array task ${SLURM_ARRAY_TASK_ID}: variant=${RUN_LABEL} exp=${EXP_NAME}"
echo "Per-rank batch size: 16; global batch size: 64; epochs: 4"

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python -m torch.distributed.run --standalone --nproc_per_node=4 \
    -m ijepa_lite.run \
      "${COMMON_ARGS[@]}" \
      "${RUN_ARGS[@]}"
