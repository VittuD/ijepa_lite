#!/bin/bash
#SBATCH --job-name=ijepa-ctp-chest224-vith14-4n
#SBATCH --account=IscrC_PRIMAL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
#SBATCH --output=sbatch_logs/continued_pretrain/chestmnist224_vith14_meta_vanilla_4n_30ep/%x_%j.out
#SBATCH --error=sbatch_logs/continued_pretrain/chestmnist224_vith14_meta_vanilla_4n_30ep/%x_%j.err

# ---------------------------------------------------------------------------
# Vanilla multiblock ChestMNIST continued pretraining from Meta's original
# ImageNet-1K ViT-H/14 I-JEPA checkpoint converted to ijepa_lite format.
#
# This is the large-batch version of the ViT-H Chest experiment:
#   4 nodes * 4 GPUs/node * 128 images/rank = global batch 2048,
# matching Meta I-JEPA's original ViT-H/14 global batch size.
#
# ChestMNIST train has ~78,468 images:
#   ceil(78468 / 2048) ~= 39 optimizer steps/epoch
#   30 epochs ~= 1,170 optimizer steps
#
# Submit:
#   sbatch slurm/continued_pretrain/continued_pretrain_chestmnist224_vith14_meta_vanilla_4n_30ep.sbatch
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

LOG_DIR="sbatch_logs/continued_pretrain/chestmnist224_vith14_meta_vanilla_4n_30ep"
mkdir -p "${LOG_DIR}"

CHECKPOINT_PATH="${WORK}/vitturini/original_jepa_weights/IN1K-vit.h.14-300e.ijepa_lite.pt"
export CHECKPOINT_PATH

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT_PATH}"
  exit 1
fi

echo "python:   $(which python)"
echo "torchrun: $(which torchrun)"
python -c "import torch; print('torch:', torch.__version__)"
python -c "import ijepa_lite; print('ijepa_lite:', ijepa_lite.__file__)"

echo "DATA_ROOT=${DATA_ROOT}"
ls -ld "$DATA_ROOT"
ls -lh "$DATA_ROOT"/chestmnist_224.npz

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT="${MASTER_PORT:-29500}"
export MASTER_ADDR MASTER_PORT

echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "SLURM_NNODES=${SLURM_NNODES}"

nvidia-smi -L

srun bash -c '
  echo "node_rank=$SLURM_NODEID CHECKPOINT_PATH=$CHECKPOINT_PATH"
  torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=4 \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    -m ijepa_lite.run \
      experiment=stl10_vits_ps8_multiblock \
      data=chestmnist \
      exp_name=chestmnist_224_vith14_vanilla_multiblock_from_meta_4n_b2048_30ep_lr2e4_wd004_ema99925 \
      init_weights=$CHECKPOINT_PATH \
      loss.kind=smooth_l1 \
      data.root=$DATA_ROOT \
      data.pretrain_split=train \
      data.train_split=train \
      data.val_split=val \
      data.download=false \
      data.size=224 \
      data.as_rgb=true \
      data.batch_size=128 \
      model.arch=vit_huge \
      model.image_size=224 \
      model.patch_size=14 \
      model.embed_dim=1280 \
      model.num_heads=16 \
      model.depth=32 \
      model.use_cls_token=false \
      model.ema_momentum=[0.99925,0.99925] \
      predictor.predictor_dim=384 \
      predictor.depth=12 \
      predictor.num_heads=12 \
      predictor.mlp_ratio=4.0 \
      optim.lr=2e-4 \
      optim.weight_decay=0.04 \
      optim.final_weight_decay=0.4 \
      sched.warmup_epochs=3 \
      sched.min_lr=1e-6 \
      train.epochs=30 \
      train.save_every_epochs=5 \
      train.eval_every_epochs=5 \
      train.inline_eval.dataset=chestmnist \
      train.inline_eval.data_root=$DATA_ROOT \
      train.inline_eval.train_split=val \
      train.inline_eval.val_split=test \
      train.inline_eval.num_classes=14 \
      +train.inline_eval.task_type=multilabel \
      +train.inline_eval.size=224 \
      +train.inline_eval.as_rgb=true \
      +train.inline_eval.download=false \
      train.inline_eval.batch_size=64 \
      train.inline_eval.num_workers=4 \
      train.inline_eval.eval_at_start=true \
      train.inline_eval.head_type=mlp \
      train.inline_eval.head_hidden_dim=1280 \
      train.inline_eval.head_num_layers=1 \
      train.inline_eval.probe_epochs=10 \
      train.inline_eval.probe_lr=0.1 \
      train.inline_eval.probe_weight_decay=0.0 \
      train.inline_eval.probe_sched=step \
      train.inline_eval.probe_step_size=5 \
      train.inline_eval.probe_gamma=0.1 \
      train.viz_enabled=true \
      +train.viz_every_epochs=5 \
      train.viz.eval_at_start=true \
      train.viz.dataset=chestmnist \
      train.viz.data_root=$DATA_ROOT \
      +train.viz.split=test \
      +train.viz.size=224 \
      train.viz.out_dir=viz_output_chestmnist_224_vith14_vanilla_multiblock_from_meta_4n_b2048_30ep \
      logger.mode=offline
'
