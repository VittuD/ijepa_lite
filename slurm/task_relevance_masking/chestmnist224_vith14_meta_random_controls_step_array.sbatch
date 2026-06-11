#!/bin/bash
#SBATCH --job-name=ijepa-claim-chest-rand
#SBATCH --account=IscrC_PRIMAL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --array=0-1
#SBATCH --output=sbatch_logs/task_relevance_masking/chestmnist224_vith14_meta_random_controls_step_array/%x_%A_%a.out
#SBATCH --error=sbatch_logs/task_relevance_masking/chestmnist224_vith14_meta_random_controls_step_array/%x_%A_%a.err

set -euo pipefail

module purge
module load profile/deeplrn
module load python/3.11.6--gcc--8.5.0

source $WORK/vitturini/ijepa_venv/bin/activate

export MED_ROOT="${CHESTMNIST_ROOT:-/leonardo_scratch/fast/IscrC_TBoneAI/datasets}"
export WANDB_MODE=offline

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export CUDA_DEVICE_MAX_CONNECTIONS=1

LOG_DIR="${SLURM_SUBMIT_DIR:-$PWD}/sbatch_logs/task_relevance_masking/chestmnist224_vith14_meta_random_controls_step_array"
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

echo "MED_ROOT=${MED_ROOT}"
ls -ld "${MED_ROOT}"
ls -lh "${MED_ROOT}/chestmnist_224.npz"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT="${MASTER_PORT:-29500}"
export MASTER_ADDR MASTER_PORT

echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "SLURM_NNODES=${SLURM_NNODES}"
nvidia-smi -L

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    RUN_LABEL="random_split_50"
    EXP_NAME="claim_chestmnist_224_vith14_random_split_50_from_meta_step1170_b2048"
    RUN_ARGS=(
      "+masking/compressor=full"
      "+masking/latent=random_split_50"
    )
    ;;
  1)
    RUN_LABEL="upper_half_random_split_50"
    EXP_NAME="claim_chestmnist_224_vith14_upper_half_random_split_50_from_meta_step1170_b2048"
    RUN_ARGS=(
      "+masking/compressor=full"
      "+masking/latent=upper_half_random_split_50"
    )
    ;;
  *)
    echo "Unsupported SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
    exit 1
    ;;
esac

COMMON_ARGS=(
  "experiment=stl10_vits_ps8_multiblock"
  "data=chestmnist"
  "exp_name=${EXP_NAME}"
  "init_weights=${CHECKPOINT_PATH}"
  "loss.kind=smooth_l1"
  "data.root=${MED_ROOT}"
  "data.pretrain_split=train"
  "data.train_split=train"
  "data.val_split=val"
  "data.download=false"
  "data.size=224"
  "data.as_rgb=true"
  "data.batch_size=128"
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
  "optim.lr=2e-4"
  "optim.weight_decay=0.04"
  "optim.final_weight_decay=0.4"
  "sched.interval=step"
  "sched.warmup_steps=117"
  "sched.total_steps=1170"
  "sched.warmup_epochs=0"
  "sched.min_lr=1e-6"
  "train.epochs=30"
  "train.max_steps=1170"
  "train.log_every=50"
  "distributed.timeout_minutes=60"
  "train.save_every_epochs=0"
  "train.save_every_steps=195"
  "train.eval_every_epochs=0"
  "train.eval_every_steps=195"
  "train.inline_eval.dataset=chestmnist"
  "train.inline_eval.data_root=${MED_ROOT}"
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
  "+train.viz_every_epochs=0"
  "train.viz_every_steps=195"
  "train.viz.eval_at_start=true"
  "train.viz.dataset=chestmnist"
  "train.viz.data_root=${MED_ROOT}"
  "+train.viz.split=test"
  "+train.viz.size=224"
  "train.viz.out_dir=viz_output_${EXP_NAME}"
  "logger.mode=offline"
)

ARGS_FILE="${LOG_DIR}/ijepa_args_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.nul"
printf '%s\0' "${COMMON_ARGS[@]}" "${RUN_ARGS[@]}" > "${ARGS_FILE}"
export ARGS_FILE

echo "Array task ${SLURM_ARRAY_TASK_ID}: variant=${RUN_LABEL} exp=${EXP_NAME}"
echo "Global batch size: 2048; max_steps: 1170; eval/viz/save every 195 steps"

srun bash -c '
  mapfile -d "" ARGS < "$ARGS_FILE"
  echo "node_rank=$SLURM_NODEID CHECKPOINT_PATH=$CHECKPOINT_PATH"
  torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=4 \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    -m ijepa_lite.run \
      "${ARGS[@]}"
'
