#!/bin/bash
#SBATCH --job-name=ijepa-in1k-mi-hmarg-sketchorthcosbs
#SBATCH --account=IscrC_TBoneAI
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=sbatch_logs/in1k/%x_%j.out
#SBATCH --error=sbatch_logs/in1k/%x_%j.err

# ---------------------------------------------------------------------------
# IN1K MI Hmarg + sketched orthogonal CosBS + floor pretraining,
# 224x224 images / patch_size=16, 300 epochs. Uses the CINECA imagenet
# module, which exports IMAGENET2012_HOME. Eval and viz are enabled every
# epoch.
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

mkdir -p sbatch_logs/in1k

echo "DATA_ROOT=$DATA_ROOT"
ls -ld "$DATA_ROOT" "$DATA_ROOT/train" "$DATA_ROOT/val"
nvidia-smi -L

EXP_NAME="in1k_224_vits_ps16_mi_HmargSketchOrthCosBS_floor_300ep"

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python -m torch.distributed.run --standalone --nproc_per_node=4 \
    -m ijepa_lite.run \
      experiment=stl10_vits_ps8_mi \
      exp_name="$EXP_NAME" \
      model.image_size=224 \
      model.patch_size=16 \
      data.name=imagenet \
      data.root="$DATA_ROOT" \
      data.pretrain_split=train \
      data.train_split=train \
      data.val_split=val \
      data.download=false \
      data.batch_size=2048 \
      masking.latent.terms.H_cond.weight="[0.0,0.0]" \
      masking.latent.terms.neg_H_marg.weight="[1.0,1.0]" \
      masking.latent.terms.neg_surprise.weight="[0.0,0.0]" \
      masking.latent.terms.neg_sketched_orthogonal_cos_surprise.weight="[0.5,0.5]" \
      masking.latent.terms.floor_penalty.weight="[5.0,5.0]" \
      model.pos_embed_kind=learned \
      train.epochs=300 \
      train.eval_every_epochs=1 \
      train.viz_enabled=true \
      +train.viz_every_epochs=1 \
      train.viz.out_dir="viz_output_${EXP_NAME}" \
      logger.mode=offline
