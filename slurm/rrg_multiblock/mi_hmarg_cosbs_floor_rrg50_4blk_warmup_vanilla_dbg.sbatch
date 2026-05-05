#!/bin/bash
#SBATCH --job-name=ijepa-mi-hmarg-cosbs-floor-rrg50-wup500-4blk-warmup-dbg
#SBATCH --account=IscrC_TBoneAI
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=16:30:00
#SBATCH --output=sbatch_logs/rrg_multiblock/%x_%j.out
#SBATCH --error=sbatch_logs/rrg_multiblock/%x_%j.err

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

nvidia-smi -L

mkdir -p sbatch_logs/rrg_multiblock

EXP_NAME="mi_HmargCosBS_floor_learned_rrg50_wup500_4blk_warmup_vanilla_dbg"

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python -m torch.distributed.run --standalone --nproc_per_node=4 \
    -m ijepa_lite.run \
      experiment=stl10_vits_ps8_mi \
      exp_name="$EXP_NAME" \
      masking.latent.terms.H_cond.weight='[0,0]' \
      masking.latent.terms.neg_H_marg.weight='[1.0,1.0]' \
      masking.latent.terms.neg_surprise.weight='[0,0]' \
      +masking.latent.terms.neg_cos_surprise.weight='[0.5,0.5]' \
      masking.latent.terms.floor_penalty.weight='[5.0,5.0]' \
      masking.latent.hard_assignment=random_region_growth_multiblock \
      masking.latent.rrg_keep_percent=50 \
      masking.latent.rrg_num_target_blocks=4 \
      masking.latent.rrg_keep_k_per_block=10 \
      masking.latent.warmup_use_vanilla_multiblock=true \
      masking.latent.warmup_epochs=10 \
      distributed.find_unused_parameters=true \
      model.pos_embed_kind=learned \
      data.root=$FAST/datasets/ \
      data.batch_size=2048 \
      train.eval_every_epochs=5 \
      train.viz_enabled=true \
      +train.viz_every_epochs=5 \
      train.viz.out_dir="viz_output_${EXP_NAME}" \
      logger.mode=offline
