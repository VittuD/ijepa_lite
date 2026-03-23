#!/bin/bash
#SBATCH --job-name=ijepa-mi-nway-xs-gumbel-05tau-4blk-ctxloss
#SBATCH --account=IscrC_PRIMAL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00
#SBATCH --output=sbatch_logs/mi_gumbel/%x_%j.out
#SBATCH --error=sbatch_logs/mi_gumbel/%x_%j.err

# ---------------------------------------------------------------------------
# N-way cross-target surprise with Gumbel sampling + V-JEPA 2.1 context loss.
#
# Same as mi_nway_cross_surprise_gumbel but adds distance-weighted
# reconstruction loss on visible (context) tokens.
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

nvidia-smi -L

export EXP_NAME="mi_nway_cross_surprise_gumbel_05tau_4blk_ctxloss"

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python -m torch.distributed.run --standalone --nproc_per_node=4 \
    -m ijepa_lite.run \
      experiment=stl10_vits_ps8_mi_nway \
      exp_name="$EXP_NAME" \
      masking.latent.hard_assignment=gumbel \
      masking.latent.gumbel_tau=0.5 \
      masking.latent.terms.nway_neg_H_marg.weight='[1.0,1.0]' \
      masking.latent.terms.nway_cross_surprise.weight='[0.1,0.1]' \
      masking.latent.terms.nway_floor_penalty.weight='[5.0,5.0]' \
      masking.latent.terms.nway_floor_penalty.h_floor=0.16 \
      masking.latent.num_tgt_blocks=4 \
      +ctx_loss.weight=1.0 \
      +ctx_loss.gamma=0.7 \
      +ctx_loss.warmup_start=0 \
      +ctx_loss.warmup_end=0 \
      data.root=$FAST/datasets/ \
      data.batch_size=2048 \
      train.viz_enabled=true \
      train.viz.out_dir="viz_output_${EXP_NAME}" \
      logger.mode=offline
