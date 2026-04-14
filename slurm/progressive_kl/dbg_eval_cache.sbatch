#!/bin/bash
#SBATCH --job-name=ijepa-eval-cache-dbg
#SBATCH --account=IscrC_PRIMAL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --output=sbatch_logs/progressive_kl/%x_%j.out
#SBATCH --error=sbatch_logs/progressive_kl/%x_%j.err

# ---------------------------------------------------------------------------
# Debug: validates the validation-step efficiency optimizations:
#   1. Feature caching in InlineEvalCallback (encoder runs once per eval, not
#      once per probe epoch × batch).
#   2. ema_param_metrics kernel fusion (flatten+cat → 6 ops vs ~300).
#   3. nway_soft mass loop vectorization (1 kernel vs M).
#   4. token_metrics std/var fusion (var.sqrt() instead of separate std).
#
# Runs eval_every_epochs=2 on a 30-epoch job so inline eval fires 15 times.
# Check logs for:
#   - inline_eval/val_acc1 appearing every 2 epochs (no crash = caching works)
#   - ema/param_l2, train/pred_std, mask/nway_tgt_*_mass all logged correctly
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

mkdir -p sbatch_logs/progressive_kl

export EXP_NAME="eval_cache_dbg"

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python -m torch.distributed.run --standalone --nproc_per_node=4 \
    -m ijepa_lite.run \
      experiment=stl10_vits_ps8_mi_nway \
      exp_name="$EXP_NAME" \
      masking.latent.hard_assignment=gumbel \
      masking.latent.gumbel_tau=0.5 \
      ~masking.latent.terms.nway_neg_H_marg \
      ~masking.latent.terms.nway_floor_penalty \
      masking.latent.terms.nway_cross_surprise.weight='[0.1,0.1]' \
      +masking.latent.terms.nway_progressive_kl.weight='[1.0,1.0]' \
      +masking.latent.terms.nway_progressive_kl.direction=forward \
      masking.latent.num_tgt_blocks=4 \
      masking.latent.ntgt_min_per_block=1 \
      +masking.latent.n_start_tgt=2 \
      +masking.latent.phase_mode=deterministic \
      data.root=$FAST/datasets/ \
      data.batch_size=2048 \
      train.epochs=30 \
      train.log_every=10 \
      train.save_every_epochs=30 \
      train.eval_every_epochs=2 \
      train.viz_enabled=false \
      logger.mode=offline
