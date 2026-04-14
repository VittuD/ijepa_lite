#!/bin/bash
#SBATCH --job-name=ijepa-prog-kl-det-nblk
#SBATCH --account=IscrC_PRIMAL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=06:30:00
#SBATCH --array=0-4
#SBATCH --output=sbatch_logs/progressive_kl/%x_%A_%a.out
#SBATCH --error=sbatch_logs/progressive_kl/%x_%A_%a.err

# ---------------------------------------------------------------------------
# Sweep: progressive KL deterministic, n_start_tgt=2, num_tgt_blocks in {4,5,6,7,8}.
# Global batch size = 8192 (2048 per GPU × 4 GPUs).
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

NBLK_VALUES=(4 5 6 7 8)
NBLK=${NBLK_VALUES[$SLURM_ARRAY_TASK_ID]}

export EXP_NAME="prog_kl_det_${NBLK}blk_fwd"

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
      masking.latent.num_tgt_blocks=$NBLK \
      masking.latent.ntgt_min_per_block=1 \
      +masking.latent.n_start_tgt=2 \
      +masking.latent.phase_mode=deterministic \
      data.root=$FAST/datasets/ \
      data.batch_size=2048 \
      train.epochs=1000 \
      train.viz_enabled=true \
      +train.viz_every_epochs=1 \
      train.viz.n_images=30 \
      train.viz.out_dir="viz_output_${EXP_NAME}" \
      logger.mode=offline
