#!/bin/bash
#SBATCH --job-name=ijepa-prog-kl-det-dbg
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
# Debug: progressive KL with deterministic phase advance.
#
# Replaces role_alive + neg_H_marg with nway_progressive_kl (forward KL).
# Starts with 2 active target blocks; unlocks one additional block per phase.
# With num_tgt_blocks=4 and n_start_tgt=2: 2 transitions auto-placed at
# ~33% and ~67% of total steps.
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

export EXP_NAME="prog_kl_det_4blk_fwd_dbg"

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python -m torch.distributed.run --standalone --nproc_per_node=1 \
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
      train.save_every_epochs=30 \
      train.eval_every_epochs=10 \
      train.viz_enabled=true \
      +train.viz_every_epochs=1 \
      train.viz.n_images=30 \
      train.viz.out_dir="viz_output_${EXP_NAME}" \
      logger.mode=offline
