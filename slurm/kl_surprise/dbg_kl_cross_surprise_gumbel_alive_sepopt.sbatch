#!/bin/bash
#SBATCH --job-name=ijepa-kl-xs-gumbel-alive-sepopt-dbg
#SBATCH --account=IscrC_PRIMAL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=00:15:00
#SBATCH --output=sbatch_logs/kl_surprise/%x_%j.out
#SBATCH --error=sbatch_logs/kl_surprise/%x_%j.err

# ---------------------------------------------------------------------------
# Debug smoke-test: KL-to-target marginal (replaces entropy) + cross-surprise
# + Gumbel sampling + role_alive + separate masker optimizer.
#
# k follows warmup+cosine schedule: k_min=0.2 -> k_max=0.5 -> k_min=0.2
# At k=1/5=0.2 (uniform for 3blk+ctx+ign=5 roles), this equals entropy.
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

mkdir -p sbatch_logs/kl_surprise

export EXP_NAME="kl_cross_surprise_gumbel_alive_sepopt_dbg"

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python -m torch.distributed.run --standalone --nproc_per_node=1 \
    -m ijepa_lite.run \
      experiment=stl10_vits_ps8_mi_nway \
      exp_name="$EXP_NAME" \
      masking.latent.hard_assignment=gumbel \
      masking.latent.gumbel_tau=0.5 \
      ~masking.latent.terms.nway_neg_H_marg \
      +masking.latent.terms.nway_kl_marg.weight='[1.0,1.0]' \
      masking.latent.terms.nway_cross_surprise.weight='[0.1,0.1]' \
      ~masking.latent.terms.nway_floor_penalty \
      +masking.latent.terms.role_alive.weight='[5.0,5.0]' \
      +masking.latent.terms.role_alive.p_min=0.05 \
      masking.latent.num_tgt_blocks=3 \
      masking.latent.k_start=0.2 \
      masking.latent.k_min=0.2 \
      masking.latent.k_max=0.5 \
      masking.latent.k_warmup_epochs=2 \
      optim.masker_separate_optimizer=true \
      data.root=$FAST/datasets/ \
      data.batch_size=512 \
      train.epochs=5 \
      train.save_every_epochs=5 \
      train.viz_enabled=true \
      train.viz.out_dir="viz_output_${EXP_NAME}" \
      logger.mode=offline
