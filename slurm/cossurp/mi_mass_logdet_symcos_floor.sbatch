#!/bin/bash
#SBATCH --job-name=ijepa-mi-mass-logdet-symcos-floor
#SBATCH --account=IscrC_PRIMAL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=06:30:00
#SBATCH --output=sbatch_logs/cossurp/%x_%j.out
#SBATCH --error=sbatch_logs/cossurp/%x_%j.err

# ---------------------------------------------------------------------------
# MI 3-way: mass-scaled logdet role diversity + symmetric cosine surprise
# + floor, with Gumbel hard assignment.
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

mkdir -p sbatch_logs/cossurp

EXP_NAME="mi_MassLogDetSymCos_floor_gumbel"

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python -m torch.distributed.run --standalone --nproc_per_node=4 \
    -m ijepa_lite.run \
      experiment=stl10_vits_ps8_mi \
      exp_name="$EXP_NAME" \
      masking.latent.hard_assignment=gumbel \
      masking.latent.gumbel_tau=0.5 \
      masking.latent.max_total_hard=128 \
      masking.latent.terms.H_cond.weight='[0,0]' \
      masking.latent.terms.neg_H_marg.weight='[0,0]' \
      masking.latent.terms.neg_surprise.weight='[0,0]' \
      masking.latent.terms.neg_symmetric_cos_surprise.weight='[0.25,0.25]' \
      masking.latent.terms.neg_symmetric_cos_surprise.role_indices='[0,1]' \
      masking.latent.terms.neg_logdet_diversity.weight='[0,0]' \
      masking.latent.terms.neg_support_logdet_diversity.weight='[0,0]' \
      masking.latent.terms.neg_mass_logdet_diversity.weight='[1.0,1.0]' \
      masking.latent.terms.neg_mass_logdet_diversity.role_indices='[0,1]' \
      masking.latent.terms.neg_mass_logdet_diversity.sketch_dim=32 \
      masking.latent.terms.neg_mass_logdet_diversity.alpha=1.0 \
      masking.latent.terms.floor_penalty.weight='[5.0,5.0]' \
      model.pos_embed_kind=learned \
      data.root=$FAST/datasets/ \
      data.batch_size=2048 \
      train.eval_every_epochs=1 \
      train.viz_enabled=true \
      +train.viz_every_epochs=1 \
      train.viz.out_dir="viz_output_${EXP_NAME}" \
      logger.mode=offline
