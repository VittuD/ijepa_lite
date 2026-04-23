#!/bin/bash
#SBATCH --job-name=ijepa-mi-signed-support-logdet-symcos-prompt-floor
#SBATCH --account=IscrC_PRIMAL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=06:30:00
#SBATCH --output=sbatch_logs/prompt_contrast/%x_%j.out
#SBATCH --error=sbatch_logs/prompt_contrast/%x_%j.err

# ---------------------------------------------------------------------------
# MI 3-way: signed support-logdet + symmetric cosine surprise + prompt
# contrast + floor, with Gumbel hard assignment.
#
# Prompt contrast:
#   - sample ctx/tgt prompts uniformly with c != t
#   - orient ctx/tgt centroid separation along the prompt axis
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

mkdir -p sbatch_logs/prompt_contrast

EXP_NAME="mi_SignedSupportLogDetSymCosPrompt_floor_gumbel"

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
      masking.latent.terms.neg_prompt_contrast.weight='[0.25,0.25]' \
      masking.latent.terms.neg_prompt_contrast.sample_mode=uniform_distinct \
      masking.latent.terms.neg_prompt_contrast.sample_temperature=0.5 \
      masking.latent.terms.neg_prompt_contrast.exclude_prompts=true \
      masking.latent.terms.neg_prompt_contrast.degenerate_cos_threshold=0.9 \
      masking.latent.terms.neg_logdet_diversity.weight='[0,0]' \
      masking.latent.terms.neg_support_logdet_diversity.weight='[0,0]' \
      masking.latent.terms.neg_mass_logdet_diversity.weight='[0,0]' \
      masking.latent.terms.neg_signed_support_logdet.weight='[1.0,1.0]' \
      masking.latent.terms.neg_signed_support_logdet.diverse_role_indices='[0,1]' \
      masking.latent.terms.neg_signed_support_logdet.compact_role_indices='[2]' \
      masking.latent.terms.neg_signed_support_logdet.compact_weight=1.0 \
      masking.latent.terms.neg_signed_support_logdet.sketch_dim=32 \
      masking.latent.terms.neg_signed_support_logdet.alpha=1.0 \
      masking.latent.terms.neg_signed_support_logdet.support_power=0.5 \
      masking.latent.terms.floor_penalty.weight='[5.0,5.0]' \
      model.pos_embed_kind=learned \
      data.root=$FAST/datasets/ \
      data.batch_size=2048 \
      train.eval_every_epochs=1 \
      train.viz_enabled=true \
      +train.viz_every_epochs=1 \
      train.viz.out_dir="viz_output_${EXP_NAME}" \
      logger.mode=offline
