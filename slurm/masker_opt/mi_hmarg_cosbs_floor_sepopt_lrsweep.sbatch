#!/bin/bash
#SBATCH --job-name=ijepa-mi-hmarg-cosbs-floor-sepopt
#SBATCH --account=IscrC_PRIMAL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=06:30:00
#SBATCH --array=0-2
#SBATCH --output=sbatch_logs/masker_opt/%x_%A_%a.out
#SBATCH --error=sbatch_logs/masker_opt/%x_%A_%a.err

# ---------------------------------------------------------------------------
# MI 3-way: Hmarg + cosine Bayesian-surprise proxy + floor, learned positions.
# Same setup as slurm/cossurp/mi_hmarg_cosbs_floor.sbatch, but with:
#   - separate optimizer for latent_masker + token_compressor
#   - array sweep over masker LR scale in {1.0, 0.1, 0.01}
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

mkdir -p sbatch_logs/masker_opt

MASKER_LR_SCALES=(1.0 0.1 0.01)
MASKER_LR_SCALE="${MASKER_LR_SCALES[$SLURM_ARRAY_TASK_ID]}"
MASKER_LR_TAG=$(printf "%s" "$MASKER_LR_SCALE" | tr '.' 'p')

EXP_NAME="mi_HmargCosBS_floor_learned_sepopt_mlr${MASKER_LR_TAG}"

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
      model.pos_embed_kind=learned \
      optim.masker.separate_optimizer=true \
      optim.masker.lr_scale="$MASKER_LR_SCALE" \
      data.root=$FAST/datasets/ \
      data.batch_size=2048 \
      train.eval_every_epochs=1 \
      train.viz_enabled=true \
      +train.viz_every_epochs=1 \
      train.viz.out_dir="viz_output_${EXP_NAME}" \
      logger.mode=offline
