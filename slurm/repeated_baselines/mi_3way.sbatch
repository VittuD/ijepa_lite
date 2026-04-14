#!/bin/bash
#SBATCH --job-name=ijepa-mi-3way
#SBATCH --account=IscrC_PRIMAL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=06:30:00
#SBATCH --array=0-1
#SBATCH --output=sbatch_logs/repeated_baselines/%x_%A_%a.out
#SBATCH --error=sbatch_logs/repeated_baselines/%x_%A_%a.err

# ---------------------------------------------------------------------------
# MI 3-way baselines (evalv2 re-run)
#   0 — mi_HmargBS_floor_learned      (H_cond zeroed out)
#   1 — mi_HcondHmargBS_floor_learned (H_cond active)
#
# Shared: neg_H_marg=[1,1], neg_surprise=[0.5,0.5], floor_penalty=[5,5],
#         pos_embed=learned
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

mkdir -p sbatch_logs/repeated_baselines

EXP_NAMES=(
  mi_HmargBS_floor_learned
  mi_HcondHmargBS_floor_learned
)
H_COND_WEIGHTS=(
  '[0.0,0.0]'
  '[1.0,1.0]'
)

EXP_NAME=${EXP_NAMES[$SLURM_ARRAY_TASK_ID]}
H_COND_W=${H_COND_WEIGHTS[$SLURM_ARRAY_TASK_ID]}

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python -m torch.distributed.run --standalone --nproc_per_node=4 \
    -m ijepa_lite.run \
      experiment=stl10_vits_ps8_mi \
      exp_name="$EXP_NAME" \
      masking.latent.terms.H_cond.weight="$H_COND_W" \
      masking.latent.terms.neg_H_marg.weight='[1.0,1.0]' \
      masking.latent.terms.neg_surprise.weight='[0.5,0.5]' \
      masking.latent.terms.floor_penalty.weight='[5.0,5.0]' \
      model.pos_embed_kind=learned \
      data.root=$FAST/datasets/ \
      data.batch_size=2048 \
      train.viz.out_dir="viz_output_${EXP_NAME}" \
      logger.mode=offline
