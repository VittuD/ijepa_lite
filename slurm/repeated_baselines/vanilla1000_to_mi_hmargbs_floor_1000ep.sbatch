#!/bin/bash
#SBATCH --job-name=ijepa-vanilla1000-to-mi-hmargbs
#SBATCH --account=IscrC_PRIMAL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --output=sbatch_logs/repeated_baselines/%x_%j.out
#SBATCH --error=sbatch_logs/repeated_baselines/%x_%j.err

# ---------------------------------------------------------------------------
# Initialize mi_HmargBS_floor_learned from the vanilla random-masker baseline
# at epoch_00999, then train 1000 fresh epochs with the MI HmargBS + floor
# objective.
#
# This uses resume_weights_only=true + resume_strict=false so compatible
# encoder / predictor / target-encoder weights are loaded from the vanilla
# checkpoint, while MI-only masker/compressor and optimizer state start fresh.
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

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
CKPT="${REPO_ROOT}/outputs/2026-04-14/15-00-53_vanilla_baseline_multiblock_stl10/rank0/checkpoints/epoch_00999.pt"

if [[ ! -f "$CKPT" ]]; then
  echo "Checkpoint not found: $CKPT" >&2
  echo "Submit this job from the ijepa_lite repo root, or edit CKPT to an absolute path." >&2
  exit 1
fi

EXP_NAME="vanilla1000_to_mi_HmargBS_floor_1000ep"

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python -m torch.distributed.run --standalone --nproc_per_node=4 \
    -m ijepa_lite.run \
      experiment=stl10_vits_ps8_mi \
      exp_name="$EXP_NAME" \
      resume="$CKPT" \
      resume_weights_only=true \
      resume_strict=false \
      masking.latent.terms.H_cond.weight='[0.0,0.0]' \
      masking.latent.terms.neg_H_marg.weight='[1.0,1.0]' \
      masking.latent.terms.neg_surprise.weight='[0.5,0.5]' \
      masking.latent.terms.floor_penalty.weight='[5.0,5.0]' \
      model.pos_embed_kind=learned \
      data.root=$FAST/datasets/ \
      data.batch_size=2048 \
      train.epochs=1000 \
      train.viz_enabled=true \
      train.viz.out_dir="viz_output_${EXP_NAME}" \
      logger.mode=offline
