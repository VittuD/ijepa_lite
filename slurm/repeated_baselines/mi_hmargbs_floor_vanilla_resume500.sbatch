#!/bin/bash
#SBATCH --job-name=ijepa-mi-hmargbs-vanilla-resume
#SBATCH --account=IscrC_PRIMAL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=06:30:00
#SBATCH --output=sbatch_logs/repeated_baselines/%x_%j.out
#SBATCH --error=sbatch_logs/repeated_baselines/%x_%j.err

# ---------------------------------------------------------------------------
# Initialize the vanilla random-masker multiblock baseline from the
# mi_HmargBS_floor_learned epoch_00499 checkpoint, then train for 500 epochs.
#
# This uses resume_weights_only=true + resume_strict=false so only compatible
# encoder / predictor / target-encoder weights are loaded.  MI-only masker and
# compressor checkpoint keys are ignored, and optimizer/scheduler state starts
# fresh for this vanilla continuation.
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
CKPT="${REPO_ROOT}/outputs/2026-04-14/15-00-51_mi_HmargBS_floor_learned/rank0/checkpoints/epoch_00499.pt"

if [[ ! -f "$CKPT" ]]; then
  echo "Checkpoint not found: $CKPT" >&2
  echo "Submit this job from the ijepa_lite repo root, or edit CKPT to an absolute path." >&2
  exit 1
fi

EXP_NAME="mi_HmargBS_floor_learned_vanilla_resume500"

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python -m torch.distributed.run --standalone --nproc_per_node=4 \
    -m ijepa_lite.run \
      experiment=stl10_vits_ps8_multiblock \
      exp_name="$EXP_NAME" \
      resume="$CKPT" \
      resume_weights_only=true \
      resume_strict=false \
      model.pos_embed_kind=learned \
      data.root=$FAST/datasets/ \
      data.batch_size=2048 \
      train.epochs=500 \
      train.viz_enabled=true \
      train.viz.out_dir="viz_output_${EXP_NAME}" \
      logger.mode=offline
