#!/bin/bash
#SBATCH --job-name=ijepa-ctp-hmarg-bs
#SBATCH --account=IscrC_PRIMAL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
#SBATCH --output=sbatch_logs/continued_pretrain/%x_%j.out
#SBATCH --error=sbatch_logs/continued_pretrain/%x_%j.err

# ---------------------------------------------------------------------------
# Continue STL10 pretraining for 200 fresh epochs from the vanilla multiblock
# epoch_00999 checkpoint. Terms match the HmargBS + floor reference; the rest
# stays aligned with the other continued-pretraining launchers.
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

mkdir -p sbatch_logs/continued_pretrain

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHECKPOINT_PATH="${REPO_ROOT}/outputs/2026-05-04/19-42-25_vanilla_baseline_multiblock_stl10/rank0/checkpoints/epoch_00999.pt"

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT_PATH}"
  exit 1
fi

nvidia-smi -L

EXP_NAME="stl10_mi_hmargbs_floor_cont_from_e999_200ep"

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python -m torch.distributed.run --standalone --nproc_per_node=4 \
    -m ijepa_lite.run \
      experiment=stl10_vits_ps8_mi \
      exp_name="$EXP_NAME" \
      init_weights="$CHECKPOINT_PATH" \
      loss.base_kind=smooth_l1 \
      data.root=$FAST/datasets/ \
      data.batch_size=2048 \
      optim.lr=3e-4 \
      masking.latent.terms.H_cond.weight="[0.0,0.0]" \
      masking.latent.terms.neg_H_marg.weight="[1.0,1.0]" \
      masking.latent.terms.neg_surprise.weight="[0.5,0.5]" \
      masking.latent.terms.floor_penalty.weight="[5.0,5.0]" \
      masking.latent.hard_assignment=gumbel \
      masking.latent.gumbel_tau=1.0 \
      masking.latent.warmup_epochs=0 \
      masking.latent.nctx_min=1 \
      masking.latent.ntgt_min=4 \
      train.epochs=200 \
      train.eval_every_epochs=1 \
      train.viz_enabled=true \
      +train.viz_every_epochs=1 \
      train.viz.out_dir="viz_output_${EXP_NAME}" \
      logger.mode=offline
