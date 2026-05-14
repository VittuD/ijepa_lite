#!/bin/bash
#SBATCH --job-name=ijepa-ctp-array
#SBATCH --account=IscrC_PRIMAL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
#SBATCH --array=0-4
#SBATCH --output=sbatch_logs/continued_pretrain/%x_%A_%a.out
#SBATCH --error=sbatch_logs/continued_pretrain/%x_%A_%a.err

# ---------------------------------------------------------------------------
# Submit all five continued-pretraining variants from one command.
#
# Matrix:
#   0  vanilla_multiblock
#   1  mi_affinity_novelty
#   2  mi_hmargbs_floor
#   3  mi_hmargcosbs_floor
#   4  mi_hmarg_sketchorthcosbs_floor
#
# Submit:
#   sbatch slurm/continued_pretrain/continued_pretrain_array.sbatch
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

CHECKPOINT_PATH="${WORK}/vitturini/outputs/2026-05-04/19-42-25_vanilla_baseline_multiblock_stl10/rank0/checkpoints/epoch_00999.pt"

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT_PATH}"
  exit 1
fi

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    RUN_LABEL="vanilla_multiblock"
    EXPERIMENT="stl10_vits_ps8_multiblock"
    EXP_NAME="stl10_vanilla_multiblock_cont_from_e999_200ep"
    LOSS_ARG="loss.kind=smooth_l1"
    RUN_ARGS=()
    ;;
  1)
    RUN_LABEL="mi_affinity_novelty"
    EXPERIMENT="stl10_vits_ps8_mi"
    EXP_NAME="stl10_mi_affinity_novelty_cont_from_e999_200ep"
    LOSS_ARG="loss.base_kind=smooth_l1"
    RUN_ARGS=(
      "masking.latent.terms.H_cond.weight=[0.0,0.0]"
      "masking.latent.terms.neg_H_marg.weight=[0.0,0.0]"
      "masking.latent.terms.neg_surprise.weight=[0.0,0.0]"
      "masking.latent.terms.floor_penalty.weight=[0.0,0.0]"
      "masking.latent.terms.neg_affinity_novelty.weight=[1.0,1.0]"
      "masking.latent.terms.neg_affinity_novelty.alpha=4.0"
      "masking.latent.terms.neg_affinity_novelty.w_tgt=1.0"
      "masking.latent.terms.neg_affinity_novelty.gamma=0.5"
      "masking.latent.hard_assignment=gumbel"
      "masking.latent.gumbel_tau=1.0"
      "masking.latent.warmup_epochs=0"
      "masking.latent.nctx_min=1"
      "masking.latent.ntgt_min=4"
    )
    ;;
  2)
    RUN_LABEL="mi_hmargbs_floor"
    EXPERIMENT="stl10_vits_ps8_mi"
    EXP_NAME="stl10_mi_hmargbs_floor_cont_from_e999_200ep"
    LOSS_ARG="loss.base_kind=smooth_l1"
    RUN_ARGS=(
      "masking.latent.terms.H_cond.weight=[0.0,0.0]"
      "masking.latent.terms.neg_H_marg.weight=[1.0,1.0]"
      "masking.latent.terms.neg_surprise.weight=[0.5,0.5]"
      "masking.latent.terms.floor_penalty.weight=[5.0,5.0]"
      "masking.latent.hard_assignment=gumbel"
      "masking.latent.gumbel_tau=1.0"
      "masking.latent.warmup_epochs=0"
      "masking.latent.nctx_min=1"
      "masking.latent.ntgt_min=4"
    )
    ;;
  3)
    RUN_LABEL="mi_hmargcosbs_floor"
    EXPERIMENT="stl10_vits_ps8_mi"
    EXP_NAME="stl10_mi_hmargcosbs_floor_cont_from_e999_200ep"
    LOSS_ARG="loss.base_kind=smooth_l1"
    RUN_ARGS=(
      "masking.latent.terms.H_cond.weight=[0.0,0.0]"
      "masking.latent.terms.neg_H_marg.weight=[1.0,1.0]"
      "masking.latent.terms.neg_surprise.weight=[0.0,0.0]"
      "+masking.latent.terms.neg_cos_surprise.weight=[0.5,0.5]"
      "masking.latent.terms.floor_penalty.weight=[5.0,5.0]"
      "masking.latent.hard_assignment=gumbel"
      "masking.latent.gumbel_tau=1.0"
      "masking.latent.warmup_epochs=0"
      "masking.latent.nctx_min=1"
      "masking.latent.ntgt_min=4"
    )
    ;;
  4)
    RUN_LABEL="mi_hmarg_sketchorthcosbs_floor"
    EXPERIMENT="stl10_vits_ps8_mi"
    EXP_NAME="stl10_mi_hmarg_sketchorthcosbs_floor_cont_from_e999_200ep"
    LOSS_ARG="loss.base_kind=smooth_l1"
    RUN_ARGS=(
      "masking.latent.terms.H_cond.weight=[0.0,0.0]"
      "masking.latent.terms.neg_H_marg.weight=[1.0,1.0]"
      "masking.latent.terms.neg_surprise.weight=[0.0,0.0]"
      "masking.latent.terms.neg_sketched_orthogonal_cos_surprise.weight=[0.5,0.5]"
      "masking.latent.terms.floor_penalty.weight=[5.0,5.0]"
      "masking.latent.hard_assignment=gumbel"
      "masking.latent.gumbel_tau=1.0"
      "masking.latent.warmup_epochs=0"
      "masking.latent.nctx_min=1"
      "masking.latent.ntgt_min=4"
    )
    ;;
  *)
    echo "Unsupported SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
    exit 1
    ;;
esac

COMMON_ARGS=(
  "experiment=${EXPERIMENT}"
  "exp_name=${EXP_NAME}"
  "init_weights=${CHECKPOINT_PATH}"
  "${LOSS_ARG}"
  "data.root=$FAST/datasets/"
  "data.batch_size=2048"
  "optim.lr=3e-4"
  "train.epochs=200"
  "train.eval_every_epochs=1"
  "train.viz_enabled=true"
  "+train.viz_every_epochs=1"
  "train.viz.out_dir=viz_output_${EXP_NAME}"
  "logger.mode=offline"
)

nvidia-smi -L

echo "Array task ${SLURM_ARRAY_TASK_ID}: variant=${RUN_LABEL} exp=${EXP_NAME}"

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python -m torch.distributed.run --standalone --nproc_per_node=4 \
    -m ijepa_lite.run \
      "${COMMON_ARGS[@]}" \
      "${RUN_ARGS[@]}"
