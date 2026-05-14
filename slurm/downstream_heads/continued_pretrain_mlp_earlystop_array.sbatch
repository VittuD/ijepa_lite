#!/bin/bash
#SBATCH --job-name=ijepa-ctp-heads-es
#SBATCH --account=IscrC_DEEPECHO
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --array=0-34
#SBATCH --output=sbatch_logs/downstream_heads/%x_%A_%a.out
#SBATCH --error=sbatch_logs/downstream_heads/%x_%A_%a.err

# ---------------------------------------------------------------------------
# Early-stopped MLP downstream sweep over 5 continued-pretraining checkpoints
# x 6 downstream datasets (excluding STL10).
#
# Checkpoint index:
#   0 -> mi_affinity_novelty
#   1 -> mi_hmarg_sketchorthcosbs_floor
#   2 -> mi_hmargbs_floor
#   3 -> mi_hmargcosbs_floor
#   4 -> vanilla_multiblock
#
# Dataset index:
#   0 -> cifar100
#   1 -> dtd
#   2 -> sun397
#   3 -> food101
#   4 -> fairface (race)
#   5 -> tissuemnist
#   6 -> vocseg
#
# Mapping:
#   ckpt_idx    = SLURM_ARRAY_TASK_ID / 7
#   dataset_idx = SLURM_ARRAY_TASK_ID % 7
#
# Submit:
#   sbatch slurm/downstream_heads/continued_pretrain_mlp_earlystop_array.sbatch
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

mkdir -p sbatch_logs/downstream_heads

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"

CKPT_LABELS=(
  mi_affinity_novelty
  mi_hmarg_sketchorthcosbs_floor
  mi_hmargbs_floor
  mi_hmargcosbs_floor
  vanilla_multiblock
)

CKPT_REL_PATHS=(
  outputs/2026-05-08/03-48-11_stl10_mi_affinity_novelty_cont_from_e999_200ep/rank0/checkpoints/epoch_00099.pt
  outputs/2026-05-08/03-48-27_stl10_mi_hmarg_sketchorthcosbs_floor_cont_from_e999_200ep/rank0/checkpoints/epoch_00099.pt
  outputs/2026-05-08/03-48-27_stl10_mi_hmargbs_floor_cont_from_e999_200ep/rank0/checkpoints/epoch_00099.pt
  outputs/2026-05-08/03-48-27_stl10_mi_hmargcosbs_floor_cont_from_e999_200ep/rank0/checkpoints/epoch_00099.pt
  outputs/2026-05-08/03-48-27_stl10_vanilla_multiblock_cont_from_e999_200ep/rank0/checkpoints/epoch_00099.pt
)

DATASETS=(
  cifar100
  dtd
  sun397
  food101
  fairface
  tissuemnist
  vocseg
)

NUM_DATASETS=${#DATASETS[@]}
CKPT_IDX=$((SLURM_ARRAY_TASK_ID / NUM_DATASETS))
DATASET_IDX=$((SLURM_ARRAY_TASK_ID % NUM_DATASETS))

if (( CKPT_IDX >= ${#CKPT_REL_PATHS[@]} )); then
  echo "Unsupported SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
  exit 1
fi

CKPT_LABEL="${CKPT_LABELS[$CKPT_IDX]}"
DATASET="${DATASETS[$DATASET_IDX]}"
export PRETRAIN_CKPT="${REPO_ROOT}/${CKPT_REL_PATHS[$CKPT_IDX]}"
export DATA_ROOT="${FAST:-/scratch}/datasets/"

if [[ ! -f "${PRETRAIN_CKPT}" ]]; then
  echo "Checkpoint not found: ${PRETRAIN_CKPT}"
  exit 1
fi

EXTRA_ARGS=()
if [[ "$DATASET" == "fairface" ]]; then
  EXTRA_ARGS+=(data.target_attr=race)
fi

EXPERIMENT="in1k_96_vits_ps8_downstream_mlp_earlystop"
if [[ "$DATASET" == "vocseg" ]]; then
  EXPERIMENT="in1k_96_vits_ps8_vocseg_probe_earlystop"
fi

EXP_NAME="in1k_96_vits_ps8_${CKPT_LABEL}_${DATASET}_es_probe"

echo "array_task=${SLURM_ARRAY_TASK_ID}"
echo "ckpt_label=${CKPT_LABEL}"
echo "dataset=${DATASET}"
echo "PRETRAIN_CKPT=${PRETRAIN_CKPT}"
echo "DATA_ROOT=${DATA_ROOT}"
nvidia-smi -L

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python -m ijepa_lite.run \
    experiment="${EXPERIMENT}" \
    data="${DATASET}" \
    data.root="${DATA_ROOT}" \
    exp_name="${EXP_NAME}" \
    logger.mode=offline \
    "${EXTRA_ARGS[@]}"
