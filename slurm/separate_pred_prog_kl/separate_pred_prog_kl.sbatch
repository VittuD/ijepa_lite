#!/bin/bash
#SBATCH --job-name=ijepa-sep-pred-prog-kl
#SBATCH --account=IscrC_PRIMAL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=10:00:00
#SBATCH --array=0-5
#SBATCH --output=sbatch_logs/separate_pred_prog_kl/%x_%A_%a.out
#SBATCH --error=sbatch_logs/separate_pred_prog_kl/%x_%A_%a.err

# ---------------------------------------------------------------------------
# Sweep: separate vs joint block prediction × progressive KL weight.
#
# Fixed: num_tgt_blocks=5, n_start_tgt=2, gumbel tau=0.5,
#        cross_surprise weight=0.1, forward KL, deterministic phase mode.
# Axes:
#   λ_kl  ∈ {0.01, 0.1, 1.0}   (klw_idx = TASK_ID / 2)
#   joint ∈ {false, true}       (joint_idx = TASK_ID % 2)
#
# TASK_ID | λ_kl | predict_blocks_jointly
#       0 | 0.01 | false
#       1 | 0.01 | true
#       2 | 0.1  | false
#       3 | 0.1  | true
#       4 | 1.0  | false
#       5 | 1.0  | true
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

mkdir -p sbatch_logs/separate_pred_prog_kl

KLW_VALUES=(0.01 0.1 1.0)
JOINT_VALUES=(false true)
JOINT_STR_VALUES=(sep joint)

KLW_IDX=$(( SLURM_ARRAY_TASK_ID / 2 ))
JOINT_IDX=$(( SLURM_ARRAY_TASK_ID % 2 ))

KLW=${KLW_VALUES[$KLW_IDX]}
JOINT=${JOINT_VALUES[$JOINT_IDX]}
JOINT_STR=${JOINT_STR_VALUES[$JOINT_IDX]}

export EXP_NAME="prog_kl_5blk_klw${KLW}_${JOINT_STR}"

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python -m torch.distributed.run --standalone --nproc_per_node=4 \
    -m ijepa_lite.run \
      experiment=stl10_vits_ps8_mi_nway \
      exp_name="$EXP_NAME" \
      masking.latent.hard_assignment=gumbel \
      masking.latent.gumbel_tau=0.5 \
      masking.latent.num_tgt_blocks=5 \
      masking.latent.ntgt_min_per_block=1 \
      +masking.latent.n_start_tgt=2 \
      +masking.latent.phase_mode=deterministic \
      ~masking.latent.terms.nway_neg_H_marg \
      ~masking.latent.terms.nway_floor_penalty \
      masking.latent.terms.nway_cross_surprise.weight='[0.1,0.1]' \
      +masking.latent.terms.nway_progressive_kl.weight="[$KLW,$KLW]" \
      +masking.latent.terms.nway_progressive_kl.direction=forward \
      +model.predict_blocks_jointly=$JOINT \
      data.root=$FAST/datasets/ \
      data.batch_size=512 \
      train.epochs=1000 \
      train.viz_enabled=true \
      logger.mode=offline
