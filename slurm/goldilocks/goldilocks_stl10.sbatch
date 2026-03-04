#!/bin/bash
#SBATCH --job-name=ijepa-gldk-stl10
#SBATCH --account=IscrC_TBoneAI
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=06:30:00
#SBATCH --output=sbatch_logs/%x_%j.out
#SBATCH --error=sbatch_logs/%x_%j.err

# ---------------------------------------------------------------------------
# P2 full pretraining — GoldilocksTeacherMasker, STL-10, 250 epochs
#
# Main comparison against MI-rate baseline (stl10_vits_ps8_mi.sbatch).
# All training hyperparameters are inherited from stl10_vits_ps8_mi via the
# experiment defaults — only the masker and loss are swapped.
#
# Key metrics to watch in W&B:
#   mask/tgt_pos_std            >0.05 and rising → content-adaptive
#   mask/tgt_assignment_entropy rising → leaving fixed-position fixed point
#   mask/p_tgt_score_std        >0 → soft scores vary with image content
#   train/reconstruction_loss   should match or beat MI baseline trajectory
#   L_goldilocks                should decrease and stabilise (not flat from step 0)
#
# After training, run linear probe:
#   export PRETRAIN_CKPT=<output_dir>/rank0/checkpoints/last.pt
#   sbatch slurm/goldilocks/goldilocks_linear_probe.sbatch
#
# Run:
#   sbatch slurm/goldilocks/goldilocks_stl10.sbatch
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

mkdir -p sbatch_logs

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python -m torch.distributed.run --standalone --nproc_per_node=4 \
    -m ijepa_lite.run \
      experiment=stl10_vits_ps8_goldilocks \
      data.root=$FAST/datasets/ \
      logger.mode=offline
