#!/bin/bash
#SBATCH --job-name=ijepa-mi-nway-cross-argmax-alive-4blk
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
# MI n-way: cross_surprise + role_alive (replaces floor), argmax, 4 blocks
# Original: mi_nway_cross_surprise_argmax_alive_4blk_0318_201254_r0
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

EXP_NAME="mi_nway_cross_surprise_argmax_alive_4blk"

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python -m torch.distributed.run --standalone --nproc_per_node=4 \
    -m ijepa_lite.run \
      experiment=stl10_vits_ps8_mi_nway \
      exp_name="$EXP_NAME" \
      masking.latent.hard_assignment=argmax \
      masking.latent.num_tgt_blocks=4 \
      masking.latent.terms.nway_neg_H_marg.weight='[1.0,1.0]' \
      masking.latent.terms.nway_cross_surprise.weight='[0.1,0.1]' \
      ~masking.latent.terms.nway_floor_penalty \
      +masking.latent.terms.role_alive.weight='[5.0,5.0]' \
      +masking.latent.terms.role_alive.p_min=0.05 \
      data.root=$FAST/datasets/ \
      data.batch_size=2048 \
      train.viz_enabled=true \
      train.viz.out_dir="viz_output_${EXP_NAME}" \
      logger.mode=offline
