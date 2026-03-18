#!/bin/bash
#SBATCH --job-name=ijepa-mi-nway-xs-argmax-alive
#SBATCH --account=IscrC_PRIMAL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00
#SBATCH --output=sbatch_logs/mi_marginal/%x_%j.out
#SBATCH --error=sbatch_logs/mi_marginal/%x_%j.err
#SBATCH --array=1-4

# ---------------------------------------------------------------------------
# N-way cross-target surprise with argmax hard assignment + role_alive penalty.
#
# Array job: SLURM_ARRAY_TASK_ID = num_tgt_blocks (1, 2, 3, 4).
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

M=${SLURM_ARRAY_TASK_ID}
export EXP_NAME="mi_nway_cross_surprise_argmax_alive_${M}blk"

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python -m torch.distributed.run --standalone --nproc_per_node=4 \
    -m ijepa_lite.run \
      experiment=stl10_vits_ps8_mi_nway \
      exp_name="$EXP_NAME" \
      masking.latent.hard_assignment=argmax \
      masking.latent.terms.nway_neg_H_marg.weight='[1.0,1.0]' \
      masking.latent.terms.nway_cross_surprise.weight='[0.1,0.1]' \
      ~masking.latent.terms.nway_floor_penalty \
      +masking.latent.terms.role_alive.weight='[5.0,5.0]' \
      +masking.latent.terms.role_alive.p_min=0.05 \
      masking.latent.num_tgt_blocks=$M \
      data.root=$FAST/datasets/ \
      data.batch_size=2048 \
      train.viz_enabled=true \
      train.viz.out_dir="viz_output_${EXP_NAME}" \
      logger.mode=offline
