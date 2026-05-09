#!/bin/bash
#SBATCH --job-name=ijepa-sigreg-stl10-dbg
#SBATCH --account=IscrC_PRIMAL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=sbatch_logs/sigreg/%x_%j.out
#SBATCH --error=sbatch_logs/sigreg/%x_%j.err

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

mkdir -p sbatch_logs/sigreg

nvidia-smi -L

EXP_NAME="sigreg_multiblock_stl10_dbg"

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python -m torch.distributed.run --standalone --nproc_per_node=4 \
    -m ijepa_lite.run \
      experiment=stl10_vits_ps8_multiblock_sigreg \
      exp_name="$EXP_NAME" \
      data.root=$FAST/datasets/ \
      data.batch_size=512 \
      train.epochs=10 \
      train.save_every_epochs=5 \
      train.eval_every_epochs=1 \
      train.viz.out_dir="viz_output_${EXP_NAME}" \
      logger.mode=offline
