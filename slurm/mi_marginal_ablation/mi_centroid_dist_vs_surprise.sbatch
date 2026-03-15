#!/bin/bash
#SBATCH --job-name=ijepa-mi-centroid
#SBATCH --account=IscrC_PRIMAL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
#SBATCH --output=sbatch_logs/mi_marginal/%x_%j.out
#SBATCH --error=sbatch_logs/mi_marginal/%x_%j.err

# ---------------------------------------------------------------------------
# Centroid distance vs Bayesian surprise ablation.
#
# Baseline (already run):
#   mi_HmargBS_floor_learned — H_cond=OFF, neg_H_marg=1, neg_surprise=0.5, floor=5
#
# This run:
#   mi_HmargCD_floor_learned — same but swap neg_surprise for neg_centroid_dist
#   (||μ_ctx − μ_tgt||² only, drops the within-target variance term)
#
# Same weight (0.5) for fair comparison.
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

export EXP_NAME="mi_HmargCD_floor_learned"

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python -m torch.distributed.run --standalone --nproc_per_node=4 \
    -m ijepa_lite.run \
      experiment=stl10_vits_ps8_mi \
      exp_name="$EXP_NAME" \
      masking.latent.terms.H_cond.weight='[0,0]' \
      masking.latent.terms.neg_H_marg.weight='[1.0,1.0]' \
      masking.latent.terms.neg_surprise.weight='[0,0]' \
      masking.latent.terms.neg_centroid_dist.weight='[0.5,0.5]' \
      masking.latent.terms.floor_penalty.weight='[5.0,5.0]' \
      data.root=$FAST/datasets/ \
      data.batch_size=2048 \
      train.viz.out_dir="viz_output_${EXP_NAME}" \
      logger.mode=offline
