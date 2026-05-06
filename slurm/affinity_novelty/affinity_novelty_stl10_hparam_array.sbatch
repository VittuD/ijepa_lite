#!/bin/bash
#SBATCH --job-name=ijepa-affinity-novelty-hp
#SBATCH --account=IscrC_PRIMAL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=05:00:00
#SBATCH --array=0-7
#SBATCH --output=sbatch_logs/affinity_novelty/%x_%A_%a.out
#SBATCH --error=sbatch_logs/affinity_novelty/%x_%A_%a.err

# ---------------------------------------------------------------------------
# STL-10 affinity-novelty hyper-parameter sweep.
#
# Goal:
#   Probe the two failure modes seen after ~epoch 50:
#   1) small / isolated main subjects drifting to ignore
#   2) too little strongly-context mass
#
# Matrix:
#   idx  exp suffix      alpha  w_tgt  gamma  nctx_min  sigma_rho
#    0   base             4.0   1.00   0.50      1        1.0
#    1   nctx4            4.0   1.00   0.50      4        1.0
#    2   g035             4.0   1.00   0.35      1        1.0
#    3   g025             4.0   1.00   0.25      1        1.0
#    4   wt075            4.0   0.75   0.50      1        1.0
#    5   a25              2.5   1.00   0.50      1        1.0
#    6   sr15             4.0   1.00   0.50      1        1.5
#    7   ctx_friendly     2.5   0.70   0.30      4        1.5
#
# Submit:
#   sbatch slurm/affinity_novelty/affinity_novelty_stl10_hparam_array.sbatch
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

mkdir -p sbatch_logs/affinity_novelty

nvidia-smi -L

EXP_SUFFIXES=(
  base
  nctx4
  g035
  g025
  wt075
  a25
  sr15
  ctx_friendly
)
ALPHAS=(
  4.0
  4.0
  4.0
  4.0
  4.0
  2.5
  4.0
  2.5
)
W_TGTS=(
  1.0
  1.0
  1.0
  1.0
  0.75
  1.0
  1.0
  0.70
)
GAMMAS=(
  0.5
  0.5
  0.35
  0.25
  0.5
  0.5
  0.5
  0.30
)
NCTX_MINS=(
  1
  4
  1
  1
  1
  1
  1
  4
)
SIGMA_RHOS=(
  1.0
  1.0
  1.0
  1.0
  1.0
  1.0
  1.5
  1.5
)

EXP_SUFFIX=${EXP_SUFFIXES[$SLURM_ARRAY_TASK_ID]}
ALPHA=${ALPHAS[$SLURM_ARRAY_TASK_ID]}
W_TGT=${W_TGTS[$SLURM_ARRAY_TASK_ID]}
GAMMA=${GAMMAS[$SLURM_ARRAY_TASK_ID]}
NCTX_MIN=${NCTX_MINS[$SLURM_ARRAY_TASK_ID]}
SIGMA_RHO=${SIGMA_RHOS[$SLURM_ARRAY_TASK_ID]}

EXP_NAME="affinity_novelty_stl10_${EXP_SUFFIX}"

echo "Array task ${SLURM_ARRAY_TASK_ID}: exp=${EXP_NAME} alpha=${ALPHA} w_tgt=${W_TGT} gamma=${GAMMA} nctx_min=${NCTX_MIN} sigma_rho=${SIGMA_RHO}"

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python -m torch.distributed.run --standalone --nproc_per_node=4 \
    -m ijepa_lite.run \
      experiment=stl10_vits_ps8_mi \
      exp_name="$EXP_NAME" \
      loss.kind=smooth_l1 \
      data.root=$FAST/datasets/ \
      data.batch_size=2048 \
      masking.latent.terms.H_cond.weight="[0.0,0.0]" \
      masking.latent.terms.neg_H_marg.weight="[0.0,0.0]" \
      masking.latent.terms.neg_surprise.weight="[0.0,0.0]" \
      masking.latent.terms.floor_penalty.weight="[0.0,0.0]" \
      masking.latent.terms.neg_affinity_novelty.weight="[1.0,1.0]" \
      masking.latent.terms.neg_affinity_novelty.alpha=${ALPHA} \
      masking.latent.terms.neg_affinity_novelty.w_tgt=${W_TGT} \
      masking.latent.terms.neg_affinity_novelty.gamma=${GAMMA} \
      masking.latent.terms.neg_affinity_novelty.sigma_rho=${SIGMA_RHO} \
      masking.latent.hard_assignment=gumbel \
      masking.latent.gumbel_tau=1.0 \
      masking.latent.warmup_epochs=0 \
      masking.latent.nctx_min=${NCTX_MIN} \
      masking.latent.ntgt_min=4 \
      train.eval_every_epochs=1 \
      train.viz_enabled=true \
      +train.viz_every_epochs=1 \
      train.viz.out_dir="viz_output_${EXP_NAME}" \
      logger.mode=offline
