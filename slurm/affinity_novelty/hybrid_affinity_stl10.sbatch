#!/bin/bash
#SBATCH --job-name=ijepa-haff-stl10
#SBATCH --account=IscrC_PRIMAL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --output=sbatch_logs/affinity_novelty/%x_%j.out
#SBATCH --error=sbatch_logs/affinity_novelty/%x_%j.err

# ---------------------------------------------------------------------------
# STL-10 hybrid feature-position affinity novelty pretraining.
# Uses neg_hybrid_affinity_novelty which multiplies the feature affinity by
# a column-normalised spatial affinity A_spat[i,j] = exp(-||pos_i-pos_j||²
# / 2*sigma_pos²) / Z_j, where Z_j normalises each column to sum 1.
# Column normalisation removes center-patch position bias; sigma_pos=0.5
# gives a tight spatial neighbourhood for intra-cluster gradient.
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

EXP_NAME="hybrid_affinity_stl10"

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
      masking.latent.terms.neg_hybrid_affinity_novelty.weight="[1.0,1.0]" \
      masking.latent.terms.neg_hybrid_affinity_novelty.alpha=4.0 \
      masking.latent.terms.neg_hybrid_affinity_novelty.w_tgt=1.0 \
      masking.latent.terms.neg_hybrid_affinity_novelty.gamma=0.5 \
      masking.latent.terms.neg_hybrid_affinity_novelty.sigma_pos=0.5 \
      masking.latent.hard_assignment=gumbel \
      masking.latent.gumbel_tau=1.0 \
      masking.latent.warmup_epochs=0 \
      masking.latent.nctx_min=1 \
      masking.latent.ntgt_min=4 \
      train.eval_every_epochs=1 \
      train.viz_enabled=true \
      +train.viz_every_epochs=1 \
      train.viz.out_dir="viz_output_${EXP_NAME}" \
      logger.mode=offline
