#!/bin/bash
#SBATCH --job-name=dbg-ijepa-affinity-novelty-stl10
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
# STL-10 affinity-coverage novelty pretraining (loss_formulation.tex /
# novelty_proxy.tex). Comparable in setup to vanilla_multiblock_stl10 —
# same data root, batch size, eval cadence, predictor loss — but with the
# 3-way MI masker driven by neg_affinity_novelty and all other terms off.
# Hard assignment via Gumbel; no warmup (reach-coupling self-warms).
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

EXP_NAME="affinity_novelty_wup750_stl10_dbg"

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
      masking.latent.terms.neg_affinity_novelty.alpha=4.0 \
      masking.latent.terms.neg_affinity_novelty.w_tgt=1.0 \
      masking.latent.terms.neg_affinity_novelty.gamma="[0.5,0.5]" \
      masking.latent.terms.neg_affinity_novelty.gamma_schedule=cosine \
      masking.latent.terms.neg_affinity_novelty.sigma_rho="[1.0,1.0]" \
      masking.latent.terms.neg_affinity_novelty.sigma_rho_schedule=linear \
      masking.latent.hard_assignment=gumbel \
      masking.latent.gumbel_tau=1.0 \
      masking.latent.warmup_epochs=0 \
      masking.latent.nctx_min=1 \
      masking.latent.ntgt_min=4 \
      train.eval_every_epochs=1 \
      train.viz_enabled=true \
      +train.viz_every_epochs=1 \
      masking.latent.warmup_use_vanilla_multiblock=true \
      +warmup_train_masker_on_soft_assignments=true \
      masking.latent.warmup_epochs=50 \
      train.viz.out_dir="viz_output_${EXP_NAME}" \
      logger.mode=offline
