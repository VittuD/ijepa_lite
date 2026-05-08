#!/bin/bash
#SBATCH --job-name=ijepa-aff-wup-stl10
#SBATCH --account=IscrC_PRIMAL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
#SBATCH --output=sbatch_logs/affinity_novelty/%x_%j.out
#SBATCH --error=sbatch_logs/affinity_novelty/%x_%j.err

# ---------------------------------------------------------------------------
# STL-10 pure-semantic affinity novelty with progressive masker warmup.
# No spatial/positional term — geometry is expected to emerge from
# semantic structure alone.
#
# Why this works (theory):
#   - warmup_train_masker_on_soft_assignments=True trains the masker on its
#     own soft assignments WHILE the encoder trains on vanilla multiblock hard
#     masks.  During warmup the encoder is partially trained and still has
#     meaningful within-cluster feature variation, so neg_affinity_novelty
#     has non-trivial intra-cluster gradient.  By the time the encoder
#     hardens its clusters the masker already has real Adam state and
#     content-adaptive tendencies.
#   - floor_penalty (information-theoretic, no geometry) prevents ign-collapse
#     if the affinity gradient ever flattens within a cluster.
#   - The Adam phantom-step bug is fixed in train_loop.py: the masker
#     optimizer is not stepped during warmup-no-train epochs, so there is no
#     3.16x LR overshoot at warmup end.
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

EXP_NAME="affinity_novelty_stl10_wup50"

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
      masking.latent.terms.floor_penalty.weight="[5.0,5.0]" \
      masking.latent.terms.neg_affinity_novelty.weight="[1.0,1.0]" \
      masking.latent.terms.neg_affinity_novelty.alpha=4.0 \
      masking.latent.terms.neg_affinity_novelty.w_tgt=1.0 \
      masking.latent.terms.neg_affinity_novelty.gamma=0.5 \
      masking.latent.hard_assignment=gumbel \
      masking.latent.gumbel_tau=1.0 \
      masking.latent.warmup_epochs=50 \
      masking.latent.warmup_use_vanilla_multiblock=true \
      masking.latent.warmup_train_masker_on_soft_assignments=true \
      masking.latent.nctx_min=1 \
      masking.latent.ntgt_min=4 \
      train.eval_every_epochs=5 \
      train.viz.out_dir="viz_output_${EXP_NAME}" \
      logger.mode=offline
