#!/bin/bash
#SBATCH --job-name=ijepa-dbg-winners-stage1-sep
#SBATCH --account=IscrC_PRIMAL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --output=sbatch_logs/separate_pred_prog_kl/%x_%j.out
#SBATCH --error=sbatch_logs/separate_pred_prog_kl/%x_%j.err

# ---------------------------------------------------------------------------
# Stage 1 winners smoke test:
# - preserve raw hard winners in MINWayMasker.aux
# - keep the current dense training path unchanged
# - run the separate-pred progressive-KL setup on a safer debug batch size
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
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

nvidia-smi -L

mkdir -p sbatch_logs/separate_pred_prog_kl

export EXP_NAME="dbg_winners_stage1_sep_prog_kl"

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python -m torch.distributed.run --standalone --nproc_per_node=4 \
    -m ijepa_lite.run \
      experiment=stl10_vits_ps8_mi_nway \
      exp_name="$EXP_NAME" \
      masking.latent.hard_assignment=gumbel \
      masking.latent.gumbel_tau=0.5 \
      masking.latent.num_tgt_blocks=5 \
      masking.latent.ntgt_min_per_block=1 \
      masking.latent.max_total_tgt=112 \
      masking.latent.max_tgt_per_block=48 \
      distributed.find_unused_parameters=true \
      +masking.latent.n_start_tgt=2 \
      +masking.latent.phase_mode=deterministic \
      ~masking.latent.terms.nway_neg_H_marg \
      ~masking.latent.terms.nway_floor_penalty \
      masking.latent.terms.nway_cross_surprise.weight='[1.0,1.0]' \
      +masking.latent.terms.nway_progressive_kl.weight='[10.0,10.0]' \
      +masking.latent.terms.nway_progressive_kl.direction=forward \
      +model.predict_blocks_jointly=false \
      data.root=$FAST/datasets/ \
      data.batch_size=512 \
      train.epochs=1 \
      train.eval_every_epochs=1000 \
      train.log_every=1 \
      train.viz_enabled=false \
      logger.mode=offline
