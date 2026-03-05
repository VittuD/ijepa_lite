#!/bin/bash
#SBATCH --job-name=ijepa-gldk-ema-dbg
#SBATCH --account=IscrC_TBoneAI
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=sbatch_logs/goldilocks_ema_debug/%x_%j.out
#SBATCH --error=sbatch_logs/goldilocks_ema_debug/%x_%j.err

# ---------------------------------------------------------------------------
# Path B smoke test — GoldilocksTeacherMasker with EMA-stabilized signal
#
# Before Path B is implemented: runs as vanilla goldilocks (ema_signal ignored).
# After Path B is implemented: verifies the extra predictor forward pass lands
# correctly and ema_patch_loss is finite and non-NaN.
#
# Pass criteria (post-implementation):
#   - No crashes
#   - mask/masker_loss logged and finite
#   - mask/marginal_score_std logged (Path B diagnostic)
#   - ema_patch_loss shape matches patch_loss (check log output)
#
# Run:
#   sbatch slurm/goldilocks_ema/goldilocks_ema_debug.sbatch
# ---------------------------------------------------------------------------

set -euo pipefail

module purge
module load profile/deeplrn
module load python/3.11.6--gcc--8.5.0

source $WORK/vitturini/ijepa_venv/bin/activate

nvidia-smi -L || true

mkdir -p sbatch_logs/goldilocks_ema_debug

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python -m ijepa_lite.run \
    experiment=stl10_vits_ps8_goldilocks_ema \
    masking/latent=goldilocks_ema_fixed_k \
    data.root=$FAST/datasets/ \
    data.batch_size=64 \
    data.num_workers=4 \
    train.epochs=1 \
    train.log_every=5 \
    train.save_every_epochs=1 \
    logger.mode=offline
