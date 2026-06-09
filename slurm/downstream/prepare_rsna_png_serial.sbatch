#!/bin/bash
#SBATCH --job-name=rsna-dcm2png
#SBATCH --account=IscrC_TBoneAI
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=sbatch_logs/downstream/rsna_prepare_png/%x_%j.out
#SBATCH --error=sbatch_logs/downstream/rsna_prepare_png/%x_%j.err

# ---------------------------------------------------------------------------
# Convert RSNA Pneumonia Detection DICOMs to PNGs off the login node.
#
# Input:
#   ${DATA_ROOT}/rsna-pneumonia-detection/stage_2_train_images/*.dcm
#
# Output:
#   ${DATA_ROOT}/rsna-pneumonia-detection/png_images/<patientId>.png
#
# Submit:
#   mkdir -p sbatch_logs/downstream/rsna_prepare_png
#   sbatch slurm/downstream/prepare_rsna_png_serial.sbatch
# ---------------------------------------------------------------------------

set -euo pipefail

module purge
module load profile/deeplrn
module load python/3.11.6--gcc--8.5.0

source $WORK/vitturini/ijepa_venv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

REPO_ROOT="${WORK}/vitturini/ijepa_lite"
DATA_ROOT="${DATA_ROOT:-${FAST:-/scratch}/datasets}"
LOG_DIR="${WORK}/vitturini/sbatch_logs/downstream/rsna_prepare_png"

mkdir -p "${LOG_DIR}"

echo "REPO_ROOT=${REPO_ROOT}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}"

srun python "${REPO_ROOT}/scripts/prepare_rsna_png.py" \
  --data-root "${DATA_ROOT}" \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --log-every 500
