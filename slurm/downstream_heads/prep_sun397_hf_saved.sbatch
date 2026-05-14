#!/bin/bash
#SBATCH --job-name=prep-hf-saved
#SBATCH --account=IscrC_DEEPECHO
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=sbatch_logs/downstream_heads/%x_%j.out
#SBATCH --error=sbatch_logs/downstream_heads/%x_%j.err

set -euo pipefail

module purge
module load profile/deeplrn
module load python/3.11.6--gcc--8.5.0

source $WORK/vitturini/ijepa_venv/bin/activate

mkdir -p sbatch_logs/downstream_heads

DATA_ROOT="${FAST:-/scratch}/datasets"
SUN397_SOURCE_DIR="${DATA_ROOT}/sun397_hf"
SUN397_DEST_DIR="${DATA_ROOT}/sun397_hf_saved"
FAIRFACE_SOURCE_DIR="${DATA_ROOT}/fairface_hf"
FAIRFACE_DEST_DIR="${DATA_ROOT}/fairface_hf_saved"

echo "DATA_ROOT=${DATA_ROOT}"
echo "SUN397_SOURCE_DIR=${SUN397_SOURCE_DIR}"
echo "SUN397_DEST_DIR=${SUN397_DEST_DIR}"
echo "FAIRFACE_SOURCE_DIR=${FAIRFACE_SOURCE_DIR}"
echo "FAIRFACE_DEST_DIR=${FAIRFACE_DEST_DIR}"

python -m ijepa_lite.scripts.save_hf_dataset \
  --repo-id tanganke/sun397 \
  --source-dir "${SUN397_SOURCE_DIR}" \
  --dest-dir "${SUN397_DEST_DIR}" \
  --cache-dir "${DATA_ROOT}" \
  --splits train test

python -m ijepa_lite.scripts.save_hf_dataset \
  --repo-id HuggingFaceM4/FairFace \
  --source-dir "${FAIRFACE_SOURCE_DIR}" \
  --dest-dir "${FAIRFACE_DEST_DIR}" \
  --cache-dir "${DATA_ROOT}" \
  --subset "0.25" \
  --splits train validation
