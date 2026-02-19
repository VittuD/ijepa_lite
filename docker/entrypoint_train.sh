#!/usr/bin/env bash
set -euo pipefail

# If you want to force torchrun even on 1 GPU, set USE_TORCHRUN=1
USE_TORCHRUN="${USE_TORCHRUN:-0}"

# Detect GPU count
GPU_COUNT="$(python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"

# Torchrun controls (set via SWARM service env vars)
NPROC_PER_NODE="${NPROC_PER_NODE:-${GPU_COUNT}}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

# If WORLD_SIZE is set externally, respect it (common in schedulers)
WORLD_SIZE_ENV="${WORLD_SIZE:-}"

# Decide if we should torchrun:
# - multi-gpu (nproc>1) OR
# - multi-node (nnodes>1) OR
# - explicit USE_TORCHRUN=1 OR
# - WORLD_SIZE already defined
if [[ "${USE_TORCHRUN}" == "1" ]] || [[ "${NNODES}" -gt 1 ]] || [[ "${NPROC_PER_NODE}" -gt 1 ]] || [[ -n "${WORLD_SIZE_ENV}" ]]; then
  echo "[entrypoint] Using torchrun"
  echo "  NNODES=${NNODES} NODE_RANK=${NODE_RANK} NPROC_PER_NODE=${NPROC_PER_NODE}"
  echo "  MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"

  if [[ "${NNODES}" -eq 1 ]]; then
    # single node
    exec torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
      -m ijepa_lite.run "$@"
  else
    # multi node
    exec torchrun \
      --nnodes="${NNODES}" \
      --node_rank="${NODE_RANK}" \
      --nproc_per_node="${NPROC_PER_NODE}" \
      --master_addr="${MASTER_ADDR}" \
      --master_port="${MASTER_PORT}" \
      -m ijepa_lite.run "$@"
  fi
else
  echo "[entrypoint] Using plain python (single process)"
  exec python -m ijepa_lite.run "$@"
fi
