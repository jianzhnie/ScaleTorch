#!/bin/bash
# FSDP2 example launcher on Ascend NPU.
#
# Sources set_env.sh for CANN toolkit + conda environment.
#
# Usage:
#   bash run_example.sh [script] [num_devices] [--extra-flags ...]
#
# Examples:
#   bash run_example.sh fsdp2_main.py 8
#   bash run_example.sh fsdp2_main.py 8 --mixed-precision
#   bash run_example.sh fsdp2_main.py 8 --explicit-prefetching
#   bash run_example.sh fsdp2_main.py 8 --dcp-api

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "${SCRIPT_DIR}")")"

# Source CANN toolkit + conda environment (set_env.sh does both).
# Temporarily relax -u because ~/.bashrc references unbound variables
# (BASHRCSOURCED) that trigger "unbound variable" under `set -u`.
SET_ENV="${SCRIPT_DIR}/../set_env.sh"
if [ -f "${SET_ENV}" ]; then
    echo "[run_example.sh] Sourcing ${SET_ENV} ..."
    set +u
    source "${SET_ENV}"
    set -u
else
    echo "[run_example.sh] ERROR: ${SET_ENV} not found" >&2
    exit 1
fi

# --- NPU-specific environment variables (same pattern as device_mesh/run.sh) ---
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export HCCL_CONNECT_TIMEOUT=3600

# Force IPv4 — hostname has no AAAA record, causing HCCL init to hang
# on repeated IPv6 resolution failures (gai error: -2).
export NCCL_SOCKET_FAMILY=AF_INET

# Bind HCCL and gloo to the detected NIC.
export HCCL_SOCKET_IFNAME="enp66s0f5"
export GLOO_SOCKET_IFNAME="enp66s0f5"

# --- parse args ---------------------------------------------------------------
SCRIPT="${1:-fsdp2_main.py}"
NGPU="${2:-8}"
shift 2 2>/dev/null || true  # remaining args go to the Python script

echo "[run_example.sh] Launching ${SCRIPT} with ${NGPU} devices ..."
echo "[run_example.sh] Extra flags: ${*:-<none>}"

torchrun \
    --nnodes=1 \
    --nproc_per_node="${NGPU}" \
    "${SCRIPT_DIR}/${SCRIPT}" \
    "$@"
