#!/bin/bash
# FSDP2 + Tensor Parallel Llama 2 example launcher on Ascend NPU.
#
# Sources set_env.sh for CANN toolkit + conda environment.
#
# Usage:
#   bash run_fsdp2_tp_llama2.sh [num_devices] [--extra-flags ...]
#
# Examples:
#   bash run_fsdp2_tp_llama2.sh 8
#   bash run_fsdp2_tp_llama2.sh 8 --model-size debug --tp-size 2
#   bash run_fsdp2_tp_llama2.sh 8 --model-size 1B --tp-size 2 --mixed-precision
#   bash run_fsdp2_tp_llama2.sh 8 --model-size debug --use-synthetic-data --epochs 1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "${SCRIPT_DIR}")")"

# Source CANN toolkit + conda environment (set_env.sh does both).
# Temporarily relax -u because ~/.bashrc references unbound variables
# (BASHRCSOURCED) that trigger "unbound variable" under `set -u`.
SET_ENV="${SCRIPT_DIR}/../set_env.sh"
if [ -f "${SET_ENV}" ]; then
    echo "[run_fsdp2_tp_llama2.sh] Sourcing ${SET_ENV} ..."
    set +u
    source "${SET_ENV}"
    set -u
else
    echo "[run_fsdp2_tp_llama2.sh] ERROR: ${SET_ENV} not found" >&2
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
NGPU="${1:-8}"
shift 1 2>/dev/null || true  # remaining args go to the Python script

echo "[run_fsdp2_tp_llama2.sh] Launching fsdp2_tp_llama2_main.py with ${NGPU} devices ..."
echo "[run_fsdp2_tp_llama2.sh] Extra flags: ${*:-<none>}"

torchrun \
    --nnodes=1 \
    --nproc_per_node="${NGPU}" \
    "${SCRIPT_DIR}/fsdp2_tp_llama2_main.py" \
    "$@"
