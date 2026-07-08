#!/bin/bash
# Run FSDP2+TP Llama2 with real wikitext2 data on Ascend NPU.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# -- Source CANN toolkit + conda environment --------------------------------
SET_ENV="${SCRIPT_DIR}/../set_env.sh"
if [ -f "${SET_ENV}" ]; then
    echo "[run_wikitext2] Sourcing ${SET_ENV} ..."
    set +u
    source "${SET_ENV}"
    set -u
else
    echo "[run_wikitext2] ERROR: ${SET_ENV} not found" >&2
    exit 1
fi

# -- Assert critical env vars -----------------------------------------------
: "${ASCEND_HOME_PATH:?ASCEND_HOME_PATH not set — CANN toolkit not sourced}"
: "${ASCEND_TOOLKIT_HOME:?ASCEND_TOOLKIT_HOME not set}"

# -- NPU-specific env vars --------------------------------------------------
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export HCCL_CONNECT_TIMEOUT=3600
export NCCL_SOCKET_FAMILY=AF_INET
export HCCL_SOCKET_IFNAME="enp66s0f5"
export GLOO_SOCKET_IFNAME="enp66s0f5"

# -- Launch -----------------------------------------------------------------
NGPU="${1:-8}"
shift 1 2>/dev/null || true

echo "[run_wikitext2] Launching with ${NGPU} devices ..."
echo "[run_wikitext2] Extra flags: ${*:-<none>}"

torchrun \
    --nnodes=1 \
    --nproc_per_node="${NGPU}" \
    "${SCRIPT_DIR}/fsdp2_tp_llama2_main.py" \
    "$@"
