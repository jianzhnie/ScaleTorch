#!/bin/bash
# DeviceMesh examples launcher on Ascend NPU.
#
# Sources set_env.sh for CANN toolkit + conda environment.
#
# Override defaults via env vars:
#   NGPU  – number of NPUs per node (default: 8)
#
# Available scripts (uncomment the one you want to run):
#   manual_process_group.py     – manual dist.new_group() approach
#   device_mesh_api.py          – init_device_mesh() API approach
#   fsdp_dp_demo.py                – HSDP with fully_shard
#   tensor_parallel_demo.py     – Tensor Parallel (Megatron-LM SPMD)
#   sequence_parallel_demo.py   – Sequence Parallel (SP)
#   fsdp_tp_demo.py             – 2D: FSDP + TP on Llama-style transformer

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Source CANN toolkit + conda environment (set_env.sh does both).
# Temporarily relax -u because ~/.bashrc references unbound variables
# (BASHRCSOURCED) that trigger "unbound variable" under `set -u`.
SET_ENV="${REPO_ROOT}/set_env.sh"
if [ -f "${SET_ENV}" ]; then
    echo "[run.sh] Sourcing ${SET_ENV} ..."
    set +u
    source "${SET_ENV}"
    set -u
else
    echo "[run.sh] ERROR: ${SET_ENV} not found" >&2
    exit 1
fi

# --- NPU-specific environment variables (same pattern as run_train.sh) ---
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export HCCL_CONNECT_TIMEOUT=3600

# Force IPv4 — hostname has no AAAA record, causing HCCL init to hang
# on repeated IPv6 resolution failures (gai error: -2).
export NCCL_SOCKET_FAMILY=AF_INET

# Bind HCCL and gloo to the detected NIC (same pattern as run_train_multinodes.sh).
export HCCL_SOCKET_IFNAME="enp66s0f5"
export GLOO_SOCKET_IFNAME="enp66s0f5"

NGPU="${NGPU:-8}"

# echo "[run.sh] Launching manual_process_group.py with ${NGPU} NPUs ..."
# torchrun \
#     --nproc_per_node="${NGPU}" \
#     "${SCRIPT_DIR}/manual_process_group.py"

# echo "[run.sh] Launching device_mesh_api.py with ${NGPU} NPUs ..."
# torchrun \
#     --nproc_per_node="${NGPU}" \
#     "${SCRIPT_DIR}/device_mesh_api.py"

# echo "[run.sh] Launching fsdp_dp_demo.py with ${NGPU} NPUs ..."
# torchrun \
#     --nproc_per_node="${NGPU}" \
#     "${SCRIPT_DIR}/fsdp_dp_demo.py"

# echo "[run.sh] Launching tensor_parallel_demo.py with ${NGPU} NPUs ..."
# torchrun \
#     --nproc_per_node="${NGPU}" \
#     "${SCRIPT_DIR}/tensor_parallel_demo.py"

# echo "[run.sh] Launching sequence_parallel_demo.py with ${NGPU} NPUs ..."
# torchrun \
#     --nproc_per_node="${NGPU}" \
#     "${SCRIPT_DIR}/sequence_parallel_demo.py"

echo "[run.sh] Launching fsdp_tp_demo.py with ${NGPU} NPUs ..."
torchrun \
    --nproc_per_node="${NGPU}" \
    "${SCRIPT_DIR}/fsdp_tp_demo.py"


