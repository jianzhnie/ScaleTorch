#!/bin/bash
set -e

# ScaleTorch NPU Optimized Training Launch Script
# Optimized for Ascend 910 (64GB HBM) with CANN 9.0 + PyTorch 2.12
#
# Usage: bash scripts/run_npu.sh [NUM_NPUS] [MODEL_PATH] [DATASET] [MODE]
#
# MODE options:
#   max_mfu    — SEQ=16384, BS=1, GC → MFU=60%, Mem=29GB (maximize compute utilization)
#   max_speed  — SEQ=2048,  BS=4, GA=2, no-GC → MFU=44%, Mem=37GB (max tokens/s)
#   balanced   — SEQ=8192,  BS=2, GC → MFU=50%, Mem=29GB (good tradeoff)
#   min_mem    — SEQ=2048,  BS=4, GC → MFU=36%, Mem=16GB (minimum memory)
#
# Measured MFU sweep (Qwen3-0.6B, Ascend 910, bf16):
#   BS=2  SEQ=2048  no-GC  → MFU=36.4%, 15.7K tok/s, 19.7GB
#   BS=4  SEQ=2048  no-GC  → MFU=43.2%, 18.4K tok/s, 34.3GB
#   BS=4  GA=2 SEQ=2048    → MFU=43.9%, 19.0K tok/s, 37.4GB  ← max throughput
#   BS=4  SEQ=2048  GC     → MFU=35.9%, 15.5K tok/s, 16.4GB  ← min memory
#   BS=4  SEQ=4096  GC     → MFU=42.2%, 14.7K tok/s, 29.3GB
#   BS=2  SEQ=8192  GC     → MFU=49.7%, 12.5K tok/s, 29.3GB
#   BS=1  SEQ=16384 GC     → MFU=60.1%, 9.7K  tok/s, 29.3GB  ← max MFU

NUM_NPUS=${1:-8}
MODEL_PATH=${2:-"/workspace/models/qwen3"}
DATASET=${3:-"/workspace/ScaleTorch/data/wikitext2"}
MODE=${4:-"balanced"}

# === NPU Performance Optimizations ===
export DTYPE=bfloat16
export FLASH_ATTEN=1
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export TASK_QUEUE_ENABLE=2
export COMBINED_ENABLE=1
export HCCL_BUFFSIZE=120
export HCCL_OP_BASE_FFTS_MODE_ENABLE=1
export ASCEND_LAUNCH_BLOCKING=0
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"

# Set config based on mode
case "$MODE" in
  max_mfu)
    MICRO_BS=1; SEQ_LEN=16384; GRAD_ACCUM=1; GC=True
    ;;
  max_speed)
    MICRO_BS=4; SEQ_LEN=2048;  GRAD_ACCUM=2; GC=False
    ;;
  min_mem)
    MICRO_BS=4; SEQ_LEN=2048;  GRAD_ACCUM=1; GC=True
    ;;
  balanced|*)
    MICRO_BS=2; SEQ_LEN=8192;  GRAD_ACCUM=1; GC=True
    ;;
esac

TP_SIZE=1; PP_SIZE=1; DP_SIZE=${NUM_NPUS}; CP_SIZE=1
GLOBAL_TOK=$((MICRO_BS * SEQ_LEN * GRAD_ACCUM * DP_SIZE))

echo "============================================"
echo " ScaleTorch NPU Training  [mode: $MODE]"
echo " NPUs: ${NUM_NPUS}, DP=${DP_SIZE}"
echo " BS=${MICRO_BS} x GA=${GRAD_ACCUM} x SEQ=${SEQ_LEN}"
echo " GC=${GC}, Global tokens/step=${GLOBAL_TOK}"
echo "============================================"

cd /workspace/ScaleTorch

torchrun \
    --nproc_per_node=${NUM_NPUS} \
    --master_addr=localhost \
    --master_port=29500 \
    tools/train.py \
    --model_name_or_path "${MODEL_PATH}" \
    --dataset_name "${DATASET}" \
    --tensor_parallel_size ${TP_SIZE} \
    --pipeline_parallel_size ${PP_SIZE} \
    --data_parallel_size ${DP_SIZE} \
    --context_parallel_size ${CP_SIZE} \
    --micro_batch_size ${MICRO_BS} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --sequence_length ${SEQ_LEN} \
    --gradient_checkpointing ${GC} \
    --learning_rate 3e-4 \
    --use_fused_adam True \
    --max_grad_norm 1.0 \
    --lr_scheduler_type cosine \
    --warmup_steps 100 \
    --save_frequency 500 \
    --log_interval 10 \
    --seed 42 \
    --num_workers 0 \
    --num_proc 1
