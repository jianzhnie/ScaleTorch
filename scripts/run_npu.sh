#!/bin/bash
set -e

# ScaleTorch NPU Optimized Training Launch Script
# Optimized for Ascend 910 (64GB HBM) with CANN 9.0 + PyTorch 2.12
#
# Usage: bash scripts/run_npu.sh [NUM_NPUS] [MODEL_PATH] [DATASET]
#
# Optimization Results (single NPU, 663M model):
#   Speed max:  BS=6, no GC, fused → 24,759 tokens/s (44.8GB)
#   Memory min: BS=4, GC         → 19,069 tokens/s (19.8GB)
#   Balanced:   BS=8, GC, fused  → 21,840 tokens/s (35.2GB)

NUM_NPUS=${1:-8}
MODEL_PATH=${2:-"/workspace/models/qwen3"}
DATASET=${3:-"/workspace/ScaleTorch/data/wikitext2"}

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
# Expandable segments reduces fragmentation, crucial for large models
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"

# Parallelism configuration
TP_SIZE=1
PP_SIZE=1
DP_SIZE=${NUM_NPUS}
CP_SIZE=1

# Training config (optimized for speed + memory balance)
MICRO_BS=8              # Optimal with gradient checkpointing
SEQ_LEN=2048
GRAD_ACCUM=2            # Effective BS per GPU = 16
MAX_GRAD_NORM=1.0

echo "============================================"
echo " ScaleTorch NPU Optimized Training"
echo " NPUs: ${NUM_NPUS}, DP=${DP_SIZE}"
echo " Micro BS: ${MICRO_BS}, Seq: ${SEQ_LEN}"
echo " Grad Accum: ${GRAD_ACCUM}"
echo " Global BS tokens: $((MICRO_BS * SEQ_LEN * GRAD_ACCUM * DP_SIZE))"
echo "============================================"

cd /workspace/ScaleTorch

torchrun \
    --nproc_per_node=${NUM_NPUS} \
    --master_addr=localhost \
    --master_port=29500 \
    train.py \
    --model_name_or_path "${MODEL_PATH}" \
    --dataset_name "${DATASET}" \
    --tensor_parallel_size ${TP_SIZE} \
    --pipeline_parallel_size ${PP_SIZE} \
    --data_parallel_size ${DP_SIZE} \
    --context_parallel_size ${CP_SIZE} \
    --micro_batch_size ${MICRO_BS} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --sequence_length ${SEQ_LEN} \
    --learning_rate 3e-4 \
    --use_fused_adam True \
    --gradient_checkpointing True \
    --max_grad_norm ${MAX_GRAD_NORM} \
    --lr_scheduler_type cosine \
    --warmup_steps 100 \
    --save_frequency 500 \
    --log_interval 1 \
    --seed 42 \
    --num_workers 0 \
    --num_proc 1
