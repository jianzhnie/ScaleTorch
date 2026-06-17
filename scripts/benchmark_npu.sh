#!/bin/bash
set -e

# Install dependencies
pip install colorama psutil datasets transformers safetensors --quiet 2>/dev/null
pip install -e /workspace/ScaleTorch --quiet 2>/dev/null

# NPU Performance Optimizations
export FLASH_ATTEN=1
export DTYPE=bfloat16
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export TASK_QUEUE_ENABLE=2
export COMBINED_ENABLE=1
export HCCL_BUFFSIZE=120
export HCCL_OP_BASE_FFTS_MODE_ENABLE=1
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export ASCEND_LAUNCH_BLOCKING=0

NUM_NPUS=${1:-8}
MODEL_PATH=${2:-"/workspace/models/qwen3"}
DATASET=${3:-"/workspace/ScaleTorch/data/wikitext2"}
MICRO_BS=${4:-4}
SEQ_LEN=${5:-2048}
GRAD_ACCUM=${6:-2}
STEPS=${7:-20}

# DP-only config for maximum throughput on single node
TP_SIZE=1
PP_SIZE=1
DP_SIZE=${NUM_NPUS}
CP_SIZE=1

echo "============================================"
echo " ScaleTorch NPU Benchmark"
echo " NPUs: ${NUM_NPUS}, TP=${TP_SIZE} PP=${PP_SIZE} DP=${DP_SIZE}"
echo " Micro BS: ${MICRO_BS}, Seq Len: ${SEQ_LEN}"
echo " Grad Accum: ${GRAD_ACCUM}, Steps: ${STEPS}"
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
    --learning_rate 3e-4 \
    --use_fused_adam True \
    --total_train_steps ${STEPS} \
    --max_grad_norm 1.0 \
    --seed 42 \
    --num_workers 0 \
    --num_proc 1 \
    --log_interval 1
