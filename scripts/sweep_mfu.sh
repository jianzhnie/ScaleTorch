#!/bin/bash
set -eo pipefail
pip install colorama psutil datasets transformers safetensors --quiet 2>/dev/null
pip install -e /workspace/ScaleTorch --quiet 2>/dev/null

export FLASH_ATTEN=1
export DTYPE=bfloat16
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"

MODEL=/workspace/models/qwen3
DATA=/workspace/ScaleTorch/data/wikitext2
PORT=29700

run_config() {
    local label="$1"; local port="$2"; shift 2
    echo "=== $label ==="
    torchrun --nproc_per_node=1 --master_addr=localhost --master_port=$port \
        tools/train.py \
        --model_name_or_path "$MODEL" \
        --dataset_name "$DATA" \
        --tensor_parallel_size 1 --pipeline_parallel_size 1 \
        --data_parallel_size 1 --context_parallel_size 1 \
        --total_train_steps 8 --log_interval 1 \
        --seed 42 --num_workers 0 --num_proc 1 \
        --learning_rate 1e-4 --max_grad_norm 1.0 --use_fused_adam True \
        "$@" 2>&1 | grep 'Tokens/s\|MFU\|Memory' | tail -3 || true
    echo "---"
    sleep 2
}

cd /workspace/ScaleTorch

# --- Batch size sweep (no gradient checkpointing) ---
run_config "BS=2 GA=1 SEQ=2048 no-GC"  $((PORT+0))  --micro_batch_size 2  --gradient_accumulation_steps 1 --sequence_length 2048
run_config "BS=4 GA=1 SEQ=2048 no-GC"  $((PORT+1))  --micro_batch_size 4  --gradient_accumulation_steps 1 --sequence_length 2048
run_config "BS=4 GA=2 SEQ=2048 no-GC"  $((PORT+2))  --micro_batch_size 4  --gradient_accumulation_steps 2 --sequence_length 2048
run_config "BS=8 GA=1 SEQ=2048 no-GC"  $((PORT+3))  --micro_batch_size 8  --gradient_accumulation_steps 1 --sequence_length 2048

# --- With gradient checkpointing ---
run_config "BS=4 GA=1 SEQ=2048 GC"     $((PORT+4))  --micro_batch_size 4  --gradient_accumulation_steps 1 --sequence_length 2048 --gradient_checkpointing True
run_config "BS=8 GA=1 SEQ=2048 GC"     $((PORT+5))  --micro_batch_size 8  --gradient_accumulation_steps 1 --sequence_length 2048 --gradient_checkpointing True
run_config "BS=8 GA=2 SEQ=2048 GC"     $((PORT+6))  --micro_batch_size 8  --gradient_accumulation_steps 2 --sequence_length 2048 --gradient_checkpointing True

# --- Longer sequences (better attention compute ratio) ---
run_config "BS=4 GA=1 SEQ=4096 GC"     $((PORT+7))  --micro_batch_size 4  --gradient_accumulation_steps 1 --sequence_length 4096 --gradient_checkpointing True
run_config "BS=4 GA=1 SEQ=8192 GC"     $((PORT+8))  --micro_batch_size 4  --gradient_accumulation_steps 1 --sequence_length 8192 --gradient_checkpointing True

echo "=== SWEEP COMPLETE ==="
