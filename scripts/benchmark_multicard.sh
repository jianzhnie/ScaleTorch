#!/bin/bash
# Quick multi-card benchmark runner with JSON-based metric extraction
set -e
source /home/jianzhnie/llmtuner/Ascend/CANN8.3.RC2/ascend-toolkit/set_env.sh
export FLASH_ATTEN=1 DTYPE=bfloat16 HCCL_CONNECT_TIMEOUT=7200 HCCL_EXEC_TIMEOUT=7200
export TASK_QUEUE_ENABLE=2 COMBINED_ENABLE=1 HCCL_BUFFSIZE=120 HCCL_OP_BASE_FFTS_MODE_ENABLE=1
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True' ASCEND_LAUNCH_BLOCKING=0

TORCHRUN=/home/jianzhnie/llmtuner/software/miniconda3/envs/rlhf/bin/torchrun
PYTHON=/home/jianzhnie/llmtuner/software/miniconda3/envs/rlhf/bin/python
MODELS=/home/jianzhnie/llmtuner/hfhub/models/Qwen
DATA=data/wikitext2
STEPS=10
PORT=29640

run() {
    local label="$1" model="$2" npus="$3" tp="$4" pp="$5" dp="$6" cp="$7"
    local bs="$8" ga="$9" seq="${10}" gc="${11}"
    echo "=== $label ==="
    rm -f performance_logs_0_*.json 2>/dev/null
    $TORCHRUN --nproc_per_node=$npus --master_addr=localhost --master_port=$PORT \
        train.py \
        --model_name_or_path "$MODELS/$model" --dataset_name "$DATA" \
        --tensor_parallel_size $tp --pipeline_parallel_size $pp \
        --data_parallel_size $dp --context_parallel_size $cp \
        --micro_batch_size $bs --gradient_accumulation_steps $ga \
        --sequence_length $seq --gradient_checkpointing $gc \
        --learning_rate 3e-4 --use_fused_adam True --max_grad_norm 1.0 \
        --total_train_steps $STEPS --seed 42 --num_workers 0 --num_proc 1 \
        --log_interval 1 2>&1 | grep -E 'Step.*Loss.*MFU|completed|Error' | tail -12
    # Extract from JSON
    local logfile=$(ls -t performance_logs_0_*.json 2>/dev/null | head -1)
    if [ -n "$logfile" ]; then
        $PYTHON -c "import json; d=json.load(open('$logfile')); a=d.get('average_stats',{}); print(f'  >> avg_tok/s={a.get(\"average_tokens_per_second\",0):,.0f} | mem={a.get(\"gpu\",{}).get(\"avg_gpu_memory_allocated\",0):.0f}MB | iters={a.get(\"total_iterations\",0)} | time={a.get(\"total_time\",0):.1f}s')"
    fi
    PORT=$((PORT+1))
    sleep 3
}

cd /home/jianzhnie/llmtuner/llm/ScaleTorch

# Multi-card configs
#         label                     model        npus tp pp dp cp bs ga seq   gc
run "0.6B-8NPU-DP8"               Qwen3-0.6B   8    1  1  8  1  2  2  2048  False
run "0.6B-8NPU-TP2-DP4"           Qwen3-0.6B   8    2  1  4  1  2  1  2048  False
run "1.7B-8NPU-DP8-GC"            Qwen3-1.7B   8    1  1  8  1  1  2  2048  True
run "1.7B-8NPU-TP2-DP4"           Qwen3-1.7B   8    2  1  4  1  1  1  2048  False
run "4B-8NPU-DP8-GC"              Qwen3-4B     8    1  1  8  1  1  1  2048  True
run "4B-8NPU-TP2-DP4-GC"          Qwen3-4B     8    2  1  4  1  1  1  2048  True
run "4B-8NPU-TP4-DP2"             Qwen3-4B     8    4  1  2  1  1  1  2048  False
run "8B-8NPU-TP2-DP4-GC"          Qwen3-8B     8    2  1  4  1  1  1  2048  True
run "8B-8NPU-TP4-DP2-GC"          Qwen3-8B     8    4  1  2  1  1  1  2048  True

echo "=== ALL DONE ==="
