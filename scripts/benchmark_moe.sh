#!/bin/bash
# Qwen3-30B-A3B MoE benchmark runner
set -e
source /home/jianzhnie/llmtuner/Ascend/CANN8.2.RC1/ascend-toolkit/set_env.sh
source /home/jianzhnie/llmtuner/Ascend/CANN8.2.RC1/nnal/atb/set_env.sh
source /home/jianzhnie/llmtuner/software/miniconda3/bin/activate vllm091

export FLASH_ATTEN=1 DTYPE=bfloat16 HCCL_CONNECT_TIMEOUT=7200 HCCL_EXEC_TIMEOUT=7200
export TASK_QUEUE_ENABLE=2 COMBINED_ENABLE=1 HCCL_BUFFSIZE=120 HCCL_OP_BASE_FFTS_MODE_ENABLE=1
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True' ASCEND_LAUNCH_BLOCKING=0

TORCHRUN=/home/jianzhnie/llmtuner/software/miniconda3/envs/vllm091/bin/torchrun
MODEL=/home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen3-30B-A3B
DATA=data/wikitext2
STEPS=10
PORT=29700
cd /home/jianzhnie/llmtuner/llm/ScaleTorch

run() {
    local label="$1" tp="$2" ep="$3" dp="$4" bs="$5" ga="$6" seq="$7" gc="$8" sp="$9"
    echo ""
    echo "================================================================"
    echo "  [$label] TP=$tp EP=$ep DP=$dp BS=$bs GA=$ga SEQ=$seq GC=$gc SP=$sp"
    echo "================================================================"
    if [ "$sp" = "True" ]; then export SEQUENCE_PARALLEL=1; else export SEQUENCE_PARALLEL=0; fi
    timeout 1800 $TORCHRUN --nproc_per_node=8 --master_addr=localhost --master_port=$PORT \
        tools/train.py \
        --model_name_or_path $MODEL --dataset_name $DATA \
        --tensor_parallel_size $tp --pipeline_parallel_size 1 \
        --data_parallel_size $dp --context_parallel_size 1 \
        --expert_parallel_size $ep \
        --micro_batch_size $bs --gradient_accumulation_steps $ga \
        --sequence_length $seq --gradient_checkpointing $gc \
        --learning_rate 3e-4 --use_fused_adam True --max_grad_norm 1.0 \
        --total_train_steps $STEPS --seed 42 --num_workers 0 --num_proc 1 \
        --log_interval 1 2>&1 | grep -E 'Step.*Loss|Number|Fatal|Error|OOM|memory' | tail -12
    local rc=$?
    if [ $rc -eq 124 ]; then echo "  TIMEOUT (1800s)"; fi
    if [ $rc -ne 0 ]; then echo "  FAILED (rc=$rc)"; fi
    PORT=$((PORT+1))
    sleep 5
}

#         label                    tp ep dp bs ga  seq  gc     sp
run "EP2-TP4-BS2-GC"               4  2  1  2  1  2048 True   False
run "EP2-TP4-SEQ4K"                 4  2  1  1  1  4096 False  False
run "EP2-TP4-SEQ4K-GC"             4  2  1  1  1  4096 True   False
run "SP-EP2-TP4-GC"                4  2  1  1  1  2048 True   True
run "EP2-TP4-GA2-GC"               4  2  1  1  2  2048 True   False
run "EP4-TP2"                       2  4  1  1  1  2048 False  False

echo ""
echo "=== ALL DONE ==="
