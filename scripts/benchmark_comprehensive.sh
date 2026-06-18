#!/bin/bash
# ──────────────────────────────────────────────────────────────
# ScaleTorch Comprehensive 8-NPU Benchmark
# Tests: DP, TP, PP, CP, SP, and mixed parallelism combos
# Usage: bash scripts/benchmark_comprehensive.sh [FILTER] [STEPS]
#   FILTER: regex to select configs (e.g. "0.6B" or "PP")
#   STEPS:  training steps per config (default: 10)
# ──────────────────────────────────────────────────────────────
set -e

source /home/jianzhnie/llmtuner/Ascend/CANN8.2.RC1/ascend-toolkit/set_env.sh
source /home/jianzhnie/llmtuner/Ascend/CANN8.2.RC1/nnal/atb/set_env.sh
source /home/jianzhnie/llmtuner/software/miniconda3/bin/activate vllm091

export FLASH_ATTEN=1 DTYPE=bfloat16 HCCL_CONNECT_TIMEOUT=7200 HCCL_EXEC_TIMEOUT=7200
export TASK_QUEUE_ENABLE=2 COMBINED_ENABLE=1 HCCL_BUFFSIZE=120 HCCL_OP_BASE_FFTS_MODE_ENABLE=1
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True' ASCEND_LAUNCH_BLOCKING=0

TORCHRUN=/home/jianzhnie/llmtuner/software/miniconda3/envs/vllm091/bin/torchrun
PYTHON=/home/jianzhnie/llmtuner/software/miniconda3/envs/vllm091/bin/python
MODELS=/home/jianzhnie/llmtuner/hfhub/models/Qwen
DATA=data/wikitext2
STEPS=${2:-10}
PORT=29600

cd /home/jianzhnie/llmtuner/llm/ScaleTorch

run() {
    local label="$1" model="$2" npus="$3" tp="$4" pp="$5" dp="$6" cp="$7"
    local bs="$8" ga="$9" seq="${10}" gc="${11}" sp="${12}" engine="${13}"

    # Filter support
    if [ -n "$FILTER" ] && ! echo "$label" | grep -qiE "$FILTER"; then
        return
    fi

    echo ""
    echo "========================================================================"
    echo "  [$label] $model | TP=$tp PP=$pp DP=$dp CP=$cp SP=$sp Engine=$engine"
    echo "  BS=$bs GA=$ga SEQ=$seq GC=$gc NPUs=$npus"
    echo "========================================================================"

    # Set SP env
    if [ "$sp" = "True" ]; then
        export SEQUENCE_PARALLEL=1
    else
        export SEQUENCE_PARALLEL=0
    fi

    rm -f performance_logs_0_*.json 2>/dev/null

    $TORCHRUN --nproc_per_node=$npus --master_addr=localhost --master_port=$PORT \
        tools/train.py \
        --model_name_or_path "$MODELS/$model" --dataset_name "$DATA" \
        --tensor_parallel_size $tp --pipeline_parallel_size $pp \
        --data_parallel_size $dp --context_parallel_size $cp \
        --pipeline_parallel_engine $engine \
        --micro_batch_size $bs --gradient_accumulation_steps $ga \
        --sequence_length $seq --gradient_checkpointing $gc \
        --learning_rate 3e-4 --use_fused_adam True --max_grad_norm 1.0 \
        --total_train_steps $STEPS --seed 42 --num_workers 0 --num_proc 1 \
        --log_interval 1 2>&1 | grep -E 'Step.*Loss.*MFU|completed|Error' | tail -15

    local logfile=$(ls -t performance_logs_0_*.json 2>/dev/null | head -1)
    if [ -n "$logfile" ]; then
        $PYTHON -c "
import json
d = json.load(open('$logfile'))
a = d.get('average_stats', {})
print(f'  >> avg_tok/s={a.get(\"average_tokens_per_second\",0):,.0f} | mem={a.get(\"gpu\",{}).get(\"avg_gpu_memory_allocated\",0):.0f}MB | iters={a.get(\"total_iterations\",0)} | time={a.get(\"total_time\",0):.1f}s')
"
    fi
    PORT=$((PORT+1))
    sleep 5
}

FILTER="${1:-}"

echo "============================================================"
echo " ScaleTorch Comprehensive 8-NPU Benchmark"
echo " Date: $(date '+%Y-%m-%d %H:%M')"
echo " Steps: $STEPS | Filter: ${FILTER:-ALL}"
echo "============================================================"

# ================================================================
#  Qwen3-0.6B (28 layers, ~1.5GB bf16)
# ================================================================
#         label                     model        npus tp pp dp cp bs ga seq   gc     sp     engine
run "0.6B-DP8"                     Qwen3-0.6B   8    1  1  8  1  2  2  2048  False  False  1f1b
run "0.6B-TP2-DP4"                 Qwen3-0.6B   8    2  1  4  1  2  1  2048  False  False  1f1b
run "0.6B-TP4-DP2"                 Qwen3-0.6B   8    4  1  2  1  2  1  2048  False  False  1f1b
run "0.6B-PP2-DP4"                 Qwen3-0.6B   8    1  2  4  1  2  2  2048  False  False  1f1b
run "0.6B-PP4-DP2"                 Qwen3-0.6B   8    1  4  2  1  2  2  2048  False  False  1f1b
run "0.6B-CP2-DP4"                 Qwen3-0.6B   8    1  1  4  2  1  1  4096  False  False  1f1b
run "0.6B-CP4-DP2"                 Qwen3-0.6B   8    1  1  2  4  1  1  8192  True   False  1f1b
run "0.6B-SP-TP2-DP4"              Qwen3-0.6B   8    2  1  4  1  2  1  2048  False  True   1f1b
run "0.6B-TP2-PP2-DP2"             Qwen3-0.6B   8    2  2  2  1  2  1  2048  False  False  1f1b
run "0.6B-TP2-CP2-DP2"             Qwen3-0.6B   8    2  1  2  2  1  1  4096  False  False  1f1b

# ================================================================
#  Qwen3-1.7B (28 layers, ~3.4GB bf16)
# ================================================================
run "1.7B-DP8-GC"                  Qwen3-1.7B   8    1  1  8  1  1  2  2048  True   False  1f1b
run "1.7B-TP2-DP4"                 Qwen3-1.7B   8    2  1  4  1  1  1  2048  False  False  1f1b
run "1.7B-TP4-DP2"                 Qwen3-1.7B   8    4  1  2  1  1  1  2048  False  False  1f1b
run "1.7B-PP2-DP4-GC"              Qwen3-1.7B   8    1  2  4  1  1  2  2048  True   False  1f1b
run "1.7B-PP4-DP2-GC"              Qwen3-1.7B   8    1  4  2  1  1  2  2048  True   False  1f1b
run "1.7B-CP2-DP4-GC"              Qwen3-1.7B   8    1  1  4  2  1  1  4096  True   False  1f1b
run "1.7B-CP4-DP2-GC"              Qwen3-1.7B   8    1  1  2  4  1  1  8192  True   False  1f1b
run "1.7B-SP-TP2-DP4"              Qwen3-1.7B   8    2  1  4  1  1  1  2048  False  True   1f1b
run "1.7B-TP2-PP2-DP2-GC"          Qwen3-1.7B   8    2  2  2  1  1  1  2048  True   False  1f1b
run "1.7B-TP2-CP2-DP2-GC"          Qwen3-1.7B   8    2  1  2  2  1  1  4096  True   False  1f1b

# ================================================================
#  Qwen3-4B (36 layers, ~8GB bf16)
# ================================================================
run "4B-DP8-GC"                    Qwen3-4B     8    1  1  8  1  1  1  2048  True   False  1f1b
run "4B-TP2-DP4-GC"                Qwen3-4B     8    2  1  4  1  1  1  2048  True   False  1f1b
run "4B-TP4-DP2"                   Qwen3-4B     8    4  1  2  1  1  1  2048  False  False  1f1b
run "4B-PP2-DP4-GC"                Qwen3-4B     8    1  2  4  1  1  1  2048  True   False  1f1b
run "4B-PP4-DP2-GC"                Qwen3-4B     8    1  4  2  1  1  1  2048  True   False  1f1b
run "4B-CP2-DP4-GC"                Qwen3-4B     8    1  1  4  2  1  1  4096  True   False  1f1b
run "4B-SP-TP2-DP4-GC"             Qwen3-4B     8    2  1  4  1  1  1  2048  True   True   1f1b
run "4B-TP2-PP2-DP2-GC"            Qwen3-4B     8    2  2  2  1  1  1  2048  True   False  1f1b
run "4B-TP2-CP2-DP2-GC"            Qwen3-4B     8    2  1  2  2  1  1  4096  True   False  1f1b
run "4B-TP2-PP2-CP2-DP1-GC"        Qwen3-4B     8    2  2  1  2  1  1  4096  True   False  1f1b

# ================================================================
#  Qwen3-8B (36 layers, ~16GB bf16)
# ================================================================
run "8B-TP2-DP4-GC"                Qwen3-8B     8    2  1  4  1  1  1  2048  True   False  1f1b
run "8B-TP4-DP2-GC"                Qwen3-8B     8    4  1  2  1  1  1  2048  True   False  1f1b
run "8B-TP8"                       Qwen3-8B     8    8  1  1  1  1  1  2048  False  False  1f1b
run "8B-PP2-DP4-GC"                Qwen3-8B     8    1  2  4  1  1  1  2048  True   False  1f1b
run "8B-PP4-DP2-GC"                Qwen3-8B     8    1  4  2  1  1  1  2048  True   False  1f1b
run "8B-CP2-DP4-GC"                Qwen3-8B     8    1  1  4  2  1  1  4096  True   False  1f1b
run "8B-SP-TP2-DP4-GC"             Qwen3-8B     8    2  1  4  1  1  1  2048  True   True   1f1b
run "8B-TP2-PP2-DP2-GC"            Qwen3-8B     8    2  2  2  1  1  1  2048  True   False  1f1b
run "8B-TP2-CP2-DP2-GC"            Qwen3-8B     8    2  1  2  2  1  1  4096  True   False  1f1b
run "8B-TP2-PP2-CP2-DP1-GC"        Qwen3-8B     8    2  2  1  2  1  1  4096  True   False  1f1b
run "8B-TP4-PP2-DP1-GC"            Qwen3-8B     8    4  2  1  1  1  1  2048  True   False  1f1b

echo ""
echo "============================================================"
echo " ALL DONE"
echo "============================================================"
