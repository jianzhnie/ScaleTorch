#!/usr/bin/env python3
"""ScaleTorch NPU Benchmark Suite — runs multiple model/parallelism configs."""

import json
import os
import re
import subprocess
import sys
import time

PYTHON = "/home/jianzhnie/llmtuner/software/miniconda3/envs/rlhf/bin/python"
TORCHRUN = "/home/jianzhnie/llmtuner/software/miniconda3/envs/rlhf/bin/torchrun"
CANN_ENV = "source /home/jianzhnie/llmtuner/Ascend/CANN8.3.RC2/ascend-toolkit/set_env.sh"
MODEL_ROOT = "/home/jianzhnie/llmtuner/hfhub/models/Qwen"
DATASET = "/home/jianzhnie/llmtuner/llm/ScaleTorch/data/wikitext2"
TRAIN_SCRIPT = "train.py"
STEPS = 10
WARMUP_STEPS = 2  # skip first N steps for avg
BASE_PORT = 29600

NPU_ENV = {
    "FLASH_ATTEN": "1",
    "DTYPE": "bfloat16",
    "HCCL_CONNECT_TIMEOUT": "7200",
    "HCCL_EXEC_TIMEOUT": "7200",
    "TASK_QUEUE_ENABLE": "2",
    "COMBINED_ENABLE": "1",
    "HCCL_BUFFSIZE": "120",
    "HCCL_OP_BASE_FFTS_MODE_ENABLE": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "ASCEND_LAUNCH_BLOCKING": "0",
}

CONFIGS = [
    # (model, tp, pp, dp, cp, bs, ga, seq, gc, label)
    # === Qwen3-0.6B (751M params, ~1.5GB bf16) ===
    ("Qwen3-0.6B", 1, 1, 1, 1,  2, 1, 2048, False, "0.6B-1NPU-DP1"),
    ("Qwen3-0.6B", 1, 1, 1, 1,  1, 1, 8192, True,  "0.6B-1NPU-SEQ8K-GC"),
    ("Qwen3-0.6B", 1, 1, 1, 1,  1, 1, 16384, True,  "0.6B-1NPU-SEQ16K-GC"),
    ("Qwen3-0.6B", 1, 1, 8, 1,  2, 2, 2048, False, "0.6B-8NPU-DP8"),
    ("Qwen3-0.6B", 2, 1, 4, 1,  2, 1, 2048, False, "0.6B-8NPU-TP2-DP4"),
    # === Qwen3-1.7B (1.7B params, ~3.4GB bf16) ===
    ("Qwen3-1.7B", 1, 1, 1, 1,  1, 1, 2048, False, "1.7B-1NPU-DP1"),
    ("Qwen3-1.7B", 1, 1, 1, 1,  1, 1, 2048, True,  "1.7B-1NPU-GC"),
    ("Qwen3-1.7B", 1, 1, 1, 1,  1, 1, 8192, True,  "1.7B-1NPU-SEQ8K-GC"),
    ("Qwen3-1.7B", 1, 1, 8, 1,  1, 2, 2048, True,  "1.7B-8NPU-DP8-GC"),
    ("Qwen3-1.7B", 2, 1, 4, 1,  1, 1, 2048, False, "1.7B-8NPU-TP2-DP4"),
    ("Qwen3-1.7B", 1, 2, 4, 1,  1, 1, 2048, True,  "1.7B-8NPU-PP2-DP4-GC"),
    # === Qwen3-4B (4B params, ~8GB bf16) ===
    ("Qwen3-4B", 1, 1, 1, 1,  1, 1, 2048, True,  "4B-1NPU-GC"),
    ("Qwen3-4B", 1, 1, 8, 1,  1, 1, 2048, True,  "4B-8NPU-DP8-GC"),
    ("Qwen3-4B", 2, 1, 4, 1,  1, 1, 2048, True,  "4B-8NPU-TP2-DP4-GC"),
    ("Qwen3-4B", 4, 1, 2, 1,  1, 1, 2048, False, "4B-8NPU-TP4-DP2"),
    ("Qwen3-4B", 1, 2, 4, 1,  1, 1, 2048, True,  "4B-8NPU-PP2-DP4-GC"),
    # === Qwen3-8B (8B params, ~16GB bf16) ===
    ("Qwen3-8B", 1, 1, 1, 1,  1, 1, 2048, True,  "8B-1NPU-GC"),
    ("Qwen3-8B", 2, 1, 4, 1,  1, 1, 2048, True,  "8B-8NPU-TP2-DP4-GC"),
    ("Qwen3-8B", 4, 1, 2, 1,  1, 1, 2048, True,  "8B-8NPU-TP4-DP2-GC"),
    ("Qwen3-8B", 1, 2, 4, 1,  1, 1, 2048, True,  "8B-8NPU-PP2-DP4-GC"),
    ("Qwen3-8B", 2, 2, 2, 1,  1, 1, 2048, True,  "8B-8NPU-TP2-PP2-DP2-GC"),
]


def run_config(cfg, port):
    model, tp, pp, dp, cp, bs, ga, seq, gc, label = cfg
    nproc = tp * pp * dp * cp
    model_path = os.path.join(MODEL_ROOT, model)

    cmd = (
        f"{CANN_ENV} && "
        + " && ".join(f"export {k}={v}" for k, v in NPU_ENV.items())
        + f" && {TORCHRUN}"
        f" --nproc_per_node={nproc}"
        f" --master_addr=localhost"
        f" --master_port={port}"
        f" {TRAIN_SCRIPT}"
        f" --model_name_or_path {model_path}"
        f" --dataset_name {DATASET}"
        f" --tensor_parallel_size {tp}"
        f" --pipeline_parallel_size {pp}"
        f" --data_parallel_size {dp}"
        f" --context_parallel_size {cp}"
        f" --micro_batch_size {bs}"
        f" --gradient_accumulation_steps {ga}"
        f" --sequence_length {seq}"
        f" --gradient_checkpointing {gc}"
        f" --learning_rate 3e-4"
        f" --use_fused_adam True"
        f" --max_grad_norm 1.0"
        f" --total_train_steps {STEPS}"
        f" --seed 42 --num_workers 0 --num_proc 1"
        f" --log_interval 1"
    )

    print(f"\n{'='*60}")
    print(f"  [{label}] {model} | TP={tp} PP={pp} DP={dp} CP={cp}")
    print(f"  BS={bs} GA={ga} SEQ={seq} GC={gc} NPUs={nproc}")
    print(f"{'='*60}")

    start = time.time()
    try:
        result = subprocess.run(
            ["bash", "-lc", cmd],
            capture_output=True, text=True, timeout=600,
            cwd="/home/jianzhnie/llmtuner/llm/ScaleTorch",
        )
        output = result.stdout + result.stderr
        elapsed = time.time() - start
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after 600s")
        return {"label": label, "status": "TIMEOUT"}
    except Exception as e:
        print(f"  ERROR: {e}")
        return {"label": label, "status": f"ERROR: {e}"}

    if result.returncode != 0:
        all_output = output
        err_lines = [l for l in all_output.split('\n') if 'Error' in l or 'error' in l or 'OOM' in l.upper() or 'out of memory' in l.lower()]
        err_msg = err_lines[0][:150] if err_lines else "unknown error"
        print(f"  FAILED (rc={result.returncode}): {err_msg}")
        return {"label": label, "status": f"FAILED: {err_msg}"}

    # Parse metrics from output lines like: [rank 0] | Step: 5 | Loss: ... | MFU: 55.75%
    steps_data = []
    for line in output.split('\n'):
        m = re.search(
            r'Step:\s*(\d+)\s*\|.*Loss:\s*([\d.]+).*Tokens/s:\s*([\d.]+K?).*'
            r'Tokens/s/GPU:\s*([\d.]+K?).*MFU:\s*([\d.]+)%.*Memory:\s*([\d.]+)GB',
            line
        )
        if m:
            step = int(m.group(1))
            loss = float(m.group(2))
            tok_s_raw = m.group(3)
            tok_s_gpu_raw = m.group(4)
            mfu = float(m.group(5))
            mem = float(m.group(6))
            tok_s = float(tok_s_raw.replace('K', '')) * (1000 if 'K' in tok_s_raw else 1)
            tok_s_gpu = float(tok_s_gpu_raw.replace('K', '')) * (1000 if 'K' in tok_s_gpu_raw else 1)
            steps_data.append({
                "step": step, "loss": loss,
                "tokens_per_sec": tok_s,
                "tokens_per_sec_gpu": tok_s_gpu,
                "mfu": mfu, "memory_gb": mem,
            })

    if not steps_data:
        print(f"  No metrics parsed from output")
        return {"label": label, "status": "NO_METRICS"}

    # Average after warmup
    steady = [s for s in steps_data if s["step"] > WARMUP_STEPS]
    if not steady:
        steady = steps_data

    avg_tok = sum(s["tokens_per_sec"] for s in steady) / len(steady)
    avg_tok_gpu = sum(s["tokens_per_sec_gpu"] for s in steady) / len(steady)
    avg_mfu = sum(s["mfu"] for s in steady) / len(steady)
    avg_mem = sum(s["memory_gb"] for s in steady) / len(steady)
    final_loss = steps_data[-1]["loss"]

    result_dict = {
        "label": label, "model": model, "status": "OK",
        "tp": tp, "pp": pp, "dp": dp, "cp": cp,
        "npus": nproc, "bs": bs, "ga": ga, "seq": seq, "gc": gc,
        "tokens_per_sec": round(avg_tok), "tokens_per_sec_gpu": round(avg_tok_gpu),
        "mfu": round(avg_mfu, 1), "memory_gb": round(avg_mem, 1),
        "final_loss": round(final_loss, 4), "elapsed": round(elapsed, 1),
    }
    print(f"  OK | Tok/s: {avg_tok:,.0f} | Tok/s/GPU: {avg_tok_gpu:,.0f} "
          f"| MFU: {avg_mfu:.1f}% | Mem: {avg_mem:.1f}GB | Loss: {final_loss:.4f}")
    return result_dict


def main():
    results = []
    port = BASE_PORT
    for i, cfg in enumerate(CONFIGS):
        r = run_config(cfg, port)
        results.append(r)
        port += 1
        # Save incremental results
        with open("benchmark_results.json", "w") as f:
            json.dump(results, f, indent=2)
        time.sleep(3)

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)

    # Print summary table
    ok = [r for r in results if r.get("status") == "OK"]
    if ok:
        print(f"\n{'Label':<30} {'NPUs':>5} {'Tok/s':>8} {'Tok/s/GPU':>10} {'MFU%':>6} {'Mem(GB)':>8}")
        print("-" * 75)
        for r in ok:
            print(f"{r['label']:<30} {r['npus']:>5} {r['tokens_per_sec']:>8,} "
                  f"{r['tokens_per_sec_gpu']:>10,} {r['mfu']:>5.1f}% {r['memory_gb']:>7.1f}")

    failed = [r for r in results if r.get("status") != "OK"]
    if failed:
        print(f"\nFailed configs ({len(failed)}):")
        for r in failed:
            print(f"  {r['label']}: {r['status']}")

    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to benchmark_results.json")


if __name__ == "__main__":
    main()
