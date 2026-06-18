#!/usr/bin/env python3
"""ScaleTorch Comprehensive 8-NPU Benchmark Suite.

Tests all parallelism strategies: DP, TP, PP, CP, SP, and mixed combos.
Usage:
    source /home/jianzhnie/llmtuner/llm/ScaleTorch/set_env.sh
    python scripts/benchmark_comprehensive.py [--filter PATTERN] [--steps N] [--seq SEQ]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

CANN_ENV = "source /home/jianzhnie/llmtuner/Ascend/CANN8.2.RC1/ascend-toolkit/set_env.sh"
CONDA_ENV = "source /home/jianzhnie/llmtuner/software/miniconda3/bin/activate vllm091"
TORCHRUN = "/home/jianzhnie/llmtuner/software/miniconda3/envs/vllm091/bin/torchrun"
PYTHON = "/home/jianzhnie/llmtuner/software/miniconda3/envs/vllm091/bin/python"
MODEL_ROOT = "/home/jianzhnie/llmtuner/hfhub/models/Qwen"
DATASET = "/home/jianzhnie/llmtuner/llm/ScaleTorch/data/wikitext2"
TRAIN_SCRIPT = "tools/train.py"
WORKDIR = "/home/jianzhnie/llmtuner/llm/ScaleTorch"
DEFAULT_STEPS = 10
WARMUP_STEPS = 2
BASE_PORT = 29600
TIMEOUT = 900

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

# ──────────────────────────────────────────────────────────────
#  Benchmark configurations
#  (label, model, tp, pp, dp, cp, bs, ga, seq, gc, sp, pp_engine)
#  sp: whether to enable SEQUENCE_PARALLEL env var (requires TP>1)
#  pp_engine: "1f1b" or "afab" (only matters when PP>1)
# ──────────────────────────────────────────────────────────────

# fmt: off
CONFIGS = [
    # ================================================================
    #  Qwen3-0.6B  (28 layers, ~1.5GB bf16)
    # ================================================================
    # --- Pure DP ---
    ("0.6B-DP8",               "Qwen3-0.6B", 1, 1, 8, 1, 2, 2, 2048, False, False, "1f1b"),
    # --- TP + DP ---
    ("0.6B-TP2-DP4",           "Qwen3-0.6B", 2, 1, 4, 1, 2, 1, 2048, False, False, "1f1b"),
    ("0.6B-TP4-DP2",           "Qwen3-0.6B", 4, 1, 2, 1, 2, 1, 2048, False, False, "1f1b"),
    # --- PP + DP ---
    ("0.6B-PP2-DP4",           "Qwen3-0.6B", 1, 2, 4, 1, 2, 2, 2048, False, False, "1f1b"),
    ("0.6B-PP4-DP2",           "Qwen3-0.6B", 1, 4, 2, 1, 2, 2, 2048, False, False, "1f1b"),
    # --- CP + DP (need longer seq for CP to shine) ---
    ("0.6B-CP2-DP4",           "Qwen3-0.6B", 1, 1, 4, 2, 1, 1, 4096, False, False, "1f1b"),
    ("0.6B-CP4-DP2",           "Qwen3-0.6B", 1, 1, 2, 4, 1, 1, 8192, True,  False, "1f1b"),
    # --- SP (requires TP>1) ---
    ("0.6B-SP-TP2-DP4",        "Qwen3-0.6B", 2, 1, 4, 1, 2, 1, 2048, False, True,  "1f1b"),
    # --- Mixed: TP + PP + DP ---
    ("0.6B-TP2-PP2-DP2",       "Qwen3-0.6B", 2, 2, 2, 1, 2, 1, 2048, False, False, "1f1b"),
    # --- Mixed: TP + CP + DP ---
    ("0.6B-TP2-CP2-DP2",       "Qwen3-0.6B", 2, 1, 2, 2, 1, 1, 4096, False, False, "1f1b"),

    # ================================================================
    #  Qwen3-1.7B  (28 layers, ~3.4GB bf16)
    # ================================================================
    # --- Pure DP ---
    ("1.7B-DP8-GC",            "Qwen3-1.7B", 1, 1, 8, 1, 1, 2, 2048, True,  False, "1f1b"),
    # --- TP + DP ---
    ("1.7B-TP2-DP4",           "Qwen3-1.7B", 2, 1, 4, 1, 1, 1, 2048, False, False, "1f1b"),
    ("1.7B-TP4-DP2",           "Qwen3-1.7B", 4, 1, 2, 1, 1, 1, 2048, False, False, "1f1b"),
    # --- PP + DP ---
    ("1.7B-PP2-DP4-GC",        "Qwen3-1.7B", 1, 2, 4, 1, 1, 2, 2048, True,  False, "1f1b"),
    ("1.7B-PP4-DP2-GC",        "Qwen3-1.7B", 1, 4, 2, 1, 1, 2, 2048, True,  False, "1f1b"),
    # --- CP + DP ---
    ("1.7B-CP2-DP4-GC",        "Qwen3-1.7B", 1, 1, 4, 2, 1, 1, 4096, True,  False, "1f1b"),
    ("1.7B-CP4-DP2-GC",        "Qwen3-1.7B", 1, 1, 2, 4, 1, 1, 8192, True,  False, "1f1b"),
    # --- SP ---
    ("1.7B-SP-TP2-DP4",        "Qwen3-1.7B", 2, 1, 4, 1, 1, 1, 2048, False, True,  "1f1b"),
    # --- Mixed ---
    ("1.7B-TP2-PP2-DP2-GC",    "Qwen3-1.7B", 2, 2, 2, 1, 1, 1, 2048, True,  False, "1f1b"),
    ("1.7B-TP2-CP2-DP2-GC",    "Qwen3-1.7B", 2, 1, 2, 2, 1, 1, 4096, True,  False, "1f1b"),

    # ================================================================
    #  Qwen3-4B  (36 layers, ~8GB bf16)
    # ================================================================
    # --- Pure DP ---
    ("4B-DP8-GC",              "Qwen3-4B",   1, 1, 8, 1, 1, 1, 2048, True,  False, "1f1b"),
    # --- TP + DP ---
    ("4B-TP2-DP4-GC",          "Qwen3-4B",   2, 1, 4, 1, 1, 1, 2048, True,  False, "1f1b"),
    ("4B-TP4-DP2",             "Qwen3-4B",   4, 1, 2, 1, 1, 1, 2048, False, False, "1f1b"),
    # --- PP + DP ---
    ("4B-PP2-DP4-GC",          "Qwen3-4B",   1, 2, 4, 1, 1, 1, 2048, True,  False, "1f1b"),
    ("4B-PP4-DP2-GC",          "Qwen3-4B",   1, 4, 2, 1, 1, 1, 2048, True,  False, "1f1b"),
    # --- CP + DP ---
    ("4B-CP2-DP4-GC",          "Qwen3-4B",   1, 1, 4, 2, 1, 1, 4096, True,  False, "1f1b"),
    # --- SP ---
    ("4B-SP-TP2-DP4-GC",       "Qwen3-4B",   2, 1, 4, 1, 1, 1, 2048, True,  True,  "1f1b"),
    # --- Mixed ---
    ("4B-TP2-PP2-DP2-GC",      "Qwen3-4B",   2, 2, 2, 1, 1, 1, 2048, True,  False, "1f1b"),
    ("4B-TP2-CP2-DP2-GC",      "Qwen3-4B",   2, 1, 2, 2, 1, 1, 4096, True,  False, "1f1b"),
    ("4B-TP2-PP2-CP2-DP1-GC",  "Qwen3-4B",   2, 2, 1, 2, 1, 1, 4096, True,  False, "1f1b"),

    # ================================================================
    #  Qwen3-8B  (36 layers, ~16GB bf16)
    # ================================================================
    # --- TP + DP ---
    ("8B-TP2-DP4-GC",          "Qwen3-8B",   2, 1, 4, 1, 1, 1, 2048, True,  False, "1f1b"),
    ("8B-TP4-DP2-GC",          "Qwen3-8B",   4, 1, 2, 1, 1, 1, 2048, True,  False, "1f1b"),
    ("8B-TP8",                 "Qwen3-8B",   8, 1, 1, 1, 1, 1, 2048, False, False, "1f1b"),
    # --- PP + DP ---
    ("8B-PP2-DP4-GC",          "Qwen3-8B",   1, 2, 4, 1, 1, 1, 2048, True,  False, "1f1b"),
    ("8B-PP4-DP2-GC",          "Qwen3-8B",   1, 4, 2, 1, 1, 1, 2048, True,  False, "1f1b"),
    # --- CP + DP ---
    ("8B-CP2-DP4-GC",          "Qwen3-8B",   1, 1, 4, 2, 1, 1, 4096, True,  False, "1f1b"),
    # --- SP ---
    ("8B-SP-TP2-DP4-GC",       "Qwen3-8B",   2, 1, 4, 1, 1, 1, 2048, True,  True,  "1f1b"),
    # --- Mixed ---
    ("8B-TP2-PP2-DP2-GC",      "Qwen3-8B",   2, 2, 2, 1, 1, 1, 2048, True,  False, "1f1b"),
    ("8B-TP2-CP2-DP2-GC",      "Qwen3-8B",   2, 1, 2, 2, 1, 1, 4096, True,  False, "1f1b"),
    ("8B-TP2-PP2-CP2-DP1-GC",  "Qwen3-8B",   2, 2, 1, 2, 1, 1, 4096, True,  False, "1f1b"),
    ("8B-TP4-PP2-DP1-GC",      "Qwen3-8B",   4, 2, 1, 1, 1, 1, 2048, True,  False, "1f1b"),

    # ================================================================
    #  Qwen3-14B  (40 layers, hidden=5120, heads=40, kv_heads=8, ~28GB bf16)
    #  Valid TP: 2/4/8 (GCD of 40 and 8 = 8)
    #  Memory estimate (params+optim+grad ≈ 12 bytes/param):
    #    TP4-DP2 → 14B×12/4 ≈ 42GB/GPU   TP8 → 21GB/GPU
    # ================================================================
    # --- TP + DP ---
    ("14B-TP4-DP2-GC",         "Qwen3-14B",  4, 1, 2, 1, 1, 1, 2048, True,  False, "1f1b"),
    ("14B-TP4-DP2",            "Qwen3-14B",  4, 1, 2, 1, 1, 1, 2048, False, False, "1f1b"),
    ("14B-TP8",                "Qwen3-14B",  8, 1, 1, 1, 1, 1, 2048, False, False, "1f1b"),
    ("14B-TP8-GC",             "Qwen3-14B",  8, 1, 1, 1, 1, 1, 2048, True,  False, "1f1b"),
    # --- SP (requires TP > 1) ---
    ("14B-SP-TP4-DP2-GC",      "Qwen3-14B",  4, 1, 2, 1, 1, 1, 2048, True,  True,  "1f1b"),
    ("14B-SP-TP8",             "Qwen3-14B",  8, 1, 1, 1, 1, 1, 2048, False, True,  "1f1b"),
    # --- CP (seq partitioning via Ring Attention) ---
    ("14B-TP4-CP2-DP1-GC",     "Qwen3-14B",  4, 1, 1, 2, 1, 1, 4096, True,  False, "1f1b"),
    # --- PP (no DP to avoid HCCL PP+DP issue) ---
    ("14B-TP4-PP2-DP1-GC",     "Qwen3-14B",  4, 2, 1, 1, 1, 1, 2048, True,  False, "1f1b"),

    # ================================================================
    #  Qwen3-32B  (64 layers, hidden=5120, heads=64, kv_heads=8, ~64GB bf16)
    #  Valid TP: 4/8 (GCD of 64 and 8 = 8; TP2 → 96GB/GPU, exceeds 64GB)
    #  Memory estimate (params+optim+grad ≈ 12 bytes/param):
    #    TP8      → 32B×12/8 ≈ 48GB/GPU   (fits)
    #    TP4-PP2  → 16B×12/4 ≈ 48GB/GPU   (PP halves active params, fits)
    # ================================================================
    # --- TP only (TP4 alone won't fit: 96GB/GPU without PP) ---
    ("32B-TP8-GC",             "Qwen3-32B",  8, 1, 1, 1, 1, 1, 2048, True,  False, "1f1b"),
    ("32B-TP8",                "Qwen3-32B",  8, 1, 1, 1, 1, 1, 2048, False, False, "1f1b"),
    ("32B-TP8-BS2-GC",         "Qwen3-32B",  8, 1, 1, 1, 2, 1, 2048, True,  False, "1f1b"),
    ("32B-TP8-SEQ4K-GC",       "Qwen3-32B",  8, 1, 1, 1, 1, 1, 4096, True,  False, "1f1b"),
    # --- SP ---
    ("32B-SP-TP8-GC",          "Qwen3-32B",  8, 1, 1, 1, 1, 1, 2048, True,  True,  "1f1b"),
    ("32B-SP-TP8",             "Qwen3-32B",  8, 1, 1, 1, 1, 1, 2048, False, True,  "1f1b"),
    # --- PP (TP4×PP2=8, ~48GB/GPU via layer-splitting) ---
    ("32B-TP4-PP2-DP1-GC",     "Qwen3-32B",  4, 2, 1, 1, 1, 1, 2048, True,  False, "1f1b"),
    ("32B-TP4-PP2-DP1",        "Qwen3-32B",  4, 2, 1, 1, 1, 1, 2048, False, False, "1f1b"),
]
# fmt: on


def build_cmd(cfg, port, steps):
    """Build the torchrun command for a benchmark config."""
    label, model, tp, pp, dp, cp, bs, ga, seq, gc, sp, pp_engine = cfg
    nproc = tp * pp * dp * cp

    env_exports = " && ".join(f"export {k}={v}" for k, v in NPU_ENV.items())
    if sp:
        env_exports += " && export SEQUENCE_PARALLEL=1"
    else:
        env_exports += " && export SEQUENCE_PARALLEL=0"

    model_path = os.path.join(MODEL_ROOT, model)

    cmd = (
        f"{CANN_ENV} && {env_exports}"
        f" && {TORCHRUN}"
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
        f" --pipeline_parallel_engine {pp_engine}"
        f" --micro_batch_size {bs}"
        f" --gradient_accumulation_steps {ga}"
        f" --sequence_length {seq}"
        f" --gradient_checkpointing {gc}"
        f" --learning_rate 3e-4"
        f" --use_fused_adam True"
        f" --max_grad_norm 1.0"
        f" --total_train_steps {steps}"
        f" --seed 42 --num_workers 0 --num_proc 1"
        f" --log_interval 1"
    )
    return cmd, nproc


def parse_metrics(output):
    """Parse per-step metrics from training output.

    Handles K/M/B/T suffixes from to_readable_format().
    """
    suffix_mult = {'K': 1e3, 'M': 1e6, 'B': 1e9, 'T': 1e12}

    def parse_readable(s):
        s = s.strip()
        if s and s[-1] in suffix_mult:
            return float(s[:-1]) * suffix_mult[s[-1]]
        return float(s)

    steps_data = []
    for line in output.split("\n"):
        m = re.search(
            r"Step:\s*(\d+)\s*\|.*Loss:\s*([\d.]+).*"
            r"Tokens/s:\s*([\d.]+[KMBT]?)\s*.*"
            r"Tokens/s/GPU:\s*([\d.]+[KMBT]?)\s*.*"
            r"MFU:\s*([\d.]+)%.*Memory:\s*([\d.]+)GB",
            line,
        )
        if m:
            steps_data.append({
                "step": int(m.group(1)),
                "loss": float(m.group(2)),
                "tokens_per_sec": parse_readable(m.group(3)),
                "tokens_per_sec_gpu": parse_readable(m.group(4)),
                "mfu": float(m.group(5)),
                "memory_gb": float(m.group(6)),
            })
    return steps_data


def load_metrics_from_json(workdir, pre_run_files, warmup_steps, nproc):
    """Load metrics from NEW performance log JSON files created after the run.

    pre_run_files: set of filenames that existed before the run started.
    Returns a dict, or None if no suitable file found.
    """
    new_files = []
    for fname in os.listdir(workdir):
        if not re.match(r'performance_logs_\d+_\d+\.json', fname):
            continue
        if fname not in pre_run_files:
            new_files.append(os.path.join(workdir, fname))

    if not new_files:
        return None

    best = None
    best_iters = 0
    for fpath in new_files:
        try:
            d = json.load(open(fpath))
            n = len(d.get("stats", []))
            if n > best_iters:
                best_iters = n
                best = (fpath, d)
        except Exception:
            continue

    if best is None or best_iters == 0:
        return None

    fpath, data = best
    avg = data.get("average_stats", {})
    stats = data.get("stats", [])

    steady = [s for s in stats if s.get("iteration", 0) > warmup_steps]
    if not steady:
        steady = stats

    global_tok_s = avg.get("average_tokens_per_second",
                            sum(s["tokens_per_second"] for s in steady) / len(steady))
    avg_mem_mb = avg.get("gpu", {}).get("avg_gpu_memory_allocated", 0)

    return {
        "tokens_per_sec": round(global_tok_s),
        "tokens_per_sec_gpu": round(global_tok_s / nproc),
        "memory_gb": round(avg_mem_mb / 1024, 1),
    }


def parse_loss_mfu_from_output(output, warmup_steps):
    """Extract loss and MFU from stdout using simple regexes."""
    suffix_mult = {'K': 1e3, 'M': 1e6, 'B': 1e9, 'T': 1e12}

    def parse_num(s):
        s = s.strip()
        if s and s[-1] in suffix_mult:
            return float(s[:-1]) * suffix_mult[s[-1]]
        return float(s)

    losses, mfus = [], []
    for line in output.split("\n"):
        step_m = re.search(r'Step:\s*(\d+)', line)
        if not step_m:
            continue
        step = int(step_m.group(1))
        if step <= warmup_steps:
            continue
        loss_m = re.search(r'Loss:\s*([\d.]+)', line)
        mfu_m = re.search(r'MFU:\s*([\d.]+)%', line)
        if loss_m:
            losses.append(float(loss_m.group(1)))
        if mfu_m:
            mfus.append(float(mfu_m.group(1)))

    return (
        round(losses[-1], 4) if losses else 0.0,
        round(sum(mfus) / len(mfus), 1) if mfus else 0.0,
    )


def run_config(cfg, port, steps):
    """Execute a single benchmark config and return results."""
    label, model, tp, pp, dp, cp, bs, ga, seq, gc, sp, pp_engine = cfg
    cmd, nproc = build_cmd(cfg, port, steps)

    parallel_str = []
    if tp > 1: parallel_str.append(f"TP{tp}")
    if pp > 1: parallel_str.append(f"PP{pp}")
    if cp > 1: parallel_str.append(f"CP{cp}")
    if dp > 1: parallel_str.append(f"DP{dp}")
    elif dp == 1 and not parallel_str: parallel_str.append("DP1")
    if sp: parallel_str.insert(0, "SP")
    par_desc = "-".join(parallel_str) if parallel_str else "DP1"

    print(f"\n{'='*70}")
    print(f"  [{label}] {model} | {par_desc}")
    print(f"  BS={bs} GA={ga} SEQ={seq} GC={gc} SP={sp} Engine={pp_engine} NPUs={nproc}")
    print(f"{'='*70}")

    # Snapshot of existing log files before the run
    pre_run_files = {
        f for f in os.listdir(WORKDIR)
        if re.match(r'performance_logs_\d+_\d+\.json', f)
    }

    run_start = time.time()
    try:
        result = subprocess.run(
            ["bash", "-lc", cmd],
            capture_output=True, text=True, timeout=TIMEOUT, cwd=WORKDIR,
        )
        output = result.stdout + result.stderr
        elapsed = time.time() - run_start
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after {TIMEOUT}s")
        return {"label": label, "model": model, "status": "TIMEOUT", "parallel": par_desc}
    except Exception as e:
        print(f"  ERROR: {e}")
        return {"label": label, "model": model, "status": f"ERROR: {e}", "parallel": par_desc}

    if result.returncode != 0:
        err_lines = [
            line for line in output.split("\n")
            if any(kw in line.lower() for kw in ["error", "oom", "out of memory", "killed"])
        ]
        err_msg = err_lines[0][:200] if err_lines else "unknown error"
        print(f"  FAILED (rc={result.returncode}): {err_msg}")
        return {"label": label, "model": model, "status": f"FAILED: {err_msg}", "parallel": par_desc}

    # Primary: load tok/s + memory from new JSON performance logs
    json_metrics = load_metrics_from_json(WORKDIR, pre_run_files, WARMUP_STEPS, nproc)

    # Secondary: parse loss + MFU from stdout (only rank 0 prints these)
    final_loss, avg_mfu = parse_loss_mfu_from_output(output, WARMUP_STEPS)

    if json_metrics is None:
        # Fallback: full stdout regex parse
        steps_data = parse_metrics(output)
        if not steps_data:
            print(f"  No metrics found in output or JSON logs")
            return {"label": label, "model": model, "status": "NO_METRICS", "parallel": par_desc}
        steady = [s for s in steps_data if s["step"] > WARMUP_STEPS] or steps_data
        json_metrics = {
            "tokens_per_sec": round(sum(s["tokens_per_sec"] for s in steady) / len(steady)),
            "tokens_per_sec_gpu": round(sum(s["tokens_per_sec_gpu"] for s in steady) / len(steady)),
            "memory_gb": round(sum(s["memory_gb"] for s in steady) / len(steady), 1),
        }
        final_loss = round(steps_data[-1]["loss"], 4)
        avg_mfu = round(sum(s["mfu"] for s in steady) / len(steady), 1)

    result_dict = {
        "label": label, "model": model, "status": "OK", "parallel": par_desc,
        "tp": tp, "pp": pp, "dp": dp, "cp": cp, "sp": sp,
        "pp_engine": pp_engine, "npus": nproc,
        "bs": bs, "ga": ga, "seq": seq, "gc": gc,
        "tokens_per_sec": json_metrics["tokens_per_sec"],
        "tokens_per_sec_gpu": json_metrics["tokens_per_sec_gpu"],
        "mfu": avg_mfu,
        "memory_gb": json_metrics["memory_gb"],
        "final_loss": final_loss,
        "elapsed": round(elapsed, 1),
    }
    print(f"  OK | Tok/s: {result_dict['tokens_per_sec']:,} | Tok/s/GPU: {result_dict['tokens_per_sec_gpu']:,} "
          f"| MFU: {avg_mfu:.1f}% | Mem: {result_dict['memory_gb']:.1f}GB | Loss: {final_loss:.4f}")
    return result_dict


def print_summary(results):
    """Print a formatted summary table."""
    ok = [r for r in results if r.get("status") == "OK"]
    failed = [r for r in results if r.get("status") != "OK"]

    if ok:
        print(f"\n{'='*90}")
        print("BENCHMARK RESULTS SUMMARY")
        print(f"{'='*90}")
        hdr = (f"{'Label':<32} {'Parallel':<20} {'SEQ':>5} {'Tok/s':>8} "
               f"{'Tok/s/GPU':>10} {'MFU%':>6} {'HBM(GB)':>8}")
        print(hdr)
        print("-" * 90)
        for r in ok:
            print(f"{r['label']:<32} {r['parallel']:<20} {r['seq']:>5} "
                  f"{r['tokens_per_sec']:>8,} {r['tokens_per_sec_gpu']:>10,} "
                  f"{r['mfu']:>5.1f}% {r['memory_gb']:>7.1f}")

    if failed:
        print(f"\nFailed configs ({len(failed)}):")
        for r in failed:
            print(f"  {r['label']}: {r['status']}")


def print_markdown_table(results):
    """Print results as a Markdown table for README."""
    ok = [r for r in results if r.get("status") == "OK"]
    if not ok:
        return
    print("\n### Comprehensive 8-NPU Benchmark Results\n")
    print("| Model | Parallelism | BS | GA | SEQ | GC | SP | "
          "Total Tok/s | Tok/s/GPU | MFU% | HBM/GPU |")
    print("|-------|------------|-----|-----|------|------|------|"
          "-----------|----------|------|---------|")
    for r in ok:
        gc_str = "Yes" if r["gc"] else "-"
        sp_str = "Yes" if r["sp"] else "-"
        print(f"| {r['model']} | {r['parallel']} | {r['bs']} | {r['ga']} | "
              f"{r['seq']} | {gc_str} | {sp_str} | "
              f"{r['tokens_per_sec']:,} | {r['tokens_per_sec_gpu']:,} | "
              f"{r['mfu']}% | {r['memory_gb']} GB |")


def main():
    parser = argparse.ArgumentParser(description="ScaleTorch Comprehensive Benchmark")
    parser.add_argument("--filter", type=str, default=None,
                        help="Regex filter for config labels (e.g. '0.6B|1.7B')")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                        help="Training steps per config")
    parser.add_argument("--list", action="store_true",
                        help="List all configs without running")
    parser.add_argument("--output", type=str, default="benchmark_comprehensive_results.json",
                        help="Output JSON file")
    args = parser.parse_args()

    configs = CONFIGS
    if args.filter:
        pat = re.compile(args.filter, re.IGNORECASE)
        configs = [c for c in configs if pat.search(c[0])]

    if args.list:
        print(f"{'#':<3} {'Label':<35} {'Model':<12} {'TP':>3} {'PP':>3} "
              f"{'DP':>3} {'CP':>3} {'BS':>3} {'GA':>3} {'SEQ':>5} {'GC':>5} {'SP':>5}")
        print("-" * 90)
        for i, c in enumerate(configs, 1):
            label, model, tp, pp, dp, cp, bs, ga, seq, gc, sp, eng = c
            print(f"{i:<3} {label:<35} {model:<12} {tp:>3} {pp:>3} "
                  f"{dp:>3} {cp:>3} {bs:>3} {ga:>3} {seq:>5} {str(gc):>5} {str(sp):>5}")
        print(f"\nTotal: {len(configs)} configs")
        return

    print(f"ScaleTorch Comprehensive Benchmark")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Configs: {len(configs)} | Steps: {args.steps}")
    print(f"Output: {args.output}")

    results = []
    port = BASE_PORT

    for i, cfg in enumerate(configs):
        print(f"\n>>> Progress: {i+1}/{len(configs)}")
        r = run_config(cfg, port, args.steps)
        results.append(r)
        port += 1

        with open(os.path.join(WORKDIR, args.output), "w") as f:
            json.dump(results, f, indent=2)
        time.sleep(5)

    print_summary(results)
    print_markdown_table(results)

    with open(os.path.join(WORKDIR, args.output), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
