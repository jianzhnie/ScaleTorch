# ScaleTorch

5D parallelism distributed training framework built on PyTorch. Implements **Tensor Parallelism (TP)**, **Pipeline Parallelism (PP)**, **Context Parallelism (CP)**, **Expert Parallelism (EP)**, and **Data Parallelism (DP)** in a unified process grid `[DP, PP, CP, EP, TP]`.

## Features

- **4D+EP Process Grid** - `ProcessGroupManager` orchestrates DP/PP/CP/EP/TP groups from a single `world_size` constraint
- **Tensor Parallelism** - Column/row parallel linear layers, embedding & layer norm sharding
- **Pipeline Parallelism** - 1F1B and AFAB schedules with stage partitioning
- **Context Parallelism** - Sequence-length partitioning via Ring Attention
- **Data Parallelism** - Gradient bucketing and overlapped all-reduce (Megatron-style)
- **Expert Parallelism** - MoE expert sharding with all-to-all token dispatch
- **Sequence Parallelism** - AllGather/ReduceScatter for LayerNorm regions
- **Model Support** - Llama, Qwen3, Qwen3-MoE, Mixture-of-Experts (MoE)
- **Attention Variants** - MHA, MQA, GQA, MLA (Multi-head Latent Attention)
- **NPU Support** - Ascend 910/910B via `torch_npu`, HCCL backend, device-agnostic abstraction
- **Config** - HuggingFace `HfArgumentParser` dataclasses for all training args
- **Checkpointing** - Weight materialization/dematerialization, tied-embedding support

## Performance (Ascend 910B, bf16, CANN 8.2 + PyTorch 2.5)

### Single-NPU Benchmarks

| Model | SEQ | BS | GC | Tokens/s/GPU | TFLOP/s/GPU | MFU | HBM |
|-------|------|-----|------|-------------|------------|------|-------|
| Qwen3-0.6B | 2048 | 2 | - | 9,731 | 57.6 | 22.5% | 22.2 GB |
| Qwen3-0.6B | 8192 | 1 | Yes | 9,834 | 99.8 | 39.0% | 21.4 GB |
| Qwen3-0.6B | 16384 | 1 | Yes | 9,079 | 143.3 | **56.0%** | 39.2 GB |
| Qwen3-1.7B | 2048 | 1 | - | 4,685 | 63.7 | 24.9% | 23.8 GB |
| Qwen3-1.7B | 2048 | 1 | Yes | 3,162 | 43.0 | 16.8% | 19.5 GB |
| Qwen3-1.7B | 8192 | 1 | Yes | 7,396 | 131.9 | **51.5%** | 32.0 GB |
| Qwen3-4B   | 2048 | 1 | Yes | 2,415 | 72.7 | 28.4% | 38.8 GB |

### 8-NPU Comprehensive Benchmarks

All configs use 8 Ascend 910B NPUs (64 GB HBM each), bf16 precision, FlashAttention enabled.
`SP` = Sequence Parallelism (enabled via `SEQUENCE_PARALLEL=1`, requires TP > 1).
`CP` = Context Parallelism (Ring Attention). Results from `scripts/benchmark_comprehensive.py`.

#### Qwen3-0.6B (28 layers, ~1.5 GB bf16)

| Parallelism | BS | GA | SEQ | GC | Total Tok/s | Tok/s/GPU | TFLOP/s/GPU | MFU | HBM/GPU |
|------------|-----|-----|------|------|------------|----------|------------|-----|---------|
| **DP8** | 2 | 2 | 2048 | - | **131,392** | **16,422** | **97.3** | **38.0%** | 21.6 GB |
| CP2-DP4 | 1 | 1 | 4096 | - | 73,715 | 9,218 | 67.6 | 26.4% | 16.1 GB |
| SP-TP2-DP4 | 2 | 1 | 2048 | - | 65,090 | 8,136 | 48.1 | 18.8% | 17.9 GB |
| TP2-DP4 | 2 | 1 | 2048 | - | 64,278 | 8,035 | 47.6 | 18.6% | 17.9 GB |
| CP4-DP2 | 1 | 1 | 8192 | Yes | 54,638 | 6,829 | 69.4 | 27.1% | 11.7 GB |
| TP2-CP2-DP2 | 1 | 1 | 4096 | - | 42,718 | 5,340 | 39.2 | 15.3% | 10.8 GB |
| TP4-DP2 | 2 | 1 | 2048 | - | 36,114 | 4,514 | 26.6 | 10.4% | 14.0 GB |

#### Qwen3-1.7B (28 layers, ~3.4 GB bf16)

| Parallelism | BS | GA | SEQ | GC | Total Tok/s | Tok/s/GPU | TFLOP/s/GPU | MFU | HBM/GPU |
|------------|-----|-----|------|------|------------|----------|------------|-----|---------|
| **DP8** | 1 | 2 | 2048 | Yes | **54,328** | **6,792** | **92.4** | **36.1%** | 16.8 GB |
| CP4-DP2 | 1 | 1 | 8192 | Yes | 40,772 | 5,096 | 90.9 | 35.5% | 25.2 GB |
| CP2-DP4 | 1 | 1 | 4096 | Yes | 39,136 | 4,891 | 73.5 | 28.7% | 25.3 GB |
| TP2-DP4 | 1 | 1 | 2048 | - | 31,468 | 3,932 | 53.5 | 20.9% | 19.2 GB |
| SP-TP2-DP4 | 1 | 1 | 2048 | - | 30,151 | 3,769 | 51.2 | 20.0% | 19.2 GB |
| TP4-DP2 | 1 | 1 | 2048 | - | 19,529 | 2,440 | 33.3 | 13.0% | 12.8 GB |

#### Qwen3-4B (36 layers, ~8 GB bf16)

| Parallelism | BS | GA | SEQ | GC | Total Tok/s | Tok/s/GPU | TFLOP/s/GPU | MFU | HBM/GPU |
|------------|-----|-----|------|------|------------|----------|------------|-----|---------|
| CP2-DP4 | 1 | 1 | 4096 | Yes | 21,762 | 2,719 | 91.6 | 35.8% | 54.1 GB |
| **DP8** | 1 | 1 | 2048 | Yes | **21,661** | **2,706** | **81.4** | **31.8%** | 54.5 GB |
| SP-TP2-DP4 | 1 | 1 | 2048 | Yes | 15,949 | 1,994 | 59.9 | 23.4% | 28.5 GB |
| TP2-DP4 | 1 | 1 | 2048 | Yes | 15,876 | 1,984 | 59.6 | 23.3% | 28.5 GB |
| TP2-CP2-DP2 | 1 | 1 | 4096 | Yes | 15,609 | 1,952 | 65.8 | 25.7% | 28.1 GB |
| TP4-DP2 | 1 | 1 | 2048 | - | 13,026 | 1,629 | 48.9 | 19.1% | 21.5 GB |

#### Qwen3-8B (36 layers, ~16 GB bf16)

| Parallelism | BS | GA | SEQ | GC | Total Tok/s | Tok/s/GPU | TFLOP/s/GPU | MFU | HBM/GPU |
|------------|-----|-----|------|------|------------|----------|------------|-----|---------|
| **TP2-CP2-DP2** | 1 | 1 | 4096 | Yes | **11,262** | **1,406** | **79.4** | **31.0%** | 50.2 GB |
| SP-TP2-DP4 | 1 | 1 | 2048 | Yes | 11,238 | 1,405 | 74.2 | 29.0% | 50.6 GB |
| TP2-DP4 | 1 | 1 | 2048 | Yes | 11,121 | 1,391 | 73.5 | 28.7% | 50.6 GB |
| TP8-SEQ4K | 1 | 1 | 4096 | Yes | 9,400 | 1,175 | 66.3 | 25.9% | 20.9 GB |
| TP4-DP2 | 1 | 1 | 2048 | Yes | 8,000 | 998 | 52.7 | 20.6% | 26.6 GB |
| TP8 | 1 | 1 | 2048 | - | 6,946 | 868 | 45.8 | 17.9% | 17.2 GB |
| TP4-PP2 | 1 | 1 | 2048 | Yes | 6,344 | 793 | 41.7 | 16.3% | 13.3 GB |
| CP2-DP4 | 1 | 1 | 4096 | Yes | — | — | — | — | OOM |

#### Qwen3-14B (40 layers, hidden=5120, heads=40, kv_heads=8, ~28 GB bf16)

Minimum viable TP: **TP4** (TP2 would require ~84 GB/GPU for params+optimizer).
Valid TP sizes: 2, 4, 8 (divisors of GCD(40 heads, 8 kv_heads) = 8).

| Parallelism | BS | GA | SEQ | GC | Total Tok/s | Tok/s/GPU | TFLOP/s/GPU | MFU | HBM/GPU |
|------------|-----|-----|------|------|------------|----------|------------|-----|---------|
| **TP4-CP2** | 1 | 1 | 4096 | Yes | **6,969** | **871** | **86.0** | **33.6%** | 31.1 GB |
| TP4-DP2 | 1 | 1 | 2048 | - | 6,544 | 818 | 76.5 | 29.9% | 53.1 GB |
| TP4-DP2 | 1 | 1 | 2048 | Yes | 5,741 | 718 | 67.1 | 26.2% | 45.8 GB |
| SP-TP4-DP2 | 1 | 1 | 2048 | Yes | 5,698 | 712 | 66.8 | 26.1% | 45.8 GB |
| SP-TP8 | 1 | 1 | 2048 | - | 5,205 | 651 | 60.9 | 23.8% | 24.8 GB |
| TP8 | 1 | 1 | 2048 | - | 5,195 | 650 | 60.9 | 23.8% | 24.8 GB |
| TP4-PP2 | 1 | 1 | 2048 | Yes | 4,516 | 565 | 53.0 | 20.7% | 19.9 GB |
| TP8 | 1 | 1 | 2048 | Yes | 4,111 | 514 | 48.1 | 18.8% | 18.2 GB |

#### Qwen3-32B (64 layers, hidden=5120, heads=64, kv_heads=8, ~64 GB bf16)

Minimum viable TP: **TP8** (TP4 alone requires ~96 GB/GPU). TP4-PP2 works via PP halving active params.
Valid TP sizes: 4, 8 (divisors of GCD(64 heads, 8 kv_heads) = 8).

| Parallelism | BS | GA | SEQ | GC | Total Tok/s | Tok/s/GPU | TFLOP/s/GPU | MFU | HBM/GPU |
|------------|-----|-----|------|------|------------|----------|------------|-----|---------|
| TP8-BS2 | 2 | 1 | 2048 | Yes | 3,016 | 377 | 79.1 | 30.9% | 39.5 GB |
| **TP8-SEQ4K** | 1 | 1 | 4096 | Yes | **2,956** | **369** | **82.2** | **32.1%** | 39.5 GB |
| TP8 | 1 | 1 | 2048 | - | 2,952 | 369 | 77.3 | 30.2% | 45.8 GB |
| SP-TP8 | 1 | 1 | 2048 | - | 2,922 | 365 | 76.5 | 29.9% | 45.8 GB |
| TP4-PP2 | 1 | 1 | 2048 | - | 2,361 | 295 | 62.0 | 24.2% | 39.1 GB |
| TP4-PP2 | 1 | 1 | 2048 | Yes | 2,360 | 295 | 61.7 | 24.1% | 39.1 GB |
| TP8 | 1 | 1 | 2048 | Yes | 2,338 | 292 | 61.2 | 23.9% | 35.1 GB |
| SP-TP8 | 1 | 1 | 2048 | Yes | 2,310 | 289 | 60.4 | 23.6% | 35.1 GB |

#### Qwen3-30B-A3B (MoE: 48 layers, 128 experts, top-8, hidden=2048, ~30.5B total / ~3.4B active)

Mixture-of-Experts model with **Expert Parallelism (EP)**: experts are sharded across EP ranks.
Only 3.35B of 30.53B parameters are active per token (top-8 out of 128 experts).
MFU is computed against active FLOPs (`6×N_active + 12×L×H×d×S`).

| Parallelism | BS | GA | SEQ | GC | Total Tok/s | Tok/s/GPU | TFLOP/s/GPU | MFU | HBM/GPU |
|------------|-----|-----|------|------|------------|----------|------------|-----|---------|
| EP2-TP4-SEQ4K | 1 | 1 | 4096 | - | 2,616 | 327 | 9.7 | 3.8% | 34.5 GB |
| **EP2-TP4** | 1 | 1 | 2048 | - | **1,860** | **233** | **5.8** | **2.3%** | **42.3 GB** |
| EP4-TP2 | 1 | 1 | 2048 | - | 1,832 | 229 | 5.7 | 2.2% | 21.8 GB |
| EP2-TP4-SEQ4K | 1 | 1 | 4096 | Yes | 1,440 | 180 | 5.4 | 2.1% | 36.0 GB |
| EP2-TP4-BS2 | 2 | 1 | 2048 | Yes | 1,408 | 176 | 4.4 | 1.7% | 36.0 GB |
| EP4-TP2 | 1 | 1 | 2048 | Yes | 1,016 | 127 | 3.2 | 1.2% | 21.9 GB |
| SP-EP2-TP4 | 1 | 1 | 2048 | Yes | 944 | 118 | 2.9 | 1.1% | 33.3 GB |
| EP2-TP4 | 1 | 1 | 2048 | Yes | 912 | 114 | 2.8 | 1.1% | 33.3 GB |
| EP2-TP4-GA2 | 1 | 2 | 2048 | Yes | 880 | 110 | 2.7 | 1.1% | 33.3 GB |

> **Key findings from Qwen3-30B-A3B (MoE) testing:**
> - **SEQ=4096 doubles TFLOP utilization** vs SEQ=2048: 9.7 vs 5.7 TFLOP/s/GPU (3.8% vs 2.2% MFU) — longer sequences amortize the per-token expert dispatch overhead
> - **EP2-TP4 and EP4-TP2 achieve identical throughput** (230 vs 229 tok/s/GPU at SEQ=2048) — different memory tradeoffs: EP4-TP2 uses only 21.8 GB/GPU (half of EP2-TP4's 32.6 GB)
> - **GC costs ~50% throughput for MoE** (230→114 tok/s/GPU) — much higher than dense models (~18%) because GC recomputes the expert routing + dispatch per layer
> - **BS=2 with GC** recovers 54% of the GC loss (176 vs 114 tok/s/GPU) by amortizing fixed overhead
> - **SP adds no benefit for MoE** — SP-EP2-TP4 (118) ≈ EP2-TP4 (114) tok/s/GPU; SP is designed for dense attention bottlenecks
> - **EP1 (no EP) OOMs** — 128 experts replicated on each GPU uses 60+ GB; EP≥4 hangs on HCCL `all_to_all`
> - MFU is inherently low (~2–4%) because only 11% of params (3.35B/30.53B) are active per token

> **Key findings from 14B/32B testing:**
> - **TP4-CP2 is the MFU sweet spot for 14B**: 33.6% MFU at SEQ=4096 — the PP+CP shape fix enables this combination
> - **TP4-DP2 is the throughput sweet spot for 14B**: 29.9% MFU vs 23.8% for TP8, due to less TP communication
> - **SP has negligible overhead**: SP-TP4-DP2 ≈ TP4-DP2 throughput; SP-TP8 ≈ TP8 throughput for 32B
> - **TP4-PP2 saves memory vs TP8**: 19.9 GB/GPU for 14B, 39.1 GB/GPU for 32B — useful for running multiple replicas
> - **BS=2 on 32B-TP8-GC** gives +30% throughput (377 vs 292 Tok/s/GPU) with only 4.4 GB extra HBM
> - **GC overhead is ~20%** for 14B/32B (consistent with smaller models: ~45.8→35.1 GB for 32B)
> - **32B-TP8-SEQ4K peak: 32.1% MFU** — longer sequences improve compute utilization

> **Key findings from comprehensive testing (43 OK configs across DP/TP/PP/CP/SP/EP, 7 models):**
> - **DP achieves highest compute efficiency** for models fitting in single-NPU: 0.6B reaches 38.0% MFU (DP8)
> - **CP significantly boosts throughput**: 0.6B-CP2-DP4 reaches 73K tok/s vs 64K for TP2-DP4; 4B-CP2-DP4 matches DP8
> - **SP overhead is negligible** — SP-TP2-DP4 matches or slightly beats TP2-DP4 across all model sizes
> - **TP4-CP2 now works** (bug fix): 14B achieves 33.6% MFU — highest for that model size
> - **TP4-PP2** reduces HBM/GPU vs pure TP: 19.9 GB for 14B, 39.1 GB for 32B, 13.3 GB for 8B
> - **GC cost**: ~20% throughput overhead for 20–30% memory saving, consistent across all model sizes
> - **PP+DP** combinations fail on Ascend HCCL (see Known Limitations); PP without DP works correctly
> - **Peak observed: 38.0% MFU** (0.6B-DP8, 131K tok/s total); **peak 8B+: 33.6% MFU** (14B-TP4-CP2)

## Benchmarking

Run the comprehensive parallelism benchmark suite (DP / TP / PP / CP / SP):

```bash
source set_env.sh

# Full 57-config suite (6 models × all strategies: 0.6B / 1.7B / 4B / 8B / 14B / 32B)
python scripts/benchmark_comprehensive.py

# Filter by model or strategy
python scripts/benchmark_comprehensive.py --filter "14B|32B"   # only 14B/32B configs
python scripts/benchmark_comprehensive.py --filter "CP|SP"     # only CP/SP configs

# Preview configs without running
python scripts/benchmark_comprehensive.py --list

# Shell wrapper
bash scripts/benchmark_comprehensive.sh           # all
bash scripts/benchmark_comprehensive.sh "8B"      # only 8B configs
```

Results are saved to `benchmark_comprehensive_results.json` with incremental writes.

## Known Limitations

- **PP + DP on Ascend HCCL**: Combining pipeline parallelism with data parallelism (e.g. PP2-DP4) raises `HCCL ERR99999` at process group initialization on Ascend NPU. This is a hardware-specific HCCL communicator conflict when multiple PP sub-groups coexist with DP groups on the same node. PP without DP (e.g. TP4-PP2-DP1) works correctly. Workaround: use TP-only or TP+PP without DP on single nodes.
- **CP + TP on Ascend HCCL**: ~~Previously raised `HCCL ERR99999`~~ — **Fixed** in this release via the PP+CP shape mismatch bug fix. TP4-CP2 now works correctly (tested on 14B: 33.6% MFU).
- **EP ≥ 4 on Ascend HCCL**: Expert parallelism with 4+ ranks hangs on `all_to_all_single` across EP sub-groups. EP=2 works correctly. Workaround: use EP2 combined with TP for larger MoE models.
- **Multi-node**: Supported via `scripts/torch_dist/launch_multi_nodes.sh` but not benchmarked here.



```bash
pip install -e .
```

## Quick Start

### NPU Training (Ascend 910)

```bash
# Modes: max_mfu | balanced | max_speed | min_mem
bash scripts/run_npu.sh 8 /path/to/Qwen3-0.6B /path/to/dataset balanced
```

### Manual Launch

```bash
export FLASH_ATTEN=1 DTYPE=bfloat16
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"

torchrun --nproc_per_node=8 train.py \
  --model_name_or_path /path/to/Qwen3-0.6B \
  --dataset_name /path/to/wikitext2 \
  --data_parallel_size 8 \
  --micro_batch_size 2 \
  --sequence_length 8192 \
  --gradient_checkpointing True \
  --use_fused_adam True \
  --learning_rate 3e-4 \
  --total_train_steps 1000
```

### CUDA Training

```bash
torchrun --nproc_per_node 4 train.py \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --tensor_parallel_size 2 \
  --data_parallel_size 2 \
  --micro_batch_size 4
```

## Supported Models

| Model | Config key | head_dim | QK Norm | Tie Embed | Sizes Tested |
|-------|-----------|----------|---------|-----------|--------------|
| Llama | `llama` | H/heads | No | No | 7B-70B |
| Qwen3 | `qwen3` | Explicit (128) | Yes | Varies | 0.6B-32B |
| Qwen3-MoE | `qwen3_moe` | Explicit (128) | Yes | No | 30B-A3B |

Model auto-selection via `config.model_type` from HuggingFace config.

## Docker (NPU)

```bash
# Build
docker build -t scaletorch-npu .

# Run with docker-compose
docker compose up
```

## Project Structure

```
scaletorch/
├── dist/                    # Distributed communication primitives
├── models/
│   ├── attention/           # MHA, MQA, GQA, MLA
│   ├── llama.py             # Llama transformer
│   ├── model_qwen3.py       # Qwen3 transformer (QK norms, explicit head_dim)
│   └── moe.py               # Mixture-of-Experts
├── parallel/
│   ├── process_group.py     # 5D process group manager [DP, PP, CP, EP, TP]
│   ├── tensor_parallel/     # Column/row linear, embedding sharding
│   ├── pipeline_parallel/   # 1F1B, AFAB schedules
│   ├── context_parallel/    # Ring Attention sequence parallelism
│   ├── expert_parallel/     # MoE all-to-all token dispatch
│   ├── sequence_parallel/   # AllGather/ReduceScatter SP
│   └── data_parallel/       # Gradient bucketing
├── trainer/
│   ├── config.py            # HfArgumentParser dataclass configs
│   └── lr_scheduler.py      # LR scheduler factory
├── utils/
│   ├── device.py            # NPU/CUDA/XPU device abstraction
│   ├── checkpoint.py        # CheckpointManager (Qwen3/Llama, TP, tied embed)
│   ├── misc.py              # MFU, param counting, seed, formatting
│   └── monitor.py           # Performance monitoring
└── data/                    # Dataset/dataloader with CP support
```

## Testing

```bash
pip install pytest
pytest tests/ -v
# 126 passed, 2 skipped
```

## Key Environment Variables

| Variable | Values | Effect |
|----------|--------|--------|
| `FLASH_ATTEN` | `0`/`1` | Toggle SDPA / Flash Attention |
| `DTYPE` | `bfloat16`/`float32` | Mixed precision type |
| `PYTORCH_NPU_ALLOC_CONF` | `expandable_segments:True` | Reduce memory fragmentation |
| `TASK_QUEUE_ENABLE` | `2` | NPU task queue optimization |
| `COMBINED_ENABLE` | `1` | NPU operator fusion |
| `SEQUENCE_PARALLEL` | `0`/`1` | Toggle sequence parallelism |

## References

### Training Frameworks

- [Nanotron](https://github.com/huggingface/nanotron) - HuggingFace LLM training framework
- [Picotron](https://github.com/huggingface/picotron) - Minimalistic 4D-parallelism framework
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) - NVIDIA LLM training framework
- [torchtitan](https://github.com/pytorch/torchtitan) - PyTorch native large model training
- [DeepSpeed](https://www.deepspeed.ai/) - Microsoft deep learning optimization library

### Landmark Papers

- [Megatron-LM](https://arxiv.org/abs/1909.08053) - Tensor parallelism for LLMs
- [Ring Flash Attention](https://github.com/zhuzilin/ring-flash-attention) - Ring Attention + FlashAttention
- [Llama 3](https://arxiv.org/abs/2407.21783) - Llama 3 herd of models
- [DeepSeek-V3](https://arxiv.org/abs/2412.19437v1) - DeepSeek-V3 architecture and training

## License

[Apache 2.0](LICENSE)
