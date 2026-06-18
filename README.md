# ScaleTorch

4D parallelism distributed training framework built on PyTorch. Implements **Tensor Parallelism (TP)**, **Pipeline Parallelism (PP)**, **Context Parallelism (CP)**, and **Data Parallelism (DP)** in a unified process grid `[DP, PP, CP, TP]`.

## Features

- **4D Process Grid** - `ProcessGroupManager` orchestrates DP/PP/CP/TP groups from a single `world_size` constraint
- **Tensor Parallelism** - Column/row parallel linear layers, embedding & layer norm sharding
- **Pipeline Parallelism** - 1F1B and AFAB schedules with stage partitioning
- **Context Parallelism** - Sequence-length partitioning via Ring Attention
- **Data Parallelism** - Gradient bucketing and overlapped all-reduce (Megatron-style)
- **Sequence Parallelism** - AllGather/ReduceScatter for LayerNorm regions
- **Model Support** - Llama, Qwen3, Mixture-of-Experts (MoE)
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
| DP8 | 2 | 2 | 2048 | - | 121,317 | 15,165 | 89.8 | 35.1% | 4.8 GB |
| SP-TP2-DP4 | 2 | 1 | 2048 | - | 62,819 | 7,852 | 46.5 | 18.2% | 4.6 GB |
| TP2-DP4 | 2 | 1 | 2048 | - | 58,377 | 7,297 | 43.2 | 16.9% | 4.6 GB |
| TP4-DP2 | 2 | 1 | 2048 | - | 37,296 | 4,662 | 27.6 | 10.8% | 2.6 GB |
| CP2-DP4 | 1 | 1 | 4096 | - | 21,019 | 2,627 | 19.3 | 7.5% | 8.4 GB |
| TP2-CP2-DP2 | 1 | 1 | 4096 | - | 15,537 | 1,942 | 14.2 | 5.6% | 4.3 GB |
| CP4-DP2 | 1 | 1 | 8192 | - | 8,972 | 1,122 | 11.4 | 4.4% | 8.3 GB |

#### Qwen3-1.7B (28 layers, ~3.4 GB bf16)

| Parallelism | BS | GA | SEQ | GC | Total Tok/s | Tok/s/GPU | TFLOP/s/GPU | MFU | HBM/GPU |
|------------|-----|-----|------|------|------------|----------|------------|-----|---------|
| DP8 | 1 | 2 | 2048 | Yes | 51,936 | 6,492 | 88.3 | 34.5% | 11.9 GB |
| SP-TP2-DP4 | 1 | 1 | 2048 | - | 29,729 | 3,716 | 50.5 | 19.7% | 11.5 GB |
| TP2-DP4 | 1 | 1 | 2048 | - | 29,710 | 3,714 | 50.5 | 19.7% | 11.5 GB |
| TP4-DP2 | 1 | 1 | 2048 | - | 17,322 | 2,165 | 29.4 | 11.5% | 6.0 GB |
| CP2-DP4 | 1 | 1 | 4096 | Yes | 13,657 | 1,707 | 25.6 | 10.0% | 22.2 GB |
| TP2-CP2-DP2 | 1 | 1 | 4096 | Yes | 10,392 | 1,299 | 19.5 | 7.6% | 11.2 GB |
| CP4-DP2 | 1 | 1 | 8192 | Yes | 8,050 | 1,006 | 17.9 | 7.0% | 22.1 GB |

#### Qwen3-4B (36 layers, ~8 GB bf16)

| Parallelism | BS | GA | SEQ | GC | Total Tok/s | Tok/s/GPU | TFLOP/s/GPU | MFU | HBM/GPU |
|------------|-----|-----|------|------|------------|----------|------------|-----|---------|
| DP8 | 1 | 1 | 2048 | Yes | 21,025 | 2,628 | 79.1 | 30.9% | 48.4 GB |
| SP-TP2-DP4 | 1 | 1 | 2048 | Yes | 14,924 | 1,866 | 56.1 | 21.9% | 24.5 GB |
| TP2-DP4 | 1 | 1 | 2048 | Yes | 14,381 | 1,798 | 54.1 | 21.1% | 24.5 GB |
| TP4-DP2 | 1 | 1 | 2048 | - | 12,527 | 1,566 | 47.1 | 18.4% | 12.6 GB |
| CP2-DP4 | 1 | 1 | 4096 | Yes | 6,268 | 784 | 26.4 | 10.3% | 48.0 GB |
| TP2-CP2-DP2 | 1 | 1 | 4096 | Yes | 5,107 | 638 | 21.5 | 8.4% | 24.2 GB |

#### Qwen3-8B (36 layers, ~16 GB bf16)

| Parallelism | BS | GA | SEQ | GC | Total Tok/s | Tok/s/GPU | TFLOP/s/GPU | MFU | HBM/GPU |
|------------|-----|-----|------|------|------------|----------|------------|-----|---------|
| TP2-DP4 | 1 | 1 | 2048 | Yes | 10,851 | 1,356 | 71.5 | 27.9% | 45.0 GB |
| SP-TP2-DP4 | 1 | 1 | 2048 | Yes | 10,734 | 1,342 | 70.8 | 27.7% | 45.0 GB |
| TP4-DP2 | 1 | 1 | 2048 | Yes | 7,655 | 957 | 50.5 | 19.7% | 22.8 GB |
| TP8 | 1 | 1 | 2048 | - | 6,534 | 817 | 43.1 | 16.8% | 8.0 GB |
| **TP4-PP2** | 1 | 1 | 2048 | Yes | **6,440** | **805** | **42.5** | **16.6%** | **13.4 GB** |
| TP2-CP2-DP2 | 1 | 1 | 4096 | Yes | 4,651 | 581 | 32.8 | 12.8% | 44.6 GB |
| CP2-DP4 | 1 | 1 | 4096 | Yes | — | — | — | — | OOM |

#### Qwen3-14B (40 layers, hidden=5120, heads=40, kv_heads=8, ~28 GB bf16)

Minimum viable TP: **TP4** (TP2 would require ~84 GB/GPU for params+optimizer).
Valid TP sizes: 2, 4, 8 (divisors of GCD(40 heads, 8 kv_heads) = 8).

| Parallelism | BS | GA | SEQ | GC | Total Tok/s | Tok/s/GPU | TFLOP/s/GPU | MFU | HBM/GPU |
|------------|-----|-----|------|------|------------|----------|------------|-----|---------|
| TP4-DP2 | 1 | 1 | 2048 | - | 6,501 | 813 | 76.1 | 29.7% | 53.2 GB |
| **TP4-DP2** | 1 | 1 | 2048 | Yes | 5,702 | 713 | 66.8 | **26.1%** | **45.8 GB** |
| SP-TP4-DP2 | 1 | 1 | 2048 | Yes | 5,662 | 708 | 66.3 | 25.9% | 45.8 GB |
| SP-TP8 | 1 | 1 | 2048 | - | 5,078 | 635 | 59.5 | 23.2% | 25.0 GB |
| TP8 | 1 | 1 | 2048 | - | 5,021 | 628 | 58.8 | 23.0% | 25.0 GB |
| TP4-PP2 | 1 | 1 | 2048 | Yes | 4,526 | 566 | 53.0 | 20.7% | 20.0 GB |
| TP8 | 1 | 1 | 2048 | Yes | 3,981 | 498 | 46.6 | 18.2% | 18.2 GB |

#### Qwen3-32B (64 layers, hidden=5120, heads=64, kv_heads=8, ~64 GB bf16)

Minimum viable TP: **TP8** (TP4 alone requires ~96 GB/GPU). TP4-PP2 works via PP halving active params.
Valid TP sizes: 4, 8 (divisors of GCD(64 heads, 8 kv_heads) = 8).

| Parallelism | BS | GA | SEQ | GC | Total Tok/s | Tok/s/GPU | TFLOP/s/GPU | MFU | HBM/GPU |
|------------|-----|-----|------|------|------------|----------|------------|-----|---------|
| TP8-BS2 | 2 | 1 | 2048 | Yes | 3,000 | 375 | 78.5 | 30.7% | 40.6 GB |
| SP-TP8 | 1 | 1 | 2048 | - | 2,934 | 367 | 76.9 | 30.0% | 46.5 GB |
| **TP8** | 1 | 1 | 2048 | - | **2,924** | **365** | **76.4** | **29.9%** | **46.5 GB** |
| TP8 | 1 | 1 | 2048 | Yes | 2,385 | 298 | 62.4 | 24.4% | 35.1 GB |
| TP4-PP2 | 1 | 1 | 2048 | - | 2,370 | 296 | 62.0 | 24.2% | 39.1 GB |
| TP4-PP2 | 1 | 1 | 2048 | Yes | 2,368 | 296 | 62.0 | 24.2% | 39.1 GB |
| SP-TP8 | 1 | 1 | 2048 | Yes | 2,242 | 280 | 58.6 | 22.9% | 35.1 GB |

> **Key findings from 14B/32B testing:**
> - **TP4-DP2 is the throughput sweet spot for 14B**: 76.1 TFLOP/s/GPU (29.7% MFU) vs 58.8 for TP8, due to less TP communication
> - **SP has negligible overhead**: SP-TP4-DP2 ≈ TP4-DP2 throughput; SP-TP8 ≈ TP8 throughput for 32B
> - **TP4-PP2 saves memory vs TP8**: 20.0 GB/GPU for 14B, 39.1 GB/GPU for 32B — useful for running multiple replicas
> - **BS=2 on 32B-TP8-GC** gives +3% throughput (78.5 vs 76.4 TFLOP/s/GPU) with only 5.5 GB extra HBM
> - **GC overhead is ~18%** for 14B/32B (consistent with smaller models: ~46.5→35.1 GB for 32B)
> - **TP4+CP2 combination fails** on HCCL (ring attention doesn't support concurrent TP+CP communicators)
> - **32B-TP8 peak: 76.4 TFLOP/s/GPU** (29.9% MFU on Ascend 910B's 256 TFLOP/s bf16 peak)

> **Key findings from comprehensive testing (40 OK configs across DP/TP/PP/CP/SP, 6 models):**
> - **DP achieves highest compute efficiency** for models fitting in single-NPU: 0.6B reaches 89.8 TFLOP/s/GPU (35% MFU)
> - **Larger models achieve better per-GPU TFLOP utilization** at same TP degree (32B-TP8: 76.4 vs 8B-TP8: 43.1 TFLOP/s/GPU)
> - **SP overhead is negligible** — SP-TP2-DP4 matches or slightly beats TP2-DP4 across all model sizes
> - **CP trades throughput for sequence length**: CP2 at SEQ=4096 drops to ~20–26 TFLOP/s/GPU; CP+TP combination fails on HCCL
> - **TP4-PP2** reduces HBM/GPU vs pure TP: 20.0 GB for 14B, 39.1 GB for 32B, 13.4 GB for 8B
> - **GC cost**: ~18% throughput overhead for 20–30% memory saving, consistent across all model sizes
> - **PP+DP** combinations fail on Ascend HCCL (see Known Limitations); PP without DP works correctly
> - **Peak observed: 143.3 TFLOP/s/GPU** (0.6B single-NPU SEQ=16K, 56% MFU); **peak 8-NPU: 89.8 TFLOP/s** (0.6B-DP8, 35% MFU)

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
- **CP + TP on Ascend HCCL**: Combining context parallelism with tensor parallelism (e.g. TP4-CP2-DP1) also raises `HCCL ERR99999`. CP currently only works with pure DP parallelism (TP=1). The ring attention implementation needs changes to support concurrent TP+CP communicator use.
- **Expert Parallelism (EP)**: Placeholder only — not yet implemented.
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
| Qwen3 | `qwen3` | Explicit (128) | Yes | Varies | 0.6B-8B |

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
│   ├── process_group.py     # 4D process group manager (proxy pattern)
│   ├── tensor_parallel/     # Column/row linear, embedding sharding
│   ├── pipeline_parallel/   # 1F1B, AFAB schedules
│   ├── context_parallel/    # Ring Attention sequence parallelism
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
