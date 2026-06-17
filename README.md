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

## Performance (Ascend 910, bf16, CANN 8.3 + PyTorch 2.5)

### Single-NPU Benchmarks

| Model | SEQ | BS | GC | Tokens/s/GPU | MFU | HBM |
|-------|------|-----|------|-------------|------|-------|
| Qwen3-0.6B | 2048 | 2 | - | 9,731 | 22.5% | 22.2 GB |
| Qwen3-0.6B | 8192 | 1 | Yes | 9,834 | 39.0% | 21.4 GB |
| Qwen3-0.6B | 16384 | 1 | Yes | 9,079 | **56.0%** | 39.2 GB |
| Qwen3-1.7B | 2048 | 1 | - | 4,685 | 24.9% | 23.8 GB |
| Qwen3-1.7B | 2048 | 1 | Yes | 3,162 | 16.8% | 19.5 GB |
| Qwen3-1.7B | 8192 | 1 | Yes | 7,396 | **51.5%** | 32.0 GB |
| Qwen3-4B   | 2048 | 1 | Yes | 2,415 | 28.4% | 38.8 GB |

### 8-NPU Multi-Card Benchmarks

| Model | Parallelism | BS | GA | GC | Total Tok/s | Tok/s/GPU | HBM/GPU |
|-------|------------|-----|-----|------|-----------|----------|---------|
| Qwen3-0.6B | DP8 | 2 | 2 | - | 79,013 | 9,877 | 4.7 GB |
| Qwen3-0.6B | TP2-DP4 | 2 | 1 | - | 36,779 | 4,597 | 4.6 GB |
| Qwen3-1.7B | DP8 | 1 | 2 | Yes | 30,342 | 3,793 | 11.9 GB |
| Qwen3-1.7B | TP2-DP4 | 1 | 1 | - | 18,870 | 2,359 | 11.5 GB |
| Qwen3-4B   | DP8 | 1 | 1 | Yes | 15,735 | 1,967 | 48.4 GB |
| Qwen3-4B   | TP2-DP4 | 1 | 1 | Yes | 8,627 | 1,078 | 24.5 GB |
| Qwen3-4B   | TP4-DP2 | 1 | 1 | - | 7,757 | 970 | 12.6 GB |
| Qwen3-8B   | TP2-DP4 | 1 | 1 | Yes | 7,832 | 979 | 44.9 GB |
| Qwen3-8B   | TP4-DP2 | 1 | 1 | Yes | 4,417 | 552 | 22.8 GB |

> **Key findings:**
> - Longer sequences dramatically improve MFU (22% at SEQ=2048 vs 56% at SEQ=16384) due to higher attention compute density
> - DP scales near-linearly for small models (0.6B: 9.7K/GPU single → 9.9K/GPU 8-card)
> - TP introduces communication overhead; DP8 outperforms TP2-DP4 for models that fit in single-NPU memory
> - Gradient checkpointing reduces memory ~30-40% with ~25-30% throughput overhead
> - Qwen3-8B requires at least TP2 to fit in 64GB HBM

## Installation

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
