# ScaleTorch

4D parallelism distributed training framework built on PyTorch. Implements **Tensor Parallelism (TP)**, **Pipeline Parallelism (PP)**, **Context Parallelism (CP)**, and **Data Parallelism (DP)** in a unified process grid `[DP, PP, CP, TP]`.

## Features

- **4D Process Grid** — `ProcessGroupManager` orchestrates DP/PP/CP/TP groups from a single `world_size` constraint
- **Tensor Parallelism** — Column/row parallel linear layers, embedding & layer norm sharding
- **Pipeline Parallelism** — 1F1B and AFAB schedules with stage partitioning
- **Context Parallelism** — Sequence-length partitioning via Ring Attention
- **Data Parallelism** — Gradient bucketing and overlapped all-reduce (Megatron-style)
- **Model Support** — LLaMA (with GQA/KV-head configs), Mixture-of-Experts (MoE)
- **Attention Variants** — MHA, MQA, GQA, MLA (Multi-head Latent Attention)
- **Distributed Primitives** — `scaletorch.dist`: all_gather, all_reduce, all_to_all, broadcast, scatter, reduce_scatter, and more
- **Config** — HuggingFace `HfArgumentParser` dataclasses for model, parallelism, optimizer, LR, training, checkpoint, and logging args
- **Checkpointing** — Weight materialization/dematerialization via `CheckpointManager`

## Installation

```bash
pip install -e .
```

Optional dependencies:

```bash
pip install -e ".[flash-attn]"   # Flash Attention
pip install -e ".[wandb]"        # experiment tracking
pip install -e ".[hydra]"        # Hydra config
pip install -e ".[dev]"          # pre-commit, flake8, yapf, isort
```

## Quick Start

Single-node training with 4D parallelism:

```bash
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node 4 train.py \
  --model_name_or_path gpt2 \
  --batch_size 32 \
  --tensor_parallel_size 2 \
  --data_parallel_size 2
```

Launch scripts in `scripts/torch_dist/`:

```bash
bash scripts/torch_dist/launch_single_node.sh
bash scripts/torch_dist/launch_multi_nodes.sh
```

## Project Structure

```
scaletorch/
├── dist/                    # Distributed communication primitives
├── models/
│   ├── attention/           # MHA, MQA, GQA, MLA
│   ├── model_llama.py       # LLaMA transformer
│   └── moe_model.py         # Mixture-of-Experts
├── parallel/
│   ├── pg_manager.py        # 4D process group manager
│   ├── tensor_parallel/     # Column/row linear, embedding sharding
│   ├── pipeline_parallel/   # 1F1B, AFAB schedules
│   ├── context_parallel/    # Ring Attention sequence parallelism
│   └── data_parallel/       # Gradient bucketing
├── trainer/
│   └── config.py            # HfArgumentParser dataclass configs
├── utils/
│   └── checkpoint.py        # CheckpointManager
└── data/                    # Dataset/dataloader utilities
```

## Testing

```bash
python run_tests.py                                        # all tests
python -m unittest tests.test_dist -v                      # single module
python -m unittest discover -s tests -p "test_*.py" -v     # manual discovery
```

## Examples

| Directory | Description |
|-----------|-------------|
| `examples/mnist/` | Basic & multi-GPU MNIST training |
| `examples/fsdp/` | FSDP training example |
| `examples/imagenet/` | Distributed ImageNet training |
| `examples/mingpt/` | minGPT training with GPT-2 |
| `examples/picotron/` | Picotron-style 4D parallel training |
| `examples/mmengine/` | mmengine integration example |

## Documentation

Technical docs in `doc/` (Chinese):

- `doc/parallel/` — Overview, communication primitives, and detailed docs for each parallelism strategy
- `doc/ddp_intro.md`, `doc/fsdp_intro.md` — DDP and FSDP introductions

## References

### Training Frameworks

- [Nanotron](https://github.com/huggingface/nanotron) — HuggingFace LLM training framework
- [Picotron](https://github.com/huggingface/picotron) — Minimalistic 4D-parallelism framework
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) — NVIDIA LLM training framework
- [DeepSpeed](https://www.deepspeed.ai/) — Microsoft deep learning optimization library
- [torchtitan](https://github.com/pytorch/torchtitan) — PyTorch native large model training
- [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) — EleutherAI LLM framework
- [LitGPT](https://github.com/Lightning-AI/litgpt) — Lightning AI LLM implementations
- [FairScale](https://github.com/facebookresearch/fairscale/tree/main) — Facebook large-scale training extensions
- [Colossal-AI](https://colossalai.org/) — Integrated large-scale training system
- [OpenDiLoCo](https://github.com/PrimeIntellect-ai/OpenDiLoCo) — Distributed training with DiLoCo
- [torchgpipe](https://github.com/kakaobrain/torchgpipe) — GPipe in PyTorch
- [OSLO](https://github.com/EleutherAI/oslo) — Large-scale optimization framework

### Parallelism Techniques

- [Data Parallelism](https://siboehm.com/articles/22/data-parallel-training) — Data parallel training explained
- [ZeRO](https://arxiv.org/abs/1910.02054) — Zero Redundancy Optimizer
- [FSDP](https://arxiv.org/abs/2304.11277) — Fully Sharded Data Parallel
- [Tensor & Sequence Parallelism](https://arxiv.org/abs/2205.05198) — TP + SP + selective recomputation
- [Pipeline Parallelism](https://developer.nvidia.com/blog/scaling-language-model-training-to-a-trillion-parameters-using-megatron/#pipeline_parallelism) — NVIDIA PP guide
- [Breadth-first Pipeline Parallelism](https://arxiv.org/abs/2211.05953) — PP schedule analysis
- [Ring All-Reduce](https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/) — Ring all-reduce algorithm
- [Ring Flash Attention](https://github.com/zhuzilin/ring-flash-attention) — Ring Attention + FlashAttention
- [Ring Attention Tutorial](https://coconut-mode.com/posts/ring-attention/) — Ring Attention concepts
- [ZeRO and 3D Parallelism](https://www.deepspeed.ai/tutorials/large-models-w-deepspeed/#understanding-performance-tradeoff-between-zero-and-3d-parallelism) — ZeRO vs 3D trade-offs
- [Mixed Precision Training](https://arxiv.org/abs/1710.03740) — Mixed precision techniques
- [Visualizing 6D Mesh Parallelism](https://main-horse.github.io/posts/visualizing-6d/) — 6D parallel mesh communication

### Landmark Papers

- [Megatron-LM](https://arxiv.org/abs/1909.08053) — Tensor parallelism for LLMs
- [Megatron-Turing NLG 530B](https://developer.nvidia.com/blog/using-deepspeed-and-megatron-to-train-megatron-turing-nlg-530b-the-worlds-largest-and-most-powerful-generative-language-model/) — 530B model training
- [PaLM](https://arxiv.org/abs/2204.02311) — Google Pathways Language Model
- [Gemini](https://arxiv.org/abs/2312.11805) — Google multimodal model
- [Llama 3](https://arxiv.org/abs/2407.21783) — Llama 3 herd of models
- [DeepSeek-V3](https://arxiv.org/abs/2412.19437v1) — DeepSeek-V3 architecture and training

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

[Apache 2.0](LICENSE)
