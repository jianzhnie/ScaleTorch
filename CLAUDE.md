# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Install

```bash
pip install -e .                  # install package in editable mode
pip install -r requirements.txt   # full dependencies (flash-attn, hydra-core, wandb, etc.)
pre-commit install                # install git hooks
```

## Test

```bash
python run_tests.py                                          # run all tests
python -m unittest discover -s tests -p "test_*.py" -v      # manual run
python -m unittest tests.test_dist -v                        # single test module
```

Tests use **unittest** (not pytest). Base class: `tests/test_base.py` — `BaseTestCase` with helpers for mocking process groups and distributed ops.

## Lint & Format

```bash
pre-commit run --all-files   # run all hooks
flake8 .                     # linter (max 79 chars, ignores W503/W504/E251/E501/E126)
```

Formatter: **yapf** + **isort** via pre-commit. Double quotes for strings. Absolute imports. LF line endings.

## Running Training

```bash
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node 4 train.py \
  --model_name_or_path gpt2 --batch_size 32 \
  --tensor_parallel_size 2 --data_parallel_size 2
```

Launch scripts in `scripts/torch_dist/` for single-node and multi-node setups.

## Architecture

ScaleTorch implements **4D parallelism** — a process grid of `[DP, PP, CP, TP]`:

- **`scaletorch/parallel/pg_manager.py`** — `ProcessGroupManager`: creates and manages the 4D process group grid. All parallelism modules depend on it.
- **`scaletorch/parallel/tensor_parallel/`** — Tensor parallelism (column/row linear, embedding, layer norm sharding)
- **`scaletorch/parallel/context_parallel/`** — Context (sequence) parallelism with Ring Attention
- **`scaletorch/parallel/pipeline_parallel/`** — Pipeline parallelism with 1F1B and AFAB schedules
- **`scaletorch/parallel/data_parallel/`** — Data parallelism with gradient bucketing

Other key modules:
- **`scaletorch/dist/`** — Low-level distributed primitives (all_gather, all_reduce, broadcast, scatter, all_to_all, etc.) and env utilities (`init_dist`, `get_rank`, `infer_launcher`)
- **`scaletorch/models/`** — Model architectures: Llama, MoE, attention variants (MHA/MQA/GQA/MLA)
- **`scaletorch/trainer/config.py`** — Dataclass configs parsed via `HfArgumentParser`: `ScaleTorchArguments` aggregates `ModelArguments`, `ParallelArguments`, `TrainingArguments`, etc.
- **`scaletorch/utils/checkpoint.py`** — `CheckpointManager` with weight materialization/dematerialization
- **`train.py`** — Main entry point: config parsing → dist init → model creation (with TP/PP/CP/DP) → optimizer → training loop → checkpointing → wandb logging

## Conventions

- Python >=3.10, PyTorch + HuggingFace Transformers
- Config uses HuggingFace `HfArgumentParser` with dataclass argument groups
- Distributed code uses `scaletorch.dist` utilities, not raw `torch.distributed` directly
- Models inherit `torch.nn.Module`; parallelism modules hook into model layers via `ProcessGroupManager`
- Documentation in `doc/` (Chinese), covering all parallelism strategies in detail
