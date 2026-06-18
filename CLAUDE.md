# CLAUDE.md

Guidance for Claude Code working in this repo.

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

Formatter: **yapf** + **isort** via pre-commit. Double quotes. Absolute imports. LF line endings.

## Running Training

```bash
# NPU (Ascend) — source CANN env first
source set_env.sh
torchrun --nproc_per_node 8 tools/train.py \
  --model_name_or_path /path/to/Qwen3-0.6B \
  --data_parallel_size 8 \
  --micro_batch_size 2 --sequence_length 2048

# CUDA
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node 4 tools/train.py \
  --model_name_or_path gpt2 --micro_batch_size 32 \
  --tensor_parallel_size 2 --data_parallel_size 2
```

Launch scripts in `scripts/torch_dist/` for single-node and multi-node setups.
Comprehensive benchmark: `python scripts/benchmark_comprehensive.py --list` / `--filter`.

## Architecture

ScaleTorch implements **4D parallelism** — process grid `[DP, PP, CP, TP]`:

- **`scaletorch/parallel/process_group.py`** — `ProcessGroupManager`: creates/manages 4D process group grid. All parallelism modules depend on it.
- **`scaletorch/parallel/tensor_parallel/`** — Tensor parallelism (column/row linear, embedding, layer norm sharding)
- **`scaletorch/parallel/context_parallel/`** — Context (sequence) parallelism with Ring Attention
- **`scaletorch/parallel/pipeline_parallel/`** — Pipeline parallelism with 1F1B and AFAB schedules
- **`scaletorch/parallel/data_parallel/`** — Data parallelism with gradient bucketing

Other key modules:
- **`scaletorch/dist/`** — Low-level distributed primitives (all_gather, all_reduce, broadcast, scatter, all_to_all, etc.) and env utilities (`init_dist`, `get_rank`, `infer_launcher`)
- **`scaletorch/models/`** — Model architectures: Llama, MoE, attention variants (MHA/MQA/GQA/MLA)
- **`scaletorch/trainer/config.py`** — Dataclass configs via `HfArgumentParser`: `ScaleTorchArguments` aggregates `ModelArguments`, `ParallelArguments`, `TrainingArguments`, etc.
- **`scaletorch/utils/checkpoint.py`** — `CheckpointManager` with weight materialization/dematerialization
- **`tools/train.py`** — Entry point: config → dist init → model creation (TP/PP/CP/DP) → optimizer → training loop → checkpointing → wandb logging

## Conventions

- Python >=3.10, PyTorch + HuggingFace Transformers
- Config via HuggingFace `HfArgumentParser` with dataclass argument groups
- Distributed code uses `scaletorch.dist` utilities, not raw `torch.distributed`
- Models inherit `torch.nn.Module`; parallelism modules hook into model layers via `ProcessGroupManager`
- Documentation in `docs/` (Chinese), covering all parallelism strategies

## Known Issues & Fixes Applied

### P2P Communication on Ascend HCCL
`cp_comms.py` and `pp_comms.py` must pass `torch.distributed.isend`/`irecv` **directly** to `P2POp`, not the `st_dist` wrapper functions. HCCL validates op identity by reference. See:
- `scaletorch/parallel/context_parallel/cp_comms.py` — `import torch.distributed as torch_dist`, use `torch_dist.isend/irecv`
- `scaletorch/parallel/pipeline_parallel/pp_comms.py` — same fix + replaced `batch_isend_irecv` with direct blocking `send/recv` (more stable on HCCL)

### PP + DP Concurrent HCCL Conflict
Pipeline P2P (PP group) and DP allreduce fire concurrently on Ascend HCCL, causing `ERR99999`. Fix:
- `scaletorch/parallel/data_parallel/data_parallel.py` — added `sync_grads_manually()` method
- `scaletorch/parallel/pipeline_parallel/pipeline_parallel.py` — keep `require_backward_grad_sync=False` throughout; call `model.sync_grads_manually()` after all PP P2P comms complete
- **Note**: PP+DP still fails at HCCL process group level on this hardware. PP without DP (e.g. TP4-PP2-DP1) works.

### Qwen3 Pipeline Parallelism
`Qwen3Attention` and `Qwen3RMSNorm` lacked `reset_parameters()` required by `PipelineParallel.__init__`. Added standard weight-init implementations in `scaletorch/models/model_qwen3.py`.

### `model.config` AttributeError After PP/DP Wrapping
After `PipelineParallel` or `DataParallelBucket` wrapping, `model.config` is inaccessible. Fix in `tools/train.py`:
- Load `model_config = AutoConfig.from_pretrained(...)` separately after `create_model()`
- `get_tensor_shapes()` now returns the hidden-state shape tuple `(batch, seq, hidden_size)` instead of a dict

### Benchmark Metric Collection
Performance logs are written by all ranks as `performance_logs_{rank}_{ts}.json`. The benchmark driver (`scripts/benchmark_comprehensive.py`) uses a pre-run file snapshot to identify new log files after each run, avoiding false matches from previous runs.
