# FSDP2 Example

FSDP2 (Fully Sharded Data Parallel v2) training on a toy Transformer model.
Supports both **Ascend NPU** (HCCL) and **CUDA** (NCCL) backends with
auto-detection.

## Quick Start

```bash
cd examples/FSDP2
bash run_example.sh fsdp2_main.py 8
```

- First run creates a `checkpoints/` folder and saves state dicts.
- Second run loads from the previous checkpoint and resumes.
- Minimum 2 devices required.

## Options

```bash
# Enable explicit forward/backward prefetching
bash run_example.sh fsdp2_main.py 8 --explicit-prefetching

# Enable bfloat16 mixed precision
bash run_example.sh fsdp2_main.py 8 --mixed-precision

# Use DCP (Distributed Checkpoint) API
bash run_example.sh fsdp2_main.py 8 --dcp-api
```

## Files

| File | Purpose |
|------|---------|
| `fsdp2_main.py` | Training entry point with NPU/CUDA auto-detection |
| `model.py` | Toy Transformer (attention, FFN, embeddings) |
| `checkpoint.py` | Checkpoint save/load via DTensor API and DCP API |
| `utils.py` | Debug helpers for model inspection |
| `run_example.sh` | Launcher script with CANN env vars |

## Requirements

- PyTorch >= 2.7
- For NPU: CANN toolkit + `torch_npu`
