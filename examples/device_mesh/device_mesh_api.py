"""2-D DeviceMesh setup demo on Ascend NPU / CUDA.

Equivalent to 2d_setup.py but uses ``init_device_mesh`` instead of manual
``dist.new_group()`` calls.

Example for 8 devices:
  mesh_shape = (2, 4) → 2 replicate dim × 4 shard dim
"""

import os
import sys

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

# ===========================================================================
# Detect backend (NPU → hccl, otherwise CUDA → nccl)
# ===========================================================================
try:
    import torch_npu  # noqa: F401
except ImportError:
    device_mod, backend, device_type = torch.cuda, "nccl", "cuda"
else:
    if torch.npu.is_available():
        device_mod, backend, device_type = torch.npu, "hccl", "npu"
    else:
        sys.exit(
            "[2d_device_mesh] torch_npu found but NPU is not available. "
            "Is the CANN toolkit sourced?\n"
        )

# ===========================================================================
# Bootstrap
# ===========================================================================
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
local_rank = int(os.environ.get("LOCAL_RANK", os.environ["RANK"]))

dist.init_process_group(backend)
device_mod.set_device(local_rank)
num_devices = device_mod.device_count()

print(
    f"[rank={rank}] local_rank={local_rank} world_size={world_size} "
    f"device={device_type} device_count={num_devices}",
    flush=True,
)

# ===========================================================================
# 2-D process groups
# ===========================================================================
if num_devices % 2 != 0:
    dist.destroy_process_group()
    sys.exit(
        f"[2d_device_mesh] device_count={num_devices} is odd — "
        "2-D split requires an even number of devices.\n"
    )

shard_size = num_devices // 2

mesh_2d = init_device_mesh(
    device_type,
    mesh_shape=(2, shard_size),
    mesh_dim_names=("replicate", "shard"),
)

shard_group = mesh_2d.get_group(mesh_dim="shard")
replicate_group = mesh_2d.get_group(mesh_dim="replicate")

# Compute shard / replicate rank lists for readable output.
shard_ranks = (
    list(range(0, shard_size)) if rank < shard_size
    else list(range(shard_size, num_devices))
)
replicate_ranks = (
    [rank, rank + shard_size] if rank < shard_size
    else [rank - shard_size, rank]
)

# ===========================================================================
# Smoke test: all-reduce within the shard group
# ===========================================================================
tensor = torch.ones(1, device=device_mod.current_device()) * (rank + 1)
dist.all_reduce(tensor, group=shard_group)
expected = float(sum(r + 1 for r in shard_ranks))

assert abs(tensor.item() - expected) < 0.5, (
    f"all_reduce mismatch: rank={rank} shard={shard_ranks} "
    f"got={tensor.item():.1f} expected={expected:.1f}"
)

print(
    f"[rank={rank}] shard_group={shard_ranks} "
    f"replicate_group={replicate_ranks} "
    f"all_reduce={tensor.item():.1f} ✓",
    flush=True,
)

dist.destroy_process_group()
print(f"[rank={rank}] Done.", flush=True)
