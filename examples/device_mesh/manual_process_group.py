"""2-D parallelism process group setup demo on Ascend NPU / CUDA.

Creates two process-group dimensions from a flat 1-D world:
  * shard groups     – split devices into halves (tensor / expert parallelism axis)
  * replicate groups – pair ranks across halves (data / FSDP axis)

Example for 8 devices:
  shard     : (0,1,2,3)  and  (4,5,6,7)
  replicate : (0,4), (1,5), (2,6), (3,7)
"""

import os
import sys

import torch
import torch.distributed as dist

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
            "[2d_setup] torch_npu found but NPU is not available. "
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
        f"[2d_setup] device_count={num_devices} is odd — "
        "2-D split requires an even number of devices.\n"
    )

# Create shard groups (e.g. (0,1,2,3), (4,5,6,7))
# and assign the correct shard group to each rank.
shard_rank_lists = (
    list(range(0, num_devices // 2)),
    list(range(num_devices // 2, num_devices)),
)
shard_groups = (
    dist.new_group(shard_rank_lists[0]),
    dist.new_group(shard_rank_lists[1]),
)
current_shard_group = (
    shard_groups[0] if rank in shard_rank_lists[0] else shard_groups[1]
)

# Create replicate groups (e.g. (0,4), (1,5), (2,6), (3,7))
# and assign the correct replicate group to each rank.
current_replicate_group = None
current_replicate_ranks = None
shard_factor = len(shard_rank_lists[0])
for i in range(num_devices // 2):
    replicate_group_ranks = list(range(i, num_devices, shard_factor))
    replicate_group = dist.new_group(replicate_group_ranks)
    if rank in replicate_group_ranks:
        current_replicate_group = replicate_group
        current_replicate_ranks = replicate_group_ranks

# ===========================================================================
# Smoke test: all-reduce within the shard group
# ===========================================================================
my_shard_ranks = (
    shard_rank_lists[0] if rank in shard_rank_lists[0] else shard_rank_lists[1]
)

tensor = torch.ones(1, device=device_mod.current_device()) * (rank + 1)
dist.all_reduce(tensor, group=current_shard_group)
expected = float(sum(r + 1 for r in my_shard_ranks))

assert abs(tensor.item() - expected) < 0.5, (
    f"all_reduce mismatch: rank={rank} shard={my_shard_ranks} "
    f"got={tensor.item():.1f} expected={expected:.1f}"
)

print(
    f"[rank={rank}] shard_group={my_shard_ranks} "
    f"replicate_group={current_replicate_ranks} "
    f"all_reduce={tensor.item():.1f} ✓",
    flush=True,
)

dist.destroy_process_group()
print(f"[rank={rank}] Done.", flush=True)
