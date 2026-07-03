"""HSDP (Hybrid Sharding Data Parallel) demo on Ascend NPU / CUDA.

Uses a 2-D DeviceMesh to combine FSDP sharding with data-parallel replication:
  * ``dp_replicate`` – replicate the sharded model across groups →
    gradient all-reduce across replicas (cross-host / slow link)
  * ``dp_shard``      – shard parameters within each replica group →
    FSDP all-gather / reduce-scatter inside a host (fast NVLink / HCCS)

Example for 8 devices with mesh_shape=(2, 4):
  2 replica groups × 4 shard ranks per group
"""

import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

# -- detect backend ---------------------------------------------------------
try:
    import torch_npu  # noqa: F401
except ImportError:
    device_mod, backend, device_type = torch.cuda, "nccl", "cuda"
else:
    if torch.npu.is_available():
        device_mod, backend, device_type = torch.npu, "hccl", "npu"
    else:
        sys.exit(
            "[hsdp] torch_npu found but NPU is not available. "
            "Is the CANN toolkit sourced?\n"
        )

# -- bootstrap --------------------------------------------------------------
rank = int(os.environ["RANK"])
local_rank = int(os.environ.get("LOCAL_RANK", os.environ["RANK"]))

dist.init_process_group(backend)
device_mod.set_device(local_rank)

num_devices = device_mod.device_count()
print(
    f"[rank={rank}] local_rank={local_rank} world_size={os.environ['WORLD_SIZE']} "
    f"device={device_type} device_count={num_devices}",
    flush=True,
)

# -- build HSDP mesh --------------------------------------------------------
if num_devices % 2 != 0:
    dist.destroy_process_group()
    sys.exit(
        f"[hsdp] device_count={num_devices} is odd — "
        "HSDP requires an even number of devices.\n"
    )

replicate_size = 2
shard_size = num_devices // replicate_size

mesh_2d = init_device_mesh(
    device_type,
    mesh_shape=(replicate_size, shard_size),
    mesh_dim_names=("dp_replicate", "dp_shard"),
)

print(
    f"[rank={rank}] HSDP mesh: {replicate_size}×{shard_size} "
    f"(replicate={mesh_2d['dp_replicate']} shard={mesh_2d['dp_shard']})",
    flush=True,
)


# -- model ----------------------------------------------------------------
class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


model = ToyModel().to(device_mod.current_device())
fsdp_model = fully_shard(model, mesh=mesh_2d)

# -- smoke test: forward + backward ----------------------------------------
optimizer = torch.optim.SGD(fsdp_model.parameters(), lr=0.01)
x = torch.randn(8, 10, device=device_mod.current_device())
loss = fsdp_model(x).sum()
loss.backward()
optimizer.step()

print(
    f"[rank={rank}] HSDP smoke test passed — "
    f"loss={loss.item():.4f} grad_norm={sum(p.grad.norm().item() for p in fsdp_model.parameters()):.4f} ✓",
    flush=True,
)

dist.destroy_process_group()
print(f"[rank={rank}] Done.", flush=True)
