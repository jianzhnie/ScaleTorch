"""Sequence Parallel (SP) demo on Ascend NPU / CUDA.

Megatron-LM SPMD style with sequence-dimension sharding.
Unlike Tensor Parallel, each rank receives a *different* shard of the input
along the sequence dimension, trading an all-gather (input) + reduce-scatter
(output) for higher memory efficiency on large sequence lengths.

Example for 4 devices:
  mesh_shape = (4,) → 1-D sp mesh, sequence sharded across 4 ranks.

Reference:
  https://arxiv.org/pdf/2205.05198.pdf
"""

import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._tensor import Shard
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

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
            "[sp_demo] torch_npu found but NPU is not available. "
            "Is the CANN toolkit sourced?\n"
        )

# -- bootstrap --------------------------------------------------------------
rank = int(os.environ["RANK"])
local_rank = int(os.environ.get("LOCAL_RANK", os.environ["RANK"]))
world_size = int(os.environ["WORLD_SIZE"])

dist.init_process_group(backend)
device_mod.set_device(local_rank)

num_devices = device_mod.device_count()
print(
    f"[rank={rank}] local_rank={local_rank} world_size={world_size} "
    f"device={device_type} device_count={num_devices}",
    flush=True,
)

# -- build SP mesh ----------------------------------------------------------
if num_devices % 2 != 0:
    dist.destroy_process_group()
    sys.exit(
        f"[sp_demo] device_count={num_devices} is odd — "
        "SP demo requires an even number of devices.\n"
    )

mesh_1d = init_device_mesh(
    device_type,
    mesh_shape=(num_devices,),
    mesh_dim_names=("sp",),
)

print(
    f"[rank={rank}] SP mesh: {num_devices} devices "
    f"(sp={mesh_1d['sp']})",
    flush=True,
)

# -- model ------------------------------------------------------------------
class ToyModel(nn.Module):
    """MLP model for SP: colwise shard in_proj with Shard(0) input,
    rowwise shard out_proj with Shard(0) output."""

    def __init__(self):
        super().__init__()
        self.in_proj = nn.Linear(10, 32)
        self.relu = nn.ReLU()
        self.out_proj = nn.Linear(32, 5)

    def forward(self, x):
        return self.out_proj(self.relu(self.in_proj(x)))


model = ToyModel().to(device_mod.current_device())

# Apply SP parallelization plan.
# Shard(0): shard along sequence dimension (dim 0 of input/output).
sp_model = parallelize_module(
    module=model,
    device_mesh=mesh_1d,
    parallelize_plan={
        "in_proj": ColwiseParallel(input_layouts=Shard(0)),
        "out_proj": RowwiseParallel(output_layouts=Shard(0)),
    },
)

# -- smoke test: forward + backward + optimizer step ------------------------
lr = 0.25
optimizer = torch.optim.AdamW(sp_model.parameters(), lr=lr, foreach=True)

num_iters = 10
for i in range(num_iters):
    # SP allows different input per rank — no seed needed.
    inp = torch.rand(20, 10, device=device_mod.current_device())
    output = sp_model(inp)
    output.sum().backward()
    optimizer.step()
    optimizer.zero_grad()

print(
    f"[rank={rank}] SP smoke test passed — "
    f"{num_iters} iterations completed on {num_devices} devices ✓",
    flush=True,
)

dist.destroy_process_group()
print(f"[rank={rank}] Done.", flush=True)
