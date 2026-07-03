"""Tensor Parallel (TP) demo on Ascend NPU / CUDA.

Megatron-LM SPMD style: column-wise shard the first linear layer, row-wise
shard the second, avoiding communication between the two layers.
Only one all-reduce is needed at the output of the second layer.

Example for 4 devices:
  mesh_shape = (4,) → 1-D tp mesh, model sharded across 4 ranks.

Reference:
  https://github.com/pytorch/pytorch/issues/89884
  https://arxiv.org/abs/1909.08053
"""

import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
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
            "[tp_demo] torch_npu found but NPU is not available. "
            "Is the CANN toolkit sourced?\n"
        )

# -- bootstrap --------------------------------------------------------------
rank = int(os.environ["RANK"])
local_rank = int(os.environ.get("LOCAL_RANK", os.environ["RANK"]))
world_size = int(os.environ['WORLD_SIZE'])

dist.init_process_group(backend)
device_mod.set_device(local_rank)

num_devices = device_mod.device_count()
print(
    f"[rank={rank}] local_rank={local_rank} world_size={world_size} "
    f"device={device_type} device_count={num_devices}",
    flush=True,
)

# -- build TP mesh ----------------------------------------------------------
if num_devices % 2 != 0:
    dist.destroy_process_group()
    sys.exit(
        f"[tp_demo] device_count={num_devices} is odd — "
        "TP demo requires an even number of devices.\n"
    )

mesh_1d = init_device_mesh(
    device_type,
    mesh_shape=(num_devices,),
    mesh_dim_names=("tp",),
)

print(
    f"[rank={rank}] TP mesh: {num_devices} devices "
    f"(tp={mesh_1d['tp']})",
    flush=True,
)

# -- model ------------------------------------------------------------------
class ToyModel(nn.Module):
    """MLP model for TP: colwise shard in_proj, rowwise shard out_proj."""

    def __init__(self):
        super().__init__()
        self.in_proj = nn.Linear(10, 32)
        self.relu = nn.ReLU()
        self.out_proj = nn.Linear(32, 5)

    def forward(self, x):
        return self.out_proj(self.relu(self.in_proj(x)))


model = ToyModel().to(device_mod.current_device())

# Apply TP parallelization plan.
tp_model = parallelize_module(
    module=model,
    device_mesh=mesh_1d,
    parallelize_plan={
        "in_proj": ColwiseParallel(),
        "out_proj": RowwiseParallel(),
    },
)

# -- smoke test: forward + backward + optimizer step ------------------------
lr = 0.25
optimizer = torch.optim.AdamW(tp_model.parameters(), lr=lr, foreach=True)

num_iters = 10
for i in range(num_iters):
    # TP requires identical input across all ranks — seed mimics dataloader.
    torch.manual_seed(i)
    inp = torch.rand(20, 10, device=device_mod.current_device())
    output = tp_model(inp)
    output.sum().backward()
    optimizer.step()
    optimizer.zero_grad()

print(
    f"[rank={rank}] TP smoke test passed — "
    f"{num_iters} iterations completed on {num_devices} devices ✓",
    flush=True,
)

# Synchronize all ranks before teardown to avoid HCCL hang.
device_mod.synchronize()
dist.barrier()
dist.destroy_process_group()
print(f"[rank={rank}] Done.", flush=True)
