"""DTensor basics demo on Ascend NPU / CUDA.

DTensor (Distributed Tensor) is the fundamental building block for all PyTorch
parallelism strategies — TP, SP, FSDP, PP all work by manipulating DTensors
under the hood.

This demo shows the three key placement types on a 1-D DeviceMesh:
  * ``Shard(dim)``    – tensor split along dim across devices
  * ``Replicate()``   – tensor copied to every device
  * ``Partial()``     – tensor is partial-summed (used for gradients)

Each placement is demonstrated with its DTensorSpec — the metadata that
describes global tensor shape, local shard placement, and device mesh.

Example for 8 devices:
  mesh_shape = (8,) → each rank holds 1/8 of a sharded 16-element tensor.

Reference:
  https://pytorch.org/docs/stable/distributed.tensor.html
"""

import os
import sys

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor, Partial, Replicate, Shard, distribute_tensor
from torch.distributed.device_mesh import init_device_mesh

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
            "[dtensor] torch_npu found but NPU is not available. "
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

# -- build 1-D mesh for DTensor ---------------------------------------------
mesh = init_device_mesh(
    device_type,
    mesh_shape=(num_devices,),
    mesh_dim_names=("shard",),
)


# -- helper ----------------------------------------------------------------
def _show_spec(name: str, dt: DTensor):
    """Print DTensorSpec for one DTensor on rank 0 only (to avoid noise)."""
    if rank == 0:
        s = dt._spec
        placements = [
            f"{type(p).__name__}({p.dim})" if isinstance(p, Shard) else type(p).__name__
            for p in s.placements
        ]
        print(
            f"  [{name}] spec: global_shape={tuple(s.tensor_meta.shape)}, "
            f"placements=[{', '.join(placements)}], "
            f"mesh_shape={tuple(s.mesh.shape)}",
            flush=True,
        )


# -- Placement 1: Shard via distribute_tensor -------------------------------
full_tensor = torch.arange(16, device=device_mod.current_device()).float()
sharded = distribute_tensor(full_tensor, mesh, [Shard(0)])
_show_spec("Shard(0)", sharded)
print(
    f"[rank={rank}] Shard(0): {num_devices} ranks, "
    f"global=16 elems, local={sharded.to_local().tolist()}",
    flush=True,
)

# -- Placement 2: Replicate via distribute_tensor ---------------------------
replicated = distribute_tensor(full_tensor, mesh, [Replicate()])
_show_spec("Replicate", replicated)
assert replicated.to_local().numel() == 16, "Replicate should hold all elements"
print(
    f"[rank={rank}] Replicate: all {num_devices} ranks hold full 16 elements ✓",
    flush=True,
)

# -- Placement 3: Partial via DTensor.from_local ----------------------------
elem_per_rank = 8
local_ones = torch.ones(elem_per_rank, device=device_mod.current_device()) * (rank + 1)
partial_t = DTensor.from_local(local_ones, mesh, [Partial()])
_show_spec("Partial (before reduce)", partial_t)
# redistribute Partial → Shard(0): reduce-scatter.
partial_to_shard = partial_t.redistribute(mesh, [Shard(0)])
_show_spec("Partial→Shard (after reduce-scatter)", partial_to_shard)
print(
    f"[rank={rank}] Partial→Shard: reduce-scatter, "
    f"local={partial_to_shard.to_local().tolist()}",
    flush=True,
)

# -- Redistribution round-trip: Shard(0) → Replicate → Shard(0) -------------
shard0 = distribute_tensor(full_tensor, mesh, [Shard(0)])
repl = shard0.redistribute(mesh, [Replicate()])
new_shard0 = repl.redistribute(mesh, [Shard(0)])
assert repl.to_local().numel() == 16, "Redistribute to Replicate failed"
assert torch.equal(new_shard0.to_local(), shard0.to_local()), (
    f"Redistribute round-trip mismatch: rank={rank}"
)
if rank == 0:
    print(
        f"  [Redistribute] Shard(0)→Replicate: all-gather, "
        f"Replicate→Shard(0): reduce-scatter ✓",
        flush=True,
    )
print(
    f"[rank={rank}] Redistribute: Shard(0)→Replicate→Shard(0) round-trip ✓",
    flush=True,
)

# -- Smoke test: all-reduce via Partial → Replicate -------------------------
local_grad = torch.ones(4, device=device_mod.current_device()) * (rank + 1)
partial_grad = DTensor.from_local(local_grad, mesh, [Partial()])
full_grad = partial_grad.redistribute(mesh, [Replicate()])
_show_spec("Smoke: Partial→Replicate (all-reduce)", full_grad)
expected = float(sum(r + 1 for r in range(num_devices)))
assert abs(full_grad.to_local()[0].item() - expected) < 0.5, (
    f"Partial all-reduce mismatch: rank={rank} "
    f"got={full_grad.to_local()[0].item():.1f} expected={expected:.1f}"
)
print(
    f"[rank={rank}] Smoke test: 1+2+...+{num_devices} = "
    f"{full_grad.to_local()[0].item():.1f} ✓ (all-reduce via Partial)",
    flush=True,
)

dist.destroy_process_group()
print(f"[rank={rank}] Done.", flush=True)
