"""FSDP2 training example — Ascend NPU / CUDA.

A self-contained FSDP2 (Fully Sharded Data Parallel v2) training demo on a
toy Transformer model, adapted for both Ascend NPU and CUDA backends.

Features:
  * Auto-detection of NPU (HCCL) vs CUDA (NCCL) backend
  * 1-D DeviceMesh for data-parallel sharding
  * Optional mixed precision (bfloat16 param, float32 reduce)
  * Optional explicit forward/backward prefetching
  * Checkpoint save/load with both DTensor API and DCP API
  * 10 training iterations with gradient clipping

Usage:
  torchrun --nproc_per_node=8 fsdp2_main.py
  torchrun --nproc_per_node=8 fsdp2_main.py --mixed-precision
  torchrun --nproc_per_node=8 fsdp2_main.py --explicit-prefetching
  torchrun --nproc_per_node=8 fsdp2_main.py --dcp-api

Reference:
  https://pytorch.org/docs/stable/distributed.fsdp.html
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
# FSDP2: PyTorch 2.4-2.6 uses _composable.fsdp; 2.7+ uses torch.distributed.fsdp
try:
    from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
except ImportError:
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

from checkpoint import Checkpointer
from model import ModelArgs, Transformer
from utils import inspect_mixed_precision, inspect_model


# -- backend auto-detection (NPU → CUDA fallback) ---------------------------
try:
    import torch_npu  # noqa: F401
except ImportError:
    device_mod, backend, device_type = torch.cuda, "nccl", "cuda"
else:
    if torch.npu.is_available():
        device_mod, backend, device_type = torch.npu, "hccl", "npu"
    else:
        sys.exit(
            "[fsdp2] torch_npu found but NPU is not available. "
            "Is the CANN toolkit sourced?\n"
        )


def verify_min_device_count(min_devices: int = 2) -> bool:
    """Check that we have at least ``min_devices`` accelerators available."""
    return device_mod.device_count() >= min_devices


def set_modules_to_forward_prefetch(model, num_to_forward_prefetch):
    """Configure forward prefetching: each layer prefetches the next N layers."""
    for i, layer in enumerate(model.layers):
        if i >= len(model.layers) - num_to_forward_prefetch:
            break
        layers_to_prefetch = [
            model.layers[i + j] for j in range(1, num_to_forward_prefetch + 1)
        ]
        layer.set_modules_to_forward_prefetch(layers_to_prefetch)


def set_modules_to_backward_prefetch(model, num_to_backward_prefetch):
    """Configure backward prefetching: each layer prefetches the previous N layers."""
    for i, layer in enumerate(model.layers):
        if i < num_to_backward_prefetch:
            continue
        layers_to_prefetch = [
            model.layers[i - j] for j in range(1, num_to_backward_prefetch + 1)
        ]
        layer.set_modules_to_backward_prefetch(layers_to_prefetch)


def main(args):
    # -- guard: minimum device count -----------------------------------------
    if not verify_min_device_count(min_devices=2):
        print(
            f"[fsdp2] Need at least 2 {device_type.upper()} devices, "
            f"found {device_mod.device_count()}. Exiting.",
            flush=True,
        )
        sys.exit(1)

    # -- bootstrap -----------------------------------------------------------
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ["RANK"]))
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(backend)
    device_mod.set_device(local_rank)
    device = torch.device(f"{device_type}:{local_rank}")

    print(
        f"[rank={rank}] local_rank={local_rank} world_size={world_size} "
        f"device={device_type} device_count={device_mod.device_count()}",
        flush=True,
    )

    # -- 1-D device mesh for FSDP2 data-parallel sharding --------------------
    mesh = init_device_mesh(
        device_type,
        mesh_shape=(world_size,),
        mesh_dim_names=("dp",),
    )

    # -- model (created on meta device, then materialised on NPU/CUDA) --------
    torch.manual_seed(0)
    vocab_size = 1024
    batch_size = 32
    seq_len = 64
    model_args = ModelArgs(
        n_layers=10,
        n_heads=4,
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        dropout_p=0,
    )
    with torch.device("meta"):
        model = Transformer(model_args)

    # -- build FSDP2 kwargs --------------------------------------------------
    fsdp_kwargs = {"mesh": mesh}
    if args.mixed_precision:
        fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )

    # Apply FSDP2: shard each transformer layer individually, then the top-level model.
    for layer in model.layers:
        fully_shard(layer, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)

    inspect_model(model)

    # -- explicit prefetching (optional) -------------------------------------
    if args.explicit_prefetching:
        set_modules_to_forward_prefetch(model, num_to_forward_prefetch=2)
        set_modules_to_backward_prefetch(model, num_to_backward_prefetch=2)

    # -- checkpoint: load or initialise --------------------------------------
    checkpointer = Checkpointer("checkpoints", dcp_api=args.dcp_api)
    if checkpointer.is_empty():
        model.to_empty(device=device)
        model.reset_parameters()
    else:
        checkpointer.load_model(model)

    if args.mixed_precision:
        inspect_mixed_precision(model)

    # -- optimizer -----------------------------------------------------------
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    if not checkpointer.is_empty():
        checkpointer.load_optim(model, optim)

    # -- training loop (10 iterations) ---------------------------------------
    for step in range(10):
        if args.explicit_prefetching:
            model.unshard()
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        loss = model(x).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        optim.zero_grad()

        if rank == 0:
            print(
                f"  [step {step:2d}] loss={loss.item():.4f}",
                flush=True,
            )

    # -- save checkpoint & cleanup -------------------------------------------
    checkpointer.save(model, optim)
    dist.destroy_process_group()

    if rank == 0:
        print("[fsdp2] Training complete ✓", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch FSDP2 example (NPU/CUDA)")
    parser.add_argument(
        "--explicit-prefetching", action="store_true", default=False,
        help="Enable explicit forward/backward prefetching",
    )
    parser.add_argument(
        "--mixed-precision", action="store_true", default=False,
        help="Enable bfloat16 mixed precision training",
    )
    parser.add_argument(
        "--dcp-api", action="store_true", default=False,
        help="Use DCP (Distributed Checkpoint) API for save/load",
    )
    args = parser.parse_args()

    main(args)
