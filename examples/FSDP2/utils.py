"""Debug helpers for FSDP2 training on Ascend NPU / CUDA."""

import torch
from model import Transformer
# FSDP2: PyTorch 2.4-2.6 uses _composable.fsdp; 2.7+ uses torch.distributed.fsdp
try:
    from torch.distributed._composable.fsdp import FSDPModule
except ImportError:
    from torch.distributed.fsdp import FSDPModule


def inspect_model(model: FSDPModule):
    """Print FSDP2 model structure and parameter sharding info (rank 0 only)."""
    assert isinstance(model, Transformer), (
        f"Expected Transformer model, got {type(model).__name__}"
    )
    assert isinstance(model, FSDPModule), (
        f"Expected FSDPModule, got {type(model).__name__}"
    )

    if torch.distributed.get_rank() == 0:
        print(model)
        print("--- Parameter sharding info ---")
        for name, param in model.named_parameters():
            placements = getattr(param, "placements", None)
            local_shape = tuple(param.shape)
            print(
                f"  {name}: local_shape={local_shape}, "
                f"dtype={param.dtype}, placements={placements}"
            )
        print("--- End parameter info ---")


def inspect_mixed_precision(model: FSDPModule):
    """Verify mixed precision: top-level params should be in param_dtype after unshard."""
    try:
        model.unshard()
    except AttributeError:
        print("[inspect_mixed_precision] model.unshard() not available, skipping")
        return

    for name, param in model.named_parameters(recurse=False):
        print(
            f"  [mp] {name}: dtype={param.dtype}, device={param.device}"
        )

    try:
        model.reshard()
    except AttributeError:
        pass
