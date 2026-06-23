"""Utility functions for distributed training: printing, seeds, MFU, parameter counting, loss averaging."""

import builtins
import fcntl
import os
import random
import tempfile
from typing import Any

import numpy as np
import torch

import scaletorch.dist as st_dist
from scaletorch.parallel.process_group import process_group_manager as pgm

# Constants for number formatting
TRILLION = 1e12
BILLION = 1e9
MILLION = 1e6
THOUSAND = 1e3

# Constants for MFU calculation
DEFAULT_THEORETICAL_FLOPS = 989.5 * 10**12  # FLOPS for A100 GPU


def _get_device_peak_flops() -> float:
    """Return theoretical peak bf16/fp16 FLOPS for the current accelerator."""
    try:
        from scaletorch.utils.device import get_theoretical_flops

        return get_theoretical_flops()
    except Exception:
        return DEFAULT_THEORETICAL_FLOPS


# Tensor parallelism keywords for parameter counting
TP_KEYWORDS = [
    ".q_proj.",
    ".k_proj.",
    ".v_proj.",
    ".out_proj.",
    ".gate_proj.",
    ".up_proj.",
    ".down_proj.",
    ".gate_up_proj.",
    "embedding.weight",
    "final_proj.weight",
]


def rank_print(*args: Any, is_print_rank: bool = True, **kwargs: Any) -> None:
    """Thread-safe print across distributed ranks using a dedicated lock file.

    Uses a temporary lock file for cross-process synchronization rather than
    locking the source file itself, which is unreliable across nodes.
    """
    if not is_print_rank:
        return

    lock_path = os.environ.get(
        "SCALETORCH_PRINT_LOCK",
        os.path.join(tempfile.gettempdir(), "scaletorch_print.lock"),
    )

    try:
        with open(lock_path, "w") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                builtins.print(*args, **kwargs)
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)
    except OSError as e:
        # Fallback to regular print if file locking fails
        builtins.print(
            f"Warning: File locking failed ({e}), falling back to regular print",
            file=kwargs.get("file"),
        )
        builtins.print(*args, **kwargs)
    except Exception as e:
        # Catch any other unexpected exceptions
        builtins.print(
            f"Warning: Unexpected error in print function: {e}",
            file=kwargs.get("file"),
        )
        builtins.print(*args, **kwargs)


def set_all_seed(seed: int, deterministic: bool = False) -> None:
    """Set random seed for Python, NumPy, and PyTorch (CPU and CUDA)."""
    if not isinstance(seed, int):
        raise TypeError(f"Seed must be an integer, got {type(seed).__name__}")

    if seed < 0:
        raise ValueError(f"Seed must be non-negative, got {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    elif hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.manual_seed_all(seed)


def to_readable_format(num: int | float, precision: int = 2) -> str:
    """Format large numbers with K/M/B/T suffixes (e.g. 1500000 -> '1.50M')."""
    if not isinstance(num, int | float):
        raise TypeError(f"num must be numeric (int or float), got {type(num).__name__}")

    if precision < 0:
        raise ValueError(f"precision must be non-negative, got {precision}")

    # Handle edge cases
    if num == 0:
        return f"{0:.{precision}f}"

    sign = "-" if num < 0 else ""
    num = abs(num)

    if num >= TRILLION:
        return f"{sign}{num / TRILLION:.{precision}f}T"
    elif num >= BILLION:
        return f"{sign}{num / BILLION:.{precision}f}B"
    elif num >= MILLION:
        return f"{sign}{num / MILLION:.{precision}f}M"
    elif num >= THOUSAND:
        return f"{sign}{num / THOUSAND:.{precision}f}K"
    else:
        return f"{sign}{num:.{precision}f}"


def get_mfu(
    tokens_per_second: float,
    num_params: int,
    model_config: Any,
    theoretical_flops: float | None = None,
    sequence_length: int | None = None,
) -> float:
    """Calculate Model FLOPs Utilization (MFU) percentage (0-100).

    Formula: FLOPs/token = 6*N + 12*L*num_heads*head_dim*seq_len
    where 6*N covers all linear layers (fwd+bwd ≈ 3x, 2 ops per multiply-add),
    and the attention term covers QK^T and AV products.

    Args:
        tokens_per_second: Measured throughput in tokens/sec
        num_params: Total model parameter count
        model_config: HuggingFace model config
        theoretical_flops: Peak device FLOPS (auto-detected if None)
        sequence_length: Actual training seq_len (uses config.max_position_embeddings if None)
    """
    if tokens_per_second <= 0 or num_params <= 0:
        return 0.0

    if theoretical_flops is None:
        theoretical_flops = _get_device_peak_flops()
    if theoretical_flops <= 0:
        return 0.0

    num_layers = getattr(model_config, "num_hidden_layers", 1)
    num_heads = getattr(model_config, "num_attention_heads", 1)
    head_dim = getattr(
        model_config, "head_dim", getattr(model_config, "hidden_size", 1) // num_heads
    )
    seq_len = sequence_length or getattr(model_config, "max_position_embeddings", 2048)

    flops_per_token = 6 * num_params + 12 * num_layers * num_heads * head_dim * seq_len
    mfu = tokens_per_second * flops_per_token / theoretical_flops * 100

    return max(0.0, mfu)


def get_num_params(model: torch.nn.Module) -> int:
    """Count total parameters accounting for tensor and pipeline parallelism."""
    if not pgm:
        return sum(p.numel() for p in model.parameters())

    try:
        tp_world_size = pgm.tp_world_size
    except AttributeError:
        return sum(p.numel() for p in model.parameters())

    # Count parameters in current pipeline parallel rank
    local_num_params = 0

    for name, param in model.named_parameters():
        if any(tp_keyword in name.lower() for tp_keyword in TP_KEYWORDS):
            local_num_params += param.numel() * tp_world_size
        else:
            local_num_params += param.numel()

    if not torch.distributed.is_initialized():
        return local_num_params

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    param_counts = torch.tensor(local_num_params, device=device)

    st_dist.all_reduce(param_counts, op="sum", group=pgm.pp_group)

    return param_counts.item()


def assert_no_meta_tensors(model: torch.nn.Module) -> None:
    """Raise RuntimeError if any model parameters or buffers are still on the meta device."""
    meta_tensors = []

    for name, param in model.named_parameters():
        if param.device == torch.device("meta"):
            meta_tensors.append(f"Parameter '{name}' with shape {param.shape}")

    for name, buffer in model.named_buffers():
        if buffer.device == torch.device("meta"):
            meta_tensors.append(f"Buffer '{name}' with shape {buffer.shape}")

    if meta_tensors:
        error_msg = f"Found {len(meta_tensors)} meta tensors:\n" + "\n".join(
            meta_tensors
        )
        raise RuntimeError(error_msg)


def average_loss_across_dp_cp_ranks(
    loss: float | None, device: str | torch.device
) -> float:
    """Average loss across DP and CP ranks (only on the last PP stage)."""
    if loss is not None and not isinstance(loss, int | float):
        raise TypeError(f"loss must be numeric or None, got {type(loss)}")

    if not pgm:
        return loss if loss is not None else 0.0

    # Convert loss to tensor, using 0.0 if None
    reduced_loss = torch.tensor(
        [loss if loss is not None else 0.0], dtype=torch.float32, device=device
    )

    # Only average if this is the last pipeline parallel stage
    if pgm.pp_is_last_stage:
        st_dist.all_reduce(reduced_loss, op="sum", group=pgm.cp_dp_group)
        reduced_loss /= pgm.cp_dp_world_size

    return reduced_loss.item()
