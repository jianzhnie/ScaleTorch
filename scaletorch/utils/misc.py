"""Utility functions for distributed training: printing, seeds, MFU, parameter counting, loss averaging."""

import builtins
import fcntl
import random
from typing import Any, Optional, Union

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

# Tensor parallelism keywords for parameter counting
TP_KEYWORDS = ['attention', 'mlp', 'embed', 'final_proj']


def rank_print(*args: Any, is_print_rank: bool = True, **kwargs: Any) -> None:
    """Thread-safe print using file locking to prevent interleaved output across ranks."""
    if not is_print_rank:
        return

    try:
        with open(__file__, 'r') as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                builtins.print(*args, **kwargs)
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)
    except (IOError, OSError) as e:
        # Fallback to regular print if file locking fails
        # This is not ideal but ensures output is not completely lost
        builtins.print(
            f'Warning: File locking failed ({e}), falling back to regular print',
            file=kwargs.get('file'))
        builtins.print(*args, **kwargs)
    except Exception as e:
        # Catch any other unexpected exceptions
        builtins.print(f'Warning: Unexpected error in print function: {e}',
                       file=kwargs.get('file'))
        builtins.print(*args, **kwargs)


def set_all_seed(seed: int, deterministic: bool = False) -> None:
    """Set random seed for Python, NumPy, and PyTorch (CPU and CUDA)."""
    if not isinstance(seed, int):
        raise TypeError(f'Seed must be an integer, got {type(seed).__name__}')

    if seed < 0:
        raise ValueError(f'Seed must be non-negative, got {seed}')

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def to_readable_format(num: Union[int, float], precision: int = 2) -> str:
    """Format large numbers with K/M/B/T suffixes (e.g. 1500000 -> '1.50M')."""
    if not isinstance(num, (int, float)):
        raise TypeError(
            f'num must be numeric (int or float), got {type(num).__name__}')

    if precision < 0:
        raise ValueError(f'precision must be non-negative, got {precision}')

    # Handle edge cases
    if num == 0:
        return f'{0:.{precision}f}'

    sign = '-' if num < 0 else ''
    num = abs(num)

    if num >= TRILLION:
        return f'{sign}{num / TRILLION:.{precision}f}T'
    elif num >= BILLION:
        return f'{sign}{num / BILLION:.{precision}f}B'
    elif num >= MILLION:
        return f'{sign}{num / MILLION:.{precision}f}M'
    elif num >= THOUSAND:
        return f'{sign}{num / THOUSAND:.{precision}f}K'
    else:
        return f'{sign}{num:.{precision}f}'


def get_mfu(tokens_per_second: float,
            num_params: int,
            model_config: Any,
            theoretical_flops: float = DEFAULT_THEORETICAL_FLOPS) -> float:
    """Calculate Model FLOPs Utilization (MFU) percentage (0-100).

    Based on nanoGPT and Stanford CS336 approach.
    model_config must have: num_hidden_layers, hidden_size, max_position_embeddings.
    """
    if tokens_per_second < 0:
        raise ValueError(
            f'tokens_per_second must be non-negative, got {tokens_per_second}')
    if num_params < 0:
        raise ValueError(f'num_params must be non-negative, got {num_params}')
    if theoretical_flops <= 0:
        raise ValueError(
            f'theoretical_flops must be positive, got {theoretical_flops}')

    # Validate model_config attributes
    required_attrs = [
        'num_hidden_layers', 'hidden_size', 'max_position_embeddings'
    ]
    missing_attrs = [
        attr for attr in required_attrs if not hasattr(model_config, attr)
    ]
    if missing_attrs:
        raise AttributeError(
            f'model_config missing required attributes: {missing_attrs}')

    num_layers = model_config.num_hidden_layers
    hidden_dim = model_config.hidden_size
    seq_len = model_config.max_position_embeddings

    # Calculate FLOPs per token using the standard transformer formula
    flops_per_token = 6 * num_params + 12 * num_layers * hidden_dim * seq_len

    # Calculate MFU percentage
    mfu = tokens_per_second * flops_per_token / theoretical_flops * 100

    return mfu


def get_num_params(model: torch.nn.Module) -> int:
    """Count total parameters accounting for tensor and pipeline parallelism."""
    if pgm is None:
        return sum(p.numel() for p in model.parameters())

    tp_world_size = pgm.tp_world_size

    # Count parameters in current pipeline parallel rank
    local_num_params = 0

    for name, param in model.named_parameters():
        # Parameters split across tensor parallel ranks
        # TODO: LayerNorm is also split across TP ranks for sequence parallelism
        if any(tp_keyword in name.lower() for tp_keyword in TP_KEYWORDS):
            local_num_params += param.numel() * tp_world_size
        else:
            # Parameters replicated across TP ranks (layer norm, biases)
            local_num_params += param.numel()

    # Gather parameter counts from all pipeline parallel ranks
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device('cpu')
    param_counts = torch.tensor(local_num_params, device=device)

    st_dist.all_reduce(param_counts, op='sum', group=pgm.pp_group)

    return param_counts.item()


def assert_no_meta_tensors(model: torch.nn.Module) -> None:
    """Raise RuntimeError if any model parameters or buffers are still on the meta device."""
    meta_tensors = []

    for name, param in model.named_parameters():
        if param.device == torch.device('meta'):
            meta_tensors.append(f"Parameter '{name}' with shape {param.shape}")

    for name, buffer in model.named_buffers():
        if buffer.device == torch.device('meta'):
            meta_tensors.append(f"Buffer '{name}' with shape {buffer.shape}")

    if meta_tensors:
        error_msg = f'Found {len(meta_tensors)} meta tensors:\n' + '\n'.join(
            meta_tensors)
        raise RuntimeError(error_msg)


def average_loss_across_dp_cp_ranks(loss: Optional[float],
                                    device: Union[str, torch.device]) -> float:
    """Average loss across DP and CP ranks (only on the last PP stage)."""
    if loss is not None and not isinstance(loss, (int, float)):
        raise TypeError(f'loss must be numeric or None, got {type(loss)}')

    if pgm is None:
        return loss if loss is not None else 0.0

    # Convert loss to tensor, using 0.0 if None
    reduced_loss = torch.tensor([loss if loss is not None else 0.0],
                                dtype=torch.float32,
                                device=device)

    # Only average if this is the last pipeline parallel stage
    if pgm.pp_is_last_stage:
        st_dist.all_reduce(reduced_loss, op='sum', group=pgm.cp_dp_group)
        reduced_loss /= pgm.cp_dp_world_size

    return reduced_loss.item()
