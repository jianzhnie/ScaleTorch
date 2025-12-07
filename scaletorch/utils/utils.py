"""
Utility functions for ScaleTorch distributed training framework.

This module provides essential utilities for:
- Multi-process printing coordination
- Random seed management
- Number formatting for large values
- Model performance calculations (MFU)
- Parameter counting in distributed settings
- Meta tensor validation
- Loss averaging across parallel ranks
"""

import builtins
import fcntl
import random
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.distributed as dist

from scaletorch.parallel.pg_manager import process_group_manager as pgm

# Constants for number formatting
TRILLION = 1e12
BILLION = 1e9
MILLION = 1e6
THOUSAND = 1e3

# Constants for MFU calculation
DEFAULT_THEORETICAL_FLOPS = 989.5 * 10**12  # FLOPS for A100 GPU

# Tensor parallelism keywords for parameter counting
TP_KEYWORDS = ['attention', 'mlp', 'embed', 'final_proj']


def print(*args: Any, is_print_rank: bool = True, **kwargs: Any) -> None:
    """
    Thread-safe print function for multi-process environments.

    Solves the multi-process interleaved print problem by using file locking
    to ensure atomic print operations across distributed processes.

    Args:
        *args: Arguments to print
        is_print_rank: Whether this rank should print (default: True)
        **kwargs: Additional keyword arguments for print function

    Returns:
        None
    """
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


def set_all_seed(seed: int) -> None:
    """
    Set random seed for all random number generators.

    Sets seeds for Python random, NumPy, and PyTorch (both CPU and CUDA).
    This ensures reproducible results across runs.

    Args:
        seed: Random seed value (must be non-negative)

    Returns:
        None

    Raises:
        TypeError: If seed is not an integer
        ValueError: If seed is negative
    """
    if not isinstance(seed, int):
        raise TypeError(f'Seed must be an integer, got {type(seed).__name__}')

    if seed < 0:
        raise ValueError(f'Seed must be non-negative, got {seed}')

    try:
        # Set seeds for all random number generators
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Set CUDA seeds if available
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # Ensure reproducibility on CUDA
            # Note: This may impact performance
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception as e:
        raise RuntimeError(f'Failed to set random seeds: {e}')


def to_readable_format(num: Union[int, float], precision: int = 2) -> str:
    """
    Convert large numbers to human-readable format with appropriate suffixes.

    Converts numbers to format with K (thousands), M (millions), B (billions),
    or T (trillions) suffixes.

    Args:
        num: Number to format
        precision: Number of decimal places (default: 2)

    Returns:
        Formatted string with appropriate suffix

    Raises:
        TypeError: If num is not numeric
        ValueError: If precision is negative

    Examples:
        >>> to_readable_format(1500)
        '1.50K'
        >>> to_readable_format(1500000)
        '1.50M'
        >>> to_readable_format(1500000000)
        '1.50B'
    """
    if not isinstance(num, (int, float)):
        raise TypeError(
            f'num must be numeric (int or float), got {type(num).__name__}')

    if precision < 0:
        raise ValueError(f'precision must be non-negative, got {precision}')

    # Handle edge cases
    if num == 0:
        return f'{0:.{precision}f}'

    # Handle negative numbers by working with absolute value
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
    """
    Calculate Model FLOPs Utilization (MFU) percentage.

    MFU measures how efficiently a model utilizes the available FLOPS capacity
    of the hardware. Based on the approach from nanoGPT and Stanford CS336.

    References:
    - https://github.com/karpathy/nanoGPT/blob/9755682b981a45507f6eb9b11eadef8cb83cebd5/model.py#L289
    - https://github.com/stanford-cs336/spring2024-lectures/blob/main/lecture_02.py#L950

    Args:
        tokens_per_second: Processing speed in tokens per second
        num_params: Total number of model parameters
        model_config: Model configuration object with required attributes:
            - num_hidden_layers: Number of transformer layers
            - hidden_size: Hidden dimension size
            - max_position_embeddings: Maximum sequence length
        theoretical_flops: Theoretical FLOPS of the hardware (default: A100 GPU)

    Returns:
        MFU percentage (0-100)

    Raises:
        AttributeError: If model_config is missing required attributes
        ValueError: If any input value is negative
    """
    # Input validation
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
    """
    Calculate total number of parameters accounting for tensor and pipeline parallelism.

    This function accounts for:
    - Tensor Parallelism (TP): Parameters in attention/mlp/embed/final_proj are sharded
    - Pipeline Parallelism (PP): Gathers parameter counts across pipeline stages
    - Data Parallelism (DP): Parameters are replicated, counted only once

    Args:
        model: PyTorch model instance

    Returns:
        Total number of parameters across all parallel configurations

    Raises:
        RuntimeError: If distributed operations fail
        AttributeError: If pgm is not available
    """
    try:
        tp_world_size = pgm.tp_world_size
    except AttributeError as e:
        raise AttributeError('pgm is not available') from e

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
        param_counts = torch.tensor(local_num_params, device='cuda')

        # Sum up parameters across all PP ranks
        dist.all_reduce(param_counts, op=dist.ReduceOp.SUM, group=pgm.pp_group)

        return param_counts.item()
    except (RuntimeError, dist.DistBackendError) as e:
        raise RuntimeError(
            f'Failed to gather parameters across PP ranks: {e}') from e


def assert_no_meta_tensors(model: torch.nn.Module) -> None:
    """
    Assert that the model contains no meta tensors.

    Meta tensors are placeholder tensors used for initialization without
    allocating actual memory. This function ensures all parameters and buffers
    are properly initialized.

    Args:
        model: PyTorch model to check

    Returns:
        None

    Raises:
        AssertionError: If any meta tensors are found
    """
    meta_tensors = []

    # Check parameters
    for name, param in model.named_parameters():
        if param.device == torch.device('meta'):
            meta_tensors.append(f"Parameter '{name}' with shape {param.shape}")

    # Check buffers
    for name, buffer in model.named_buffers():
        if buffer.device == torch.device('meta'):
            meta_tensors.append(f"Buffer '{name}' with shape {buffer.shape}")

    if meta_tensors:
        error_msg = f'Found {len(meta_tensors)} meta tensors:\n' + '\n'.join(
            meta_tensors)
        raise AssertionError(error_msg)


def average_loss_across_dp_cp_ranks(loss: Optional[float],
                                    device: Union[str, torch.device]) -> float:
    """
    Average loss across data parallel and context parallel ranks.

    Only performs averaging if this is the last pipeline parallel stage.

    Args:
        loss: Loss value to average (can be None)
        device: Target device for tensor operations

    Returns:
        Averaged loss value, or 0.0 if loss was None

    Raises:
        RuntimeError: If distributed operations fail
        ValueError: If device is invalid
    """
    if loss is not None and not isinstance(loss, (int, float)):
        raise TypeError(f'loss must be numeric or None, got {type(loss)}')

    try:
        # Convert loss to tensor, using 0.0 if None
        reduced_loss = torch.tensor([loss if loss is not None else 0.0],
                                    dtype=torch.float32,
                                    device=device)

        # Only average if this is the last pipeline parallel stage
        if pgm.pp_is_last_stage:
            dist.all_reduce(reduced_loss,
                            op=dist.ReduceOp.SUM,
                            group=pgm.cp_dp_group)
            reduced_loss /= pgm.cp_dp_world_size

        return reduced_loss.item()

    except (RuntimeError, dist.DistBackendError) as e:
        raise RuntimeError(f'Failed to average loss across ranks: {e}') from e
    except Exception as e:
        raise RuntimeError(
            f'Unexpected error during loss averaging: {e}') from e
