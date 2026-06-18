"""Device detection utilities for multi-backend support (CUDA, NPU, XPU, MPS, MLU, MUSA)."""

from __future__ import annotations

import os

import torch
from transformers.utils import (is_torch_cuda_available,
                                is_torch_mlu_available, is_torch_mps_available,
                                is_torch_musa_available,
                                is_torch_npu_available, is_torch_xpu_available)

from scaletorch.utils.logger_utils import get_logger

logger = get_logger(__name__)

_device_type_cache = None


def get_device_type() -> str:
    """Get the accelerator device type available on this system.

    Returns:
        One of: 'npu', 'cuda', 'xpu', 'mlu', 'musa', 'mps', 'cpu'
    """
    global _device_type_cache
    if _device_type_cache is not None:
        return _device_type_cache

    if is_torch_npu_available():
        _device_type_cache = 'npu'
    elif is_torch_cuda_available():
        _device_type_cache = 'cuda'
    elif is_torch_xpu_available():
        _device_type_cache = 'xpu'
    elif is_torch_mlu_available():
        _device_type_cache = 'mlu'
    elif is_torch_musa_available():
        _device_type_cache = 'musa'
    elif is_torch_mps_available():
        _device_type_cache = 'mps'
    else:
        _device_type_cache = 'cpu'
    return _device_type_cache


def is_accelerator_available() -> bool:
    """Check if any accelerator (GPU/NPU) is available."""
    return get_device_type() != 'cpu'


def get_dist_backend() -> str:
    """Get the appropriate distributed backend for the current device.

    Returns:
        'hccl' for NPU, 'nccl' for CUDA, 'gloo' for others
    """
    dtype = get_device_type()
    if dtype == 'npu':
        return 'hccl'
    elif dtype == 'cuda':
        return 'nccl'
    else:
        return 'gloo'


def get_visible_devices_keyword() -> str:
    """Get the environment variable name for visible devices."""
    dtype = get_device_type()
    if dtype == 'npu':
        return 'ASCEND_RT_VISIBLE_DEVICES'
    elif dtype == 'cuda':
        return 'CUDA_VISIBLE_DEVICES'
    return 'CUDA_VISIBLE_DEVICES'


def get_dist_info() -> tuple[int, int, int]:
    """Return (rank, world_size, local_rank) from environment variables."""
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    return rank, world_size, local_rank


def get_current_device(use_cpu: bool = False) -> torch.device:
    """Get the current process's device based on LOCAL_RANK.

    Args:
        use_cpu: Force CPU device

    Returns:
        torch.device for the current process
    """
    if use_cpu:
        return torch.device('cpu')

    _, _, local_rank = get_dist_info()
    dtype = get_device_type()

    if dtype == 'cpu':
        return torch.device('cpu')
    return torch.device(f'{dtype}:{local_rank}')


def set_device(device: torch.device) -> None:
    """Set the current device for the accelerator."""
    dtype = device.type
    if dtype == 'npu':
        torch.npu.set_device(device)
    elif dtype == 'cuda':
        torch.cuda.set_device(device)
    elif dtype == 'xpu':
        torch.xpu.set_device(device)


def get_device_count() -> int:
    """Get the number of available accelerator devices."""
    dtype = get_device_type()
    if dtype == 'npu':
        return torch.npu.device_count()
    elif dtype == 'cuda':
        return torch.cuda.device_count()
    elif dtype == 'xpu':
        return torch.xpu.device_count()
    return 0


def synchronize() -> None:
    """Synchronize the current accelerator device."""
    dtype = get_device_type()
    if dtype == 'npu':
        torch.npu.synchronize()
    elif dtype == 'cuda':
        torch.cuda.synchronize()
    elif dtype == 'xpu':
        torch.xpu.synchronize()


def empty_cache() -> None:
    """Empty the accelerator memory cache."""
    dtype = get_device_type()
    if dtype == 'npu':
        torch.npu.empty_cache()
    elif dtype == 'cuda':
        torch.cuda.empty_cache()


def memory_reserved() -> int:
    """Get the total memory reserved by the accelerator (bytes)."""
    dtype = get_device_type()
    if dtype == 'npu':
        return torch.npu.memory_reserved()
    elif dtype == 'cuda':
        return torch.cuda.memory_reserved()
    return 0


def memory_allocated() -> int:
    """Get the current memory allocated on the accelerator (bytes)."""
    dtype = get_device_type()
    if dtype == 'npu':
        return torch.npu.memory_allocated()
    elif dtype == 'cuda':
        return torch.cuda.memory_allocated()
    return 0


def max_memory_allocated() -> int:
    """Get the peak memory allocated on the accelerator (bytes)."""
    dtype = get_device_type()
    if dtype == 'npu':
        return torch.npu.max_memory_allocated()
    elif dtype == 'cuda':
        return torch.cuda.max_memory_allocated()
    return 0


def reset_peak_memory_stats() -> None:
    """Reset peak memory tracking."""
    dtype = get_device_type()
    if dtype == 'npu':
        torch.npu.reset_peak_memory_stats()
    elif dtype == 'cuda':
        torch.cuda.reset_peak_memory_stats()


def is_bf16_supported() -> bool:
    """Check if bfloat16 is supported on the current device."""
    dtype = get_device_type()
    if dtype == 'npu':
        return True
    elif dtype == 'cuda':
        return torch.cuda.is_bf16_supported()
    return False


def manual_seed_all(seed: int) -> None:
    """Set random seed for all accelerator devices."""
    dtype = get_device_type()
    if dtype == 'npu':
        torch.npu.manual_seed_all(seed)
    elif dtype == 'cuda':
        torch.cuda.manual_seed_all(seed)


def get_theoretical_flops() -> float:
    """Get theoretical peak FLOPS for the current accelerator.

    Returns FLOPS in operations per second for mixed-precision (bf16/fp16).
    """
    dtype = get_device_type()
    if dtype == 'npu':
        try:
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            name = torch.npu.get_device_name(local_rank)
            if '910B4' in name or '910b4' in name:
                return 352.0e12
            elif '910B3' in name or '910b3' in name:
                return 320.0e12
            elif '910B' in name or '910b' in name:
                return 320.0e12
        except Exception:
            pass
        return 256.0e12
    elif dtype == 'cuda':
        name = torch.cuda.get_device_name(0).upper()
        if 'H100' in name or 'H200' in name:
            return 1979e12  # H100 SXM bf16
        elif 'A100' in name:
            return 312e12   # A100 SXM bf16
        elif 'A10' in name:
            return 125e12   # A10 bf16
        elif 'L40' in name:
            return 181e12   # L40S bf16
        elif 'V100' in name:
            return 125e12   # V100 FP16
        elif '4090' in name:
            return 330e12   # RTX 4090 bf16
        elif '3090' in name:
            return 142e12   # RTX 3090 bf16
        return 312e12  # default A100-level
    return 100.0e12
