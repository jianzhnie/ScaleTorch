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


def get_visible_devices_keyword() -> str:
    """Return 'CUDA_VISIBLE_DEVICES' or 'ASCEND_RT_VISIBLE_DEVICES' depending on backend."""
    return 'CUDA_VISIBLE_DEVICES' if is_torch_cuda_available(
    ) else 'ASCEND_RT_VISIBLE_DEVICES'


def get_process_info() -> tuple[int, int, int]:
    """Return (rank, world_size, local_rank) from environment variables."""
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    return rank, world_size, local_rank


def get_device() -> torch.device:
    """Return the best available device (npu > cuda > musa > mlu > xpu > cpu)."""
    if is_torch_npu_available():
        device = torch.device('npu:0')
        torch.npu.set_device(device)
    elif is_torch_cuda_available():
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
    elif is_torch_musa_available():
        device = torch.device('musa:0')
    elif is_torch_mlu_available():
        device = torch.device('mlu:0')
    elif is_torch_xpu_available():
        device = torch.device('xpu:0')
        torch.xpu.set_device(device)
    else:
        device = torch.device('cpu')
    return device


def get_current_device(use_cpu: bool = False) -> torch.device:
    """Return device for this process based on LOCAL_RANK (e.g. cuda:3, npu:1)."""
    _, _, local_rank = get_process_info()

    if use_cpu:
        device = 'cpu'
    elif is_torch_cuda_available():
        device = f'cuda:{local_rank}'
    elif is_torch_npu_available():
        device = f'npu:{local_rank}'
    elif is_torch_xpu_available():
        device = f'xpu:{local_rank}'
    elif is_torch_mps_available():
        device = f'mps:{local_rank}'
    elif is_torch_mlu_available():
        device = f'mlu:{local_rank}'
    elif is_torch_musa_available():
        device = f'musa:{local_rank}'
    else:
        device = 'cpu'
    return torch.device(device)


def get_device_count() -> int:
    """Return the number of available GPU/NPU/XPU devices."""
    if is_torch_npu_available():
        num_devices = torch.npu.device_count()
    elif is_torch_xpu_available():
        num_devices = torch.xpu.device_count()
    elif is_torch_cuda_available():
        num_devices = torch.cuda.device_count()
    else:
        num_devices = 0
    return num_devices
