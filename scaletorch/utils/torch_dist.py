import os
from datetime import timedelta
from typing import List, Optional

import torch
import torch.distributed as dist
from transformers.utils import (is_accelerate_available, is_ipex_available,
                                is_torch_cuda_available,
                                is_torch_mps_available, is_torch_npu_available,
                                is_torch_xpu_available)

from scaletorch.utils.logger_utils import get_logger

logger = get_logger(__name__)


def get_device() -> torch.device:
    """Retrieve PyTorch device. It checks that the requested device is
    available first. For now, it supports cpu and cuda, xpu, npu. By default,
    it tries to use the gpu.

    :param device: One for 'auto', 'cuda', 'cpu'
    :return: Supported Pytorch device
    """
    if is_torch_xpu_available():
        if not is_ipex_available() and not is_accelerate_available(
                '0.32.0.dev'):
            raise ImportError(
                'Using the XPU PyTorch backend requires `accelerate>=0.32.0.dev`'
            )
        device = torch.device('xpu:0')
        torch.xpu.set_device(device)
    elif is_torch_npu_available():
        device = torch.device('npu:0')
        torch.npu.set_device(device)
    elif torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    return device


def get_current_device() -> 'torch.device':
    r"""
    Gets the current available device.
    """
    if is_torch_xpu_available():
        device = 'xpu:{}'.format(os.environ.get('LOCAL_RANK', '0'))
    elif is_torch_npu_available():
        device = 'npu:{}'.format(os.environ.get('LOCAL_RANK', '0'))
    elif is_torch_mps_available():
        device = 'mps:{}'.format(os.environ.get('LOCAL_RANK', '0'))
    elif is_torch_cuda_available():
        device = 'cuda:{}'.format(os.environ.get('LOCAL_RANK', '0'))
    else:
        device = 'cpu'

    return torch.device(device)


def get_device_count() -> int:
    r"""
    Gets the number of available GPU or NPU devices.
    """
    if is_torch_npu_available():
        num_devices = torch.npu.device_count()
    if is_torch_xpu_available():
        num_devices = torch.xpu.device_count()
    elif is_torch_cuda_available():
        num_devices = torch.cuda.device_count()
    else:
        num_devices = 0
    return num_devices


def setup_distributed_environment(
    backend: str = 'nccl',
    init_method: str = 'env',
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    master_addr: Optional[str] = None,
    master_port: Optional[str] = None,
    gpu_ids: Optional[List[int]] = None,
    **init_process_group_kwargs,
) -> None:
    """Set up the Distributed Data Parallel (DDP) environment with
    comprehensive checks.

    Args:
        backend (str): Backend to use for distributed training. Default is "nccl".
        init_method (str): Method to initialize the process group. Must be either "env" or "tcp".
        rank (Optional[int]): Specific rank to set. If None, uses environment variable.
        world_size (Optional[int]): Total number of processes. If None, uses environment variable.
        master_addr (Optional[str]): Address of the master node. If None, uses environment variable.
        master_port (Optional[str]): Port of the master node. If None, uses environment variable.
        gpu_ids (Optional[List[int]]): List of GPU IDs to be used. If None, uses all available GPUs.
        **init_process_group_kwargs: Additional arguments to pass to `init_process_group`.

    Raises:
        ValueError: If the provided `init_method` is not supported or `gpu_ids` are invalid.
        RuntimeError: If distributed environment setup fails.
    """

    try:
        # Section 1: Fallback to environment variables if arguments are not provided
        current_rank = rank if rank is not None else int(
            os.environ.get('RANK', 0))
        current_world_size = (world_size if world_size is not None else int(
            os.environ.get('WORLD_SIZE', 1)))
        current_master_addr = (master_addr if master_addr is not None else
                               os.environ.get('MASTER_ADDR', 'localhost'))
        current_master_port = (master_port if master_port is not None else
                               os.environ.get('MASTER_PORT', '12355'))

        # Validate and set GPU IDs
        available_gpus = list(range(torch.cuda.device_count()))
        if gpu_ids is None:
            current_gpu_ids = available_gpus
        else:
            if not all(gpu_id in available_gpus for gpu_id in gpu_ids):
                raise ValueError(
                    f'Invalid GPU IDs: {gpu_ids}. Available GPUs: {available_gpus}'
                )
            current_gpu_ids = gpu_ids

        # Section 2: Set up initialization method
        if init_method == 'env':
            os.environ['MASTER_ADDR'] = current_master_addr
            os.environ['MASTER_PORT'] = current_master_port
            url = 'env://'
        elif init_method == 'tcp':
            url = f'tcp://{current_master_addr}:{current_master_port}'
        else:
            raise ValueError(
                f"Unsupported init_method: {init_method}. Must be either 'env' or 'tcp'."
            )

        # Section 3: Configure backend
        if backend == 'nccl':
            os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
                str(gpu_id) for gpu_id in current_gpu_ids)

        # Section 4: Initialize the process group
        init_process_group_kwargs.update({
            'backend': backend,
            'init_method': url,
            'rank': current_rank,
            'world_size': current_world_size,
        })
        init_process_group_kwargs.setdefault('timeout',
                                             timedelta(seconds=1800))

        dist.init_process_group(**init_process_group_kwargs)

        # Set environment variables for distributed training
        os.environ['RANK'] = str(current_rank)
        os.environ['WORLD_SIZE'] = str(current_world_size)

        # Section 5: Set environment variables for distributed training
        os.environ['RANK'] = str(current_rank)
        os.environ['WORLD_SIZE'] = str(current_world_size)

        logger.info(
            f'DDP Setup Complete: Rank {current_rank}, World Size {current_world_size}, '
            f'Master Address {current_master_addr}, Master Port {current_master_port}, '
            f'GPUs {current_gpu_ids}')
    except Exception as e:
        logger.error(f'Failed to setup distributed environment: {e}')
        raise RuntimeError(f'Distributed setup failed: {e}')


def cleanup_distribute_environment() -> None:
    """Safely clean up the distributed environment.

    This function will attempt to destroy the distributed process group and
    handle any errors gracefully.
    """
    try:
        # Check if the process group is initialized
        if dist.is_initialized():
            dist.destroy_process_group()
            logger.info('Distributed process group destroyed successfully.')
        else:
            logger.info(
                'Distributed process group was not initialized. No cleanup needed.'
            )
    except RuntimeError as e:
        logger.info(f'Error during distributed cleanup: {e}')
    except Exception as e:
        logger.info(f'Unexpected error during distributed cleanup: {e}')


# Example usage
if __name__ == '__main__':
    setup_distributed_environment(
        backend='nccl',
        init_method='env',
        rank=0,
        world_size=2,
        master_addr='127.0.0.1',
        master_port='29500',
        gpu_ids=[0, 1],
    )
