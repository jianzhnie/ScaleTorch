import logging
import os
from datetime import timedelta
from typing import List, Optional

import torch
import torch.distributed as dist
from transformers.utils import (is_torch_cuda_available,
                                is_torch_mps_available, is_torch_npu_available,
                                is_torch_xpu_available)

# Configure global logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


def validate_distributed_setup() -> bool:
    """Validate if the distributed environment is properly configured.

    This is a placeholder function and should be implemented based on your specific requirements.
    Returns:
        bool: True if distributed environment is correctly configured
    """
    required_env_vars = [
        'MASTER_ADDR',
        'MASTER_PORT',
        'WORLD_SIZE',
        'LOCAL_RANK',
        'RANK',
    ]
    for var in required_env_vars:
        if var not in os.environ:
            logger.error(f'Missing required environment variable: {var}')
            return False
    return True


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
        return torch.npu.device_count()
    elif is_torch_cuda_available():
        return torch.cuda.device_count()
    else:
        return 0


def setup_distributed_environment(
    backend: str = 'nccl',
    init_method: str = 'env',
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    local_rank: Optional[int] = None,
    local_world_size: Optional[int] = None,
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
        local_rank (Optional[int]): Local rank of the process. If None, uses environment variable.
        local_world_size (Optional[int]): Local world size of the process. If None, uses environment variable.
        master_addr (Optional[str]): Address of the master node. If None, uses environment variable.
        master_port (Optional[str]): Port of the master node. If None, uses environment variable.
        gpu_ids (Optional[List[int]]): List of GPU IDs to be used. If None, uses all available GPUs.
        **init_process_group_kwargs: Additional arguments to pass to `init_process_group`.

    Raises:
        RuntimeError: If distributed environment is not properly configured.
        ValueError: If the provided `init_method` is not supported.
    """
    if not validate_distributed_setup():
        raise RuntimeError(
            'Distributed environment not properly configured. '
            "Ensure you're using torchrun or torch.distributed.launch with correct environment variables."
        )

    try:
        # Use provided values or fall back to environment variables
        current_rank = rank or int(os.environ.get('RANK', 0))
        current_world_size = world_size or int(os.environ.get('WORLD_SIZE', 1))
        current_local_rank = local_rank or int(os.environ.get('LOCAL_RANK', 0))
        current_local_world_size = local_world_size or int(
            os.environ.get('LOCAL_WORLD_SIZE', 1))
        current_master_addr = master_addr or os.environ.get(
            'MASTER_ADDR', 'localhost')
        current_master_port = master_port or os.environ.get(
            'MASTER_PORT', '12355')
        current_gpu_ids = gpu_ids or list(range(torch.cuda.device_count()))

        # Initialize distributed process group
        if init_method == 'env':
            os.environ['MASTER_ADDR'] = current_master_addr
            os.environ['MASTER_PORT'] = current_master_port
            url = 'env://'
        elif init_method == 'tcp':
            url = f'tcp://{current_master_addr}:{current_master_port}'
        else:
            raise ValueError(
                f'The provided init_method ({init_method}) is not supported. Must '
                f"be either 'env' or 'tcp'.")

        if backend == 'nccl':
            os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
                str(gid) for gid in current_gpu_ids)

        init_process_group_kwargs.update(
            dict(
                backend=backend,
                init_method=url,
                rank=current_rank,
                world_size=current_world_size,
            ))
        init_process_group_kwargs.setdefault('timeout',
                                             timedelta(seconds=1800))

        dist.init_process_group(**init_process_group_kwargs)

        # Set environment variables for distributed training
        os.environ['RANK'] = str(current_rank)
        os.environ['LOCAL_RANK'] = str(current_local_rank)
        os.environ['WORLD_SIZE'] = str(current_world_size)
        os.environ['LOCAL_WORLD_SIZE'] = str(current_local_world_size)

        # Set the current device
        torch.cuda.set_device(current_local_rank)

        logger.info(
            f'DDP Setup Complete: Rank {current_rank}, World Size {current_world_size}, '
            f'Local Rank {current_local_rank}, Local World Size {current_local_world_size}'
        )

    except Exception as e:
        logger.info(f'Failed to setup distributed environment: {e}')
        raise RuntimeError(f'Distributed setup failed: {e}')


def cleanup_ddp_env() -> None:
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
