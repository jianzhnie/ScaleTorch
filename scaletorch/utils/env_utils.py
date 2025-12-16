import datetime
import platform
import socket

import psutil
import torch
import torch.distributed as dist
from transformers.utils import (is_torch_bf16_gpu_available,
                                is_torch_cuda_available,
                                is_torch_npu_available)

from scaletorch.utils.device import get_dist_info
from scaletorch.utils.logger_utils import get_logger

logger = get_logger(__name__)


def rank_print(rank: int, msg: str) -> None:
    """Print message with rank prefix for distributed debugging.

    Args:
        rank: Process rank
        msg: Message to print
    """
    print(f'[Rank {rank}] {msg}', flush=True)


def init_dist_pytorch(backend: str = 'nccl', **kwargs) -> None:
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
    if dist.is_initialized():
        return

    rank, world_size, local_rank = get_dist_info()

    timeout = kwargs.get('timeout', None)
    if timeout is not None:
        # If a timeout (in seconds) is specified, it must be converted
        # to a timedelta object before forwarding the call to
        # the respective backend, because they expect a timedelta object.
        try:
            kwargs['timeout'] = datetime.timedelta(seconds=timeout)
        except TypeError as exception:
            raise TypeError(
                f'Timeout for distributed training must be provided as '
                f"timeout in seconds, but we've received the type "
                f'{type(timeout)}. Please specify the timeout like this: '
                f"dist_cfg=dict(backend='nccl', timeout=1800)") from exception

    if world_size == 1:
        return

    elif is_torch_cuda_available():
        backend = 'nccl'
        torch.cuda.set_device(local_rank)
    elif is_torch_npu_available():
        backend = 'hccl'
        torch.npu.set_device(local_rank)
    else:
        backend = 'gloo'

    try:
        dist.init_process_group(backend=backend,
                                rank=rank,
                                world_size=world_size,
                                **kwargs)
        rank_print(
            rank,
            f'Initialized distributed process group with backend: {backend}')
    except Exception as e:
        rank_print(rank,
                   f'Failed to initialize distributed process group: {e}')
        raise


def cleanup_dis() -> None:
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


def get_system_info():
    logger.info('System Diagnostic Information:')
    # Basic system information
    info = {
        'Operating System': platform.platform(),
        'Python Version': platform.python_version(),
        'CPU Count': psutil.cpu_count(),
        'CPU Physical Count': psutil.cpu_count(logical=False),
        'CPU Frequency': f'{psutil.cpu_freq().current:.2f}MHz',
        'Memory Total': f'{psutil.virtual_memory().total / (1024**3):.2f}GB',
        'Memory Available':
        f'{psutil.virtual_memory().available / (1024**3):.2f}GB',
        'Memory Used': f'{psutil.virtual_memory().used / (1024**3):.2f}GB',
        'Disk Usage':
        f"{psutil.disk_usage('/').used / (1024**3):.2f}GB / {psutil.disk_usage('/').total / (1024**3):.2f}GB",
        'Network Interfaces': {
            interface:
            [addr.address for addr in addrs if addr.family == socket.AF_INET]
            for interface, addrs in psutil.net_if_addrs().items()
        },
        'Hostname': socket.gethostname(),
    }

    # Check CUDA GPU
    if is_torch_cuda_available():
        info['PyTorch Version'] += ' (GPU)'
        info['GPU Type'] = torch.cuda.get_device_name()
        info['GPU Memory'] = (
            f'{torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f}GB'
        )
        # Check BF16 support
        if is_torch_bf16_gpu_available():
            info['BF16 GPU Available'] = True

    # Check NPU
    if is_torch_npu_available():
        info['PyTorch Version'] += ' (NPU)'
        info['NPU Type'] = torch.npu.get_device_name()
        try:
            info['CANN Version'] = torch.version.cann
        except AttributeError:
            info['CANN Version'] = 'Unknown'

    # Log all information
    for key, value in info.items():
        logger.info(f'{key}: {value}')

    return info


# Example usage
if __name__ == '__main__':
    get_system_info()
    init_dist_pytorch(backend='nccl')
