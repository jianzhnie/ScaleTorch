import logging
import platform
import socket

import psutil
import torch
from transformers.utils import (is_torch_bf16_gpu_available,
                                is_torch_cuda_available,
                                is_torch_npu_available)

# Configure global logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


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

    # PyTorch and CUDA information
    dl_info = {
        'CUDA Available':
        torch.cuda.is_available(),
        'CUDA Version':
        torch.version.cuda if torch.cuda.is_available() else 'N/A',
        'CUDA Device Count':
        torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'cuDNN Version':
        torch.backends.cudnn.version() if torch.cuda.is_available() else 'N/A',
        'PyTorch Version':
        torch.__version__,
    }
    info.update(dl_info)

    # Check BF16 support
    if is_torch_bf16_gpu_available():
        info['PyTorch Version'] += ' (BF16)'
        info['BF16 GPU Available'] = True

    # Check CUDA GPU
    if is_torch_cuda_available():
        info['PyTorch Version'] += ' (GPU)'
        info['GPU Type'] = torch.cuda.get_device_name()
        info['GPU Memory'] = (
            f'{torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f}GB'
        )

    # Check NPU
    if is_torch_npu_available():
        info['PyTorch Version'] += ' (NPU)'
        info['NPU Type'] = torch.npu.get_device_name()
        try:
            info['CANN Version'] = torch.version.cann
        except AttributeError:
            info['CANN Version'] = 'Unknown'

    # Check optional dependencies
    try:
        import deepspeed

        info['DeepSpeed Version'] = getattr(deepspeed, '__version__',
                                            'Unknown')
    except ImportError:
        pass

    try:
        import bitsandbytes

        info['Bitsandbytes Version'] = getattr(bitsandbytes, '__version__',
                                               'Unknown')
    except ImportError:
        pass

    try:
        import vllm

        info['vLLM Version'] = getattr(vllm, '__version__', 'Unknown')
    except ImportError:
        pass

    # Log all information
    for key, value in info.items():
        logger.info(f'{key}: {value}')

    return info


if __name__ == '__main__':
    get_system_info()
