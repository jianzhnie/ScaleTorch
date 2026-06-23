import platform
import socket
import warnings
from typing import Any

import psutil
import torch
from transformers.utils import (
    is_torch_bf16_gpu_available,
    is_torch_cuda_available,
    is_torch_mlu_available,
    is_torch_musa_available,
    is_torch_npu_available,
    is_torch_xpu_available,
)

from scaletorch.utils.logger_utils import get_logger

logger = get_logger(__name__)


def init_dist_pytorch(backend: str = "nccl", **kwargs) -> None:
    """Set up the Distributed Data Parallel (DDP) environment.

    .. deprecated:: 0.2.0
        Use ``scaletorch.dist.utils.init_dist(launcher='pytorch')`` instead.
        This wrapper will be removed in v0.3.0.

    Args:
        backend (str): Backend to use for distributed training. Default is "nccl".
        **kwargs: Additional arguments passed to init_process_group.
    """
    warnings.warn(
        "init_dist_pytorch is deprecated. "
        "Use scaletorch.dist.utils.init_dist(launcher='pytorch') instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from scaletorch.dist.utils import init_dist

    init_dist(launcher="pytorch", backend=backend, **kwargs)


def cleanup_dist() -> None:
    """Clean up distributed training resources.

    .. deprecated:: 0.2.0
        Use ``scaletorch.dist.utils.cleanup_dist`` directly.
        This wrapper will be removed in v0.3.0.
    """
    warnings.warn(
        "cleanup_dist is deprecated. Use scaletorch.dist.utils.cleanup_dist instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from scaletorch.dist.utils import cleanup_dist as _cleanup

    _cleanup()


def get_system_info() -> dict[str, Any]:
    """Collect comprehensive system diagnostic information.

    Returns:
        Dictionary containing system information.
    """
    logger.info("System Diagnostic Information:")

    # Basic system information
    info = {
        "Operating System": platform.platform(),
        "Python Version": platform.python_version(),
        "CPU Count": psutil.cpu_count(),
        "CPU Physical Count": psutil.cpu_count(logical=False),
        "CPU Frequency": f"{psutil.cpu_freq().current:.2f}MHz"
        if psutil.cpu_freq()
        else "Unknown",
        "Memory Total": f"{psutil.virtual_memory().total / (1024**3):.2f}GB",
        "Memory Available": f"{psutil.virtual_memory().available / (1024**3):.2f}GB",
        "Memory Used": f"{psutil.virtual_memory().used / (1024**3):.2f}GB",
        "Disk Usage": f"{psutil.disk_usage('/').used / (1024**3):.2f}GB / {psutil.disk_usage('/').total / (1024**3):.2f}GB",
        "Network Interfaces": {
            interface: [addr.address for addr in addrs if addr.family == socket.AF_INET]
            for interface, addrs in psutil.net_if_addrs().items()
        },
        "Hostname": socket.gethostname(),
    }

    # Check CUDA GPU
    if is_torch_cuda_available():
        info["Device Type"] = "CUDA"
        info["GPU Type"] = torch.cuda.get_device_name(0)
        info["GPU Memory"] = (
            f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f}GB"
        )
        # Check BF16 support
        if is_torch_bf16_gpu_available():
            info["BF16 Support"] = True

    # Check NPU
    elif is_torch_npu_available():
        info["Device Type"] = "NPU"
        info["NPU Type"] = torch.npu.get_device_name(0)
        try:
            info["CANN Version"] = torch.version.cann
        except AttributeError:
            info["CANN Version"] = "Unknown"

    # Check MLU
    elif is_torch_mlu_available():
        info["Device Type"] = "MLU"
        info["MLU Type"] = torch.mlu.get_device_name(0)

    # Check MUSA
    elif is_torch_musa_available():
        info["Device Type"] = "MUSA"
        info["MUSA Type"] = torch.musa.get_device_name(0)

    # Check XPU
    elif is_torch_xpu_available():
        info["Device Type"] = "XPU"
        info["XPU Type"] = torch.xpu.get_device_name(0)

    else:
        info["Device Type"] = "CPU"

    # Log all information
    for key, value in info.items():
        logger.info("%s: %s", key, value)

    return info


# Example usage
if __name__ == "__main__":
    get_system_info()
    init_dist_pytorch()
