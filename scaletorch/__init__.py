"""ScaleTorch: A distributed training framework for large language models.

Supports 5D parallelism: Data (DP), Pipeline (PP), Context (CP),
Tensor (TP), and Expert (EP) parallelism.
"""

from scaletorch.parallel.process_group import (
    ProcessGroupManager,
    get_process_group_manager,
    process_group_manager,
    setup_process_group_manager,
)
from scaletorch.trainer.config import ScaleTorchArguments
from scaletorch.trainer.lr_scheduler import create_lr_scheduler
from scaletorch.utils.logger_utils import get_logger

__all__ = [
    "ProcessGroupManager",
    "ScaleTorchArguments",
    "create_lr_scheduler",
    "get_logger",
    "get_process_group_manager",
    "process_group_manager",
    "setup_process_group_manager",
]
