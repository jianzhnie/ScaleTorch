"""Distributed training setup: initialization, cleanup, validation, and dtype."""

from __future__ import annotations

import datetime
import os

import torch
import torch.distributed as dist

from scaletorch.parallel.process_group import setup_process_group_manager
from scaletorch.trainer.config import ScaleTorchArguments
from scaletorch.utils.device import (
    get_current_device,
    get_dist_backend,
    is_accelerator_available,
    is_bf16_supported,
    set_device,
)
from scaletorch.utils.logger_utils import get_logger

logger = get_logger(__name__)


def get_dtype(config: ScaleTorchArguments) -> torch.dtype:
    """Determine the appropriate dtype based on config and hardware support."""
    use_cpu = getattr(config, "use_cpu", False)
    dtype = torch.float32

    if not use_cpu and is_accelerator_available():
        if is_bf16_supported():
            dtype = torch.bfloat16
            logger.info("Using bfloat16 dtype for training")
        else:
            logger.info("Using float32 dtype (bfloat16 not supported)")

    flash_atten_enabled = os.getenv("FLASH_ATTEN") == "1"
    if flash_atten_enabled:
        logger.info("Flash attention enabled via PyTorch SDPA")

    return dtype


def validate_config(config: ScaleTorchArguments, world_size: int) -> None:
    """Validate parallelism configuration against world size.

    Raises:
        ValueError: If parallelism configuration is inconsistent.
    """
    if (
        config.sequence_length
        and config.context_parallel_size
        and config.sequence_length % config.context_parallel_size != 0
    ):
        raise ValueError(
            f"sequence_length ({config.sequence_length}) must be divisible by "
            f"context_parallel_size ({config.context_parallel_size}) "
            f"for Context Parallelism to work correctly."
        )

    expected_world_size = (
        config.tensor_parallel_size
        * config.pipeline_parallel_size
        * config.data_parallel_size
        * config.context_parallel_size
        * config.expert_parallel_size
    )
    if world_size != expected_world_size:
        raise ValueError(
            f"world_size ({world_size}) != TP ({config.tensor_parallel_size}) * "
            f"PP ({config.pipeline_parallel_size}) * "
            f"DP ({config.data_parallel_size}) * "
            f"CP ({config.context_parallel_size}) * "
            f"EP ({config.expert_parallel_size}) = {expected_world_size}. "
            f"Please ensure your distributed setup matches the configured parallelism dimensions."
        )

    logger.info("Configuration validation passed")


def initialize_distributed_training(
    config: ScaleTorchArguments,
) -> tuple[int, int, int, str, torch.device]:
    """Initialize distributed training environment.

    Returns:
        Tuple of (local_rank, global_rank, world_size, backend, device).
    """
    try:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        global_rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
    except ValueError as e:
        raise ValueError(f"Environment variables must be integers: {e}")

    if local_rank < 0 or global_rank < 0 or world_size <= 0:
        raise ValueError(
            f"Invalid rank values: local_rank={local_rank}, "
            f"global_rank={global_rank}, world_size={world_size}"
        )
    if global_rank >= world_size:
        raise ValueError(
            f"global_rank ({global_rank}) must be less than world_size ({world_size})"
        )

    backend = "gloo" if getattr(config, "use_cpu", False) else get_dist_backend()

    if world_size > 1:
        try:
            dist.init_process_group(
                rank=global_rank,
                world_size=world_size,
                backend=backend,
                init_method="env://",
                timeout=datetime.timedelta(minutes=3),
            )
        except RuntimeError as e:
            raise RuntimeError(f"Failed to initialize process group: {e}")

        try:
            device = get_current_device(use_cpu=getattr(config, "use_cpu", False))
            set_device(device)
        except RuntimeError as e:
            raise RuntimeError(f"Failed to set device: {e}")

        try:
            setup_process_group_manager(
                tp_size=config.tensor_parallel_size,
                cp_size=config.context_parallel_size,
                pp_size=config.pipeline_parallel_size,
                dp_size=config.data_parallel_size,
                ep_size=config.expert_parallel_size,
            )
        except Exception as e:
            dist.destroy_process_group()
            raise RuntimeError(f"Failed to setup process group manager: {e}")
    else:
        logger.info("Running in single process mode.")
        device = get_current_device(use_cpu=getattr(config, "use_cpu", False))

    return local_rank, global_rank, world_size, backend, device


def cleanup_distributed_training(world_size: int) -> None:
    """Clean up distributed training resources."""
    try:
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()
            logger.info("Distributed process group destroyed successfully")
    except Exception as e:
        logger.warning(f"Failed to destroy process group: {e}")
