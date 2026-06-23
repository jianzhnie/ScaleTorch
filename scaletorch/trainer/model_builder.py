"""Model and optimizer construction with parallelism wrappers."""

from __future__ import annotations

import inspect
import os
import time

import torch
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer as OptimizerBase
from transformers import AutoConfig, PretrainedConfig

from scaletorch.models.llama import Llama
from scaletorch.models.model_qwen3 import Qwen3
from scaletorch.models.model_qwen3_moe import Qwen3MoE
from scaletorch.parallel.context_parallel.context_parallel import apply_context_parallel
from scaletorch.parallel.data_parallel.data_parallel import DataParallelBucket
from scaletorch.parallel.pipeline_parallel.pipeline_parallel import PipelineParallel
from scaletorch.parallel.process_group import process_group_manager as pgm
from scaletorch.parallel.tensor_parallel.tensor_parallel import apply_tensor_parallel
from scaletorch.trainer.config import ScaleTorchArguments
from scaletorch.utils.checkpoint import (
    init_model_with_dematerialized_weights,
    init_model_with_materialized_weights,
)
from scaletorch.utils.logger_utils import get_logger
from scaletorch.utils.misc import rank_print

logger = get_logger(__name__)


def create_model(
    config: ScaleTorchArguments, dtype: torch.dtype, device: torch.device
) -> tuple[torch.nn.Module, PretrainedConfig]:
    """Create and configure the model with parallelism.

    Returns:
        Tuple of (model, model_config).

    Raises:
        RuntimeError: If model creation fails.
    """
    is_print_rank = False
    if pgm:
        is_print_rank = (
            pgm.tp_rank == 0
            and pgm.dp_rank == 0
            and pgm.cp_rank == 0
            and pgm.pp_is_last_stage
        )

    rank_print(
        f"rank {pgm.global_rank if pgm else 0}: Initializing model meta device",
        is_print_rank=is_print_rank,
    )

    start_time = time.time()

    try:
        model_config = AutoConfig.from_pretrained(
            config.model_name_or_path, trust_remote_code=True
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load model config: {e}")

    with init_model_with_dematerialized_weights():
        model_type = getattr(model_config, "model_type", "llama")
        if model_type in ("qwen3",):
            model = Qwen3(config=model_config)
        elif model_type in ("qwen3_moe",):
            model = Qwen3MoE(config=model_config)
        else:
            model = Llama(config=model_config)

        if pgm and pgm.tp_world_size > 1:
            model = apply_tensor_parallel(model)

        if pgm and pgm.pp_world_size > 1:
            model = PipelineParallel(model, model_config)

    model_safetensors_dir = os.path.join(config.model_name_or_path, "")
    model = init_model_with_materialized_weights(
        model, model_config, save_dir=model_safetensors_dir
    )

    if pgm and pgm.cp_world_size > 1:
        model = apply_context_parallel(model)

    model.to(dtype).to(device)

    if pgm and pgm.dp_world_size > 1:
        model = DataParallelBucket(model)

    rank_print(
        f"init model parallel time: {time.time() - start_time:.2f}s",
        is_print_rank=is_print_rank,
    )

    return model, model_config


def create_optimizer(
    model: torch.nn.Module, config: ScaleTorchArguments, device: torch.device
) -> OptimizerBase:
    """Create and configure the optimizer."""
    extra_args = {}
    if getattr(config, "use_fused_adam", False):
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device.type in ("cuda", "npu")
        extra_args = {"fused": True} if use_fused else {}

        if use_fused:
            logger.info(f"Using fused AdamW optimizer on {device.type}")

    try:
        optimizer = AdamW(model.parameters(), lr=config.learning_rate, **extra_args)
    except Exception as e:
        raise RuntimeError(f"Failed to create optimizer: {e}")

    return optimizer


def get_tensor_shapes(
    config: ScaleTorchArguments, model_config: PretrainedConfig
) -> tuple[int, ...]:
    """Calculate hidden-state tensor shape for pipeline parallelism.

    Returns:
        Tuple of (micro_batch_size, sequence_length, hidden_size).
    """
    hidden_size = getattr(
        model_config, "hidden_size", getattr(model_config, "d_model", 768)
    )
    return (config.micro_batch_size, config.sequence_length, hidden_size)
