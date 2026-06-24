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
    """Create and configure the optimizer based on config settings.

    Supports optimizer_type: adamw, adam, sgd, lamb.
    Respects config.weight_decay, config.betas, and config.learning_rate.
    """
    optimizer_type = getattr(config, "optimizer_type", "adamw")
    weight_decay = getattr(config, "weight_decay", 0.0)
    betas = getattr(config, "betas", (0.9, 0.999))
    lr = getattr(config, "learning_rate", 1e-3)

    params = model.parameters()

    try:
        if optimizer_type == "adamw":
            extra_args = {}
            if getattr(config, "use_fused_adam", False):
                fused_available = (
                    "fused" in inspect.signature(torch.optim.AdamW).parameters
                )
                if fused_available and device.type in ("cuda", "npu"):
                    extra_args["fused"] = True
                    logger.info("Using fused AdamW on %s", device.type)
            optimizer = AdamW(
                params,
                lr=lr,
                betas=betas,
                weight_decay=weight_decay,
                **extra_args,
            )
        elif optimizer_type == "adam":
            from torch.optim import Adam

            optimizer = Adam(
                params, lr=lr, betas=betas, weight_decay=weight_decay
            )
        elif optimizer_type == "sgd":
            from torch.optim import SGD

            optimizer = SGD(
                params, lr=lr, weight_decay=weight_decay, momentum=0.9
            )
        elif optimizer_type == "lamb":
            try:
                from torch.optim import Lamb

                optimizer = Lamb(
                    params, lr=lr, betas=betas, weight_decay=weight_decay
                )
            except ImportError:
                logger.warning(
                    "LAMB optimizer not available in this PyTorch version, "
                    "falling back to AdamW."
                )
                optimizer = AdamW(
                    params, lr=lr, betas=betas, weight_decay=weight_decay
                )
        else:
            raise ValueError(
                f"Unsupported optimizer_type: {optimizer_type!r}. "
                f"Supported: adamw, adam, sgd, lamb."
            )
    except Exception as e:
        raise RuntimeError(
            f"Failed to create optimizer ({optimizer_type}): {e}"
        ) from e

    return optimizer


def get_tensor_shapes(
    config: ScaleTorchArguments, model_config: PretrainedConfig
) -> tuple[int, ...]:
    """Calculate hidden-state tensor shape for pipeline parallelism.

    When context parallelism (CP) is enabled, the sequence is split across CP
    ranks *before* entering the pipeline stages, so the per-rank sequence
    length seen by PP communication is ``sequence_length // cp_size``.

    Returns:
        Tuple of (micro_batch_size, local_sequence_length, hidden_size).
    """
    hidden_size = getattr(
        model_config, "hidden_size", getattr(model_config, "d_model", 768)
    )
    seq_len = config.sequence_length
    cp_size = getattr(config, "context_parallel_size", 1) or 1
    if cp_size > 1:
        seq_len = seq_len // cp_size
    return (config.micro_batch_size, seq_len, hidden_size)
