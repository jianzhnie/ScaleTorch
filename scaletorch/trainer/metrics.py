"""Training metrics logging (console and Weights & Biases)."""

from __future__ import annotations

from torch.optim.lr_scheduler import _LRScheduler as LRSchedulerBase
from torch.optim.optimizer import Optimizer as OptimizerBase
from transformers import PretrainedConfig

from scaletorch.parallel.process_group import process_group_manager as pgm
from scaletorch.trainer.config import ScaleTorchArguments
from scaletorch.utils.device import is_accelerator_available, memory_reserved
from scaletorch.utils.logger_utils import get_logger
from scaletorch.utils.misc import get_mfu, rank_print, to_readable_format

try:
    import wandb
except ImportError:
    wandb = None

logger = get_logger(__name__)


def log_training_metrics(
    step: int,
    loss: float,
    tokens_per_step: int,
    step_duration: float,
    trained_tokens: int,
    num_params: int,
    model_config: PretrainedConfig,
    world_size: int,
    config: ScaleTorchArguments,
    optimizer: OptimizerBase | None = None,
    lr_scheduler: LRSchedulerBase | None = None,
    grad_norm: float | None = None,
) -> None:
    """Log training metrics to console and optionally to Weights & Biases."""
    tokens_per_second = tokens_per_step / step_duration
    tokens_per_second_per_gpu = tokens_per_second / world_size
    mfu = get_mfu(
        tokens_per_second_per_gpu,
        num_params,
        model_config,
        sequence_length=getattr(config, "sequence_length", None),
    )

    current_lr = None
    if lr_scheduler is not None:
        current_lr = lr_scheduler.get_last_lr()[0]
    elif optimizer is not None:
        current_lr = optimizer.param_groups[0]["lr"]

    if pgm:
        is_wandb_rank = (
            pgm.tp_rank == 0
            and pgm.dp_rank == 0
            and pgm.cp_rank == 0
            and pgm.pp_is_last_stage
        )
    else:
        is_wandb_rank = True

    if is_wandb_rank:
        max_tokens = getattr(config, "max_tokens", None)
        max_tokens_str = ("/" + to_readable_format(max_tokens)) if max_tokens else ""
        global_rank = pgm.global_rank if pgm else 0

        log_parts = [
            f"[rank {global_rank}]",
            f"Step: {step:<5d}",
            f"Loss: {loss:6.4f}",
        ]

        if current_lr is not None:
            log_parts.append(f"LR: {current_lr:.2e}")

        if grad_norm is not None:
            log_parts.append(f"GradNorm: {grad_norm:.2f}")

        log_parts.extend(
            [
                f"Global batch size: {to_readable_format(tokens_per_step):>7s}",
                f"Tokens/s: {to_readable_format(tokens_per_second):>7s}",
                f"Tokens/s/GPU: {to_readable_format(tokens_per_second_per_gpu):>7s}",
                f"Tokens: {to_readable_format(trained_tokens):>7s}{max_tokens_str}",
                f"MFU: {mfu:5.2f}%",
            ]
        )

        if is_accelerator_available():
            log_parts.append(f"Memory: {memory_reserved() / 1e9:6.2f}GB")

        rank_print(" | ".join(log_parts), is_print_rank=is_wandb_rank)

        if getattr(config, "use_wandb", False) and wandb is not None:
            log_dict = {
                "loss": loss,
                "tokens_per_step": tokens_per_step,
                "tokens_per_second": tokens_per_second,
                "mfu": mfu,
                "tokens_per_second_per_gpu": tokens_per_second_per_gpu,
                "trained_tokens": trained_tokens,
            }

            if current_lr is not None:
                log_dict["learning_rate"] = current_lr

            if grad_norm is not None:
                log_dict["grad_norm"] = grad_norm

            if is_accelerator_available():
                log_dict["memory_usage"] = memory_reserved() / 1e9

            wandb.log(log_dict, step=step)
