"""
Training entry point for ScaleTorch distributed training.

Usage examples:
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node 4 --master_addr localhost --master_port 25500 train.py --model_name_or_path gpt2 --batch_size 32 --tensor_parallel_size 2 --data_parallel_size 2
"""

import datetime
import gc
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.amp import GradScaler
from transformers import HfArgumentParser

from scaletorch.data.dataloader import MicroBatchDataLoader
from scaletorch.parallel.pipeline_parallel.pipeline_parallel import (
    train_step_pipeline_1f1b,
    train_step_pipeline_afab,
)
from scaletorch.parallel.process_group import process_group_manager as pgm
from scaletorch.trainer.config import ScaleTorchArguments
from scaletorch.trainer.dist_setup import (
    cleanup_distributed_training,
    get_dtype,
    initialize_distributed_training,
    validate_config,
)
from scaletorch.trainer.lr_scheduler import create_lr_scheduler
from scaletorch.trainer.metrics import log_training_metrics
from scaletorch.trainer.model_builder import create_model, create_optimizer, get_tensor_shapes
from scaletorch.trainer.train_step import clip_gradients, train_step
from scaletorch.utils.checkpoint import CheckpointManager
from scaletorch.utils.device import empty_cache, is_accelerator_available
from scaletorch.utils.logger_utils import get_logger
from scaletorch.utils.misc import (
    average_loss_across_dp_cp_ranks,
    get_num_params,
    rank_print,
    set_all_seed,
    to_readable_format,
)
from scaletorch.utils.monitor import PerformanceMonitor

try:
    import wandb
except ImportError:
    wandb = None

logger = get_logger(__name__)

def main() -> None:
    """Main training function — orchestrates the full training pipeline."""
    world_size = 1
    is_wandb_rank = True

    try:
        parser = HfArgumentParser(ScaleTorchArguments)
        (config,) = parser.parse_args_into_dataclasses()

        logger.info("Starting ScaleTorch training script")
        logger.info(f"Configuration: {config}")

        dtype = get_dtype(config)
        logger.info(f"Using dtype: {dtype}")

        local_rank, global_rank, world_size, backend, device = (
            initialize_distributed_training(config)
        )
        logger.info(
            f"Distributed training initialized: local_rank={local_rank}, "
            f"global_rank={global_rank}, world_size={world_size}, backend={backend}"
        )

        validate_config(config, world_size)

        if pgm:
            is_wandb_rank = (
                pgm.tp_rank == 0
                and pgm.dp_rank == 0
                and pgm.cp_rank == 0
                and pgm.pp_is_last_stage
            )

        seed = getattr(config, "seed", 42)
        if not isinstance(seed, int) or seed < 0:
            raise ValueError(f"Seed must be a non-negative integer, got {seed}")
        set_all_seed(seed)
        logger.info(f"Random seed set to: {seed}")

        # ---- Data loader ----
        logger.info("Initializing data loader...")
        start_time = time.time()
        data_loader = MicroBatchDataLoader(
            micro_batch_size=config.micro_batch_size,
            sequence_length=config.sequence_length,
            dataset_name=config.dataset_name,
            tokenizer_name=config.model_name_or_path,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            device=device,
            num_workers=getattr(config, "num_workers", 0),
            num_proc=getattr(config, "num_proc", 1),
            num_samples=getattr(config, "num_samples", None),
            subset_name=getattr(config, "subset_name", None),
            split=getattr(config, "split", "train"),
        )

        if world_size > 1 and dist.is_initialized():
            dist.barrier()

        rank_print(
            f"init dataloader time: {time.time() - start_time:.2f}s",
            is_print_rank=is_wandb_rank,
        )

        tokens_per_step = data_loader.global_batch_size * (config.sequence_length or 1)
        if is_wandb_rank:
            rank_print("Tokens per step:", to_readable_format(tokens_per_step))

        # ---- Wandb ----
        if is_wandb_rank and getattr(config, "use_wandb", False) and wandb is not None:
            try:
                wandb.init(
                    project=getattr(config, "project_name", "scaletorch"),
                    name=f"{getattr(config, 'experiment_name', 'experiment')}_{to_readable_format(tokens_per_step)}",
                    config={
                        "tensor_parallel_size": pgm.tp_world_size if pgm else 1,
                        "context_parallel_size": pgm.cp_world_size if pgm else 1,
                        "pipeline_parallel_size": pgm.pp_world_size if pgm else 1,
                        "data_parallel_size": pgm.dp_world_size if pgm else 1,
                        "model": config.model_name_or_path,
                        "dataset": config.dataset_name,
                        "max_tokens": getattr(config, "max_tokens", None),
                        "learning_rate": config.learning_rate,
                        "seed": config.seed,
                        "micro_batch_size": data_loader.micro_batch_size,
                        "global_batch_size": data_loader.global_batch_size,
                        "gradient_accumulation": data_loader.gradient_accumulation_steps,
                    },
                )
                logger.info("Weights & Biases initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize wandb: {e}")

        if world_size > 1 and dist.is_initialized():
            dist.barrier()

        # ---- Model / optimizer / scheduler ----
        logger.info("Creating model...")
        model, model_config = create_model(config, dtype, device)
        model.train()

        try:
            num_params = get_num_params(model)
            rank_print(
                f"Number of parameters: {to_readable_format(num_params)}",
                is_print_rank=is_wandb_rank,
            )
        except Exception as e:
            logger.warning(f"Failed to calculate model parameters: {e}")
            num_params = 0

        logger.info("Creating optimizer...")
        optimizer = create_optimizer(model, config, device)

        lr_scheduler = None
        try:
            total_train_steps = getattr(config, "total_train_steps", None)
            if (
                total_train_steps is None
                and getattr(config, "max_tokens", None) is not None
                and tokens_per_step > 0
            ):
                total_train_steps = config.max_tokens // tokens_per_step
            lr_scheduler = create_lr_scheduler(optimizer, config, total_train_steps)
            if lr_scheduler is not None:
                logger.info(
                    f"Learning rate scheduler created: {type(lr_scheduler).__name__}"
                )
        except Exception as e:
            logger.warning(f"Failed to create learning rate scheduler: {e}")

        scaler = None
        if dtype == torch.float16 and not getattr(config, "use_cpu", False):
            scaler = GradScaler(device=device.type, enabled=True)
            logger.info(f"GradScaler initialized for fp16 on {device.type}")

        checkpoint_manager = CheckpointManager()
        monitor = PerformanceMonitor(config=config, log_dir=None)

        # ---- Resume from checkpoint ----
        trained_tokens, step = 0, 0
        resume_path = getattr(config, "resume_path", None)
        if resume_path:
            try:
                step, trained_tokens = checkpoint_manager.load_checkpoint(
                    model, optimizer, resume_path
                )
                rank_print(
                    f"Loaded checkpoint at step {step}, trained_tokens={trained_tokens}",
                    is_print_rank=is_wandb_rank,
                )
                if lr_scheduler is not None:
                    scheduler_path = os.path.join(resume_path, "scheduler.pt")
                    if os.path.exists(scheduler_path):
                        try:
                            lr_scheduler.load_state_dict(torch.load(scheduler_path))
                            rank_print(
                                f"Loaded scheduler state from {scheduler_path}",
                                is_print_rank=is_wandb_rank,
                            )
                        except Exception as e:
                            logger.warning(f"Failed to load scheduler state: {e}")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")

        if world_size > 1 and dist.is_initialized():
            dist.barrier()

        tensor_shapes = None
        if pgm and pgm.pp_world_size > 1:
            tensor_shapes = get_tensor_shapes(config, model_config)

        # ---- Training loop ----
        logger.info("Starting training loop...")
        monitor.start()

        max_tokens = getattr(config, "max_tokens", None)
        total_train_steps = getattr(config, "total_train_steps", None) or float("inf")

        try:
            _run_training_loop(
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                scaler=scaler,
                data_loader=data_loader,
                monitor=monitor,
                checkpoint_manager=checkpoint_manager,
                config=config,
                model_config=model_config,
                device=device,
                dtype=dtype,
                tensor_shapes=tensor_shapes,
                tokens_per_step=tokens_per_step,
                num_params=num_params,
                world_size=world_size,
                max_tokens=max_tokens,
                total_train_steps=total_train_steps,
                is_wandb_rank=is_wandb_rank,
                trained_tokens=trained_tokens,
                step=step,
            )
        except KeyboardInterrupt:
            logger.info("Training interrupted by user.")
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise
        finally:
            _cleanup(config, monitor, world_size, is_wandb_rank)

    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        cleanup_distributed_training(world_size)
        raise


def _run_training_loop(
    *,
    model,
    optimizer,
    lr_scheduler,
    scaler,
    data_loader,
    monitor,
    checkpoint_manager,
    config,
    model_config,
    device,
    dtype,
    tensor_shapes,
    tokens_per_step,
    num_params,
    world_size,
    max_tokens,
    total_train_steps,
    is_wandb_rank,
    trained_tokens,
    step,
) -> None:
    """Inner training loop, extracted from main() for readability."""
    while max_tokens is None or trained_tokens < max_tokens:
        monitor.start_iteration(tokens_processed=tokens_per_step)
        step_start_time = time.time()

        optimizer.zero_grad(set_to_none=True)

        try:
            if pgm and pgm.pp_world_size > 1:
                if config.pipeline_parallel_engine == "afab":
                    loss = train_step_pipeline_afab(
                        model, data_loader, tensor_shapes, device, dtype
                    )
                elif config.pipeline_parallel_engine == "1f1b":
                    loss = train_step_pipeline_1f1b(
                        model, data_loader, tensor_shapes, device, dtype
                    )
                else:
                    raise ValueError(
                        f"Invalid pipeline parallel engine: {config.pipeline_parallel_engine}"
                    )
            else:
                gradient_checkpointing = getattr(
                    config, "gradient_checkpointing", False
                )
                loss = train_step(
                    model=model,
                    data_loader=data_loader,
                    device=device,
                    dtype=dtype,
                    scaler=scaler,
                    gradient_accumulation_steps=config.gradient_accumulation_steps,
                    gradient_checkpointing=gradient_checkpointing,
                )
        except Exception as e:
            raise RuntimeError(f"Training step failed at step {step}: {e}")

        metrics = monitor.end_iteration()

        try:
            if pgm:
                loss = average_loss_across_dp_cp_ranks(loss, device)
        except Exception as e:
            raise RuntimeError(f"Failed to average loss across ranks: {e}") from e

        grad_norm = None
        max_grad_norm = getattr(config, "max_grad_norm", None)
        if max_grad_norm is not None and max_grad_norm > 0:
            try:
                grad_norm = clip_gradients(model, max_grad_norm)
            except Exception as e:
                raise RuntimeError(f"Gradient clipping failed: {e}") from e

        try:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
        except Exception as e:
            raise RuntimeError(f"Optimizer step failed at step {step}: {e}")

        if lr_scheduler is not None:
            try:
                lr_scheduler.step()
            except Exception as e:
                logger.warning(f"Failed to update learning rate scheduler: {e}")

        trained_tokens += tokens_per_step
        step += 1

        if hasattr(model, "reset"):
            try:
                model.reset()
            except Exception as e:
                logger.warning(f"Model reset failed: {e}")

        step_duration = time.time() - step_start_time

        if step % 100 == 0 and is_accelerator_available():
            empty_cache()
            if step % 1000 == 0:
                gc.collect()

        try:
            log_training_metrics(
                step=step,
                loss=loss,
                tokens_per_step=tokens_per_step,
                step_duration=step_duration,
                trained_tokens=trained_tokens,
                num_params=num_params,
                model_config=model_config,
                world_size=world_size,
                config=config,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                grad_norm=grad_norm,
            )
            if (
                is_wandb_rank
                and getattr(config, "use_wandb", False)
                and wandb is not None
            ):
                wandb.log(metrics, step=step)
        except Exception as e:
            logger.warning(f"Failed to log metrics: {e}")

        try:
            save_frequency = getattr(config, "save_frequency", 0)
            if save_frequency > 0 and step % save_frequency == 0:
                checkpoint_dir = Path(config.work_dir) / f"{step}"
                checkpoint_manager.save_checkpoint(
                    model, optimizer, step, trained_tokens, str(checkpoint_dir)
                )
                if lr_scheduler is not None and is_wandb_rank:
                    scheduler_path = checkpoint_dir / "scheduler.pt"
                    torch.save(lr_scheduler.state_dict(), scheduler_path)
                    logger.info(f"Saved scheduler state to {scheduler_path}")
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

        if step >= total_train_steps:
            logger.info(f"Reached maximum training steps: {total_train_steps}")
            break


def _cleanup(config, monitor, world_size, is_wandb_rank) -> None:
    """Post-training cleanup: wandb, performance logs, distributed teardown."""
    logger.info("Cleaning up training resources...")

    try:
        if (
            is_wandb_rank
            and getattr(config, "use_wandb", False)
            and wandb is not None
        ):
            wandb.finish()
            logger.info("Weights & Biases logging finished")
    except Exception as e:
        logger.warning(f"Failed to finish wandb: {e}")

    try:
        import datetime as dt

        global_rank_val = pgm.global_rank if pgm else 0
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        monitor.save_stats(
            f"performance_logs_{global_rank_val}_{timestamp}.json"
        )
        logger.info("Performance logs saved successfully")
    except Exception as e:
        logger.error(f"Error saving performance logs: {e}")

    cleanup_distributed_training(world_size)
    logger.info("Training completed successfully")


if __name__ == "__main__":
    main()
