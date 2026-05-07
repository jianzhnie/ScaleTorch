"""
Training script for LLaMA model with distributed parallelism support.

This script implements distributed training for LLaMA models with support for:
- Tensor Parallelism
- Pipeline Parallelism (1F1B and AFAB)
- Context Parallelism
- Data Parallelism
- Gradient accumulation
- Checkpointing
- Weights & Biases logging

Usage examples:
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node 4 --master_addr localhost --master_port 25500 train.py --model_name_or_path gpt2 --batch_size 32 --tensor_parallel_size 2 --data_parallel_size 2
CUDA_DEVICE_MAX_CONNECTIONS=1 debugpy-run -p 5678 -m torch.distributed.run -- --nproc_per_node=4 --nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:29400 train.py --model_name_or_path llama2 --batch_size 64 --tensor_parallel_size 1 --data_parallel_size 4 --use_flash_attention True
"""

from __future__ import annotations

import contextlib
import datetime
import inspect
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import scaletorch.dist as st_dist
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.optimizer import Optimizer as OptimizerBase
from transformers import AutoConfig, HfArgumentParser, PretrainedConfig

from scaletorch.data.dataloader import MicroBatchDataLoader
from scaletorch.models.model_llama import Llama
from scaletorch.parallel.context_parallel.context_parallel import (
    apply_context_parallel)
from scaletorch.parallel.data_parallel.data_parallel import DataParallelBucket
from scaletorch.parallel.pg_manager import (
    process_group_manager as pgm, setup_process_group_manager)
from scaletorch.parallel.pipeline_parallel.pipeline_parallel import (
    PipelineParallel, train_step_pipeline_1f1b, train_step_pipeline_afab)
from scaletorch.parallel.tensor_parallel.tensor_parallel import (
    apply_tensor_parallel)
from scaletorch.trainer.config import ScaleTorchArguments
from scaletorch.trainer.lr_scheduler import create_lr_scheduler
from scaletorch.utils.checkpoint import (
    CheckpointManager, init_model_with_dematerialized_weights,
    init_model_with_materialized_weights)
from scaletorch.utils.logger_utils import get_logger
from scaletorch.utils.monitor import PerformanceMonitor
from scaletorch.utils.utils import (average_loss_across_dp_cp_ranks, get_mfu,
                                    get_num_params, print, set_all_seed,
                                    to_readable_format)

_FUSED_ADAM_AVAILABLE = "fused" in inspect.signature(AdamW).parameters

# Optional imports
try:
    import wandb
except ImportError:
    wandb = None

logger = get_logger(__name__)


def _is_log_rank() -> bool:
    """Check if current rank should log (TP=0, DP=0, CP=0, last PP stage)."""
    if pgm is None:
        return True
    return (pgm.tp_rank == 0 and pgm.dp_rank == 0
            and pgm.cp_rank == 0 and pgm.pp_is_last_stage)


def _wandb_enabled(config: ScaleTorchArguments) -> bool:
    """Check if wandb is available and enabled in config."""
    return config.use_wandb and wandb is not None


def train_step(model: torch.nn.Module,
               data_loader: MicroBatchDataLoader,
               device: torch.device,
               dtype: torch.dtype,
               scaler: Optional[GradScaler] = None,
               gradient_accumulation_steps: int = 1,
               gradient_checkpointing: bool = False,
               use_no_sync: bool = True) -> float:
    """Single training step with gradient accumulation and mixed precision."""
    if not model.training:
        model.train()

    accumulation_loss = torch.tensor(0.0, device=device)
    sync_context = (model.no_sync() if
                    (use_no_sync and hasattr(model, 'no_sync')
                     and gradient_accumulation_steps > 1) else
                    contextlib.nullcontext())

    with sync_context:
        for i in range(gradient_accumulation_steps):
            try:
                batch = next(data_loader)
            except StopIteration:
                raise RuntimeError(
                    f'Data loader exhausted after {i}/{gradient_accumulation_steps} steps. '
                    'Check dataset size and gradient accumulation config.'
                )

            input_ids = batch['input_ids'].to(device, non_blocking=True)
            target_ids = batch['target_ids'].to(device, non_blocking=True)

            forward_context = (autocast(device.type, dtype=dtype)
                               if scaler is not None
                               else torch.enable_grad())

            with forward_context:
                outputs = model(input_ids=input_ids,
                                gradient_checkpointing=gradient_checkpointing)
                target_ids_flat = target_ids.reshape(-1)
                outputs_reshaped = outputs.view(-1, outputs.size(-1))
                loss = F.cross_entropy(
                    outputs_reshaped, target_ids_flat,
                    reduction='mean') / gradient_accumulation_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accumulation_loss = accumulation_loss + loss.detach()
            del outputs, outputs_reshaped, target_ids_flat, input_ids, target_ids

    return accumulation_loss.item()


def get_dtype(config: ScaleTorchArguments) -> torch.dtype:
    """Return best dtype for training — bfloat16 if GPU supports it, else float32."""
    if config.use_cpu or not torch.cuda.is_available():
        return torch.float32

    if torch.cuda.is_bf16_supported():
        logger.info('Using bfloat16 dtype for training')
        return torch.bfloat16

    logger.info('Using float32 dtype for training (bfloat16 not supported)')
    return torch.float32


def initialize_distributed_training(
        config: ScaleTorchArguments
) -> Tuple[int, int, int, str, torch.device]:
    """Init distributed env. Returns (local_rank, global_rank, world_size, backend, device)."""
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    global_rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    if local_rank < 0 or global_rank < 0 or world_size <= 0:
        raise ValueError(f'Invalid rank values: local_rank={local_rank}, '
                         f'global_rank={global_rank}, world_size={world_size}')
    if global_rank >= world_size:
        raise ValueError(
            f'global_rank ({global_rank}) must be less than world_size ({world_size})'
        )

    backend = 'gloo' if config.use_cpu else 'nccl'

    if world_size > 1:
        st_dist.init_dist('pytorch',
                          backend=backend,
                          timeout=180)

        if backend == 'nccl':
            device = torch.device('cuda', local_rank)
        else:
            device = torch.device('cpu')

        try:
            setup_process_group_manager(tp_size=config.tensor_parallel_size,
                                        cp_size=config.context_parallel_size,
                                        pp_size=config.pipeline_parallel_size,
                                        dp_size=config.data_parallel_size)
        except Exception:
            st_dist.cleanup_dist()
            raise
    else:
        logger.info('Running in single process mode.')
        if config.use_cpu:
            device = torch.device('cpu')
        else:
            device = torch.device(
                'cuda:0' if torch.cuda.is_available() else 'cpu')

    return local_rank, global_rank, world_size, backend, device


def create_model(
        config: ScaleTorchArguments, dtype: torch.dtype,
        device: torch.device) -> Tuple[torch.nn.Module, PretrainedConfig]:
    """Create model with parallelism layers applied.

    Returns:
        Tuple of (model, model_config).
    """
    is_print_rank = _is_log_rank()

    print(
        f'rank {pgm.global_rank if pgm is not None else 0}: Initializing model meta device',
        is_print_rank=is_print_rank)

    start_time = time.time()
    model_config = AutoConfig.from_pretrained(config.model_name_or_path,
                                              trust_remote_code=True)

    # Normalize config for Llama constructor compatibility
    if not hasattr(model_config, 'num_key_value_heads'):
        model_config.num_key_value_heads = model_config.num_attention_heads
    if not hasattr(model_config, 'intermediate_size'):
        model_config.intermediate_size = getattr(model_config, 'ffn_dim',
                                                 4 * model_config.hidden_size)
    if not hasattr(model_config, 'rms_norm_eps'):
        model_config.rms_norm_eps = 1e-6
    if not hasattr(model_config, 'rope_theta'):
        model_config.rope_theta = 10000.0

    with init_model_with_dematerialized_weights():
        model = Llama(config=model_config)

        if pgm is not None and pgm.tp_world_size > 1:
            model = apply_tensor_parallel(model)

        if pgm is not None and pgm.pp_world_size > 1:
            model = PipelineParallel(model, model_config)

    model = init_model_with_materialized_weights(
        model, model_config, save_dir=config.work_dir)

    # TODO: Load existing checkpoint here to continue pre-training

    if pgm is not None and pgm.cp_world_size > 1:
        model = apply_context_parallel(model)

    model.to(dtype).to(device)

    if pgm is not None and pgm.dp_world_size > 1:
        model = DataParallelBucket(model)

    print(f'init model parallel time: {time.time() - start_time:.2f}s',
          is_print_rank=is_print_rank)

    return model, model_config


def create_optimizer(model: torch.nn.Module, config: ScaleTorchArguments,
                     device: torch.device) -> OptimizerBase:
    """Create AdamW optimizer, using fused variant when available on CUDA."""
    extra_args = {}
    if config.use_fused_adam:
        use_fused = _FUSED_ADAM_AVAILABLE and device.type == 'cuda'
        extra_args = {"fused": True} if use_fused else {}
        logger.info("Using %s AdamW optimizer",
                    "fused" if use_fused else "standard")

    return AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.betas,
        **extra_args,
    )


def clip_gradients(model: torch.nn.Module, max_norm: float) -> float:
    """Clip gradients. Returns norm before clipping."""
    if max_norm <= 0:
        return 0.0
    return torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm).item()


def log_training_metrics(step: int,
                         loss: float,
                         tokens_per_step: int,
                         step_duration: float,
                         trained_tokens: int,
                         num_params: int,
                         model_config: PretrainedConfig,
                         world_size: int,
                         config: ScaleTorchArguments,
                         optimizer: Optional[OptimizerBase] = None,
                         grad_norm: Optional[float] = None) -> None:
    """Log training metrics to console and optionally wandb."""
    tokens_per_second = tokens_per_step / step_duration
    tokens_per_second_per_gpu = tokens_per_second / world_size
    mfu = get_mfu(tokens_per_second_per_gpu, num_params, model_config)
    current_lr = optimizer.param_groups[0]['lr'] if optimizer else None
    is_wandb_rank = _is_log_rank()

    if not is_wandb_rank:
        return

    # Console logging
    max_tokens_str = ('/' +
                      to_readable_format(config.max_tokens)
                      ) if config.max_tokens else ''
    global_rank = pgm.global_rank if pgm is not None else 0

    log_parts = [
        f'[rank {global_rank}]',
        f'Step: {step:<5d}',
        f'Loss: {loss:6.4f}',
    ]
    if current_lr is not None:
        log_parts.append(f'LR: {current_lr:.2e}')
    if grad_norm is not None:
        log_parts.append(f'GradNorm: {grad_norm:.2f}')

    log_parts.extend([
        f'Tokens/s: {to_readable_format(tokens_per_second):>7s}',
        f'Tokens/s/GPU: {to_readable_format(tokens_per_second_per_gpu):>7s}',
        f'Tokens: {to_readable_format(trained_tokens):>7s}{max_tokens_str}',
        f'MFU: {mfu:5.2f}%',
    ])
    if torch.cuda.is_available():
        log_parts.append(
            f'Memory: {torch.cuda.memory_reserved() / 1e9:6.2f}GB')

    print(' | '.join(log_parts), is_print_rank=True)

    # Wandb logging
    if not _wandb_enabled(config):
        return

    log_dict = {
        'loss': loss,
        'tokens_per_step': tokens_per_step,
        'tokens_per_second': tokens_per_second,
        'mfu': mfu,
        'tokens_per_second_per_gpu': tokens_per_second_per_gpu,
        'trained_tokens': trained_tokens,
    }
    if current_lr is not None:
        log_dict['learning_rate'] = current_lr
    if grad_norm is not None:
        log_dict['grad_norm'] = grad_norm
    if torch.cuda.is_available():
        log_dict['memory_usage'] = torch.cuda.memory_reserved() / 1e9

    wandb.log(log_dict, step=step)


def get_tensor_shapes(config: ScaleTorchArguments,
                      model_config: PretrainedConfig) -> Tuple[int, ...]:
    """Compute tensor shapes for pipeline parallelism.

    Returns hidden state shape shared by forward/backward communication.
    """
    return (config.micro_batch_size, config.sequence_length,
            model_config.hidden_size)


def cleanup_distributed_training(world_size: int) -> None:
    """Destroy distributed process group."""
    try:
        if world_size > 1 and st_dist.is_distributed():
            st_dist.cleanup_dist()
    except Exception as e:
        logger.warning("Failed to destroy process group: %s", e)


def _init_wandb(config: ScaleTorchArguments,
                data_loader: MicroBatchDataLoader,
                tokens_per_step: int) -> None:
    """Initialize Weights & Biases logging on the log rank."""
    if not (_is_log_rank() and _wandb_enabled(config)):
        return
    try:
        wandb.init(
            project=config.project_name,
            name=f"{config.experiment_name}_{to_readable_format(tokens_per_step)}",
            config={
                'tensor_parallel_size':
                pgm.tp_world_size if pgm is not None else 1,
                'context_parallel_size':
                pgm.cp_world_size if pgm is not None else 1,
                'pipeline_parallel_size':
                pgm.pp_world_size if pgm is not None else 1,
                'data_parallel_size':
                pgm.dp_world_size if pgm is not None else 1,
                'model':
                config.model_name_or_path,
                'dataset':
                config.dataset_name,
                'max_tokens':
                config.max_tokens,
                'learning_rate':
                config.learning_rate,
                'seed':
                config.seed,
                'micro_batch_size':
                data_loader.micro_batch_size,
                'global_batch_size':
                data_loader.global_batch_size,
                'gradient_accumulation':
                data_loader.gradient_accumulation_steps,
            },
        )
    except Exception as e:
        logger.warning("Failed to initialize wandb: %s", e)


def _resume_checkpoint(model: torch.nn.Module,
                       optimizer: OptimizerBase,
                       lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
                       checkpoint_manager: CheckpointManager,
                       resume_path: str,
                       is_print_rank: bool) -> Tuple[int, int]:
    """Load checkpoint and scheduler state. Returns (step, trained_tokens)."""
    try:
        step, trained_tokens = checkpoint_manager.load_checkpoint(
            model, optimizer, resume_path)
        print(
            f'Loaded checkpoint at step {step}, trained_tokens={trained_tokens}',
            is_print_rank=is_print_rank)

        if lr_scheduler is not None:
            scheduler_path = os.path.join(resume_path, 'scheduler.pt')
            if os.path.exists(scheduler_path):
                try:
                    lr_scheduler.load_state_dict(
                        torch.load(scheduler_path))
                except Exception as e:
                    logger.warning("Failed to load scheduler state: %s", e)
    except Exception as e:
        logger.warning("Failed to load checkpoint: %s", e)
        step, trained_tokens = 0, 0

    return step, trained_tokens


def _run_training_step(
        model: torch.nn.Module,
        data_loader: MicroBatchDataLoader,
        tensor_shapes: Optional[Tuple[int, ...]],
        device: torch.device,
        dtype: torch.dtype,
        scaler: Optional[GradScaler],
        config: ScaleTorchArguments,
        step: int) -> float:
    """Dispatch forward/backward pass to appropriate parallelism path."""
    if pgm is not None and pgm.pp_world_size > 1:
        if config.pipeline_parallel_engine == 'afab':
            return train_step_pipeline_afab(model, data_loader,
                                            tensor_shapes, device, dtype)
        if config.pipeline_parallel_engine == '1f1b':
            return train_step_pipeline_1f1b(model, data_loader,
                                            tensor_shapes, device, dtype)
        raise ValueError(
            f'Invalid pipeline parallel engine: {config.pipeline_parallel_engine}'
        )

    return train_step(
        model=model,
        data_loader=data_loader,
        device=device,
        dtype=dtype,
        scaler=scaler,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing)


def _save_step_checkpoint(model: torch.nn.Module,
                          optimizer: OptimizerBase,
                          lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
                          checkpoint_manager: CheckpointManager,
                          config: ScaleTorchArguments,
                          step: int,
                          trained_tokens: int,
                          is_log_rank: bool) -> None:
    """Save checkpoint and scheduler state if save_frequency is met."""
    if config.save_frequency <= 0 or step % config.save_frequency != 0:
        return

    checkpoint_dir = Path(config.work_dir) / str(step)
    checkpoint_manager.save_checkpoint(model, optimizer, step,
                                       trained_tokens,
                                       str(checkpoint_dir))

    if lr_scheduler is not None and is_log_rank:
        scheduler_path = checkpoint_dir / 'scheduler.pt'
        torch.save(lr_scheduler.state_dict(), scheduler_path)
        logger.info("Saved scheduler state to %s", scheduler_path)


def main() -> None:
    """Parse config, init distributed, create model/optimizer, run training loop."""
    world_size = 1

    try:
        # Parse command-line arguments
        parser = HfArgumentParser(ScaleTorchArguments)
        config, = parser.parse_args_into_dataclasses()

        logger.info('Starting ScaleTorch training script')
        logger.info("Configuration: %s", config)

        # Get appropriate data type
        dtype = get_dtype(config)
        logger.info("Using dtype: %s", dtype)

        # Initialize distributed training
        local_rank, global_rank, world_size, backend, device = initialize_distributed_training(
            config)
        logger.info(
            f'Distributed training initialized: local_rank={local_rank}, global_rank={global_rank}, world_size={world_size}, backend={backend}'
        )

        # Validate configuration
        config.validate_world_size(world_size)

        # Determine if this rank should log to wandb
        is_wandb_rank = _is_log_rank()

        # Set random seed for reproducibility
        seed = config.seed
        if not isinstance(seed, int) or seed < 0:
            raise ValueError(
                f'Seed must be a non-negative integer, got {seed}')
        set_all_seed(seed)
        logger.info("Random seed set to: %s", seed)

        # Initialize data loader
        logger.info('Initializing data loader...')
        start_time = time.time()
        data_loader = MicroBatchDataLoader(
            micro_batch_size=config.micro_batch_size,
            sequence_length=config.sequence_length,
            dataset_name=config.dataset_name,
            tokenizer_name=config.model_name_or_path,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            device=device,
            num_workers=config.num_workers,
            num_proc=config.num_proc,
            num_samples=config.num_samples,
            subset_name=config.subset_name,
            split=config.split)

        if world_size > 1 and st_dist.is_distributed():
            st_dist.barrier()

        print(f'init dataloader time: {time.time() - start_time:.2f}s',
              is_print_rank=is_wandb_rank)

        # Calculate tokens per step
        tokens_per_step = data_loader.global_batch_size * (
            config.sequence_length or 1)
        if is_wandb_rank:
            print('Tokens per step:', to_readable_format(tokens_per_step))

        _init_wandb(config, data_loader, tokens_per_step)

        if world_size > 1 and st_dist.is_distributed():
            st_dist.barrier()

        # Create model
        logger.info('Creating model...')
        model, model_config = create_model(config, dtype, device)

        # Set model to training mode
        model.train()

        # Print model size
        try:
            num_params = get_num_params(model)
            print(f'Number of parameters: {to_readable_format(num_params)}',
                  is_print_rank=is_wandb_rank)
        except Exception as e:
            logger.warning("Failed to calculate model parameters: %s", e)
            num_params = 0

        # Create optimizer
        logger.info('Creating optimizer...')
        optimizer = create_optimizer(model, config, device)

        # Create learning rate scheduler
        lr_scheduler = None
        try:
            total_train_steps = config.total_train_steps
            if total_train_steps is None and config.max_tokens is not None and tokens_per_step > 0:
                # Estimate total steps from max_tokens
                total_train_steps = config.max_tokens // tokens_per_step
            lr_scheduler = create_lr_scheduler(optimizer, config,
                                               total_train_steps)
            if lr_scheduler is not None:
                logger.info(
                    f'Learning rate scheduler created: {type(lr_scheduler).__name__}'
                )
        except Exception as e:
            logger.warning("Failed to create learning rate scheduler: %s", e)

        # Initialize GradScaler for mixed precision training
        scaler = None
        if dtype in (torch.float16, torch.bfloat16) and not config.use_cpu:
            scaler = GradScaler(device.type,
                                enabled=dtype == torch.float16)
            logger.info("GradScaler initialized for %s", dtype)

        # Initialize checkpoint manager
        checkpoint_manager = CheckpointManager()

        # Initialize performance monitor
        monitor = PerformanceMonitor(config=config, log_dir=None)

        # Initialize training state
        trained_tokens, step = 0, 0
        if config.resume_path:
            step, trained_tokens = _resume_checkpoint(
                model, optimizer, lr_scheduler, checkpoint_manager,
                config.resume_path, is_wandb_rank)

        if world_size > 1 and st_dist.is_distributed():
            st_dist.barrier()

        # Get tensor shapes for pipeline parallelism
        tensor_shapes = None
        if pgm is not None and pgm.pp_world_size > 1:
            tensor_shapes = get_tensor_shapes(config, model_config)

        # Training loop with error handling
        logger.info('Starting training loop...')
        monitor.start()

        max_tokens = config.max_tokens
        total_train_steps = config.total_train_steps

        try:
            while max_tokens is None or trained_tokens < max_tokens:
                monitor.start_iteration(tokens_processed=tokens_per_step)
                step_start_time = time.time()

                optimizer.zero_grad(set_to_none=True)

                # Forward + backward
                try:
                    loss = _run_training_step(model, data_loader,
                                              tensor_shapes, device, dtype,
                                              scaler, config, step)
                except Exception as e:
                    raise RuntimeError(
                        f'Training step failed at step {step}: {e}') from e

                metrics = monitor.end_iteration()

                if pgm is not None:
                    loss = average_loss_across_dp_cp_ranks(loss, device)

                # Gradient clipping
                grad_norm = None
                if config.max_grad_norm is not None and config.max_grad_norm > 0:
                    grad_norm = clip_gradients(model, config.max_grad_norm)

                # Optimizer step
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                # LR scheduler step
                if lr_scheduler is not None:
                    try:
                        if isinstance(lr_scheduler, ReduceLROnPlateau):
                            lr_scheduler.step(loss)
                        else:
                            lr_scheduler.step()
                    except Exception as e:
                        logger.warning(
                            "Failed to update learning rate scheduler: %s", e)

                trained_tokens += tokens_per_step
                step += 1

                if hasattr(model, 'reset'):
                    try:
                        model.reset()
                    except Exception as e:
                        logger.warning("Model reset failed: %s", e)

                step_duration = time.time() - step_start_time

                # Logging
                try:
                    log_training_metrics(step=step,
                                         loss=loss,
                                         tokens_per_step=tokens_per_step,
                                         step_duration=step_duration,
                                         trained_tokens=trained_tokens,
                                         num_params=num_params,
                                         model_config=model_config,
                                         world_size=world_size,
                                         config=config,
                                         optimizer=optimizer,
                                         grad_norm=grad_norm)
                    if is_wandb_rank and _wandb_enabled(config):
                        wandb.log(metrics, step=step)
                except Exception as e:
                    logger.warning("Failed to log metrics: %s", e)

                # Checkpoint
                try:
                    _save_step_checkpoint(model, optimizer, lr_scheduler,
                                          checkpoint_manager, config, step,
                                          trained_tokens, is_wandb_rank)
                except Exception as e:
                    logger.warning("Failed to save checkpoint: %s", e)

                if total_train_steps is not None and step >= total_train_steps:
                    logger.info(
                        f'Reached maximum training steps: {total_train_steps}')
                    break

        except KeyboardInterrupt:
            logger.info('Training interrupted by user.')
        except Exception as e:
            logger.error("Error during training: %s", e)
            raise
        finally:
            # Cleanup resources
            logger.info('Cleaning up training resources...')

            # Finish Weights & Biases logging if enabled and available
            try:
                if is_wandb_rank and _wandb_enabled(config):
                    wandb.finish()
                    logger.info('Weights & Biases logging finished')
            except Exception as e:
                logger.warning("Failed to finish wandb: %s", e)

            # Save performance logs
            try:
                global_rank_val = pgm.global_rank if pgm is not None else 0
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                monitor.save_stats(
                    f'performance_logs_{global_rank_val}_{timestamp}.json')
                logger.info('Performance logs saved successfully')
            except Exception as e:
                logger.error("Error saving performance logs: %s", e)

            # Clean up distributed training
            cleanup_distributed_training(world_size)

            logger.info('Training completed successfully')

    except Exception as e:
        logger.error("Fatal error in main: %s", e)
        # Attempt cleanup
        cleanup_distributed_training(world_size)
        raise


if __name__ == '__main__':
    main()
