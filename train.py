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

import contextlib
import datetime
import gc
import inspect
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import _LRScheduler as LRSchedulerBase
from torch.optim.optimizer import Optimizer as OptimizerBase
from transformers import AutoConfig, HfArgumentParser, PretrainedConfig

# ScaleTorch imports
from scaletorch.data.dataloader import MicroBatchDataLoader
from scaletorch.model.model_llama import Llama
from scaletorch.parallel.context_parallel.context_parallel import \
    apply_context_parallel
from scaletorch.parallel.data_parallel.data_parallel import DataParallelBucket
from scaletorch.parallel.pg_manager import process_group_manager as pgm
from scaletorch.parallel.pg_manager import setup_process_group_manager
from scaletorch.parallel.pipeline_parallel.pipeline_parallel import (
    PipelineParallel, train_step_pipeline_1f1b, train_step_pipeline_afab)
from scaletorch.parallel.tensor_parallel.tensor_parallel import \
    apply_tensor_parallel
from scaletorch.trainer.config import ScaleTorchArguments
from scaletorch.trainer.lr_scheduler_config import create_lr_scheduler
from scaletorch.utils.checkpoint import (
    CheckpointManager, init_model_with_dematerialized_weights,
    init_model_with_materialized_weights)
from scaletorch.utils.logger_utils import get_logger
from scaletorch.utils.monitor import PerformanceMonitor
from scaletorch.utils.utils import (average_loss_across_dp_cp_ranks, get_mfu,
                                    get_num_params, print, set_all_seed,
                                    to_readable_format)

# Optional imports
try:
    import wandb
except ImportError:
    wandb = None

logger = get_logger(__name__)


def train_step(model: torch.nn.Module,
               data_loader: MicroBatchDataLoader,
               device: torch.device,
               dtype: torch.dtype,
               scaler: Optional[GradScaler] = None,
               gradient_accumulation_steps: int = 1,
               gradient_checkpointing: bool = False,
               use_no_sync: bool = True) -> float:
    """
    Perform a single training step with gradient accumulation and mixed precision support.

    Args:
        model: The neural network model to train
        data_loader: DataLoader providing batches of data
        device: Device to perform computations on
        dtype: Data type for mixed precision training
        scaler: Gradient scaler for mixed precision training (None if disabled)
        gradient_accumulation_steps: Number of steps to accumulate gradients
        gradient_checkpointing: Whether to use gradient checkpointing
        use_no_sync: Whether to use no_sync context for gradient accumulation

    Returns:
        Accumulated loss across all gradient accumulation steps

    Raises:
        RuntimeError: If model is not in training mode or data loading fails
        ValueError: If input dimensions are invalid
    """
    if not model.training:
        model.train()

    accumulation_loss = 0.0

    # Use no_sync context for gradient accumulation if supported
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
                    f'Data loader exhausted after {i} gradient accumulation steps. '
                    f'Expected {gradient_accumulation_steps} steps. '
                    'Check your dataset size and gradient accumulation configuration.'
                )

            # Validate batch format
            if not isinstance(
                    batch, dict
            ) or 'input_ids' not in batch or 'target_ids' not in batch:
                raise ValueError(
                    f'Invalid batch format. Expected dict with input_ids and target_ids, got {type(batch)}'
                )

            # Move tensors to device (non-blocking for better performance)
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            target_ids = batch['target_ids'].to(device, non_blocking=True)

            # Validate tensor dimensions
            if input_ids.ndim != 2:
                raise ValueError(
                    f'Expected input_ids to be 2D, got shape {input_ids.shape}'
                )
            if target_ids.ndim not in [1, 2]:
                raise ValueError(
                    f'Expected target_ids to be 1D or 2D, got shape {target_ids.shape}'
                )

            # Forward pass with mixed precision if enabled
            try:
                forward_context = (autocast(dtype=dtype) if scaler is not None
                                   else torch.enable_grad())

                with forward_context:
                    outputs = model(
                        input_ids=input_ids,
                        gradient_checkpointing=gradient_checkpointing)

                    # Compute the loss
                    batch_size, seq_len = input_ids.shape
                    target_ids_flat = target_ids.reshape(-1)
                    outputs_reshaped = outputs.view(seq_len * batch_size, -1)

                    # Validate shapes match
                    if outputs_reshaped.size(0) != target_ids_flat.size(0):
                        raise ValueError(
                            f'Shape mismatch: outputs {outputs_reshaped.shape} vs targets {target_ids_flat.shape}'
                        )

                    loss = F.cross_entropy(
                        outputs_reshaped, target_ids_flat,
                        reduction='mean') / gradient_accumulation_steps
            except Exception as e:
                raise RuntimeError(f'Model forward pass failed: {e}')

            # Backward pass with gradient scaling if enabled
            try:
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            except Exception as e:
                raise RuntimeError(f'Backward pass failed: {e}')

            # Accumulate loss (detach to avoid keeping computation graph)
            accumulation_loss += loss.detach().item()

            # Clear intermediate variables to free memory
            del outputs, outputs_reshaped, target_ids_flat, input_ids, target_ids

            # Periodic memory cleanup during gradient accumulation for large models
            if i % max(1, gradient_accumulation_steps //
                       4) == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

    return accumulation_loss


def get_dtype(config: ScaleTorchArguments) -> torch.dtype:
    """
    Determine the appropriate data type based on configuration and hardware support.

    Args:
        config: Configuration object containing training settings

    Returns:
        The appropriate torch.dtype for training (torch.bfloat16 or torch.float32)

    Raises:
        ValueError: If configuration is invalid
    """
    use_cpu = getattr(config, 'use_cpu', False)

    # Default to float32
    dtype = torch.float32

    if not use_cpu and torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
            logger.info('Using bfloat16 dtype for training')
        else:
            logger.info(
                'Using float32 dtype for training (bfloat16 not supported)')

    # Flash attention is now using PyTorch native implementation, no dtype restriction
    flash_atten_enabled = os.getenv('FLASH_ATTEN') == '1'
    if flash_atten_enabled:
        logger.info(
            'Using PyTorch native scaled dot product attention (flash attention implementation)'
        )

    return dtype


def validate_config(config: ScaleTorchArguments, world_size: int) -> None:
    """
    Validate the configuration parameters.

    Args:
        config: ScaleTorchArguments object containing all configuration parameters
        world_size: Total number of processes in the distributed setup

    Raises:
        ValueError: If parallelism configuration is inconsistent
    """
    # Validate cp_size divisibility
    if (config.sequence_length and config.context_parallel_size
            and config.sequence_length % config.context_parallel_size != 0):
        raise ValueError(
            f'sequence_length ({config.sequence_length}) must be divisible by '
            f'context_parallel_size ({config.context_parallel_size}) '
            f'for Context Parallelism to work correctly.')

    # Validate world size matches product of all parallelism dimensions
    expected_world_size = (config.tensor_parallel_size *
                           config.pipeline_parallel_size *
                           config.data_parallel_size *
                           config.context_parallel_size)
    if world_size != expected_world_size:
        raise ValueError(
            f'world_size ({world_size}) != tensor_parallel_size ({config.tensor_parallel_size}) * '
            f'pipeline_parallel_size ({config.pipeline_parallel_size}) * '
            f'data_parallel_size ({config.data_parallel_size}) * '
            f'context_parallel_size ({config.context_parallel_size}) = {expected_world_size}. '
            f'Please ensure your distributed setup matches the configured parallelism dimensions.'
        )

    logger.info('Configuration validation passed')


def initialize_distributed_training(
        config: ScaleTorchArguments
) -> Tuple[int, int, int, str, torch.device]:
    """
    Initialize distributed training environment.

    Args:
        config: Configuration object containing distributed settings

    Returns:
        Tuple of (local_rank, global_rank, world_size, backend, device)

    Raises:
        RuntimeError: If distributed environment is not properly configured
        ValueError: If required environment variables are missing
    """
    try:
        # Get rank information with validation
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        global_rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
    except ValueError as e:
        raise ValueError(f'Environment variables must be integers: {e}')

    # Validate rank values
    if local_rank < 0 or global_rank < 0 or world_size <= 0:
        raise ValueError(f'Invalid rank values: local_rank={local_rank}, '
                         f'global_rank={global_rank}, world_size={world_size}')
    if global_rank >= world_size:
        raise ValueError(
            f'global_rank ({global_rank}) must be less than world_size ({world_size})'
        )

    # Set backend based on configuration
    backend = 'gloo' if getattr(config, 'use_cpu', False) else 'nccl'

    # Initialize process group only if world_size > 1
    if world_size > 1:
        try:
            dist.init_process_group(rank=global_rank,
                                    world_size=world_size,
                                    backend=backend,
                                    init_method='env://',
                                    timeout=datetime.timedelta(minutes=3))
        except RuntimeError as e:
            raise RuntimeError(f'Failed to initialize process group: {e}')

        # Set device after initializing the process group
        try:
            if backend == 'nccl':
                torch.cuda.set_device(local_rank)
                device = torch.device('cuda', local_rank)
            else:
                device = torch.device('cpu')
        except RuntimeError as e:
            raise RuntimeError(f'Failed to set device: {e}')

        # Setup process group manager
        try:
            setup_process_group_manager(tp_size=config.tensor_parallel_size,
                                        cp_size=config.context_parallel_size,
                                        pp_size=config.pipeline_parallel_size,
                                        dp_size=config.data_parallel_size)
        except Exception as e:
            dist.destroy_process_group()
            raise RuntimeError(f'Failed to setup process group manager: {e}')
    else:
        # Single process mode
        logger.info('Running in single process mode.')
        # Set device
        if getattr(config, 'use_cpu', False):
            device = torch.device('cpu')
        else:
            device = torch.device(
                'cuda:0' if torch.cuda.is_available() else 'cpu')

    return local_rank, global_rank, world_size, backend, device


def create_model(config: ScaleTorchArguments, dtype: torch.dtype,
                 device: torch.device) -> torch.nn.Module:
    """
    Create and configure the model with parallelism.

    Args:
        config: Configuration object
        dtype: Data type for model parameters
        device: Device to place the model on

    Returns:
        The configured model

    Raises:
        RuntimeError: If model creation fails
        ValueError: If configuration is invalid
    """
    is_print_rank = False
    if pgm is not None:
        is_print_rank = (pgm.tp_rank == 0 and pgm.dp_rank == 0
                         and pgm.cp_rank == 0 and pgm.pp_is_last_stage)

    print(
        f'rank {pgm.global_rank if pgm is not None else 0}: Initializing model meta device',
        is_print_rank=is_print_rank)

    start_time = time.time()

    # Load model configuration
    try:
        model_config = AutoConfig.from_pretrained(config.model_name_or_path,
                                                  trust_remote_code=True)
    except Exception as e:
        raise RuntimeError(f'Failed to load model config: {e}')

    # Initialize model with dematerialized weights
    with init_model_with_dematerialized_weights():
        model = Llama(config=model_config)

        # Apply tensor parallelism if needed
        if pgm is not None and pgm.tp_world_size > 1:
            model = apply_tensor_parallel(model)

        # Apply pipeline parallelism if needed
        if pgm is not None and pgm.pp_world_size > 1:
            model = PipelineParallel(model, model_config)

    # Materialize weights
    model = init_model_with_materialized_weights(
        model, model_config, save_dir='./hf_model_safetensors/')

    # TODO: Load existing checkpoint here to continue pre-training

    # Apply context parallelism if needed
    if pgm is not None and pgm.cp_world_size > 1:
        model = apply_context_parallel(model)

    # Move model to device and set dtype
    model.to(dtype).to(device)

    # Apply data parallelism if needed
    if pgm is not None and pgm.dp_world_size > 1:
        model = DataParallelBucket(model)

    print(f'init model parallel time: {time.time()-start_time:.2f}s',
          is_print_rank=is_print_rank)

    return model


def create_optimizer(model: torch.nn.Module, config: ScaleTorchArguments,
                     device: torch.device) -> OptimizerBase:
    """
    Create and configure the optimizer.

    Args:
        model: The model to optimize
        config: Configuration object containing optimizer settings
        device: Device being used for training

    Returns:
        The configured optimizer

    Raises:
        RuntimeError: If optimizer creation fails
    """
    # Check if fused AdamW is available and should be used
    extra_args = {}
    if getattr(config, 'use_fused_adam', False):
        fused_available = 'fused' in inspect.signature(
            torch.optim.AdamW).parameters
        use_fused = fused_available and device.type == 'cuda'
        extra_args = {'fused': True} if use_fused else {}

        if use_fused:
            logger.info('Using fused AdamW optimizer')
        else:
            logger.info('Using standard AdamW optimizer')

    # Create optimizer
    try:
        optimizer = AdamW(model.parameters(),
                          lr=config.learning_rate,
                          **extra_args)
    except Exception as e:
        raise RuntimeError(f'Failed to create optimizer: {e}')

    return optimizer


def clip_gradients(model: torch.nn.Module, max_norm: float) -> float:
    """
    Clip gradients to prevent exploding gradients.

    Args:
        model: The model whose gradients to clip
        max_norm: Maximum gradient norm

    Returns:
        The gradient norm before clipping
    """
    if max_norm <= 0:
        return 0.0

    # Compute gradient norm
    total_norm = 0.0
    param_count = 0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item()**2
            param_count += 1

    if param_count == 0:
        return 0.0

    total_norm = total_norm**(1. / 2)

    # Clip gradients
    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + 1e-6)
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data.mul_(clip_coef)

    return total_norm


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
                         lr_scheduler: Optional[LRSchedulerBase] = None,
                         grad_norm: Optional[float] = None) -> None:
    """
    Log training metrics to console and optionally to Weights & Biases.

    Args:
        step: Current training step
        loss: Current loss value
        tokens_per_step: Number of tokens processed per step
        step_duration: Time taken for the current step
        trained_tokens: Total number of tokens trained so far
        num_params: Number of parameters in the model
        model_config: Model configuration
        world_size: Total number of processes
        config: Configuration object
        optimizer: Optimizer instance (for learning rate logging)
        lr_scheduler: Learning rate scheduler instance
        grad_norm: Gradient norm (if gradient clipping was applied)
    """
    # Calculate metrics
    tokens_per_second = tokens_per_step / step_duration
    tokens_per_second_per_gpu = tokens_per_second / world_size
    mfu = get_mfu(tokens_per_second_per_gpu, num_params, model_config)

    # Get current learning rate
    current_lr = None
    if optimizer is not None:
        current_lr = optimizer.param_groups[0]['lr']

    # Determine if this rank should log to wandb
    if pgm is not None:
        is_wandb_rank = (pgm.tp_rank == 0 and pgm.dp_rank == 0
                         and pgm.cp_rank == 0 and pgm.pp_is_last_stage)
    else:
        # Single process mode, always log
        is_wandb_rank = True

    if is_wandb_rank:
        # Log to console
        max_tokens = getattr(config, 'max_tokens', None)
        max_tokens_str = ('/' +
                          to_readable_format(max_tokens)) if max_tokens else ''

        # Get rank for logging
        global_rank = pgm.global_rank if pgm is not None else 0

        # Build log message
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
            f'Global batch size: {to_readable_format(tokens_per_step):>7s}',
            f'Tokens/s: {to_readable_format(tokens_per_second):>7s}',
            f'Tokens/s/GPU: {to_readable_format(tokens_per_second_per_gpu):>7s}',
            f'Tokens: {to_readable_format(trained_tokens):>7s}{max_tokens_str}',
            f'MFU: {mfu:5.2f}%',
        ])

        if torch.cuda.is_available():
            log_parts.append(
                f'Memory: {torch.cuda.memory_reserved() / 1e9:6.2f}GB')

        print(' | '.join(log_parts), is_print_rank=is_wandb_rank)

        # Log to Weights & Biases if enabled and available
        if getattr(config, 'use_wandb', False) and wandb is not None:
            log_dict = {
                'loss': loss,
                'tokens_per_step': tokens_per_step,
                'tokens_per_second': tokens_per_second,
                'mfu': mfu,
                'tokens_per_second_per_gpu': tokens_per_second_per_gpu,
                'trained_tokens': trained_tokens
            }

            if current_lr is not None:
                log_dict['learning_rate'] = current_lr

            if grad_norm is not None:
                log_dict['grad_norm'] = grad_norm

            if torch.cuda.is_available():
                log_dict['memory_usage'] = torch.cuda.memory_reserved() / 1e9

            wandb.log(log_dict, step=step)


def get_tensor_shapes(config: ScaleTorchArguments,
                      model_config: PretrainedConfig) -> Dict[str, Any]:
    """
    Calculate tensor shapes for pipeline parallelism.

    Args:
        config: Configuration object
        model_config: Model configuration

    Returns:
        Dictionary containing tensor shape information
    """
    return {
        'input_shape': (config.micro_batch_size, config.sequence_length),
        'output_shape': (config.micro_batch_size, config.sequence_length,
                         model_config.vocab_size),
        'dtype':
        get_dtype(config)
    }


def cleanup_distributed_training(world_size: int) -> None:
    """
    Clean up distributed training resources.

    Args:
        world_size: Total number of processes
    """
    try:
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()
            logger.info('Distributed process group destroyed successfully')
    except Exception as e:
        logger.warning(f'Failed to destroy process group: {e}')


def main() -> None:
    """
    Main training function with comprehensive error handling and resource management.

    This function orchestrates the entire training pipeline including:
    - Configuration parsing and validation
    - Distributed environment setup
    - Model, optimizer, and scheduler initialization
    - Training loop execution
    - Checkpointing and logging
    - Resource cleanup

    Raises:
        RuntimeError: If any critical component fails during initialization or training
    """
    world_size = 1
    is_wandb_rank = True

    try:
        # Parse command-line arguments
        parser = HfArgumentParser(ScaleTorchArguments)
        config, = parser.parse_args_into_dataclasses()

        logger.info('Starting ScaleTorch training script')
        logger.info(f'Configuration: {config}')

        # Get appropriate data type
        dtype = get_dtype(config)
        logger.info(f'Using dtype: {dtype}')

        # Initialize distributed training
        local_rank, global_rank, world_size, backend, device = initialize_distributed_training(
            config)
        logger.info(
            f'Distributed training initialized: local_rank={local_rank}, global_rank={global_rank}, world_size={world_size}, backend={backend}'
        )

        # Validate configuration
        validate_config(config, world_size)

        # Determine if this rank should log to wandb
        if pgm is not None:
            is_wandb_rank = (pgm.tp_rank == 0 and pgm.dp_rank == 0
                             and pgm.cp_rank == 0 and pgm.pp_is_last_stage)

        # Set random seed for reproducibility
        seed = getattr(config, 'seed', 42)
        if not isinstance(seed, int) or seed < 0:
            raise ValueError(
                f'Seed must be a non-negative integer, got {seed}')
        set_all_seed(seed)
        logger.info(f'Random seed set to: {seed}')

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
            num_workers=getattr(config, 'num_workers', 0),
            num_proc=getattr(config, 'num_proc', 1),
            num_samples=getattr(config, 'num_samples', None),
            subset_name=getattr(config, 'subset_name', None),
            split=getattr(config, 'split', 'train'))

        if world_size > 1 and dist.is_initialized():
            dist.barrier()

        print(f'init dataloader time: {time.time()-start_time:.2f}s',
              is_print_rank=is_wandb_rank)

        # Calculate tokens per step
        tokens_per_step = data_loader.global_batch_size * (
            config.sequence_length or 1)
        if is_wandb_rank:
            print('Tokens per step:', to_readable_format(tokens_per_step))

        # Initialize Weights & Biases if enabled and available
        if is_wandb_rank and getattr(config, 'use_wandb',
                                     False) and wandb is not None:
            try:
                wandb.init(
                    project=getattr(config, 'project_name', 'scaletorch'),
                    name=
                    f"{getattr(config, 'experiment_name', 'experiment')}_{to_readable_format(tokens_per_step)}",
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
                        getattr(config, 'max_tokens', None),
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
                logger.info('Weights & Biases initialized successfully')
            except Exception as e:
                logger.warning(f'Failed to initialize wandb: {e}')

        if world_size > 1 and dist.is_initialized():
            dist.barrier()

        # Create model
        logger.info('Creating model...')
        model = create_model(config, dtype, device)

        # Set model to training mode
        model.train()

        # Print model size
        try:
            num_params = get_num_params(model)
            print(f'Number of parameters: {to_readable_format(num_params)}',
                  is_print_rank=is_wandb_rank)
        except Exception as e:
            logger.warning(f'Failed to calculate model parameters: {e}')
            num_params = 0

        # Create optimizer
        logger.info('Creating optimizer...')
        optimizer = create_optimizer(model, config, device)

        # Create learning rate scheduler
        lr_scheduler = None
        try:
            total_train_steps = getattr(config, 'total_train_steps', None)
            if total_train_steps is None and getattr(
                    config, 'max_tokens',
                    None) is not None and tokens_per_step > 0:
                # Estimate total steps from max_tokens
                total_train_steps = getattr(config,
                                            'max_tokens') // tokens_per_step
            lr_scheduler = create_lr_scheduler(optimizer, config,
                                               total_train_steps)
            if lr_scheduler is not None:
                logger.info(
                    f'Learning rate scheduler created: {type(lr_scheduler).__name__}'
                )
        except Exception as e:
            logger.warning(f'Failed to create learning rate scheduler: {e}')

        # Initialize GradScaler for mixed precision training
        scaler = None
        if dtype in [torch.float16, torch.bfloat16
                     ] and not getattr(config, 'use_cpu', False):
            scaler = GradScaler(enabled=dtype == torch.float16)
            logger.info(f'GradScaler initialized for {dtype}')

        # Initialize checkpoint manager
        checkpoint_manager = CheckpointManager()

        # Initialize performance monitor
        monitor = PerformanceMonitor(config=config, log_dir=None)

        # Initialize training state
        trained_tokens, step = 0, 0
        resume_path = getattr(config, 'resume_path', None)
        if resume_path:
            try:
                step, trained_tokens = checkpoint_manager.load_checkpoint(
                    model, optimizer, resume_path)
                print(
                    f'Loaded checkpoint at step {step}, trained_tokens={trained_tokens}',
                    is_print_rank=is_wandb_rank)

                # Load scheduler state if available
                if lr_scheduler is not None:
                    scheduler_path = os.path.join(resume_path, 'scheduler.pt')
                    if os.path.exists(scheduler_path):
                        try:
                            lr_scheduler.load_state_dict(
                                torch.load(scheduler_path))
                            print(
                                f'Loaded scheduler state from {scheduler_path}',
                                is_print_rank=is_wandb_rank)
                        except Exception as e:
                            logger.warning(
                                f'Failed to load scheduler state: {e}')
            except Exception as e:
                logger.warning(f'Failed to load checkpoint: {e}')

        if world_size > 1 and dist.is_initialized():
            dist.barrier()

        # Get tensor shapes for pipeline parallelism
        tensor_shapes = None
        if pgm is not None and pgm.pp_world_size > 1:
            tensor_shapes = get_tensor_shapes(config, model.config)

        # Training loop with error handling
        logger.info('Starting training loop...')
        monitor.start()

        max_tokens = getattr(config, 'max_tokens', None)
        total_train_steps = getattr(config, 'total_train_steps', float('inf'))

        try:
            while max_tokens is None or trained_tokens < max_tokens:
                # Track iteration performance
                monitor.start_iteration(tokens_processed=tokens_per_step)
                step_start_time = time.time()

                # Zero gradients (set_to_none=True for better performance)
                optimizer.zero_grad(set_to_none=True)

                # Perform training step
                try:
                    if pgm is not None and pgm.pp_world_size > 1:
                        # Pipeline parallel training
                        if config.pipeline_parallel_engine == 'afab':
                            loss = train_step_pipeline_afab(
                                model, data_loader, tensor_shapes, device,
                                dtype)
                        elif config.pipeline_parallel_engine == '1f1b':
                            loss = train_step_pipeline_1f1b(
                                model, data_loader, tensor_shapes, device,
                                dtype)
                        else:
                            raise ValueError(
                                f'Invalid pipeline parallel engine: {config.pipeline_parallel_engine}'
                            )
                    else:
                        # Standard training step with mixed precision support
                        gradient_checkpointing = getattr(
                            config, 'gradient_checkpointing', False)
                        loss = train_step(
                            model=model,
                            data_loader=data_loader,
                            device=device,
                            dtype=dtype,
                            scaler=scaler,
                            gradient_accumulation_steps=config.
                            gradient_accumulation_steps,
                            gradient_checkpointing=gradient_checkpointing)
                except Exception as e:
                    raise RuntimeError(
                        f'Training step failed at step {step}: {e}')

                # Calculate tokens processed and end iteration
                metrics = monitor.end_iteration()

                # Average loss across data and context parallel ranks
                try:
                    if pgm is not None:
                        loss = average_loss_across_dp_cp_ranks(loss, device)
                except Exception as e:
                    logger.warning(f'Failed to average loss: {e}')

                # Clip gradients if enabled
                grad_norm = None
                max_grad_norm = getattr(config, 'max_grad_norm', None)
                if max_grad_norm is not None and max_grad_norm > 0:
                    try:
                        grad_norm = clip_gradients(model, max_grad_norm)
                    except Exception as e:
                        logger.warning(f'Failed to clip gradients: {e}')

                # Update parameters with gradient scaling if enabled
                try:
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                except Exception as e:
                    raise RuntimeError(
                        f'Optimizer step failed at step {step}: {e}')

                # Update learning rate scheduler
                if lr_scheduler is not None:
                    try:
                        if isinstance(lr_scheduler, ReduceLROnPlateau):
                            lr_scheduler.step(loss)
                        else:
                            lr_scheduler.step()
                    except Exception as e:
                        logger.warning(
                            f'Failed to update learning rate scheduler: {e}')

                # Update training state
                trained_tokens += tokens_per_step
                step += 1

                # Reset model state if needed
                if hasattr(model, 'reset'):
                    try:
                        model.reset()
                    except Exception as e:
                        logger.warning(f'Model reset failed: {e}')

                # Calculate step duration
                step_duration = time.time() - step_start_time

                # Periodic memory cleanup (every 100 steps) with more aggressive cleanup
                if step % 100 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    # Force synchronization to ensure cache is actually cleared
                    if step % 500 == 0:
                        torch.cuda.synchronize()
                        gc.collect()

                # Log training metrics
                try:
                    log_training_metrics(step=step,
                                         loss=loss,
                                         tokens_per_step=tokens_per_step,
                                         step_duration=step_duration,
                                         trained_tokens=trained_tokens,
                                         num_params=num_params,
                                         model_config=model.config,
                                         world_size=world_size,
                                         config=config,
                                         optimizer=optimizer,
                                         lr_scheduler=lr_scheduler,
                                         grad_norm=grad_norm)

                    # Log performance metrics if this is the wandb rank and wandb is available
                    if is_wandb_rank and getattr(config, 'use_wandb',
                                                 False) and wandb is not None:
                        wandb.log(metrics, step=step)
                except Exception as e:
                    logger.warning(f'Failed to log metrics: {e}')

                # Save checkpoint if needed
                try:
                    save_frequency = getattr(config, 'save_frequency', 0)
                    if save_frequency > 0 and step % save_frequency == 0:
                        checkpoint_dir = Path(config.work_dir) / f'{step}'
                        checkpoint_manager.save_checkpoint(
                            model, optimizer, step, trained_tokens,
                            str(checkpoint_dir))

                        # Save scheduler state separately if available
                        if lr_scheduler is not None and is_wandb_rank:
                            scheduler_path = checkpoint_dir / 'scheduler.pt'
                            torch.save(lr_scheduler.state_dict(),
                                       scheduler_path)
                            logger.info(
                                f'Saved scheduler state to {scheduler_path}')
                except Exception as e:
                    logger.warning(f'Failed to save checkpoint: {e}')

                # Check if we've reached the maximum number of steps
                if step >= total_train_steps:
                    logger.info(
                        f'Reached maximum training steps: {total_train_steps}')
                    break

        except KeyboardInterrupt:
            logger.info('Training interrupted by user.')
        except Exception as e:
            logger.error(f'Error during training: {e}')
            raise
        finally:
            # Cleanup resources
            logger.info('Cleaning up training resources...')

            # Finish Weights & Biases logging if enabled and available
            try:
                if is_wandb_rank and getattr(config, 'use_wandb',
                                             False) and wandb is not None:
                    wandb.finish()
                    logger.info('Weights & Biases logging finished')
            except Exception as e:
                logger.warning(f'Failed to finish wandb: {e}')

            # Save performance logs
            try:
                global_rank_val = pgm.global_rank if pgm is not None else 0
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                monitor.save_stats(
                    f'performance_logs_{global_rank_val}_{timestamp}.json')
                logger.info('Performance logs saved successfully')
            except Exception as e:
                logger.error(f'Error saving performance logs: {e}')

            # Clean up distributed training
            cleanup_distributed_training(world_size)

            logger.info('Training completed successfully')

    except Exception as e:
        logger.error(f'Fatal error in main: {e}')
        # Attempt cleanup
        cleanup_distributed_training(world_size)
        raise


if __name__ == '__main__':
    main()
