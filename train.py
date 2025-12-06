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
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node 4 --master_addr localhost --master_port 25500 train.py --config tmp/fast_benchmark/120M_model_tiny_stories_dp=4.json
CUDA_DEVICE_MAX_CONNECTIONS=1 debugpy-run -p 5678 -m torch.distributed.run -- --nproc_per_node=4 --nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:29400 train.py --config tmp/dummy/llama2_7b_benchmark.json
"""
import argparse
import contextlib
import datetime
import inspect
import json
import os
import time
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

# Make wandb optional
wandb = None
try:
    import wandb
except ImportError:
    pass
from torch.optim import AdamW
from torch.optim.lr_scheduler import (LambdaLR, OneCycleLR, Optimizer,
                                      PolynomialLR, ReduceLROnPlateau, StepLR)
from torch.optim.optimizer import Optimizer as OptimizerBase
from transformers import AutoConfig, PretrainedConfig

from scaletorch.dataset.dataloader import MicroBatchDataLoader
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
from scaletorch.utils.checkpoint import (
    CheckpointManager, init_model_with_dematerialized_weights,
    init_model_with_materialized_weights)
from scaletorch.utils.performance_monitor import PerformanceMonitor
from scaletorch.utils.utils import (average_loss_across_dp_cp_ranks, get_mfu,
                                    get_num_params, print, set_all_seed,
                                    to_readable_format)


def train_step(model: torch.nn.Module,
               data_loader: MicroBatchDataLoader,
               device: torch.device,
               dtype: torch.dtype,
               scaler: Optional[GradScaler] = None,
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
        gradient_checkpointing: Whether to use gradient checkpointing

    Returns:
        Accumulated loss across all gradient accumulation steps

    Raises:
        RuntimeError: If model is not in training mode or data loading fails
        ValueError: If input dimensions are invalid
    """
    try:
        # Ensure model is in training mode
        if not model.training:
            model.train()

        accumulation_loss = 0.0
        grad_accum_steps = data_loader.gradient_accumulation_steps

        # Use no_sync context for gradient accumulation if supported
        # This avoids unnecessary gradient synchronization during accumulation
        sync_context = (model.no_sync() if
                        (use_no_sync and hasattr(model, 'no_sync')
                         and grad_accum_steps > 1) else
                        contextlib.nullcontext())

        # Loop through gradient accumulation steps
        with sync_context:
            for i in range(grad_accum_steps):
                try:
                    # Get the next batch
                    batch = next(data_loader)
                except StopIteration:
                    raise RuntimeError(
                        f'Data loader exhausted after {i} gradient accumulation steps. '
                        f'Expected {grad_accum_steps} steps. '
                        'Check your dataset size and gradient accumulation configuration.'
                    )

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
                if target_ids.ndim != 2 and target_ids.ndim != 1:
                    raise ValueError(
                        f'Expected target_ids to be 1D or 2D, got shape {target_ids.shape}'
                    )

                # Forward pass with mixed precision if enabled
                try:
                    # Compute outputs with or without mixed precision
                    if scaler is not None:
                        forward_context = autocast(dtype=dtype)
                    else:
                        forward_context = torch.enable_grad()

                    with forward_context:
                        outputs = model(
                            input_ids=input_ids,
                            gradient_checkpointing=gradient_checkpointing)

                        # Compute the loss
                        batch_size, seq_len = input_ids.shape
                        target_ids_flat = target_ids.reshape(-1)
                        outputs_reshaped = outputs.view(
                            seq_len * batch_size, -1)

                        # Validate shapes match
                        if outputs_reshaped.size(0) != target_ids_flat.size(0):
                            raise ValueError(
                                f'Shape mismatch: outputs {outputs_reshaped.shape} vs targets {target_ids_flat.shape}'
                            )

                        loss = F.cross_entropy(
                            outputs_reshaped,
                            target_ids_flat,
                            reduction='mean') / grad_accum_steps
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
                if i % max(1, grad_accum_steps //
                           4) == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return accumulation_loss

    except Exception as e:
        print(f'Error in train_step: {e}')
        raise


def setup_environment(config: Dict[str, Any]) -> None:
    """
    Set up environment variables based on the configuration.

    Args:
        config: Configuration dictionary containing environment settings

    Raises:
        ValueError: If required config keys are missing or HF_TOKEN is not set
        KeyError: If expected configuration keys are missing
    """
    try:
        # Validate required configuration sections
        required_sections = ['environment', 'distributed']
        for section in required_sections:
            if section not in config:
                raise KeyError(f"Missing required config section: '{section}'")

        # Validate required environment settings
        required_env_keys = [
            'OMP_NUM_THREADS', 'TOKENIZERS_PARALLELISM', 'FLASH_ATTEN'
        ]
        for key in required_env_keys:
            if key not in config['environment']:
                raise KeyError(
                    f"Missing required environment setting: '{key}'")

        # Set environment variables with type validation
        try:
            omp_threads = int(config['environment']['OMP_NUM_THREADS'])
            if omp_threads <= 0:
                raise ValueError(
                    f'OMP_NUM_THREADS must be positive, got {omp_threads}')
            os.environ['OMP_NUM_THREADS'] = str(omp_threads)
        except (ValueError, TypeError) as e:
            raise ValueError(f'Invalid OMP_NUM_THREADS value: {e}')

        os.environ['TOKENIZERS_PARALLELISM'] = str(
            config['environment']['TOKENIZERS_PARALLELISM'])
        os.environ['FLASH_ATTEN'] = str(config['environment']['FLASH_ATTEN'])
        os.environ[
            'DEVICE'] = 'cpu' if config['distributed']['use_cpu'] else 'cuda'

    except (KeyError, ValueError, TypeError) as e:
        print(f'Error in setup_environment: {e}')
        raise


def get_dtype(config: Dict[str, Any]) -> torch.dtype:
    """
    Determine the appropriate data type based on configuration and hardware support.

    Args:
        config: Configuration dictionary containing training settings

    Returns:
        The appropriate torch.dtype for training (torch.bfloat16 or torch.float32)

    Raises:
        AssertionError: If FLASH_ATTEN is enabled but dtype is not bfloat16
        KeyError: If required config keys are missing
    """
    try:
        use_cpu = config['distributed'].get('use_cpu', False)

        # Determine dtype based on hardware capabilities
        dtype = torch.float32  # Default to float32

        if not use_cpu and torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16

        # Flash attention is now using PyTorch native implementation, no dtype restriction
        flash_atten_enabled = os.getenv('FLASH_ATTEN') == '1'
        if flash_atten_enabled:
            print(
                'Info: Using PyTorch native scaled dot product attention (flash attention implementation)'
            )

        return dtype

    except KeyError as e:
        raise KeyError(f'Missing required config key: {e}')
    except Exception as e:
        print(f'Error in get_dtype: {e}')
        raise


def validate_config(config: Dict[str, Any], world_size: int) -> None:
    """
    Validate the configuration parameters.

    Args:
        config: Configuration dictionary
        world_size: Total number of processes in the distributed setup

    Raises:
        AssertionError: If configuration parameters are invalid
        KeyError: If required config keys are missing
        ValueError: If parallelism configuration is inconsistent
    """
    # Add gradient_checkpointing to training config if not present
    if 'gradient_checkpointing' not in config['training']:
        config['training']['gradient_checkpointing'] = False
    try:
        # Validate required distributed config keys
        # Support both short and full key names
        key_mappings = {
            'tp_size': ['tp_size', 'tensor_parallel_size'],
            'pp_size': ['pp_size', 'pipeline_parallel_size'],
            'dp_size': ['dp_size', 'data_parallel_size'],
            'cp_size': ['cp_size', 'context_parallel_size']
        }

        for short_key, possible_keys in key_mappings.items():
            found = False
            for key in possible_keys:
                if key in config['distributed']:
                    # Use the found key value
                    config['distributed'][short_key] = config['distributed'][
                        key]
                    found = True
                    break
            if not found:
                raise KeyError(
                    f"Missing required distributed config: '{short_key}' (or its full name equivalent)"
                )

        # Validate training config keys
        # Support both sequence_length and max_sequence_length for compatibility
        if 'sequence_length' in config['training']:
            sequence_length = config['training']['sequence_length']
        elif 'max_sequence_length' in config['training']:
            sequence_length = config['training']['max_sequence_length']
            # Add sequence_length for compatibility
            config['training']['sequence_length'] = sequence_length
        else:
            raise KeyError(
                "Missing required training config: 'sequence_length' or 'max_sequence_length'"
            )

        # Validate cp_size divisibility
        cp_size = config['distributed']['cp_size']
        sequence_length = config['training']['sequence_length']

        if cp_size <= 0:
            raise ValueError(f'cp_size must be positive, got {cp_size}')
        if sequence_length <= 0:
            raise ValueError(
                f'sequence_length must be positive, got {sequence_length}')

        if sequence_length % cp_size != 0:
            raise ValueError(
                f'sequence_length ({sequence_length}) must be divisible by cp_size ({cp_size}) '
                f'for Context Parallelism to work correctly.')

        # Validate world size matches product of all parallelism dimensions
        tp_size = config['distributed']['tp_size']
        pp_size = config['distributed']['pp_size']
        dp_size = config['distributed']['dp_size']

        for size in [tp_size, pp_size, dp_size]:
            if size <= 0:
                raise ValueError(
                    f'Parallelism size must be positive, got {size}')

        expected_world_size = tp_size * pp_size * dp_size * cp_size
        if world_size != expected_world_size:
            raise ValueError(
                f'world_size ({world_size}) != tp_size ({tp_size}) * pp_size ({pp_size}) * '
                f'dp_size ({dp_size}) * cp_size ({cp_size}) = {expected_world_size}. '
                f'Please ensure your distributed setup matches the configured parallelism dimensions.'
            )

    except (KeyError, ValueError) as e:
        print(f'Configuration validation error: {e}')
        raise


def initialize_distributed_training(
        config: Dict[str, Any]) -> Tuple[int, int, int, str, torch.device]:
    """
    Initialize distributed training environment.

    Args:
        config: Configuration dictionary containing distributed settings

    Returns:
        Tuple of (local_rank, global_rank, world_size, backend, device)

    Raises:
        RuntimeError: If distributed environment is not properly configured
        ValueError: If required environment variables are missing
    """
    try:
        # Get rank information with validation
        try:
            local_rank = int(os.environ['LOCAL_RANK'])
            global_rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
        except KeyError as e:
            # Fallback to single process mode if no distributed environment variables are set
            print(
                f'Warning: Missing distributed environment variable {e}. Falling back to single process mode.'
            )
            local_rank = 0
            global_rank = 0
            world_size = 1
        except ValueError as e:
            raise ValueError(f'Environment variables must be integers: {e}')

        # Validate rank values
        if local_rank < 0 or global_rank < 0 or world_size <= 0:
            raise ValueError(
                f'Invalid rank values: local_rank={local_rank}, '
                f'global_rank={global_rank}, world_size={world_size}')
        if global_rank >= world_size:
            raise ValueError(
                f'global_rank ({global_rank}) must be less than world_size ({world_size})'
            )

        # Set backend based on configuration
        backend = 'gloo' if config['distributed']['use_cpu'] else 'nccl'

        # Initialize process group only if world_size > 1
        if world_size > 1:
            # Initialize process group with error handling
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
                setup_process_group_manager(
                    tp_size=config['distributed']['tp_size'],
                    cp_size=config['distributed']['cp_size'],
                    pp_size=config['distributed']['pp_size'],
                    dp_size=config['distributed']['dp_size'])
            except Exception as e:
                dist.destroy_process_group()
                raise RuntimeError(
                    f'Failed to setup process group manager: {e}')
        else:
            # Single process mode
            print('Running in single process mode.')
            # Set device
            if config['distributed']['use_cpu']:
                device = torch.device('cpu')
            else:
                device = torch.device(
                    'cuda:0' if torch.cuda.is_available() else 'cpu')

            # No need to setup process group manager in single process mode

        return local_rank, global_rank, world_size, backend, device

    except Exception as e:
        print(f'Error in initialize_distributed_training: {e}')
        raise


def create_model_config(config: Dict[str, Any],
                        device: torch.device) -> PretrainedConfig:
    """
    Create and configure the model configuration.

    Args:
        config: Configuration dictionary containing model settings
        device: Device to use for broadcasting the model config

    Returns:
        The configured PretrainedConfig object
    """
    if pgm.global_rank == 0:
        print(f'rank {pgm.global_rank}: Creating model config')
        model_config = AutoConfig.from_pretrained(
            config['model']['model_name_or_path'])

        # Modify model structure if specified in config
        model_config.num_hidden_layers = (config['model'].get(
            'num_hidden_layers', model_config.num_hidden_layers))
        model_config.num_attention_heads = (config['model'].get(
            'num_attention_heads', model_config.num_attention_heads))
        model_config.num_key_value_heads = (config['model'].get(
            'num_key_value_heads', model_config.num_key_value_heads))
        model_config.max_position_embeddings = config['training'][
            'sequence_length']

        objects = [model_config]
    else:
        objects = [None]

        # Broadcast model config to all ranks if distributed
        if dist.is_initialized():
            dist.broadcast_object_list(objects, src=0, device=device)
        model_config = objects[0]

    print(f'rank {pgm.global_rank}: Broadcasting model_config to all ranks',
          is_print_rank=pgm.global_rank == 0)

    return model_config


def create_model(model_config: PretrainedConfig, config: Dict[str, Any],
                 dtype: torch.dtype, device: torch.device) -> torch.nn.Module:
    """
    Create and configure the model with parallelism.

    Args:
        model_config: The model configuration
        config: Configuration dictionary
        dtype: Data type for model parameters
        device: Device to place the model on

    Returns:
        The configured model
    """
    print(f'rank {pgm.global_rank}: Initializing model meta device',
          is_print_rank=pgm.tp_rank == 0 and pgm.dp_rank == 0
          and pgm.cp_rank == 0 and pgm.pp_is_last_stage)

    start_time = time.time()

    # Initialize model with dematerialized weights
    with init_model_with_dematerialized_weights():
        model = Llama(config=model_config)

        # Apply tensor parallelism if needed
        if pgm.tp_world_size > 1:
            model = apply_tensor_parallel(model)

        # Apply pipeline parallelism if needed
        if pgm.pp_world_size > 1:
            model = PipelineParallel(model, model_config)

    # Materialize weights
    model = init_model_with_materialized_weights(
        model, model_config, save_dir='./hf_model_safetensors/')

    # TODO: Load existing checkpoint here to continue pre-training

    # Apply context parallelism if needed
    if pgm.cp_world_size > 1:
        model = apply_context_parallel(model)

    # Move model to device and set dtype
    model.to(dtype).to(device)

    # Apply data parallelism if needed
    if pgm.dp_world_size > 1:
        model = DataParallelBucket(model)

    print(f'init model parallel time: {time.time()-start_time:.2f}s',
          is_print_rank=pgm.tp_rank == 0 and pgm.dp_rank == 0
          and pgm.cp_rank == 0 and pgm.pp_is_last_stage)

    return model


def create_optimizer(model: torch.nn.Module, config: Dict[str, Any],
                     device: torch.device) -> OptimizerBase:
    """
    Create and configure the optimizer.

    Args:
        model: The model to optimize
        config: Configuration dictionary containing optimizer settings
        device: Device being used for training

    Returns:
        The configured optimizer
    """
    # Check if fused AdamW is available and should be used
    extra_args = {}
    if config['model']['use_fused_adam']:
        fused_available = 'fused' in inspect.signature(
            torch.optim.AdamW).parameters
        use_fused = fused_available and device.type == 'cuda'
        extra_args = {'fused': True} if use_fused else {}

    # Create optimizer
    optimizer = AdamW(model.parameters(),
                      lr=config['training']['learning_rate'],
                      **extra_args)

    return optimizer


def create_lr_scheduler(
    optimizer: OptimizerBase,
    config: Dict[str, Any],
    num_training_steps: Optional[int] = None
) -> Optional[Union[Optimizer, ReduceLROnPlateau]]:
    """
    Create and configure the learning rate scheduler.

    Args:
        optimizer: The optimizer to schedule
        config: Configuration dictionary containing scheduler settings
        num_training_steps: Total number of training steps (for some schedulers)

    Returns:
        The configured learning rate scheduler, or None if not configured
    """
    scheduler_config = config['training'].get('lr_scheduler', None)
    if scheduler_config is None:
        return None

    scheduler_type = scheduler_config.get('type', 'linear').lower()

    if scheduler_type == 'linear':
        # Linear warmup + linear decay
        warmup_steps = scheduler_config.get('warmup_steps', 0)
        if num_training_steps is None:
            return None
        return LambdaLR(optimizer,
                        lr_lambda=lambda step: min(1.0, step / warmup_steps)
                        if step < warmup_steps else max(
                            0.0, (num_training_steps - step) /
                            (num_training_steps - warmup_steps)))
    elif scheduler_type == 'cosine':
        # Cosine annealing with optional warmup
        T_max = scheduler_config.get('T_max', num_training_steps)
        eta_min = scheduler_config.get('eta_min', 0.0)
        warmup_steps = scheduler_config.get('warmup_steps', 0)
        if num_training_steps is None:
            return None

        def lr_lambda(step):
            if step < warmup_steps:
                return min(1.0, step / warmup_steps)
            else:
                progress = (step - warmup_steps) / (T_max - warmup_steps)
                return eta_min + (1.0 - eta_min) * 0.5 * (
                    1 + np.cos(np.pi * progress))

        return LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif scheduler_type == 'polynomial':
        # Polynomial decay
        power = scheduler_config.get('power', 1.0)
        warmup_steps = scheduler_config.get('warmup_steps', 0)
        if num_training_steps is None:
            return None
        return PolynomialLR(optimizer,
                            total_iters=num_training_steps,
                            power=power)
    elif scheduler_type == 'step':
        # Step decay
        step_size = scheduler_config.get('step_size', 1)
        gamma = scheduler_config.get('gamma', 0.1)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == 'onecycle':
        # OneCycleLR
        max_lr = scheduler_config.get('max_lr',
                                      config['training']['learning_rate'])
        pct_start = scheduler_config.get('pct_start', 0.3)
        if num_training_steps is None:
            return None
        return OneCycleLR(optimizer,
                          max_lr=max_lr,
                          total_steps=num_training_steps,
                          pct_start=pct_start)
    elif scheduler_type == 'reduce_on_plateau':
        # ReduceLROnPlateau (requires manual step with metric)
        mode = scheduler_config.get('mode', 'min')
        factor = scheduler_config.get('factor', 0.1)
        patience = scheduler_config.get('patience', 10)
        return ReduceLROnPlateau(optimizer,
                                 mode=mode,
                                 factor=factor,
                                 patience=patience)
    else:
        print(
            f'Warning: Unknown scheduler type {scheduler_type}, using no scheduler'
        )
        return None


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
                         config: Dict[str, Any],
                         optimizer: Optional[OptimizerBase] = None,
                         lr_scheduler: Optional[Optimizer] = None,
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
        config: Configuration dictionary
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
        max_tokens_str = (
            '/' + to_readable_format(config['training'].get('max_tokens'))
            if config['training'].get('max_tokens') else '')

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
        if config['logging']['use_wandb'] and wandb is not None:
            log_dict = {
                'loss': loss,
                'tokens_per_step': tokens_per_step,
                'tokens_per_second': tokens_per_step / step_duration,
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


def main() -> None:
    """Main training function with comprehensive error handling."""
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(
            description='Train LLaMA model with distributed parallelism')
        parser.add_argument('--config',
                            type=str,
                            default='',
                            help='Path to config file')
        args = parser.parse_args()

        # Validate config file path
        if not args.config:
            raise ValueError('--config argument is required')
        if not os.path.isfile(args.config):
            raise FileNotFoundError(f'Config file not found: {args.config}')

        # Load configuration
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f'Invalid JSON in config file: {e}')
        except Exception as e:
            raise RuntimeError(f'Failed to load config file: {e}')

        # Setup environment
        try:
            setup_environment(config)
        except Exception as e:
            raise RuntimeError(f'Environment setup failed: {e}')

        # Get appropriate data type
        try:
            dtype = get_dtype(config)
        except Exception as e:
            raise RuntimeError(f'Failed to determine dtype: {e}')

        # Initialize distributed training
        try:
            local_rank, global_rank, world_size, backend, device = initialize_distributed_training(
                config)
        except Exception as e:
            raise RuntimeError(
                f'Distributed training initialization failed: {e}')

        # Validate configuration
        try:
            validate_config(config, world_size)
        except Exception as e:
            # Only destroy process group if it's initialized
            if world_size > 1 and dist.is_initialized():
                dist.destroy_process_group()
            raise RuntimeError(f'Configuration validation failed: {e}')

        # Determine if this rank should log to wandb
        if pgm is not None:
            is_wandb_rank = (pgm.tp_rank == 0 and pgm.dp_rank == 0
                             and pgm.cp_rank == 0 and pgm.pp_is_last_stage)
        else:
            # Single process mode, always log
            is_wandb_rank = True

        # Set random seed for reproducibility
        try:
            seed = config['training'].get('seed', 42)
            if not isinstance(seed, int) or seed < 0:
                raise ValueError(
                    f'Seed must be a non-negative integer, got {seed}')
            set_all_seed(seed)
        except Exception as e:
            raise RuntimeError(f'Failed to set random seed: {e}')

        # Initialize data loader
        try:
            start_time = time.time()
            data_loader = MicroBatchDataLoader(
                micro_batch_size=config['training']['micro_batch_size'],
                sequence_length=config['training']['sequence_length'],
                dataset_name=config['dataset']['name'],
                tokenizer_name=config['model']['model_name_or_path'],
                gradient_accumulation_steps=config['training']
                ['gradient_accumulation_steps'],
                device=device,
                num_workers=config['dataset']['num_workers'],
                num_proc=config['dataset']['num_proc'],
                num_samples=config['training'].get('num_samples', None),
                subset_name=config['dataset'].get('subset_name', None),
                split=config['dataset'].get('split', 'train'))

            if world_size > 1 and dist.is_initialized():
                dist.barrier()

            print(f'init dataloader time: {time.time()-start_time:.2f}s',
                  is_print_rank=is_wandb_rank)
        except Exception as e:
            # Only destroy process group if it's initialized
            if world_size > 1 and dist.is_initialized():
                dist.destroy_process_group()
            raise RuntimeError(f'Data loader initialization failed: {e}')

        # Calculate tokens per step
        try:
            tokens_per_step = data_loader.global_batch_size * config[
                'training']['sequence_length']

            # Print tokens per step
            if (pgm is None or pgm.global_rank == 0):
                print('Tokens per step:',
                      to_readable_format(tokens_per_step),
                      is_print_rank=is_wandb_rank)
        except Exception as e:
            dist.destroy_process_group()
            raise RuntimeError(f'Failed to calculate tokens per step: {e}')

        # Initialize Weights & Biases if enabled and available
        try:
            if is_wandb_rank and config['logging'][
                    'use_wandb'] and wandb is not None:
                wandb.init(
                    project='scaletorch',
                    name=
                    f"{config['logging']['run_name']}_{to_readable_format(tokens_per_step)}_{pgm}",
                    config={
                        'tensor_parallel_size':
                        pgm.tp_world_size,
                        'context_parallel_size':
                        pgm.cp_world_size,
                        'pipeline_parallel_size':
                        pgm.pp_world_size,
                        'data_parallel_size':
                        pgm.dp_world_size,
                        'model':
                        config['model']['model_name_or_path'],
                        'dataset':
                        config['dataset']['name'],
                        'max_tokens':
                        config['training']['max_tokens'],
                        'learning_rate':
                        config['training']['learning_rate'],
                        'seed':
                        config['training']['seed'],
                        'micro_batch_size':
                        data_loader.micro_batch_size,
                        'global_batch_size':
                        data_loader.global_batch_size,
                        'gradient_accumulation':
                        data_loader.gradient_accumulation_steps,
                    },
                )
        except Exception as e:
            print(f'Warning: Failed to initialize wandb: {e}')

        # Create model configuration
        try:
            model_config = create_model_config(config, device)
        except Exception as e:
            dist.destroy_process_group()
            raise RuntimeError(f'Failed to create model configuration: {e}')

        dist.barrier()

        # Create model
        try:
            model = create_model(model_config, config, dtype, device)
        except Exception as e:
            dist.destroy_process_group()
            raise RuntimeError(f'Model creation failed: {e}')

        # Set model to training mode
        model.train()

        # Print model size
        try:
            num_params = get_num_params(model)
            print(f'Number of parameters: {to_readable_format(num_params)}',
                  is_print_rank=is_wandb_rank)
        except Exception as e:
            print(f'Warning: Failed to calculate model parameters: {e}')
            num_params = 0

        # Define tensor shapes for pipeline parallelism
        tensor_shapes = (data_loader.micro_batch_size,
                         data_loader.sequence_length_per_gpu,
                         model_config.hidden_size)

        # Create optimizer
        try:
            optimizer = create_optimizer(model, config, device)
        except Exception as e:
            dist.destroy_process_group()
            raise RuntimeError(f'Optimizer creation failed: {e}')

        # Create learning rate scheduler
        lr_scheduler = None
        try:
            total_train_steps = config['training'].get('total_train_steps',
                                                       None)
            if total_train_steps is None and config['training'].get(
                    'max_tokens') is not None:
                # Estimate total steps from max_tokens
                total_train_steps = config['training'][
                    'max_tokens'] // tokens_per_step
            lr_scheduler = create_lr_scheduler(optimizer, config,
                                               total_train_steps)
        except Exception as e:
            print(f'Warning: Failed to create learning rate scheduler: {e}')

        # Initialize GradScaler for mixed precision training
        scaler = None
        if dtype in [torch.float16, torch.bfloat16
                     ] and not config['distributed']['use_cpu']:
            scaler = GradScaler(
                enabled=dtype == torch.float16
            )  # Only use scaler for float16, bfloat16 doesn't need scaling

        # Initialize checkpoint manager
        checkpoint_manager = CheckpointManager()

        # Initialize performance monitor
        monitor = PerformanceMonitor(config=config, log_dir=None)

        # Initialize training state
        trained_tokens, step = 0, 0
        if config['checkpoint'].get('load_path'):
            try:
                step, trained_tokens = checkpoint_manager.load_checkpoint(
                    model, optimizer, config['checkpoint']['load_path'])
                print(
                    f'Loaded checkpoint at step {step}, trained_tokens={trained_tokens}',
                    is_print_rank=is_wandb_rank)

                # Load scheduler state if available
                if lr_scheduler is not None:
                    scheduler_path = os.path.join(
                        config['checkpoint']['load_path'], 'scheduler.pt')
                    if os.path.exists(scheduler_path):
                        try:
                            lr_scheduler.load_state_dict(
                                torch.load(scheduler_path))
                            print(
                                f'Loaded scheduler state from {scheduler_path}',
                                is_print_rank=is_wandb_rank)
                        except Exception as e:
                            print(
                                f'Warning: Failed to load scheduler state: {e}'
                            )
            except Exception as e:
                print(f'Warning: Failed to load checkpoint: {e}')

        dist.barrier()

        # Training loop with error handling
        try:
            # Start performance monitoring
            monitor.start()

            while (config['training'].get('max_tokens') is None
                   or trained_tokens < config['training'].get('max_tokens')):
                # Track iteration performance
                monitor.start_iteration(tokens_processed=tokens_per_step)
                step_start_time = time.time()

                # Zero gradients (set_to_none=True for better performance)
                optimizer.zero_grad(set_to_none=True)

                # Perform training step
                try:
                    if pgm and pgm.pp_world_size > 1:
                        # Pipeline parallel training
                        if config['distributed']['pp_engine'] == 'afab':
                            loss = train_step_pipeline_afab(
                                model, data_loader, tensor_shapes, device,
                                dtype)
                        elif config['distributed']['pp_engine'] == '1f1b':
                            loss = train_step_pipeline_1f1b(
                                model, data_loader, tensor_shapes, device,
                                dtype)
                        else:
                            raise ValueError(
                                f"Invalid pipeline parallel engine: {config['distributed']['pp_engine']}"
                            )
                    else:
                        # Standard training step with mixed precision support
                        # Get gradient checkpointing configuration
                        gradient_checkpointing = config['training'].get(
                            'gradient_checkpointing', False)
                        loss = train_step(
                            model,
                            data_loader,
                            device,
                            dtype,
                            scaler,
                            gradient_checkpointing=gradient_checkpointing)
                except Exception as e:
                    raise RuntimeError(
                        f'Training step failed at step {step}: {e}')

                # Calculate tokens processed and end iteration
                metrics = monitor.end_iteration()

                # Average loss across data and context parallel ranks
                try:
                    loss = average_loss_across_dp_cp_ranks(loss, device)
                except Exception as e:
                    print(f'Warning: Failed to average loss: {e}')

                # Clip gradients if enabled
                grad_norm = None
                max_grad_norm = config['training'].get('max_grad_norm', None)
                if max_grad_norm is not None and max_grad_norm > 0:
                    try:
                        grad_norm = clip_gradients(model, max_grad_norm)
                    except Exception as e:
                        print(f'Warning: Failed to clip gradients: {e}')

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
                        print(
                            f'Warning: Failed to update learning rate scheduler: {e}'
                        )

                # Update training state
                trained_tokens += tokens_per_step
                step += 1

                # Reset model state if needed
                if hasattr(model, 'reset'):
                    try:
                        model.reset()
                    except Exception as e:
                        print(f'Warning: Model reset failed: {e}')

                # Calculate step duration
                step_duration = time.time() - step_start_time

                # Periodic memory cleanup (every 100 steps) with more aggressive cleanup
                if step % 100 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    # Force synchronization to ensure cache is actually cleared
                    if step % 500 == 0:
                        torch.cuda.synchronize()
                        import gc
                        gc.collect()

                # Log training metrics
                try:
                    log_training_metrics(step, loss, tokens_per_step,
                                         step_duration, trained_tokens,
                                         num_params, model_config, world_size,
                                         config, optimizer, lr_scheduler,
                                         grad_norm)

                    # Log performance metrics if this is the wandb rank and wandb is available
                    if is_wandb_rank and config['logging'][
                            'use_wandb'] and wandb is not None:
                        wandb.log(metrics, step=step)
                except Exception as e:
                    print(f'Warning: Failed to log metrics: {e}')

                # Save checkpoint if needed
                try:
                    if (config['checkpoint'].get('save_frequency', 0) > 0
                            and step % config['checkpoint']['save_frequency']
                            == 0):
                        checkpoint_manager.save_checkpoint(
                            model, optimizer, step, trained_tokens,
                            config['checkpoint']['save_dir'] + f'/{step}')

                        # Save scheduler state separately if available
                        if lr_scheduler is not None and is_wandb_rank:
                            scheduler_path = os.path.join(
                                config['checkpoint']['save_dir'], f'{step}',
                                'scheduler.pt')
                            os.makedirs(os.path.dirname(scheduler_path),
                                        exist_ok=True)
                            torch.save(lr_scheduler.state_dict(),
                                       scheduler_path)
                except Exception as e:
                    print(f'Warning: Failed to save checkpoint: {e}')

                # Check if we've reached the maximum number of steps
                if step >= config['training'].get('total_train_steps',
                                                  float('inf')):
                    break

        except KeyboardInterrupt:
            print('Training interrupted by user.')
        except Exception as e:
            print(f'Error during training: {e}')
            raise
        finally:
            # Finish Weights & Biases logging if enabled and available
            try:
                if is_wandb_rank and config['logging'][
                        'use_wandb'] and wandb is not None:
                    wandb.finish()
            except Exception as e:
                print(f'Warning: Failed to finish wandb: {e}')

            # Save performance logs
            try:
                global_rank = pgm.global_rank if pgm else 0
                monitor.save_stats(
                    f"performance_logs_{global_rank}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )
            except Exception as e:
                print(f'Error saving performance logs: {e}')

            # Clean up distributed training
            try:
                dist.destroy_process_group()
            except Exception as e:
                print(f'Warning: Failed to destroy process group: {e}')

    except Exception as e:
        print(f'Fatal error in main: {e}')
        # Attempt cleanup
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except:
            pass
        raise


if __name__ == '__main__':
    main()
