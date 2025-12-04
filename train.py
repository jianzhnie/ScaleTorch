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
import datetime
import inspect
import json
import os
import time
from typing import Any, Dict, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer
from transformers import AutoConfig, PretrainedConfig

import scaletorch.parallel.pg_manager as pgm
from scaletorch.dataset.dataloader import MicroBatchDataLoader
from scaletorch.model.model_llama import Llama
from scaletorch.parallel.context_parallel.context_parallel import \
    apply_context_parallel
from scaletorch.parallel.data_parallel.data_parallel import DataParallelBucket
from scaletorch.parallel.pg_manager import setup_process_group_manager
from scaletorch.parallel.pipeline_parallel.pipeline_parallel import (
    PipelineParallel, train_step_pipeline_1f1b, train_step_pipeline_afab)
from scaletorch.parallel.tensor_parallel.tensor_parallel import \
    apply_tensor_parallel
from scaletorch.utils.checkpoint import (
    CheckpointManager, init_model_with_dematerialized_weights,
    init_model_with_materialized_weights)
from scaletorch.utils.utils import (average_loss_across_dp_cp_ranks, get_mfu,
                                    get_num_params, print, set_all_seed,
                                    to_readable_format)


def train_step(model: torch.nn.Module, data_loader: MicroBatchDataLoader,
               device: torch.device) -> float:
    """
    Perform a single training step with gradient accumulation.

    Args:
        model: The neural network model to train
        data_loader: DataLoader providing batches of data
        device: Device to perform computations on

    Returns:
        Accumulated loss across all gradient accumulation steps
    """
    # Ensure model is in training mode
    if not model.training:
        model.train()

    acc_loss = 0.0

    # Loop through gradient accumulation steps
    for i in range(data_loader.grad_acc_steps):
        # Get the next batch
        batch = next(data_loader)
        input_ids = batch['input_ids'].to(device)
        target_ids = batch['target_ids'].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids)

        # Compute the loss
        batch_size, seq_len = input_ids.shape
        target_ids = target_ids.reshape(-1)
        outputs = outputs.view(seq_len * batch_size, -1)
        loss = F.cross_entropy(outputs, target_ids,
                               reduction='mean') / data_loader.grad_acc_steps

        # Backward pass
        loss.backward()

        # Accumulate loss
        acc_loss += loss.item()

    return acc_loss


def setup_environment(config: Dict[str, Any]) -> None:
    """
    Set up environment variables based on the configuration.

    Args:
        config: Configuration dictionary containing environment settings

    Raises:
        ValueError: If HF_TOKEN is not set in config or environment
    """
    # Set environment variables
    os.environ['OMP_NUM_THREADS'] = str(
        config['environment']['OMP_NUM_THREADS'])
    os.environ['TOKENIZERS_PARALLELISM'] = config['environment'][
        'TOKENIZERS_PARALLELISM']
    os.environ['FLASH_ATTEN'] = config['environment']['FLASH_ATTEN']
    os.environ[
        'DEVICE'] = 'cpu' if config['distributed']['use_cpu'] else 'cuda'

    # Handle HF_TOKEN
    if config['environment'].get('HF_TOKEN') is None:
        if 'HF_TOKEN' not in os.environ:
            raise ValueError(
                'HF_TOKEN is neither set in the config file nor in the environment'
            )
    else:
        if 'HF_TOKEN' not in os.environ:
            os.environ['HF_TOKEN'] = config['environment']['HF_TOKEN']
        else:
            print(
                'Warning: HF_TOKEN is set in the environment and the config file. Using the environment variable.'
            )


def get_dtype(config: Dict[str, Any]) -> torch.dtype:
    """
    Determine the appropriate data type based on configuration and hardware support.

    Args:
        config: Configuration dictionary containing training settings

    Returns:
        The appropriate torch.dtype for training

    Raises:
        AssertionError: If FLASH_ATTEN is enabled but dtype is not bfloat16
    """
    dtype = (torch.bfloat16
             if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
             and not config['distributed']['use_cpu'] else torch.float32)

    assert ((dtype == torch.bfloat16 and os.getenv('FLASH_ATTEN') == '1')
            or os.getenv('FLASH_ATTEN') != '1'
            ), 'Kernel operations requires dtype=torch.bfloat16'

    return dtype


def validate_config(config: Dict[str, Any], world_size: int) -> None:
    """
    Validate the configuration parameters.

    Args:
        config: Configuration dictionary
        world_size: Total number of processes in the distributed setup

    Raises:
        AssertionError: If configuration parameters are invalid
    """
    assert config['training']['seq_length'] % config['distributed'][
        'cp_size'] == 0, (
            'seq_length must be divisible by cp_size for Context Parallelism')

    assert world_size == (
        config['distributed']['tp_size'] * config['distributed']['pp_size'] *
        config['distributed']['dp_size'] * config['distributed']['cp_size']
    ), 'world_size must be equal to tp_size * pp_size * dp_size * cp_size'


def initialize_distributed_training(
        config: Dict[str, Any]) -> Tuple[int, int, int, str, torch.device]:
    """
    Initialize distributed training environment.

    Args:
        config: Configuration dictionary containing distributed settings

    Returns:
        Tuple of (local_rank, global_rank, world_size, backend, device)
    """
    # Get rank information
    local_rank = int(os.environ['LOCAL_RANK'])
    global_rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    # Set backend based on configuration
    backend = 'gloo' if config['distributed']['use_cpu'] else 'nccl'

    # Initialize process group
    dist.init_process_group(rank=global_rank,
                            world_size=world_size,
                            backend=backend,
                            init_method='env://',
                            timeout=datetime.timedelta(minutes=3))

    # Set device after initializing the process group
    if backend == 'nccl':
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cpu')

    # Setup process group manager
    setup_process_group_manager(tp_size=config['distributed']['tp_size'],
                                cp_size=config['distributed']['cp_size'],
                                pp_size=config['distributed']['pp_size'],
                                dp_size=config['distributed']['dp_size'])

    return local_rank, global_rank, world_size, backend, device


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
    if pgm.process_group_manager.global_rank == 0:
        print(
            f'rank {pgm.process_group_manager.global_rank}: Creating model config'
        )
        model_config = AutoConfig.from_pretrained(config['model']['name'])

        # Modify model structure if specified in config
        model_config.num_hidden_layers = (config['model'].get(
            'num_hidden_layers', model_config.num_hidden_layers))
        model_config.num_attention_heads = (config['model'].get(
            'num_attention_heads', model_config.num_attention_heads))
        model_config.num_key_value_heads = (config['model'].get(
            'num_key_value_heads', model_config.num_key_value_heads))
        model_config.max_position_embeddings = config['training']['seq_length']

        objects = [model_config]
    else:
        objects = [None]

    # Broadcast model config to all ranks
    dist.broadcast_object_list(objects, src=0, device=device)
    model_config = objects[0]

    print(
        f'rank {pgm.process_group_manager.global_rank}: Broadcasting model_config to all ranks',
        is_print_rank=pgm.process_group_manager.global_rank == 0)

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
    print(
        f'rank {pgm.process_group_manager.global_rank}: Initializing model meta device',
        is_print_rank=pgm.process_group_manager.tp_rank == 0
        and pgm.process_group_manager.dp_rank == 0
        and pgm.process_group_manager.cp_rank == 0
        and pgm.process_group_manager.pp_is_last_stage)

    start_time = time.time()

    # Initialize model with dematerialized weights
    with init_model_with_dematerialized_weights():
        model = Llama(config=model_config)

        # Apply tensor parallelism if needed
        if pgm.process_group_manager.tp_world_size > 1:
            model = apply_tensor_parallel(model)

        # Apply pipeline parallelism if needed
        if pgm.process_group_manager.pp_world_size > 1:
            model = PipelineParallel(model, model_config)

    # Materialize weights
    model = init_model_with_materialized_weights(
        model, model_config, save_dir='./hf_model_safetensors/')

    # TODO: Load existing checkpoint here to continue pre-training

    # Apply context parallelism if needed
    if pgm.process_group_manager.cp_world_size > 1:
        model = apply_context_parallel(model)

    # Move model to device and set dtype
    model.to(dtype).to(device)

    # Apply data parallelism if needed
    if pgm.process_group_manager.dp_world_size > 1:
        model = DataParallelBucket(model)

    print(f'init model parallel time: {time.time()-start_time:.2f}s',
          is_print_rank=pgm.process_group_manager.tp_rank == 0
          and pgm.process_group_manager.dp_rank == 0
          and pgm.process_group_manager.cp_rank == 0
          and pgm.process_group_manager.pp_is_last_stage)

    return model


def create_optimizer(model: torch.nn.Module, config: Dict[str, Any],
                     device: torch.device) -> Optimizer:
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


def log_training_metrics(step: int, loss: float, tokens_per_step: int,
                         step_duration: float, trained_tokens: int,
                         num_params: int, model_config: PretrainedConfig,
                         world_size: int, config: Dict[str, Any]) -> None:
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
    """
    # Calculate metrics
    tokens_per_second = tokens_per_step / step_duration
    tokens_per_second_per_gpu = tokens_per_second / world_size
    mfu = get_mfu(tokens_per_second_per_gpu, num_params, model_config)

    # Determine if this rank should log
    is_wandb_rank = (pgm.process_group_manager.tp_rank == 0
                     and pgm.process_group_manager.dp_rank == 0
                     and pgm.process_group_manager.cp_rank == 0
                     and pgm.process_group_manager.pp_is_last_stage)

    if is_wandb_rank:
        # Log to console
        max_tokens_str = ('/' +
                          to_readable_format(config['training']['max_tokens'])
                          if config['training']['max_tokens'] else '')

        print(
            f'[rank {pgm.process_group_manager.global_rank}] '
            f'Step: {step:<5d} | '
            f'Loss: {loss:6.4f} | '
            f'Global batch size: {to_readable_format(tokens_per_step):>7s} | '
            f'Tokens/s: {to_readable_format(tokens_per_second):>7s} | '
            f'Tokens/s/GPU: {to_readable_format(tokens_per_second_per_gpu):>7s} | '
            f'Tokens: {to_readable_format(trained_tokens):>7s}{max_tokens_str} | '
            f'MFU: {mfu:5.2f}% | '
            f'Memory usage: {torch.cuda.memory_reserved() / 1e9:6.2f}GB',
            is_print_rank=is_wandb_rank)

        # Log to Weights & Biases if enabled
        if config['logging']['use_wandb']:
            wandb.log({
                'loss': loss,
                'tokens_per_step': tokens_per_step,
                'tokens_per_second': tokens_per_step / step_duration,
                'mfu': mfu,
                'tokens_per_second_per_gpu': tokens_per_second_per_gpu,
                'memory_usage': torch.cuda.memory_reserved() / 1e9,
                'trained_tokens': trained_tokens
            })


def main() -> None:
    """Main training function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train LLaMA model with distributed parallelism')
    parser.add_argument('--config',
                        type=str,
                        default='',
                        help='Path to config file')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Setup environment
    setup_environment(config)

    # Get appropriate data type
    dtype = get_dtype(config)

    # Initialize distributed training
    local_rank, global_rank, world_size, backend, device = initialize_distributed_training(
        config)

    # Validate configuration
    validate_config(config, world_size)

    # Determine if this rank should log to wandb
    is_wandb_rank = (pgm.process_group_manager.tp_rank == 0
                     and pgm.process_group_manager.dp_rank == 0
                     and pgm.process_group_manager.cp_rank == 0
                     and pgm.process_group_manager.pp_is_last_stage)

    # Set random seed for reproducibility
    set_all_seed(config['training']['seed'])

    # Initialize data loader
    start_time = time.time()
    data_loader = MicroBatchDataLoader(
        micro_batch_size=config['training']['micro_batch_size'],
        seq_length=config['training']['seq_length'],
        dataset_name=config['dataset']['name'],
        tokenizer_name=config['model']['name'],
        grad_acc_steps=config['training']['gradient_accumulation_steps'],
        device=device,
        num_workers=config['dataset']['num_workers'],
        num_proc=config['dataset']['num_proc'],
        num_samples=config['training'].get('num_samples', None),
        subset_name=config['dataset'].get('subset_name', None),
        split=config['dataset'].get('split', 'train'))

    dist.barrier()

    print(f'init dataloader time: {time.time()-start_time:.2f}s',
          is_print_rank=is_wandb_rank)

    # Calculate tokens per step
    tokens_per_step = data_loader.global_batch_size * config['training'][
        'seq_length']

    if pgm.process_group_manager.global_rank == 0:
        print('Tokens per step:',
              to_readable_format(tokens_per_step),
              is_print_rank=is_wandb_rank)

    # Initialize Weights & Biases if enabled
    if is_wandb_rank and config['logging']['use_wandb']:
        wandb.init(
            project='scaletorch',
            name=
            f"{config['logging']['run_name']}_{to_readable_format(tokens_per_step)}_{pgm.process_group_manager}",
            config={
                'tensor_parallel_size':
                pgm.process_group_manager.tp_world_size,
                'context_parallel_size':
                pgm.process_group_manager.cp_world_size,
                'pipeline_parallel_size':
                pgm.process_group_manager.pp_world_size,
                'data_parallel_size': pgm.process_group_manager.dp_world_size,
                'model': config['model']['name'],
                'dataset': config['dataset']['name'],
                'max_tokens': config['training']['max_tokens'],
                'learning_rate': config['training']['learning_rate'],
                'seed': config['training']['seed'],
                'micro_batch_size': data_loader.micro_batch_size,
                'global_batch_size': data_loader.global_batch_size,
                'gradient_accumulation': data_loader.grad_acc_steps,
            },
        )

    # Create model configuration
    model_config = create_model_config(config, device)

    dist.barrier()

    # Create model
    model = create_model(model_config, config, dtype, device)

    # Set model to training mode
    model.train()

    # Print model size
    num_params = get_num_params(model)
    print(f'Number of parameters: {to_readable_format(num_params)}',
          is_print_rank=is_wandb_rank)

    # Define tensor shapes for pipeline parallelism
    tensor_shapes = (data_loader.micro_batch_size,
                     data_loader.seq_length_per_gpu, model_config.hidden_size)

    # Create optimizer
    optimizer = create_optimizer(model, config, device)

    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager()

    # Initialize training state
    trained_tokens, step = 0, 0
    if config['checkpoint']['load_path']:
        step, trained_tokens = checkpoint_manager.load_checkpoint(
            model, optimizer, config['checkpoint']['load_path'])

    dist.barrier()

    # Training loop
    while (config['training']['max_tokens'] is None
           or trained_tokens < config['training']['max_tokens']):
        step_start_time = time.time()

        # Zero gradients
        optimizer.zero_grad()

        # Perform training step
        if pgm.process_group_manager.pp_world_size > 1:
            # Pipeline parallel training
            if config['distributed']['pp_engine'] == 'afab':
                loss = train_step_pipeline_afab(model, data_loader,
                                                tensor_shapes, device, dtype)
            elif config['distributed']['pp_engine'] == '1f1b':
                loss = train_step_pipeline_1f1b(model, data_loader,
                                                tensor_shapes, device, dtype)
            else:
                raise ValueError(
                    f"Invalid pipeline parallel engine: {config['distributed']['pp_engine']}"
                )
        else:
            # Standard training step
            loss = train_step(model, data_loader, device)

        # Average loss across data and context parallel ranks
        loss = average_loss_across_dp_cp_ranks(loss, device)

        # Update parameters
        optimizer.step()

        # Update training state
        trained_tokens += tokens_per_step
        step += 1

        # Reset model state if needed
        if hasattr(model, 'reset'):
            model.reset()

        # Calculate step duration
        step_duration = time.time() - step_start_time

        # Log training metrics
        log_training_metrics(step, loss, tokens_per_step, step_duration,
                             trained_tokens, num_params, model_config,
                             world_size, config)

        # Save checkpoint if needed
        if (config['checkpoint']['save_frequency'] > 0
                and step % config['checkpoint']['save_frequency'] == 0):
            checkpoint_manager.save_checkpoint(
                model, optimizer, step, trained_tokens,
                config['checkpoint']['save_dir'] + f'/{step}')

        # Check if we've reached the maximum number of steps
        if step >= config['training']['total_train_steps']:
            break

    # Finish Weights & Biases logging if enabled
    if is_wandb_rank and config['logging']['use_wandb']:
        wandb.finish()

    # Clean up distributed training
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
