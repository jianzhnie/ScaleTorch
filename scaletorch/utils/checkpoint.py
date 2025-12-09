"""
Checkpoint management utilities for ScaleTorch.

This module provides functionality for:
- Model initialization with dematerialized weights (memory-efficient)
- Loading pre-trained weights from safetensors checkpoints
- Distributed checkpoint saving and loading
- Tensor parallelism and pipeline parallelism support
"""

import contextlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from safetensors import safe_open

from scaletorch.model.model_llama import FinalProjection
from scaletorch.parallel.pg_manager import process_group_manager as pgm
from scaletorch.parallel.pipeline_parallel.pipeline_parallel import \
    PipelineParallel
from scaletorch.utils.utils import assert_no_meta_tensors, print


@contextlib.contextmanager
def init_model_with_dematerialized_weights(
        include_buffers: bool = False) -> Iterator[None]:
    """
    Context manager for initializing models with empty weights (no memory allocation).

    Adapted from Accelerate library:
    https://github.com/huggingface/accelerate/blob/v0.11.0/src/accelerate/big_modeling.py#L254

    This is useful for initializing large models without allocating memory,
    allowing for memory-efficient model creation before loading actual weights.

    Args:
        include_buffers: Whether to also skip buffer initialization.

    Yields:
        None

    Example:
        >>> with init_model_with_dematerialized_weights():
        ...     model = MyLargeModel()  # Creates model without allocating memory
    """
    old_register_parameter = nn.Module.register_parameter
    old_register_buffer = nn.Module.register_buffer if include_buffers else None

    def register_empty_parameter(module: nn.Module, name: str,
                                 param: Optional[nn.Parameter]) -> None:
        """Register parameter with meta device to avoid memory allocation."""
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            module._parameters[name] = param_cls(
                module._parameters[name].to(torch.device('meta')), **kwargs)

    def register_empty_buffer(module: nn.Module, name: str,
                              buffer: Optional[torch.Tensor]) -> None:
        """Register buffer with meta device to avoid memory allocation."""
        old_register_buffer(module, name, buffer)
        if buffer is not None:
            module._buffers[name] = module._buffers[name].to(
                torch.device('meta'))

    try:
        nn.Module.register_parameter = register_empty_parameter
        if include_buffers:
            nn.Module.register_buffer = register_empty_buffer
        yield
    finally:
        # Restore original registration methods
        nn.Module.register_parameter = old_register_parameter
        if include_buffers:
            nn.Module.register_buffer = old_register_buffer


def init_model_with_materialized_weights(
        model: nn.Module, model_config: Any,
        save_dir: Union[str, Path]) -> nn.Module:
    """
    Initialize model with correct tensor shapes and load pre-trained weights from checkpoint.

    This function:
    1. Determines which layers this rank is responsible for
    2. Loads weights from safetensors checkpoint (sharded or single file)
    3. Adjusts tensor sizes for tensor parallelism
    4. Creates final projection layer if needed
    5. Loads weights into the model

    Args:
        model: The model to initialize
        model_config: Model configuration object
        save_dir: Directory containing the checkpoint files

    Returns:
        The initialized model with loaded weights

    Raises:
        Exception: If this rank has no layers to process
        FileNotFoundError: If checkpoint files are not found
        RuntimeError: If there are issues loading or processing weights
    """
    save_dir = Path(save_dir)
    initialization_manager = InitializationManager(model, model_config)
    layer_names = initialization_manager.get_layer_names_in_sft_format()

    # Validate that this rank has work to do
    if not layer_names:
        raise Exception(
            'This rank has no layers to process. There are too many ranks '
            'and not enough layers to distribute.')

    print(f'Rank {pgm.global_rank}: Processing {len(layer_names)} layers')

    state_dict: Dict[str, torch.Tensor] = {}

    # Determine checkpoint format and load accordingly
    index_path = save_dir / 'model.safetensors.index.json'

    try:
        if index_path.exists():
            # Handle sharded checkpoint
            state_dict = _load_sharded_checkpoint(save_dir, index_path,
                                                  layer_names,
                                                  initialization_manager)
        else:
            # Handle single file checkpoint
            safetensors_path = save_dir / 'model.safetensors'
            if not safetensors_path.exists():
                raise FileNotFoundError(
                    f'No checkpoint found in {save_dir}. Expected either '
                    f"'model.safetensors.index.json' or 'model.safetensors'")

            state_dict = _load_single_checkpoint(safetensors_path, layer_names,
                                                 initialization_manager)

        # Handle final projection layer creation and initialization
        _handle_final_projection(model, model_config, state_dict)

        # Synchronize across distributed processes and load weights
        # Use non-blocking barrier if possible for better performance
        if dist.is_initialized():
            dist.barrier()
        model.load_state_dict(state_dict, strict=True, assign=True)
        if dist.is_initialized():
            dist.barrier()

        # Verify no meta tensors remain and initialize parameters
        assert_no_meta_tensors(model)
        initialization_manager.init_model_parameters()
        if dist.is_initialized():
            dist.barrier()

        return model

    except Exception as e:
        raise RuntimeError(
            f'Failed to initialize model with materialized weights: {e}')


def _load_sharded_checkpoint(
    save_dir: Path, index_path: Path, layer_names: List[str],
    initialization_manager: 'InitializationManager'
) -> Dict[str, torch.Tensor]:
    """Load weights from a sharded safetensors checkpoint."""
    state_dict: Dict[str, torch.Tensor] = {}

    try:
        with open(index_path, 'r', encoding='utf-8') as f:
            index = json.load(f)

        weight_map = index.get('weight_map', {})

        for sft_name in layer_names:
            if sft_name not in weight_map:
                print(
                    f"Warning: Layer '{sft_name}' not found in checkpoint index"
                )
                continue

            shard_path = save_dir / weight_map[sft_name]
            if not shard_path.exists():
                raise FileNotFoundError(
                    f"Shard file '{shard_path}' not found for layer '{sft_name}'"
                )

            with safe_open(shard_path, framework='pytorch', device='cpu') as f:
                hf_name = initialization_manager.convert_safetensors_to_hf_name(
                    sft_name)
                tensor = f.get_tensor(sft_name)
                tensor = initialization_manager.adjust_tensor_size(
                    tensor, hf_name)
                state_dict[hf_name] = tensor

    except json.JSONDecodeError as e:
        raise RuntimeError(f'Failed to parse checkpoint index file: {e}')
    except Exception as e:
        raise RuntimeError(f'Error loading sharded checkpoint: {e}')

    return state_dict


def _load_single_checkpoint(
    safetensors_path: Path, layer_names: List[str],
    initialization_manager: 'InitializationManager'
) -> Dict[str, torch.Tensor]:
    """Load weights from a single safetensors checkpoint file."""
    state_dict: Dict[str, torch.Tensor] = {}

    try:
        with safe_open(safetensors_path, framework='pytorch',
                       device='cpu') as f:
            checkpoint_keys = set(f.keys())
            layer_names_set = set(layer_names)

            # Warn about mismatched layer counts
            if len(checkpoint_keys) > len(layer_names):
                print(
                    f'Warning: Checkpoint has {len(checkpoint_keys)} layers but '
                    f'model only has {len(layer_names)} layers')

            # Check for missing layers
            missing_layers = layer_names_set - checkpoint_keys
            if missing_layers:
                print(
                    f'Warning: Missing layers in checkpoint: {missing_layers}')

            for sft_name in layer_names:
                if sft_name not in checkpoint_keys:
                    continue

                hf_name = initialization_manager.convert_safetensors_to_hf_name(
                    sft_name)
                tensor = f.get_tensor(sft_name)
                tensor = initialization_manager.adjust_tensor_size(
                    tensor, hf_name)
                state_dict[hf_name] = tensor

    except Exception as e:
        raise RuntimeError(f'Error loading single checkpoint file: {e}')

    return state_dict


def _handle_final_projection(model: nn.Module, model_config: Any,
                             state_dict: Dict[str, torch.Tensor]) -> None:
    """Handle final projection layer creation and weight initialization."""
    if not (pgm.pp_is_last_stage or not isinstance(model, PipelineParallel)):
        return

    vocab_size = model_config.vocab_size

    if pgm.tp_world_size > 1:
        # For TP>1, the final_proj is already wrapped in ColumnParallel
        # Just need to initialize state_dict with correct sharded size
        vocab_per_rank = vocab_size // pgm.tp_world_size
        state_dict['final_proj.weight'] = torch.zeros(vocab_per_rank,
                                                      model_config.hidden_size)
    else:
        # For TP=1, create the full layer
        model.final_proj = FinalProjection(model_config.hidden_size,
                                           vocab_size,
                                           bias=False)
        state_dict['final_proj.weight'] = torch.zeros(vocab_size,
                                                      model_config.hidden_size)


class InitializationManager:
    """Manages model initialization and weight processing for distributed training."""

    def __init__(self, model: nn.Module, model_config: Any) -> None:
        """
        Initialize the initialization manager.

        Args:
            model: The model to manage
            model_config: Model configuration object
        """
        self.model = model
        self.model_config = model_config

    def init_model_parameters(self) -> None:
        """Initialize model parameters using the model's reset_parameters method."""
        self.model.reset_parameters()

    def get_layer_names_in_sft_format(self) -> List[str]:
        """
        Get layer names in safetensors format based on model's layer distribution.

        This method determines which layers this specific rank is responsible for
        based on pipeline parallelism configuration.

        Returns:
            List of layer names in safetensors format
        """
        decoder_components = [
            'input_layernorm',
            'mlp.down_proj',
            'mlp.gate_proj',
            'mlp.up_proj',
            'post_attention_layernorm',
            'self_attn.k_proj',
            'self_attn.o_proj',
            'self_attn.q_proj',
            'self_attn.v_proj',
        ]

        # Generate base layer names based on pipeline parallelism
        layer_names: List[str] = []

        if isinstance(self.model, PipelineParallel):
            base_names = [
                f'model.layers.{layer_id}'
                for layer_id in self.model.layer_distribution
            ]
        else:
            base_names = [
                f'model.layers.{layer_id}'
                for layer_id in range(self.model_config.num_hidden_layers)
            ]

        # Add decoder components for each layer
        for layer in base_names:
            for component in decoder_components:
                layer_names.append(f'{layer}.{component}.weight')

        # Add special layers based on pipeline stage
        # NOTE: Safetensors may have tied embeddings, but Picotron does not support it
        if isinstance(self.model, PipelineParallel):
            if pgm.pp_is_first_stage:
                layer_names.insert(0, 'model.embed_tokens.weight')
            elif pgm.pp_is_last_stage:
                layer_names.append('model.norm.weight')
        else:
            layer_names.insert(0, 'model.embed_tokens.weight')
            layer_names.append('model.norm.weight')

        return layer_names

    def adjust_tensor_size(self, tensor: torch.Tensor,
                           name: str) -> torch.Tensor:
        """
        Resize tensor based on tensor parallelism configuration.

        This method adjusts tensor dimensions to match the current tensor parallelism
        setup, ensuring proper sharding across ranks.

        Args:
            tensor: The tensor to adjust
            name: The layer name (used to determine adjustment strategy)

        Returns:
            The adjusted tensor
        """
        tp_rank = pgm.tp_rank
        tp_size = pgm.tp_world_size
        hidden_size = self.model_config.hidden_size

        # Handle embedding and final projection layers
        if 'embedding.weight' in name or 'final_proj.weight' in name:
            vocab_size = self.model_config.vocab_size
            vocab_per_rank = vocab_size // tp_size

            if tensor.shape[0] != vocab_per_rank:
                start_idx = tp_rank * vocab_per_rank
                end_idx = start_idx + vocab_per_rank
                tensor = tensor[start_idx:end_idx, :]
            return tensor

        # Handle attention layers
        if 'attention' in name:
            head_dim = hidden_size // self.model_config.num_attention_heads

            if 'q_proj.weight' in name:
                total_heads = self.model_config.num_attention_heads
                heads_per_rank = total_heads // tp_size
                target_dim = heads_per_rank * head_dim
            elif 'k_proj.weight' in name or 'v_proj.weight' in name:
                total_heads = self.model_config.num_key_value_heads
                heads_per_rank = total_heads // tp_size
                target_dim = heads_per_rank * head_dim
            elif 'out_proj.weight' in name:
                # For out_proj, split along the second dimension
                if tensor.shape[1] != hidden_size // tp_size:
                    start_idx = (hidden_size // tp_size) * tp_rank
                    end_idx = (hidden_size // tp_size) * (tp_rank + 1)
                    tensor = tensor[:, start_idx:end_idx]
                return tensor
            else:
                return tensor

            # Adjust tensor dimension if needed
            if tensor.shape[0] != target_dim:
                if target_dim > tensor.shape[0]:
                    # Pad tensor if target is larger
                    pad_tensor = torch.empty(target_dim - tensor.shape[0],
                                             tensor.shape[1],
                                             dtype=tensor.dtype,
                                             device=tensor.device)
                    tensor = torch.cat([tensor, pad_tensor], dim=0)
                else:
                    # Truncate tensor if target is smaller
                    tensor = tensor[:target_dim, :]

        # Handle MLP layers
        elif 'mlp' in name:
            intermediate_size = self.model_config.intermediate_size
            intermediate_size_per_rank = intermediate_size // tp_size

            if 'up_proj.weight' in name or 'gate_proj.weight' in name:
                if tensor.shape[0] != intermediate_size_per_rank:
                    start_idx = tp_rank * intermediate_size_per_rank
                    end_idx = start_idx + intermediate_size_per_rank
                    tensor = tensor[start_idx:end_idx, :]
            elif 'down_proj.weight' in name:
                if tensor.shape[1] != intermediate_size_per_rank:
                    start_idx = tp_rank * intermediate_size_per_rank
                    end_idx = start_idx + intermediate_size_per_rank
                    tensor = tensor[:, start_idx:end_idx]

        return tensor

    def convert_safetensors_to_hf_name(self, sft_name: str) -> str:
        """
        Convert safetensors naming convention to HuggingFace naming convention.

        Args:
            sft_name: Layer name in safetensors format

        Returns:
            Layer name in HuggingFace format
        """
        name_mapping = {
            'model.': '',
            'layers.': 'decoder_layers.',
            'embed_tokens': 'embedding',
            'self_attn.': 'attention.',
            'o_proj': 'out_proj',
            'lm_head': 'final_proj',
            'input_layernorm': 'input_layernorm',
            'post_attention_layernorm': 'post_attention_layernorm',
            r'^norm': 'final_norm'
        }

        result = sft_name
        for pattern, replacement in name_mapping.items():
            result = re.sub(pattern, replacement, result)

        return result


class CheckpointManager:
    """Manages checkpoint saving and loading for distributed training."""

    def __init__(self) -> None:
        """Initialize checkpoint manager with process group information."""
        self.tp_rank = pgm.tp_rank
        self.pp_rank = pgm.pp_rank
        self.tp_world_size = pgm.tp_world_size
        self.pp_world_size = pgm.pp_world_size
        self.cp_dp_world_size = pgm.cp_dp_world_size
        self.dp_rank = pgm.dp_rank
        self.cp_rank = pgm.cp_rank

    def _get_checkpoint_path(self, out_dir: Union[str, Path]) -> Path:
        """
        Generate checkpoint file path based on tensor and pipeline parallelism configuration.

        Args:
            out_dir: Output directory for checkpoints

        Returns:
            Path object for the checkpoint file
        """
        out_dir = Path(out_dir)
        ckpt_name = (
            f'weights_tp_rank_world_size={self.tp_rank}_{self.tp_world_size}_'
            f'pp_rank_world_size={self.pp_rank}_{self.pp_world_size}.pth')
        return out_dir / ckpt_name

    def save_checkpoint(self, model: nn.Module,
                        optimizer: torch.optim.Optimizer, trained_steps: int,
                        trained_tokens: int, out_dir: Union[str,
                                                            Path]) -> None:
        """
        Save model and optimizer states to a checkpoint file.

        Only DP/CP rank 0 saves the model since weights are identical across all ranks.

        Args:
            model: The model to save
            optimizer: The optimizer to save
            trained_steps: Number of training steps completed
            trained_tokens: Number of tokens processed
            out_dir: Directory to save the checkpoint
        """
        out_dir = Path(out_dir)
        path = self._get_checkpoint_path(out_dir)

        # Only save from DP/CP rank 0
        if self.dp_rank == 0 and self.cp_rank == 0:
            try:
                out_dir.mkdir(parents=True, exist_ok=True)

                # Handle DataParallel wrapper if present
                raw_model = model.module if self.cp_dp_world_size > 1 else model

                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'trained_steps': trained_steps,
                    'trained_tokens': trained_tokens
                }

                # Use efficient checkpoint saving with proper device placement
                # Move model state dict to CPU before saving to reduce GPU memory pressure
                if torch.cuda.is_available():
                    cpu_model_state = {}
                    for key, value in checkpoint['model'].items():
                        if isinstance(value, torch.Tensor):
                            cpu_model_state[key] = value.cpu()
                        else:
                            cpu_model_state[key] = value
                    checkpoint['model'] = cpu_model_state

                # Use new zipfile serialization for better compression and speed
                torch.save(checkpoint,
                           path,
                           _use_new_zipfile_serialization=True)
                print(f'Checkpoint saved to {path}')

            except Exception as e:
                raise RuntimeError(f'Failed to save checkpoint: {e}')

    def load_checkpoint(self, model: nn.Module,
                        optimizer: torch.optim.Optimizer,
                        out_dir: Union[str, Path]) -> tuple[int, int]:
        """
        Load model and optimizer states from checkpoint.

        Args:
            model: The model to load weights into
            optimizer: The optimizer to load state into
            out_dir: Directory containing the checkpoint

        Returns:
            Tuple of (trained_steps, trained_tokens)

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            RuntimeError: If there are issues loading the checkpoint
        """
        path = self._get_checkpoint_path(out_dir)

        if not path.exists():
            raise FileNotFoundError(f'Checkpoint not found at {path}')

        try:
            # Load checkpoint efficiently with proper device mapping
            checkpoint = torch.load(path,
                                    map_location='cpu',
                                    weights_only=False)

            # Validate checkpoint contents
            required_keys = {
                'model', 'optimizer', 'trained_steps', 'trained_tokens'
            }
            if not all(key in checkpoint for key in required_keys):
                raise RuntimeError(
                    f'Invalid checkpoint format. Expected keys: {required_keys}, '
                    f'got: {checkpoint.keys()}')

            # Handle DataParallel wrapper if present
            raw_model = model.module if self.cp_dp_world_size > 1 else model

            # Load model weights
            raw_model.load_state_dict(checkpoint['model'])

            # Load optimizer state
            optimizer.load_state_dict(checkpoint['optimizer'])

            trained_steps = checkpoint['trained_steps']
            trained_tokens = checkpoint['trained_tokens']

            print(f'Checkpoint loaded from {path} '
                  f'(steps: {trained_steps}, tokens: {trained_tokens})')

            return trained_steps, trained_tokens

        except Exception as e:
            raise RuntimeError(f'Failed to load checkpoint: {e}')
