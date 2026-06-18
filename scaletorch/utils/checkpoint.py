"""Checkpoint management: save/load, weight materialization, and safetensors loading."""

from __future__ import annotations

import contextlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import torch
import torch.nn as nn
from safetensors import safe_open

import scaletorch.dist as st_dist
from scaletorch.models.llama import FinalProjection
from scaletorch.parallel.pipeline_parallel.pipeline_parallel import \
    PipelineParallel
from scaletorch.parallel.process_group import process_group_manager as pgm
from scaletorch.utils.misc import assert_no_meta_tensors, rank_print


@contextlib.contextmanager
def init_model_with_dematerialized_weights(
        include_buffers: bool = False) -> Iterator[None]:
    """Context manager that initializes model parameters on the meta device (no memory allocation).

    Adapted from HuggingFace Accelerate.
    """
    old_register_parameter = nn.Module.register_parameter
    old_register_buffer = nn.Module.register_buffer if include_buffers else None

    def register_empty_parameter(module: nn.Module, name: str,
                                 param: Optional[nn.Parameter]) -> None:
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            module._parameters[name] = param_cls(
                module._parameters[name].to(torch.device('meta')), **kwargs)

    def register_empty_buffer(module: nn.Module, name: str,
                              buffer: Optional[torch.Tensor]) -> None:
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
    """Materialize model weights: determine layers for this rank, load from safetensors, adjust for TP."""
    save_dir = Path(save_dir)
    initialization_manager = InitializationManager(model, model_config)
    layer_names = initialization_manager.get_layer_names_in_sft_format()

    # Validate that this rank has work to do
    if not layer_names:
        raise RuntimeError(
            'This rank has no layers to process. There are too many ranks '
            'and not enough layers to distribute.')

    rank = pgm.global_rank if pgm else 0
    rank_print(f'Rank {rank}: Processing {len(layer_names)} weight tensors')

    state_dict: Dict[str, torch.Tensor] = {}

    # Determine checkpoint format and load accordingly
    index_path = save_dir / 'model.safetensors.index.json'
    safetensors_path = save_dir / 'model.safetensors'

    has_checkpoint = index_path.exists() or safetensors_path.exists()

    if not has_checkpoint:
        rank_print(f'Rank {rank}: No checkpoint in {save_dir}. '
              'Materializing with random init.')
        model.to_empty(device='cpu')
        model.reset_parameters()
        assert_no_meta_tensors(model)
        return model

    if index_path.exists():
        state_dict = _load_sharded_checkpoint(save_dir, index_path,
                                              layer_names,
                                              initialization_manager)
    else:
        state_dict = _load_single_checkpoint(safetensors_path, layer_names,
                                             initialization_manager)

    # Handle final projection layer creation and initialization
    _handle_final_projection(model, model_config, state_dict)

    # Synchronize across distributed processes and load weights
    if st_dist.is_distributed():
        st_dist.barrier()
    model.load_state_dict(state_dict, strict=True, assign=True)
    if st_dist.is_distributed():
        st_dist.barrier()

    # Verify no meta tensors remain and initialize parameters
    assert_no_meta_tensors(model)
    initialization_manager.init_model_parameters()
    if st_dist.is_distributed():
        st_dist.barrier()

    return model


def _load_sharded_checkpoint(
    save_dir: Path, index_path: Path, layer_names: List[str],
    initialization_manager: 'InitializationManager'
) -> Dict[str, torch.Tensor]:
    """Load weights from a sharded safetensors checkpoint."""
    state_dict: Dict[str, torch.Tensor] = {}

    with open(index_path, 'r', encoding='utf-8') as f:
        index = json.load(f)

    weight_map = index.get('weight_map', {})

    for sft_name in layer_names:
        if sft_name not in weight_map:
            rank_print(f"Warning: Layer '{sft_name}' not found in checkpoint index")
            continue

        shard_path = save_dir / weight_map[sft_name]
        if not shard_path.exists():
            raise FileNotFoundError(
                f"Shard file '{shard_path}' not found for layer '{sft_name}'")

        with safe_open(shard_path, framework='pytorch', device='cpu') as f:
            hf_name = initialization_manager.convert_safetensors_to_hf_name(
                sft_name)
            tensor = f.get_tensor(sft_name)
            tensor = initialization_manager.adjust_tensor_size(tensor, hf_name)
            state_dict[hf_name] = tensor

    return state_dict


def _load_single_checkpoint(
    safetensors_path: Path, layer_names: List[str],
    initialization_manager: 'InitializationManager'
) -> Dict[str, torch.Tensor]:
    """Load weights from a single safetensors checkpoint file."""
    state_dict: Dict[str, torch.Tensor] = {}

    with safe_open(safetensors_path, framework='pytorch', device='cpu') as f:
        checkpoint_keys = set(f.keys())

        if len(checkpoint_keys) > len(layer_names):
            rank_print(f'Warning: Checkpoint has {len(checkpoint_keys)} layers but '
                  f'model only has {len(layer_names)} layers')

        missing_layers = set(layer_names) - checkpoint_keys
        if missing_layers:
            rank_print(f'Warning: Missing layers in checkpoint: {missing_layers}')

        for sft_name in layer_names:
            if sft_name not in checkpoint_keys:
                continue

            hf_name = initialization_manager.convert_safetensors_to_hf_name(
                sft_name)
            tensor = f.get_tensor(sft_name)
            tensor = initialization_manager.adjust_tensor_size(tensor, hf_name)
            state_dict[hf_name] = tensor

    return state_dict


def _handle_final_projection(model: nn.Module, model_config: Any,
                             state_dict: Dict[str, torch.Tensor]) -> None:
    """Handle final projection layer weight in state_dict."""
    if not (getattr(pgm, 'pp_is_last_stage', True)
            or not isinstance(model, PipelineParallel)):
        return

    if 'final_proj.weight' in state_dict:
        return

    tie_word_embeddings = getattr(model_config, 'tie_word_embeddings', False)
    if tie_word_embeddings and 'embedding.weight' in state_dict:
        state_dict['final_proj.weight'] = state_dict['embedding.weight']
        return

    vocab_size = model_config.vocab_size
    tp_world_size = getattr(pgm, 'tp_world_size', 1)
    if tp_world_size > 1:
        vocab_per_rank = vocab_size // tp_world_size
        state_dict['final_proj.weight'] = torch.zeros(
            vocab_per_rank, model_config.hidden_size)
    else:
        state_dict['final_proj.weight'] = torch.zeros(
            vocab_size, model_config.hidden_size)


class InitializationManager:
    """Manages model initialization and weight processing for distributed training."""

    def __init__(self, model: nn.Module, model_config: Any) -> None:
        self.model = model
        self.model_config = model_config

    def init_model_parameters(self) -> None:
        """Initialize model parameters using the model's reset_parameters method."""
        self.model.reset_parameters()

    def get_layer_names_in_sft_format(self) -> List[str]:
        """Get layer names in safetensors format for this rank's pipeline stage."""
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

        model_type = getattr(self.model_config, 'model_type', 'llama')
        if model_type in ('qwen3', 'qwen2', 'qwen3_moe'):
            decoder_components.extend([
                'self_attn.q_norm',
                'self_attn.k_norm',
            ])

        # For MoE models: replace mlp.{gate,up,down}_proj with expert components
        if model_type == 'qwen3_moe':
            decoder_components = [c for c in decoder_components
                                  if not c.startswith('mlp.')]
            num_experts = getattr(self.model_config, 'num_experts', 128)
            ep_size = getattr(pgm, 'ep_world_size', 1)
            ep_rank = getattr(pgm, 'ep_rank', 0)
            experts_per_rank = num_experts // ep_size
            start_expert = ep_rank * experts_per_rank
            end_expert = start_expert + experts_per_rank
            for eid in range(start_expert, end_expert):
                for proj in ('gate_proj', 'up_proj', 'down_proj'):
                    decoder_components.append(
                        f'mlp.experts.{eid}.{proj}')
            decoder_components.append('mlp.gate')

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
        tie_word_embeddings = getattr(self.model_config, 'tie_word_embeddings', False)
        if isinstance(self.model, PipelineParallel):
            if getattr(pgm, 'pp_is_first_stage', True):
                layer_names.insert(0, 'model.embed_tokens.weight')
            elif getattr(pgm, 'pp_is_last_stage', True):
                layer_names.append('model.norm.weight')
                if not tie_word_embeddings:
                    layer_names.append('lm_head.weight')
        else:
            layer_names.insert(0, 'model.embed_tokens.weight')
            layer_names.append('model.norm.weight')
            if not tie_word_embeddings:
                layer_names.append('lm_head.weight')

        return layer_names

    def adjust_tensor_size(self, tensor: torch.Tensor,
                           name: str) -> torch.Tensor:
        """Shard or truncate tensor dimensions for the current tensor parallelism rank."""
        tp_rank = pgm.tp_rank if pgm else 0
        tp_size = getattr(pgm, 'tp_world_size', 1)
        hidden_size = self.model_config.hidden_size

        # MoE expert and router weights: EP-sharded (already handled by name
        # mapping), NOT TP-sharded — return as-is.
        if 'moe.experts.' in name or 'moe.router.' in name:
            return tensor
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
            head_dim = getattr(self.model_config, 'head_dim',
                               hidden_size // self.model_config.num_attention_heads)

            if 'q_norm' in name or 'k_norm' in name:
                return tensor

            if 'q_proj.weight' in name:
                total_heads = self.model_config.num_attention_heads
                heads_per_rank = total_heads // tp_size
                target_dim = heads_per_rank * head_dim
            elif 'k_proj.weight' in name or 'v_proj.weight' in name:
                total_heads = self.model_config.num_key_value_heads
                heads_per_rank = total_heads // tp_size
                target_dim = heads_per_rank * head_dim
            elif 'out_proj.weight' in name:
                total_qkv_dim = self.model_config.num_attention_heads * head_dim
                dim_per_rank = total_qkv_dim // tp_size
                if tensor.shape[1] != dim_per_rank:
                    start_idx = dim_per_rank * tp_rank
                    end_idx = dim_per_rank * (tp_rank + 1)
                    tensor = tensor[:, start_idx:end_idx]
                return tensor
            else:
                return tensor

            # Adjust tensor dimension if needed
            if tensor.shape[0] != target_dim:
                if target_dim > tensor.shape[0]:
                    # Pad tensor if target is larger
                    pad_tensor = torch.zeros(target_dim - tensor.shape[0],
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
        """Convert safetensors naming convention to model parameter names."""
        result = sft_name

        # Strip top-level 'model.' prefix
        result = re.sub(r'^model\.', '', result)
        # Layer indexing
        result = result.replace('layers.', 'decoder_layers.')
        # Embedding
        result = result.replace('embed_tokens', 'embedding')
        # Final layers
        result = result.replace('lm_head', 'final_proj')
        result = re.sub(r'^norm', 'final_norm', result)

        # MoE expert naming: mlp.experts.N.proj → moe.experts.experts.LOCAL.proj
        # MoE router: mlp.gate → moe.router.gate
        if 'mlp.experts.' in result:
            ep_size = getattr(pgm, 'ep_world_size', 1)
            ep_rank = getattr(pgm, 'ep_rank', 0)
            num_experts = getattr(self.model_config, 'num_experts', 128)
            experts_per_rank = num_experts // ep_size

            m = re.search(r'mlp\.experts\.(\d+)\.', result)
            if m:
                global_id = int(m.group(1))
                local_id = global_id - ep_rank * experts_per_rank
                result = result.replace(f'mlp.experts.{global_id}.',
                                        f'moe.experts.experts.{local_id}.')
        elif 'mlp.gate.' in result:
            result = result.replace('mlp.gate.', 'moe.router.gate.')
        else:
            # Dense MLP (non-MoE models) — no change needed for mlp.*
            pass

        # Attention
        result = result.replace('self_attn.', 'attention.')
        result = result.replace('.o_proj', '.out_proj')

        return result


class CheckpointManager:
    """Manages checkpoint saving and loading for distributed training."""

    def __init__(self) -> None:
        """Initialize checkpoint manager with process group information."""
        self.tp_rank = pgm.tp_rank if pgm else 0
        self.pp_rank = pgm.pp_rank if pgm else 0
        self.tp_world_size = getattr(pgm, 'tp_world_size', 1)
        self.pp_world_size = getattr(pgm, 'pp_world_size', 1)
        self.cp_dp_world_size = getattr(pgm, 'cp_dp_world_size', 1)
        self.dp_rank = pgm.dp_rank if pgm else 0
        self.cp_rank = pgm.cp_rank if pgm else 0

    def _get_checkpoint_path(self, out_dir: Union[str, Path]) -> Path:
        """Generate checkpoint file path based on TP and PP configuration."""
        out_dir = Path(out_dir)
        ckpt_name = (
            f'weights_tp_rank_world_size={self.tp_rank}_{self.tp_world_size}_'
            f'pp_rank_world_size={self.pp_rank}_{self.pp_world_size}.pth')
        return out_dir / ckpt_name

    def save_checkpoint(self, model: nn.Module,
                        optimizer: torch.optim.Optimizer, trained_steps: int,
                        trained_tokens: int, out_dir: Union[str,
                                                            Path]) -> None:
        """Save model and optimizer state. Only DP/CP rank 0 writes to disk."""
        out_dir = Path(out_dir)
        path = self._get_checkpoint_path(out_dir)

        # Only save from DP/CP rank 0
        if self.dp_rank == 0 and self.cp_rank == 0:
            try:
                out_dir.mkdir(parents=True, exist_ok=True)

                raw_model = model.module if self.cp_dp_world_size > 1 else model

                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'trained_steps': trained_steps,
                    'trained_tokens': trained_tokens
                }

                cpu_model_state = {}
                for key, value in checkpoint['model'].items():
                    if isinstance(value, torch.Tensor) and not value.is_cpu:
                        cpu_model_state[key] = value.cpu()
                    else:
                        cpu_model_state[key] = value
                checkpoint['model'] = cpu_model_state

                torch.save(checkpoint,
                           path,
                           _use_new_zipfile_serialization=True)
                rank_print(f'Checkpoint saved to {path}')

            except Exception as e:
                raise RuntimeError(f'Failed to save checkpoint: {e}')

    def load_checkpoint(self, model: nn.Module,
                        optimizer: torch.optim.Optimizer,
                        out_dir: Union[str, Path]) -> tuple[int, int]:
        """Load model and optimizer states from checkpoint. Returns (trained_steps, trained_tokens)."""
        path = self._get_checkpoint_path(out_dir)

        if not path.exists():
            raise FileNotFoundError(f'Checkpoint not found at {path}')

        checkpoint = torch.load(path, map_location='cpu', weights_only=False)

        # Validate checkpoint contents
        required_keys = {
            'model', 'optimizer', 'trained_steps', 'trained_tokens'
        }
        if not all(key in checkpoint for key in required_keys):
            raise RuntimeError(
                f'Invalid checkpoint format. Expected keys: {required_keys}, '
                f'got: {checkpoint.keys()}')

        raw_model = model.module if self.cp_dp_world_size > 1 else model
        raw_model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        trained_steps = checkpoint['trained_steps']
        trained_tokens = checkpoint['trained_tokens']

        rank_print(f'Checkpoint loaded from {path} '
              f'(steps: {trained_steps}, tokens: {trained_tokens})')

        return trained_steps, trained_tokens
