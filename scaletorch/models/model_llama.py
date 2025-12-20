"""
Llama model implementation with support for various attention mechanisms and optimizations.

This module implements the Llama transformer architecture with support for:
- Flash Attention for efficient attention computation
- Context parallelism for long sequences
- Tensor parallelism for distributed training
- Multiple RMSNorm implementations (Triton and standard)
- Rotary Position Embedding (RoPE)
"""

import inspect
import math
import os
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential
from transformers.utils import (is_flash_attn_2_available,
                                is_flash_attn_greater_or_equal_2_10,
                                is_torch_npu_available)

from scaletorch.parallel.context_parallel import context_parallel
from scaletorch.parallel.pg_manager import process_group_manager as pgm

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func

    _flash_supports_window_size = 'window_size' in inspect.signature(
        flash_attn_func).parameters
    _flash_supports_deterministic = 'deterministic' in inspect.signature(
        flash_attn_func).parameters
    _flash_use_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

if is_torch_npu_available:
    from transformers.integrations.npu_flash_attention import \
        npu_flash_attn_func as flash_attn_func
    from transformers.integrations.npu_flash_attention import \
        npu_flash_attn_varlen_func as flash_attn_varlen_func
    from transformers.modeling_flash_attention_utils import \
        flash_attn_supports_top_left_mask

    _flash_supports_window_size = 'window_size' in inspect.signature(
        flash_attn_func).parameters
    _flash_supports_deterministic = 'deterministic' in inspect.signature(
        flash_attn_func).parameters
    _flash_use_top_left_mask = flash_attn_supports_top_left_mask()


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor,
                         sin: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary position embedding to input tensor.

    Args:
        x: Input tensor of shape [batch_size, num_heads, sequence_length, head_dim]
        cos: Cosine values for rotation
        sin: Sine values for rotation

    Returns:
        Rotated tensor with same shape as input
    """
    # TODO: Consider implementing RotaryEmbedding as a class for better modularity
    batch_size, num_head, sequence_length, head_dim = x.size()

    # Split tensor into two halves for rotation
    x1 = x[..., :head_dim // 2]
    x2 = x[..., head_dim // 2:]

    # Create rotated version by negating and swapping halves
    rotate_half = torch.cat([-x2, x1], dim=-1)

    # Apply rotation using cos and sin
    return x * cos + rotate_half * sin


def get_cos_sin(sequence_length: int,
                head_dim: int,
                base: float = 500000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate cosine and sine values for rotary position embedding.

    Args:
        sequence_length: Length of the sequence
        head_dim: Dimension of each attention head
        base: Base for frequency calculation (default: 500000.0)

    Returns:
        Tuple of (cos, sin) tensors for position embedding

    Raises:
        AssertionError: If head_dim is not even
    """
    assert head_dim % 2 == 0, f'head_dim must be even, got {head_dim}'

    # Compute frequencies on CPU to match transformers implementation
    theta = 1.0 / (base**(torch.arange(
        0, head_dim, 2, dtype=torch.int64).float().to('cpu') / head_dim))

    # Determine data type based on environment
    dtype = torch.bfloat16 if os.getenv(
        'DTYPE', 'bfloat16') == 'bfloat16' else torch.float32

    # Determine device based on environment
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device('cuda', local_rank) if os.getenv(
        'DEVICE', 'cuda') == 'cuda' else torch.device('cpu')

    # Create position tensor
    position = torch.arange(sequence_length).to(device).unsqueeze(
        1).float()  # [sequence_length, 1]

    # Move theta to device for computation
    theta = theta.to(device)

    # Compute cos and sin values, repeat for full head dimension
    cos = torch.cos(position.float() * theta.float()).to(dtype).repeat(1, 2)
    sin = torch.sin(position.float() * theta.float()).to(dtype).repeat(1, 2)

    return cos, sin  # [sequence_length, head_dim], [sequence_length, head_dim]


def flash_attention(q: torch.Tensor,
                    k: torch.Tensor,
                    v: torch.Tensor,
                    causal: bool = True) -> torch.Tensor:
    """
    Apply flash attention to query, key, value tensors.

    Args:
        q: Query tensor of shape [batch_size, num_heads, sequence_length, head_dim]
        k: Key tensor of shape [batch_size, num_heads, sequence_length, head_dim]
        v: Value tensor of shape [batch_size, num_heads, sequence_length, head_dim]
        causal: Whether to use causal masking

    Returns:
        Attention output tensor
    """
    # Rearrange dimensions for flash attention
    # [batch_size, sequence_length, num_head, head_dim]
    q = q.permute(0, 2, 1, 3)
    # [batch_size, sequence_length, num_head, head_dim]
    k = k.permute(0, 2, 1, 3)
    # [batch_size, sequence_length, num_head, head_dim]
    v = v.permute(0, 2, 1, 3)

    # Use PyTorch native scaled dot product attention
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


class TritonRMSNorm(nn.Module):
    """Triton-optimized RMSNorm implementation."""

    def __init__(self,
                 hidden_size: int,
                 eps: float = 1e-5,
                 device: Optional[Union[torch.device, str]] = None,
                 dtype: Optional[torch.dtype] = None):
        """
        Initialize TritonRMSNorm.

        Args:
            hidden_size: Size of the hidden dimension
            eps: Epsilon for numerical stability
            device: Device to place parameters on
            dtype: Data type for parameters
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(
            torch.empty(hidden_size, device=device, dtype=dtype))
        self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weight parameters to ones."""
        nn.init.ones_(self.weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        prenorm: bool = False,
        residual_in_fp32: bool = False,
        return_dropout_mask: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Apply Triton-optimized RMS normalization.

        Args:
            hidden_states: Input tensor to normalize
            residual: Optional residual connection
            dropout_p: Dropout probability
            prenorm: Whether this is a pre-normalization
            residual_in_fp32: Whether residual should be in fp32
            return_dropout_mask: Whether to return dropout mask

        Returns:
            Normalized tensor, optionally with residual and dropout mask
        """
        # Compute RMSNorm using PyTorch native operations
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        normed = self.weight * hidden_states.to(input_dtype)

        # Handle residual connection if provided
        if residual is not None:
            normed = normed + residual

        return normed


class LlamaRMSNorm(nn.Module):
    """Standard RMSNorm implementation equivalent to T5LayerNorm."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        """
        Initialize LlamaRMSNorm.

        Args:
            hidden_size: Size of the hidden dimension
            eps: Epsilon for numerical stability
        """
        super().__init__()
        self.weight = nn.Parameter(torch.empty(hidden_size))
        self.variance_epsilon = eps
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weight parameters to ones."""
        nn.init.ones_(self.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization.

        Args:
            hidden_states: Input tensor to normalize

        Returns:
            Normalized tensor
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance +
                                                    self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Attention(nn.Module):
    """Multi-head attention layer with support for tensor parallelism."""

    def __init__(self, config: Any, layer_idx: int):
        """
        Initialize attention layer.

        Args:
            config: Model configuration object
            layer_idx: Index of this layer in the model

        Raises:
            AssertionError: If attention heads are not divisible by tensor parallel world size
        """
        super().__init__()

        # Store configuration
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_values = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.layer_idx = layer_idx

        # Validate tensor parallelism compatibility
        tp_world_size = pgm.tp_world_size
        assert self.num_heads % tp_world_size == 0, (
            f'num_attention_heads ({self.num_heads}) should be divisible by tp world size ({tp_world_size})'
        )
        assert self.num_key_values % tp_world_size == 0, (
            f'num_key_value_heads ({self.num_key_values}) should be divisible by tp world size ({tp_world_size})'
        )

        # Calculate local heads for tensor parallelism
        self.num_local_heads = self.num_heads // tp_world_size
        self.num_local_kv_heads = self.num_key_values // tp_world_size

        # Initialize projection layers
        self.q_proj = nn.Linear(config.hidden_size,
                                self.num_heads * self.head_dim,
                                bias=False)
        self.k_proj = nn.Linear(config.hidden_size,
                                self.num_key_values * self.head_dim,
                                bias=False)
        self.v_proj = nn.Linear(config.hidden_size,
                                self.num_key_values * self.head_dim,
                                bias=False)
        self.out_proj = nn.Linear(config.hidden_size,
                                  config.hidden_size,
                                  bias=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize attention weights using uniform distribution."""

        def _init_weights(tensor: torch.Tensor) -> None:
            k = 1 / tensor.size(1)
            bound = math.sqrt(k)
            torch.nn.init.uniform_(tensor, -bound, bound)

        _init_weights(self.q_proj.weight)
        _init_weights(self.k_proj.weight)
        _init_weights(self.v_proj.weight)
        _init_weights(self.out_proj.weight)

    def forward(self,
                x: torch.Tensor,
                cos: torch.Tensor,
                sin: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply multi-head attention.

        Args:
            x: Input tensor of shape [batch_size, sequence_length, hidden_dim]
            cos: Cosine values for rotary embedding
            sin: Sine values for rotary embedding
            attention_mask: Optional attention mask
            position_ids: Optional position IDs

        Returns:
            Attention output tensor
        """
        batch_size, sequence_length, hidden_dim = x.size()

        # Project input to query, key, value
        q = self.q_proj(x)  # [batch_size, sequence_length, num_heads*head_dim]
        k = self.k_proj(
            x)  # [batch_size, sequence_length, num_key_values*head_dim]
        v = self.v_proj(
            x)  # [batch_size, sequence_length, num_key_values*head_dim]

        # Apply rotary position embedding based on attention type
        if os.getenv('FLASH_ATTEN', '1') != '1':
            # Standard attention with custom rotary embedding
            q = q.view(batch_size, sequence_length, self.num_local_heads,
                       self.head_dim).transpose(1, 2)
            k = k.view(batch_size, sequence_length, self.num_local_kv_heads,
                       self.head_dim).transpose(1, 2)
            v = v.view(batch_size, sequence_length, self.num_local_kv_heads,
                       self.head_dim).transpose(1, 2)
            q = apply_rotary_pos_emb(q, cos, sin)
            k = apply_rotary_pos_emb(k, cos, sin)
        else:
            # Use manual rotary embedding
            q = q.view(batch_size, sequence_length, self.num_local_heads,
                       self.head_dim)
            k = k.view(batch_size, sequence_length, self.num_local_kv_heads,
                       self.head_dim)

            # Apply rotary embeddings manually
            def apply_rotary_emb_manual(x, cos, sin):
                x_rot = x[..., :x.shape[-1] // 2]
                x_pass = x[..., x.shape[-1] // 2:]
                x_rot = torch.stack([-x_rot[..., 1::2], x_rot[..., ::2]],
                                    dim=-1)
                x_rot = x_rot.flatten(-2)
                return torch.cat(
                    [x_rot * cos + x_pass * sin, -x_rot * sin + x_pass * cos],
                    dim=-1)

            q = apply_rotary_emb_manual(q, cos[:, :self.head_dim // 2],
                                        sin[:, :self.head_dim // 2])
            k = apply_rotary_emb_manual(k, cos[:, :self.head_dim // 2],
                                        sin[:, :self.head_dim // 2])
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.view(batch_size, sequence_length, self.num_local_kv_heads,
                       self.head_dim).transpose(1, 2)

        # Repeat key-value heads to match query heads
        k = k.repeat_interleave(self.num_local_heads //
                                self.num_local_kv_heads,
                                dim=1)
        v = v.repeat_interleave(self.num_local_heads //
                                self.num_local_kv_heads,
                                dim=1)

        # Determine causal masking
        causal = q.size(2) == k.size(
            2)  # During decoding, q length is usually 1

        # Apply attention based on configuration
        if os.getenv('CONTEXT_PARALLEL', '0') == '1':
            # Ring attention for context parallelism
            sm_scale = 1.0 / (q.size(-1)**0.5)
            out = context_parallel.ring_attention(q, k, v, sm_scale,
                                                  causal).transpose(1, 2)
        elif os.getenv('FLASH_ATTEN', '1') == '1':
            # Flash attention for efficiency
            out = flash_attention(q, k, v, causal=causal)
        else:
            # Standard PyTorch scaled dot-product attention
            out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
            out = out.transpose(1, 2)

        # Reshape and project output
        out = out.reshape(batch_size, sequence_length,
                          self.num_local_heads * self.head_dim)
        return self.out_proj(out)


class MLP(nn.Module):
    """Multi-layer perceptron with SwiGLU activation."""

    def __init__(self, config: Any) -> None:
        """
        Initialize MLP layer.

        Args:
            config: Model configuration object
        """
        super().__init__()
        self.up_proj = nn.Linear(config.hidden_size,
                                 config.intermediate_size,
                                 bias=False)
        self.gate_proj = nn.Linear(config.hidden_size,
                                   config.intermediate_size,
                                   bias=False)
        self.down_proj = nn.Linear(config.intermediate_size,
                                   config.hidden_size,
                                   bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize MLP weights using uniform distribution."""

        def _init_weights(tensor: torch.Tensor) -> None:
            k = 1 / tensor.size(1)
            bound = math.sqrt(k)
            torch.nn.init.uniform_(tensor, -bound, bound)

        _init_weights(self.up_proj.weight)
        _init_weights(self.gate_proj.weight)
        _init_weights(self.down_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply MLP transformation with SwiGLU activation.

        Args:
            x: Input tensor

        Returns:
            Transformed tensor
        """
        # TODO: Consider breaking this into multiple lines for better debugging
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class FinalProjection(nn.Module):
    """Final projection layer for vocabulary output."""

    def __init__(self, hidden_size: int, vocab_size: int, bias: bool = False):
        """
        Initialize final projection layer.

        Args:
            hidden_size: Size of hidden dimension
            vocab_size: Size of vocabulary
            bias: Whether to include bias term
        """
        super().__init__()
        self.in_features = hidden_size
        self.out_features = vocab_size

        # Note: torch.nn.functional.linear performs XW^T + b so we exchange dimensions
        self.weight = nn.Parameter(
            torch.empty(self.out_features, self.in_features))
        self.bias = nn.Parameter(torch.empty(
            self.out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize projection weights using uniform distribution."""

        def _init_weights(tensor: torch.Tensor) -> None:
            k = 1 / tensor.size(1)
            bound = math.sqrt(k)
            torch.nn.init.uniform_(tensor, -bound, bound)

        _init_weights(self.weight)
        if self.bias is not None:
            _init_weights(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply final linear projection.

        Args:
            x: Input tensor

        Returns:
            Projected tensor
        """
        return F.linear(x, self.weight, self.bias)


class DecoderLayer(nn.Module):
    """Transformer decoder layer with RMSNorm -> Attention -> Residual -> RMSNorm -> MLP -> Residual."""

    def __init__(self, config: Any, layer_idx: int):
        """
        Initialize decoder layer.

        Args:
            config: Model configuration object
            layer_idx: Index of this layer
        """
        super().__init__()

        # Select RMSNorm implementation based on attention type
        RMSNorm = LlamaRMSNorm if os.getenv('FLASH_ATTEN',
                                            '1') != '1' else TritonRMSNorm

        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)
        self.attention = Attention(config, layer_idx=layer_idx)
        self.mlp = MLP(config)
        self.layer_idx = layer_idx

        # Precompute rotary embeddings
        head_dim = config.hidden_size // config.num_attention_heads
        self.cos, self.sin = get_cos_sin(
            config.max_position_embeddings,
            head_dim=head_dim,
            base=config.rope_theta)  # [max_position_embeddings, head_dim]

        # Update for context parallelism if enabled
        self.cos, self.sin = context_parallel.update_rope_for_context_parallel(
            self.cos, self.sin)

    def forward(self,
                x: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply decoder layer transformation.

        Args:
            x: Input tensor
            attention_mask: Optional attention mask
            position_ids: Optional position IDs

        Returns:
            Transformed tensor
        """
        # TODO: Implement proper position_ids handling for generation
        cos, sin = self.cos, self.sin

        # Attention block with residual connection
        x = x + self.attention(self.input_layernorm(x), cos, sin,
                               attention_mask, position_ids)

        # MLP block with residual connection
        x = x + self.mlp(self.post_attention_layernorm(x))

        return x


class Embedding(nn.Module):
    """Embedding layer with custom initialization."""

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 padding_idx: Optional[int] = None):
        """
        Initialize embedding layer.

        Args:
            num_embeddings: Size of vocabulary
            embedding_dim: Dimension of embeddings
            padding_idx: Index for padding token
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize embeddings with normal distribution."""
        torch.nn.init.normal_(self.weight, mean=0.0, std=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply embedding lookup.

        Args:
            x: Input token indices

        Returns:
            Embedded representations
        """
        return F.embedding(x, self.weight, self.padding_idx)


class Llama(nn.Module):
    """Llama transformer model implementation."""

    def __init__(self, config: Any) -> None:
        """
        Initialize Llama model.

        Args:
            config: Model configuration object

        Raises:
            AssertionError: If model dimensions are incompatible
        """
        super().__init__()

        # Validate model configuration
        assert config.hidden_size % config.num_attention_heads == 0, (
            f'hidden_size ({config.hidden_size}) must be divisible by '
            f'num_attention_heads ({config.num_attention_heads})')
        assert config.num_attention_heads % config.num_key_value_heads == 0, (
            f'num_attention_heads ({config.num_attention_heads}) must be divisible by '
            f'num_key_value_heads ({config.num_key_value_heads})')

        # Store model parameters
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_values = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.num_layers = config.num_hidden_layers
        self.model_config = config

        # Initialize model components
        self.embedding = Embedding(self.vocab_size, self.hidden_size)
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(config, layer_idx=i) for i in range(self.num_layers)
        ])
        self.final_proj = FinalProjection(self.hidden_size,
                                          self.vocab_size,
                                          bias=False)

        # Select final RMSNorm implementation
        RMSNorm = LlamaRMSNorm if os.getenv('FLASH_ATTEN',
                                            '1') != '1' else TritonRMSNorm
        self.final_norm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset all model parameters."""
        self.embedding.reset_parameters()

        for layer in self.decoder_layers:
            layer.input_layernorm.reset_parameters()
            layer.attention.reset_parameters()
            layer.post_attention_layernorm.reset_parameters()
            layer.mlp.reset_parameters()

        self.final_norm.reset_parameters()
        self.final_proj.reset_parameters()

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                gradient_checkpointing: bool = False) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            input_ids: Input token indices of shape [batch_size, sequence_length]
            attention_mask: Optional attention mask
            position_ids: Optional position IDs
            gradient_checkpointing: Whether to use gradient checkpointing

        Returns:
            Logits tensor of shape [batch_size, sequence_length, vocab_size]
        """
        # Embed input tokens
        x = self.embedding(input_ids)

        # Apply transformer layers with optional gradient checkpointing
        if gradient_checkpointing:
            # Define a function to pass to checkpoint_sequential
            def create_layer_fn(module):

                def forward_fn(x):
                    return module(x, attention_mask, position_ids)

                return forward_fn

            # Apply gradient checkpointing to decoder layers
            x = checkpoint_sequential(self.decoder_layers,
                                      chunks=1,
                                      input=x,
                                      create_layer_fn=create_layer_fn)
        else:
            # Standard forward pass
            for layer in self.decoder_layers:
                x = layer(x, attention_mask, position_ids)

        # Apply final normalization and projection
        x = self.final_norm(x)
        logits = self.final_proj(x)

        return logits  # [batch_size, sequence_length, vocab_size]
