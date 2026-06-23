"""Llama model with flash attention, context/tensor parallelism, and RoPE."""

from __future__ import annotations

import inspect
import math
import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from transformers.utils import (
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    is_torch_npu_available,
)

from scaletorch.env import (
    DEFAULT_DTYPE,
    ENV_CONTEXT_PARALLEL,
    ENV_DTYPE,
    ENV_FLASH_ATTENTION,
    ENV_SEQUENCE_PARALLEL,
)
from scaletorch.parallel.context_parallel import context_parallel
from scaletorch.parallel.process_group import process_group_manager as pgm
from scaletorch.parallel.sequence_parallel.sp_comms import (
    AllGatherFromSequenceParallelRegion,
    ReduceScatterToSequenceParallelRegion,
)
from scaletorch.utils.device import get_current_device

# ---------------------------------------------------------------------------
# Attention backend registry
# ---------------------------------------------------------------------------
_ATTENTION_REGISTRY: dict[str, Any] = {}


def register_attention_backend(name: str):
    """Decorator to register a named attention backend.

    Each backend is a callable with the signature::

        fn(q, k, v, *, causal: bool) -> Tensor

    where ``q/k/v`` have shape ``[B, H, S, D]`` and the return value has
    shape ``[B, S, H, D]``.

    Example::

        @register_attention_backend("my_custom")
        def _my_attn(q, k, v, *, causal=True):
            ...
    """

    def decorator(fn):
        _ATTENTION_REGISTRY[name] = fn
        return fn

    return decorator


@register_attention_backend("ring")
def _ring_attention_backend(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool = True
) -> torch.Tensor:
    sm_scale = 1.0 / (q.size(-1) ** 0.5)
    return context_parallel.ring_attention(q, k, v, sm_scale, causal).transpose(1, 2)


@register_attention_backend("flash")
def _flash_attention_backend(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool = True
) -> torch.Tensor:
    return flash_attention(q, k, v, causal=causal)


@register_attention_backend("sdpa")
def _sdpa_attention_backend(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool = True
) -> torch.Tensor:
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal).transpose(1, 2)


def get_attention_backend(name: str):
    """Look up a registered attention backend by *name*.

    Raises:
        KeyError: If *name* is not registered.
    """
    if name not in _ATTENTION_REGISTRY:
        raise KeyError(
            f"Unknown attention backend '{name}'. "
            f"Registered: {list(_ATTENTION_REGISTRY.keys())}"
        )
    return _ATTENTION_REGISTRY[name]


def _resolve_attention_backend_name(
    use_context_parallel: bool, use_flash_attn: bool
) -> str:
    """Choose the best attention backend name from env flags."""
    if use_context_parallel:
        return "ring"
    if use_flash_attn:
        return "flash"
    return "sdpa"


class AttentionBackend:
    """Probe and cache the best available flash attention implementation.

    Encapsulates module-level probing (inspect, try/except) into a testable,
    injectable singleton. Access via ``AttentionBackend.get()``.
    """

    _instance: AttentionBackend | None = None

    def __init__(self) -> None:
        self.supports_window_size: bool = False
        self.supports_deterministic: bool = False
        self.use_top_left_mask: bool = False
        self.use_npu: bool = False
        self.use_flash: bool = False
        self._npu_func: Any = None
        self._flash_func: Any = None
        self._probe()

    def _probe(self) -> None:
        if is_torch_npu_available():
            self.use_npu = True
            try:
                from transformers.integrations.npu_flash_attention import (
                    npu_flash_attn_func,
                )

                self._npu_func = npu_flash_attn_func
                self.supports_window_size = (
                    "window_size" in inspect.signature(npu_flash_attn_func).parameters
                )
            except ImportError:
                self.use_npu = False

        elif is_flash_attn_2_available():
            from flash_attn import flash_attn_func

            self._flash_func = flash_attn_func
            self.supports_window_size = (
                "window_size" in inspect.signature(flash_attn_func).parameters
            )
            self.supports_deterministic = (
                "deterministic" in inspect.signature(flash_attn_func).parameters
            )
            self.use_top_left_mask = not is_flash_attn_greater_or_equal_2_10()
            self.use_flash = True

    @classmethod
    def get(cls) -> AttentionBackend:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


# Singleton — kept as private module variable for backward compatibility
_attention_backend: AttentionBackend = AttentionBackend.get()

# Legacy module-level aliases for backward compatibility
_flash_supports_window_size: bool = _attention_backend.supports_window_size
_flash_supports_deterministic: bool = _attention_backend.supports_deterministic
_flash_use_top_left_mask: bool = _attention_backend.use_top_left_mask
_use_npu_flash_attn: bool = _attention_backend.use_npu
_use_flash_attn: bool = _attention_backend.use_flash


def _init_weights(tensor: torch.Tensor) -> None:
    if tensor.ndim < 2:
        torch.nn.init.zeros_(tensor)
        return
    k = 1 / tensor.size(1)
    bound = math.sqrt(k)
    torch.nn.init.uniform_(tensor, -bound, bound)


def apply_rotary_pos_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """
    Apply rotary position embedding to input tensor.

    Args:
        x: Input tensor of shape [batch_size, num_heads, sequence_length, head_dim]
        cos: Cosine values for rotation
        sin: Sine values for rotation

    Returns:
        Rotated tensor with same shape as input
    """
    _batch_size, _num_head, _sequence_length, head_dim = x.size()

    # Ensure cos/sin have correct shape for broadcasting [1, 1, seq_len, head_dim]
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]
    rotate_half = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotate_half * sin


def get_cos_sin(
    sequence_length: int,
    head_dim: int,
    base: float = 500000.0,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate cosine and sine values for rotary position embedding.

    Args:
        sequence_length: Length of the sequence
        head_dim: Dimension of each attention head
        base: Base for frequency calculation (default: 500000.0)
        device: Target device (auto-detected if None)
        dtype: Target dtype (auto-detected if None)

    Returns:
        Tuple of (cos, sin) tensors for position embedding.

    Raises:
        ValueError: If head_dim is not even
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")

    if device is None:
        device = get_current_device()
    if dtype is None:
        dtype = (
            torch.bfloat16
            if os.getenv(ENV_DTYPE, DEFAULT_DTYPE) == "bfloat16"
            else torch.float32
        )

    theta = 1.0 / (
        base
        ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )

    position = torch.arange(
        sequence_length, dtype=torch.float32, device=device
    ).unsqueeze(1)

    freqs = position * theta
    cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1).to(dtype)
    sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1).to(dtype)

    return cos, sin


def flash_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True
) -> torch.Tensor:
    """
    Apply flash attention using the best available implementation.

    Args:
        q: Query tensor of shape [batch_size, num_heads, sequence_length, head_dim]
        k: Key tensor of shape [batch_size, num_heads, sequence_length, head_dim]
        v: Value tensor of shape [batch_size, num_heads, sequence_length, head_dim]
        causal: Whether to use causal masking

    Returns:
        Attention output tensor of shape [batch_size, sequence_length, num_heads, head_dim]
    """
    backend = _attention_backend
    if backend.use_npu and backend._npu_func is not None:
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        softmax_scale = 1.0 / (q.size(-1) ** 0.5)
        out = backend._npu_func(q, k, v, softmax_scale=softmax_scale, causal=causal)
        return out

    # F.scaled_dot_product_attention expects [B, H, S, D] and returns [B, H, S, D]
    out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    # Transpose to [B, S, H, D] for consistent output format
    return out.transpose(1, 2)


class RMSNorm(nn.Module):
    """RMSNorm implementation supporting optional device/dtype placement.

    Replaces the former ``FusedRMSNorm`` and ``LlamaRMSNorm`` classes which
    were functionally identical.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Initialize RMSNorm.

        Args:
            hidden_size: Size of the hidden dimension
            eps: Epsilon for numerical stability
            device: Device to place parameters on
            dtype: Data type for parameters
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(hidden_size, device=device, dtype=dtype))
        self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weight parameters to ones."""
        nn.init.ones_(self.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization."""
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


# Backward-compatible aliases
FusedRMSNorm = RMSNorm
LlamaRMSNorm = RMSNorm


class LlamaAttention(nn.Module):
    """Multi-head attention layer with support for tensor parallelism."""

    def __init__(self, config: Any, layer_idx: int):
        """
        Initialize attention layer.

        Args:
            config: Model configuration object
            layer_idx: Index of this layer in the model

        Raises:
            ValueError: If attention heads are not divisible by tensor parallel world size
        """
        super().__init__()

        # Store configuration
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.layer_idx = layer_idx

        # Validate tensor parallelism compatibility
        tp_world_size = pgm.tp_world_size if pgm else 1
        if self.num_heads % tp_world_size != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_heads}) should be divisible by tp world size ({tp_world_size})"
            )
        if self.num_key_value_heads % tp_world_size != 0:
            raise ValueError(
                f"num_key_value_heads ({self.num_key_value_heads}) should be divisible by tp world size ({tp_world_size})"
            )

        # Calculate local heads for tensor parallelism
        self.num_local_heads = self.num_heads // tp_world_size
        self.num_local_kv_heads = self.num_key_value_heads // tp_world_size

        # Initialize projection layers
        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.reset_parameters()

        # Resolve env-dependent flags once at init
        self._use_flash_attn = os.getenv(ENV_FLASH_ATTENTION, "1") == "1"
        self._use_context_parallel = os.getenv(ENV_CONTEXT_PARALLEL, "0") == "1"
        self._attn_backend_name = _resolve_attention_backend_name(
            self._use_context_parallel, self._use_flash_attn
        )

    def reset_parameters(self) -> None:
        """Initialize attention weights using uniform distribution."""

        _init_weights(self.q_proj.weight)
        _init_weights(self.k_proj.weight)
        _init_weights(self.v_proj.weight)
        _init_weights(self.out_proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
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
        batch_size, sequence_length, _hidden_dim = x.size()

        # Project input to query, key, value
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to [batch, heads, seq_len, head_dim]
        q = q.view(
            batch_size, sequence_length, self.num_local_heads, self.head_dim
        ).transpose(1, 2)
        k = k.view(
            batch_size, sequence_length, self.num_local_kv_heads, self.head_dim
        ).transpose(1, 2)
        v = v.view(
            batch_size, sequence_length, self.num_local_kv_heads, self.head_dim
        ).transpose(1, 2)

        # Apply rotary position embedding
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        # Expand key-value heads to match query heads (zero-copy via expand)
        n_rep = self.num_local_heads // self.num_local_kv_heads
        if n_rep > 1:
            k = (
                k.unsqueeze(2)
                .expand(-1, -1, n_rep, -1, -1)
                .reshape(
                    batch_size, self.num_local_heads, sequence_length, self.head_dim
                )
            )
            v = (
                v.unsqueeze(2)
                .expand(-1, -1, n_rep, -1, -1)
                .reshape(
                    batch_size, self.num_local_heads, sequence_length, self.head_dim
                )
            )

        # Determine causal masking
        causal = q.size(2) == k.size(2)  # During decoding, q length is usually 1

        # Apply attention via registered backend
        attn_fn = get_attention_backend(self._attn_backend_name)
        out = attn_fn(q, k, v, causal=causal)

        # Reshape and project output
        out = out.reshape(
            batch_size, sequence_length, self.num_local_heads * self.head_dim
        )
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
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize MLP weights using uniform distribution."""

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
        # SwiGLU activation: down_proj(silu(gate_proj(x)) * up_proj(x))
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
        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.bias = nn.Parameter(torch.empty(self.out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize projection weights using uniform distribution."""

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

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.attention = LlamaAttention(config, layer_idx=layer_idx)
        self.mlp = MLP(config)
        self.layer_idx = layer_idx

        self._use_sp = (
            os.getenv(ENV_SEQUENCE_PARALLEL, "0") == "1"
            and pgm
            and pgm.tp_world_size > 1
        )

        # Precompute rotary embeddings (registered as buffers so they move with model.to(device))
        head_dim = config.hidden_size // config.num_attention_heads
        # Handle both old (config.rope_theta) and new (config.rope_scaling['rope_theta']) formats
        rope_theta = getattr(config, "rope_theta", None)
        if rope_theta is None:
            rope_scaling = getattr(config, "rope_scaling", {}) or {}
            rope_theta = rope_scaling.get("rope_theta", 500000.0)
        cos, sin = get_cos_sin(
            config.max_position_embeddings,
            head_dim=head_dim,
            base=rope_theta,
            device=torch.device("cpu"),
        )

        # Update for context parallelism if enabled
        if pgm and pgm.cp_world_size > 1:
            cos, sin = context_parallel.update_rope_for_context_parallel(cos, sin)

        # Register as buffers so they move with model.to(device)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Apply decoder layer transformation.

        Args:
            x: Input tensor
            attention_mask: Optional attention mask
            position_ids: Optional position IDs

        Returns:
            Transformed tensor
        """
        # When position_ids is provided (e.g., during generation with KV cache),
        # use it to index cos/sin. Otherwise, slice by sequence length.
        seq_len = x.size(1)
        if position_ids is not None:
            cos = self.cos[position_ids].unsqueeze(0)
            sin = self.sin[position_ids].unsqueeze(0)
        else:
            cos = self.cos[:seq_len].unsqueeze(0).unsqueeze(0)
            sin = self.sin[:seq_len].unsqueeze(0).unsqueeze(0)

        # Attention block with residual connection
        if self._use_sp:
            x_full = AllGatherFromSequenceParallelRegion.apply(x)
            attn_out = self.attention(
                self.input_layernorm(x_full), cos, sin, attention_mask, position_ids
            )
            attn_out = ReduceScatterToSequenceParallelRegion.apply(attn_out)
            x = x + attn_out
        else:
            x = x + self.attention(
                self.input_layernorm(x), cos, sin, attention_mask, position_ids
            )

        # MLP block with residual connection
        if self._use_sp:
            x_full = AllGatherFromSequenceParallelRegion.apply(x)
            mlp_out = self.mlp(self.post_attention_layernorm(x_full))
            mlp_out = ReduceScatterToSequenceParallelRegion.apply(mlp_out)
            x = x + mlp_out
        else:
            x = x + self.mlp(self.post_attention_layernorm(x))

        return x


class LlamaEmbedding(nn.Module):
    """Embedding layer with custom initialization."""

    def __init__(
        self, num_embeddings: int, embedding_dim: int, padding_idx: int | None = None
    ):
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
        torch.nn.init.normal_(
            self.weight, mean=0.0, std=1.0 / math.sqrt(self.embedding_dim)
        )

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
            ValueError: If model dimensions are incompatible
        """
        super().__init__()

        # Validate model configuration
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({config.hidden_size}) must be divisible by "
                f"num_attention_heads ({config.num_attention_heads})"
            )
        if config.num_attention_heads % config.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({config.num_attention_heads}) must be "
                f"divisible by num_key_value_heads ({config.num_key_value_heads})"
            )

        # Store model parameters
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.num_layers = config.num_hidden_layers
        self.model_config = config
        self.config = config

        # Initialize model components
        self.embedding = LlamaEmbedding(self.vocab_size, self.hidden_size)
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(config, layer_idx=i) for i in range(self.num_layers)]
        )
        self.final_proj = FinalProjection(self.hidden_size, self.vocab_size, bias=False)

        self.final_norm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

        self._use_sp = (
            os.getenv(ENV_SEQUENCE_PARALLEL, "0") == "1"
            and pgm
            and pgm.tp_world_size > 1
        )

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

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        gradient_checkpointing: bool = False,
    ) -> torch.Tensor:
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

        # Partition along sequence dimension for SP
        if self._use_sp:
            x = ReduceScatterToSequenceParallelRegion.apply(x)

        # Apply transformer layers with optional gradient checkpointing
        if gradient_checkpointing:
            for layer in self.decoder_layers:
                x = torch_checkpoint(
                    layer, x, attention_mask, position_ids, use_reentrant=False
                )
        else:
            # Standard forward pass
            for layer in self.decoder_layers:
                x = layer(x, attention_mask, position_ids)

        # Apply final normalization and projection
        if self._use_sp:
            x = AllGatherFromSequenceParallelRegion.apply(x)
        x = self.final_norm(x)
        logits = self.final_proj(x)

        return logits  # [batch_size, sequence_length, vocab_size]
