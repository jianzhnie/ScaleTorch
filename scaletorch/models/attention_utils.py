"""Shared attention backend utilities for all model implementations.

Provides the AttentionBackend singleton, attention registry, flash_attention
helper, and common utilities (RoPE, RMSNorm, weight init) used by both
Llama and Qwen3 models.
"""

from __future__ import annotations

import inspect
import math
import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils import (
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    is_torch_npu_available,
)

from scaletorch.env import DEFAULT_DTYPE, ENV_CONTEXT_PARALLEL, ENV_DTYPE, ENV_FLASH_ATTENTION
from scaletorch.utils.device import get_current_device

# ---------------------------------------------------------------------------
# Attention backend registry
# ---------------------------------------------------------------------------
_ATTENTION_REGISTRY: dict[str, Any] = {}


def register_attention_backend(name: str):
    """Decorator to register a named attention backend."""

    def decorator(fn):
        _ATTENTION_REGISTRY[name] = fn
        return fn

    return decorator


def get_attention_backend(name: str):
    """Look up a registered attention backend by *name*."""
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


# ---------------------------------------------------------------------------
# AttentionBackend singleton
# ---------------------------------------------------------------------------


class AttentionBackend:
    """Probe and cache the best available flash attention implementation."""

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


_attention_backend: AttentionBackend = AttentionBackend.get()


# ---------------------------------------------------------------------------
# Flash attention dispatch
# ---------------------------------------------------------------------------


def flash_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True
) -> torch.Tensor:
    """Apply flash attention using the best available implementation.

    Args:
        q, k, v: Tensors of shape [batch_size, num_heads, sequence_length, head_dim]
        causal: Whether to use causal masking

    Returns:
        Attention output of shape [batch_size, sequence_length, num_heads, head_dim]
    """
    backend = _attention_backend
    if backend.use_npu and backend._npu_func is not None:
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        softmax_scale = 1.0 / (q.size(-1) ** 0.5)
        out = backend._npu_func(q, k, v, softmax_scale=softmax_scale, causal=causal)
        return out

    out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    return out.transpose(1, 2)


# ---------------------------------------------------------------------------
# Common utilities
# ---------------------------------------------------------------------------


def _init_weights(tensor: torch.Tensor) -> None:
    """Initialize tensor with uniform distribution based on fan-in."""
    if tensor.ndim < 2:
        torch.nn.init.zeros_(tensor)
        return
    k = 1 / tensor.size(1)
    bound = math.sqrt(k)
    torch.nn.init.uniform_(tensor, -bound, bound)


def apply_rotary_pos_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply rotary position embedding to input tensor.

    Args:
        x: Input tensor of shape [batch_size, num_heads, sequence_length, head_dim]
        cos: Cosine values for rotation
        sin: Sine values for rotation

    Returns:
        Rotated tensor with same shape as input
    """
    _batch_size, _num_head, _sequence_length, head_dim = x.size()

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
    """Generate cosine and sine values for rotary position embedding.

    Args:
        sequence_length: Length of the sequence
        head_dim: Dimension of each attention head
        base: Base for frequency calculation (default: 500000.0)
        device: Target device (auto-detected if None)
        dtype: Target dtype (auto-detected if None)

    Returns:
        Tuple of (cos, sin) tensors for position embedding.
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


# ---------------------------------------------------------------------------
# RMSNorm (shared across all models)
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """RMSNorm implementation supporting optional device/dtype placement."""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(hidden_size, device=device, dtype=dtype))
        self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)
