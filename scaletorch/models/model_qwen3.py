"""Qwen3 model implementation for ScaleTorch.

Key differences from Llama:
- Explicit head_dim from config (may differ from hidden_size // num_heads)
- QK norms (RMSNorm on Q and K after projection, before RoPE)
- tie_word_embeddings support
"""

from __future__ import annotations

import math
import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from scaletorch.env import ENV_CONTEXT_PARALLEL, ENV_FLASH_ATTENTION
from scaletorch.models.attention_utils import (
    RMSNorm,
    _init_weights,
    _resolve_attention_backend_name,
    apply_rotary_pos_emb,
    flash_attention,
    get_attention_backend,
    get_cos_sin,
    register_attention_backend,
)
from scaletorch.parallel.context_parallel import context_parallel
from scaletorch.parallel.process_group import process_group_manager as pgm


# Register backends into the shared registry (idempotent — re-registration overwrites)
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


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class Qwen3MLP(nn.Module):
    """Multi-layer perceptron with SwiGLU activation for Qwen3."""

    def __init__(self, config: Any) -> None:
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
        _init_weights(self.up_proj.weight)
        _init_weights(self.gate_proj.weight)
        _init_weights(self.down_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class Qwen3Embedding(nn.Module):
    """Embedding layer with custom initialization for Qwen3."""

    def __init__(
        self, num_embeddings: int, embedding_dim: int, padding_idx: int | None = None
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.normal_(
            self.weight, mean=0.0, std=1.0 / math.sqrt(self.embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.embedding(x, self.weight, self.padding_idx)


class FinalProjection(nn.Module):
    """Final projection layer for vocabulary output."""

    def __init__(self, hidden_size: int, vocab_size: int, bias: bool = False):
        super().__init__()
        self.in_features = hidden_size
        self.out_features = vocab_size

        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.bias = nn.Parameter(torch.empty(self.out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _init_weights(self.weight)
        if self.bias is not None:
            _init_weights(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


# ---------------------------------------------------------------------------
# Qwen3 Attention
# ---------------------------------------------------------------------------


class Qwen3Attention(nn.Module):
    """Qwen3 attention with explicit head_dim and QK norms."""

    def __init__(self, config: Any, layer_idx: int):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_values = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.layer_idx = layer_idx

        tp_world_size = pgm.tp_world_size if pgm else 1
        if self.num_heads % tp_world_size != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_heads}) must be divisible "
                f"by tp_world_size ({tp_world_size})"
            )
        if self.num_key_values % tp_world_size != 0:
            raise ValueError(
                f"num_key_value_heads ({self.num_key_values}) must be divisible "
                f"by tp_world_size ({tp_world_size})"
            )

        self.num_local_heads = self.num_heads // tp_world_size
        self.num_local_kv_heads = self.num_key_values // tp_world_size

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_values * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_values * self.head_dim, bias=False
        )
        self.out_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self._use_cp = os.getenv(ENV_CONTEXT_PARALLEL, "0") == "1"
        self._use_flash = os.getenv(ENV_FLASH_ATTENTION, "1") == "1"
        self._attn_backend_name = _resolve_attention_backend_name(
            self._use_cp, self._use_flash
        )

    def reset_parameters(self) -> None:
        nn.init.normal_(self.q_proj.weight, std=0.02)
        nn.init.normal_(self.k_proj.weight, std=0.02)
        nn.init.normal_(self.v_proj.weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        self.q_norm.reset_parameters()
        self.k_norm.reset_parameters()

    def forward(self, x, cos, sin, attention_mask=None, position_ids=None):
        batch_size, seq_len, _ = x.size()

        q = self.q_proj(x).view(
            batch_size, seq_len, self.num_local_heads, self.head_dim
        )
        k = self.k_proj(x).view(
            batch_size, seq_len, self.num_local_kv_heads, self.head_dim
        )
        v = self.v_proj(x).view(
            batch_size, seq_len, self.num_local_kv_heads, self.head_dim
        )

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        num_kv_groups = self.num_local_heads // self.num_local_kv_heads
        if num_kv_groups > 1:
            k = (
                k.unsqueeze(2)
                .expand(-1, -1, num_kv_groups, -1, -1)
                .reshape(batch_size, self.num_local_heads, seq_len, self.head_dim)
            )
            v = (
                v.unsqueeze(2)
                .expand(-1, -1, num_kv_groups, -1, -1)
                .reshape(batch_size, self.num_local_heads, seq_len, self.head_dim)
            )

        causal = q.size(2) == k.size(2)

        attn_fn = get_attention_backend(self._attn_backend_name)
        out = attn_fn(q, k, v, causal=causal)

        out = out.reshape(batch_size, seq_len, self.num_local_heads * self.head_dim)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Qwen3 Decoder Layer & Model
# ---------------------------------------------------------------------------


class Qwen3DecoderLayer(nn.Module):
    """Qwen3 decoder layer (RoPE passed from parent model)."""

    def __init__(self, config: Any, layer_idx: int):
        super().__init__()

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.attention = Qwen3Attention(config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.layer_idx = layer_idx

    def forward(self, x, attention_mask=None, position_ids=None, cos=None, sin=None):
        seq_len = x.size(1)
        cos_slice = cos[:seq_len]
        sin_slice = sin[:seq_len]

        x = x + self.attention(
            self.input_layernorm(x), cos_slice, sin_slice, attention_mask, position_ids
        )
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen3(nn.Module):
    """Qwen3 transformer model with shared RoPE buffers."""

    def __init__(self, config: Any) -> None:
        super().__init__()

        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_values = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.max_position_embeddings = config.max_position_embeddings
        self.num_layers = config.num_hidden_layers
        self.tie_word_embeddings = getattr(config, "tie_word_embeddings", False)
        self.model_config = config
        self.config = config

        self.embedding = Qwen3Embedding(self.vocab_size, self.hidden_size)
        self.decoder_layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx=i) for i in range(self.num_layers)]
        )

        self.final_norm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.final_proj = FinalProjection(self.hidden_size, self.vocab_size, bias=False)

        if self.tie_word_embeddings:
            self.final_proj.weight = self.embedding.weight

        # Shared RoPE: compute once, pass to all layers
        rope_theta = getattr(config, "rope_theta", 500000.0)
        cos, sin = get_cos_sin(
            config.max_position_embeddings,
            head_dim=self.head_dim,
            base=rope_theta,
            device=torch.device("cpu"),
        )
        if pgm and pgm.cp_world_size > 1:
            cos, sin = context_parallel.update_rope_for_context_parallel(cos, sin)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def reset_parameters(self) -> None:
        self.embedding.reset_parameters()
        for layer in self.decoder_layers:
            layer.input_layernorm.reset_parameters()
            layer.attention.reset_parameters()
            layer.post_attention_layernorm.reset_parameters()
            layer.mlp.reset_parameters()
        self.final_norm.reset_parameters()
        if not self.tie_word_embeddings:
            self.final_proj.reset_parameters()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        gradient_checkpointing=False,
    ):
        x = self.embedding(input_ids)

        if gradient_checkpointing:
            for layer in self.decoder_layers:
                x = torch_checkpoint(
                    layer, x, attention_mask, position_ids, self.cos, self.sin,
                    use_reentrant=False
                )
        else:
            for layer in self.decoder_layers:
                x = layer(x, attention_mask, position_ids, self.cos, self.sin)

        x = self.final_norm(x)
        logits = self.final_proj(x)
        return logits
