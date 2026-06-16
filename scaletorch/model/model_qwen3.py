"""
Qwen3 model implementation for ScaleTorch.

Key differences from Llama:
- Explicit head_dim from config (may differ from hidden_size // num_heads)
- QK norms (RMSNorm on Q and K after projection, before RoPE)
- tie_word_embeddings support
"""

import os
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from scaletorch.parallel.context_parallel import context_parallel
from scaletorch.parallel.pg_manager import process_group_manager as pgm
from scaletorch.model.model_llama import (
    LlamaRMSNorm,
    TritonRMSNorm,
    Embedding,
    MLP,
    FinalProjection,
    apply_rotary_pos_emb,
    flash_attention,
    get_cos_sin,
)


class Qwen3RMSNorm(nn.Module):
    """RMSNorm used for QK head norms in Qwen3."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(input_dtype)


class Qwen3Attention(nn.Module):
    """Qwen3 attention with explicit head_dim and QK norms."""

    def __init__(self, config: Any, layer_idx: int):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_values = config.num_key_value_heads
        self.head_dim = getattr(config, 'head_dim', self.hidden_size // self.num_heads)
        self.layer_idx = layer_idx

        tp_world_size = pgm.tp_world_size if pgm else 1
        assert self.num_heads % tp_world_size == 0
        assert self.num_key_values % tp_world_size == 0

        self.num_local_heads = self.num_heads // tp_world_size
        self.num_local_kv_heads = self.num_key_values // tp_world_size

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_values * self.head_dim, bias=False)
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_values * self.head_dim, bias=False)
        self.out_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, x, cos, sin, attention_mask=None, position_ids=None):
        batch_size, seq_len, _ = x.size()

        q = self.q_proj(x).view(
            batch_size, seq_len, self.num_local_heads, self.head_dim)
        k = self.k_proj(x).view(
            batch_size, seq_len, self.num_local_kv_heads, self.head_dim)
        v = self.v_proj(x).view(
            batch_size, seq_len, self.num_local_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        num_kv_groups = self.num_local_heads // self.num_local_kv_heads
        if num_kv_groups > 1:
            k = k.repeat_interleave(num_kv_groups, dim=1)
            v = v.repeat_interleave(num_kv_groups, dim=1)

        causal = q.size(2) == k.size(2)

        if os.getenv('CONTEXT_PARALLEL', '0') == '1':
            sm_scale = 1.0 / (q.size(-1) ** 0.5)
            out = context_parallel.ring_attention(
                q, k, v, sm_scale, causal).transpose(1, 2)
        elif os.getenv('FLASH_ATTEN', '1') == '1':
            out = flash_attention(q, k, v, causal=causal)
        else:
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=causal).transpose(1, 2)

        out = out.reshape(batch_size, seq_len,
                          self.num_local_heads * self.head_dim)
        return self.out_proj(out)


class Qwen3DecoderLayer(nn.Module):
    """Qwen3 decoder layer."""

    def __init__(self, config: Any, layer_idx: int):
        super().__init__()

        RMSNorm = LlamaRMSNorm if os.getenv(
            'FLASH_ATTEN', '1') != '1' else TritonRMSNorm

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)
        self.attention = Qwen3Attention(config, layer_idx=layer_idx)
        self.mlp = MLP(config)
        self.layer_idx = layer_idx

        head_dim = getattr(config, 'head_dim',
                           config.hidden_size // config.num_attention_heads)
        rope_theta = getattr(config, 'rope_theta', 500000.0)
        cos, sin = get_cos_sin(
            config.max_position_embeddings,
            head_dim=head_dim,
            base=rope_theta,
            device=torch.device('cpu'))
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

        self.cos, self.sin = context_parallel.update_rope_for_context_parallel(
            self.cos, self.sin)

    def forward(self, x, attention_mask=None, position_ids=None):
        seq_len = x.size(1)
        cos = self.cos[:seq_len]
        sin = self.sin[:seq_len]

        x = x + self.attention(
            self.input_layernorm(x), cos, sin, attention_mask, position_ids)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen3(nn.Module):
    """Qwen3 transformer model."""

    def __init__(self, config: Any) -> None:
        super().__init__()

        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_values = config.num_key_value_heads
        self.head_dim = getattr(config, 'head_dim',
                                self.hidden_size // self.num_heads)
        self.max_position_embeddings = config.max_position_embeddings
        self.num_layers = config.num_hidden_layers
        self.tie_word_embeddings = getattr(config, 'tie_word_embeddings', False)
        self.model_config = config
        self.config = config

        self.embedding = Embedding(self.vocab_size, self.hidden_size)
        self.decoder_layers = nn.ModuleList([
            Qwen3DecoderLayer(config, layer_idx=i)
            for i in range(self.num_layers)
        ])

        RMSNorm = LlamaRMSNorm if os.getenv(
            'FLASH_ATTEN', '1') != '1' else TritonRMSNorm
        self.final_norm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

        self.final_proj = FinalProjection(
            self.hidden_size, self.vocab_size, bias=False)

        if self.tie_word_embeddings:
            self.final_proj.weight = self.embedding.weight

    def reset_parameters(self) -> None:
        self.embedding.reset_parameters()
        for layer in self.decoder_layers:
            layer.input_layernorm.reset_parameters()
            layer.post_attention_layernorm.reset_parameters()
        self.final_norm.reset_parameters()
        if not self.tie_word_embeddings:
            self.final_proj.reset_parameters()

    def forward(self, input_ids, attention_mask=None, position_ids=None,
                gradient_checkpointing=False):
        x = self.embedding(input_ids)

        if gradient_checkpointing:
            from torch.utils.checkpoint import checkpoint
            for layer in self.decoder_layers:
                x = checkpoint(
                    layer, x, attention_mask, position_ids,
                    use_reentrant=False)
        else:
            for layer in self.decoder_layers:
                x = layer(x, attention_mask, position_ids)

        x = self.final_norm(x)
        logits = self.final_proj(x)
        return logits
