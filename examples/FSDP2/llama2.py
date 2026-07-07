"""Llama 2 model — GQA + RoPE + SwiGLU, DTensor/FSDP2-compatible.

Architecture:
  * RotaryPositionEncoding (module with cos/sin buffers)
  * Grouped Query Attention (enable_gqa via PyTorch SDPA)
  * SwiGLU MLP (gate_proj + up_proj + down_proj)
  * Pre-norm with nn.RMSNorm
  * LlamaModel (base) + LlamaForPretraining (base + lm_head)

Model presets for ``--model-size``:
  debug   hidden=256,  layers=4,   heads=8,   kv_heads=4
  1B      hidden=2048, layers=24,  heads=16,  kv_heads=4
  7B      hidden=4096, layers=32,  heads=32,  kv_heads=8
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# -- model presets -----------------------------------------------------------
_MODEL_PRESETS = {
    "debug": dict(
        vocab_size=1024,
        max_position_embeddings=256,
        hidden_size=256,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
    ),
    "1B": dict(
        vocab_size=32000,
        max_position_embeddings=2048,
        hidden_size=2048,
        intermediate_size=5632,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_key_value_heads=4,
    ),
    "7B": dict(
        vocab_size=32000,
        max_position_embeddings=4096,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
    ),
}


class LlamaConfig:
    """Llama model hyperparameters."""

    def __init__(
        self,
        vocab_size: int = 50000,
        max_position_embeddings: int = 2048,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        num_key_value_heads: int = 3,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

    @classmethod
    def from_preset(cls, name: str) -> "LlamaConfig":
        """Build config from a named preset: 'debug', '1B', '7B'."""
        if name not in _MODEL_PRESETS:
            raise ValueError(
                f"Unknown model preset '{name}'. "
                f"Available: {list(_MODEL_PRESETS.keys())}"
            )
        return cls(**_MODEL_PRESETS[name])


# -- modules ----------------------------------------------------------------
class RotaryPositionEncoding(nn.Module):
    """Rotary Position Embedding (RoPE) as a reusable module.

    Computes cos/sin tables once in __init__ (registered as non-persistent
    buffers so they are NOT saved in state_dict / checkpoint).
    """

    def __init__(self, dim: int, max_position_embeddings: int = 2048) -> None:
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings

        N = 10_000.0
        inv_freq = 1.0 / (N ** (torch.arange(0, dim, 2).float() / dim))
        inv_freq = torch.cat((inv_freq, inv_freq), dim=-1)
        position = torch.arange(max_position_embeddings).float()
        sinusoid_inp = torch.outer(position, inv_freq)
        self.register_buffer("cos", sinusoid_inp.cos(), persistent=False)
        self.register_buffer("sin", sinusoid_inp.sin(), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """Apply RoPE to tensor x.

        Args:
            x: (batch_size, seq_len, num_heads, head_dim)

        Returns:
            Tensor of same shape with rotary encoding applied.
        """
        batch_size, seq_len, num_heads, head_dim = x.shape
        device = x.device
        dtype = x.dtype
        # Slice and broadcast cos/sin to 4-D: (1, seq_len, 1, head_dim)
        cos = self.cos.to(device, dtype)[:seq_len].reshape(1, seq_len, 1, -1)
        sin = self.sin.to(device, dtype)[:seq_len].reshape(1, seq_len, 1, -1)
        x1, x2 = x.chunk(2, dim=-1)
        rotated = torch.cat((-x2, x1), dim=-1)
        return (x * cos) + (rotated * sin)


class LlamaAttention(nn.Module):
    """Grouped-Query Attention (GQA) with RoPE."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_heads = config.num_key_value_heads

        assert (self.head_dim * self.num_heads) == self.hidden_size

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

    def reset_parameters(self):
        self.q_proj.reset_parameters()
        self.k_proj.reset_parameters()
        self.v_proj.reset_parameters()
        self.o_proj.reset_parameters()

    def forward(
        self,
        hidden_states: Tensor,
        rope: RotaryPositionEncoding,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        bs, seq_len, _ = hidden_states.size()

        # Project to Q, K, V — use reshape (not view) for DTensor safety.
        query_states = self.q_proj(hidden_states).reshape(
            bs, seq_len, self.num_heads, self.head_dim
        )
        key_states = self.k_proj(hidden_states).reshape(
            bs, seq_len, self.num_kv_heads, self.head_dim
        )
        value_states = self.v_proj(hidden_states).reshape(
            bs, seq_len, self.num_kv_heads, self.head_dim
        )

        # Apply RoPE
        query_states = rope(query_states)
        key_states = rope(key_states)

        # Transpose to BHSD for SDPA
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Expand KV heads to match Q heads for GQA (manual repeat_kv).
        n_rep = self.num_heads // self.num_kv_heads
        if n_rep > 1:
            key_states = (
                key_states.unsqueeze(2)
                .expand(-1, -1, n_rep, -1, -1)
                .reshape(key_states.shape[0], self.num_heads, *key_states.shape[2:])
            )
            value_states = (
                value_states.unsqueeze(2)
                .expand(-1, -1, n_rep, -1, -1)
                .reshape(value_states.shape[0], self.num_heads, *value_states.shape[2:])
            )

        # SDPA
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attn_mask,
            dropout_p=0.0,
        )

        # BHSD → BSHD → (bs, seq, hidden)
        attn_output = attn_output.transpose(1, 2).reshape(bs, seq_len, self.hidden_size)
        return self.o_proj(attn_output)


class LlamaMLP(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def reset_parameters(self):
        self.gate_proj.reset_parameters()
        self.up_proj.reset_parameters()
        self.down_proj.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class LlamaDecoderLayer(nn.Module):
    """Single transformer decoder layer (pre-norm)."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=1e-5)
        self.self_attn = LlamaAttention(config)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=1e-5)
        self.mlp = LlamaMLP(config)

    def reset_parameters(self):
        self.input_layernorm.reset_parameters()
        self.self_attn.reset_parameters()
        self.post_attention_layernorm.reset_parameters()
        self.mlp.reset_parameters()

    def forward(
        self,
        hidden_states: Tensor,
        rope: RotaryPositionEncoding,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        # Self-attention block
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, rope=rope, attn_mask=attn_mask)
        hidden_states = hidden_states + residual

        # MLP block
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states) + residual
        return hidden_states


class LlamaModel(nn.Module):
    """Llama base model (no LM head)."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.rotary_emb = RotaryPositionEncoding(
            config.hidden_size // config.num_attention_heads,
            config.max_position_embeddings,
        )
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = nn.RMSNorm(config.hidden_size, eps=1e-5)

    def reset_parameters(self):
        self.embed_tokens.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()
        self.norm.reset_parameters()

    def forward(
        self, input_ids: Tensor, attn_mask: Tensor | None = None
    ) -> Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, rope=self.rotary_emb, attn_mask=attn_mask
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states


class LlamaForPretraining(nn.Module):
    """Llama model with LM head for causal language modeling."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.base_model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def reset_parameters(self):
        self.base_model.reset_parameters()
        self.lm_head.reset_parameters()

    def forward(
        self, input_ids: Tensor, attn_mask: Tensor | None = None
    ) -> Tensor:
        hidden_states = self.base_model(input_ids, attn_mask)
        return self.lm_head(hidden_states)
