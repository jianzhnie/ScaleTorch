"""Model implementations with parallelism support.

Models:
    - Llama: Transformer decoder with flash attention, TP, CP, and RoPE
    - Qwen3: Qwen3 transformer with QK norms and explicit head_dim
    - Qwen3MoE: Qwen3 Mixture-of-Experts with Expert Parallelism
    - GPT (MoE): GPT-style transformer with MoE support
    - LeNet: Simple CNN for educational purposes
"""

from scaletorch.models.llama import (
    MLP,
    DecoderLayer,
    FinalProjection,
    FusedRMSNorm,
    Llama,
    LlamaAttention,
    LlamaEmbedding,
    LlamaRMSNorm,
    RMSNorm,
    apply_rotary_pos_emb,
    flash_attention,
    get_attention_backend,
    get_cos_sin,
    register_attention_backend,
)
from scaletorch.models.model_qwen3 import Qwen3
from scaletorch.models.model_qwen3_moe import Qwen3MoE
from scaletorch.models.moe import GPT, GPTConfig

__all__ = [
    "MLP",
    "DecoderLayer",
    "FinalProjection",
    "FusedRMSNorm",
    "GPT",
    "GPTConfig",
    "Llama",
    "LlamaAttention",
    "LlamaEmbedding",
    "LlamaRMSNorm",
    "Qwen3",
    "Qwen3MoE",
    "RMSNorm",
    "apply_rotary_pos_emb",
    "flash_attention",
    "get_attention_backend",
    "get_cos_sin",
    "register_attention_backend",
]
