"""Model implementations with parallelism support.

Models:
    - Llama: Transformer decoder with flash attention, TP, CP, and RoPE
    - LeNet: Simple CNN for educational purposes
    - MoE: Mixture-of-Experts transformer components
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
    apply_rotary_pos_emb,
    flash_attention,
    get_cos_sin,
)

__all__ = [
    "MLP",
    "DecoderLayer",
    "FinalProjection",
    "FusedRMSNorm",
    "Llama",
    "LlamaAttention",
    "LlamaEmbedding",
    "LlamaRMSNorm",
    "apply_rotary_pos_emb",
    "flash_attention",
    "get_cos_sin",
]
