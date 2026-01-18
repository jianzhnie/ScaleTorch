"""
ScaleTorch Attention Modules

This package provides various attention mechanisms for transformer models:
- MultiHeadAttention: Standard multi-head attention from "Attention is All You Need"
- MultiQueryAttention: Multi-query attention for efficient inference
- GroupQueryAttention: Grouped query attention balancing quality and efficiency
- MultiHeadLatentAttention: Attention operating on latent space representations

All modules support:
- Configurable dropout rates
- Optional bias terms
- Attention weight return functionality
- Proper parameter initialization
- Memory-efficient implementations
"""

from .gqa import GroupQueryAttention
from .mha import MultiHeadAttention
from .mla import MultiHeadLatentAttention
from .mqa import MultiQueryAttention

__all__ = [
    'MultiHeadAttention', 'MultiQueryAttention', 'GroupQueryAttention',
    'MultiHeadLatentAttention'
]
