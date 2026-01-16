"""
Attention modules for ScaleTorch.

This module provides implementations of various attention mechanisms:
- MultiHeadAttention: Standard multi-head attention from "Attention is All You Need"
- MultiQueryAttention: Memory-efficient attention with single key/value head
- GroupQueryAttention: Hybrid attention with grouped key/value heads
- MultiHeadLatentAttention: Multi-head attention operating on latent space representations
"""

from .gqa import GroupQueryAttention
from .mha import MultiHeadAttention
from .mla import MultiHeadLatentAttention
from .mqa import MultiQueryAttention

__all__ = [
    'MultiHeadAttention',
    'MultiQueryAttention',
    'GroupQueryAttention',
    'MultiHeadLatentAttention',
]
