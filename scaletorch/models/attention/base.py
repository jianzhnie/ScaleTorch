"""Base attention module providing common functionality for all attention mechanisms."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


class BaseAttention(nn.Module, ABC):
    """Abstract base class for attention mechanisms.

    This class provides common functionality and interface for all attention
    implementations, ensuring consistency across different attention types.
    """

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 dropout: float = 0.0,
                 bias: bool = True) -> None:
        """Initialize base attention parameters.

        Args:
            hidden_size: Dimensionality of input and output features
            num_heads: Number of attention heads
            dropout: Dropout probability for attention weights
            bias: Whether to use bias in linear projections

        Raises:
            ValueError: If hidden_size is not divisible by num_heads
        """
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError(
                f'hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})'
            )

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout_prob = dropout
        self.bias = bias

        # Pre-compute scaling factor for efficiency
        self.scale_factor = 1.0 / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32))

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    @abstractmethod
    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass for attention mechanism.

        Args:
            hidden_state: Input tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Optional attention mask
            return_attention_weights: Whether to return attention weights

        Returns:
            Output tensor and optionally attention weights
        """
        pass

    def split_head(self, x: torch.Tensor) -> torch.Tensor:
        """Split input tensor into multiple attention heads.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)

        Returns:
            Tensor of shape (batch_size, num_heads, seq_len, head_dim)
        """
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads,
                      self.head_dim).transpose(1, 2)

    def combine_head(self, x: torch.Tensor) -> torch.Tensor:
        """Combine multiple attention heads into single tensor.

        Args:
            x: Input tensor of shape (batch_size, num_heads, seq_len, head_dim)

        Returns:
            Tensor of shape (batch_size, seq_len, hidden_size)
        """
        batch_size, _, seq_len, _ = x.size()
        return (x.transpose(1, 2).contiguous().view(batch_size, seq_len,
                                                    self.hidden_size))

    def apply_attention_mask(
            self, attention_scores: torch.Tensor,
            attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Apply attention mask to scores.

        Args:
            attention_scores: Attention scores tensor
            attention_mask: Optional attention mask

        Returns:
            Masked attention scores
        """
        if attention_mask is not None:
            attention_scores = torch.masked_fill(attention_scores,
                                                 attention_mask == 0,
                                                 float('-inf'))
        return attention_scores

    def compute_attention_weights(
        self,
        attention_scores: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute attention weights from scores.

        Args:
            attention_scores: Raw attention scores
            attention_mask: Optional attention mask

        Returns:
            Normalized attention weights
        """
        attention_scores = self.apply_attention_mask(attention_scores,
                                                     attention_mask)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        return self.dropout(attention_weights)

    def extra_repr(self) -> str:
        """Return string representation of module parameters."""
        return (f'hidden_size={self.hidden_size}, num_heads={self.num_heads}, '
                f'head_dim={self.head_dim}, bias={self.bias}')


def validate_attention_inputs(hidden_state: torch.Tensor,
                              attention_mask: Optional[torch.Tensor],
                              num_heads: int) -> Tuple[int, int]:
    """Validate attention input tensors.

    Args:
        hidden_state: Input hidden state tensor
        attention_mask: Optional attention mask
        num_heads: Number of attention heads

    Returns:
        Tuple of (batch_size, seq_len)

    Raises:
        ValueError: If input tensors have invalid shapes
    """
    if hidden_state.dim() != 3:
        raise ValueError(f'hidden_state must be 3D, got {hidden_state.dim()}D')

    batch_size, seq_len, hidden_size = hidden_state.size()

    if attention_mask is not None:
        if attention_mask.dim() not in [3, 4]:
            raise ValueError(
                f'attention_mask must be 3D or 4D, got {attention_mask.dim()}D'
            )

        if attention_mask.size(0) != batch_size:
            raise ValueError(
                f'attention_mask batch size {attention_mask.size(0)} '
                f'must match hidden_state batch size {batch_size}')

    return batch_size, seq_len
