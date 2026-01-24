from typing import Optional

import torch
from torch import nn

from .base import BaseAttention


class MultiHeadAttention(BaseAttention):
    """
    Multi-Head Attention module as described in "Attention is All You Need" (Vaswani et al., 2017).

    This implementation splits the input into multiple heads, computes attention independently for each head,
    and then concatenates the results. This allows the model to focus on different parts of the input simultaneously.

    Args:
        hidden_size (int): Dimensionality of the input and output features.
        num_heads (int): Number of attention heads to use. Must divide hidden_size evenly.
        dropout (float, optional): Dropout probability for attention weights. Defaults to 0.1.
        bias (bool, optional): Whether to use bias in linear projections. Defaults to True.

    Attributes:
        num_heads (int): Number of attention heads.
        head_dim (int): Dimensionality of each attention head.
        scale_factor (torch.Tensor): Scaling factor for dot-product attention.
        q_proj (nn.Linear): Linear projection for query vectors.
        k_proj (nn.Linear): Linear projection for key vectors.
        v_proj (nn.Linear): Linear projection for value vectors.
        o_proj (nn.Linear): Linear projection for output vectors.
        dropout (nn.Dropout): Dropout layer for attention weights.
    """

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 dropout: float = 0.1,
                 bias: bool = True) -> None:
        super().__init__(hidden_size, num_heads, dropout, bias)

        # Projection matrices for Q, K, V
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

        # Output projection
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize parameters using Xavier uniform initialization."""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass of the Multi-Head Attention module.

        Args:
            hidden_state (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_size).
            attention_mask (Optional[torch.Tensor]): Attention mask of shape (batch_size, 1, 1, seq_len)
                or (batch_size, 1, seq_len, seq_len). 1 indicates positions to attend to, 0 indicates positions to mask out.
            return_attention_weights (bool): Whether to return attention weights along with the output. Defaults to False.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                Output tensor of shape (batch_size, seq_len, hidden_size).
                If return_attention_weights is True, returns a tuple (output, attention_weights).
        """
        batch_size, seq_len, _ = hidden_state.size()

        # Linear projections
        # query, key, value each has shape: (batch_size, seq_len, hidden_size)
        query = self.q_proj(hidden_state)
        key = self.k_proj(hidden_state)
        value = self.v_proj(hidden_state)

        # Split into multiple heads
        query = self.split_head(query)
        key = self.split_head(key)
        value = self.split_head(value)

        # Compute scaled dot-product attention
        # Matrix multiplication: (batch_size, num_heads, seq_len, head_dim) * (batch_size, num_heads, head_dim, seq_len)
        # Resulting shape: (batch_size, num_heads, seq_len, seq_len)
        attention_scores = (torch.matmul(query, key.transpose(-1, -2)) *
                            self.scale_factor)

        # Apply attention mask if provided
        if attention_mask is not None:
            # Ensure mask has correct shape
            expected_mask_shape = (batch_size, self.num_heads, seq_len,
                                   seq_len)
            assert attention_mask.size() == expected_mask_shape, (
                f'Attention mask size must match {expected_mask_shape}, got {attention_mask.size()}'
            )

            attention_scores = torch.masked_fill(attention_scores,
                                                 attention_mask == 0,
                                                 float('-inf'))

        # Softmax to get attention weights
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # Dropout on attention weights
        attention_weights = self.dropout(attention_weights)

        # Weighted sum of values
        # Multiply attention weights with V:
        # (batch_size, num_heads, seq_len, seq_len) * (batch_size, num_heads, seq_len, head_dim)
        # -> (batch_size, num_heads, seq_len, head_dim)
        output = torch.matmul(attention_weights, value)

        # Reshape and apply output projection
        # Transpose back: (batch_size, num_heads, seq_len, head_dim)
        # -> (batch_size, seq_len, num_heads, head_dim)
        # -> (batch_size, seq_len, hidden_size)
        output = (output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size))
        output = self.o_proj(output)

        if return_attention_weights:
            return output, attention_weights
        return output

    def split_head(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split the input tensor into multiple attention heads.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_size).

        Returns:
            torch.Tensor: Tensor of shape (batch_size, num_heads, seq_len, head_dim).
        """
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads,
                      self.head_dim).transpose(1, 2)

    def extra_repr(self) -> str:
        """Return a string representation of the module's extra information."""
        return (f'hidden_size={self.hidden_size}, num_heads={self.num_heads}, '
                f'head_dim={self.head_dim}, bias={self.bias}')
