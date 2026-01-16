from typing import Optional

import torch
from torch import nn as nn


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module as described in "Attention is All You Need" (Vaswani et al., 2017).

    This implementation splits the input into multiple heads, computes attention independently for each head,
    and then concatenates the results. This allows the model to focus on different parts of the input simultaneously.

    Args:
        hidden_size (int): Dimensionality of the input and output features.
        num_heads (int): Number of attention heads to use. Must divide hidden_size evenly.

    Attributes:
        num_heads (int): Number of attention heads.
        head_dim (int): Dimensionality of each attention head.
        q_proj (nn.Linear): Linear projection for query vectors.
        k_proj (nn.Linear): Linear projection for key vectors.
        v_proj (nn.Linear): Linear projection for value vectors.
        o_proj (nn.Linear): Linear projection for output vectors.
    """

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        assert hidden_size % num_heads == 0, 'hidden_size must be divisible by num_heads'

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Projection matrices for Q, K, V
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)

        # Output projection
        self.o_proj = nn.Linear(hidden_size, hidden_size)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                hidden_state: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the Multi-Head Attention module.

        Args:
            hidden_state (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_size).
            attention_mask (Optional[torch.Tensor]): Attention mask of shape (batch_size, 1, 1, seq_len)
                or (batch_size, 1, seq_len, seq_len). 1 indicates positions to attend to, 0 indicates positions to mask out.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_size).
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
        # Note that the scaling factor uses head_dim, not hidden_size.
        attention_scores = torch.matmul(query, key.transpose(
            -1, -2)) / torch.sqrt(
                torch.tensor(self.head_dim, dtype=torch.float32))

        # Apply attention mask if provided
        if attention_mask is not None:
            # If attention_mask is provided, it should have shape (batch_size, num_heads, seq_len, seq_len).
            assert attention_scores.size() == attention_mask.size(
            ), 'Attention mask size must match attention scores size. get {} and {}'.format(
                attention_scores.size(), attention_mask.size())
            attention_scores = torch.masked_fill(attention_scores,
                                                 attention_mask == 0,
                                                 float('-inf'))

        # Softmax to get attention probabilities
        attention_probs = torch.softmax(attention_scores, dim=-1)

        # Dropout on attention weights
        attention_probs = self.dropout(attention_probs)

        # Weighted sum of values
        # Multiply attention weights with V:
        # (batch_size, num_heads, seq_len, seq_len) * (batch_size, num_heads, seq_len, head_dim)
        # -> (batch_size, num_heads, seq_len, head_dim)
        output = torch.matmul(attention_probs, value)

        # Reshape and apply output projection
        # Transpose back: (batch_size, num_heads, seq_len, head_dim)
        # -> (batch_size, seq_len, num_heads, head_dim)
        # -> (batch_size, seq_len, hidden_size)
        #
        # Note: The transpose operation changes the dimension ordering but does not change the memory layout,
        # resulting in a non-contiguous tensor. The contiguous() method makes the tensor contiguous in memory,
        # allowing subsequent view or reshape operations without error.
        output = output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.head_dim * self.num_heads)
        output = self.o_proj(output)

        return output

    def split_head(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split the input tensor into multiple attention heads.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_size).

        Returns:
            torch.Tensor: Tensor of shape (batch_size, num_heads, seq_len, head_dim).
        """
        # Reshaping from (batch_size, seq_len, hidden_size) to (batch_size, seq_len, num_heads, head_dim)
        # Then transpose to (batch_size, num_heads, seq_len, head_dim)
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads,
                      self.head_dim).transpose(1, 2)
