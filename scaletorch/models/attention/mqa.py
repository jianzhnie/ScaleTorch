from typing import Optional

import torch
from torch import nn


class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention module as described in "Fast Transformer Decoding: One Write-Head is All You Need" (Shazeer, 2019).

    This implementation uses a single key and value head for all query heads, which reduces memory usage and speeds up inference
    compared to standard Multi-Head Attention.

    Args:
        hidden_size (int): Dimensionality of the input and output features.
        num_heads (int): Number of query heads to use. Must divide hidden_size evenly.
        dropout (float, optional): Dropout probability for attention weights. Defaults to 0.1.
        bias (bool, optional): Whether to use bias in linear projections. Defaults to True.

    Attributes:
        num_heads (int): Number of query heads.
        head_dim (int): Dimensionality of each attention head.
        scale_factor (torch.Tensor): Scaling factor for dot-product attention.
        q_proj (nn.Linear): Linear projection for query vectors.
        k_proj (nn.Linear): Linear projection for key vectors (single head).
        v_proj (nn.Linear): Linear projection for value vectors (single head).
        o_proj (nn.Linear): Linear projection for output vectors.
        dropout (nn.Dropout): Dropout layer for attention weights.
    """

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 dropout: float = 0.1,
                 bias: bool = True) -> None:
        super().__init__()
        assert hidden_size % num_heads == 0, f'hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})'

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size
        self.bias = bias

        # Scaling factor for attention scores (pre-compute for efficiency)
        self.scale_factor = 1.0 / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32))

        # Projection matrices: multiple heads for queries, single head for keys and values
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, self.head_dim, bias=bias)

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

    def forward(self,
                hidden_state: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                return_attention_weights: bool = False) -> torch.Tensor:
        """
        Forward pass of the Multi-Query Attention module.

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
        query = self.q_proj(hidden_state)
        key = self.k_proj(hidden_state)
        value = self.v_proj(hidden_state)

        # Split into heads: multiple heads for queries, single head for keys and values
        query = self.split_head(query)
        key = self.split_head(key, head_num=1)
        value = self.split_head(value, head_num=1)

        # Compute scaled dot-product attention
        # (batch_size, num_heads, seq_len, head_dim) * (batch_size, 1, head_dim, seq_len)
        # -> (batch_size, num_heads, seq_len, seq_len)
        attention_scores = torch.matmul(query, key.transpose(
            -1, -2)) * self.scale_factor

        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = torch.masked_fill(attention_scores,
                                                 attention_mask == 0,
                                                 float('-inf'))

        # Softmax to get attention probabilities
        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Weighted sum of values
        # (batch_size, num_heads, seq_len, seq_len) * (batch_size, 1, seq_len, head_dim)
        # -> (batch_size, num_heads, seq_len, head_dim)
        output = torch.matmul(attention_probs, value)

        # Reshape and apply output projection
        output = output.transpose(1,
                                  2).contiguous().view(batch_size, seq_len,
                                                       self.hidden_size)
        output = self.o_proj(output)

        if return_attention_weights:
            return output, attention_probs
        return output

    def split_head(self,
                   x: torch.Tensor,
                   head_num: Optional[int] = None) -> torch.Tensor:
        """
        Split the input tensor into multiple attention heads.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_size) or (batch_size, seq_len, head_dim).
            head_num (Optional[int]): Number of heads to split into. If None, uses self.num_heads.

        Returns:
            torch.Tensor: Tensor of shape (batch_size, num_heads, seq_len, head_dim).
        """
        batch_size, seq_len, hidden_size = x.size()
        current_num_heads = head_num or self.num_heads

        return x.view(batch_size, seq_len, current_num_heads,
                      self.head_dim).transpose(1, 2)

    def extra_repr(self) -> str:
        """Return a string representation of the module's extra information."""
        return (f'hidden_size={self.hidden_size}, num_heads={self.num_heads}, '
                f'head_dim={self.head_dim}, bias={self.bias}')
