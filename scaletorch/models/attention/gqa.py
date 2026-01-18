from typing import Optional

import torch
from torch import nn


class GroupQueryAttention(nn.Module):
    """
    Group Query Attention module as described in "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (Chen et al., 2023).

    This implementation is a hybrid between Multi-Head Attention and Multi-Query Attention, where queries are grouped
    and each group shares a single key and value head. This reduces memory usage and speeds up inference while
    maintaining performance closer to Multi-Head Attention than Multi-Query Attention.

    By configuring `num_kv_groups` (G, the number of groups), this module supports:
        - When num_kv_groups == num_heads: Multi-Head Attention (MHA)
        - When num_kv_groups == 1: Multi-Query Attention (MQA)
        - When 1 < num_kv_groups < num_heads: Generic Grouped Query Attention (GQA)

    Args:
        hidden_size (int): Dimensionality of the input and output features.
        num_heads (int): Number of query heads to use. Must divide hidden_size evenly.
        num_kv_groups (int): Number of groups to divide query heads into. Must divide num_heads evenly.
        dropout (float, optional): Dropout probability for attention weights. Defaults to 0.1.
        bias (bool, optional): Whether to use bias in linear projections. Defaults to True.

    Attributes:
        num_heads (int): Number of query heads.
        head_dim (int): Dimensionality of each attention head.
        num_kv_groups (int): Number of groups.
        heads_per_group (int): Number of heads per group.
        scale_factor (torch.Tensor): Scaling factor for dot-product attention.
        q_proj (nn.Linear): Linear projection for query vectors.
        k_proj (nn.Linear): Linear projection for key vectors (one per group).
        v_proj (nn.Linear): Linear projection for value vectors (one per group).
        o_proj (nn.Linear): Linear projection for output vectors.
        dropout (nn.Dropout): Dropout layer for attention weights.
    """

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 num_kv_groups: int,
                 dropout: float = 0.1,
                 bias: bool = True) -> None:
        super().__init__()
        assert hidden_size % num_heads == 0, f'hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})'
        assert num_heads % num_kv_groups == 0, f'num_heads ({num_heads}) must be divisible by num_kv_groups ({num_kv_groups})'

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.num_kv_groups = num_kv_groups
        self.hidden_size = hidden_size
        self.bias = bias

        # Number of heads per group
        self.heads_per_group = num_heads // num_kv_groups

        # Scaling factor for attention scores (pre-compute for efficiency)
        self.scale_factor = 1.0 / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32))

        # Linear projections for queries, keys, and values
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size,
                                self.num_kv_groups * self.head_dim,
                                bias=bias)
        self.v_proj = nn.Linear(hidden_size,
                                self.num_kv_groups * self.head_dim,
                                bias=bias)

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
        Forward pass of the Group Query Attention module.

        Args:
            hidden_state (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_size).
            attention_mask (Optional[torch.Tensor]): Attention mask of shape (batch_size, 1, 1, seq_len)
                or (batch_size, 1, seq_len, seq_len). 1 indicates positions to attend to, 0 indicates positions to mask out.
            return_attention_weights (bool, optional): Whether to return attention weights. Defaults to False.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_size).
                If return_attention_weights is True, returns a tuple (output, attention_weights).
        """
        batch_size, seq_len, _ = hidden_state.size()

        # Linear projections
        # query has shape (batch_size, seq_len, hidden_size)
        query = self.q_proj(hidden_state)
        # key and value have shape (batch_size, seq_len, num_kv_groups * head_dim)
        key = self.k_proj(hidden_state)
        value = self.v_proj(hidden_state)

        # Split into heads and expand key/value for grouped attention
        query = self.split_head(query)
        key = self.split_head_grouped(key)
        value = self.split_head_grouped(value)

        # Compute scaled dot-product attention
        # (batch_size, num_heads, seq_len, head_dim) * (batch_size, num_heads, head_dim, seq_len)
        # -> (batch_size, num_heads, seq_len, seq_len)
        attention_scores = torch.matmul(query, key.transpose(
            -1, -2)) * self.scale_factor

        # Apply attention mask if provided
        if attention_mask is not None:
            # Ensure mask has correct shape
            expected_mask_shape = (batch_size, self.num_heads, seq_len,
                                   seq_len)
            assert attention_mask.size() == expected_mask_shape, \
                f'Attention mask size must match {expected_mask_shape}, got {attention_mask.size()}'
            attention_scores = torch.masked_fill(attention_scores,
                                                 attention_mask == 0,
                                                 float('-inf'))

        # Softmax to get attention weights
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # Apply dropout to attention weights
        attention_weights = self.dropout(attention_weights)

        # Weighted sum of values
        # (batch_size, num_heads, seq_len, seq_len) * (batch_size, num_heads, seq_len, head_dim)
        # -> (batch_size, num_heads, seq_len, head_dim)
        output = torch.matmul(attention_weights, value)

        # Reshape and apply output projection
        output = output.transpose(1,
                                  2).contiguous().view(batch_size, seq_len,
                                                       self.hidden_size)
        output = self.o_proj(output)

        if return_attention_weights:
            return output, attention_weights
        return output

    def split_head(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split the input tensor into multiple attention heads for queries.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_size).

        Returns:
            torch.Tensor: Tensor of shape (batch_size, num_heads, seq_len, head_dim).
        """
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads,
                      self.head_dim).transpose(1, 2)

    def split_head_grouped(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split the input tensor into grouped attention heads for keys and values.

        This method splits keys/values into groups, then expands each to serve multiple query heads in the same group.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, num_kv_groups * head_dim).

        Returns:
            torch.Tensor: Tensor of shape (batch_size, num_heads, seq_len, head_dim).
        """
        batch_size, seq_len, _ = x.size()

        # Split into groups: (batch_size, seq_len, num_kv_groups * head_dim) -> (batch_size, num_kv_groups, seq_len, head_dim)
        x = x.view(batch_size, seq_len, self.num_kv_groups,
                   self.head_dim).transpose(1, 2)

        # Expand each group's key/value to serve multiple query heads
        # (batch_size, num_kv_groups, seq_len, head_dim) -> (batch_size, num_kv_groups, heads_per_group, seq_len, head_dim)
        x = x.unsqueeze(2).expand(batch_size, self.num_kv_groups,
                                  self.heads_per_group, seq_len, self.head_dim)

        # Reshape to match query heads: (batch_size, num_heads, seq_len, head_dim)
        x = x.reshape(batch_size, self.num_heads, seq_len, self.head_dim)

        return x

    def extra_repr(self) -> str:
        """Return a string representation of the module's extra information."""
        return (
            f'hidden_size={self.hidden_size}, num_heads={self.num_heads}, '
            f'num_kv_groups={self.num_kv_groups}, head_dim={self.head_dim}, '
            f'heads_per_group={self.heads_per_group}, bias={self.bias}')
