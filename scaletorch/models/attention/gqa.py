from typing import Optional

import torch
from torch import nn


class GroupQueryAttention(nn.Module):
    """
    Group Query Attention module as described in "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (Chen et al., 2023).

    This implementation is a hybrid between Multi-Head Attention and Multi-Query Attention, where queries are grouped and each group shares a single key and value head.
    This reduces memory usage and speeds up inference while maintaining performance closer to Multi-Head Attention than Multi-Query Attention.

    By configuring `group_num` (G, the number of groups), this module supports:
        - When group_num == num_heads: Multi-Head Attention (MHA)
        - When group_num == 1: Multi-Query Attention (MQA)
        - When 1 < group_num < num_heads: Generic Grouped Query Attention (GQA)

    Args:
        hidden_size (int): Dimensionality of the input and output features.
        num_heads (int): Number of query heads to use. Must divide hidden_size evenly.
        group_num (int): Number of groups to divide query heads into. Must divide num_heads evenly.

    Attributes:
        num_heads (int): Number of query heads.
        head_dim (int): Dimensionality of each attention head.
        group_num (int): Number of groups.
        q_proj (nn.Linear): Linear projection for query vectors.
        k_proj (nn.Linear): Linear projection for key vectors (one per group).
        v_proj (nn.Linear): Linear projection for value vectors (one per group).
        o_proj (nn.Linear): Linear projection for output vectors.
    """

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 group_num: int,
                 dropout: float = 0.1):
        super(GroupQueryAttention, self).__init__()
        assert hidden_size % num_heads == 0, 'hidden_size must be divisible by num_heads'
        assert num_heads % group_num == 0, 'num_heads must be divisible by group_num'

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.group_num = group_num

        # Number of query heads per group
        self.q_heads_per_group = num_heads // group_num

        # Linear projections for queries, keys, and values
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, self.group_num * self.head_dim)
        self.v_proj = nn.Linear(hidden_size, self.group_num * self.head_dim)

        # Output projection
        self.o_proj = nn.Linear(hidden_size, hidden_size)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                hidden_state: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the Group Query Attention module.

        Args:
            hidden_state (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_size).
            attention_mask (Optional[torch.Tensor]): Attention mask of shape (batch_size, 1, 1, seq_len)
                or (batch_size, 1, seq_len, seq_len). 1 indicates positions to attend to, 0 indicates positions to mask out.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_size).
        """
        batch_size = hidden_state.size()[0]

        # Linear projections
        query = self.q_proj(hidden_state)  # (batch_size, seq_len, hidden_size)
        key = self.k_proj(
            hidden_state)  # (batch_size, seq_len, group_num * head_dim)
        value = self.v_proj(
            hidden_state)  # (batch_size, seq_len, group_num * head_dim)

        # Split into heads: multiple heads for queries, one per group for keys and values
        query = self.split_head(query)
        key = self.split_head(key, self.group_num)
        value = self.split_head(value, self.group_num)

        # Compute scaled dot-product attention
        attention_scores = torch.matmul(query, key.transpose(
            -1, -2)) / torch.sqrt(
                torch.tensor(self.head_dim, dtype=torch.float32))

        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = torch.masked_fill(attention_scores,
                                                 attention_mask == 0,
                                                 float('-inf'))

        # Softmax to get attention probabilities
        attention_probs = torch.softmax(attention_scores, dim=-1)

        # Apply dropout to attention weights
        attention_probs = self.dropout(attention_probs)

        # Weighted sum of values
        # (batch_size, group_num, q_heads_per_group, seq_len, seq_len) * (batch_size, group_num, 1, seq_len, head_dim)
        # -> (batch_size, group_num, q_heads_per_group, seq_len, head_dim)
        output = torch.matmul(attention_probs, value)

        # Reshape and apply output projection
        # (batch_size, group_num, q_heads_per_group, seq_len, head_dim)
        # -> (batch_size, num_heads, seq_len, head_dim)
        # -> (batch_size, seq_len, num_heads, head_dim) -> (batch_size, seq_len, hidden_dim)
        output = output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.head_dim * self.num_heads)
        output = self.o_proj(output)

        return output

    def split_head(self,
                   x: torch.Tensor,
                   group_num: Optional[int] = None) -> torch.Tensor:
        """
        Split the input tensor into multiple attention heads.

        For queries: splits into self.num_heads heads.
        For keys and values: splits into group_num heads, then expands each to serve multiple query heads in the same group.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_size) or (batch_size, seq_len, group_num * head_dim).
            group_num (Optional[int]): Number of groups to split into. If None, uses self.num_heads.

        Returns:
            torch.Tensor: Tensor of shape (batch_size, num_heads, seq_len, head_dim).
        """
        batch_size, seq_len = x.size()[:2]

        if group_num is None:
            # Split queries into multiple heads
            return x.view(batch_size, -1, self.num_heads,
                          self.head_dim).transpose(1, 2)
        else:
            # Split keys/values into groups, then expand to match query heads
            x = x.view(batch_size, -1, group_num,
                       self.head_dim).transpose(1, 2)
            # Expand each group's key/value to serve multiple query heads
            x = x[:, :, None, :, :].expand(
                batch_size, group_num, self.num_heads // group_num, seq_len,
                self.head_dim).reshape(batch_size,
                                       self.num_heads // group_num * group_num,
                                       seq_len, self.head_dim)
            return x
