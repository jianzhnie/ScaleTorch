from typing import Optional

import torch
from torch import nn


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-head Latent Attention module that operates on latent space representations.

    This implementation extends the standard multi-head attention mechanism to work with
    latent space representations, allowing for more flexible and powerful attention patterns.

    Args:
        hidden_size (int): Dimensionality of the input and output features.
        num_heads (int): Number of attention heads to use. Must divide hidden_size evenly.
        latent_size (int): Dimensionality of the latent space.
        dropout (float, optional): Dropout probability for attention weights. Defaults to 0.0.

    Attributes:
        num_heads (int): Number of attention heads.
        head_dim (int): Dimensionality of each attention head.
        latent_size (int): Dimensionality of the latent space.
        scale_factor (torch.Tensor): Scaling factor for dot-product attention.
        q_proj (nn.Linear): Linear projection for query vectors.
        k_proj (nn.Linear): Linear projection for key vectors.
        v_proj (nn.Linear): Linear projection for value vectors.
        latent_proj (nn.Linear): Linear projection to latent space.
        output_proj (nn.Linear): Linear projection for output vectors.
        dropout (nn.Dropout): Dropout layer for attention weights.
    """

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 q_latent_size: int,
                 kv_latent_size: int,
                 dropout: float = 0.0,
                 bias: bool = True) -> None:
        super().__init__()
        assert hidden_size % num_heads == 0, f'hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})'

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_latent_size = q_latent_size
        self.kv_latent_size = kv_latent_size
        self.hidden_size = hidden_size

        # Scaling factor for attention scores (pre-compute for efficiency)
        self.scale_factor = 1.0 / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32))

        # Projection matrices for Q, K, V (operating on latent space)
        self.q_down_proj = nn.Linear(hidden_size, q_latent_size, bias=bias)
        self.q_up_proj = nn.Linear(q_latent_size, hidden_size, bias=bias)
        self.kv_down_proj = nn.Linear(self.hidden_size,
                                      kv_latent_size,
                                      bias=bias)
        self.k_up_proj = nn.Linear(kv_latent_size, hidden_size, bias=bias)
        self.v_up_proj = nn.Linear(kv_latent_size, hidden_size, bias=bias)

        # Output projection
        self.output_proj = nn.Linear(hidden_size, hidden_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                hidden_state: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the Multi-head Latent Attention module.

        Args:
            hidden_state (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_size).
            attention_mask (Optional[torch.Tensor]): Attention mask of shape (batch_size, 1, 1, seq_len)
                or (batch_size, 1, seq_len, seq_len). 1 indicates positions to attend to, 0 indicates positions to mask out.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_size).
        """
        batch_size, seq_len, _ = hidden_state.size()

        # Query projection
        query_latent = self.q_down_proj(hidden_state)
        query = self.q_up_proj(query_latent)

        # Down-project to latent space
        latent_state = self.kv_down_proj(hidden_state)

        # Key and Value projections
        key = self.k_up_proj(latent_state)
        value = self.v_up_proj(latent_state)

        # Split into multiple heads
        query = self.split_head(query)
        key = self.split_head(key)
        value = self.split_head(value)

        # Compute scaled dot-product attention
        attention_scores = torch.matmul(query, key.transpose(
            -1, -2)) * self.scale_factor

        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = torch.masked_fill(attention_scores,
                                                 attention_mask == 0,
                                                 float('-inf'))

        # Softmax to get attention weights
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # Apply dropout
        attention_weights = self.dropout(attention_weights)

        # Weighted sum of values
        output = torch.matmul(attention_weights, value)

        # Reshape and apply output projection
        output = output.transpose(1,
                                  2).contiguous().view(batch_size, seq_len,
                                                       self.hidden_size)
        output = self.output_proj(output)

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
