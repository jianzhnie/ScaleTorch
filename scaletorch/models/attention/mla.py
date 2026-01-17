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
                 latent_size: int,
                 dropout: float = 0.0) -> None:
        super().__init__()
        assert hidden_size % num_heads == 0, f'hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})'

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.latent_size = latent_size
        self.hidden_size = hidden_size

        # Scaling factor for attention scores (pre-compute for efficiency)
        self.scale_factor = 1.0 / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32))

        # Projection matrices for Q, K, V (operating on latent space)
        self.q_proj = nn.Linear(latent_size, hidden_size)
        self.k_proj = nn.Linear(latent_size, hidden_size)
        self.v_proj = nn.Linear(latent_size, hidden_size)

        # Latent space projection
        self.latent_proj = nn.Linear(hidden_size, latent_size)

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

        # Project to latent space
        latent_state = self.latent_proj(hidden_state)

        # Linear projections for Q, K, V (operating on latent space)
        query = self.q_proj(latent_state)
        key = self.k_proj(latent_state)
        value = self.v_proj(latent_state)

        # Split into multiple heads
        query = self.split_head(query)
        key = self.split_head(key)
        value = self.split_head(value)

        # Compute scaled dot-product attention
        attention_scores = torch.matmul(query, key.transpose(
            -1, -2)) * self.scale_factor

        # Apply attention mask if provided
        if attention_mask is not None:
            # Using additive mask approach (alternative to masked_fill)
            attention_scores += attention_mask * -1e-9

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
