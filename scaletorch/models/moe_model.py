"""
Enhanced GPT Language Model with Mixture of Experts (MoE) implementation.

This module provides a complete implementation of a GPT-style transformer with
Mixture of Experts layers, featuring improved type safety, documentation, and
code organization.

References:
1. GPT-2: https://github.com/openai/gpt-2/blob/master/src/model.py
2. HuggingFace GPT-2: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
3. OpenMoE: https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/moe/experts.py
4. ST-MoE: https://arxiv.org/abs/2202.08906
5. Switch Transformer: https://arxiv.org/abs/2101.03961
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

from scaletorch.utils import get_logger

# Configure logging
logger = get_logger(__name__)


@dataclass
class GPTConfig:
    """Configuration class for GPT model with MoE support.

    Attributes:
        block_size: Maximum sequence length the model can process
        vocab_size: Size of the vocabulary (default: 50304, padded from 50257)
        n_layer: Number of transformer layers
        n_head: Number of attention heads
        n_embd: Embedding dimension size
        dropout: Dropout probability (default: 0.0)
        bias: Whether to use bias in Linear and LayerNorm layers (default: True)
        use_moe: Whether to use Mixture of Experts layers (default: False)
        moe_layers: Specific layer indices to use MoE (if None, uses default pattern)
        n_experts: Number of experts in MoE layers (default: 8)
        top_k: Number of experts to route each token to (default: 2)
        capacity_factor: Expert capacity multiplier (default: 1.25)
        use_noisy_top_k: Whether to add noise to routing (default: True)
        use_aux_loss: Whether to use auxiliary loss for load balancing (default: False)
        use_router_z_loss: Whether to use router z-loss for stability (default: False)
        aux_loss_weight: Weight for auxiliary loss (default: 0.01)
        router_z_loss_weight: Weight for router z-loss (default: 0.01)
        train_capacity: Capacity factor during training (default: 1.25)
        eval_capacity: Capacity factor during evaluation (default: 1.25)
        min_capacity: Minimum expert capacity (default: 0.0)
        use_switch_tfm_init: Whether to use Switch Transformer initialization (default: False)
        switch_tfm_init_scale: Scale factor for Switch Transformer initialization (default: 1.0)
    """
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    use_moe: bool = False
    moe_layers: Optional[List[int]] = None
    n_experts: int = 8
    top_k: int = 2
    capacity_factor: float = 1.25
    use_aux_loss: bool = False
    use_router_z_loss: bool = False
    use_noisy_top_k: bool = True
    aux_loss_weight: float = 0.01
    router_z_loss_weight: float = 0.01
    train_capacity: float = 1.25
    eval_capacity: float = 1.25
    min_capacity: float = 0.0
    use_switch_tfm_init: bool = False
    switch_tfm_init_scale: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.block_size <= 0:
            raise ValueError(
                f'block_size must be positive, got {self.block_size}')
        if self.vocab_size <= 0:
            raise ValueError(
                f'vocab_size must be positive, got {self.vocab_size}')
        if self.n_layer <= 0:
            raise ValueError(f'n_layer must be positive, got {self.n_layer}')
        if self.n_head <= 0:
            raise ValueError(f'n_head must be positive, got {self.n_head}')
        if self.n_embd <= 0:
            raise ValueError(f'n_embd must be positive, got {self.n_embd}')
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f'n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})'
            )
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError(f'dropout must be in [0, 1], got {self.dropout}')
        if self.use_moe:
            if self.n_experts <= 0:
                raise ValueError(
                    f'n_experts must be positive, got {self.n_experts}')
            if not 1 <= self.top_k <= self.n_experts:
                raise ValueError(
                    f'top_k ({self.top_k}) must be between 1 and n_experts ({self.n_experts})'
                )
            if self.capacity_factor <= 0:
                raise ValueError(
                    f'capacity_factor must be positive, got {self.capacity_factor}'
                )
        if self.switch_tfm_init_scale <= 0:
            raise ValueError(
                f'switch_tfm_init_scale must be positive, got {self.switch_tfm_init_scale}'
            )


class LayerNorm(nn.Module):
    """Layer normalization with optional bias.

    PyTorch's LayerNorm doesn't support bias=False directly, so this wrapper
    provides that functionality.
    """

    def __init__(self, ndim: int, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply layer normalization.

        Args:
            input: Input tensor of shape [..., ndim]

        Returns:
            Normalized tensor of same shape as input
        """
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias,
                            1e-5)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with optional Flash Attention support."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError(
                f'Embedding dimension {config.n_embd} must be divisible by num_heads {config.n_head}'
            )

        # Key, query, value projections for all heads in a batch
        self.c_attn = nn.Linear(config.n_embd,
                                3 * config.n_embd,
                                bias=config.bias)

        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Store configuration
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # Flash attention support (PyTorch >= 2.0)
        self.flash = hasattr(torch.nn.functional,
                             'scaled_dot_product_attention')
        if not self.flash:
            logger.warning(
                'Using slow attention. Flash Attention requires PyTorch >= 2.0'
            )
            # Causal mask to ensure attention is only applied to left in input sequence
            self.register_buffer(
                'bias',
                torch.tril(torch.ones(config.block_size,
                                      config.block_size)).view(
                                          1, 1, config.block_size,
                                          config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of causal self-attention.

        Args:
            x: Input tensor of shape [batch_size, seq_len, n_embd]

        Returns:
            Output tensor of shape [batch_size, seq_len, n_embd]
        """
        B, T, C = x.size(
        )  # batch size, sequence length, embedding dimensionality

        # Calculate query, key, values for all heads in batch
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # Reshape for multi-head attention: [B, T, n_head, head_dim]
        head_dim = C // self.n_head
        k = k.view(B, T, self.n_head,
                   head_dim).transpose(1, 2)  # [B, n_head, T, head_dim]
        q = q.view(B, T, self.n_head,
                   head_dim).transpose(1, 2)  # [B, n_head, T, head_dim]
        v = v.view(B, T, self.n_head,
                   head_dim).transpose(1, 2)  # [B, n_head, T, head_dim]

        # Causal self-attention
        if self.flash:
            # Efficient attention using Flash Attention CUDA kernels
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True)
        else:
            # Manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # [B, n_head, T, T] x [B, n_head, T, head_dim] -> [B, n_head, T, head_dim]

        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Standard Multi-Layer Perceptron for non-MoE transformer blocks."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd,
                              4 * config.n_embd,
                              bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd,
                                config.n_embd,
                                bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of MLP.

        Args:
            x: Input tensor of shape [batch_size, seq_len, n_embd]

        Returns:
            Output tensor of shape [batch_size, seq_len, n_embd]
        """
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Standard transformer block with self-attention and MLP."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of transformer block.

        Args:
            x: Input tensor of shape [batch_size, seq_len, n_embd]

        Returns:
            Output tensor of shape [batch_size, seq_len, n_embd]
        """
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MLPExperts(nn.Module):
    """Group of MLP experts for Mixture of Experts layers.

    This module implements a collection of expert MLPs that process different
    subsets of tokens based on routing decisions.
    """

    def __init__(self, config: GPTConfig) -> None:
        """Initialize MLP experts.

        Args:
            config: Configuration for the model.
        """
        super().__init__()

        self.bias = config.bias
        self.n_experts = config.n_experts

        # Expert weights: [n_experts, d_model, 4 * d_model]
        self.c_fc = nn.Parameter(
            torch.empty(config.n_experts, config.n_embd, 4 * config.n_embd))
        self.c_proj = nn.Parameter(
            torch.empty(config.n_experts, 4 * config.n_embd, config.n_embd))

        # Optional biases
        self.fc_bias = nn.Parameter(
            torch.zeros(config.n_experts, 1, 4 *
                        config.n_embd)) if config.bias else None
        self.proj_bias = nn.Parameter(
            torch.zeros(config.n_experts, 1,
                        config.n_embd)) if config.bias else None

        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize expert weights using normal distribution."""
        nn.init.normal_(self.c_fc, mean=0.0, std=0.02)
        nn.init.normal_(self.c_proj, mean=0.0, std=0.02)
        if self.bias:
            nn.init.zeros_(self.fc_bias)
            nn.init.zeros_(self.proj_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through expert MLPs.

        Args:
            x: Input tensor of shape [n_experts, expert_capacity, d_model]

        Returns:
            Output tensor of shape [n_experts, expert_capacity, d_model]
        """
        if x.dim() != 3:
            raise ValueError(f'Expected 3D input tensor, got {x.dim()}D')

        if x.size(0) != self.n_experts:
            raise ValueError(
                f'Expected {self.n_experts} experts, got {x.size(0)}')

        # First linear transformation with optional bias
        x = torch.bmm(x, self.c_fc)  # [n_experts, capacity, 4 * d_model]
        if self.bias:
            x = x + self.fc_bias

        # Activation function
        x = self.gelu(x)

        # Second linear transformation with optional bias
        x = torch.bmm(x, self.c_proj)  # [n_experts, capacity, d_model]
        if self.bias:
            x = x + self.proj_bias

        # Dropout
        x = self.dropout(x)

        return x


class Router(nn.Module):
    """Router module for expert selection in Mixture of Experts.

    This module implements the routing mechanism that decides which experts
    should process each token, including optional noisy top-k routing.
    """

    def __init__(self, config: GPTConfig) -> None:
        """Initialize router.

        Args:
            config: Configuration for router
        """
        super().__init__()

        self.top_k = config.top_k
        self.n_experts = config.n_experts
        self.capacity_factor = config.capacity_factor
        self.use_noisy_top_k = config.use_noisy_top_k
        self.use_aux_loss = config.use_aux_loss
        self.aux_loss_weight = config.aux_loss_weight

        # Routing network
        self.gate = nn.Linear(config.n_embd, config.n_experts, bias=False)

        # Noise network for load balancing (optional)
        if config.use_noisy_top_k:
            self.noise = nn.Linear(config.n_embd, config.n_experts, bias=False)

    def _compute_gate_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Compute gate scores with optional noise for load balancing.

        Args:
            x: Input tensor of shape [batch_size * seq_len, d_model]

        Returns:
            Gate scores of shape [batch_size * seq_len, n_experts]
        """
        if x.dim() != 2:
            raise ValueError(f'Expected 2D input tensor, got {x.dim()}D')

        # Base gate scores
        gate_scores = self.gate(x)

        if self.use_noisy_top_k:
            # Add noise for load balancing (eq. 4 in https://arxiv.org/abs/1701.06538)
            noise = F.softplus(self.noise(x))
            noise = noise * torch.randn_like(noise)
            gate_scores = gate_scores + noise

        return gate_scores

    def _select_experts(
            self,
            gate_scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select top-k experts for each token.

        Args:
            gate_scores: Gate scores of shape [num_tokens, n_experts]

        Returns:
            Tuple of (expert_weights, expert_indices) both of shape [num_tokens, top_k]
        """
        if gate_scores.dim() != 2:
            raise ValueError(
                f'Expected 2D gate scores, got {gate_scores.dim()}D')

        # Top-k expert selection
        top_k_weights, top_k_indices = torch.topk(gate_scores,
                                                  self.top_k,
                                                  dim=-1)

        # Compute probabilities over selected experts
        expert_weights = torch.full_like(gate_scores, float('-inf'))
        expert_weights.scatter_(-1, top_k_indices, top_k_weights)
        expert_weights = F.softmax(expert_weights, dim=-1)

        # Extract weights for selected experts
        expert_weights = expert_weights.gather(-1, top_k_indices)

        return expert_weights, top_k_indices

    def _compute_expert_capacity(self, num_tokens: int) -> int:
        """Compute expert capacity based on configuration.

        Args:
            num_tokens: Total number of tokens in batch

        Returns:
            Expert capacity (ensured to be even)
        """
        if num_tokens <= 0:
            raise ValueError(f'num_tokens must be positive, got {num_tokens}')

        capacity = math.floor(self.top_k * self.capacity_factor * num_tokens /
                              self.n_experts)
        capacity += capacity % 2  # Ensure even capacity
        return max(int(capacity), 1)  # Ensure at least 1 capacity

    def compute_aux_loss(self, expert_probs: torch.Tensor,
                         indices: torch.Tensor) -> torch.Tensor:
        """Compute Switch Transformer auxiliary loss for load balancing.

        This loss encourages balanced expert usage by penalizing uneven distribution
        of tokens across experts.

        Args:
            expert_probs: Expert probabilities of shape [batch_size, seq_len, top_k]
            indices: Expert indices of shape [batch_size, seq_len, top_k]

        Returns:
            Auxiliary loss scalar tensor
        """
        if expert_probs.dim() != 3 or indices.dim() != 3:
            raise ValueError(
                'Expected 3D tensors for expert_probs and indices')

        if expert_probs.size() != indices.size():
            raise ValueError(
                'expert_probs and indices must have the same shape')

        # equation (5): compute ratio of tokens allocated to each expert
        # total number of tokens is defined as total tokens in batch * k
        with torch.no_grad():
            one_hot_indices = F.one_hot(
                indices, num_classes=self.n_experts)  # [B, T, k, n_experts]
            one_hot_indices = torch.sum(
                one_hot_indices.float(),
                dim=2)  # [B, T, n_experts] (sum over k dimension)
            tokens_per_expert = torch.mean(one_hot_indices.float(), dim=(0, 1))

        # equation (6): compute ratio of router probability allocated to each expert
        prob_per_expert = torch.mean(expert_probs.float(), dim=(0, 1))

        # equation (4): take a scaled dot product between prob/token allocation vectors
        # multiply the result by the number of experts
        aux_loss = self.n_experts * torch.sum(
            prob_per_expert * tokens_per_expert)

        return self.aux_loss_weight * aux_loss

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               Optional[torch.Tensor]]:
        """Route tokens to experts.

        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]

        Returns:
            Tuple containing:
            - expert_weights: Routing weights of shape [num_tokens, n_experts, capacity]
            - expert_mask: Binary mask of shape [num_tokens, n_experts, capacity]
            - expert_batches: Expert input batches of shape [n_experts, capacity, d_model]
            - aux_loss: Auxiliary loss for load balancing (if enabled)
        """
        if x.dim() != 3:
            raise ValueError(f'Expected 3D input tensor, got {x.dim()}D')

        batch_size, seq_len, d_model = x.size()
        num_tokens = batch_size * seq_len

        # Flatten input for routing
        x_flat = x.view(num_tokens, d_model)

        # Compute gate scores
        gate_scores = self._compute_gate_scores(x_flat)

        # Select top-k experts
        expert_weights, expert_indices = self._select_experts(gate_scores)

        # Compute auxiliary loss if enabled
        aux_loss = None
        if self.use_aux_loss:
            # Reshape for aux loss computation
            expert_probs_reshaped = expert_weights.view(
                batch_size, seq_len, -1)
            aux_loss = self.compute_aux_loss(
                expert_probs_reshaped,
                expert_indices.view(batch_size, seq_len, -1))

        # Compute expert capacity
        expert_capacity = self._compute_expert_capacity(num_tokens)

        # Create expert assignment masks with vectorized approach
        expert_mask = torch.zeros(num_tokens,
                                  self.n_experts,
                                  expert_capacity,
                                  dtype=torch.bool,
                                  device=x.device)

        # Vectorized assignment of tokens to experts
        token_indices = torch.arange(num_tokens, device=x.device).unsqueeze(1)  # [num_tokens, 1]
        expert_indices_expanded = expert_indices.unsqueeze(-1)  # [num_tokens, top_k, 1]
        
        # Create indices for capacity dimension
        capacity_indices = torch.zeros(num_tokens, self.top_k, 1, dtype=torch.long, device=x.device)
        
        # Scatter assignments to expert_mask
        try:
            expert_mask.scatter_(1, expert_indices_expanded, True)
        except RuntimeError:
            # Handle case where some experts exceed capacity
            # Limit assignments to capacity
            for i in range(self.top_k):
                expert_idx = expert_indices[:, i]  # [num_tokens]
                # Count current assignments for each expert
                current_counts = expert_mask.sum(dim=(0, 2))  # [n_experts]
                # Check which tokens can still be assigned
                can_assign = current_counts[expert_idx] < expert_capacity
                # Assign tokens that can be assigned
                assign_token_indices = token_indices[can_assign].squeeze(-1)
                assign_expert_indices = expert_idx[can_assign]
                expert_mask[assign_token_indices, assign_expert_indices, 0] = True

        # Compute final routing weights
        final_weights = torch.zeros(num_tokens,
                                    self.n_experts,
                                    expert_capacity,
                                    dtype=x.dtype,
                                    device=x.device)

        # Populate weights using advanced indexing
        weight_indices = torch.arange(num_tokens, device=x.device)
        for k in range(self.top_k):
            exp_idx = expert_indices[:, k]
            weights = expert_weights[:, k]
            # Assign weights to corresponding positions
            final_weights[weight_indices, exp_idx, 0] = weights

        # Prepare expert input batches using vectorized operations
        expert_batches = torch.zeros(self.n_experts,
                                     expert_capacity,
                                     d_model,
                                     dtype=x.dtype,
                                     device=x.device)

        # Group tokens by expert assignment using advanced indexing
        for expert_idx in range(self.n_experts):
            # Get all tokens assigned to this expert
            assigned_tokens_mask = expert_mask[:, expert_idx, 0]  # [num_tokens]
            assigned_token_indices = torch.where(assigned_tokens_mask)[0]
            
            if len(assigned_token_indices) > 0:
                # Limit to expert capacity
                limited_indices = assigned_token_indices[:expert_capacity]
                tokens = x_flat[limited_indices]
                expert_batches[expert_idx, :len(tokens), :] = tokens

        return final_weights, expert_mask.float(), expert_batches, aux_loss


class MOELayer(nn.Module):
    """Mixture of Experts layer implementing routing and expert processing.

    This layer combines the router and expert networks to implement the full
    Mixture of Experts mechanism.
    """

    def __init__(
        self,
        config: GPTConfig
    ) -> None:
        """Initialize MoE layer.

        Args:
            config: Configuration for the Mixture of Experts layer.
        """
        super().__init__()

        # Router for token-to-expert assignment
        self.router = Router(config)

        # Expert networks
        self.experts = MLPExperts(config)

    def forward(
            self,
            x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through MoE layer.

        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]

        Returns:
            Tuple of (output, aux_loss) where aux_loss may be None
        """
        if x.dim() != 3:
            raise ValueError(f'Expected 3D input tensor, got {x.dim()}D')

        batch_size, seq_len, d_model = x.size()
        num_tokens = batch_size * seq_len

        # Route tokens to experts
        expert_weights, expert_mask, expert_batches, aux_loss = self.router(x)

        # Process tokens through expert networks
        expert_outputs = self.experts(expert_batches)
        # [n_experts, capacity, d_model]

        # More efficient aggregation of expert outputs
        # Reshape for efficient computation
        expert_outputs_flat = expert_outputs.view(-1, d_model)  # [n_experts * capacity, d_model]
        
        # Reshape expert weights for batch matrix multiplication
        # expert_weights: [num_tokens, n_experts, capacity]
        expert_weights_flat = expert_weights.view(num_tokens, -1)  # [num_tokens, n_experts * capacity]

        # Weighted combination of expert outputs using einsum for better performance
        # Equivalent to: output = expert_weights_flat @ expert_outputs_flat
        output = torch.einsum('nc,cd->nd', expert_weights_flat, expert_outputs_flat)
        # [num_tokens, d_model]

        # Reshape back to original dimensions
        return output.view(batch_size, seq_len, d_model), aux_loss


class MoEBlock(nn.Module):
    """Transformer block with MoE layer instead of standard MLP."""

    def __init__(self, config: GPTConfig) -> None:
        """Initialize MoE transformer block.

        Args:
            config: GPT configuration
        """
        super().__init__()

        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.moe = MOELayer(config=config)

    def forward(
            self,
            x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of MoE transformer block.

        Args:
            x: Input tensor of shape [batch_size, seq_len, n_embd]

        Returns:
            Tuple of (output, aux_loss) where aux_loss may be None
        """
        if x.dim() != 3:
            raise ValueError(f'Expected 3D input tensor, got {x.dim()}D')

        # Self-attention with residual connection
        x = x + self.attn(self.ln_1(x))

        # MoE layer with residual connection
        moe_output, aux_loss = self.moe(self.ln_2(x))
        x = x + moe_output

        return x, aux_loss


class GPT(nn.Module):
    """GPT Language Model with optional Mixture of Experts layers."""

    def __init__(self, config: GPTConfig) -> None:
        """Initialize GPT model.

        Args:
            config: Model configuration
        """
        super().__init__()

        # Validate configuration
        if config.vocab_size is None or config.vocab_size <= 0:
            raise ValueError(
                f'vocab_size must be specified and positive, got {config.vocab_size}'
            )
        if config.block_size is None or config.block_size <= 0:
            raise ValueError(
                f'block_size must be specified and positive, got {config.block_size}'
            )

        self.config = config

        # Determine which layers should use MoE
        if config.use_moe and config.moe_layers is None:
            # Default pattern: use MoE in last half of layers
            moe_layers = list(range(config.n_layer // 2, config.n_layer))
        else:
            moe_layers = config.moe_layers or []

        # Transformer components
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size,
                                 config.n_embd),  # Token embeddings
                wpe=nn.Embedding(config.block_size,
                                 config.n_embd),  # Position embeddings
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList(),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            ))

        # Build transformer layers
        for i in range(config.n_layer):
            if i in moe_layers:
                # Use MoE block
                layer = MoEBlock(config=config)
            else:
                # Use standard transformer block
                layer = Block(config)

            self.transformer.h.append(layer)

        # Language modeling head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying (token embeddings with output layer)
        # Note: This may generate warnings with torch.compile() but is generally harmless
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize all weights
        self.apply(self._init_weights)

        # Special initialization for residual projections (per GPT-2 paper)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p,
                                      mean=0.0,
                                      std=0.02 / math.sqrt(2 * config.n_layer))

        # Report parameter count
        logger.info(
            f'Number of parameters: {self.get_num_params() / 1e6:.2f}M')

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Return the number of parameters in the model.

        For non-embedding count (default), position embeddings are subtracted.
        Token embeddings are included due to weight tying with the output layer.

        Args:
            non_embedding: Whether to exclude embedding parameters

        Returns:
            Number of parameters
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    @torch.no_grad()
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights for the given module.

        Args:
            module: Module to initialize
        """
        if isinstance(module, nn.Linear):
            if self.config.use_switch_tfm_init:
                scale = self.config.switch_tfm_init_scale

                # linear layers have flipped dimensions in torch
                # size of weights is [out_dim, in_dim]
                w_fan_in = module.weight.shape[-1]
                w_std = (scale / w_fan_in)**0.5
                torch.nn.init.trunc_normal_(
                    module.weight,
                    mean=0.0,
                    std=w_std,
                    a=-2 * w_std,
                    b=2 * w_std,
                )
            else:
                # perform standard (normal) initialization of weights
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

            # always initialize bias to zero
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, MLPExperts):
            # we have to init expert weights manually because
            # nn.Parameter is not a type of module in torch
            if self.config.use_switch_tfm_init:
                scale = self.config.switch_tfm_init_scale

                c_fc_fan_in = module.c_fc.shape[-2]
                c_fc_std = (scale / c_fc_fan_in)**0.5
                torch.nn.init.trunc_normal_(
                    module.c_fc,
                    mean=0.0,
                    std=c_fc_std,
                    a=-2 * c_fc_std,
                    b=2 * c_fc_std,
                )

                c_proj_fan_in = module.c_proj.shape[-2]
                c_proj_std = (scale / c_proj_fan_in)**0.5
                torch.nn.init.trunc_normal_(
                    module.c_proj,
                    mean=0.0,
                    std=c_proj_std,
                    a=-2 * c_proj_std,
                    b=2 * c_proj_std,
                )
            else:
                # perform standard (normal) initialization of weights
                torch.nn.init.normal_(module.c_fc, mean=0.0, std=0.02)
                torch.nn.init.normal_(module.c_proj, mean=0.0, std=0.02)

            # bias is always initialized to zero
            if module.fc_bias is not None:
                torch.nn.init.zeros_(module.fc_bias)
                torch.nn.init.zeros_(module.proj_bias)
        elif isinstance(module, nn.Embedding):
            # just use standard initialization scheme for embedding always
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through the model.

        Args:
            idx: Input tensor of token indices with shape [batch_size, seq_len]
            targets: Target tensor for training with shape [batch_size, seq_len]

        Returns:
            Tuple of (logits, loss) where loss includes auxiliary losses if enabled
        """
        if idx.dim() != 2:
            raise ValueError(f'Expected 2D input tensor, got {idx.dim()}D')

        device = idx.device
        batch_size, seq_len = idx.size()

        # Validate sequence length
        if seq_len > self.config.block_size:
            raise ValueError(f'Cannot forward sequence of length {seq_len}, '
                             f'block size is only {self.config.block_size}')

        # Create position indices
        pos = torch.arange(0, seq_len, dtype=torch.long, device=device)

        # Forward pass through transformer
        tok_emb = self.transformer.wte(idx)  # [batch_size, seq_len, n_embd]
        pos_emb = self.transformer.wpe(pos)  # [seq_len, n_embd]
        x = self.transformer.drop(tok_emb + pos_emb)

        # Apply transformer blocks and collect auxiliary losses
        total_aux_loss = 0.0
        aux_loss_count = 0

        for block in self.transformer.h:
            if isinstance(block, MoEBlock):
                x, aux_loss = block(x)
                if aux_loss is not None:
                    total_aux_loss += aux_loss
                    aux_loss_count += 1
            else:
                x = block(x)

        # Final layer normalization
        x = self.transformer.ln_f(x)

        # Language modeling head
        if targets is not None:
            # Training mode: compute loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1),
                                   ignore_index=-1)

            # Add auxiliary loss if any MoE layers produced one
            if aux_loss_count > 0:
                loss = loss + total_aux_loss / aux_loss_count
        else:
            # Inference mode: only compute logits for last position (optimization)
            logits = self.lm_head(x[:, [-1], :])
            # Note: using list [-1] preserves time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size: int) -> None:
        """Reduce the model's block size if necessary.

        This is useful when loading a pretrained model with a larger block size
        but wanting to use a smaller one for efficiency.

        Args:
            block_size: New block size (must be <= current block size)
        """
        if block_size > self.config.block_size:
            raise ValueError(
                f'New block size {block_size} must be <= current block size {self.config.block_size}'
            )

        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(
            self.transformer.wpe.weight[:block_size])

        # Update attention masks if they exist
        for block in self.transformer.h:
            if hasattr(block, 'attn') and hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :
                                                  block_size]

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate new tokens from the model.

        Args:
            idx: Conditioning sequence of indices with shape [batch_size, seq_len]
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If specified, only sample from top k tokens

        Returns:
            Generated sequence with shape [batch_size, seq_len + max_new_tokens]
        """
        if temperature <= 0:
            raise ValueError(
                f'Temperature must be positive, got {temperature}')
        if max_new_tokens < 0:
            raise ValueError(
                f'max_new_tokens must be non-negative, got {max_new_tokens}')

        for _ in range(max_new_tokens):
            # Crop sequence if it gets too long
            idx_cond = idx if idx.size(
                1) <= self.config.block_size else idx[:,
                                                      -self.config.block_size:]

            # Forward pass to get logits
            logits, _ = self(idx_cond)

            # Pluck logits at final step and scale by temperature
            logits = logits[:, -1, :] / temperature

            # Optionally crop to top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')

            # Convert to probabilities and sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        """Estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS.

        Args:
            fwdbwd_per_iter: Number of forward-backward passes per iteration
            dt: Time elapsed in seconds

        Returns:
            MFU value between 0 and 1
        """
        if fwdbwd_per_iter <= 0:
            raise ValueError(
                f'fwdbwd_per_iter must be positive, got {fwdbwd_per_iter}')
        if dt <= 0:
            raise ValueError(f'dt must be positive, got {dt}')

        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu


# Additional utility functions for model analysis and debugging
def analyze_moe_usage(model: GPT) -> Dict[str, Any]:
    """Analyze MoE usage in the model.

    Args:
        model: GPT model instance

    Returns:
        Dictionary containing MoE statistics
    """
    if not isinstance(model, GPT):
        raise TypeError(f'Expected GPT model, got {type(model)}')

    moe_layers = []
    total_params = model.get_num_params()

    for i, block in enumerate(model.transformer.h):
        if isinstance(block, MoEBlock):
            moe_layers.append(i)

    return {
        'total_layers': len(model.transformer.h),
        'moe_layers': moe_layers,
        'moe_layer_count': len(moe_layers),
        'total_parameters': total_params,
        'uses_moe': len(moe_layers) > 0,
    }


def get_moe_layer_info(model: GPT, layer_idx: int) -> Optional[Dict[str, Any]]:
    """Get detailed information about a specific MoE layer.

    Args:
        model: GPT model instance
        layer_idx: Layer index to inspect

    Returns:
        Dictionary with layer information or None if not an MoE layer
    """
    if not isinstance(model, GPT):
        raise TypeError(f'Expected GPT model, got {type(model)}')

    if layer_idx >= len(model.transformer.h):
        return None

    block = model.transformer.h[layer_idx]
    if not isinstance(block, MoEBlock):
        return None

    moe_layer = block.moe
    return {
        'layer_idx': layer_idx,
        'n_experts': moe_layer.router.n_experts,
        'top_k': moe_layer.router.top_k,
        'capacity_factor': moe_layer.router.capacity_factor,
        'use_noisy_top_k': moe_layer.router.use_noisy_top_k,
        'use_aux_loss': moe_layer.router.use_aux_loss,
        'expert_params':
        sum(p.numel() for p in moe_layer.experts.parameters()),
        'router_params': sum(p.numel() for p in moe_layer.router.parameters()),
    }
