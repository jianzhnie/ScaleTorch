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

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

# Configure logging
logger = logging.getLogger(__name__)


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
    use_noisy_top_k: bool = True

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

    def __init__(
        self,
        d_model: int,
        n_experts: int = 8,
        bias: bool = False,
        dropout: float = 0.2,
    ) -> None:
        """Initialize MLP experts.

        Args:
            d_model: Model dimension size
            n_experts: Number of experts to create
            bias: Whether to use bias in linear layers
            dropout: Dropout probability
        """
        super().__init__()

        if n_experts <= 0:
            raise ValueError(
                f'Number of experts must be positive, got {n_experts}')
        if d_model <= 0:
            raise ValueError(
                f'Model dimension must be positive, got {d_model}')
        if not 0.0 <= dropout <= 1.0:
            raise ValueError(f'Dropout must be in [0, 1], got {dropout}')

        self.n_experts = n_experts
        self.d_model = d_model
        self.bias = bias

        # Expert weights: [n_experts, d_model, 4 * d_model]
        self.w1 = nn.Parameter(torch.empty(n_experts, d_model, 4 * d_model))
        self.w2 = nn.Parameter(torch.empty(n_experts, 4 * d_model, d_model))

        # Optional biases
        self.b1 = nn.Parameter(torch.zeros(n_experts, 1, 4 *
                                           d_model)) if bias else None
        self.b2 = nn.Parameter(torch.zeros(n_experts, 1,
                                           d_model)) if bias else None

        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize expert weights using normal distribution."""
        nn.init.normal_(self.w1, mean=0.0, std=0.02)
        nn.init.normal_(self.w2, mean=0.0, std=0.02)
        if self.bias:
            nn.init.zeros_(self.b1)
            nn.init.zeros_(self.b2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through expert MLPs.

        Args:
            x: Input tensor of shape [n_experts, expert_capacity, d_model]

        Returns:
            Output tensor of shape [n_experts, expert_capacity, d_model]
        """
        # First linear transformation with optional bias
        x = torch.bmm(x, self.w1)  # [n_experts, capacity, 4 * d_model]
        if self.bias:
            x = x + self.b1

        # Activation function
        x = self.gelu(x)

        # Second linear transformation with optional bias
        x = torch.bmm(x, self.w2)  # [n_experts, capacity, d_model]
        if self.bias:
            x = x + self.b2

        # Dropout
        x = self.dropout(x)

        return x


class Router(nn.Module):
    """Router module for expert selection in Mixture of Experts.

    This module implements the routing mechanism that decides which experts
    should process each token, including optional noisy top-k routing.
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int = 8,
        top_k: int = 2,
        use_noisy_top_k: bool = True,
        capacity_factor: float = 1.25,
    ) -> None:
        """Initialize router.

        Args:
            d_model: Model dimension size
            n_experts: Number of available experts
            top_k: Number of experts to route each token to
            use_noisy_top_k: Whether to add noise to routing decisions
            capacity_factor: Expert capacity multiplier
        """
        super().__init__()

        if n_experts <= 0:
            raise ValueError(
                f'Number of experts must be positive, got {n_experts}')
        if d_model <= 0:
            raise ValueError(
                f'Model dimension must be positive, got {d_model}')
        if not 1 <= top_k <= n_experts:
            raise ValueError(
                f'top_k ({top_k}) must be between 1 and n_experts ({n_experts})'
            )
        if capacity_factor <= 0:
            raise ValueError(
                f'capacity_factor must be positive, got {capacity_factor}')

        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k
        self.use_noisy_top_k = use_noisy_top_k
        self.capacity_factor = capacity_factor

        # Routing network
        self.gate = nn.Linear(d_model, n_experts, bias=False)

        # Noise network for load balancing (optional)
        if use_noisy_top_k:
            self.noise = nn.Linear(d_model, n_experts, bias=False)

    def _compute_gate_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Compute gate scores with optional noise for load balancing.

        Args:
            x: Input tensor of shape [batch_size * seq_len, d_model]

        Returns:
            Gate scores of shape [batch_size * seq_len, n_experts]
        """
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
        capacity = math.floor(self.top_k * self.capacity_factor * num_tokens /
                              self.n_experts)
        capacity += capacity % 2  # Ensure even capacity
        return int(capacity)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route tokens to experts.

        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]

        Returns:
            Tuple containing:
            - expert_weights: Routing weights of shape [num_tokens, n_experts, capacity]
            - expert_mask: Binary mask of shape [num_tokens, n_experts, capacity]
            - expert_batches: Expert input batches of shape [n_experts, capacity, d_model]
        """
        batch_size, seq_len, d_model = x.size()
        num_tokens = batch_size * seq_len

        # Flatten input for routing
        x_flat = x.view(num_tokens, d_model)

        # Compute gate scores
        gate_scores = self._compute_gate_scores(x_flat)

        # Select top-k experts
        expert_weights, expert_indices = self._select_experts(gate_scores)

        # Compute expert capacity
        expert_capacity = self._compute_expert_capacity(num_tokens)

        # Create expert assignment masks
        # This is a complex operation that assigns tokens to experts while respecting capacity limits
        expert_mask = torch.zeros(num_tokens,
                                  self.n_experts,
                                  expert_capacity,
                                  dtype=torch.bool,
                                  device=x.device)

        # For each expert, track which tokens are assigned and their positions
        for expert_idx in range(self.n_experts):
            # Find tokens routed to this expert
            tokens_for_expert = (expert_indices == expert_idx).any(dim=-1)
            token_indices = torch.where(tokens_for_expert)[0]

            # Limit to capacity
            if len(token_indices) > expert_capacity:
                token_indices = token_indices[:expert_capacity]

            # Assign tokens to expert positions
            if len(token_indices) > 0:
                expert_mask[token_indices,
                            expert_idx, :len(token_indices)] = True

        # Compute final routing weights
        final_weights = torch.zeros(num_tokens,
                                    self.n_experts,
                                    expert_capacity,
                                    dtype=x.dtype,
                                    device=x.device)

        # Populate weights based on expert assignments
        for i in range(num_tokens):
            for j in range(self.n_experts):
                assigned_positions = expert_mask[i,
                                                 j].nonzero(as_tuple=True)[0]
                if len(assigned_positions) > 0:
                    # Use the weight for this expert (sum if multiple top-k selections)
                    expert_weight = expert_weights[i, (
                        expert_indices[i] == j)].sum()
                    final_weights[i, j, assigned_positions] = expert_weight

        # Prepare expert input batches
        expert_batches = torch.zeros(self.n_experts,
                                     expert_capacity,
                                     d_model,
                                     dtype=x.dtype,
                                     device=x.device)

        for expert_idx in range(self.n_experts):
            # Get tokens assigned to this expert
            assigned_tokens = expert_mask[:, expert_idx, :].any(dim=-1)
            token_indices = torch.where(assigned_tokens)[0]

            if len(token_indices) > 0:
                # Get the actual token embeddings
                tokens = x_flat[token_indices]

                # Limit to capacity
                if len(tokens) > expert_capacity:
                    tokens = tokens[:expert_capacity]

                expert_batches[expert_idx, :len(tokens)] = tokens

        return final_weights, expert_mask.float(), expert_batches


class MOELayer(nn.Module):
    """Mixture of Experts layer implementing routing and expert processing.

    This layer combines the router and expert networks to implement the full
    Mixture of Experts mechanism.
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int = 8,
        top_k: int = 2,
        use_noisy_top_k: bool = True,
        capacity_factor: float = 1.25,
        bias: bool = False,
        dropout: float = 0.2,
    ) -> None:
        """Initialize MoE layer.

        Args:
            d_model: Model dimension size
            n_experts: Number of experts
            top_k: Number of experts to route each token to
            use_noisy_top_k: Whether to add noise to routing
            capacity_factor: Expert capacity multiplier
            bias: Whether to use bias in expert networks
            dropout: Dropout probability
        """
        super().__init__()

        self.d_model = d_model
        self.n_experts = n_experts

        # Router for token-to-expert assignment
        self.router = Router(
            d_model=d_model,
            n_experts=n_experts,
            top_k=top_k,
            use_noisy_top_k=use_noisy_top_k,
            capacity_factor=capacity_factor,
        )

        # Expert networks
        self.experts = MLPExperts(
            d_model=d_model,
            n_experts=n_experts,
            bias=bias,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MoE layer.

        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]

        Returns:
            Output tensor of shape [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.size()
        num_tokens = batch_size * seq_len

        # Route tokens to experts
        expert_weights, expert_mask, expert_batches = self.router(x)

        # Process tokens through expert networks
        expert_outputs = self.experts(
            expert_batches)  # [n_experts, capacity, d_model]

        # Aggregate expert outputs (eq. 2 in ST-MoE: https://arxiv.org/abs/2202.08906)
        expert_outputs_flat = expert_outputs.view(
            -1, d_model)  # [n_experts * capacity, d_model]
        expert_weights_flat = expert_weights.view(
            num_tokens, -1)  # [num_tokens, n_experts * capacity]

        # Weighted combination of expert outputs
        output = expert_weights_flat @ expert_outputs_flat  # [num_tokens, d_model]

        # Reshape back to original dimensions
        return output.view(batch_size, seq_len, d_model)


class MoEBlock(nn.Module):
    """Transformer block with MoE layer instead of standard MLP."""

    def __init__(
        self,
        config: GPTConfig,
        n_experts: int = 8,
        top_k: int = 2,
        use_noisy_top_k: bool = True,
        capacity_factor: float = 1.25,
    ) -> None:
        """Initialize MoE transformer block.

        Args:
            config: GPT configuration
            n_experts: Number of experts
            top_k: Number of experts to route each token to
            use_noisy_top_k: Whether to add noise to routing
            capacity_factor: Expert capacity multiplier
        """
        super().__init__()

        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.moe = MOELayer(
            d_model=config.n_embd,
            n_experts=n_experts,
            top_k=top_k,
            use_noisy_top_k=use_noisy_top_k,
            capacity_factor=capacity_factor,
            bias=config.bias,
            dropout=config.dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of MoE transformer block.

        Args:
            x: Input tensor of shape [batch_size, seq_len, n_embd]

        Returns:
            Output tensor of shape [batch_size, seq_len, n_embd]
        """
        x = x + self.attn(self.ln_1(x))
        x = x + self.moe(self.ln_2(x))
        return x


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
                layer = MoEBlock(
                    config=config,
                    n_experts=config.n_experts,
                    top_k=config.top_k,
                    use_noisy_top_k=config.use_noisy_top_k,
                    capacity_factor=config.capacity_factor,
                )
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

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights for different module types."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
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
            Tuple of (logits, loss) where loss is None during inference
        """
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

        # Apply transformer blocks
        for block in self.transformer.h:
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
        else:
            # Inference mode: only compute logits for last position (optimization)
            logits = self.lm_head(
                x[:, [-1], :])  # Note: using list [-1] preserves time dim
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


# Additional utility functions for model analysis and debugging
def analyze_moe_usage(model: GPT) -> Dict[str, any]:
    """Analyze MoE usage in the model.

    Args:
        model: GPT model instance

    Returns:
        Dictionary containing MoE statistics
    """
    moe_layers = []
    total_params = model.get_num_params()

    for i, block in enumerate(model.transformer.h):
        if hasattr(block, 'moe'):
            moe_layers.append(i)

    return {
        'total_layers': len(model.transformer.h),
        'moe_layers': moe_layers,
        'moe_layer_count': len(moe_layers),
        'total_parameters': total_params,
        'uses_moe': len(moe_layers) > 0,
    }


def get_moe_layer_info(model: GPT, layer_idx: int) -> Optional[Dict[str, any]]:
    """Get detailed information about a specific MoE layer.

    Args:
        model: GPT model instance
        layer_idx: Layer index to inspect

    Returns:
        Dictionary with layer information or None if not an MoE layer
    """
    if layer_idx >= len(model.transformer.h):
        return None

    block = model.transformer.h[layer_idx]
    if not hasattr(block, 'moe'):
        return None

    moe_layer = block.moe
    return {
        'layer_idx': layer_idx,
        'n_experts': moe_layer.n_experts,
        'top_k': moe_layer.router.top_k,
        'capacity_factor': moe_layer.router.capacity_factor,
        'use_noisy_top_k': moe_layer.router.use_noisy_top_k,
        'expert_params':
        sum(p.numel() for p in moe_layer.experts.parameters()),
        'router_params': sum(p.numel() for p in moe_layer.router.parameters()),
    }
