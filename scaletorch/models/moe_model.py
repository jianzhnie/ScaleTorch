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
        use_optimized_routing: Whether to use optimized routing (default: True)
        expert_capacity_roundup: Whether to round up expert capacity to even number (default: True)
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
    eval_capacity: float = 2.0
    min_capacity: float = 0.0
    use_switch_tfm_init: bool = False
    switch_tfm_init_scale: float = 1.0
    use_optimized_routing: bool = True
    use_einsum_aggregation: bool = True
    expert_capacity_roundup: bool = True

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
    """Layer normalization with optional bias."""

    def __init__(self, ndim: int, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias,
                            1e-5)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with optional Flash Attention support."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Key, query, value projections for all heads in a batch
        self.c_attn = nn.Linear(config.n_embd,
                                3 * config.n_embd,
                                bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # Flash attention support
        self.flash = hasattr(torch.nn.functional,
                             'scaled_dot_product_attention')
        if not self.flash:
            logger.warning(
                'Using slow attention. Flash Attention requires PyTorch >= 2.0'
            )
            self.register_buffer(
                'bias',
                torch.tril(torch.ones(config.block_size,
                                      config.block_size)).view(
                                          1, 1, config.block_size,
                                          config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, n_embd]
        Returns:
            Output: [batch_size, seq_len, n_embd]
        """
        B, T, C = x.size(
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # Calculate query, key, values for all heads in batch
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # k, q, v shape: [B, T, n_head, head_dim] -> [B, n_head, T, head_dim]
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

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
            # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

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
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MLPExperts(nn.Module):
    """Group of MLP experts for Mixture of Experts layers.

    This module implements a collection of expert MLPs that process different
    subsets of tokens based on routing decisions. Optimized for efficient
    batch processing using matrix multiplications.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.bias = config.bias
        self.n_experts = config.n_experts
        self.dropout_prob = config.dropout
        self.intermediate_size = 4 * config.n_embd

        # Optimized expert weights
        self.c_fc = nn.Parameter(
            torch.empty(config.n_experts, config.n_embd,
                        self.intermediate_size))
        self.c_proj = nn.Parameter(
            torch.empty(config.n_experts, self.intermediate_size,
                        config.n_embd))

        # Optional biases
        if config.bias:
            self.fc_bias = nn.Parameter(
                torch.zeros(config.n_experts, 1, self.intermediate_size))
            self.proj_bias = nn.Parameter(
                torch.zeros(config.n_experts, 1, config.n_embd))
        else:
            self.register_buffer('fc_bias', None)
            self.register_buffer('proj_bias', None)

        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(self.dropout_prob)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize expert weights."""
        nn.init.normal_(self.c_fc, mean=0.0, std=0.02)
        nn.init.normal_(self.c_proj, mean=0.0, std=0.02)
        # Biases are initialized to zero in __init__ if they exist

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [n_experts, expert_capacity, n_embd]
        Returns:
            Output tensor [n_experts, expert_capacity, n_embd]
        """
        if x.dim() != 3:
            raise ValueError(f'Expected 3D input tensor, got {x.dim()}D')

        # x: [n_experts, capacity, n_embd]
        # c_fc: [n_experts, n_embd, intermediate_size]
        # Result: [n_experts, capacity, intermediate_size]
        x = torch.einsum('ecm,emi->eci', x, self.c_fc)

        if self.bias:
            x = x + self.fc_bias

        x = self.gelu(x)

        # c_proj: [n_experts, intermediate_size, n_embd]
        # Result: [n_experts, capacity, n_embd]
        x = torch.einsum('eci,eim->ecm', x, self.c_proj)

        if self.bias:
            x = x + self.proj_bias

        x = self.dropout(x)
        return x


class Router(nn.Module):
    """Router module for expert selection in Mixture of Experts."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.top_k = config.top_k
        self.n_experts = config.n_experts
        self.capacity_factor = config.capacity_factor
        self.use_noisy_top_k = config.use_noisy_top_k
        self.use_aux_loss = config.use_aux_loss
        self.aux_loss_weight = config.aux_loss_weight
        self.use_router_z_loss = config.use_router_z_loss

        self.w_gate = nn.Linear(config.n_embd, config.n_experts, bias=False)

        if config.use_noisy_top_k:
            self.w_noise = nn.Linear(config.n_embd,
                                     config.n_experts,
                                     bias=False)

    def _compute_gate_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Compute gate scores with optional noise."""
        gate_scores = self.w_gate(x)

        if self.use_noisy_top_k and self.training:
            # Add noise for load balancing (eq. 4 in https://arxiv.org/abs/1701.06538)
            noise = F.softplus(self.w_noise(x))
            noise = noise * torch.randn_like(noise)
            gate_scores = gate_scores + noise

        return gate_scores

    def _select_experts(
            self,
            gate_scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select top-k experts for each token.

        Args:
            gate_scores: [num_tokens, n_experts]
        Returns:
            expert_weights: [num_tokens, top_k] (softmax probabilities)
            top_k_indices: [num_tokens, top_k]
        """
        # Top-k expert selection
        top_k_weights, top_k_indices = torch.topk(gate_scores,
                                                  self.top_k,
                                                  dim=-1)

        # Softmax over top-k
        expert_weights = F.softmax(top_k_weights, dim=-1)

        return expert_weights, top_k_indices

    def _compute_expert_capacity(self, tokens_per_batch: int) -> int:
        """Compute expert capacity."""
        capacity_factor = self.config.train_capacity if self.training else self.config.eval_capacity

        capacity = math.floor(self.config.top_k * capacity_factor *
                              tokens_per_batch / self.config.n_experts)
        if self.config.expert_capacity_roundup:
            capacity += capacity % 2
        return max(int(capacity), 1)

    def compute_router_z_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Computes ST-MoE router z loss."""
        z_loss = torch.logsumexp(logits, dim=-1)**2.0
        return torch.mean(z_loss)

    def compute_aux_loss(self, expert_probs: torch.Tensor,
                         indices: torch.Tensor) -> torch.Tensor:
        """Compute Switch Transformer auxiliary loss for load balancing."""
        # expert_probs: [B, T, top_k] (or flattened [N, top_k])
        # indices: [B, T, top_k] (or flattened [N, top_k])

        # Flatten if necessary
        if expert_probs.dim() == 3:
            expert_probs = expert_probs.view(-1, expert_probs.size(-1))
            indices = indices.view(-1, indices.size(-1))

        # Density: fraction of tokens routed to each expert
        # Create a mask for all selected experts
        mask = F.one_hot(
            indices,
            num_classes=self.n_experts).float()  # [N, top_k, n_experts]
        mask = mask.sum(
            dim=1
        )  # [N, n_experts] - how many times each expert was selected for each token (usually 0 or 1)
        density = mask.mean(dim=0)  # [n_experts]

        # Probability: mean router probability for each expert
        # We need the probability of each expert being selected.
        # This is tricky because expert_probs only contains top-k probs.
        # But aux loss typically uses the probabilities from the router BEFORE top-k selection,
        # or the sum of probabilities allocated to each expert.
        # Here we use the sum of probabilities allocated to the expert in the top-k selection.

        # Create a tensor of probabilities for all experts (0 for non-selected)
        probs_all = torch.zeros(expert_probs.size(0),
                                self.n_experts,
                                device=expert_probs.device)
        probs_all.scatter_add_(1, indices, expert_probs)  # [N, n_experts]
        prob_per_expert = probs_all.mean(dim=0)  # [n_experts]

        # Aux loss = N * sum(density * probability)
        aux_loss = self.n_experts * torch.sum(density * prob_per_expert)
        return self.aux_loss_weight * aux_loss

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: [batch_size, seq_len, n_embd]
        Returns:
            dispatch_weights: [num_tokens, n_experts, capacity]
            expert_batches: [n_experts, capacity, n_embd]
            aux_loss: Scalar or None
        """
        batch_size, seq_len, n_embd = x.size()
        num_tokens = batch_size * seq_len
        x_flat = x.view(num_tokens, n_embd)

        # 1. Gate scores
        gate_scores = self._compute_gate_scores(x_flat)

        # 2. Router Z-loss
        z_loss = None
        if self.use_router_z_loss and self.training:
            z_loss = self.compute_router_z_loss(
                gate_scores) * self.config.router_z_loss_weight

        # 3. Select experts
        expert_weights, expert_indices = self._select_experts(gate_scores)

        # 4. Aux loss
        aux_loss = None
        if self.use_aux_loss and self.training:
            aux_loss = self.compute_aux_loss(expert_weights, expert_indices)

        # Combine losses
        total_aux_loss = None
        if z_loss is not None and aux_loss is not None:
            total_aux_loss = z_loss + aux_loss
        elif z_loss is not None:
            total_aux_loss = z_loss
        elif aux_loss is not None:
            total_aux_loss = aux_loss

        # 5. Dispatch tokens
        expert_capacity = self._compute_expert_capacity(num_tokens)

        dispatch_weights, expert_batches = self._dispatch_tokens_optimized(
            x_flat, expert_weights, expert_indices, expert_capacity, n_embd)

        return dispatch_weights, expert_batches, total_aux_loss

    def _dispatch_tokens_optimized(
        self,
        x_flat: torch.Tensor,
        expert_weights: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_capacity: int,
        n_embd: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Optimized token dispatching."""
        num_tokens = x_flat.size(0)
        top_k = self.top_k

        # Flatten assignments
        flat_expert_indices = expert_indices.view(-1)  # [num_tokens * top_k]
        flat_expert_weights = expert_weights.view(-1)  # [num_tokens * top_k]

        # Token indices corresponding to each assignment
        token_indices = torch.arange(
            num_tokens, device=x_flat.device).repeat_interleave(top_k)

        # Assign positions within experts
        # We need to assign a unique position [0, capacity-1] for each token assigned to an expert.
        # We can do this by counting occurrences.

        expert_positions = torch.zeros_like(flat_expert_indices)

        # TODO: Optimize this loop for large n_experts?
        # For typical n_experts (8-64), this loop is fast enough.
        for i in range(self.n_experts):
            mask = flat_expert_indices == i
            if mask.any():
                # Assign 0, 1, 2... to tokens assigned to expert i
                expert_positions[mask] = torch.arange(mask.sum(),
                                                      device=x_flat.device)

        # Filter assignments that exceed capacity
        valid_mask = expert_positions < expert_capacity

        # Apply mask
        valid_token_indices = token_indices[valid_mask]
        valid_expert_indices = flat_expert_indices[valid_mask]
        valid_expert_positions = expert_positions[valid_mask]
        valid_expert_weights = flat_expert_weights[valid_mask]

        # Create dispatch weights (sparse tensor effectively)
        # Shape: [num_tokens, n_experts, capacity]
        dispatch_weights = torch.zeros(num_tokens,
                                       self.n_experts,
                                       expert_capacity,
                                       dtype=x_flat.dtype,
                                       device=x_flat.device)

        if valid_mask.any():
            dispatch_weights[valid_token_indices, valid_expert_indices,
                             valid_expert_positions] = valid_expert_weights

        # Create expert batches
        # Shape: [n_experts, capacity, n_embd]
        expert_batches = torch.zeros(self.n_experts,
                                     expert_capacity,
                                     n_embd,
                                     dtype=x_flat.dtype,
                                     device=x_flat.device)

        if valid_mask.any():
            # Gather tokens for valid assignments
            # Note: A token might be copied to multiple experts if top_k > 1
            tokens_to_assign = x_flat[valid_token_indices]

            # Use index_put_ or direct assignment
            # We can treat expert_batches as [n_experts * capacity, n_embd] for flatter assignment
            # Flatten indices: global_pos = expert_idx * capacity + pos_in_expert

            # However, direct indexing on dimensions works:
            expert_batches[valid_expert_indices,
                           valid_expert_positions] = tokens_to_assign

        return dispatch_weights, expert_batches


class MOELayer(nn.Module):
    """Mixture of Experts layer."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.router = Router(config)
        self.experts = MLPExperts(config)

    def forward(
            self,
            x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: [batch_size, seq_len, n_embd]
        Returns:
            output: [batch_size, seq_len, n_embd]
            aux_loss: scalar or None
        """
        batch_size, seq_len, n_embd = x.size()

        # 1. Route
        # dispatch_weights: [num_tokens, n_experts, capacity]
        # expert_batches: [n_experts, capacity, n_embd]
        dispatch_weights, expert_batches, aux_loss = self.router(x)

        # 2. Process with experts
        # expert_outputs: [n_experts, capacity, n_embd]
        expert_outputs = self.experts(expert_batches)

        # 3. Aggregate
        # output = sum(weight * expert_output)
        # dispatch_weights: [T, E, C]
        # expert_outputs: [E, C, M]
        # output: [T, M]
        output = torch.einsum('tec,ecm->tm', dispatch_weights, expert_outputs)

        return output.view(batch_size, seq_len, n_embd), aux_loss


class MoEBlock(nn.Module):
    """Transformer block with MoE layer."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.moe = MOELayer(config)

    def forward(
            self,
            x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = x + self.attn(self.ln_1(x))
        moe_out, aux_loss = self.moe(self.ln_2(x))
        x = x + moe_out
        return x, aux_loss


class GPT(nn.Module):
    """GPT Language Model with optional Mixture of Experts layers."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        # Determine which layers should use MoE
        if config.use_moe and config.moe_layers is None:
            moe_layers = list(range(config.n_layer // 2, config.n_layer))
        else:
            moe_layers = config.moe_layers or []

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                blocks=nn.ModuleList([
                    MoEBlock(config) if i in moe_layers else Block(config)
                    for i in range(config.n_layer)
                ]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # Weight tying

        self.apply(self._init_weights)

        # Special initialization for residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p,
                                      mean=0.0,
                                      std=0.02 / math.sqrt(2 * config.n_layer))

        logger.info(
            f'Number of parameters: {self.get_num_params() / 1e6:.2f}M')

    def get_num_params(self, non_embedding: bool = True) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            if self.config.use_switch_tfm_init:
                scale = self.config.switch_tfm_init_scale
                w_fan_in = module.weight.shape[-1]
                w_std = (scale / w_fan_in)**0.5
                torch.nn.init.trunc_normal_(module.weight,
                                            mean=0.0,
                                            std=w_std,
                                            a=-2 * w_std,
                                            b=2 * w_std)
            else:
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, MLPExperts):
            # Experts initialized in their own class, but we re-init here to respect global config/seeds if needed
            # Actually MLPExperts calls _init_weights in __init__, so we might skip or re-do.
            # Re-doing ensures consistency if GPT uses specific init logic (like switch_tfm_init).
            pass  # MLPExperts handles its own init, and if we want switch init, we should pass it to MLPExperts
            # However, the original code manually re-inits MLPExperts here.
            # Let's keep the manual re-init logic if it differs from MLPExperts default.

            # The original code logic for MLPExperts init was:
            if self.config.use_switch_tfm_init:
                scale = self.config.switch_tfm_init_scale
                # ... trunc_normal ...
                # I'll rely on MLPExperts._init_weights usually, but since switch_tfm_init logic is here:
                c_fc_fan_in = module.c_fc.shape[
                    -2]  # [experts, in, out] -> in is -2
                c_fc_std = (scale / c_fc_fan_in)**0.5
                torch.nn.init.trunc_normal_(module.c_fc,
                                            mean=0.0,
                                            std=c_fc_std,
                                            a=-2 * c_fc_std,
                                            b=2 * c_fc_std)

                c_proj_fan_in = module.c_proj.shape[
                    -2]  # [experts, mid, out] -> mid is -2
                c_proj_std = (scale / c_proj_fan_in)**0.5
                torch.nn.init.trunc_normal_(module.c_proj,
                                            mean=0.0,
                                            std=c_proj_std,
                                            a=-2 * c_proj_std,
                                            b=2 * c_proj_std)
            # If not switch init, MLPExperts.__init__ already did normal(0, 0.02)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = idx.device
        b, t = idx.size()

        if t > self.config.block_size:
            raise ValueError(
                f'Sequence length {t} exceeds block size {self.config.block_size}'
            )

        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # Token + Position embeddings
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Blocks
        total_aux_loss = 0.0
        aux_loss_count = 0

        for block in self.transformer.blocks:
            if isinstance(block, MoEBlock):
                x, aux_loss = block(x)
                if aux_loss is not None:
                    total_aux_loss += aux_loss
                    aux_loss_count += 1
            else:
                x = block(x)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1),
                                   ignore_index=-1)
            if aux_loss_count > 0:
                loss = loss + total_aux_loss / aux_loss_count
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def crop_block_size(self, block_size: int) -> None:
        """Reduce block size."""
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(
            self.transformer.wpe.weight[:block_size])
        for block in self.transformer.blocks:
            if hasattr(block, 'attn') and hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :
                                                  block_size]

    @torch.no_grad()
    def generate(self,
                 idx: torch.Tensor,
                 max_new_tokens: int,
                 temperature: float = 1.0,
                 top_k: Optional[int] = None) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(
                1) <= self.config.block_size else idx[:,
                                                      -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        """Estimate model flops utilization (MFU)."""
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12
        return flops_achieved / flops_promised


def analyze_moe_usage(model: GPT) -> Dict[str, Any]:
    """Analyze MoE usage."""
    moe_layers = [
        i for i, b in enumerate(model.transformer.blocks)
        if isinstance(b, MoEBlock)
    ]
    return {
        'total_layers': len(model.transformer.blocks),
        'moe_layers': moe_layers,
        'moe_layer_count': len(moe_layers),
        'total_parameters': model.get_num_params(),
        'uses_moe': len(moe_layers) > 0,
    }


def get_moe_layer_info(model: GPT, layer_idx: int) -> Optional[Dict[str, Any]]:
    """Get info for specific MoE layer."""
    if layer_idx >= len(model.transformer.blocks):
        return None
    block = model.transformer.blocks[layer_idx]
    if not isinstance(block, MoEBlock):
        return None

    moe = block.moe
    return {
        'layer_idx': layer_idx,
        'n_experts': moe.router.n_experts,
        'top_k': moe.router.top_k,
        'capacity_factor': moe.router.capacity_factor,
        'expert_params': sum(p.numel() for p in moe.experts.parameters()),
        'router_params': sum(p.numel() for p in moe.router.parameters()),
    }
