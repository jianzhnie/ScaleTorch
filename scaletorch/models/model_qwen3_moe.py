"""Qwen3 Mixture-of-Experts (MoE) model with Expert Parallelism support.

Architecture (Qwen3-30B-A3B):
  - 48 layers, hidden=2048, heads=32, kv_heads=4, head_dim=128
  - 128 experts per layer, top-8 routing, moe_intermediate=768
  - All layers are MoE (decoder_sparse_step=1)
  - Supports EP (expert parallel): experts sharded across EP ranks
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from scaletorch.models.llama import (
    FinalProjection,
    LlamaEmbedding,
    RMSNorm,
    get_cos_sin,
)
from scaletorch.models.model_qwen3 import Qwen3Attention
from scaletorch.parallel.process_group import process_group_manager as pgm
from scaletorch.utils.logger_utils import get_logger

logger = get_logger(__name__)


class MoERouter(nn.Module):
    """Top-k gating router for MoE with optional load-balancing auxiliary loss."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        norm_topk_prob: bool = True,
        aux_loss_coef: float = 0.001,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_coef = aux_loss_coef
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, hidden_states: torch.Tensor):
        """Compute top-k expert routing.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]

        Returns:
            topk_weights: [num_tokens, top_k]
            topk_indices: [num_tokens, top_k]
            aux_loss:     scalar load-balancing loss (0 if disabled)
        """
        bsz, seq_len, hidden = hidden_states.shape
        hidden_flat = hidden_states.reshape(-1, hidden)
        logits = self.gate(hidden_flat).float()  # [num_tokens, num_experts]

        topk_weights, topk_indices = torch.topk(
            logits, self.top_k, dim=-1
        )  # [num_tokens, top_k]
        topk_weights = F.softmax(topk_weights, dim=-1)

        if self.norm_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        topk_weights = topk_weights.to(hidden_states.dtype)

        # Load-balancing auxiliary loss (Switch Transformer style)
        aux_loss = torch.tensor(0.0, device=hidden_states.device)
        if self.training and self.aux_loss_coef > 0:
            probs = F.softmax(logits, dim=-1)
            # f_i = fraction of tokens routed to expert i
            tokens_per_expert = torch.zeros(self.num_experts, device=logits.device)
            tokens_per_expert.scatter_add_(
                0,
                topk_indices.reshape(-1),
                torch.ones(topk_indices.numel(), device=logits.device),
            )
            f = tokens_per_expert / (bsz * seq_len)
            # p_i = mean routing probability for expert i
            p = probs.mean(dim=0)
            aux_loss = self.aux_loss_coef * self.num_experts * (f * p).sum()

        return topk_weights, topk_indices, aux_loss

    def reset_parameters(self):
        nn.init.normal_(self.gate.weight, std=0.02)


# --- PLACEHOLDER:MOELAYER ---


class MoEExperts(nn.Module):
    """Individual expert MLPs matching HuggingFace Qwen3MoE weight format.

    Each expert is a separate nn.Module with gate_proj, up_proj, down_proj
    so that `load_state_dict` can directly map HuggingFace weights.
    """

    def __init__(self, num_experts: int, hidden_size: int, moe_intermediate_size: int):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList(
            [
                self._make_expert(hidden_size, moe_intermediate_size)
                for _ in range(num_experts)
            ]
        )

    @staticmethod
    def _make_expert(hidden_size, intermediate_size):
        expert = nn.Module()
        expert.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        expert.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        expert.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        return expert

    def reset_parameters(self):
        for expert in self.experts:
            nn.init.normal_(expert.gate_proj.weight, std=0.02)
            nn.init.normal_(expert.up_proj.weight, std=0.02)
            nn.init.normal_(expert.down_proj.weight, std=0.02)

    def forward(
        self, hidden_states: torch.Tensor, expert_ids: torch.Tensor
    ) -> torch.Tensor:
        """Process tokens through assigned local experts.

        Uses scatter_add for NPU compatibility (avoids index_put/index_copy).

        Args:
            hidden_states: [num_tokens, hidden_size]
            expert_ids:    [num_tokens] — local expert indices

        Returns:
            output: [num_tokens, hidden_size]
        """
        num_tokens, hidden = hidden_states.shape
        # Group tokens per expert, process, then scatter results back
        results = []
        indices = []
        for idx, expert in enumerate(self.experts):
            token_idx = (expert_ids == idx).nonzero(as_tuple=False).squeeze(-1)
            if token_idx.numel() == 0:
                continue
            tokens = hidden_states[token_idx]
            gate = F.silu(expert.gate_proj(tokens))
            up = expert.up_proj(tokens)
            expert_out = expert.down_proj(gate * up)
            results.append(expert_out)
            indices.append(token_idx)

        if not results:
            return torch.zeros_like(hidden_states)

        # Concatenate results and scatter back into output tensor
        all_results = torch.cat(results, dim=0)  # [total, hidden]
        all_indices = torch.cat(indices, dim=0)  # [total]

        # Use scatter via expanded index (NPU-compatible, no in-place indexed assignment)
        output = torch.zeros(
            num_tokens, hidden, dtype=all_results.dtype, device=all_results.device
        )
        idx_expanded = all_indices.unsqueeze(-1).expand_as(all_results)
        output.scatter_(0, idx_expanded, all_results)
        return output


class MoELayer(nn.Module):
    """Mixture-of-Experts layer with optional Expert Parallelism.

    When EP is enabled (ep_size > 1):
      - Experts are sharded: each EP rank holds num_experts // ep_size experts.
      - Router is replicated on all EP ranks.
      - Token dispatch uses all-to-all across EP group.
    When EP is disabled (ep_size = 1):
      - All experts are local, no communication needed.
    """

    def __init__(self, config: Any):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.moe_intermediate_size = config.moe_intermediate_size

        ep_size = pgm.ep_world_size if pgm else 1
        ep_rank = pgm.ep_rank if pgm else 0
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.experts_per_rank = self.num_experts // ep_size

        self.router = MoERouter(
            config.hidden_size,
            self.num_experts,
            self.top_k,
            norm_topk_prob=getattr(config, "norm_topk_prob", True),
            aux_loss_coef=getattr(config, "router_aux_loss_coef", 0.001),
        )

        self.experts = MoEExperts(
            self.experts_per_rank, config.hidden_size, config.moe_intermediate_size
        )

    def forward(self, hidden_states: torch.Tensor) -> tuple:
        """Forward pass through MoE layer.

        Args:
            hidden_states: [batch, seq_len, hidden_size]

        Returns:
            output: [batch, seq_len, hidden_size]
            aux_loss: scalar
        """
        bsz, seq_len, hidden = hidden_states.shape
        topk_weights, topk_indices, aux_loss = self.router(hidden_states)
        # topk_weights: [num_tokens, top_k]
        # topk_indices: [num_tokens, top_k]  — global expert IDs

        if self.ep_size > 1:
            output = self._forward_ep(
                hidden_states.reshape(-1, hidden), topk_weights, topk_indices
            )
        else:
            output = self._forward_local(
                hidden_states.reshape(-1, hidden), topk_weights, topk_indices
            )

        return output.reshape(bsz, seq_len, hidden), aux_loss

    def _forward_local(self, hidden_flat, topk_weights, topk_indices):
        """No EP: all experts are local."""
        output = torch.zeros_like(hidden_flat)
        for k in range(self.top_k):
            expert_out = self.experts(hidden_flat, topk_indices[:, k])
            output += topk_weights[:, k].unsqueeze(-1) * expert_out
        return output

    def _forward_ep(self, hidden_flat, topk_weights, topk_indices):
        """With EP: dispatch tokens via all-to-all, process, gather."""
        from scaletorch.parallel.expert_parallel.ep_comms import (
            dispatch_tokens,
            gather_tokens,
        )

        num_tokens, hidden = hidden_flat.shape

        # Dispatch: send tokens to EP ranks owning the selected experts
        (
            recv_tokens,
            recv_expert_ids,
            recv_weights,
            send_splits,
            recv_splits,
            reorder_idx,
        ) = dispatch_tokens(
            hidden_flat,
            topk_indices,
            topk_weights,
            self.num_experts,
            self.ep_size,
            self.ep_rank,
        )

        # Process local experts on received tokens
        expert_output = self.experts(recv_tokens, recv_expert_ids)
        # Apply weights
        expert_output = expert_output * recv_weights.unsqueeze(-1)

        # Gather: send outputs back to original EP ranks
        gathered = gather_tokens(
            expert_output,
            send_splits,
            recv_splits,
            reorder_idx,
            num_tokens,
            self.top_k,
            hidden,
        )

        # Sum over top_k contributions
        gathered = gathered.reshape(num_tokens, self.top_k, hidden)
        return gathered.sum(dim=1)

    def reset_parameters(self):
        self.router.reset_parameters()
        self.experts.reset_parameters()


class Qwen3MoEDecoderLayer(nn.Module):
    """Qwen3 MoE decoder layer: Attention + MoE (replaces dense MLP)."""

    def __init__(self, config: Any, layer_idx: int):
        super().__init__()

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.attention = Qwen3Attention(config, layer_idx=layer_idx)
        self.moe = MoELayer(config)
        self.layer_idx = layer_idx

        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        rope_theta = getattr(config, "rope_theta", 1000000.0)
        cos, sin = get_cos_sin(
            config.max_position_embeddings,
            head_dim=head_dim,
            base=rope_theta,
            device=torch.device("cpu"),
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        from scaletorch.parallel.context_parallel import context_parallel

        cos, sin = context_parallel.update_rope_for_context_parallel(self.cos, self.sin)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x, attention_mask=None, position_ids=None):
        seq_len = x.size(1)
        cos = self.cos[:seq_len]
        sin = self.sin[:seq_len]

        x = x + self.attention(
            self.input_layernorm(x), cos, sin, attention_mask, position_ids
        )

        moe_out, aux_loss = self.moe(self.post_attention_layernorm(x))
        x = x + moe_out

        # Store aux_loss for later aggregation
        self._aux_loss = aux_loss
        return x


class Qwen3MoE(nn.Module):
    """Qwen3 MoE transformer model."""

    def __init__(self, config: Any) -> None:
        super().__init__()

        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_hidden_layers
        self.config = config
        self.model_config = config

        self.embedding = LlamaEmbedding(self.vocab_size, self.hidden_size)
        self.decoder_layers = nn.ModuleList(
            [Qwen3MoEDecoderLayer(config, layer_idx=i) for i in range(self.num_layers)]
        )

        self.final_norm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.final_proj = FinalProjection(self.hidden_size, self.vocab_size, bias=False)

        if getattr(config, "tie_word_embeddings", False):
            self.final_proj.weight = self.embedding.weight

    def reset_parameters(self) -> None:
        self.embedding.reset_parameters()
        for layer in self.decoder_layers:
            layer.input_layernorm.reset_parameters()
            layer.post_attention_layernorm.reset_parameters()
            layer.attention.reset_parameters()
            layer.moe.reset_parameters()
        self.final_norm.reset_parameters()
        if not getattr(self.config, "tie_word_embeddings", False):
            self.final_proj.reset_parameters()

    def get_aux_loss(self) -> torch.Tensor:
        """Aggregate auxiliary losses from all MoE layers."""
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.decoder_layers:
            if hasattr(layer, "_aux_loss"):
                total = total + layer._aux_loss
        return total

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        gradient_checkpointing=False,
    ):
        x = self.embedding(input_ids)

        if gradient_checkpointing:
            for layer in self.decoder_layers:
                x = torch_checkpoint(
                    layer, x, attention_mask, position_ids, use_reentrant=False
                )
        else:
            for layer in self.decoder_layers:
                x = layer(x, attention_mask, position_ids)

        x = self.final_norm(x)
        logits = self.final_proj(x)
        return logits
