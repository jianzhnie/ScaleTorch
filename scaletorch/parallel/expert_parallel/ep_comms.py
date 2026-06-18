"""Expert parallel all-to-all communication for MoE token dispatch."""

from __future__ import annotations

import torch
import torch.distributed as dist

from scaletorch.parallel.process_group import process_group_manager as pgm
from scaletorch.utils.logger_utils import get_logger

logger = get_logger(__name__)


def all_to_all(input_tensor: torch.Tensor,
               output_splits: list[int],
               input_splits: list[int],
               group=None) -> torch.Tensor:
    """All-to-all with variable-length splits on the EP group.

    Args:
        input_tensor: 1-D contiguous tensor (concatenated per-rank chunks).
        output_splits: list of sizes to *receive* from each rank.
        input_splits:  list of sizes to *send* to each rank.
        group: process group (defaults to EP group).
    """
    if group is None:
        group = pgm.ep_group
    output_tensor = input_tensor.new_empty(sum(output_splits))
    dist.all_to_all_single(
        output_tensor, input_tensor,
        output_split_sizes=output_splits,
        input_split_sizes=input_splits,
        group=group)
    return output_tensor


def dispatch_tokens(
    hidden_states: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    ep_size: int,
    ep_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], list[int], torch.Tensor]:
    """Dispatch tokens to EP ranks that own the selected experts.

    Args:
        hidden_states: [num_tokens, hidden_size]
        topk_indices:  [num_tokens, top_k]  — global expert ids (0..num_experts-1)
        topk_weights:  [num_tokens, top_k]  — gating weights
        num_experts:   total number of experts
        ep_size:       expert parallel world size
        ep_rank:       this rank's expert parallel index

    Returns:
        recv_tokens:   tokens received from all ranks [total_recv, hidden]
        recv_expert_ids: local expert ids for received tokens [total_recv]
        recv_weights:  gating weights for received tokens [total_recv]
        send_splits:   number of (token, top_k) entries sent to each rank
        recv_splits:   number of entries received from each rank
        reorder_idx:   index to scatter results back after expert computation
    """
    num_tokens, top_k = topk_indices.shape
    hidden = hidden_states.shape[-1]
    experts_per_rank = num_experts // ep_size
    device = hidden_states.device

    # For each (token, expert) pair, find destination EP rank
    dest_rank = topk_indices // experts_per_rank  # [num_tokens, top_k]

    # Flatten to 1-D list of assignments
    flat_tokens_idx = torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, top_k).reshape(-1)
    flat_expert_ids = topk_indices.reshape(-1)  # global
    flat_weights = topk_weights.reshape(-1)
    flat_dest = dest_rank.reshape(-1)

    # Sort by destination rank for contiguous all-to-all sends
    sort_idx = torch.argsort(flat_dest, stable=True)
    flat_tokens_idx = flat_tokens_idx[sort_idx]
    flat_expert_ids = flat_expert_ids[sort_idx]
    flat_weights = flat_weights[sort_idx]
    flat_dest = flat_dest[sort_idx]

    # Compute send splits (how many entries go to each EP rank)
    send_splits = [(flat_dest == r).sum().item() for r in range(ep_size)]

    # Exchange split counts so each rank knows how much to receive
    send_counts = torch.tensor(send_splits, device=device, dtype=torch.long)
    recv_counts = torch.empty_like(send_counts)
    dist.all_to_all_single(recv_counts, send_counts, group=pgm.ep_group)
    recv_splits = recv_counts.tolist()

    # All-to-all: exchange hidden states
    send_hidden = hidden_states[flat_tokens_idx].contiguous()  # [total_send, hidden]
    send_hidden_flat = send_hidden.reshape(-1)
    recv_hidden_flat = all_to_all(
        send_hidden_flat,
        output_splits=[s * hidden for s in recv_splits],
        input_splits=[s * hidden for s in send_splits])
    recv_tokens = recv_hidden_flat.reshape(-1, hidden)

    # All-to-all: exchange expert ids (convert to local ids)
    send_ids = flat_expert_ids.contiguous()
    recv_ids = all_to_all(send_ids, recv_splits, send_splits)
    recv_expert_ids = recv_ids % experts_per_rank  # convert to local expert id

    # All-to-all: exchange weights
    send_w = flat_weights.contiguous()
    recv_weights = all_to_all(send_w, recv_splits, send_splits)

    # Save reorder info to scatter results back
    reorder_idx = sort_idx

    return recv_tokens, recv_expert_ids, recv_weights, send_splits, recv_splits, reorder_idx


def gather_tokens(
    expert_output: torch.Tensor,
    send_splits: list[int],
    recv_splits: list[int],
    reorder_idx: torch.Tensor,
    num_tokens: int,
    top_k: int,
    hidden: int,
) -> torch.Tensor:
    """Gather expert outputs back to the original EP ranks.

    Args:
        expert_output: [total_local, hidden] — outputs from local experts
        send_splits:   original send_splits from dispatch (now recv for gather)
        recv_splits:   original recv_splits from dispatch (now send for gather)
        reorder_idx:   sort index from dispatch to undo the reordering
        num_tokens:    original number of tokens
        top_k:         number of experts per token
        hidden:        hidden dimension

    Returns:
        gathered: [num_tokens * top_k, hidden] in original token order
    """
    # All-to-all: send expert outputs back (reverse direction)
    send_flat = expert_output.reshape(-1).contiguous()
    recv_flat = all_to_all(
        send_flat,
        output_splits=[s * hidden for s in send_splits],
        input_splits=[s * hidden for s in recv_splits])
    recv = recv_flat.reshape(-1, hidden)

    # Undo the sort reordering
    output = torch.empty_like(recv)
    output[reorder_idx] = recv
    return output
