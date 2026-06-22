"""Sequence parallel communication primitives: all-gather and reduce-scatter along sequence dimension."""

from __future__ import annotations

import torch

import scaletorch.dist as st_dist
from scaletorch.parallel.process_group import process_group_manager as pgm

SEQ_DIM = 1


def _gather_along_seq_dim(tensor_list: list[torch.Tensor]) -> torch.Tensor:
    """Concatenate gathered tensors along the sequence dimension (dim=1)."""
    return torch.cat(tensor_list, dim=SEQ_DIM)


def _split_along_seq_dim(
    tensor: torch.Tensor, num_partitions: int
) -> list[torch.Tensor]:
    """Split tensor along the sequence dimension into equal chunks."""
    seq_len = tensor.size(SEQ_DIM)
    if seq_len % num_partitions != 0:
        raise ValueError(
            f"Sequence length {seq_len} is not divisible by {num_partitions}"
        )
    chunk_size = seq_len // num_partitions
    return list(torch.split(tensor, chunk_size, dim=SEQ_DIM))


class AllGatherFromSequenceParallelRegion(torch.autograd.Function):
    """All-gather along sequence dim in forward, reduce-scatter in backward.

    Used before column-parallel layers to reconstruct the full sequence from
    sequence-partitioned activations.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        if not pgm or pgm.tp_world_size == 1:
            return x

        if not x.is_contiguous():
            x = x.contiguous()

        tensor_list = st_dist.all_gather(x, group=pgm.tp_group)
        return _gather_along_seq_dim(tensor_list)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        if not pgm or pgm.tp_world_size == 1:
            return grad_output

        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()

        tp_world_size = pgm.tp_world_size
        chunks = _split_along_seq_dim(grad_output, tp_world_size)
        grad_partition = torch.empty_like(chunks[0])
        st_dist.reduce_scatter(grad_partition, chunks, group=pgm.tp_group)
        return grad_partition


class ReduceScatterToSequenceParallelRegion(torch.autograd.Function):
    """Reduce-scatter along sequence dim in forward, all-gather in backward.

    Used after row-parallel layers to partition the output along the sequence
    dimension across TP ranks.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        if not pgm or pgm.tp_world_size == 1:
            return x

        if not x.is_contiguous():
            x = x.contiguous()

        tp_world_size = pgm.tp_world_size
        chunks = _split_along_seq_dim(x, tp_world_size)
        output = torch.empty_like(chunks[0])
        st_dist.reduce_scatter(output, chunks, group=pgm.tp_group)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        if not pgm or pgm.tp_world_size == 1:
            return grad_output

        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()

        tensor_list = st_dist.all_gather(grad_output, group=pgm.tp_group)
        return _gather_along_seq_dim(tensor_list)
