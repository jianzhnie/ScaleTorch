"""Tensor parallel communication primitives: copy, reduce, gather, and async all-reduce linear layers."""

from __future__ import annotations

import torch
import torch.nn.functional as F

import scaletorch.dist as st_dist
from scaletorch.parallel.process_group import process_group_manager as pgm


def merge_first_two_dims(
    grad_output: torch.Tensor, input_: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Merge the first two dimensions of tensors for efficient matrix operations.

    Args:
        grad_output: Gradient tensor with shape (batch, seq, ...)
        input_: Input tensor with shape (batch, seq, ...)

    Returns:
        Tuple of tensors with merged first two dimensions
    """
    return grad_output.contiguous().view(
        -1, *grad_output.shape[2:]
    ), input_.contiguous().view(-1, *input_.shape[2:])


def split_tensor_along_last_dim(
    tensor: torch.Tensor, num_partitions: int
) -> list[torch.Tensor]:
    """
    Split a tensor along its last dimension into num_partitions chunks.

    Args:
        tensor: Input tensor to split
        num_partitions: Number of partitions to split the tensor into

    Returns:
        List of tensor chunks

    Raises:
        ValueError: If tensor's last dimension is not divisible by num_partitions
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")

    if not isinstance(num_partitions, int) or num_partitions <= 0:
        raise ValueError(
            f"num_partitions must be positive integer, got {num_partitions}"
        )

    last_dim = tensor.dim() - 1
    if tensor.size()[last_dim] % num_partitions != 0:
        raise ValueError(
            f"Tensor last dimension {tensor.size()[last_dim]} is not divisible by {num_partitions}"
        )

    last_dim_size = tensor.size()[last_dim] // num_partitions
    return torch.split(tensor, last_dim_size, dim=last_dim)


class CopyToModelParallelRegion(torch.autograd.Function):
    """
    Copy in forward pass, all-reduce in backward pass.

    This implements the `f` function from the paper:
    "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism"
    https://arxiv.org/abs/1909.08053

    This function is used in row-parallel linear layers where the input needs to be
    copied to all tensor parallel ranks in forward pass, and gradients need to be
    all-reduced in backward pass.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: simply return the input tensor.

        Args:
            x: Input tensor

        Returns:
            Same tensor as input
        """
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Backward pass: all-reduce gradients across tensor parallel group.

        Args:
            grad_output: Gradient tensor

        Returns:
            All-reduced gradient tensor
        """
        if pgm.tp_world_size == 1:
            return grad_output

        try:
            # Ensure tensor is contiguous for efficient communication
            if not grad_output.is_contiguous():
                grad_output = grad_output.contiguous()
            st_dist.all_reduce(grad_output, op="sum", group=pgm.tp_group)
        except Exception as e:
            raise RuntimeError(
                f"Failed to all-reduce gradients (shape={grad_output.shape}): {e}"
            ) from e

        return grad_output


class ReduceFromModelParallelRegion(torch.autograd.Function):
    """
    All-reduce in forward pass, identity in backward pass.

    This implements the `g` function from the Megatron-LM paper.
    Used in column-parallel linear layers where outputs need to be
    all-reduced across tensor parallel ranks in forward pass.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: all-reduce input tensor across tensor parallel group.

        Uses an out-of-place copy so that the original input is not modified
        when other consumers (e.g. residual connections) still reference it.

        Args:
            x: Input tensor

        Returns:
            All-reduced tensor (a new tensor; *x* is not mutated)
        """
        if pgm.tp_world_size == 1:
            return x

        try:
            output = x.clone()
            if not output.is_contiguous():
                output = output.contiguous()
            st_dist.all_reduce(output, op="sum", group=pgm.tp_group)
        except Exception as e:
            raise RuntimeError(
                f"Failed to all-reduce tensor (shape={x.shape}): {e}"
            ) from e

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Backward pass: simply return the gradient as-is.

        Args:
            grad_output: Gradient tensor

        Returns:
            Same gradient tensor
        """
        return grad_output


class GatherFromModelParallelRegion(torch.autograd.Function):
    """
    Gather in forward pass, split in backward pass.

    Used in tensor parallel operations where tensors need to be gathered
    from all ranks in forward pass and split in backward pass.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: gather tensors from all tensor parallel ranks.

        Args:
            x: Input tensor from current rank

        Returns:
            Gathered tensor from all ranks concatenated along last dimension
        """
        if pgm.tp_world_size == 1:
            return x

        last_dim = x.dim() - 1
        # Need contiguous tensors for collectives
        # Reference: https://github.com/pytorch/pytorch/blob/main/torch/distributed/nn/functional.py#L321
        if not x.is_contiguous():
            x = x.contiguous()

        try:
            tensor_list = st_dist.all_gather(x, group=pgm.tp_group)
            output = torch.cat(tensor_list, dim=last_dim).contiguous()
        except Exception as e:
            raise RuntimeError(
                f"Failed to gather tensors (shape={x.shape}, tp_world_size={pgm.tp_world_size}): {e}"
            ) from e

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Backward pass: split gradient according to tensor parallel size.

        Args:
            grad_output: Gradient tensor

        Returns:
            Split gradient chunk for current rank
        """
        if pgm.tp_world_size == 1:
            return grad_output

        try:
            # Split gradient according to TP size
            chunks = split_tensor_along_last_dim(grad_output, pgm.tp_world_size)
            return chunks[pgm.tp_rank].contiguous()
        except Exception as e:
            raise RuntimeError(f"Failed to split gradient: {e}") from e


class LinearWithAsyncAllReduce(torch.autograd.Function):
    """
    Linear layer with asynchronous all-reduce for gradient overlap.

    This implementation overlaps computation and communication by performing
    the all-reduce of input gradients before calculating weight and bias gradients.

    Key difference from synchronous version:
    Before: grad_output -> grad_input, grad_weight, grad_bias -> grad_input all_reduce
    Now:    grad_output -> grad_input -> grad_input all_reduce -> grad_weight, grad_bias

    This is particularly beneficial for Column Parallel Linear layers.
    """

    @staticmethod
    def forward(
        ctx,
        input_: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass: perform linear transformation.

        Args:
            input_: Input tensor with shape (batch, seq, input_size)
            weight: Weight tensor with shape (output_size, input_size)
            bias: Optional bias tensor with shape (output_size,)

        Returns:
            Output tensor with shape (batch, seq, output_size)
        """
        ctx.save_for_backward(input_, weight)
        ctx.use_bias = bias is not None

        try:
            output = F.linear(input_, weight, bias)
            # input_ @ weight.t() + bias
        except Exception as e:
            raise RuntimeError(f"Failed to compute linear transformation: {e}") from e

        return output

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Backward pass with asynchronous all-reduce for gradient overlap.

        The key optimization is to start the all-reduce of input gradients early,
        allowing it to overlap with weight and bias gradient computations.

        Args:
            grad_output: Gradient tensor with shape (batch, seq, output_size)

        Returns:
            Tuple of (grad_input, grad_weight, grad_bias)
        """
        input_, weight = ctx.saved_tensors

        try:
            # Compute input gradient: (b, s, out_size) @ (out_size, input_size) = (b, s, input_size)
            grad_input = grad_output @ weight

            # Start asynchronous all-reduce of input gradient
            # Ensure tensor is contiguous for efficient communication
            input_gradient_all_reduce_handle = None
            if pgm.tp_world_size > 1:
                if not grad_input.is_contiguous():
                    grad_input = grad_input.contiguous()
                input_gradient_all_reduce_handle = st_dist.all_reduce(
                    grad_input, group=pgm.tp_group, async_op=True
                )

            # Merge first two dimensions for efficient matrix multiplication
            grad_output_flat, input_flat = merge_first_two_dims(grad_output, input_)

            # Compute weight gradient: (out_size, b*s) @ (b*s, input_size) -> (out_size, input_size)
            grad_weight = grad_output_flat.t() @ input_flat

            # Compute bias gradient if needed
            grad_bias = grad_output_flat.sum(0) if ctx.use_bias else None

            # Wait for asynchronous all-reduce to complete
            if input_gradient_all_reduce_handle is not None:
                input_gradient_all_reduce_handle.wait()

        except Exception as e:
            raise RuntimeError(f"Failed to compute backward pass: {e}") from e

        return grad_input, grad_weight, grad_bias


def linear_with_all_reduce(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Linear layer with all-reduce communication.

    Uses CopyToModelParallelRegion to handle gradient synchronization
    across tensor parallel ranks.

    Args:
        x: Input tensor
        weight: Weight tensor
        bias: Optional bias tensor

    Returns:
        Output of linear transformation with gradient synchronization
    """
    input_parallel = CopyToModelParallelRegion.apply(x)
    return F.linear(input_parallel, weight, bias)


def linear_with_async_all_reduce(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Linear layer with asynchronous all-reduce for gradient overlap.

    Uses LinearWithAsyncAllReduce to overlap computation and communication.

    Args:
        x: Input tensor
        weight: Weight tensor
        bias: Optional bias tensor

    Returns:
        Output of linear transformation with asynchronous gradient synchronization
    """
    return LinearWithAsyncAllReduce.apply(x, weight, bias)
