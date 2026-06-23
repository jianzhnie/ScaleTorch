"""Collective communication primitives: scatter, reduce, all_reduce, all_gather, etc."""

from __future__ import annotations

from collections import OrderedDict
from types import GeneratorType
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch import distributed as torch_dist
from torch._utils import (
    _flatten_dense_tensors,
    _take_tensors,
    _unflatten_dense_tensors,
)
from torch.distributed import ProcessGroup

from ._reduce_op import _get_reduce_op
from .utils import (
    cast_data_device,
    get_comm_device,
    get_data_device,
    get_default_group,
    get_rank,
    get_world_size,
)


def scatter(
    tensor_out: Tensor,
    scatter_list: list[Tensor] | None = None,
    src: int = 0,
    group: ProcessGroup | None = None,
) -> None:
    """Scatters tensors from source to all processes in a group.

    Each process will receive exactly one tensor and store its data in the
    ``tensor_out`` argument.

    Complex tensors are supported.

    Note:
        Calling ``scatter`` in non-distributed environment does nothing.

    Args:
        tensor_out (Tensor): Output tensor to store the scattered data.
        scatter_list (list[Tensor] | None): List of tensors to scatter from
            the source rank. Must be the same length as the world size.
            Only used on the source rank. Defaults to None.
        src (int): Source rank. Defaults to 0.
        group (ProcessGroup | None): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Raises:
        ValueError: If scatter_list is not provided on source rank or has incorrect length.
        ValueError: If tensors in scatter_list have different shapes than output.
        TypeError: If tensor_out is not a Tensor, scatter_list is not a list, or src is not an int.

    Examples:
        >>> import torch
        >>> import scaletorch.dist as dist

        >>> # non-distributed environment
        >>> tensor_out = torch.zeros(2, dtype=torch.int64)
        >>> scatter_list = [torch.arange(2, dtype=torch.int64)]
        >>> dist.scatter(tensor_out, scatter_list=scatter_list)
        >>> tensor_out
        tensor([0, 1])

        >>> # distributed environment
        >>> # We have 2 process groups, 2 ranks.
        >>> tensor_out = torch.zeros(2, dtype=torch.int64)
        >>> if dist.get_rank() == 0:
        ...     scatter_list = [torch.tensor([1, 2]), torch.tensor([3, 4])]
        >>> else:
        ...     scatter_list = None
        >>> dist.scatter(tensor_out, scatter_list=scatter_list)
        >>> tensor_out
        tensor([1, 2]) # Rank 0
        tensor([3, 4]) # Rank 1
    """
    if not isinstance(tensor_out, torch.Tensor):
        raise TypeError(f'tensor_out must be a torch.Tensor, got {type(tensor_out)}')
    if scatter_list is not None and not isinstance(scatter_list, list):
        raise TypeError(f'scatter_list must be a list or None, got {type(scatter_list)}')
    if not isinstance(src, int):
        raise TypeError(f'src must be an int, got {type(src)}')

    world_size = get_world_size(group)
    if world_size == 1:
        if scatter_list is not None:
            if not scatter_list:
                raise ValueError('scatter_list must not be empty on source rank')
            tensor_out.copy_(scatter_list[0])
        return

    if group is None:
        group = get_default_group()

    output_device = get_data_device(tensor_out)
    backend_device = get_comm_device(group)
    tensor_on_device = cast_data_device(tensor_out, backend_device)
    this_rank = get_rank(group)

    scatter_list_on_device: list[Tensor] | None = None
    if this_rank == src:
        if scatter_list is None:
            raise ValueError(f'scatter_list must be provided on source rank {src}')
        if len(scatter_list) != world_size:
            raise ValueError(
                f'scatter_list length ({len(scatter_list)}) must equal '
                f'world size ({world_size})')
        scatter_list_on_device = []
        for tensor in scatter_list:
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f'All items in scatter_list must be torch.Tensor, got {type(tensor)}')
            if tensor.shape != tensor_out.shape:
                raise ValueError(
                    f'All tensors in scatter_list must have the same shape as output. '
                    f'Expected {tensor_out.shape}, got {tensor.shape}')
            scatter_list_on_device.append(cast_data_device(tensor, backend_device))
    else:
        scatter_list_on_device = []

    torch_dist.scatter(tensor_on_device, scatter_list_on_device, src, group)
    cast_data_device(tensor_on_device, output_device, out=tensor_out)


def reduce(
    data: Tensor,
    dst: int = 0,
    op: str = 'sum',
    group: ProcessGroup | None = None,
) -> None:
    """Reduces the tensor data across all processes and sends the result to the
    specified destination process.

    After the call, only the destination process will have the final result.

    Note:
        Calling ``reduce`` in non-distributed environment does nothing.

    Args:
        data (Tensor): Input and output of the collective. The function
            operates in-place.
        dst (int): Destination rank to receive the reduced result. Defaults to 0.
        op (str): Operation to reduce data. Defaults to 'sum'. Optional values
            are 'sum', 'mean', 'product', 'min', 'max', 'band', 'bor' and
            'bxor'.
        group (ProcessGroup | None): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Raises:
        TypeError: If data is not a Tensor.
        TypeError: If dst is not an int.
        TypeError: If op is not a str.
        ValueError: If op is not supported.

    Examples:
        >>> import torch
        >>> import scaletorch.dist as dist

        >>> # non-distributed environment
        >>> data = torch.arange(2, dtype=torch.int64)
        >>> dist.reduce(data)
        >>> data
        tensor([0, 1])

        >>> # distributed environment
        >>> # We have 2 process groups, 2 ranks.
        >>> data = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank
        >>> data
        tensor([1, 2]) # Rank 0
        tensor([3, 4]) # Rank 1
        >>> dist.reduce(data, dst=0, op='sum')
        >>> data
        tensor([4, 6]) # Rank 0 (destination process)
        tensor([3, 4]) # Rank 1 (original data remains)
    """
    if not isinstance(data, torch.Tensor):
        raise TypeError(f'data must be a torch.Tensor, got {type(data)}')
    if not isinstance(dst, int):
        raise TypeError(f'dst must be an int, got {type(dst)}')
    if not isinstance(op, str):
        raise TypeError(f'op must be a str, got {type(op)}')

    world_size = get_world_size(group)
    if world_size > 1:
        if group is None:
            group = get_default_group()

        input_device = get_data_device(data)
        backend_device = get_comm_device(group)
        data_on_device = cast_data_device(data, backend_device)

        reduce_op = 'sum' if op.lower() == 'mean' else op
        torch_dist.reduce(data_on_device, dst, _get_reduce_op(reduce_op), group)
        if op.lower() == 'mean' and get_rank(group) == dst:
            data_on_device.div_(world_size)

        if get_rank(group) == dst:
            cast_data_device(data_on_device, input_device, out=data)


def reduce_scatter(
    tensor_out: Tensor,
    input_list: list[Tensor] | None = None,
    op: str = 'sum',
    group: ProcessGroup | None = None,
) -> None:
    """Reduces the tensor data across all machines and scatters the result
    to all processes in a group.

    Each process will receive a portion of the reduced result.

    Note:
        Calling ``reduce_scatter`` in non-distributed environment does nothing.

    Args:
        tensor_out (Tensor): Output tensor to store the scattered reduced result.
        input_list (list[Tensor] | None): List of tensors to be reduced and scattered.
            Its size should be tensor_out size times the world size.
        op (str): Operation to reduce data. Defaults to 'sum'. Optional values
            are 'sum', 'mean' and 'product', 'min', 'max', 'band', 'bor' and
            'bxor'.
        group (ProcessGroup | None): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Raises:
        TypeError: If tensor_out is not a Tensor or input_list is not a list.
        TypeError: If op is not a str.
        ValueError: If input_list is not provided or has incorrect size.

    Examples:
        >>> import torch
        >>> import scaletorch.dist as dist

        >>> # non-distributed environment
        >>> tensor_out = torch.zeros(2, dtype=torch.int64)
        >>> input_list = [torch.arange(2, dtype=torch.int64), torch.arange(2, dtype=torch.int64)]
        >>> dist.reduce_scatter(tensor_out, input_list=input_list)
        >>> tensor_out
        tensor([0, 1])

        >>> # distributed environment
        >>> tensor_out = torch.zeros(4, dtype=torch.int64)
        >>> input_list = [torch.arange(4, dtype=torch.int64), torch.arange(4, dtype=torch.int64)]
        >>> input_list
        [tensor([0, 1, 2, 3]), tensor([0, 1, 2, 3])] # Rank 0
        [tensor([0, 1, 2, 3]), tensor([0, 1, 2, 3])] # Rank 1
        >>> dist.reduce_scatter(tensor_out, input_list)
        >>> tensor_out
        tensor([0, 2]) # Rank 0
        tensor([4, 6]) # Rank 1
    """
    if not isinstance(tensor_out, torch.Tensor):
        raise TypeError(f'tensor_out must be a torch.Tensor, got {type(tensor_out)}')
    if input_list is not None and not isinstance(input_list, list):
        raise TypeError(f'input_list must be a list or None, got {type(input_list)}')
    if not isinstance(op, str):
        raise TypeError(f'op must be a str, got {type(op)}')

    world_size = get_world_size(group)
    if world_size == 1:
        if input_list is not None and len(input_list) > 0:
            tensor_out.copy_(input_list[0])
        return

    if group is None:
        group = get_default_group()

    if (input_list is None or not isinstance(input_list, list)
            or len(input_list) != world_size):
        raise ValueError(
            f'input_list must be a list of {world_size} tensors, but got '
            + str(type(input_list)) + ' with length '
            + (str(len(input_list)) if isinstance(input_list, list) else 'N/A'))

    total_input_size = sum(t.numel() for t in input_list)
    expected_size = tensor_out.numel() * world_size
    if total_input_size != expected_size:
        raise ValueError(
            f'Total input size ({total_input_size}) must equal '
            f'output size * world_size ({expected_size})')

    for idx, tensor in enumerate(input_list):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f'All items in input_list must be torch.Tensor, '
                f'got {type(tensor)} at index {idx}')
        if tensor.shape != tensor_out.shape:
            raise ValueError(
                f'Tensor at index {idx} in input_list must have shape '
                f'{tensor_out.shape}, but got {tensor.shape}')

    output_device = get_data_device(tensor_out)
    backend_device = get_comm_device(group)
    tensor_on_device = cast_data_device(tensor_out, backend_device)
    input_on_device = [cast_data_device(t, backend_device) for t in input_list]

    reduce_op = 'sum' if op.lower() == 'mean' else op
    torch_dist.reduce_scatter(
        tensor_on_device, input_on_device, _get_reduce_op(reduce_op), group)
    if op.lower() == 'mean':
        tensor_on_device.div_(world_size)

    cast_data_device(tensor_on_device, output_device, out=tensor_out)


def all_to_all(
    output_tensor_list: list[Tensor],
    input_tensor_list: list[Tensor],
    group: ProcessGroup | None = None,
) -> list[Tensor]:
    """All-to-All communication operation.

    Scatters list of input tensors to all processes in a group and return gathered list of tensors in output list.

    Note:
        Calling ``all_to_all`` in non-distributed environment does nothing.

    Args:
        output_tensor_list (list[Tensor]): Output tensor list to store the gathered result.
        input_tensor_list (list[Tensor]): Input tensor list to be scattered. Its size
            should be output tensor size times the world size.
        group (ProcessGroup | None): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Returns:
        list[Tensor]: List of output tensors containing chunks from all processes.

    Raises:
        ValueError: If input_tensor_list or output_tensor_list lengths don't match world size.
        TypeError: If input_tensor_list or output_tensor_list are not lists.

    Examples:
        >>> import torch
        >>> import scaletorch.dist as dist

        >>> # non-distributed environment
        >>> input = torch.arange(4, dtype=torch.int64) + rank * 4
        >>> input = list(input.chunk(4))
        >>> input
        [tensor([0]), tensor([1]), tensor([2]), tensor([3])]     # Rank 0
        [tensor([4]), tensor([5]), tensor([6]), tensor([7])]     # Rank 1
        [tensor([8]), tensor([9]), tensor([10]), tensor([11])]   # Rank 2
        [tensor([12]), tensor([13]), tensor([14]), tensor([15])] # Rank 3
        >>> output = list(torch.empty([4], dtype=torch.int64).chunk(4))
        >>> dist.all_to_all(output, input)
        >>> output
        [tensor([0]), tensor([4]), tensor([8]), tensor([12])]    # Rank 0
        [tensor([1]), tensor([5]), tensor([9]), tensor([13])]    # Rank 1
        [tensor([2]), tensor([6]), tensor([10]), tensor([14])]   # Rank 2
        [tensor([3]), tensor([7]), tensor([11]), tensor([15])]   # Rank 3

        >>> # distributed environment with 2 processes
        >>> # Rank 0 has input tensors [tensor([0, 1]), tensor([2, 3])]
        >>> # Rank 1 has input tensors [tensor([4, 5]), tensor([6, 7])]
        >>> if dist.get_rank() == 0:
        ...     input_tensors = [torch.tensor([0, 1]), torch.tensor([2, 3])]
        ... else:
        ...     input_tensors = [torch.tensor([4, 5]), torch.tensor([6, 7])]
        >>> output_tensors = [torch.empty(2, dtype=torch.int64), torch.empty(2, dtype=torch.int64)]
        >>> dist.all_to_all(output_tensors, input_tensors)
        >>> output_tensors
        [tensor([0, 1]), tensor([4, 5])]  # Rank 0 receives from rank 0 and rank 1
        [tensor([2, 3]), tensor([6, 7])]  # Rank 1 receives from rank 0 and rank 1

        >>> # Another example showing data movement:
        >>> # Each rank creates data where values identify the source rank
        >>> rank = dist.get_rank()
        >>> world_size = dist.get_world_size()
        >>> input_tensors = [torch.full((2,), rank*world_size+i) for i in range(world_size)]
        >>> output_tensors = [torch.empty(2, dtype=torch.int64) for _ in range(world_size)]
        >>> input_tensors  # Before all_to_all
        [tensor([0, 0]), tensor([1, 1])]  # Rank 0
        [tensor([2, 2]), tensor([3, 3])]  # Rank 1  (with world_size=2)
        >>> dist.all_to_all(output_tensors, input_tensors)
        >>> output_tensors  # After all_to_all
        [tensor([0, 0]), tensor([2, 2])]  # Rank 0 receives data from rank 0 and rank 1
        [tensor([1, 1]), tensor([3, 3])]  # Rank 1 receives data from rank 0 and rank 1
    """
    if not isinstance(input_tensor_list, list):
        raise TypeError(f'input_tensor_list must be a list, got {type(input_tensor_list)}')
    if not isinstance(output_tensor_list, list):
        raise TypeError(f'output_tensor_list must be a list, got {type(output_tensor_list)}')

    world_size = get_world_size(group)
    if world_size == 1:
        if input_tensor_list and output_tensor_list:
            output_tensor_list[0].copy_(input_tensor_list[0])
        return output_tensor_list

    if group is None:
        group = get_default_group()

    if len(input_tensor_list) != world_size or len(output_tensor_list) != world_size:
        raise ValueError(
            f'Input and output tensor lists must have length equal to world size '
            f'({world_size}). Got input length: {len(input_tensor_list)}, '
            f'output length: {len(output_tensor_list)}')

    for i, tensor in enumerate(input_tensor_list):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f'All items in input_tensor_list must be torch.Tensor, '
                f'got {type(tensor)} at index {i}')

    output_device = get_data_device(output_tensor_list)
    backend_device = get_comm_device(group)
    input_on_device = cast_data_device(input_tensor_list, backend_device)
    tensor_on_device = cast_data_device(output_tensor_list, backend_device)

    torch_dist.all_to_all(tensor_on_device, input_on_device, group=group)
    cast_data_device(tensor_on_device, output_device, out=output_tensor_list)
    return output_tensor_list


def all_reduce(
    data: Tensor, op: str = 'sum', group: ProcessGroup | None = None,
    async_op: bool = False,
) -> Any | None:
    """Reduces the tensor data across all machines in such a way that all get
    the final result.

    After the call ``data`` is going to be bitwise identical in all
    processes.

    Note:
        Calling ``all_reduce`` in non-distributed environment does nothing.

    Args:
        data (Tensor): Input and output of the collective. The function
            operates in-place.
        op (str): Operation to reduce data. Defaults to 'sum'. Optional values
            are 'sum', 'mean' and 'product', 'min', 'max', 'band', 'bor' and
            'bxor'.
        group (ProcessGroup | None): The process group to work on. If None,
            the default process group will be used. Defaults to None.
        async_op (bool): If True, perform the operation asynchronously and
            return a work handle. Caller must ensure tensor is on the correct
            device. Defaults to False.

    Raises:
        TypeError: If data is not a Tensor.
        ValueError: If op is not supported.

    Examples:
        >>> import torch
        >>> import scaletorch.dist as dist

        >>> # non-distributed environment
        >>> data = torch.arange(2, dtype=torch.int64)
        >>> dist.all_reduce(data)
        >>> data
        tensor([0, 1])

        >>> # distributed environment
        >>> # We have 2 process groups, 2 ranks.
        >>> data = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank
        >>> data
        tensor([1, 2]) # Rank 0
        tensor([3, 4]) # Rank 1
        >>> dist.all_reduce(data, op='sum')
        >>> data
        tensor([4, 6]) # Rank 0
        tensor([4, 6]) # Rank 1
    """
    if not isinstance(data, torch.Tensor):
        raise TypeError(f'data must be a torch.Tensor, got {type(data)}')
    if not isinstance(op, str):
        raise TypeError(f'op must be a str, got {type(op)}')

    world_size = get_world_size(group)
    if world_size > 1:
        if group is None:
            group = get_default_group()
        reduce_op = 'sum' if op.lower() == 'mean' else op
        if async_op:
            return torch_dist.all_reduce(data, _get_reduce_op(reduce_op), group, async_op=True)
        input_device = get_data_device(data)
        backend_device = get_comm_device(group)
        data_on_device = cast_data_device(data, backend_device)
        torch_dist.all_reduce(data_on_device, _get_reduce_op(reduce_op), group)
        if op.lower() == 'mean':
            data_on_device.div_(world_size)
        cast_data_device(data_on_device, input_device, out=data)
    return None


def all_gather(data: Tensor, group: ProcessGroup | None = None) -> list[Tensor]:
    """Gather data from the whole group in a list.

    Note:
        Calling ``all_gather`` in non-distributed environment does nothing
        and just returns a list containing :attr:`data` itself.

    Note:
        Unlike PyTorch ``torch.distributed.all_gather``, :meth:`all_gather` in
        scaletorch does not pass in an empty list ``gather_list`` and returns
        the ``gather_list`` directly, which is more convenient. The difference
        between their interfaces is as below:

        - scaletorch: all_gather(data, group) -> gather_list
        - PyTorch: all_gather(gather_list, data, group) -> None

    Args:
        data (Tensor): Tensor to be gathered.
        group (ProcessGroup | None): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Returns:
        list[Tensor]: Return a list containing data from the whole group if
        in distributed environment, otherwise a list only containing
        :attr:`data` itself.

    Raises:
        TypeError: If data is not a Tensor.

    Examples:
        >>> import torch
        >>> import scaletorch.dist as dist

        >>> # non-distributed environment
        >>> data = torch.arange(2, dtype=torch.int64)
        >>> data
        tensor([0, 1])
        >>> output = dist.all_gather(data)
        >>> output
        [tensor([0, 1])]

        >>> # distributed environment
        >>> # We have 2 process groups, 2 ranks.
        >>> data = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank
        >>> data
        tensor([1, 2])  # Rank 0
        tensor([3, 4])  # Rank 1
        >>> output = dist.all_gather(data)
        >>> output
        [tensor([1, 2]), tensor([3, 4])]  # Rank 0
        [tensor([1, 2]), tensor([3, 4])]  # Rank 1
    """
    if not isinstance(data, torch.Tensor):
        raise TypeError(f'data must be a torch.Tensor, got {type(data)}')

    world_size = get_world_size(group)
    if world_size == 1:
        return [data.clone()]

    if group is None:
        group = get_default_group()
    input_device = get_data_device(data)
    backend_device = get_comm_device(group)
    data_on_device = cast_data_device(data, backend_device)
    gather_list = [torch.empty_like(data, device=backend_device) for _ in range(world_size)]
    torch_dist.all_gather(gather_list, data_on_device, group)
    return cast_data_device(gather_list, input_device)  # type: ignore


def gather(
    data: Tensor, dst: int = 0, group: ProcessGroup | None = None,
) -> list[Tensor | None]:
    """Gather data from the whole group to ``dst`` process.

    Note:
        Calling ``gather`` in non-distributed environment dose nothing
        and just returns a list containing :attr:`data` itself.

    Note:
        ``NCCL`` backend does not support ``gather``.

    Note:
        Unlike PyTorch ``torch.distributed.gather``, :meth:`gather` in
        scaletorch does not pass in an empty list ``gather_list`` and returns
        the ``gather_list`` directly, which is more convenient. The difference
        between their interfaces is as below:

        - scaletorch: gather(data, dst, group) -> gather_list
        - PyTorch: gather(data, gather_list, dst, group) -> None

    Args:
        data (Tensor): Tensor to be gathered. CUDA tensor is not supported.
        dst (int): Destination rank. Defaults to 0.
        group (ProcessGroup | None): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Returns:
        list[Tensor]: ``dst`` process will get a list of tensor gathering from
        the whole group. Other process will get a empty list. If in
        non-distributed environment, just return a list containing
        :attr:`data` itself.

    Raises:
        TypeError: If data is not a Tensor or dst is not an int.

    Examples:
        >>> import torch
        >>> import scaletorch.dist as dist

        >>> # non-distributed environment
        >>> data = torch.arange(2, dtype=torch.int64)
        >>> data
        tensor([0, 1])
        >>> output = dist.gather(data)
        >>> output
        [tensor([0, 1])]

        >>> # distributed environment
        >>> # We have 2 process groups, 2 ranks.
        >>> data = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank
        >>> data
        tensor([1, 2]) # Rank 0
        tensor([3, 4]) # Rank 1
        >>> output = dist.gather(data)
        >>> output
        [tensor([1, 2]), tensor([3, 4])]  # Rank 0
        []  # Rank 1
    """
    if not isinstance(data, torch.Tensor):
        raise TypeError(f'data must be a torch.Tensor, got {type(data)}')
    if not isinstance(dst, int):
        raise TypeError(f'dst must be an int, got {type(dst)}')

    world_size = get_world_size(group)
    if world_size == 1:
        return [data]

    if group is None:
        group = get_default_group()
    input_device = get_data_device(data)
    backend_device = get_comm_device(group)
    data_on_device = cast_data_device(data, backend_device)
    if get_rank(group) == dst:
        gather_list = [torch.empty_like(data, device=backend_device) for _ in range(world_size)]
    else:
        gather_list = []
    torch_dist.gather(data_on_device, gather_list, dst, group)
    if get_rank(group) == dst:
        return cast_data_device(gather_list, input_device)  # type: ignore
    return gather_list


def broadcast(data: Tensor, src: int = 0, group: ProcessGroup | None = None) -> None:
    """Broadcast the data from ``src`` process to the whole group.

    ``data`` must have the same number of elements in all processes
    participating in the collective.

    Note:
        Calling ``broadcast`` in non-distributed environment does nothing.

    Args:
        data (Tensor): Data to be sent if ``src`` is the rank of current
            process, and data to be used to save received data otherwise.
        src (int): Source rank. Defaults to 0.
        group (ProcessGroup | None): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Raises:
        TypeError: If data is not a Tensor or src is not an int.

    Examples:
        >>> import torch
        >>> import scaletorch.dist as dist

        >>> # non-distributed environment
        >>> data = torch.arange(2, dtype=torch.int64)
        >>> data
        tensor([0, 1])
        >>> dist.broadcast(data)
        >>> data
        tensor([0, 1])

        >>> # distributed environment
        >>> # We have 2 process groups, 2 ranks.
        >>> data = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank
        >>> data
        tensor([1, 2]) # Rank 0
        tensor([3, 4]) # Rank 1
        >>> dist.broadcast(data)
        >>> data
        tensor([1, 2]) # Rank 0
        tensor([1, 2]) # Rank 1
    """
    if not isinstance(data, torch.Tensor):
        raise TypeError(f'data must be a torch.Tensor, got {type(data)}')
    if not isinstance(src, int):
        raise TypeError(f'src must be an int, got {type(src)}')

    if get_world_size(group) > 1:
        if group is None:
            group = get_default_group()
        input_device = get_data_device(data)
        backend_device = get_comm_device(group)
        data_on_device = cast_data_device(data, backend_device)
        data_on_device = data_on_device.contiguous()  # type: ignore
        torch_dist.broadcast(data_on_device, src, group)
        cast_data_device(data_on_device, input_device, out=data)


def sync_random_seed(group: ProcessGroup | None = None) -> int:
    """Synchronize a random seed to all processes.

    In distributed sampling, different ranks should sample non-overlapped
    data in the dataset. Therefore, this function is used to make sure that
    each rank shuffles the data indices in the same order based
    on the same seed. Then different ranks could use different indices
    to select non-overlapped data from the same data list.

    Args:
        group (ProcessGroup | None): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Returns:
        int: Random seed.

    Examples:
        >>> import torch
        >>> import scaletorch.dist as dist

        >>> # non-distributed environment
        >>> seed = dist.sync_random_seed()
        >>> seed  # which a random number
        587791752

        >>> # distributed environment
        >>> # We have 2 process groups, 2 ranks.
        >>> seed = dist.sync_random_seed()
        >>> seed
        587791752  # Rank 0
        587791752  # Rank 1
    """
    seed = np.random.randint(2**31)
    if get_world_size(group) == 1:
        return seed
    if group is None:
        group = get_default_group()
    backend_device = get_comm_device(group)
    if get_rank(group) == 0:
        random_num = torch.tensor(seed, dtype=torch.int32).to(backend_device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32).to(backend_device)
    torch_dist.broadcast(random_num, src=0, group=group)
    return random_num.item()


# ---------------------------------------------------------------------------
# Dict / parameter-level helpers
# ---------------------------------------------------------------------------


def all_reduce_dict(
    data: dict[str, Tensor], op: str = 'sum', group: ProcessGroup | None = None,
) -> None:
    """Reduces the dict across all machines in such a way that all get the
    final result.

    The code is modified from https://github.com/Megvii-
    BaseDetection/YOLOX/blob/main/yolox/utils/allreduce_norm.py.

    Args:
        data (dict[str, Tensor]): Data to be reduced.
        op (str): Operation to reduce data. Defaults to 'sum'. Optional values
            are 'sum', 'mean' and 'product', 'min', 'max', 'band', 'bor' and
            'bxor'.
        group (ProcessGroup | None): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Raises:
        TypeError: If data is not a dict or op is not a str.
        TypeError: If any value in data is not a Tensor.

    Examples:
        >>> import torch
        >>> import scaletorch.dist as dist

        >>> # non-distributed environment
        >>> data = {
                'key1': torch.arange(2, dtype=torch.int64),
                'key2': torch.arange(3, dtype=torch.int64)
            }
        >>> dist.all_reduce_dict(data)
        >>> data
            {'key1': tensor([0, 1]), 'key2': tensor([0, 1, 2])}

        >>> # distributed environment
        >>> # We have 2 process groups, 2 ranks.
        >>> data = {
                'key1': torch.arange(2, dtype=torch.int64),
                'key2': torch.arange(3, dtype=torch.int64)
            }
        >>> dist.all_reduce_dict(data)
        >>> data
        {'key1': tensor([0, 2]), 'key2': tensor([0, 2, 4])}  # Rank 0
        {'key1': tensor([0, 2]), 'key2': tensor([0, 2, 4])}  # Rank 1
    """
    if not isinstance(data, dict):
        raise TypeError(f'data must be a dict, got {type(data)}')
    if not isinstance(op, str):
        raise TypeError(f'op must be a str, got {type(op)}')
    for key, val in data.items():
        if not isinstance(val, torch.Tensor):
            raise TypeError(f'Values must be Tensor, got {type(val)} for key "{key}"')

    if get_world_size(group) > 1:
        if group is None:
            group = get_default_group()
        keys = sorted(data.keys())
        tensor_shapes = [data[k].shape for k in keys]
        tensor_sizes = [data[k].numel() for k in keys]
        flatten_tensor = torch.cat([data[k].flatten() for k in keys])
        all_reduce(flatten_tensor, op=op, group=group)
        split_tensors = [
            x.reshape(shape) for x, shape in zip(
                torch.split(flatten_tensor, tensor_sizes), tensor_shapes)]
        for key, val in zip(keys, split_tensors):
            data[key] = val


def _all_reduce_coalesced(
    tensors: list[torch.Tensor], bucket_size_mb: int = -1,
    op: str = 'sum', group: ProcessGroup | None = None,
) -> None:
    """All-reduce a sequence of tensors as a whole.

    Args:
        tensors (list[torch.Tensor]): A sequence of tensors to be
            all-reduced.
        bucket_size_mb (int): The limit of each chunk in megabytes
            for grouping tensors into chunks. Defaults to -1.
        op (str): Operation to reduce data. Defaults to 'sum'. Optional values
            are 'sum', 'mean' and 'product', 'min', 'max', 'band', 'bor' and
            'bxor'.
        group (ProcessGroup | None): The process group to work on. If None,
            the default process group will be used. Defaults to None.
    """
    if not isinstance(tensors, list):
        raise TypeError(f'tensors must be a list, got {type(tensors)}')
    for i, tensor in enumerate(tensors):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f'Item at index {i} is not a Tensor: {type(tensor)}')

    if bucket_size_mb > 0:
        bucket_size_bytes = bucket_size_mb * 1024 * 1024
        buckets = _take_tensors(tensors, bucket_size_bytes)
    else:
        buckets_dict: OrderedDict[str, list[torch.Tensor]] = OrderedDict()
        for tensor in tensors:
            tp = tensor.type()
            if tp not in buckets_dict:
                buckets_dict[tp] = []
            buckets_dict[tp].append(tensor)
        buckets = buckets_dict.values()

    for bucket in buckets:
        flat_tensors = _flatten_dense_tensors(bucket)
        all_reduce(flat_tensors, op=op, group=group)
        for tensor, synced in zip(bucket, _unflatten_dense_tensors(flat_tensors, bucket)):
            tensor.copy_(synced)


def all_reduce_params(
    params: list | GeneratorType, coalesce: bool = True,
    bucket_size_mb: int = -1, op: str = 'sum',
    group: ProcessGroup | None = None,
) -> None:
    """All-reduce parameters.

    Args:
        params (list | GeneratorType): List of
            parameters or buffers of a model.
        coalesce (bool, optional): Whether to reduce parameters as a whole.
            Defaults to True.
        bucket_size_mb (int, optional): Size of bucket, the unit is MB.
            Defaults to -1.
        op (str): Operation to reduce data. Defaults to 'sum'. Optional values
            are 'sum', 'mean' and 'product', 'min', 'max', 'band', 'bor' and
            'bxor'.
        group (ProcessGroup | None): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Examples:
        >>> import torch
        >>> import scaletorch.dist as dist

        >>> # non-distributed environment
        >>> data = [torch.arange(2), torch.arange(3)]
        >>> dist.all_reduce_params(data)
        >>> data
            [tensor([0, 1]), tensor([0, 1, 2])]

        >>> # distributed environment
        >>> # We have 2 process groups, 2 ranks.
        >>> if dist.get_rank() == 0:
        ...     data = [torch.tensor([1, 2]), torch.tensor([3, 4])]
        ... else:
        ...     data = [torch.tensor([2, 3]), torch.tensor([4, 5])]

        >>> dist.all_reduce_params(data)
        >>> data
            [torch.tensor([3, 5]), torch.tensor([7, 9])]
    """
    if not isinstance(params, (list, GeneratorType)):
        raise TypeError(f'params must be a list or generator, got {type(params)}')
    if get_world_size(group) == 1:
        return
    params_data = [param.data if hasattr(param, 'data') else param for param in params]
    if coalesce:
        _all_reduce_coalesced(params_data, bucket_size_mb, op=op, group=group)
    else:
        for tensor in params_data:
            all_reduce(tensor, op=op, group=group)
