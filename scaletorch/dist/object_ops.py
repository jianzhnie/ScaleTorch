"""Object-level communication: broadcast/gather picklable Python objects."""

from __future__ import annotations

import pickle
from typing import Any

import torch
from torch import Tensor
from torch import distributed as torch_dist
from torch.distributed import ProcessGroup
from transformers.utils import is_torch_npu_available

from .utils import (
    get_backend,
    get_default_group,
    get_rank,
    get_world_size,
)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _object_to_tensor(obj: Any) -> tuple[Tensor, Tensor]:
    """Serialize a picklable Python object to a pair of tensors."""
    data = pickle.dumps(obj)
    # Use torch.frombuffer (non-deprecated) rather than ByteStorage.from_buffer.
    # ByteTensor from a bytearray is fast; torch.tensor(…, dtype=…) is 100× slower.
    # See: https://github.com/pytorch/pytorch/issues/65696
    byte_tensor = torch.ByteTensor(bytearray(data))
    local_size = torch.LongTensor([byte_tensor.numel()])
    return byte_tensor, local_size


def _tensor_to_object(tensor: Tensor, tensor_size: int) -> Any:
    """Deserialize a tensor back into a Python object."""
    buf = tensor.cpu().numpy().tobytes()[:tensor_size]
    return pickle.loads(buf)


# ---------------------------------------------------------------------------
# NPU-compatible broadcast (manual serialisation path)
# ---------------------------------------------------------------------------

def _broadcast_object_list(
    object_list: list[Any],
    src: int = 0,
    group: ProcessGroup | None = None,
) -> None:
    """Broadcast picklable objects — manual implementation for NPU/HCCL."""
    if group is not None and get_rank(group) == -1:
        return

    my_rank = get_rank()

    # Serialize on src rank
    if my_rank == src:
        tensor_list, size_list = zip(
            *[_object_to_tensor(obj) for obj in object_list])
        object_sizes_tensor = torch.cat(size_list)
    else:
        object_sizes_tensor = torch.empty(len(object_list), dtype=torch.long)

    # Determine communication device from backend
    group_backend = get_backend(group)
    is_nccl_backend = group_backend == torch_dist.Backend.NCCL
    current_device = torch.device('cpu')
    is_hccl_backend = group_backend == 'hccl'
    is_cncl_backend = group_backend == 'cncl'
    is_mccl_backend = group_backend == 'mccl'

    if is_hccl_backend:
        current_device = torch.device('npu', torch.npu.current_device())
        object_sizes_tensor = object_sizes_tensor.to(current_device)
    elif is_cncl_backend:
        current_device = torch.device('mlu', torch.mlu.current_device())
        object_sizes_tensor = object_sizes_tensor.to(current_device)
    elif is_mccl_backend:
        current_device = torch.device('musa', torch.musa.current_device())
        object_sizes_tensor = object_sizes_tensor.to(current_device)
    elif is_nccl_backend:
        current_device = torch.device('cuda', torch.cuda.current_device())
        object_sizes_tensor = object_sizes_tensor.to(current_device)

    # Broadcast sizes
    torch_dist.broadcast(object_sizes_tensor, src=src, group=group)

    # Broadcast serialised bytes
    if my_rank == src:
        object_tensor = torch.cat(tensor_list)
    else:
        object_tensor = torch.empty(
            torch.sum(object_sizes_tensor).int().item(), dtype=torch.uint8)

    if is_nccl_backend or is_hccl_backend or is_cncl_backend or is_mccl_backend:
        object_tensor = object_tensor.to(current_device)
    torch_dist.broadcast(object_tensor, src=src, group=group)

    # Deserialize on non-src ranks
    offset = 0
    if my_rank != src:
        for i, obj_size in enumerate(object_sizes_tensor):
            obj_view = object_tensor[offset:offset + obj_size]
            obj_view = obj_view.type(torch.uint8)
            if obj_view.device != torch.device('cpu'):
                obj_view = obj_view.cpu()
            offset += obj_size
            object_list[i] = _tensor_to_object(obj_view, obj_size)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def broadcast_object_list(
    data: list[Any],
    src: int = 0,
    group: ProcessGroup | None = None,
    device: torch.device | None = None,
) -> None:
    """Broadcasts picklable objects in ``object_list`` to the whole group.
    Similar to :func:`broadcast`, but Python objects can be passed in. Note
    that all objects in ``object_list`` must be picklable in order to be
    broadcasted.

    Note:
        Calling ``broadcast_object_list`` in non-distributed environment does
        nothing.

    Args:
        data (list[Any]): List of input objects to broadcast.
            Each object must be picklable. Only objects on the ``src`` rank
            will be broadcast, but each rank must provide lists of equal sizes.
        src (int): Source rank from which to broadcast ``object_list``.
        group: (ProcessGroup | None): The process group to work on. If None,
            the default process group will be used. Default is ``None``.
        device (torch.device | None): If not None, the objects are
            serialized and converted to tensors which are moved to the
            ``device`` before broadcasting. Default is ``None``.

    Note:
        For NCCL-based process groups, internal tensor representations of
        objects must be moved to the GPU device before communication starts.
        In this case, the used device is given by
        ``torch.cuda.current_device()`` and it is the user's responsibility to
        ensure that this is correctly set so that each rank has an individual
        GPU, via ``torch.cuda.set_device()``.

    Examples:
        >>> import torch
        >>> import scaletorch.dist as dist

        >>> # non-distributed environment
        >>> data = ['foo', 12, {1: 2}]
        >>> dist.broadcast_object_list(data)
        >>> data
        ['foo', 12, {1: 2}]

        >>> # distributed environment
        >>> # We have 2 process groups, 2 ranks.
        >>> if dist.get_rank() == 0:
        >>>     # Assumes world_size of 3.
        >>>     data = ["foo", 12, {1: 2}]  # any picklable object
        >>> else:
        >>>     data = [None, None, None]
        >>> dist.broadcast_object_list(data)
        >>> data
        ["foo", 12, {1: 2}]  # Rank 0
        ["foo", 12, {1: 2}]  # Rank 1
    """
    if not isinstance(data, list):
        raise TypeError(f'data must be a list, got {type(data)}')

    if get_world_size(group) > 1:
        if group is None:
            group = get_default_group()

        if not is_torch_npu_available():
            torch_dist.broadcast_object_list(data, src, group, device=device)
        else:
            _broadcast_object_list(data, src, group)


def all_gather_object(
    data: Any,
    group: ProcessGroup | None = None,
) -> list[Any]:
    """Gather picklable objects from the whole group into a list. Similar to
    :func:`all_gather`, but Python objects can be passed in. Note that the
    object must be picklable in order to be gathered.

    Note:
        Calling ``all_gather_object`` in non-distributed environment does
        nothing and just returns a list containing :attr:`data` itself.

    Note:
        Unlike PyTorch ``torch.distributed.all_gather_object``,
        :meth:`all_gather_object` in scaletorch does not pass in an empty list
        ``gather_list`` and returns the ``gather_list`` directly, which is
        more convenient. The difference between their interfaces is as below:

        - scaletorch: all_gather_object(data, group) -> gather_list
        - PyTorch: all_gather_object(gather_list, data, group) -> None

    Args:
        data (Any): Pickable Python object to be broadcast from current
            process.
        group (ProcessGroup | None): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Returns:
        list[Tensor]: Return a list containing data from the whole group if
        in distributed environment, otherwise a list only containing
        :attr:`data` itself.

    Note:
        For NCCL-based process groups, internal tensor representations
        of objects must be moved to the GPU device before communication starts.
        In this case, the used device is given by
        ``torch.cuda.current_device()`` and it is the user's responsibility to
        ensure that this is correctly set so that each rank has an individual
        GPU, via ``torch.cuda.set_device()``.

    Examples:
        >>> import torch
        >>> import scaletorch.dist as dist

        >>> # non-distributed environment
        >>> data = ['foo', 12, {1: 2}]  # any picklable object
        >>> gather_objects = dist.all_gather_object(data[dist.get_rank()])
        >>> output
        ['foo']

        >>> # distributed environment
        >>> # We have 3 process groups, 3 ranks.
        >>> output = dist.all_gather_object(data[dist.get_rank()])
        >>> output
        ['foo', 12, {1: 2}]  # Rank 0
        ['foo', 12, {1: 2}]  # Rank 1
        ['foo', 12, {1: 2}]  # Rank 2
    """
    world_size = get_world_size(group)
    if world_size == 1:
        return [data]

    if group is None:
        group = get_default_group()

    gather_list = [None] * world_size
    torch_dist.all_gather_object(gather_list, data, group)
    return gather_list


def gather_object(
    data: Any,
    dst: int = 0,
    group: ProcessGroup | None = None,
) -> list[Any] | None:
    """Gathers picklable objects from the whole group in a single process.
    Similar to :func:`gather`, but Python objects can be passed in. Note that
    the object must be picklable in order to be gathered.

    Note:
        ``NCCL backend`` does not support ``gather_object``.

    Note:
        Unlike PyTorch ``torch.distributed.gather_object``,
        :meth:`gather_object` in scaletorch does not pass in an empty list
        ``gather_list`` and returns the ``gather_list`` directly, which is
        more convenient. The difference between their interfaces is as below:

        - scaletorch: gather_object(data, dst, group) -> gather_list
        - PyTorch: gather_object(data, gather_list, data, group) -> None

    Args:
        data (Any): Input object. Must be picklable.
        dst (int): Destination rank. Defaults to 0.
        group: (ProcessGroup | None): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Returns:
        list[Any]. On the ``dst`` rank, return ``gather_list`` which contains
        the output of the collective.

    Examples:
        >>> import torch
        >>> import scaletorch.dist as dist

        >>> # non-distributed environment
        >>> data = ['foo', 12, {1: 2}]  # any picklable object
        >>> gather_objects = dist.gather_object(data[dist.get_rank()])
        >>> output
        ['foo']

        >>> # distributed environment
        >>> # We have 3 process groups, 3 ranks.
        >>> dist.gather_object(gather_objects[dist.get_rank()], dst=0)
        >>> output
        ['foo', 12, {1: 2}]  # Rank 0
        None  # Rank 1
        None  # Rank 2
    """
    world_size = get_world_size(group)
    if world_size == 1:
        return [data]

    if group is None:
        group = get_default_group()

    gather_list = [None] * world_size if get_rank(group) == dst else None
    torch_dist.gather_object(data, gather_list, dst, group)
    return gather_list
