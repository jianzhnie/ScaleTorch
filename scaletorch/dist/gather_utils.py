"""Distributed result collection utilities (CPU tmpdir and GPU/NPU paths)."""

from __future__ import annotations

import os.path as osp
import pickle
import shutil
import tempfile
from itertools import chain, zip_longest

import torch

from scaletorch.utils import mkdir_or_exist

from .collective_ops import broadcast
from .object_ops import all_gather_object
from .utils import barrier, get_dist_info

# Maximum length of a temporary directory path in bytes when broadcast
# across ranks. 512 is more than sufficient for typical POSIX paths.
_TMPDIR_PATH_MAX_LEN = 512


def collect_results(
    results: list,
    size: int,
    device: str = 'cpu',
    tmpdir: str | None = None,
) -> list | None:
    """Collected results in distributed environments.

    Args:
        results (list[object]): Result list containing result parts to be
            collected. Each item of ``result_part`` should be a picklable
            object.
        size (int): Size of the results, commonly equal to length of
            the results.
        device (str): Device name. Optional values are 'cpu', 'gpu' or 'npu'.
        tmpdir (str | None): Temporal directory for collected results to
            store. If set to None, it will create a temporal directory for it.
            ``tmpdir`` should be None when device is 'gpu' or 'npu'.
            Defaults to None.

    Returns:
        list or None: The collected results.

    Examples:
        >>> # distributed environment
        >>> # We have 2 process groups, 2 ranks.
        >>> import scaletorch.dist as dist
        >>> if dist.get_rank() == 0:
                data = ['foo', {1: 2}]
            else:
                data = [24, {'a': 'b'}]
        >>> size = 4
        >>> output = dist.collect_results(data, size, device='cpu')
        >>> output
        ['foo', 24, {1: 2}, {'a': 'b'}]  # rank 0
        None  # rank 1
    """
    if device not in {'cpu', 'gpu', 'npu'}:
        raise NotImplementedError(
            f"device must be 'cpu', 'gpu' or 'npu', but got {device}")

    if device in ('gpu', 'npu'):
        if tmpdir is not None:
            raise ValueError(
                f'tmpdir should be None when device is {device}, got {tmpdir}')
        return _collect_results_device(results, size)

    return collect_results_cpu(results, size, tmpdir)


def collect_results_cpu(
    result_part: list,
    size: int,
    tmpdir: str | None = None,
) -> list | None:
    """Collect results under cpu mode.

    On cpu mode, this function will save the results on different gpus to
    ``tmpdir`` and collect them by the rank 0 worker.

    Args:
        result_part (list): Result list containing result parts
            to be collected. Each item of ``result_part`` should be a picklable
            object.
        size (int): Size of the results, commonly equal to length of
            the results.
        tmpdir (str | None): Temporal directory for collected results to
            store. If set to None, it will create a random temporal directory
            for it. Defaults to None.

    Returns:
        list or None: The collected results.

    Examples:
        >>> # distributed environment
        >>> # We have 2 process groups, 2 ranks.
        >>> import scaletorch.dist as dist
        >>> if dist.get_rank() == 0:
                data = ['foo', {1: 2}]
            else:
                data = [24, {'a': 'b'}]
        >>> size = 4
        >>> output = dist.collect_results_cpu(data, size)
        >>> output
        ['foo', 24, {1: 2}, {'a': 'b'}]  # rank 0
        None  # rank 1
    """
    rank, world_size = get_dist_info()
    if world_size == 1:
        return result_part[:size]

    # Create / broadcast a shared tmpdir
    if tmpdir is None:
        dir_tensor = torch.full((_TMPDIR_PATH_MAX_LEN,), 32, dtype=torch.uint8)
        if rank == 0:
            mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir_tensor = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8)
            dir_tensor[:len(tmpdir_tensor)] = tmpdir_tensor
        broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.numpy().tobytes().decode().rstrip()
    else:
        mkdir_or_exist(tmpdir)

    # Dump partial results
    with open(osp.join(tmpdir, f'part_{rank}.pkl'), 'wb') as fh:
        pickle.dump(result_part, fh, protocol=2)

    barrier()

    if rank != 0:
        return None

    # Rank 0: load, interleave, and trim
    part_list = []
    for i in range(world_size):
        path = osp.join(tmpdir, f'part_{i}.pkl')
        if not osp.exists(path):
            raise FileNotFoundError(
                f'{tmpdir} is not a shared directory for rank {i}, '
                f'please make sure {tmpdir} is shared across all ranks!')
        with open(path, 'rb') as fh:
            part_list.append(pickle.load(fh))

    ordered_results = [
        item
        for item in chain.from_iterable(zip_longest(*part_list))
        if item is not None
    ]
    ordered_results = ordered_results[:size]
    shutil.rmtree(tmpdir)
    return ordered_results


def _collect_results_device(
    result_part: list,
    size: int,
) -> list | None:
    """Collect results via ``all_gather_object`` (GPU / NPU path)."""
    rank, world_size = get_dist_info()
    if world_size == 1:
        return result_part[:size]

    part_list = all_gather_object(result_part)

    if rank == 0:
        ordered_results = [
            item
            for item in chain.from_iterable(zip_longest(*part_list))
            if item is not None
        ]
        return ordered_results[:size]

    return None


def collect_results_gpu(result_part: list, size: int) -> list | None:
    """Collect results under GPU mode (delegates to device path).

    On gpu mode, this function will encode results to gpu tensors and use gpu
    communication for results collection.

    Args:
        result_part (list[object]): Result list containing result parts
            to be collected. Each item of ``result_part`` should be a picklable
            object.
        size (int): Size of the results, commonly equal to length of
            the results.

    Returns:
        list or None: The collected results.

    Examples:
        >>> # distributed environment
        >>> # We have 2 process groups, 2 ranks.
        >>> import scaletorch.dist as dist
        >>> if dist.get_rank() == 0:
                data = ['foo', {1: 2}]
            else:
                data = [24, {'a': 'b'}]
        >>> size = 4
        >>> output = dist.collect_results_gpu(data, size)
        >>> output
        ['foo', 24, {1: 2}, {'a': 'b'}]  # rank 0
        None  # rank 1
    """
    return _collect_results_device(result_part, size)
