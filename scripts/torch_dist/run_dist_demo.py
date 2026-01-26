import datetime
import os
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
from transformers.utils import is_torch_cuda_available, is_torch_npu_available


def rank_print(rank: int, msg: str) -> None:
    """Print message with rank prefix for distributed debugging.

    Args:
        rank: Process rank
        msg: Message to print
    """
    print(f'[Rank {rank}] {msg}', flush=True)


def get_dist_info() -> Tuple[int, int, int]:
    """Get distributed training information.

    Returns:
        Tuple of (rank, world_size, local_rank)
    """
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    return rank, world_size, local_rank


def get_current_device(use_cpu: bool = False) -> torch.device:
    """Get current device based on available backends.

    Args:
        use_cpu: Whether to force CPU usage

    Returns:
        Current device
    """
    _, _, local_rank = get_dist_info()

    if use_cpu:
        return torch.device('cpu')
    if is_torch_cuda_available():
        return torch.device('cuda', local_rank)
    if is_torch_npu_available():
        return torch.device('npu', local_rank)
    return torch.device('cpu')


def init_dist_process(use_cpu: bool = False, timeout: int = 120) -> None:
    """Initialize distributed process group.

    Args:
        use_cpu: Whether to use CPU for distributed training
        timeout: Timeout in seconds for distributed operations
    """
    if dist.is_initialized():
        return

    rank, world_size, local_rank = get_dist_info()

    if world_size == 1:
        return  # Single process, no need for distributed training

    if use_cpu:
        backend = 'gloo'
    elif is_torch_cuda_available():
        backend = 'nccl'
        torch.cuda.set_device(local_rank)
    elif is_torch_npu_available():
        backend = 'hccl'
        torch.npu.set_device(local_rank)
    else:
        backend = 'gloo'

    try:
        dist.init_process_group(backend=backend,
                                rank=rank,
                                world_size=world_size,
                                timeout=datetime.timedelta(seconds=timeout))
        rank_print(
            rank,
            f'Initialized distributed process group with backend: {backend}')
    except Exception as e:
        rank_print(rank,
                   f'Failed to initialize distributed process group: {e}')
        raise


def test_broadcast(rank: int, world_size: int, device: torch.device) -> None:
    """Test broadcast communication.

    Args:
        rank: Process rank
        world_size: Total number of processes
        device: Device to use for tensors
    """
    rank_print(rank, 'Testing broadcast')
    try:
        # Create different data for each rank
        if rank == 0:
            tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        else:
            tensor = torch.zeros(4, device=device)

        rank_print(rank, f'Before broadcast - Rank {rank} has data: {tensor}')
        dist.broadcast(tensor, src=0)
        rank_print(rank, f'After broadcast - Rank {rank} has data: {tensor}')
    except Exception as e:
        rank_print(rank, f'Broadcast test failed: {e}')
        raise


def test_gather(rank: int, world_size: int, device: torch.device) -> None:
    """Test gather communication.

    Args:
        rank: Process rank
        world_size: Total number of processes
        device: Device to use for tensors
    """
    rank_print(rank, 'Testing gather')
    try:
        # Create local data
        local_data = torch.tensor([rank, rank * 2],
                                  dtype=torch.float32,
                                  device=device)
        rank_print(rank, f'Local data - Rank {rank} has data: {local_data}')

        # Prepare receive buffer only on rank 0
        gather_list: Optional[List[torch.Tensor]] = None
        if rank == 0:
            gather_list = [
                torch.zeros_like(local_data) for _ in range(world_size)
            ]

        # Perform gather operation
        dist.gather(local_data, gather_list, dst=0)
        if rank == 0 and gather_list is not None:
            rank_print(
                rank,
                f'Gathered data - Rank {rank} received: {[t.tolist() for t in gather_list]}'
            )
    except Exception as e:
        rank_print(rank, f'Gather test failed: {e}')
        raise


def test_scatter(rank: int, world_size: int, device: torch.device) -> None:
    """Test scatter communication.

    Args:
        rank: Process rank
        world_size: Total number of processes
        device: Device to use for tensors
    """
    rank_print(rank, 'Testing scatter')
    try:
        # Receive tensor
        recv_tensor = torch.zeros(4, dtype=torch.int64, device=device)

        # Prepare scatter data only on rank 0
        scatter_list = None
        if rank == 0:
            # Create full data and split it
            full_data = torch.arange(world_size * 4,
                                     dtype=torch.int64,
                                     device=device)
            # Use split instead of chunk for more predictable behavior
            scatter_list = list(torch.split(full_data, 4))
            rank_print(
                rank,
                f'Rank 0 scattering data: {[t.tolist() for t in scatter_list]}'
            )

        rank_print(rank,
                   f'Before scatter - Rank {rank} has data: {recv_tensor}')
        # Perform scatter operation
        dist.scatter(recv_tensor, scatter_list, src=0)
        rank_print(
            rank,
            f'After scatter - Rank {rank} received: {recv_tensor.tolist()}')
    except Exception as e:
        rank_print(rank, f'Scatter test failed: {e}')
        raise


def test_reduce(rank: int, world_size: int, device: torch.device) -> None:
    """Test reduce communication.

    Args:
        rank: Process rank
        world_size: Total number of processes
        device: Device to use for tensors
    """
    rank_print(rank, 'Testing reduce')
    try:
        # Create local data
        tensor = torch.tensor([rank, rank + 1, rank + 2],
                              dtype=torch.float32,
                              device=device)
        rank_print(rank, f'Before reduce - Rank {rank} has data: {tensor}')

        # Perform reduce operation, only rank 0 receives result
        dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)
        if rank == 0:
            rank_print(
                rank,
                f'After reduce - Rank {rank} (destination) has data: {tensor}')
    except Exception as e:
        rank_print(rank, f'Reduce test failed: {e}')
        raise


def test_all_gather(rank: int, world_size: int, device: torch.device) -> None:
    """Test all_gather communication.

    Args:
        rank: Process rank
        world_size: Total number of processes
        device: Device to use for tensors
    """
    rank_print(rank, 'Testing all_gather')
    try:
        # Create local data
        local_data = torch.tensor([rank, rank * 2, rank * 3],
                                  dtype=torch.float32,
                                  device=device)
        rank_print(rank, f'Local data - Rank {rank} has data: {local_data}')

        # Gather data from all processes
        gathered_list = [
            torch.zeros_like(local_data) for _ in range(world_size)
        ]
        dist.all_gather(gathered_list, local_data)
        rank_print(
            rank,
            f'Gathered data - Rank {rank} received: {[t.tolist() for t in gathered_list]}'
        )
    except Exception as e:
        rank_print(rank, f'All-gather test failed: {e}')
        raise


def test_all_reduce(rank: int, world_size: int, device: torch.device) -> None:
    """Test all_reduce communication.

    Args:
        rank: Process rank
        world_size: Total number of processes
        device: Device to use for tensors
    """
    rank_print(rank, 'Testing all_reduce')
    try:
        # Create local data
        tensor = torch.tensor([1.0, 2.0, 3.0], device=device) * (rank + 1)
        rank_print(rank, f'Before all_reduce - Rank {rank} has data: {tensor}')

        # Perform global sum reduction
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        rank_print(rank,
                   f'After all_reduce sum - Rank {rank} has data: {tensor}')
    except Exception as e:
        rank_print(rank, f'All-reduce test failed: {e}')
        raise


def test_reduce_scatter(rank: int, world_size: int,
                        device: torch.device) -> None:
    """Test reduce_scatter communication.

    Args:
        rank: Process rank
        world_size: Total number of processes
        device: Device to use for tensors
    """
    rank_print(rank, 'Testing reduce_scatter')
    try:
        # Each rank creates local data
        tensor_dim = 4
        input_list = []
        for i in range(world_size):
            tensor = torch.ones(tensor_dim,
                                device=device) * (rank + 1) * (i + 1)
            input_list.append(tensor)

        output_tensor = torch.zeros(tensor_dim, device=device)
        print(f'Before reduce_scatter - Rank {rank} has data: {input_list}')
        # Perform reduce_scatter operation
        dist.reduce_scatter(output_tensor, input_list, op=dist.ReduceOp.SUM)
        rank_print(
            rank,
            f'After reduce_scatter - Rank {rank} has data: {output_tensor}')
    except Exception as e:
        rank_print(rank, f'Reduce-scatter test failed: {e}')
        raise


def test_all_to_all(rank: int, world_size: int, device: torch.device) -> None:
    """Test all_to_all communication.

    Args:
        rank: Process rank
        world_size: Total number of processes
        device: Device to use for tensors
    """
    rank_print(rank, 'Testing all_to_all')
    try:
        # Each rank creates local data
        input_list = []
        for i in range(world_size):
            tensor = torch.tensor([rank * world_size + i],
                                  dtype=torch.float32,
                                  device=device)
            input_list.append(tensor)

        rank_print(
            rank,
            f'Input data - Rank {rank} has data: {[t.tolist() for t in input_list]}'
        )

        # Prepare output list
        output_list = [
            torch.zeros(1, dtype=torch.float32, device=device)
            for _ in range(world_size)
        ]

        # Perform all_to_all operation
        dist.all_to_all(output_list, input_list)
        rank_print(
            rank,
            f'Output data - Rank {rank} received: {[t.tolist() for t in output_list]}'
        )
    except Exception as e:
        rank_print(rank, f'All-to-all test failed: {e}')
        raise


def test_object_broadcast(rank: int, world_size: int,
                          device: torch.device) -> None:
    """Test broadcasting Python objects.

    Args:
        rank: Process rank
        world_size: Total number of processes
        device: Device to use for tensors
    """
    rank_print(rank, 'Testing object broadcast')
    try:
        # Prepare objects to broadcast
        if rank == 0:
            obj_list = [{'loss': 0.5, 'rank': rank}, ['accuracy', 0.95], 100]
        else:
            obj_list = [None, None, None]

        rank_print(
            rank,
            f'Before object broadcast - Rank {rank} has data: {obj_list}')
        dist.broadcast_object_list(obj_list, src=0)
        rank_print(
            rank, f'After object broadcast - Rank {rank} has data: {obj_list}')
    except Exception as e:
        rank_print(rank, f'Object broadcast test failed: {e}')
        raise


def run_all_tests(use_cpu: bool = False) -> None:
    """Run all distributed communication tests.

    Args:
        use_cpu: Whether to use CPU for testing
    """
    rank, world_size, _ = get_dist_info()
    device = get_current_device(use_cpu)

    rank_print(rank,
               f'Starting distributed tests on Rank {rank}/{world_size-1}')

    try:
        # Run various tests
        test_broadcast(rank, world_size, device)

        dist.barrier()  # Synchronize between tests

        test_gather(rank, world_size, device)

        dist.barrier()

        test_scatter(rank, world_size, device)

        dist.barrier()

        test_reduce(rank, world_size, device)

        dist.barrier()

        test_all_gather(rank, world_size, device)

        dist.barrier()

        test_all_reduce(rank, world_size, device)

        dist.barrier()

        test_reduce_scatter(rank, world_size, device)

        dist.barrier()

        test_all_to_all(rank, world_size, device)

        dist.barrier()

        if world_size > 1:
            test_object_broadcast(rank, world_size, device)
            dist.barrier()

        rank_print(rank, 'All tests finished successfully')
    except Exception as e:
        rank_print(rank, f'Tests failed with error: {e}')
        raise


if __name__ == '__main__':
    # Initialize distributed environment (if in distributed environment)
    # Can control CPU usage via environment variable
    use_cpu = os.environ.get('USE_CPU', 'false').lower() == 'true'
    try:
        init_dist_process(use_cpu)
        run_all_tests(use_cpu)
    except Exception as e:
        print(f'Distributed test failed: {e}')
        raise
