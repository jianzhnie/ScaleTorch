import os

import torch
import torch.distributed as dist
from transformers.utils import is_torch_cuda_available, is_torch_npu_available


def get_current_device(use_cpu=False):
    """Get current device based on available backends."""
    if not use_cpu and is_torch_cuda_available():
        return torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0)))
    elif not use_cpu and is_torch_npu_available():
        return torch.device('npu', int(os.environ.get('LOCAL_RANK', 0)))
    else:
        return torch.device('cpu')


def init_dist_process(use_cpu=False) -> None:
    """Initialize distributed process group."""

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    # LOCAL_RANK is set by `torch.distributed.launch` since PyTorch 1.1
    local_rank = int(os.environ['LOCAL_RANK'])

    if use_cpu:
        dist.init_process_group('gloo', rank=rank, world_size=world_size)
    elif is_torch_cuda_available():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl',
                                rank=rank,
                                world_size=world_size)

    elif is_torch_npu_available():
        import torch_npu  # noqa: F401
        torch.npu.set_device(local_rank)
        dist.init_process_group(backend='hccl',
                                rank=rank,
                                world_size=world_size)
    else:
        dist.init_process_group('gloo', rank=rank, world_size=world_size)


def test_broadcast(rank_id, world_size, use_cpu=False):
    """Test broadcast communication."""
    print(f'\n=== Testing Broadcast (Rank {rank_id}) ===')
    device = get_current_device(use_cpu)

    # 创建不同rank的数据
    if rank_id == 0:
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
    else:
        tensor = torch.zeros(4, device=device)

    print(f'Before broadcast - Rank {rank_id} has data: {tensor}')
    dist.broadcast(tensor, src=0)
    print(f'After broadcast - Rank {rank_id} has data: {tensor}')


def test_all_reduce(rank_id, world_size, use_cpu=False):
    """Test all_reduce communication."""
    print(f'\n=== Testing AllReduce (Rank {rank_id}) ===')
    device = get_current_device(use_cpu)

    # 创建本地数据
    tensor = torch.tensor([1.0, 2.0, 3.0], device=device) * (rank_id + 1)
    print(f'Before all_reduce - Rank {rank_id} has data: {tensor}')

    # 执行全局求和归约
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f'After all_reduce sum - Rank {rank_id} has data: {tensor}')


def test_all_gather(rank_id, world_size, use_cpu=False):
    """Test all_gather communication."""
    print(f'\n=== Testing AllGather (Rank {rank_id}) ===')
    device = get_current_device(use_cpu)

    # 创建本地数据
    local_data = torch.tensor([rank_id, rank_id * 2, rank_id * 3],
                              dtype=torch.float32,
                              device=device)
    print(f'Local data - Rank {rank_id} has data: {local_data}')

    # 收集所有进程的数据
    gathered_list = [torch.zeros_like(local_data) for _ in range(world_size)]
    dist.all_gather(gathered_list, local_data)
    print(
        f'Gathered data - Rank {rank_id} received: {[t.tolist() for t in gathered_list]}'
    )


def test_scatter(rank_id, world_size, use_cpu=False):
    """Test scatter communication."""
    print(f'\n=== Testing Scatter (Rank {rank_id}) ===')
    device = get_current_device(use_cpu)

    # 接收张量
    recv_tensor = torch.zeros(4, dtype=torch.int64, device=device)

    # 只在rank 0准备scatter数据
    scatter_list = None
    if rank_id == 0:
        # 创建总数据并拆分
        full_data = torch.arange(world_size * 4,
                                 dtype=torch.int64,
                                 device=device)
        scatter_list = list(torch.chunk(full_data, world_size))
        print(f'Rank 0 scattering data: {[t.tolist() for t in scatter_list]}')

    # 执行scatter操作
    dist.scatter(recv_tensor, scatter_list, src=0)
    print(f'After scatter - Rank {rank_id} received: {recv_tensor.tolist()}')


def test_reduce(rank_id, world_size, use_cpu=False):
    """Test reduce communication."""
    print(f'\n=== Testing Reduce (Rank {rank_id}) ===')
    device = get_current_device(use_cpu)

    # 创建本地数据
    tensor = torch.tensor([rank_id, rank_id + 1, rank_id + 2],
                          dtype=torch.float32,
                          device=device)
    print(f'Before reduce - Rank {rank_id} has data: {tensor}')

    # 执行reduce操作，只在rank 0接收结果
    dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)
    if rank_id == 0:
        print(
            f'After reduce - Rank {rank_id} (destination) has data: {tensor}')


def test_reduce_scatter(rank_id, world_size, use_cpu=False):
    """Test reduce_scatter communication."""
    print(f'\n=== Testing ReduceScatter (Rank {rank_id}) ===')
    device = get_current_device(use_cpu)

    # 每个rank创建本地数据
    input_list = []
    for i in range(world_size):
        tensor = torch.tensor(
            [i + 1,
             (i + 1) * 2], dtype=torch.float32, device=device) * (rank_id + 1)
        input_list.append(tensor)

    input_tensor = torch.cat(input_list)
    output_tensor = torch.zeros_like(input_list[0])

    print(f'Before reduce_scatter - Rank {rank_id} has data: {input_tensor}')

    # 执行reduce_scatter操作
    dist.reduce_scatter(output_tensor, input_list, op=dist.ReduceOp.SUM)
    print(f'After reduce_scatter - Rank {rank_id} has data: {output_tensor}')


def test_object_broadcast(rank_id, world_size, use_cpu=False):
    """Test broadcasting Python objects."""
    print(f'\n=== Testing Object Broadcast (Rank {rank_id}) ===')

    # 准备要广播的对象
    if rank_id == 0:
        obj_list = [{'loss': 0.5, 'rank': rank_id}, ['accuracy', 0.95], 100]
    else:
        obj_list = [None, None, None]

    print(f'Before object broadcast - Rank {rank_id} has data: {obj_list}')
    dist.broadcast_object_list(obj_list, src=0)
    print(f'After object broadcast - Rank {rank_id} has data: {obj_list}')


def run_all_tests(rank_id, world_size, use_cpu=False):
    """Run all distributed communication tests."""
    print(f'Starting distributed tests on Rank {rank_id}/{world_size-1}')

    # 同步所有进程
    dist.barrier()

    # 运行各项测试
    test_broadcast(rank_id, world_size, use_cpu)

    dist.barrier()  # 确保测试之间同步

    test_all_reduce(rank_id, world_size, use_cpu)

    dist.barrier()

    test_all_gather(rank_id, world_size, use_cpu)

    dist.barrier()

    test_scatter(rank_id, world_size, use_cpu)

    dist.barrier()

    test_reduce(rank_id, world_size, use_cpu)

    dist.barrier()

    # test_reduce_scatter(rank_id, world_size, use_cpu)

    # dist.barrier()

    test_object_broadcast(rank_id, world_size, use_cpu)

    # 最终同步
    dist.barrier()
    print(f'\nAll tests completed on Rank {rank_id}')


if __name__ == '__main__':

    # 初始化分布式环境（如果处于分布式环境）
    init_dist_process(use_cpu=True)

    # 获取本地数据
    rank_id = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    # 运行所有测试
    run_all_tests(rank_id, world_size, use_cpu=True)
