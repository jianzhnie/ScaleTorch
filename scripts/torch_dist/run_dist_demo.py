import os

import torch
import torch.distributed as dist
from transformers.utils import is_torch_cuda_available, is_torch_npu_available


def rank_print(rank, msg):
    print(f'[Rank {rank}] {msg}', flush=True)


def get_dist_info():
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    return rank, world_size, local_rank


def get_current_device(device):
    """Get current device based on available backends."""
    _, _, local_rank = get_dist_info()

    if device:
        return torch.device('cpu')
    if is_torch_cuda_available():
        return torch.device('cuda', local_rank)
    if is_torch_npu_available():
        return torch.device('npu', local_rank)
    return torch.device('cpu')


def init_dist_process(device, timeout=120) -> None:
    """Initialize distributed process group."""
    if dist.is_initialized():
        return

    rank, world_size, local_rank = get_dist_info()

    if world_size == 1:
        return  # 单进程直接返回

    if device:
        backend = 'gloo'
    elif is_torch_cuda_available():
        backend = 'nccl'
        torch.cuda.set_device(local_rank)
    elif is_torch_npu_available():
        backend = 'hccl'
        torch.npu.set_device(local_rank)
    else:
        backend = 'gloo'

    dist.init_process_group(backend=backend,
                            rank=rank,
                            world_size=world_size,
                            timeout=dist.timedelta(seconds=timeout))


def test_broadcast(rank, world_size, device):
    """Test broadcast communication."""
    rank_print(rank, 'Testing broadcast')
    # 创建不同rank的数据
    if rank == 0:
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
    else:
        tensor = torch.zeros(4, device=device)

    rank_print(rank, f'Before broadcast - Rank {rank} has data: {tensor}')
    dist.broadcast(tensor, src=0)
    rank_print(rank, f'After broadcast - Rank {rank} has data: {tensor}')


def test_gather(rank, world_size, device):
    """Test gather communication."""
    rank_print(rank, 'Testing gather')
    # 创建本地数据
    local_data = torch.tensor([rank, rank * 2],
                              dtype=torch.float32,
                              device=device)
    rank_print(rank, f'Local data - Rank {rank} has data: {local_data}')

    # 只在rank 0上准备接收缓冲区
    gather_list = None
    if rank == 0:
        gather_list = [torch.zeros_like(local_data) for _ in range(world_size)]

    # 执行gather操作
    dist.gather(local_data, gather_list, dst=0)
    if rank == 0:
        rank_print(
            rank,
            f'Gathered data - Rank {rank} received: {[t.tolist() for t in gather_list]}'
        )


def test_scatter(rank, world_size, device):
    """Test scatter communication."""
    rank_print(rank, 'Testing scatter')

    # 接收张量
    recv_tensor = torch.zeros(4, dtype=torch.int64, device=device)

    # 只在rank 0准备scatter数据
    scatter_list = None
    if rank == 0:
        # 创建总数据并拆分
        full_data = torch.arange(world_size * 4,
                                 dtype=torch.int64,
                                 device=device)
        scatter_list = list(torch.chunk(full_data, world_size))
        rank_print(
            rank,
            f'Rank 0 scattering data: {[t.tolist() for t in scatter_list]}')

    rank_print(rank, f'Before scatter - Rank {rank} has data: {recv_tensor}')
    # 执行scatter操作
    dist.scatter(recv_tensor, scatter_list, src=0)
    rank_print(
        rank, f'After scatter - Rank {rank} received: {recv_tensor.tolist()}')


def test_reduce(rank, world_size, device):
    """Test reduce communication."""
    rank_print(rank, 'Testing reduce')

    # 创建本地数据
    tensor = torch.tensor([rank, rank + 1, rank + 2],
                          dtype=torch.float32,
                          device=device)
    rank_print(rank, f' Before reduce - Rank {rank} has data: {tensor}')

    # 执行reduce操作，只在rank 0接收结果
    dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)
    if rank == 0:
        rank_print(
            rank,
            f'After reduce - Rank {rank} (destination) has data: {tensor}')


def test_all_gather(rank, world_size, device):
    """Test all_gather communication."""
    rank_print(rank, 'Testing all_gather')

    # 创建本地数据
    local_data = torch.tensor([rank, rank * 2, rank * 3],
                              dtype=torch.float32,
                              device=device)
    rank_print(rank, f'Local data - Rank {rank} has data: {local_data}')

    # 收集所有进程的数据
    gathered_list = [torch.zeros_like(local_data) for _ in range(world_size)]
    dist.all_gather(gathered_list, local_data)
    rank_print(
        rank,
        f'Gathered data - Rank {rank} received: {[t.tolist() for t in gathered_list]}'
    )


def test_all_reduce(rank, world_size, device):
    """Test all_reduce communication."""
    rank_print(rank, 'Testing all_reduce')

    # 创建本地数据
    tensor = torch.tensor([1.0, 2.0, 3.0], device=device) * (rank + 1)
    rank_print(rank, f'Before all_reduce - Rank {rank} has data: {tensor}')

    # 执行全局求和归约
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    rank_print(rank, f'After all_reduce sum - Rank {rank} has data: {tensor}')


def test_reduce_scatter(rank, world_size, device):
    """Test reduce_scatter communication."""
    rank_print(rank, 'Testing reduce_scatter')

    # 每个rank创建本地数据
    input_list = []
    for i in range(world_size):
        tensor = torch.tensor(
            [i + 1,
             (i + 1) * 2], dtype=torch.float32, device=device) * (rank + 1)
        input_list.append(tensor)

    input_tensor = torch.cat(input_list)
    output_tensor = torch.zeros_like(input_list[0])

    rank_print(
        rank, f'Before reduce_scatter - Rank {rank} has data: {input_tensor}')

    # 执行reduce_scatter操作
    dist.reduce_scatter(output_tensor, input_list, op=dist.ReduceOp.SUM)
    rank_print(
        rank, f'After reduce_scatter - Rank {rank} has data: {output_tensor}')


def test_all_to_all(rank, world_size, device):
    """Test all_to_all communication."""
    rank_print(rank, 'Testing all_to_all')

    # 每个rank创建本地数据
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

    # 准备输出列表
    output_list = [
        torch.zeros(1, dtype=torch.float32, device=device)
        for _ in range(world_size)
    ]

    # 执行all_to_all操作
    dist.all_to_all(output_list, input_list)
    rank_print(
        rank,
        f'Output data - Rank {rank} received: {[t.tolist() for t in output_list]}'
    )


def test_object_broadcast(rank, world_size, device):
    """Test broadcasting Python objects."""
    rank_print(rank, 'Testing object broadcast')

    # 准备要广播的对象
    if rank == 0:
        obj_list = [{'loss': 0.5, 'rank': rank}, ['accuracy', 0.95], 100]
    else:
        obj_list = [None, None, None]

    rank_print(rank,
               f'Before object broadcast - Rank {rank} has data: {obj_list}')
    dist.broadcast_object_list(obj_list, src=0)
    rank_print(rank,
               f'After object broadcast - Rank {rank} has data: {obj_list}')


def run_all_tests(use_cpu: bool = False):
    """Run all distributed communication tests."""
    rank, world_size, _ = get_dist_info()
    device = get_current_device(use_cpu)

    rank_print(rank,
               f'Starting distributed tests on Rank {rank}/{world_size-1}')

    # 运行各项测试
    test_broadcast(rank, world_size, device)

    dist.barrier()  # 确保测试之间同步

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
        dist.barrier()

    rank_print(rank, 'All tests finished')


if __name__ == '__main__':
    # 初始化分布式环境（如果处于分布式环境）
    # 可以通过环境变量控制是否使用CPU进行测试
    use_cpu = os.environ.get('USE_CPU', 'false').lower() == 'true'
    init_dist_process(use_cpu)
    run_all_tests(use_cpu)
