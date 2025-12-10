from typing import Any, Dict, List, Optional

import torch

import scaletorch.dist as dist
from scaletorch.dist import get_device
from scaletorch.utils.logger_utils import get_logger

logger = get_logger(__name__)


def test_scatter() -> None:
    """Test scatter function.

    Verifies that data from the source process is correctly scattered
    to all processes in the group.
    """
    logger.info('Testing scatter...')
    world_size = dist.get_world_size()
    tensor_size = 4
    device = get_device()

    # 接收 Tensor
    data_recv = torch.zeros(tensor_size, dtype=torch.int64, device=device)

    scatter_list_split: Optional[List[torch.Tensor]] = None
    if dist.get_rank() == 0:
        # 总数据
        full_data = torch.arange(world_size * tensor_size,
                                 dtype=torch.int64,
                                 device=device)
        # 拆分成 world_size 份
        scatter_list_split = list(torch.chunk(full_data, world_size))

    dist.scatter(data_recv, src=0, scatter_list=scatter_list_split)

    # 验证结果
    expected = torch.arange(dist.get_rank() * tensor_size,
                            (dist.get_rank() + 1) * tensor_size,
                            dtype=torch.int64,
                            device=device)
    assert torch.equal(
        data_recv,
        expected), f'Scatter failed: expected {expected}, got {data_recv}'
    logger.info(f'Rank {dist.get_rank()}: scatter test passed')


def test_reduce() -> None:
    """Test reduce function.

    Verifies that tensor data is correctly reduced across all processes
    and sent to the destination process.
    """
    logger.info('Testing reduce...')
    world_size = dist.get_world_size()
    device = get_device()

    # 测试 sum 操作
    data = torch.ones(4, dtype=torch.float32,
                      device=device) * (dist.get_rank() + 1)
    original_data = data.clone()
    dist.reduce(data, dst=0, op='sum')

    if dist.get_rank() == 0:
        expected_sum = torch.ones(4, dtype=torch.float32, device=device) * sum(
            range(1, world_size + 1))
        assert torch.allclose(
            data, expected_sum
        ), f'Reduce sum failed: expected {expected_sum}, got {data}'
    logger.info(f'Rank {dist.get_rank()}: reduce sum test passed')

    # 测试 max 操作
    data = original_data.clone()
    dist.reduce(data, dst=0, op='max')

    if dist.get_rank() == 0:
        expected_max = torch.ones(4, dtype=torch.float32,
                                  device=device) * world_size
        assert torch.allclose(
            data, expected_max
        ), f'Reduce max failed: expected {expected_max}, got {data}'
    logger.info(f'Rank {dist.get_rank()}: reduce max test passed')

    # 测试 mean 操作
    data = original_data.clone()
    dist.reduce(data, dst=0, op='mean')

    if dist.get_rank() == 0:
        expected_mean = torch.ones(4, dtype=torch.float32,
                                   device=device) * (world_size + 1) / 2
        assert torch.allclose(
            data, expected_mean
        ), f'Reduce mean failed: expected {expected_mean}, got {data}'
    logger.info(f'Rank {dist.get_rank()}: reduce mean test passed')


def test_reduce_scatter() -> None:
    """Test reduce_scatter function.

    Verifies that tensor data is correctly reduced across all processes
    and scattered to all processes in the group.
    """
    logger.info('Testing reduce_scatter...')
    world_size = dist.get_world_size()
    # Ensure tensor_size is divisible by world_size to create equal chunks
    tensor_size = world_size * 2
    device = get_device()

    # 每个进程的数据都是不同的
    data = torch.ones(tensor_size, dtype=torch.float32,
                      device=device) * (dist.get_rank() + 1)

    dist.reduce_scatter(data, op='sum')

    # 验证结果 - 每个进程应该得到对应块的总和
    expected = torch.ones(tensor_size // world_size,
                          dtype=torch.float32,
                          device=device) * sum(range(1, world_size + 1))
    assert torch.allclose(
        data,
        expected), f'Reduce scatter failed: expected {expected}, got {data}'
    logger.info(f'Rank {dist.get_rank()}: reduce_scatter test passed')

    # 测试 max 操作
    data = torch.ones(tensor_size, dtype=torch.float32,
                      device=device) * (dist.get_rank() + 1)
    dist.reduce_scatter(data, op='max')

    expected = torch.ones(tensor_size // world_size,
                          dtype=torch.float32,
                          device=device) * world_size
    assert torch.allclose(
        data, expected
    ), f'Reduce scatter max failed: expected {expected}, got {data}'
    logger.info(f'Rank {dist.get_rank()}: reduce_scatter max test passed')


def test_all_to_all() -> None:
    """Test all_to_all function.

    Verifies that tensor data is correctly redistributed among all processes,
    with each process sending and receiving data from all other processes.
    """
    logger.info('Testing all_to_all...')
    world_size = dist.get_world_size()
    tensor_size = world_size * 4  # 确保可以被world_size整除
    device = get_device()

    # 生成以rank区分的输入数据
    start_value = dist.get_rank() * tensor_size
    input_data = torch.arange(start_value,
                              start_value + tensor_size,
                              dtype=torch.int64,
                              device=device)

    output_data = dist.all_to_all(input_data)

    # 验证结果
    chunk_size = tensor_size // world_size
    expected_chunks = []
    for src_rank in range(world_size):
        # src_rank发送给当前rank的数据块
        chunk_start = src_rank * tensor_size + dist.get_rank() * chunk_size
        chunk_end = src_rank * tensor_size + (dist.get_rank() + 1) * chunk_size
        expected_chunks.append(
            torch.arange(chunk_start,
                         chunk_end,
                         dtype=torch.int64,
                         device=device))

    expected_output = torch.cat(expected_chunks)
    assert torch.equal(
        output_data, expected_output
    ), f'All-to-all failed: expected {expected_output}, got {output_data}'
    logger.info(f'Rank {dist.get_rank()}: all_to_all test passed')


def test_all_reduce() -> None:
    """Test all_reduce function.

    Verifies that tensor data is correctly reduced across all processes
    and the result is available on all processes.
    """
    logger.info('Testing all_reduce...')
    world_size = dist.get_world_size()
    device = get_device()

    # 测试 sum 操作
    data = torch.ones(4, dtype=torch.float32,
                      device=device) * (dist.get_rank() + 1)
    dist.all_reduce(data, op='sum')

    expected_sum = torch.ones(4, dtype=torch.float32, device=device) * sum(
        range(1, world_size + 1))
    assert torch.allclose(
        data, expected_sum
    ), f'All-reduce sum failed: expected {expected_sum}, got {data}'
    logger.info(f'Rank {dist.get_rank()}: all_reduce sum test passed')

    # 测试 max 操作
    data = torch.ones(4, dtype=torch.float32,
                      device=device) * (dist.get_rank() + 1)
    dist.all_reduce(data, op='max')

    expected_max = torch.ones(4, dtype=torch.float32,
                              device=device) * world_size
    assert torch.allclose(
        data, expected_max
    ), f'All-reduce max failed: expected {expected_max}, got {data}'
    logger.info(f'Rank {dist.get_rank()}: all_reduce max test passed')

    # 测试 mean 操作
    data = torch.ones(4, dtype=torch.float32,
                      device=device) * (dist.get_rank() + 1)
    dist.all_reduce(data, op='mean')

    expected_mean = torch.ones(4, dtype=torch.float32,
                               device=device) * (world_size + 1) / 2
    assert torch.allclose(
        data, expected_mean
    ), f'All-reduce mean failed: expected {expected_mean}, got {data}'
    logger.info(f'Rank {dist.get_rank()}: all_reduce mean test passed')


def test_all_gather() -> None:
    """Test all_gather function.

    Verifies that tensor data from all processes is correctly gathered
    and made available on all processes.
    """
    logger.info('Testing all_gather...')
    world_size = dist.get_world_size()
    device = get_device()

    # 每个进程准备自己的数据
    local_data = torch.ones(3, dtype=torch.float32,
                            device=device) * (dist.get_rank() + 1)

    gathered_data = dist.all_gather(local_data)

    # 验证结果
    assert len(
        gathered_data
    ) == world_size, f'Expected {world_size} tensors, got {len(gathered_data)}'
    for i in range(world_size):
        expected = torch.ones(3, dtype=torch.float32, device=device) * (i + 1)
        assert torch.allclose(gathered_data[i],
                              expected), f'All-gather failed for rank {i}'
    logger.info(f'Rank {dist.get_rank()}: all_gather test passed')


def test_gather() -> None:
    """Test gather function.

    Verifies that tensor data from all processes is correctly gathered
    to the destination process.
    """
    logger.info('Testing gather...')
    world_size = dist.get_world_size()
    device = get_device()

    # 每个进程准备自己的数据
    local_data = torch.ones(3, dtype=torch.float32,
                            device=device) * (dist.get_rank() + 1)

    gathered_data = dist.gather(local_data, dst=0)

    # 验证结果
    if dist.get_rank() == 0:
        assert len(
            gathered_data
        ) == world_size, f'Expected {world_size} tensors, got {len(gathered_data)}'
        for i in range(world_size):
            expected = torch.ones(3, dtype=torch.float32,
                                  device=device) * (i + 1)
            assert torch.allclose(gathered_data[i],
                                  expected), f'Gather failed for rank {i}'
        logger.info(f'Rank {dist.get_rank()}: gather test passed')
    else:
        assert gathered_data == [], f'Expected empty list, got {gathered_data}'
        logger.info(f'Rank {dist.get_rank()}: gather test passed (empty list)')


def test_broadcast() -> None:
    """Test broadcast function.

    Verifies that data from the source process is correctly broadcast
    to all processes in the group.
    """
    logger.info('Testing broadcast...')
    device = get_device()

    # 只有rank 0有有效数据
    if dist.get_rank() == 0:
        data = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
    else:
        data = torch.tensor([0.0, 0.0, 0.0, 0.0], device=device)  # 其他进程用占位符数据

    dist.broadcast(data, src=0)

    # 验证所有进程都有相同数据
    expected = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
    assert torch.allclose(
        data, expected), f'Broadcast failed: expected {expected}, got {data}'
    logger.info(f'Rank {dist.get_rank()}: broadcast test passed')


def test_sync_random_seed() -> None:
    """Test sync_random_seed function.

    Verifies that a random seed is synchronized across all processes.
    """
    logger.info('Testing sync_random_seed...')
    seed = dist.sync_random_seed()
    device = get_device()

    # 所有进程应该获得相同的种子
    seeds_tensor = torch.tensor(seed, device=device)
    seeds = dist.all_gather(seeds_tensor)
    assert all(
        s.item() == seed
        for s in seeds), 'Sync random seed failed: not all seeds are equal'
    logger.info(
        f'Rank {dist.get_rank()}: sync_random_seed test passed (seed={seed})')


def test_broadcast_object_list() -> None:
    """Test broadcast_object_list function.

    Verifies that Python objects are correctly broadcast from source
    process to all processes in the group.
    """
    logger.info('Testing broadcast_object_list...')
    # 只有rank 0有有效数据
    data: List[Any] = []
    if dist.get_rank() == 0:
        data = ['test_string', {'key': 'value'}, [1, 2, 3]]
    else:
        data = [None, None, None]  # 其他进程用占位符数据

    dist.broadcast_object_list(data, src=0)

    # 验证所有进程都有相同数据
    expected: List[Any] = ['test_string', {'key': 'value'}, [1, 2, 3]]
    assert data == expected, f'Broadcast object list failed: expected {expected}, got {data}'
    logger.info(f'Rank {dist.get_rank()}: broadcast_object_list test passed')


def test_all_reduce_dict() -> None:
    """Test all_reduce_dict function.

    Verifies that dictionary data is correctly reduced across all processes
    and the result is available on all processes.
    """
    logger.info('Testing all_reduce_dict...')
    world_size = dist.get_world_size()
    device = get_device()

    # 每个进程准备自己的字典数据
    data: Dict[str, torch.Tensor] = {
        'key1':
        torch.ones(2, dtype=torch.float32, device=device) *
        (dist.get_rank() + 1),
        'key2':
        torch.ones(3, dtype=torch.float32, device=device) *
        (dist.get_rank() + 1)
    }

    dist.all_reduce_dict(data, op='sum')

    # 验证结果
    expected_sum = sum(range(1, world_size + 1))
    expected: Dict[str, torch.Tensor] = {
        'key1':
        torch.ones(2, dtype=torch.float32, device=device) * expected_sum,
        'key2':
        torch.ones(3, dtype=torch.float32, device=device) * expected_sum
    }

    for k in data.keys():
        assert torch.allclose(
            data[k], expected[k]), f'All-reduce dict failed for key {k}'
    logger.info(f'Rank {dist.get_rank()}: all_reduce_dict test passed')

    # 测试 max 操作
    data_max: Dict[str, torch.Tensor] = {
        'key1':
        torch.ones(2, dtype=torch.float32, device=device) *
        (dist.get_rank() + 1),
        'key2':
        torch.ones(3, dtype=torch.float32, device=device) *
        (dist.get_rank() + 1)
    }

    dist.all_reduce_dict(data_max, op='max')

    expected_max: Dict[str, torch.Tensor] = {
        'key1': torch.ones(2, dtype=torch.float32, device=device) * world_size,
        'key2': torch.ones(3, dtype=torch.float32, device=device) * world_size
    }

    for k in data_max.keys():
        assert torch.allclose(
            data_max[k],
            expected_max[k]), f'All-reduce dict max failed for key {k}'
    logger.info(f'Rank {dist.get_rank()}: all_reduce_dict max test passed')


def test_all_gather_object() -> None:
    """Test all_gather_object function.

    Verifies that Python objects from all processes are correctly gathered
    and made available on all processes.
    """
    logger.info('Testing all_gather_object...')
    world_size = dist.get_world_size()

    # 每个进程准备自己的对象数据
    local_data = f'data_from_rank_{dist.get_rank()}'

    gathered_data = dist.all_gather_object(local_data)

    # 验证结果
    assert len(
        gathered_data
    ) == world_size, f'Expected {world_size} objects, got {len(gathered_data)}'
    for i in range(world_size):
        expected = f'data_from_rank_{i}'
        assert gathered_data[
            i] == expected, f'All-gather object failed for rank {i}'
    logger.info(f'Rank {dist.get_rank()}: all_gather_object test passed')


def test_gather_object() -> None:
    """Test gather_object function.

    Verifies that Python objects from all processes are correctly gathered
    to the destination process.
    """
    logger.info('Testing gather_object...')
    world_size = dist.get_world_size()

    # 每个进程准备自己的对象数据
    local_data = f'data_from_rank_{dist.get_rank()}'

    gathered_data = dist.gather_object(local_data, dst=0)

    # 验证结果
    if dist.get_rank() == 0:
        assert len(
            gathered_data
        ) == world_size, f'Expected {world_size} objects, got {len(gathered_data)}'
        for i in range(world_size):
            expected = f'data_from_rank_{i}'
            assert gathered_data[
                i] == expected, f'Gather object failed for rank {i}'
        logger.info(f'Rank {dist.get_rank()}: gather_object test passed')
    else:
        assert gathered_data is None, f'Expected None, got {gathered_data}'
        logger.info(
            f'Rank {dist.get_rank()}: gather_object test passed (None)')


def test_all_reduce_params() -> None:
    """Test all_reduce_params function.

    Verifies that parameters/buffers are correctly reduced across all processes.
    """
    logger.info('Testing all_reduce_params...')
    world_size = dist.get_world_size()
    device = get_device()

    # 准备参数列表
    params: List[torch.Tensor] = [
        torch.ones(2, dtype=torch.float32, device=device) *
        (dist.get_rank() + 1),
        torch.ones(3, dtype=torch.float32, device=device) *
        (dist.get_rank() + 1)
    ]

    dist.all_reduce_params(params, op='sum')

    # 验证结果
    expected_sum = sum(range(1, world_size + 1))
    expected: List[torch.Tensor] = [
        torch.ones(2, dtype=torch.float32, device=device) * expected_sum,
        torch.ones(3, dtype=torch.float32, device=device) * expected_sum
    ]

    for i in range(len(params)):
        assert torch.allclose(
            params[i], expected[i]), f'All-reduce params failed for param {i}'
    logger.info(f'Rank {dist.get_rank()}: all_reduce_params test passed')

    # 测试 max 操作
    params_max: List[torch.Tensor] = [
        torch.ones(2, dtype=torch.float32, device=device) *
        (dist.get_rank() + 1),
        torch.ones(3, dtype=torch.float32, device=device) *
        (dist.get_rank() + 1)
    ]

    dist.all_reduce_params(params_max, op='max')

    expected_max: List[torch.Tensor] = [
        torch.ones(2, dtype=torch.float32, device=device) * world_size,
        torch.ones(3, dtype=torch.float32, device=device) * world_size
    ]

    for i in range(len(params_max)):
        assert torch.allclose(
            params_max[i],
            expected_max[i]), f'All-reduce params max failed for param {i}'
    logger.info(f'Rank {dist.get_rank()}: all_reduce_params max test passed')


def run_all_tests() -> None:
    """Run all distributed tests."""
    # Run all test functions

    tests = [
        test_scatter,
        test_reduce,
        test_reduce_scatter,
        test_all_to_all,
        test_all_reduce,
        test_all_gather,
        test_gather,
        test_broadcast,
        test_sync_random_seed,
        test_broadcast_object_list,
        test_all_reduce_dict,
        test_all_gather_object,
        test_gather_object,
        test_all_reduce_params,
    ]

    failed_tests = []
    passed_tests = []

    for test_func in tests:
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f'Running {test_func.__name__}...')
            test_func()
            passed_tests.append(test_func.__name__)
            logger.info(f'✓ {test_func.__name__} passed')
        except Exception as e:
            failed_tests.append((test_func.__name__, str(e)))
            logger.error(f'✗ {test_func.__name__} failed: {e}')
            # 可以选择是否继续运行其他测试
            continue

    # 记录测试总结
    logger.info(f"\n{'='*60}")
    logger.info('Test Summary:')
    logger.info(f'Passed: {len(passed_tests)}/{len(tests)}')
    logger.info(f'Failed: {len(failed_tests)}/{len(tests)}')

    if failed_tests:
        logger.info('\nFailed tests:')
        for test_name, error in failed_tests:
            logger.info(f'  - {test_name}: {error}')

    if failed_tests:
        raise RuntimeError(f'{len(failed_tests)} tests failed')


if __name__ == '__main__':

    try:
        # 初始化分布式环境
        dist.init_dist(launcher='pytorch', backend='nccl')

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # 设置日志
        logger.info(f'Running tests on Rank {rank}/{world_size-1}')

        # 运行所有测试
        run_all_tests()
        logger.info(f'Rank {rank}: All tests passed!')

    except Exception as e:
        logger.error(f'Rank {rank}: Test failed with error: {e}')
        raise
    finally:
        dist.cleanup_dist()
