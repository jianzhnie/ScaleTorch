import torch

import scaletorch.dist as dist

if __name__ == '__main__':

    # 初始化分布式环境（如果处于分布式环境）

    dist.init_dist(launcher='pytorch', backend='nccl')

    # 创建本地数据
    data = torch.tensor([1.0, 2.0, 3.0])

    # 执行全局求和归约
    dist.all_reduce(data, op='sum')
    print(f'Rank {dist.get_rank()}: {data}')

    # 执行全局平均归约
    data = torch.tensor([1.0, 2.0, 3.0])
    dist.all_reduce(data, op='mean')
    print(f'Rank {dist.get_rank()}: {data}')

    print('Scatter Example:')
    world_size = dist.get_world_size()  # 假设 world_size = 8
    tensor_size = 4  # 每个进程接收 4 个元素

    # 接收 Tensor 的形状必须是 (4,)
    data_recv = torch.zeros(tensor_size, dtype=torch.int64)
    print(f'Rank {dist.get_rank()}: Recv Tensor Shape {data_recv.shape}')

    if dist.get_rank() == 0:
        # 总数据 (8*4,)
        full_data = torch.arange(world_size * tensor_size, dtype=torch.int64)

        # 拆分成 8 份，每份 (4,)
        scatter_list_split = list(torch.chunk(full_data, world_size))
        print(
            f'Rank 0: Scatter List Shapes {[t.shape for t in scatter_list_split]}'
        )
    else:
        scatter_list_split = None

    dist.scatter(data_recv, src=0, scatter_list=scatter_list_split)
    print(f'Rank {dist.get_rank()}: Scatter result (Split): {data_recv}')

    # Reduce 求和示例
    rank = dist.get_rank()
    initial_data = torch.arange(8, dtype=torch.float32)
    data_sum = initial_data.clone()  # 使用克隆的张量进行操作

    print(f'Rank {rank} 初始数据: {data_sum}')

    # 执行 Reduce 求和操作到目标进程 dst=0 (默认)
    # 所有进程的数据在 Rank 0 上求和。
    # 假设世界大小为 W，Rank i 上的数据为 D_i
    # Rank 0 结果 = SUM(D_i)
    # 其他 Rank 结果保持 D_i 不变
    dist.reduce(data_sum, dst=0, op='sum')

    # 只有 Rank 0 会打印归约后的结果
    if rank == 0:
        # 预期结果：world_size * initial_data (如果所有 Rank 初始数据相同)
        print(f'➡️ Rank {rank} 归约求和 (SUM) 结果: {data_sum}')
    else:
        # 其他 Rank 的数据保持不变
        print(f'   Rank {rank} 归约求和后数据: {data_sum} (未更新)')

    # 2. 示例：求最大值 (op='max')
    # 创建一个不同的初始数据，用于更清晰地展示 max 效果
    max_data = initial_data.clone()
    # 仅 Rank 1 修改数据，使其值更高
    if rank == 1:
        max_data += 10

    print(f'Rank {rank} 新初始数据: {max_data}')

    # 执行 Reduce 求最大值操作到目标进程 dst=0 (默认)
    dist.reduce(max_data, dst=0, op='max')

    if rank == 0:
        # Rank 0 结果是所有进程对应元素的 Max
        print(f'➡️ Rank {rank} 归约求最大值 (MAX) 结果: {max_data}')
    else:
        print(f'   Rank {rank} 归约求最大值后数据: {max_data} (未更新)')

    # 假设 WORLD_SIZE = 8
    # 假设 rank 为当前进程 ID (0, 1, 2, or 3)

    # --- 优化后的输入数据定义 ---
    # 1. 定义输入数据
    # 使用 size = 16 确保能被 WORLD_SIZE 整除 (16/4 = 4)
    TENSOR_SIZE = 16
    CHUNK_SIZE = TENSOR_SIZE // world_size

    # 生成一个以 Rank 区分的、可预测的输入张量。
    # 例如，如果 rank=1, WORLD_SIZE=4:
    # input_data = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    # 这确保了 Rank 1 的数据块总是比 Rank 0 的数据块大。
    start_value = rank * TENSOR_SIZE
    input_data = torch.arange(start_value,
                              start_value + TENSOR_SIZE,
                              dtype=torch.int64)

    print(
        f'Rank {rank}: Input Data Shape {input_data.shape}, Content: {input_data}'
    )

    # 2. 调用待测试的 all_to_all 函数
    # 假设 dist 已经导入并包含了 all_to_all 函数
    output_data = dist.all_to_all(input_data)

    print(
        f'Rank {rank}: Output Data Shape {output_data.shape}, Content: {output_data}'
    )

    # --- 3. 结果校验（关键） ---
    # 检查输出张量是否与预期结果一致
    # 预期的输出张量形状与输入张量形状相同 (16,)
    #
    # Rank i 接收的数据：
    # - 来自 Rank 0 的第 i 个分块
    # - 来自 Rank 1 的第 i 个分块
    # - ...
    # - 来自 Rank W-1 的第 i 个分块
    #
    # 每个 Rank j 的第 i 个分块是：
    # torch.arange(j*TENSOR_SIZE + i*CHUNK_SIZE, j*TENSOR_SIZE + (i+1)*CHUNK_SIZE)
    expected_chunks = []
    for src_rank in range(world_size):
        # Rank src_rank 发送给 Rank rank 的分块
        # 该分块是 Rank src_rank 输入张量的第 rank 个分块
        chunk_start = src_rank * TENSOR_SIZE + rank * CHUNK_SIZE
        chunk_end = src_rank * TENSOR_SIZE + (rank + 1) * CHUNK_SIZE
        expected_chunks.append(
            torch.arange(chunk_start, chunk_end, dtype=torch.int64))

    expected_output = torch.cat(expected_chunks)

    assert torch.equal(output_data, expected_output), \
        f'Rank {rank} FAILED: Expected {expected_output}, but got {output_data}'

    print(f'Rank {rank} SUCCESS! Verified Output.')
