# ScaleTorch 分布式计算模块文档

## 概述
在分布式训练或测试的过程中，不同进程有时需要根据分布式的环境信息执行不同的代码逻辑，同时不同进程之间也经常会有相互通信的需求，对一些数据进行同步等操作。 PyTorch 提供了一套基础的通信原语用于多进程之间张量的通信，基于这套原语，ScaleTorch 实现了更高层次的通信原语封装以满足更加丰富的需求。

## 1. 功能介绍

`scaletorch.dist.dist` 是 ScaleTorch 库中提供的分布式计算核心模块，它封装并扩展了 PyTorch 的分布式通信功能，为用户提供了更便捷、更高效的分布式训练支持。该模块主要功能包括：

### 1.1 核心功能

- **分布式通信原语**：提供 broadcast、gather、scatter、reduce、 reduce_scatter、all_reduce、all_gather、all_to_all 等常用分布式操作
- **Python对象支持**：支持直接广播和收集Python对象（如字典、列表等），无需手动序列化
- **参数归约**：高效的模型参数和梯度归约
- **设备兼容性**：支持 CPU、GPU、NPU、MLU、MUSA 等多种计算设备
- **多后端支持**：兼容 NCCL、Gloo、MPI、HCCL、CNCL、MCCL 等多种通信后端
- **简化接口**：相比原生 PyTorch 分布式接口，提供了更简洁易用的 API

### 1.2 主要特性

- **非分布式环境兼容**：所有函数在非分布式环境下自动降级，保证代码一致性
- **自动设备管理**：自动处理不同设备间的数据转换和通信
- **错误处理**：提供详细的错误提示和边界情况处理
- **性能优化**：支持张量合并归约（coalesced reduction）以减少通信次数
- **多框架集成**：支持 DeepSpeed、ColossalAI 等分布式训练框架

### 1.3 适用场景

- 多GPU/多节点分布式训练
- 模型参数同步和梯度聚合
- 跨进程数据收集和分发
- 分布式评估和结果收集
- 随机种子同步确保可重复性

## 2. 使用教程

### 2.1 安装与导入

#### 安装 ScaleTorch

```bash
pip install scaletorch
```

#### 导入模块
```python
import torch
import scaletorch.dist as dist
```

### 2.2 初始化分布式环境

- [init_dist](dist.init_dist)： 是分布式训练的启动函数，目前支持 pytorch，slurm，MPI 3 种分布式启动方式，同时允许设置通信的后端，默认使用 NCCL。

#### 方法一：使用 PyTorch 启动器

```python
import torch
import torch.distributed as dist
import os

# 设置环境变量（通常由启动脚本设置）
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'

# 初始化分布式环境
dist.init_process_group(backend='nccl')

# 在命令行使用torchrun启动
# torchrun --nproc_per_node=2 your_script.py
```

#### 方法二：使用 SLURM 集群
```python
dist.init_dist(launcher='slurm', backend='nccl', port=29500)
```

#### 方法三：使用 MPI
```python
dist.init_dist(launcher='mpi', backend='nccl')
```

#### 方法四：非分布式环境（单进程）
```python
# 无需初始化，所有函数会自动降级处理
```

### 2.3 基本使用示例

#### 示例1：张量规约（all_reduce）
```python
import torch
import scaletorch.dist as dist

# 初始化分布式环境（如果处于分布式环境）
dist.init_dist(launcher='pytorch', backend='nccl')

# 创建本地数据
data = torch.tensor([1.0, 2.0, 3.0])

# 执行全局求和归约
dist.all_reduce(data, op='sum')
print(f"Rank {dist.get_rank()}: {data}")

# 执行全局平均归约
data = torch.tensor([1.0, 2.0, 3.0])
dist.all_reduce(data, op='mean')
print(f"Rank {dist.get_rank()}: {data}")

# 也可以使用其他操作，如平均、最小值、最大值等
# dist.all_reduce(tensor, op='mean')
# dist.all_reduce(tensor, op='min')
# dist.all_reduce(tensor, op='max')
```

#### 示例2：张量收集（all_gather）


```python
import torch
import scaletorch.dist as dist

# 每个进程创建不同的数据
local_data = torch.tensor([dist.get_rank() * 10 + i for i in range(3)])

# 收集所有进程的数据
gathered_data = dist.all_gather(local_data)
print(f"Rank {dist.get_rank()}: gathered {gathered_data}")

# 收集到指定进程（如rank 0）
collected_data = dist.gather(local_data, dst=0)
if dist.get_rank() == 0:
    print(f"Collected on rank 0: {collected_data}")
```

#### 示例3：广播操作 (broadcast)

```python
import torch
import scaletorch.dist as dist

if dist.get_rank() == 0:
    # 只在 rank 0 创建数据
    data_to_broadcast = torch.tensor([1.0, 2.0, 3.0, 4.0])
else:
    data_to_broadcast = torch.zeros(4)

# 广播数据到所有进程
dist.broadcast(data_to_broadcast, src=0)
print(f"Rank {dist.get_rank()}: {data_to_broadcast}")
```

#### 示例4：Python 对象通信
```python
import scaletorch.dist as dist

# 广播Python对象
if dist.get_rank() == 0:
    obj_list = [{"loss": 0.5}, ["accuracy", 0.95], 100]
else:
    obj_list = [None, None, None]

dist.broadcast_object_list(obj_list, src=0)
print(f"Rank {dist.get_rank()}: {obj_list}")

# 收集Python对象
local_obj = {"rank": dist.get_rank(), "value": dist.get_rank() * 10}
all_objs = dist.all_gather_object(local_obj)
print(f"Rank {dist.get_rank()}: {all_objs}")
```

#### 示例5：全到全通信操作 (all_to_all)
```python
import torch
import scaletorch.dist as dist

# 初始化分布式环境（如果处于分布式环境）
dist.init_dist(launcher='pytorch', backend='nccl')

# 每个进程创建不同的数据
data = torch.arange(4, dtype=torch.int64) + dist.get_rank() * 4
print(f"Rank {dist.get_rank()} input: {data}")

# 执行全到全通信
output = dist.all_to_all(data)
print(f"Rank {dist.get_rank()} output: {output}")

# 预期输出（2个进程的情况）：
# Rank 0 input: tensor([0, 1, 2, 3])
# Rank 1 input: tensor([4, 5, 6, 7])
# Rank 0 output: tensor([0, 1, 4, 5])  # 接收两个进程的第一半
# Rank 1 output: tensor([2, 3, 6, 7])  # 接收两个进程的第二半
```

### 2.4 高级功能

#### 参数归约（用于模型训练）
```python
import torch
import torch.nn as nn
import scaletorch.dist as dist

# 创建模型
model = nn.Linear(10, 5)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 训练步骤
for epoch in range(10):
    # 前向传播、反向传播...

    # 同步所有进程的模型参数
    dist.all_reduce_params(model.parameters(), op='mean')

    # 优化器步骤
    optimizer.step()
```

#### 分布式结果收集

```python
import scaletorch.dist as dist

# 每个进程生成部分结果
local_results = [f"result_{dist.get_rank()}_{i}" for i in range(3)]

# 收集所有结果到 rank 0
all_results = dist.collect_results(local_results, size=12, device='cpu')
if dist.get_rank() == 0:
    print(f"All results: {all_results}")
```

#### 同步随机种子
```python
import numpy as np
import scaletorch.dist as dist

# 同步所有进程的随机种子
seed = dist.sync_random_seed()
np.random.seed(seed)
torch.manual_seed(seed)

print(f"Rank {dist.get_rank()}: synchronized seed = {seed}")
```

### 2.5 分布式工具函数

分布式信息的获取与控制函数没有参数，这些函数兼容非分布式训练的情况，功能如下

- [get_world_size](scaletorch.dist.get_world_size)：获取当前进程组的进程总数，非分布式情况下返回 1
- [get_rank](scaletorch.dist.get_rank)：获取当前进程对应的全局 rank 数，非分布式情况下返回 0
- [get_backend](scaletorch.dist.get_backend)：获取当前通信使用的后端，非分布式情况下返回 None
- [get_local_rank](scaletorch.dist.get_local_rank)：获取当前进程对应到当前机器的 rank 数，非分布式情况下返回 0
- [get_local_size](scaletorch.dist.get_local_size)：获取当前进程所在机器的总进程数，非分布式情况下返回 0
- [get_dist_info](scaletorch.dist.get_dist_info)：获取当前任务的进程总数和当前进程对应到全局的 rank 数，非分布式情况下 word_size = 1，rank = 0
- [is_main_process](scaletorch.dist.is_main_process)：判断是否为 0 号主进程，非分布式情况下返回 True
- [master_only](scaletorch.dist.master_only)：函数装饰器，用于修饰只需要全局 0 号进程（rank 0 而不是 local rank 0）执行的函数
- [barrier](scaletorch.dist.barrier)：同步所有进程到达相同位置

```python
import scaletorch.dist as dist

# 获取分布式信息
rank, world_size = dist.get_dist_info()
print(f"Rank: {rank}, World Size: {world_size}")

# 检查是否为分布式环境
is_dist = dist.is_distributed()
print(f"Is distributed: {is_dist}")

# 检查是否为主进程
is_main = dist.is_main_process()
print(f"Is main process: {is_main}")

# 进程同步
dist.barrier()
print(f"Rank {rank} passed barrier")
```

## 3. 分布式通信函数 API

通信函数 （Collective functions），主要用于进程间数据的通信，基于 PyTorch 原生的 all_reduce，all_gather，gather，broadcast 接口，ScaleTorch 提供了如下接口，兼容非分布式训练的情况，并支持更丰富数据类型的通信。

### 3.1 核心函数接口说明

#### `all_reduce(data, op='sum', group=None)`

- **功能**：全局归约操作，所有进程获得相同结果
- **输入参数**：
  - `data` (Tensor)：输入张量，函数原地修改此张量
  - `op` (str)：归约操作类型，可选值：'sum', 'mean', 'product', 'min', 'max', 'band', 'bor', 'bxor'，默认为'sum'
  - `group` (ProcessGroup)：进程组，默认为None（使用默认进程组）
- **输出**：无返回值，原地修改`data`
- **样例**：

```python
# 输入（rank 0）：tensor([1, 2]), (rank 1)：tensor([3, 4])
dist.all_reduce(data, op='sum')
# 输出（所有rank）：tensor([4, 6])
```

#### `all_gather(data, group=None) -> List[Tensor]`
- **功能**：从所有进程收集数据
- **输入参数**：
  - `data` (Tensor)：要收集的本地张量
  - `group` (ProcessGroup)：进程组，默认为None
- **输出**：包含所有进程数据的列表，列表长度为进程数
- **样例**：

```python
# 输入（rank 0）：tensor([1, 2]), (rank 1)：tensor([3, 4])
gathered = dist.all_gather(data)
# 输出（所有rank）：[tensor([1, 2]), tensor([3, 4])]
```

#### `gather(data, dst=0, group=None) -> List[Optional[Tensor]]`
- **功能**：收集所有进程数据到指定目标进程
- **输入参数**：
  - `data` (Tensor)：要收集的本地张量
  - `dst` (int)：目标进程rank，默认为0
  - `group` (ProcessGroup)：进程组，默认为None
- **输出**：
  - 目标进程：包含所有进程数据的列表
  - 非目标进程：空列表
- **样例**：

```python
# 输入（rank 0）：tensor([1, 2]), (rank 1)：tensor([3, 4])
result = dist.gather(data, dst=0)
# 输出（rank 0）：[tensor([1, 2]), tensor([3, 4])]
# 输出（rank 1）：[]
```

#### `broadcast(data, src=0, group=None)`
- **功能**：从源进程广播数据到所有进程
- **输入参数**：
  - `data` (Tensor)：源进程为发送数据，其他进程为接收缓冲区
  - `src` (int)：源进程rank，默认为0
  - `group` (ProcessGroup)：进程组，默认为None
- **输出**：无返回值，原地修改`data`
- **样例**：

```python
# 输入（rank 0）：tensor([1, 2]), (rank 1)：tensor([0, 0])
dist.broadcast(data, src=0)
# 输出（所有rank）：tensor([1, 2])
```

#### `all_to_all(data, group=None) -> Tensor`
- **功能**：全到全通信操作。每个进程将输入张量分割成`world_size`个块，并将第i个块发送给第i个进程。操作完成后，每个进程将接收来自所有其他进程的块并将它们拼接成输出张量。
- **输入参数**：
  - `data` (Tensor)：要发送的输入张量，张量将沿第一个维度分割成`world_size`个块
  - `group` (ProcessGroup)：进程组，默认为None
- **输出**：包含来自所有进程块的输出张量
- **样例**：
```python
# 输入（rank 0）：tensor([0, 1, 2, 3]), (rank 1)：tensor([4, 5, 6, 7])
# 执行：output = dist.all_to_all(data)
# 输出（rank 0）：tensor([0, 1, 4, 5])  # 接收两个进程的第一半
# 输出（rank 1）：tensor([2, 3, 6, 7])  # 接收两个进程的第二半
```

#### `sync_random_seed(group=None) -> int`
- **功能**：同步所有进程的随机种子
- **输入参数**：
  - `group` (ProcessGroup)：进程组，默认为None
- **输出**：同步后的随机种子（整数）
- **样例**：
```python
# 执行：seed = dist.sync_random_seed()
# 输出（所有rank）：相同的随机种子，如587791752
```

#### `broadcast_object_list(data, src=0, group=None)`
- **功能**：支持对任意可被 Pickle 序列化的 Python 对象列表进行广播，基于 broadcast 接口实现
- **输入参数**：
  - `data` (List[Any])：Python对象列表，必须可序列化
  - `src` (int)：源进程rank，默认为0
  - `group` (ProcessGroup)：进程组，默认为None
- **输出**：无返回值，原地修改`data`
- **样例**：

```python
# 输入（rank 0）：['foo', 12, {1: 2}], (rank 1)：[None, None, None]

dist.broadcast_object_list(data, src=0)
# 输出（所有rank）：['foo', 12, {1: 2}]
```

#### `all_reduce_dict(data, op='sum', group=None)`
- **功能**：对 dict 中的内容进行 all_reduce 操作，基于 broadcast 和 all_reduce 接口实现
- **输入参数**：
  - `data` (Dict[str, Tensor])：张量字典，键为字符串
  - `op` (str)：归约操作类型，默认为'sum'
  - `group` (ProcessGroup)：进程组，默认为None
- **输出**：无返回值，原地修改字典值
- **样例**：
```python
# 输入（rank 0）：{'loss': tensor(0.5), 'acc': tensor(0.8)}
#       (rank 1)：{'loss': tensor(0.3), 'acc': tensor(0.9)}
dist.all_reduce_dict(data, op='sum')
# 输出（所有rank）：{'loss': tensor(0.8), 'acc': tensor(1.7)}
```

#### `all_gather_object(data, group=None) -> List[Any]`
- **功能**：基于 all_gather 实现对任意可以被 Pickle 序列化的 Python 对象进行 all_gather 操作
- **输入参数**：
  - `data` (Any)：要收集的本地Python对象，必须可序列化
  - `group` (ProcessGroup)：进程组，默认为None
- **输出**：包含所有进程对象的列表
- **样例**：
```python
# 输入（rank 0）：{'a': 1}, (rank 1)：{'b': 2}
objs = dist.all_gather_object(data)
# 输出（所有rank）：[{'a': 1}, {'b': 2}]
```

#### `gather_object(data, dst=0, group=None) -> Optional[List[Any]]`
- **功能**：将 group 里每个 rank 中任意可被 Pickle 序列化的 Python 对象 gather 到指定的目标 rank
- **输入参数**：
  - `data` (Any)：要收集的本地Python对象
  - `dst` (int)：目标进程rank，默认为0
  - `group` (ProcessGroup)：进程组，默认为None
- **输出**：
  - 目标进程：包含所有进程对象的列表
  - 非目标进程：None
- **样例**：
```python
# 输入（rank 0）：'data0', (rank 1)：'data1'
result = dist.gather_object(data, dst=0)
# 输出（rank 0）：['data0', 'data1']
# 输出（rank 1）：None
```

#### `collect_results(results, size, device='cpu', tmpdir=None) -> Optional[list]`
- **功能**：支持基于 CPU 通信或者 GPU 通信对不同进程间的列表数据进行收集
- **输入参数**：
  - `results` (list)：本地结果列表
  - `size` (int)：总结果数量
  - `device` (str)：设备类型，'cpu'、'gpu'或'npu'
  - `tmpdir` (str)：临时目录路径（仅CPU模式）
- **输出**：
  - rank 0：收集到的完整结果列表
  - 其他rank：None
- **样例**：
```python
# 输入（rank 0）：['res0_1', 'res0_2'], (rank 1)：['res1_1', 'res1_2']
all_res = dist.collect_results(results, size=4, device='cpu')
# 输出（rank 0）：['res0_1', 'res1_1', 'res0_2', 'res1_2']
# 输出（rank 1）：None
```

#### `all_reduce_params(params, coalesce=True, bucket_size_mb=-1, op='sum', group=None)`
- **功能**：归约模型参数或缓冲区
- **输入参数**：
  - `params` (Union[List, Generator])：参数或缓冲区列表/生成器
  - `coalesce` (bool)：是否合并归约，默认为True
  - `bucket_size_mb` (int)：归约桶大小（MB），默认为-1（不限制）
  - `op` (str)：归约操作类型，默认为'sum'
  - `group` (ProcessGroup)：进程组，默认为None
- **输出**：无返回值，原地修改参数
- **样例**：
```python
# 输入（rank 0）：[tensor([1, 2]), tensor([3, 4])]
#       (rank 1)：[tensor([2, 3]), tensor([4, 5])]
dist.all_reduce_params(params, op='sum')
# 输出（所有rank）：[tensor([3, 5]), tensor([7, 9])]
```

### 3.2 设备与数据类型约束

#### 设备支持
- **CPU**：所有操作均支持
- **GPU**：需要NCCL后端
- **NPU**：需要HCCL后端（华为昇腾）
- **MLU**：需要CNCL后端（寒武纪）
- **MUSA**：需要MCCL后端（沐曦）

#### 数据类型约束
1. **张量操作**：支持所有PyTorch数值类型
2. **Python对象操作**：对象必须可通过`pickle`序列化
3. **字典操作**：键必须为字符串，值为张量
4. **跨设备通信**：自动处理设备间数据转换

#### 通信后端兼容性
| 操作          | NCCL | Gloo | MPI | HCCL | CNCL | MCCL |
| ------------- | ---- | ---- | --- | ---- | ---- | ---- |
| all_reduce    | ✓    | ✓    | ✓   | ✓    | ✓    | ✓    |
| all_gather    | ✓    | ✓    | ✓   | ✓    | ✓    | ✓    |
| all_to_all    | ✓    | ✓    | ✓   | ✓    | ✓    | ✓    |
| gather        | ✗    | ✓    | ✓   | ✗    | ✗    | ✗    |
| broadcast     | ✓    | ✓    | ✓   | ✓    | ✓    | ✓    |
| gather_object | ✗    | ✓    | ✓   | ✗    | ✗    | ✗    |

### 3.3 错误处理与边界情况

#### 常见错误
1. **未初始化分布式环境**：函数会自动降级处理，不会报错
2. **张量形状不一致**：`broadcast`要求所有进程张量形状相同
3. **不可序列化对象**：对象通信时对象必须可pickle序列化
4. **设备不匹配**：通信张量应在相同类型设备上
5. **NCCL不支持的操作**：如`gather`在NCCL后端不可用
6. **张量大小不可被进程数整除**：`all_to_all`要求输入张量大小可被`world_size`整除

#### 降级行为
- 非分布式环境：所有函数自动降级为单进程版本
- 单进程调用：函数执行无操作或返回本地数据
- 设备不支持：自动转换为CPU通信

### 3.4 性能建议

1. **批量操作**：使用`all_reduce_params`代替多次`all_reduce`调用
2. **合并归约**：设置`coalesce=True`减少通信次数
3. **适当桶大小**：根据网络带宽调整`bucket_size_mb`
4. **设备选择**：GPU/NPU通信通常比CPU快
5. **避免小张量**：合并小张量减少通信开销
6. **选择合适的通信原语**：根据实际需求选择最适合的通信操作，如当需要每个进程都获取所有进程的数据块时，`all_to_all`可能比多次`send/recv`更高效

## 总结

ScaleTorch分布式计算模块为PyTorch分布式训练提供了完整、易用且高效的工具集。其主要优势包括：

1. **接口统一**：分布式与非分布式环境使用相同接口
2. **功能全面**：覆盖大多数分布式训练需求，包括新增的`all_to_all`操作
3. **设备广泛**：支持多种AI加速硬件
4. **易于集成**：与现有PyTorch代码无缝集成
5. **性能优化**：提供多种性能优化选项

该模块特别适合需要跨多GPU/多节点进行模型训练、评估和部署的深度学习应用场景。
