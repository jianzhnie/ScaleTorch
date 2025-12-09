# 数据并行 (Data Parallelism)

## 概述

数据并行是分布式训练中最常用的并行策略。每个 GPU 保存完整的模型副本，但处理不同的数据批次。梯度通过 AllReduce 在所有 GPU 上同步，确保所有模型副本参数保持一致。

### 核心思想

```
GPU-0:  模型副本 + 数据批次[0:256]
           |
GPU-1:  模型副本 + 数据批次[256:512]  ---> AllReduce 梯度 ---> 参数更新
           |
GPU-N:  模型副本 + 数据批次[N*256:(N+1)*256]
```

## 基本概念

### 数据分割

```python
# 全局批次大小：1024
# GPU 数量：4
# 每个 GPU 上的批次大小：256

batch_size = 256  # 每个 GPU 上的批次大小
world_size = 4    # GPU 总数
global_batch_size = batch_size * world_size  # 1024
```

### 梯度同步

```python
# 每个 GPU 计算梯度
loss = model(x)
loss.backward()

# AllReduce 同步梯度
dist.all_reduce(grad, op=dist.ReduceOp.SUM)

# 平均梯度（隐含）
grad /= world_size
```

## 数据并行实现

### DataParallelNaive

简单的数据并行实现，适合学习和测试。

```python
class DataParallelNaive(nn.Module):
    """
    基础数据并行实现。

    特性：
    - 前向传播后立即同步梯度
    - 简单易懂，适合理解原理
    - 性能不如优化版本
    """
```

**使用方法：**

```python
import torch.distributed as dist
from scaletorch.parallel.data_parallel import DataParallelNaive

# 初始化
dist.init_process_group(backend='nccl')

# 包装模型
model = MyModel()
model = DataParallelNaive(model)

# 训练循环
for batch in dataloader:
    output = model(batch)
    loss = compute_loss(output)
    loss.backward()  # 自动同步梯度
    optimizer.step()
```

### DataParallelBucket

优化的数据并行实现，使用梯度桶化提高效率。

```python
class DataParallelBucket(nn.Module):
    """
    基于梯度桶的优化数据并行。

    特性：
    - 参数分组成多个桶
    - 梯度就绪时立即同步
    - 通信与计算重叠
    - 更好的性能
    """
```

## 梯度桶化 (Gradient Bucketing)

### 概念

将参数分组成若干个"桶"，每个桶包含多个参数。当桶内所有参数的梯度就绪时，对这个桶进行一次 AllReduce，从而提高通信效率。

### Bucket 类

```python
class Bucket:
    """
    管理一组参数的梯度同步。

    属性：
        params: 分配给该桶的参数集合
        params_with_grad_ready: 梯度已就绪的参数
        grad_data: 存储梯度的张量
        handle: 异步 AllReduce 操作的句柄
    """
```

### 梯度同步流程

```python
# 1. 桶创建时初始化
bucket = Bucket(
    params=[param1, param2, param3],
    grad_data=grad_tensor,
    process_group=dp_group
)

# 2. 梯度就绪时注册
bucket.add_ready_param(param)

# 3. 当所有参数梯度就绪时
if bucket.is_ready():
    bucket.sync_gradient()  # 启动异步 AllReduce

# 4. 优化器更新前等待完成
bucket.wait()
```

### BucketManager 类

```python
class BucketManager:
    """
    管理所有梯度桶。

    负责：
    - 参数分组
    - 桶创建和管理
    - 梯度同步的协调
    """
```

## 性能优化技术

### 1. 梯度累积（Gradient Accumulation）

```python
from scaletorch.parallel.data_parallel import no_sync

accumulation_steps = 4

for step, batch in enumerate(dataloader):
    output = model(batch)
    loss = compute_loss(output) / accumulation_steps

    # 梯度累积：不同步
    with model.no_sync() if (step + 1) % accumulation_steps != 0 else nullcontext():
        loss.backward()

    # 每 N 步同步一次梯度
    if (step + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**优势：**
- 增加有效批次大小
- 减少同步频率
- 提高内存利用率

### 2. 异步 AllReduce

```python
# 启动异步 AllReduce
handle = dist.all_reduce(grad, async_op=True)

# 同时进行计算
next_loss = model(next_batch)

# 等待通信完成
handle.wait()
```

**优势：**
- 通信与计算重叠
- 减少总体训练时间

### 3. AllGather 优化

```python
# 使用 AllGather 替代多个 AllReduce
gathered_grads = dist.all_gather(grad_list, grad)

# 一次通信获取所有Rank的梯度
```

## 使用示例

### 示例 1: 基本数据并行

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from scaletorch.parallel.data_parallel import DataParallelBucket
from scaletorch.parallel.pg_manager import setup_process_group_manager

# 初始化分布式环境
dist.init_process_group(backend='nccl')

# 设置进程组（4 个秩进行 DP）
pgm = setup_process_group_manager(
    tp_size=1, cp_size=1, pp_size=1, dp_size=4
)

# 创建模型
model = MyModel()
model.to('cuda')

# 应用数据并行
model = DataParallelBucket(model)

# 数据加载器
train_loader = DataLoader(dataset, batch_size=256, sampler=DistributedSampler(dataset))

# 训练循环
optimizer = torch.optim.AdamW(model.parameters())
for epoch in range(num_epochs):
    for batch in train_loader:
        output = model(batch['input_ids'])
        loss = compute_loss(output, batch['labels'])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

### 示例 2: 梯度累积

```python
from scaletorch.parallel.data_parallel import no_sync
from contextlib import nullcontext

accumulation_steps = 4
global_step = 0

for batch in dataloader:
    output = model(batch)
    loss = compute_loss(output) / accumulation_steps

    # 决定是否同步梯度
    should_sync = (global_step + 1) % accumulation_steps == 0
    sync_context = nullcontext() if should_sync else model.no_sync()

    with sync_context:
        loss.backward()

    global_step += 1

    # 累积 N 步后更新
    if should_sync:
        optimizer.step()
        optimizer.zero_grad()
```

### 示例 3: 检查桶配置

```python
from scaletorch.parallel.data_parallel import DataParallelBucket

model = DataParallelBucket(model)

# 打印桶配置
for i, bucket in enumerate(model.bucket_manager.buckets):
    print(f"Bucket {i}:")
    print(f"  Parameters: {len(bucket.params)}")
    print(f"  Grad size: {bucket.grad_data.numel()} elements")
    print(f"  Grad shape: {bucket.grad_data.shape}")
```

### 示例 4: 分布式数据加载

```python
from torch.utils.data import DataLoader, DistributedSampler

# 创建采样器确保每个 GPU 获得不同的数据
sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True,
    seed=seed
)

# 创建数据加载器
train_loader = DataLoader(
    dataset,
    batch_size=256,
    sampler=sampler,
    num_workers=4
)

# 训练循环
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)  # 确保不同 epoch 数据打乱不同
    for batch in train_loader:
        # 训练代码
        pass
```

## 性能特性

### 通信开销

| 操作 | 通信量 | 说明 |
|------|--------|------|
| AllReduce | O(P) | P = 参数总数 |
| AllGather | O(P) | 单次获取所有秩数据 |
| Reduce-Scatter | O(P) | 分散完整张量 |

### 加速比

```
N GPU 的加速比 ≈ N / (1 + α)

其中 α = 通信时间 / 计算时间

- 通常 α ≈ 5-15%（对于大模型）
- 加速比可达 8-10x for 8 GPU
```

### 内存占用

```
每个 GPU 内存 = 模型参数 + 激活 + 优化器状态 + 梯度

对于 AdamW 优化器：
- 模型参数：P
- 优化器状态（动量、方差）：2P
- 总计：~3P + 激活缓存
```

## 最佳实践

1. **批次大小规划：**

   ```python
   # 全局批次大小应能被 GPU 数量整除
   global_batch_size = 1024
   num_gpus = 4
   batch_size_per_gpu = global_batch_size // num_gpus  # 256

   assert global_batch_size % num_gpus == 0
   ```

2. **梯度检查：**

   ```python
   # 确保所有 GPU 上梯度相同
   dist.all_reduce(grad)
   assert grad.std() < 1e-5, "Gradient mismatch across GPUs!"
   ```

3. **通信延迟隐藏：**

   ```python
   # 在一批数据反向传播时准备下一批数据
   for batch, next_batch in zip(dataloader, dataloader):
       with model.no_sync():
           loss = model(batch)
           loss.backward()

       # 准备下一批
       next_batch = prepare_batch(next_batch)

       # 等待梯度同步
       optimizer.step()
   ```

4. **内存优化：**

   ```python
   # 启用梯度检查点
   from torch.utils.checkpoint import checkpoint

   for layer in model.layers:
       # 以计算换内存
       output = checkpoint(layer, input)
   ```

## 常见问题

### Q1: 如何确保数据并行的正确性？

```python
# 在所有 GPU 上验证梯度
dist.all_reduce(grad, op=dist.ReduceOp.SUM)
expected_grad = grad / world_size

# 与单 GPU 的梯度比较
```

### Q2: 梯度不同步的原因？

常见原因：
- 模型在不同 GPU 上初始化不同
- 随机数生成器种子设置不同
- 数据预处理不一致

### Q3: 如何调试 AllReduce 通信问题？

```python
# 启用通信调试
import os
os.environ['NCCL_DEBUG'] = 'INFO'

# 检查通信是否挂起
dist.monitored_barrier(timeout=timedelta(seconds=30))

# 验证秩连接
print(f"Rank: {dist.get_rank()}, World size: {dist.get_world_size()}")
```

### Q4: 不同 GPU 上的损失值应该相同吗？

不一定。损失值在梯度同步前可能略有不同，这是正常的。关键是梯度应该相同。

## 参考资源

- [PyTorch DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
- [Horovod Data Parallelism](https://horovod.ai/)
- [Gradient Compression for Distributed Training](https://arxiv.org/abs/1902.00146)
