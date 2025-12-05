# 梯度桶化 (Gradient Bucketing)

## 概述

梯度桶化是一种优化分布式训练通信的技术。它将模型参数分组成若干个"桶"，当一个桶内所有参数的梯度就绪时，对这个桶进行一次 AllReduce，而不是为每个参数单独通信。这样可以显著减少通信开销和同步等待时间。

### 核心思想

```
单参数同步：            梯度桶化：
AllReduce(p1)      Bucket-1: [p1, p2, p3]
AllReduce(p2)                   AllReduce
AllReduce(p3)      Bucket-2: [p4, p5, p6]
...                            AllReduce
```

**优势：**
- 通信次数减少 N 倍（N = 参数数量）
- 梯度就绪时立即同步（不等待其他梯度）
- 通信与计算重叠机会增多
- 总体训练时间显著降低

## Bucket 类

```python
class Bucket:
    """
    管理一组参数的梯度同步。

    属性：
        params: 分配给该桶的参数集合
        params_with_grad_ready: 梯度已就绪的参数
        grad_data: 存储梯度的连续张量
        process_group: 通信组
        handle: 异步 AllReduce 的操作句柄
    """
```

### 初始化

```python
from scaletorch.parallel.data_parallel.bucket import Bucket
import torch.distributed as dist

# 创建桶
bucket = Bucket(
    params=[param1, param2, param3],  # 该桶包含的参数
    grad_data=grad_tensor,             # 存储梯度的缓冲区
    process_group=dp_group             # 通信组
)
```

### 核心方法

#### add_ready_param

```python
def add_ready_param(self, param: nn.Parameter) -> None:
    """
    标记参数梯度已就绪。

    当梯度全部就绪时，自动启动 AllReduce。
    """
    self.params_with_grad_ready.add(param)

    # 检查是否所有参数梯度都就绪
    if self.is_ready():
        self.sync_gradient()
```

#### sync_gradient

```python
def sync_gradient(self) -> None:
    """
    启动异步 AllReduce 操作。

    流程：
    1. 梯度平均（除以 world_size）
    2. 启动异步 AllReduce
    3. 返回处理句柄以供后续等待
    """
    # 梯度平均
    self.grad_data.div_(self.process_group_size)

    # 启动异步 AllReduce
    self.handle = dist.all_reduce(
        self.grad_data,
        group=self.process_group,
        async_op=True
    )
```

#### wait

```python
def wait(self) -> None:
    """
    等待 AllReduce 操作完成。

    优化器更新参数前必须调用此方法。
    """
    if self.handle is not None:
        self.handle.wait()
        self.reset()
```

#### reset

```python
def reset(self) -> None:
    """
    重置桶状态，为下一个训练步骤做准备。

    清空：
    - 就绪参数集合
    - 梯度数据
    - 操作句柄
    """
    self.handle = None
    self.params_with_grad_ready.clear()
    self.grad_data.zero_()
```

## BucketManager 类

```python
class BucketManager:
    """
    管理所有梯度桶的生命周期。

    职责：
    - 参数分组
    - 桶的创建和维护
    - 梯度同步的协调
    - 优化器更新的同步
    """
```

### 初始化与配置

```python
from scaletorch.parallel.data_parallel.bucket import BucketManager

# 创建桶管理器
manager = BucketManager(
    model=model,
    bucket_size_mb=25,      # 每个桶的大小（MB）
    process_group=dp_group
)

# 自动将参数分组成若干个桶
manager.build_buckets()
```

### 桶分组算法

```python
def build_buckets(self, bucket_size_bytes: int = 25 * 1024 * 1024) -> None:
    """
    贪心算法分组参数到桶中。

    算法：
    1. 遍历所有参数
    2. 如果加入参数后桶大小超过 bucket_size_bytes，创建新桶
    3. 否则，将参数加入当前桶

    目标：
    - 每个桶大小约为 bucket_size_bytes
    - 减少桶数量
    - 优化通信效率
    """
```

### 梯度就绪回调

```python
def register_gradient_ready_hook(self) -> None:
    """
    为每个参数注册梯度就绪钩子。

    当参数梯度计算完成时自动触发，
    通知对应的桶该参数梯度已就绪。
    """
    for param in self.model.parameters():
        if param.requires_grad:
            # 注册反向钩子
            param.register_hook(
                lambda grad, p=param: self._param_gradient_ready(p)
            )
```

### 通信协调

```python
def synchronize(self) -> None:
    """
    等待所有桶的 AllReduce 操作完成。

    优化器更新前调用此方法确保所有梯度已同步。
    """
    for bucket in self.buckets:
        bucket.wait()
```

## 工作流程

### 第一步：初始化

```python
import torch
import torch.distributed as dist
from scaletorch.parallel.data_parallel.bucket import BucketManager

# 初始化分布式
dist.init_process_group(backend='nccl')

# 创建模型
model = MyModel()
model.to('cuda')

# 创建桶管理器
bucket_manager = BucketManager(model, bucket_size_mb=25)
bucket_manager.build_buckets()
bucket_manager.register_gradient_ready_hook()

print(f"Created {len(bucket_manager.buckets)} buckets")
```

### 第二步：前向传播

```python
# 正常的前向传播
batch = next(dataloader)
output = model(batch['input_ids'])
loss = compute_loss(output, batch['labels'])
```

### 第三步：反向传播（自动触发梯度同步）

```python
# 反向传播
loss.backward()

# 钩子自动触发，梯度就绪时启动 AllReduce
# 通信与计算重叠进行
```

### 第四步：优化器更新（等待通信完成）

```python
# 等待所有梯度同步完成
bucket_manager.synchronize()

# 优化器更新
optimizer.step()
optimizer.zero_grad()
```

## 性能优化

### 1. 桶大小优化

```python
# 太小的桶
BucketManager(model, bucket_size_mb=5)  # 很多小桶，通信开销大

# 适中的桶
BucketManager(model, bucket_size_mb=25)  # 平衡通信和计算

# 太大的桶
BucketManager(model, bucket_size_mb=100)  # 少数大桶，等待时间长
```

**最佳实践：** 根据模型大小选择 20-50 MB 的桶大小

### 2. 梯度累积

```python
from torch.utils.checkpoint import checkpoint

accumulation_steps = 4

for step, batch in enumerate(dataloader):
    output = model(batch)
    loss = compute_loss(output) / accumulation_steps

    loss.backward()

    # 只在梯度准备好更新时同步
    if (step + 1) % accumulation_steps == 0:
        bucket_manager.synchronize()
        optimizer.step()
        optimizer.zero_grad()
```

### 3. 通信与计算重叠

```python
# 梯度就绪时立即启动 AllReduce
# 后续层的反向传播在 AllReduce 进行时继续计算
# 优化器更新前等待完成

# 时间轴示例：
# Layer-N backprop: |====|
# Layer-N AllReduce:        |=====|
# Layer-N-1 backprop:           |====|
```

## 配置示例

### 示例 1: 小模型（<1GB）

```python
# 单个大桶
BucketManager(model, bucket_size_mb=50)
```

### 示例 2: 中等模型（1-10GB）

```python
# 多个适中大小的桶
BucketManager(model, bucket_size_mb=25)  # 约 40-50 个桶
```

### 示例 3: 大模型（>10GB）

```python
# 较小的桶以增加通信与计算重叠机会
BucketManager(model, bucket_size_mb=10)  # 约 100+ 个桶
```

## 监控和调试

### 打印桶信息

```python
def print_bucket_stats(bucket_manager):
    """打印桶统计信息"""
    total_params = 0
    for i, bucket in enumerate(bucket_manager.buckets):
        param_count = sum(p.numel() for p in bucket.params)
        total_params += param_count
        grad_size_mb = bucket.grad_data.numel() * 4 / 1024 / 1024
        print(f"Bucket {i}: {param_count:>10} params, "
              f"{grad_size_mb:>6.2f} MB")
    print(f"Total: {total_params:>10} params")

print_bucket_stats(bucket_manager)
```

### 验证梯度同步

```python
def verify_gradient_sync(model, dp_group):
    """验证梯度是否正确同步"""
    for param in model.parameters():
        if param.grad is None:
            continue

        grad_copy = param.grad.clone()
        dist.all_reduce(grad_copy, op=dist.ReduceOp.SUM, group=dp_group)
        grad_copy /= dist.get_world_size(group=dp_group)

        assert torch.allclose(param.grad, grad_copy, rtol=1e-5, atol=1e-5), \
            f"Gradient mismatch for {param.name}"

verify_gradient_sync(model, dp_group)
```

### 性能分析

```python
import time
import torch.distributed as dist

class BucketTimer:
    def __init__(self, bucket_manager):
        self.manager = bucket_manager
        self.timings = {i: [] for i in range(len(bucket_manager.buckets))}

    def record_sync(self, bucket_id: int, elapsed: float):
        self.timings[bucket_id].append(elapsed)

    def report(self):
        for bucket_id, times in self.timings.items():
            if times:
                avg = sum(times) / len(times)
                print(f"Bucket {bucket_id}: avg {avg*1000:.2f} ms")

timer = BucketTimer(bucket_manager)
```

## 常见问题

### Q1: 为什么梯度同步前某些参数梯度为 None？

这是正常的。某些参数可能在该批次中未被使用，因此梯度为 None。桶管理器会自动处理这种情况。

### Q2: 能否自定义桶的参数分组？

可以。通过继承 `BucketManager` 并重写 `build_buckets` 方法：

```python
class CustomBucketManager(BucketManager):
    def build_buckets(self, bucket_size_bytes):
        # 自定义分组逻辑
        pass
```

### Q3: 使用梯度桶化后数值精度会降低吗？

不会。梯度桶化不改变数值精度，只是改变了同步的方式和时机。

### Q4: 与 PyTorch DDP 相比如何？

| 特性       | DDP  | 梯度桶化 |
| ---------- | ---- | -------- |
| 通信效率   | 中等 | 高       |
| 实现复杂度 | 低   | 中       |
| 自定义性   | 低   | 高       |
| 调试难度   | 低   | 中       |

## 参考资源

- [PyTorch Gradient Bucketing](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
- [DeepSpeed ZeRO Optimization](https://www.deepspeed.ai/tutorials/zero-offload/)
- [Efficient Gradient Communication](https://arxiv.org/abs/1802.06955)
