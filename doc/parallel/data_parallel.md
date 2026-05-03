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

## 类层次

```
DataParallelBase (nn.Module)
├── DataParallelNaive    # 简单实现，每参数即时 AllReduce
└── DataParallelBucket   # 生产级实现，梯度分桶优化
```

## DataParallelBase

所有数据并行实现的基类。

```python
from scaletorch.parallel.data_parallel.data_parallel import DataParallelBase

class DataParallelBase(nn.Module):
    def __init__(self, module: nn.Module):
        self.module = module
        self.require_backward_grad_sync: bool  # 控制是否同步梯度

    def forward(self, *inputs, **kwargs) -> Any  # 委托给 self.module

    def no_sync(self) -> contextmanager  # 临时禁用梯度同步
```

## DataParallelNaive

简单数据并行实现，每个参数梯度就绪时立即 AllReduce。适合学习原理和调试。

```python
from scaletorch.parallel.data_parallel.data_parallel import DataParallelNaive

model = DataParallelNaive(model)
```

**工作原理：**
- 通过 `register_post_accumulate_grad_hook` 为每个参数注册回调
- 梯度就绪时在 `cp_dp_group` 上执行 `dist.all_reduce(SUM)`
- 结果除以 `cp_dp_world_size` 得到平均梯度

## DataParallelBucket

生产级数据并行，使用梯度桶化提高效率。默认桶大小 16M 元素。

```python
from scaletorch.parallel.data_parallel.data_parallel import DataParallelBucket

model = DataParallelBucket(
    module=model,
    grad_type=None,       # 可选指定梯度数据类型
    bucket_size=2**24     # 桶大小（元素数），默认 16M
)
```

**关键方法：**

| 方法 | 说明 |
|------|------|
| `forward(*inputs, **kwargs)` | 委托给内部模块 |
| `no_sync()` | 上下文管理器，禁用梯度同步 |
| `backward(input_tensor, output_tensor, output_tensor_grad)` | 反向传播委托 |
| `reset()` | 重置桶管理器，清空所有梯度 |
| `get_bucket_info()` | 返回桶配置信息字符串 |

**关键属性：**

| 属性 | 说明 |
|------|------|
| `module` | 被包装的模型 |
| `grad_type` | 梯度数据类型 |
| `bucket_manager` | `BucketManager` 实例 |

## 使用示例

### 基本数据并行

```python
import torch.distributed as dist
from scaletorch.parallel.data_parallel.data_parallel import DataParallelBucket
from scaletorch.parallel.pg_manager import setup_process_group_manager

dist.init_process_group(backend='nccl')
pgm = setup_process_group_manager(tp_size=1, cp_size=1, pp_size=1, dp_size=4)

model = MyModel().cuda()
model = DataParallelBucket(model)

optimizer = torch.optim.AdamW(model.parameters())
for batch in train_loader:
    output = model(batch['input_ids'])
    loss = compute_loss(output, batch['labels'])
    loss.backward()
    optimizer.step()
    model.reset()
    optimizer.zero_grad()
```

### 梯度累积

```python
from contextlib import nullcontext

accumulation_steps = 4

for step, batch in enumerate(dataloader):
    output = model(batch)
    loss = compute_loss(output) / accumulation_steps

    # 仅在累积完成时同步梯度
    should_sync = (step + 1) % accumulation_steps == 0
    sync_ctx = nullcontext() if should_sync else model.no_sync()

    with sync_ctx:
        loss.backward()

    if should_sync:
        optimizer.step()
        model.reset()
        optimizer.zero_grad()
```

### 检查桶配置

```python
model = DataParallelBucket(model)
print(model.get_bucket_info())
```

## 性能特性

### 通信开销

| 操作 | 通信量 | 说明 |
|------|--------|------|
| AllReduce | O(P) | P = 参数总数 |
| AllGather | O(P) | 单次获取所有秩数据 |
| Reduce-Scatter | O(P) | 分散完整张量 |

### 内存占用

```
每个 GPU 内存 = 模型参数 + 激活 + 优化器状态 + 梯度

对于 AdamW 优化器：
- 模型参数：P
- 优化器状态（动量、方差）：2P
- 总计：~3P + 激活缓存
```

## 常见问题

### Q1: 梯度不同步的原因？

常见原因：
- 模型在不同 GPU 上初始化不同
- 随机数生成器种子设置不同
- 数据预处理不一致

### Q2: 如何调试 AllReduce 通信问题？

```python
import os
os.environ['NCCL_DEBUG'] = 'INFO'

# 验证秩连接
from scaletorch.dist import get_rank, get_world_size
print(f"Rank: {get_rank()}, World size: {get_world_size()}")
```

### Q3: 不同 GPU 上的损失值应该相同吗？

不一定。损失值在梯度同步前可能略有不同。关键是梯度应该相同。

## 参考资源

- [PyTorch DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
- [梯度桶化详细文档](./gradient_bucket.md)
