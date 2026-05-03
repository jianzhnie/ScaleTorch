# 梯度桶化 (Gradient Bucketing)

## 概述

梯度桶化将模型参数分组成若干个"桶"，当一个桶内所有参数的梯度就绪时，对这个桶进行一次 AllReduce，而不是为每个参数单独通信。显著减少通信开销和同步等待时间。

```
单参数同步：            梯度桶化：
AllReduce(p1)      Bucket-1: [p1, p2, p3]
AllReduce(p2)                   AllReduce
AllReduce(p3)      Bucket-2: [p4, p5, p6]
...                            AllReduce
```

## Bucket 类

管理一组参数的梯度同步，执行单次 AllReduce。

```python
from scaletorch.parallel.data_parallel.bucket import Bucket

bucket = Bucket(
    params=[param1, param2, param3],  # 该桶包含的参数
    grad_data=grad_tensor,             # 存储梯度的缓冲区
    process_group=dp_group             # 通信组
)
```

### 核心方法

#### mark_param_as_ready(param)

标记参数梯度已就绪。当桶内所有参数就绪时，自动启动异步 AllReduce。

```python
bucket.mark_param_as_ready(param)
# 如果所有参数都已就绪，自动调用 sync_gradient()
```

#### sync_gradient()

启动异步 AllReduce。梯度先除以 `process_group_size` 再同步。

```python
bucket.sync_gradient()  # 启动异步 AllReduce
```

#### wait()

等待 AllReduce 操作完成。优化器更新前必须调用。

```python
bucket.wait()
```

#### reset()

重置桶状态：清空就绪集合、归零梯度数据。

```python
bucket.reset()
```

#### is_synchronization_complete()

检查是否所有参数梯度都已就绪。

```python
if bucket.is_synchronization_complete():
    # 所有参数梯度就绪
    pass
```

## BucketManager 类

管理所有梯度桶的生命周期，负责参数分组、桶创建和梯度同步协调。

```python
from scaletorch.parallel.data_parallel.bucket import BucketManager

manager = BucketManager(
    params=list(model.parameters()),   # 模型参数列表
    process_group=pgm.dp_group,        # 通信组
    bucket_size=2**24,                 # 桶大小（元素数），默认 16M
    grad_type=None                     # 可选梯度数据类型
)
```

### 桶分组算法

参数按大小降序排列，使用贪心策略分组：

1. 按参数元素数降序排列
2. 尝试将参数放入最佳匹配的桶（best-fit）
3. 如果没有合适的桶，创建新桶
4. 每个桶大小不超过 `bucket_size`

### 核心方法

| 方法 | 说明 |
|------|------|
| `reset()` | 重置所有桶 |
| `wait()` | 等待所有桶的 AllReduce 完成 |
| `is_synchronization_complete()` | 检查所有桶是否同步完成 |
| `mark_param_as_ready(param)` | 标记参数就绪，触发对应桶检查 |
| `get_bucket_info()` | 返回桶配置信息字符串 |

### 关键属性

| 属性 | 说明 |
|------|------|
| `buckets` | `Bucket` 实例列表 |
| `params_to_bucket_location` | 参数到 `(start_idx, end_idx, bucket_idx)` 的映射 |
| `grad_data_list` | 每个桶的梯度张量列表 |

## 工作流程

### 1. 初始化（在 DataParallelBucket 构造时自动完成）

```python
# DataParallelBucket 内部自动创建 BucketManager
model = DataParallelBucket(model, bucket_size=2**24)
# BucketManager 将参数分组为桶
# 每个参数的 main_grad 是桶 grad_data 的视图
```

### 2. 前向传播

```python
output = model(batch['input_ids'])
loss = compute_loss(output, batch['labels'])
```

### 3. 反向传播（自动触发梯度同步）

```python
loss.backward()
# 每个参数梯度就绪时，hook 自动调用 mark_param_as_ready
# 桶内所有参数就绪时，自动启动异步 AllReduce
```

### 4. 优化器更新

```python
# DataParallelBucket._post_backward 自动等待所有桶完成
# 并将同步后的梯度复制回 param.grad
optimizer.step()
model.reset()  # 重置桶和梯度
optimizer.zero_grad()
```

## 性能优化

### 桶大小选择

```python
# 小模型 (<1GB)：大桶
bucket_size = 2**25  # 32M 元素

# 中等模型 (1-10GB)：适中桶
bucket_size = 2**24  # 16M 元素

# 大模型 (>10GB)：较小桶，增加通信与计算重叠
bucket_size = 2**23  # 8M 元素
```

### 通信与计算重叠

```
时间轴：
Layer-N 反向:   |====|
Layer-N AllReduce:    |=====|
Layer-N-1 反向:        |====|
                     通信与计算重叠
```

梯度就绪时立即启动 AllReduce，后续层的反向传播在 AllReduce 进行时继续计算。

## 监控和调试

### 打印桶信息

```python
model = DataParallelBucket(model)
print(model.get_bucket_info())
```

### 验证梯度同步

```python
for param in model.parameters():
    if param.grad is not None:
        grad_copy = param.grad.clone()
        dist.all_reduce(grad_copy, op=dist.ReduceOp.SUM, group=pgm.dp_group)
        grad_copy /= dist.get_world_size(group=pgm.dp_group)
        assert torch.allclose(param.grad, grad_copy, rtol=1e-5)
```

## 常见问题

### Q1: 为什么梯度同步前某些参数梯度为 None？

正常现象。某些参数在该批次中未被使用。桶管理器只处理有梯度的参数。

### Q2: 使用梯度桶化后数值精度会降低吗？

不会。桶化只改变同步的方式和时机，不影响数值精度。

## 参考资源

- [PyTorch DDP Gradient Bucketing](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
- [数据并行详细文档](./data_parallel.md)
