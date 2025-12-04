# 张量并行 (Tensor Parallelism)

## 概述

张量并行通过在多个 GPU 上分割模型的权重矩阵来实现。ScaleTorch 实现了列并行和行并行两种主要策略，可以高效地在大规模集群上训练超大模型。

### 核心概念

**列并行 (Column Parallel)：**
- 权重矩阵沿列方向分割
- 每个 GPU 负责计算输出的一个子集
- 无需跨 GPU 通信

**行并行 (Row Parallel)：**
- 权重矩阵沿行方向分割
- 需要在前向传播中进行 AllReduce

## 通信原语

### 1. CopyToModelParallelRegion

复制操作：前向传播复制，反向传播 AllReduce。

```python
class CopyToModelParallelRegion(torch.autograd.Function):
    """
    用于行并行线性层。

    前向：直接返回输入
    反向：对张量并行组进行 AllReduce
    """
```

**使用场景：** 行并行线性层的输入处理

### 2. ReduceFromModelParallelRegion

AllReduce 操作：前向传播 AllReduce，反向传播恒等变换。

```python
class ReduceFromModelParallelRegion(torch.autograd.Function):
    """
    用于列并行线性层。

    前向：对张量并行组进行 AllReduce
    反向：直接返回梯度
    """
```

**使用场景：** 列并行线性层的输出合并

### 3. GatherFromModelParallelRegion

Gather 操作：前向 Gather，反向 Split。

```python
class GatherFromModelParallelRegion(torch.autograd.Function):
    """
    从所有秩收集张量。

    前向：AllGather 所有秩的张量
    反向：将梯度分割到各秩
    """
```

**使用场景：** 需要完整张量的最后阶段

## 线性层实现

### ColumnParallelLinear

列并行线性层：Y = XW + b，其中 W 沿列分割。

```python
class ColumnParallelLinear(nn.Module):
    """
    列并行线性层实现。

    参数：
        in_features: 输入特征维度
        out_features: 输出特征维度
        bias: 是否使用偏置
        gather_output: 是否收集所有秩的输出
        async_all_reduce: 是否使用异步 AllReduce
    """
```

**特性：**
- 权重形状：`(out_features // tp_size, in_features)`
- 前向传播无跨 GPU 通信（gather_output=False 时）
- 适合于 MLP 的上升投影（Up Projection）

### RowParallelLinear

行并行线性层：Y = XW + b，其中 W 沿行分割。

```python
class RowParallelLinear(nn.Module):
    """
    行并行线性层实现。

    参数：
        in_features: 输入特征维度
        out_features: 输出特征维度
        bias: 是否使用偏置
        async_all_reduce: 是否使用异步 AllReduce
    """
```

**特性：**
- 权重形状：`(out_features, in_features // tp_size)`
- 前向传播需要 AllReduce 来合并输出
- 适合于 MLP 的下降投影（Down Projection）

### VocabParallelEmbedding

词汇并行嵌入层。

```python
class VocabParallelEmbedding(nn.Module):
    """
    词汇并行嵌入层。

    将词汇表分割到多个秩上，每个秩负责部分词汇。
    """
```

## 异步 AllReduce

### linear_with_async_all_reduce

异步 AllReduce 线性层，提高通信与计算的重叠。

```python
def linear_with_async_all_reduce(
    input_: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """
    使用异步 AllReduce 的线性层。

    优势：
    - 通信与计算重叠
    - 减少总体训练时间
    - 需要手动同步
    """
```

## 应用接口

### apply_tensor_parallel

自动将模型转换为张量并行版本。

```python
def apply_tensor_parallel(model: torch.nn.Module) -> torch.nn.Module:
    """
    将张量并行应用于 Transformer 模型。

    自动替换：
    - 注意力投影层 (Q, K, V, Out)
    - MLP 投影层 (Up, Gate, Down)
    - 嵌入层和最终投影层
    """
```

## 使用示例

### 示例 1: 基本张量并行

```python
import torch
import torch.distributed as dist
from scaletorch.parallel.pg_manager import setup_process_group_manager
from scaletorch.parallel.tensor_parallel import apply_tensor_parallel
from scaletorch.model.model_llama import LlamaModel

# 初始化分布式环境
dist.init_process_group(backend='nccl')

# 设置进程组（4 个秩进行 TP）
pgm = setup_process_group_manager(tp_size=4, cp_size=1, pp_size=1, dp_size=1)

# 创建模型
model = LlamaModel(config)
model.to('cuda')

# 应用张量并行
model = apply_tensor_parallel(model)

# 正常训练
optimizer = torch.optim.AdamW(model.parameters())
for batch in dataloader:
    output = model(batch['input_ids'])
    loss = compute_loss(output, batch['labels'])
    loss.backward()
    optimizer.step()
```

### 示例 2: 列并行线性层

```python
from scaletorch.parallel.tensor_parallel import ColumnParallelLinear

# 创建列并行层
# 输入：(batch, seq_len, hidden_dim)
# 输出：(batch, seq_len, hidden_dim*4)
mlp_up = ColumnParallelLinear(
    in_features=768,
    out_features=3072,
    bias=True,
    gather_output=False
)

# 前向传播
x = torch.randn(2, 1024, 768, device='cuda')
y = mlp_up(x)  # 形状：(2, 1024, 3072 // tp_size)
```

### 示例 3: 行并行线性层

```python
from scaletorch.parallel.tensor_parallel import RowParallelLinear

# 创建行并行层
mlp_down = RowParallelLinear(
    in_features=3072,
    out_features=768,
    bias=True
)

# 前向传播（需要 AllReduce）
x = torch.randn(2, 1024, 3072 // tp_size, device='cuda')
y = mlp_down(x)  # 形状：(2, 1024, 768)，经过 AllReduce
```

### 示例 4: 异步 AllReduce

```python
from scaletorch.parallel.tensor_parallel import ColumnParallelLinear

# 创建支持异步 AllReduce 的层
linear = ColumnParallelLinear(
    in_features=768,
    out_features=3072,
    gather_output=True,
    async_all_reduce=True
)

# 前向传播
x = torch.randn(2, 1024, 768, device='cuda')
y = linear(x)

# 需要手动同步
y.wait()  # 如果实现了异步等待
```

## 性能特性

### 计算复杂度

| 操作       | 复杂度                  | 说明                             |
| ---------- | ----------------------- | -------------------------------- |
| 列并行前向 | O(B·S·D²/P)             | B=batch, S=seq, D=dim, P=tp_size |
| 行并行前向 | O(B·S·D²/P + AllReduce) | 包含 AllReduce 开销              |
| 反向传播   | O(B·S·D²/P + AllReduce) | 列行互补                         |

### 通信开销

**列并行：**
- 前向：无通信（gather_output=False）
- 反向：AllReduce (D² 字节)

**行并行：**
- 前向：AllReduce (D² 字节)
- 反向：无通信

## 最佳实践

1. **层映射策略：**

   ```python
   # Q, K, V 投影：列并行
   q_proj = ColumnParallelLinear(...)

   # 注意力输出投影：行并行
   out_proj = RowParallelLinear(...)

   # MLP 上升：列并行
   up_proj = ColumnParallelLinear(...)

   # MLP 下降：行并行
   down_proj = RowParallelLinear(...)
   ```

2. **通信优化：**

   ```python
   # 使用异步 AllReduce 提高效率
   async_linear = ColumnParallelLinear(
       ...,
       async_all_reduce=True
   )
   ```

3. **内存优化：**

   ```python
   # 避免不必要的 gather
   linear = ColumnParallelLinear(
       ...,
       gather_output=False  # 只在需要完整张量时为 True
   )
   ```

## 常见问题

### Q1: 何时使用列并行 vs 行并行？

列并行适用于输出不需要在秩间同步的情况（中间层），行并行适用于需要完整输出或进行秩间操作的情况（跨秩汇聚）。

### Q2: 异步 AllReduce 何时有益？

异步 AllReduce 在通信与计算可以重叠时最有效。通常在以下情况使用：
- 大模型（通信时间长）
- 多层网络（计算充足）

### Q3: 如何调试张量形状错误？

```python
# 启用调试模式
pgm = setup_process_group_manager(tp_size=4)
print(f"TP rank: {pgm.tp_rank}, TP size: {pgm.tp_world_size}")

# 检查分割后的形状
print(f"Weight shape: {linear.weight.shape}")
print(f"Expected: ({out_features // tp_size}, {in_features})")
```

### Q4: 所有 GPU 上的权重都相同吗？

不同。每个 GPU 只存储权重矩阵的一部分：
- 列并行：每个 GPU 存储 `(out_features // tp_size, in_features)`
- 行并行：每个 GPU 存储 `(out_features, in_features // tp_size)`

## 参考文献

- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
- [PyTorch Distributed Documentation](https://pytorch.org/docs/stable/distributed.html)
