# 张量并行 (Tensor Parallelism)

## 概述

张量并行通过在多个 GPU 上分割模型的权重矩阵来实现。ScaleTorch 实现了列并行和行并行两种主要策略，可以高效地在大规模集群上训练超大模型。该方案基于 [Megatron-LM](https://arxiv.org/abs/1909.08053) 论文的思想，通过合理的通信设计实现计算与通信的平衡。

### 核心概念

**列并行 (Column Parallel)：**

- 权重矩阵 $W$ 沿列方向分割：$W = [W_1, W_2, \ldots, W_p]$（水平拼接）
- 每个 GPU $i$ 存储 $W_i \in \mathbb{R}^{D_{out}/p \times D_{in}}$
- 前向传播无需通信，反向传播需要 AllReduce 同步梯度

**行并行 (Row Parallel)：**

- 权重矩阵 $W$ 沿行方向分割：$W = [W_1; W_2; \ldots; W_p]$（竖直拼接）
- 每个 GPU $i$ 存储 $W_i \in \mathbb{R}^{D_{out} \times D_{in}/p}$
- 前向传播需要 AllReduce，反向传播无需通信

### 为什么需要张量并行

单纯的数据并行（Data Parallelism）存在以下局限：

1. **内存限制**：即使使用梯度检查点（Gradient Checkpointing），超大模型仍然无法在单 GPU 上放下。
2. **通信开销**：DP 的梯度同步通信量随批大小增加而增加。
3. **通信延迟**：DP 需要等待所有 rank 同步，容易成为训练瓶颈。

张量并行优势：

- 将模型权重分散存储，减少单 GPU 内存压力
- 通过合理的通信设计（如列并行前向无通信），降低通信开销
- 可与 DP、PP、CP 等其他并行策略组合，形成多维并行

## 数学原理与通信机制

### 列并行线性层 (Column Parallel Linear)

#### 前向传播

给定输入 $X \in \mathbb{R}^{B \times S \times D_{in}}$，权重分割为 $W = [W_1^T, W_2^T, \ldots, W_p^T]^T$，其中 $W_i^T \in \mathbb{R}^{D_{out}/p \times D_{in}}$。

GPU $i$ 上的计算：

$$Y_i = X W_i^T + b_i$$

其中 $Y_i \in \mathbb{R}^{B \times S \times D_{out}/p}$，$b_i \in \mathbb{R}^{D_{out}/p}$。

**关键特性**：前向传播无需任何跨 GPU 通信，计算完全独立。

#### 反向传播

**输入梯度**：
$$\frac{\partial L}{\partial X} = \sum_{i=1}^{p} \frac{\partial L}{\partial Y_i} \cdot W_i$$

每个 GPU $i$ 计算 $\frac{\partial L}{\partial Y_i} \cdot W_i$，随后需要对所有 GPU 的结果进行 AllReduce：

$$\frac{\partial L}{\partial X}^{final} = \text{AllReduce}\left(\sum_{i=1}^{p} \frac{\partial L}{\partial Y_i} \cdot W_i\right)$$

**权重梯度**：
$$\frac{\partial L}{\partial W_i^T} = \frac{\partial L}{\partial Y_i}^T \cdot X$$

**计算复杂度**（反向）：

```
Input:  grad_output (B, S, out_size)
Compute: grad_input = grad_output @ weight        O(B*S*out_size*in_size / p)
Comms:   AllReduce(grad_input)                    O(B*S*in_size) 网络 I/O
Output: grad_input (B, S, in_size)
```

### 行并行线性层 (Row Parallel Linear)

#### 前向传播

权重分割为 $W = [W_1; W_2; \ldots; W_p]$（竖直拼接），其中 $W_i \in \mathbb{R}^{D_{out} \times D_{in}/p}$。

输入也分割为 $X = [X_1, X_2, \ldots, X_p]$（水平拼接）。

GPU $i$ 上的计算：

$$Y_i = X_i W_i^T$$

其中 $Y_i \in \mathbb{R}^{B \times S \times D_{out}}$。

**合并**：

$$Y = \text{AllReduce}\left(\sum_{i=1}^{p} Y_i\right)$$

**关键特性**：前向传播需要 AllReduce 来同步每个 GPU 的部分输出。

#### 反向传播

**输入梯度**：
$$\frac{\partial L}{\partial X_i} = \frac{\partial L}{\partial Y} \cdot W_i$$

计算后无需通信，每个 GPU 得到其对应的梯度。

**权重梯度**：
$$\frac{\partial L}{\partial W_i^T} = \frac{\partial L}{\partial Y}^T \cdot X_i$$

**计算复杂度**（前向）：

```
Input:  x (B, S, in_size/p)
Compute: output = x @ weight              O(B*S*out_size*in_size / p)
Comms:   AllReduce(output)                O(B*S*out_size) 网络 I/O
Output: output (B, S, out_size)
```

### 列并行 vs 行并行对比

| 维度         | 列并行                                      | 行并行                                      |
| ------------ | ------------------------------------------- | ------------------------------------------- |
| **权重分割** | 沿输出维度                                  | 沿输入维度                                  |
| **前向通信** | ✗ 无                                        | ✓ AllReduce                                 |
| **反向通信** | ✓ AllReduce                                 | ✗ 无                                        |
| **前向计算** | $O(B \cdot S \cdot D_{out}/p \cdot D_{in})$ | $O(B \cdot S \cdot D_{out} \cdot D_{in}/p)$ |
| **常见应用** | MLP Up Projection                           | MLP Down Projection、注意力输出             |

### 通信原语设计

#### 设计原则

ScaleTorch 采用三层设计来处理张量并行通信：

1. **低层原语**（`CopyToModelParallelRegion`, `ReduceFromModelParallelRegion`, `GatherFromModelParallelRegion`）
   - 通过 `torch.autograd.Function` 实现，精确控制前向/反向通信
   - 支持梯度累积和自定义反向传播

2. **中层接口**（`linear_with_all_reduce`, `linear_with_async_all_reduce`）
   - 对常见线性层操作的封装
   - 支持同步和异步 AllReduce

3. **高层应用**（`ColumnParallelLinear`, `RowParallelLinear`, `VocabParallelEmbedding`）
   - PyTorch Module 形式，易于集成到现有模型
   - 自动处理权重初始化和形状管理



### 1. CopyToModelParallelRegion（前向复制，反向AllReduce）

#### 原理

实现 Megatron-LM 论文中的函数 $f$：

- **前向**：恒等变换，直接返回输入
- **反向**：对梯度进行 AllReduce，同步所有 GPU 的梯度

```
前向：Y = f(X) = X
反向：dX = AllReduce(dY)
```

#### 算法伪代码

```
Algorithm CopyToModelParallelRegion (Row Parallel Linear Backward)
┌─────────────────────────────────────────────────────────────┐
│ 前向传播                                                     │
│   Input:  X ∈ ℝ^(B,S,D_in/p)                               │
│   Output: Y = X （无修改）                                    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 反向传播                                                      │
│   Input:  dY ∈ ℝ^(B,S,D_in/p)                              │
│   Process:                                                    │
│     1. 对所有 tensor parallel rank 进行 AllReduce             │
│        dX_global = AllReduce(dY, op=SUM, group=tp_group)    │
│     2. 返回同步后的梯度                                        │
│   Output: dX ∈ ℝ^(B,S,D_in/p)                              │
└─────────────────────────────────────────────────────────────┘
```

#### 使用场景

**行并行线性层的反向梯度同步**：

在行并行线性层中，输入梯度需要从所有 GPU 汇聚。

#### 代码实现

```python
class CopyToModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x  # 恒等变换

    @staticmethod
    def backward(ctx, grad_output):
        dist.all_reduce(grad_output, op=dist.ReduceOp.SUM,
                       group=pgm.tp_group)
        return grad_output
```

### 2. ReduceFromModelParallelRegion（前向AllReduce，反向复制）

#### 原理

实现 Megatron-LM 论文中的函数 $g$：

- **前向**：对输入进行 AllReduce
- **反向**：恒等变换，直接返回梯度

```
前向：Y = g(X) = AllReduce(X)
反向：dX = dY （无修改）
```

#### 算法伪代码

```
Algorithm ReduceFromModelParallelRegion (Column Parallel Forward)
┌─────────────────────────────────────────────────────────────┐
│ 前向传播                                                      │
│   Input:  X_i ∈ ℝ^(B,S,D_out/p)  (GPU i 上的部分输出)      │
│   Process:                                                    │
│     1. AllReduce 合并所有 GPU 的输出                         │
│        Y = AllReduce(X_i, op=SUM, group=tp_group)          │
│   Output: Y ∈ ℝ^(B,S,D_out/p)                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 反向传播                                                      │
│   Input:  dY ∈ ℝ^(B,S,D_out/p)                             │
│   Output: dX = dY （无修改）                                  │
└─────────────────────────────────────────────────────────────┘
```

#### 使用场景

**列并行线性层的输出合并**

#### 代码实现

```python
class ReduceFromModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        dist.all_reduce(x, op=dist.ReduceOp.SUM, group=pgm.tp_group)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output  # 恒等变换
```

### 3. GatherFromModelParallelRegion（前向Gather，反向Split）

#### 原理

- **前向**：从所有 GPU 收集张量，沿最后一维拼接
- **反向**：将梯度按 tensor parallel size 分割

```
前向：Y = Gather([X_0, X_1, ..., X_{p-1}]) 沿 last_dim
反向：dX_i = Split(dY)[i]  沿 last_dim
```

#### 算法伪代码

```
Algorithm GatherFromModelParallelRegion (Gather Outputs for Final Layer)
┌──────────────────────────────────────────────────────────────────┐
│ 前向传播                                                           │
│   Input:  X_i ∈ ℝ^(B,S,D)  (GPU i 的部分张量)                       │
│   Process:                                                         │
│     1. 在每个 GPU 上创建容器: tensor_list = [[], [], ...]       │
│     2. AllGather: 将所有 GPU 的 X_i 收集到每个 GPU 的列表中      │
│        dist.all_gather(tensor_list, X_i, group=tp_group)       │
│     3. 沿最后维度拼接: Y = cat([X_0, X_1, ..., X_{p-1}])       │
│   Output: Y ∈ ℝ^(B,S,p*D)                                       │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ 反向传播                                                           │
│   Input:  dY ∈ ℝ^(B,S,p*D)  (合并后的梯度)                      │
│   Process:                                                         │
│     1. Split: 沿最后维度分割为 p 个块                             │
│        chunks = Split(dY, axis=last_dim, num_chunks=p)         │
│     2. 每个 GPU 保留对应的梯度块                                  │
│        dX_i = chunks[i]                                          │
│   Output: dX_i ∈ ℝ^(B,S,D)  (GPU i 的梯度分片)                 │
└──────────────────────────────────────────────────────────────────┘
```

#### 使用场景

**需要完整张量的最后阶段**（如语言模型的最终投影层）

#### 代码实现

```python
class GatherFromModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.contiguous()
        tensor_list = [torch.empty_like(x) for _ in range(pgm.tp_world_size)]
        dist.all_gather(tensor_list, x, group=pgm.tp_group)
        output = torch.cat(tensor_list, dim=-1).contiguous()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        chunks = torch.split(grad_output, grad_output.size(-1) // pgm.tp_world_size,
                           dim=-1)
        return chunks[pgm.tp_rank].contiguous()
```

### 4. 异步AllReduce优化

#### 原理

在反向传播中，计算梯度和进行通信存在依赖关系。通过异步操作和计算图重排，可以实现"计算-通信重叠"。

#### 同步 vs 异步对比

```
┌─────────────────────────────────────────────────────────┐
│ 同步方式（标准反向传播）                                   │
│                                                          │
│ GPU i:  [ grad_input计算 ] → [ AllReduce等待 ] → [ 其他梯度计算 ]
│         ├─ O(B*S*D)      ├─ O(通信) ────────┤    └─ O(B*S*D)
│         └─ 计算时间       └─ 无法隐藏通信 ────┘
└─────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│ 异步方式（计算-通信重叠）                                  │
│                                                           │
│ GPU i: [ grad_input计算 & 启动AllReduce ] ↓
│        ├─ O(B*S*D)                        │
│        └─ 非阻塞异步操作                  │
│          [ grad_weight/bias计算 ← 并行执行 ]
│          ├─ O(B*S*D)                      │
│          └─ 利用计算隐藏通信时间           │
│          [ 等待AllReduce完成 ]
│          └─ O(通信)
└──────────────────────────────────────────────────────────┘
```

#### 算法伪代码

```
Algorithm LinearWithAsyncAllReduce.backward()
┌────────────────────────────────────────────────────────────┐
│ 输入: grad_output ∈ ℝ^(B,S,D_out)                         │
│      input ∈ ℝ^(B,S,D_in)                                 │
│      weight ∈ ℝ^(D_out, D_in)                             │
└────────────────────────────────────────────────────────────┘

1. 计算输入梯度（立即开始AllReduce）
   grad_input = grad_output @ weight  ▸ 形状 (B,S,D_in)

2. 启动异步AllReduce（非阻塞）
   async_handle = AllReduce(grad_input, async_op=True)
   │
   ├─ 通信在后台进行
   │
3. 同时计算权重梯度（计算-通信重叠）
   grad_output_flat = flatten(grad_output)  ▸ (B*S, D_out)
   input_flat = flatten(input)              ▸ (B*S, D_in)
   grad_weight = grad_output_flat.T @ input_flat

4. 计算偏置梯度
   grad_bias = sum(grad_output_flat, dim=0)

5. 等待AllReduce完成
   async_handle.wait()
   │
   └─ grad_input 现已同步

返回: (grad_input, grad_weight, grad_bias)
```

#### 性能收益估计

假设计算时间为 $T_c$，通信时间为 $T_{comm}$：

- **同步方式**：总时间 = $T_c + T_{comm}$ （串行）
- **异步方式**：总时间 ≈ $\max(T_c, T_{comm})$ （并行）

在大模型训练中，$T_c \approx T_{comm}$，异步方式可以获得接近 **2x 的反向速度提升**。



## 线性层实现细节

### 列并行线性层执行流程

#### 前向传播（无通信）

```text
Input: X ∈ ℝ^(B,S,D_in)
       W_i ∈ ℝ^(D_out/p, D_in)  (GPU i 的权重分片)

GPU 0:           GPU 1:           GPU 2:           GPU 3:
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│X @ W_0^T │    │X @ W_1^T │    │X @ W_2^T │    │X @ W_3^T │
│(B,S,D/4) │    │(B,S,D/4) │    │(B,S,D/4) │    │(B,S,D/4) │
└─────┬────┘    └─────┬────┘    └─────┬────┘    └─────┬────┘
      │ 无通信         │               │               │
      └────────────────┼───────────────┼───────────────┘
      可选: Gather 合并 (gather_output=True)
```

#### 反向传播（需要AllReduce）

```text
dL/dY_i ∈ ℝ^(B,S,D_out/p)  (每个GPU)

GPU 0:              GPU 1:              GPU 2:              GPU 3:
计算 dL/dX_0      计算 dL/dX_1      计算 dL/dX_2      计算 dL/dX_3
   │                 │                 │                 │
   └─────────────────┼─────────────────┼─────────────────┘
         AllReduce: SUM(dL/dX_i)
                      ↓
            dL/dX_final ∈ ℝ^(B,S,D)
```

### 行并行线性层执行流程

#### 前向传播（需要AllReduce）

```text
Input: X_i ∈ ℝ^(B,S,D_in/p)  (各GPU的输入分片)
       W ∈ ℝ^(D_out, D_in/p)

GPU 0:           GPU 1:           GPU 2:           GPU 3:
计算 X_0@W_0^T   计算 X_1@W_1^T   计算 X_2@W_2^T   计算 X_3@W_3^T
   │                │                │                │
   └────────────────┼────────────────┼────────────────┘
         AllReduce: SUM(X_i @ W_i^T)
                      ↓
            Y ∈ ℝ^(B,S,D_out) 在所有GPU
```

#### 反向传播（无通信）

```text
dL/dY ∈ ℝ^(B,S,D_out)  (所有GPU相同)

GPU 0:           GPU 1:           GPU 2:           GPU 3:
计算 dL/dX_0     计算 dL/dX_1     计算 dL/dX_2     计算 dL/dX_3
│ (B,S,D/4)      │ (B,S,D/4)      │ (B,S,D/4)      │ (B,S,D/4)
└─────────────────┴─────────────────┴─────────────────┘
         无通信，各自返回本地梯度
```

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

| 操作 | 复杂度 | 说明 |
| --- | --- | --- |
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
