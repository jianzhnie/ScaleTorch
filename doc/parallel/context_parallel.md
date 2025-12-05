# 上下文并行 (Context Parallelism)

## 概述

上下文并行（Context Parallelism, CP）是一种在序列长度维度上进行并行化的策略。它特别适合处理超长序列的场景，例如长文档或视频处理任务。

## 核心模块

### 1. context_parallel.py - Ring Attention 实现

Ring Attention 是上下文并行的核心算法，实现了一种高效的分布式注意力计算机制。

#### 主要类和函数

##### apply_context_parallel()

应用上下文并行配置到模型。

```python
from scaletorch.parallel.context_parallel import apply_context_parallel

model = apply_context_parallel(model)
```

**功能：**
- 设置 `CONTEXT_PARALLEL` 环境变量
- 根据 CP 世界大小启用/禁用上下文并行
- 返回相同的模型对象（便于链式调用）

##### ring_attention()

计算 Ring Attention。

```python
output = ring_attention(q, k, v, sm_scale=1.0/sqrt(head_dim), is_causal=True)
```

**参数：**
- `q` (Tensor): Query 张量，形状 (batch_size, num_heads, seq_len, head_dim)
- `k` (Tensor): Key 张量，形状 (batch_size, num_heads, seq_len, head_dim)
- `v` (Tensor): Value 张量，形状 (batch_size, num_heads, seq_len, head_dim)
- `sm_scale` (float): Softmax 缩放因子，通常为 $1/\sqrt{d_k}$
- `is_causal` (bool): 是否应用因果掩码（用于自回归模型）

**返回值：**
- 注意力输出张量，形状 (batch_size, num_heads, seq_len, head_dim)

**异常：**
- `ValueError`: 输入张量形状或数据类型不兼容

##### RingAttentionFunc - 自定义自动求导函数

```python
class RingAttentionFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sm_scale, is_causal):
        # Ring attention 前向传播
        ...

    @staticmethod
    def backward(ctx, dout):
        # Ring attention 反向传播
        ...
```

**前向传播算法：**

1. 初始化通信器
2. 保存原始 K 和 V 用于反向传播
3. Ring 循环（每次迭代一个步骤）：
   - 如果不是最后一步，启动异步通信获取下一个 K 和 V
   - 计算当前步骤的注意力
   - 更新输出和 LSE（log-sum-exp）
   - 等待通信完成
4. 将输出转换为输入数据类型
5. 保存张量用于反向传播

**反向传播算法：**

1. 初始化两个通信器（一个用于 KV，一个用于梯度）
2. 预分配梯度缓冲区
3. Ring 循环进行反向梯度计算：
   - 计算每一步的梯度
   - 累积梯度
   - 管理异步通信

##### ring_attention_forward()

单步注意力前向计算。

```python
output, lse = ring_attention_forward(q, k, v, sm_scale=1.0, is_causal=False)
```

**实现细节：**
- 计算注意力分数：$S = Q \cdot K^T \cdot \text{scale}$
- 应用因果掩码（如需要）
- 使用在线 softmax 保证数值稳定性
- 返回注意力输出和 log-sum-exp

##### ring_attention_backward()

单步注意力反向计算。

```python
dq, dk, dv = ring_attention_backward(
    dout, q, k, v, output, softmax_lse,
    sm_scale=1.0, is_causal=False
)
```

**梯度计算步骤：**
1. 重建注意力概率
2. 计算对 V 的梯度
3. 计算对注意力概率的梯度
4. 计算对分数的梯度
5. 应用因果掩码
6. 计算对 Q 和 K 的梯度

##### update_out_and_lse()

使用数值稳定的方法更新输出和 LSE。

```python
out, lse = update_out_and_lse(current_out, current_lse, block_out, block_lse)
```

**数值稳定性技巧：**
- 使用 sigmoid 和 logsigmoid 进行稳定的指数更新
- 避免大数减法导致的精度丧失

##### update_rope_for_context_parallel()

为上下文并行更新 RoPE（旋转位置编码）。

```python
cos_cp, sin_cp = update_rope_for_context_parallel(cos, sin)
```

**功能：**
- 将 RoPE 张量分区给各个 CP 秩
- 每个秩获得序列的一个连续段
- 用于处理长序列时的位置编码

**参数：**
- `cos` (Tensor): RoPE 余弦分量
- `sin` (Tensor): RoPE 正弦分量

**返回值：**
- 当前秩的 RoPE 分区

### 2. cp_comms.py - 通信原语

#### ContextCommunicate 类

管理 Ring 拓扑中的点对点通信。

```python
from scaletorch.parallel.context_parallel.cp_comms import ContextCommunicate

comm = ContextCommunicate('ring_attention_forward')
```

**初始化参数：**
- `name` (str): 通信操作的名称（用于日志）

**主要方法：**

##### send_recv()

发送数据到下一个秩，从上一个秩接收数据。

```python
received = comm.send_recv(tensor_to_send)
```

**注意：** 这是异步操作，需要配合 `commit()` 和 `wait()` 使用。

##### commit()

提交所有待处理的 send_recv 操作。

```python
comm.send_recv(k)
comm.send_recv(v)
comm.commit()  # 开始实际传输
```

##### wait()

等待所有待处理的操作完成。

```python
comm.wait()  # 阻塞直到所有通信完成
```

## 设计理念

### Ring Attention 算法

Ring Attention 是一种内存高效的分布式注意力计算方法：

```
Step 0: 每个秩计算 Q[i] @ K[i] @ V[i]
Step 1: 每个秩接收来自前一个秩的 K 和 V，计算 Q[i] @ K[i-1] @ V[i-1]
...
Step N-1: 完整的注意力计算
```

**优势：**
- 内存占用低：只需保存一个 K 和 V 块
- 通信量适中：只有 K 和 V 沿 Ring 传递
- 支持梯度计算：反向传播同样高效

### 数值稳定性

所有计算都使用了数值稳定的方法：

1. **在线 Softmax**：避免中间溢出
2. **LogSumExp 技巧**：保留精度
3. **Stable Update**：使用 sigmoid 避免减法误差

## 使用示例

### 示例 1: 基本使用

```python
import torch
from scaletorch.parallel.context_parallel import ring_attention

# 设置参数
batch_size, num_heads, seq_len, head_dim = 2, 8, 1024, 64

# 创建输入张量
q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
k = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
v = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')

# 计算注意力
sm_scale = 1.0 / (head_dim ** 0.5)
output = ring_attention(q, k, v, sm_scale, is_causal=True)

print(f"Output shape: {output.shape}")
```

### 示例 2: 在模型中应用

```python
import torch.nn as nn
from scaletorch.parallel.context_parallel import apply_context_parallel

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.head_dim = hidden_dim // num_heads
        self.num_heads = num_heads

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # 转置到 (batch, heads, seq, dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 应用 Ring Attention
        output = ring_attention(q, k, v, 1.0 / (self.head_dim ** 0.5), is_causal=True)

        return output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)

# 应用上下文并行
model = AttentionLayer(hidden_dim=512, num_heads=8)
model = apply_context_parallel(model)
```

### 示例 3: 长序列处理

```python
import torch
from scaletorch.parallel.context_parallel.cp_comms import ContextCommunicate

# 处理长序列，每个 GPU 只保存一个分区
comm = ContextCommunicate('long_sequence_attention')

# 发送和接收序列分区
for step in range(world_size):
    # 异步通信
    next_kv = comm.send_recv(current_kv)
    comm.commit()

    # 计算注意力
    output_step = compute_attention(q, current_kv)

    # 等待下一个分区到达
    comm.wait()
    current_kv = next_kv
```

## 性能特征

### 内存占用

| 方案            | 内存占用                   |
| --------------- | -------------------------- |
| 标准 Attention  | O(N²) - 存储完整注意力矩阵 |
| Ring Attention  | O(N) - 只存储一个 KV 块    |
| Flash Attention | O(N) - 块级计算            |

### 通信成本

假设 CP 大小为 $p$，序列长度为 $N$：

| 操作           | 通信量                      | 步骤      |
| -------------- | --------------------------- | --------- |
| K/V 广播       | $p-1$ 次，每次大小 $O(N/p)$ | 前向 p 步 |
| 梯度 AllReduce | 1 次，大小 $O(N)$           | 反向      |

总通信：$O(N)$（相对于序列长度）

### 时间复杂度

- 前向：$O(p \cdot N/p \cdot N) = O(N^2)$ 计算 + $O(p)$ 通信步
- 反向：$O(N^2)$ 计算 + $O(p)$ 通信步

## 最佳实践

1. **序列分割：** 确保序列长度能被 CP 大小整除
2. **设备亲和性：** 在同一节点内部使用 CP 以减少网络往返
3. **梯度累积：** 适当设置梯度累积以隐藏通信延迟
4. **数据类型：** 使用 FP32 进行不稳定计算，必要时使用 FP64

## 常见问题

### Q1: Ring Attention 何时比标准 Attention 更优?
**A:** 当序列长度 N 很大时，内存节省最明显。一般 N > 1024 时有显著优势。

### Q2: CP 大小应该如何选择?
**A:** 考虑以下因素：
- GPU 间带宽（同一节点>跨节点）
- 序列长度和可整除性
- 计算与通信的重叠机会

### Q3: 能否与其他并行策略组合?
**A:** 可以，推荐的组合：
- CP + DP：长序列训练
- CP + TP：需要两个并行维度
- CP + DP + TP：完整混合并行

## 参考论文

- [Ring Attention with Blockwise Transformers for Context-Aware Generation](https://arxiv.org/abs/2310.01889)
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
