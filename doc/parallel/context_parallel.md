# 上下文并行 (Context Parallelism, CP) 完整指南

## 目录

- [1. 概述](#1-概述)
- [2. 核心概念](#2-核心概念)
- [3. 核心模块与 API](#3-核心模块与-api)
- [4. Ring Attention 算法详解](#4-ring-attention-算法详解)
- [5. 通信机制与优化](#5-通信机制与优化)
- [6. 性能分析](#6-性能分析)
- [7. 最佳实践](#7-最佳实践)
- [8. 常见问题](#8-常见问题)


## 1. 概述

### 1.1 什么是上下文并行

上下文并行（Context Parallelism, CP）是一种针对**超长序列任务**设计的分布式并行策略，特别适合处理长文档、视频等应用场景。

**核心思想**：
- 将序列长度维度切分到多卡
- 每卡只需处理部分序列
- 极大降低单卡内存压力
- 通过高效的 Ring 通信机制实现全局注意力计算

### 1.2 与其他并行方法的对比

| 特性       | DP    | TP     | PP    | CP       |
| ---------- | ----- | ------ | ----- | -------- |
| 并行维度   | batch | hidden | layer | sequence |
| 通信量     | 高    | 高     | 中    | 中       |
| 长序列     | 差    | 差     | 差    | **优**   |
| 实现复杂度 | 低    | 高     | 高    | 中       |


## 2. 核心概念

### 2.1 Ring Attention 算法原理

Ring Attention 是上下文并行的核心算法，通过环形（Ring）拓扑实现高效的分布式注意力计算。

**基本思想**：
- 将 K/V 张量在各进程间顺序传递
- 每个进程只需存储本地分区及当前步收到的 K/V
- 通过多步迭代实现完整的注意力计算

#### 算法流程

```bash
假设有 4 个 GPU，每个 GPU 存储一个序列分区：
- GPU-0: [K₀, V₀]
- GPU-1: [K₁, V₁]
- GPU-2: [K₂, V₂]
- GPU-3: [K₃, V₃]

Step 0: 每个 GPU 计算 Attn(Q_i, K_i, V_i)
Step 1: GPU-0 接收 [K₃, V₃] 并计算 Attn(Q_0, K_3, V_3)
Step 2: GPU-0 接收 [K₂, V₂] 并计算 Attn(Q_0, K_2, V_2)
Step 3: GPU-0 接收 [K₁, V₁] 并计算 Attn(Q_0, K_1, V_1)
最终：完整的 Attention(Q_0, [K_0; K_1; K_2; K_3], [V_0; V_1; V_2; V_3])
```

### 2.2 序列分区机制

```python
# 序列长度 N，CP 大小 P
# 每个 GPU 分得的序列长度：N / P

GPU-0: Q[0:N/P],     K[0:N/P],     V[0:N/P]
GPU-1: Q[N/P:2N/P],  K[N/P:2N/P],  V[N/P:2N/P]
GPU-2: Q[2N/P:3N/P], K[2N/P:3N/P], V[2N/P:3N/P]
...
```

### 2.3 关键优势

1. **内存高效**
   - 标准注意力：$O(N^2)$ 内存存储完整注意力矩阵
   - Ring Attention：$O(N)$ 只存储一个 KV 块

2. **通信优化**
   - 通信量：$O(N)$ 相对于序列长度
   - 支持异步通信隐藏通信延迟

3. **数值稳定**
   - 在线 Softmax：避免中间溢出
   - LogSumExp 技巧：保留精度
   - Stable Update：使用 sigmoid 避免减法误差

## 3. 核心模块与 API

### 3.1 Ring Attention 实现 (context_parallel.py)

#### 3.1.1 主要函数

##### apply_context_parallel(model)

配置模型以支持上下文并行。

```python
from scaletorch.parallel.context_parallel import apply_context_parallel

model = apply_context_parallel(model)
```

**功能**：
- 设置 `CONTEXT_PARALLEL` 环境变量
- 根据 CP 世界大小启用/禁用上下文并行
- 返回相同的模型对象（便于链式调用）

**参数**：
- `model` (nn.Module): 要配置的模型

**返回值**：
- 配置后的模型对象

##### ring_attention(q, k, v, sm_scale, is_causal)

执行 Ring Attention 前向计算。

```python
output = ring_attention(
    q, k, v,
    sm_scale=1.0 / (head_dim ** 0.5),
    is_causal=True
)
```

**参数**：
- `q` (Tensor): Query 张量，形状 (batch_size, num_heads, seq_len, head_dim)
- `k` (Tensor): Key 张量，形状 (batch_size, num_heads, seq_len, head_dim)
- `v` (Tensor): Value 张量，形状 (batch_size, num_heads, seq_len, head_dim)
- `sm_scale` (float): Softmax 缩放因子，通常为 $1/\sqrt{d_k}$
- `is_causal` (bool): 是否应用因果掩码（用于自回归模型）

**返回值**：
- 注意力输出张量，形状 (batch_size, num_heads, seq_len, head_dim)

**异常**：
- `ValueError`: 输入张量形状或数据类型不兼容

**算法复杂度**：
- 时间：$O(p \cdot N/p \cdot N) = O(N^2)$ 计算 + $O(p)$ 通信步
- 空间：$O(N/p + head\_dim)$ （每个 GPU）

#### 3.1.2 自定义自动求导函数

##### RingAttentionFunc

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

**前向传播算法**：

1. 初始化通信器
2. 保存原始 K 和 V 用于反向传播
3. Ring 循环（每次迭代一个步骤）：
   ```
   for step in range(cp_world_size):
       if step != cp_world_size - 1:
           启动异步通信获取下一个 K 和 V

       计算当前步骤的注意力
       更新输出和 LSE（log-sum-exp）

       等待通信完成
   ```
4. 将输出转换为输入数据类型
5. 保存张量用于反向传播

**反向传播算法**：

1. 初始化两个通信器（一个用于 KV，一个用于梯度）
2. 预分配梯度缓冲区
3. Ring 循环进行反向梯度计算：
   ```
   for step in range(cp_world_size):
       计算每一步的梯度
       累积梯度
       管理异步通信
   ```

#### 3.1.3 单步注意力计算

##### ring_attention_forward(q, k, v, sm_scale, is_causal)

单步注意力前向计算。

```python
output, lse = ring_attention_forward(
    q, k, v,
    sm_scale=1.0 / (head_dim ** 0.5),
    is_causal=False
)
```

**实现细节**：
- 计算注意力分数：$S = Q \cdot K^T \cdot \text{scale}$
- 应用因果掩码：如果 $j > i$，则 $S[i,j] = -\infty$
- 使用在线 Softmax：避免数值溢出
- 返回注意力输出和 log-sum-exp

**数学形式**：

$$\text{output} = \text{softmax}(Q K^T / \sqrt{d}) V$$

$$\text{LSE} = \log(\sum_j e^{Q_i K_j^T / \sqrt{d}})$$

##### ring_attention_backward(dout, q, k, v, output, lse, sm_scale, is_causal)

单步注意力反向计算。

```python
dq, dk, dv = ring_attention_backward(
    dout, q, k, v, output, softmax_lse,
    sm_scale=1.0 / (head_dim ** 0.5),
    is_causal=False
)
```

**梯度计算步骤**：
1. 重建注意力概率：$P = \text{softmax}(S)$
2. 计算对 V 的梯度：$\frac{\partial L}{\partial V} = P^T \frac{\partial L}{\partial O}$
3. 计算对注意力概率的梯度
4. 计算对分数的梯度
5. 应用因果掩码
6. 计算对 Q 和 K 的梯度

#### 3.1.4 数值稳定性优化

##### update_out_and_lse(current_out, current_lse, block_out, block_lse)

使用数值稳定的方法更新输出和 LSE。

```python
out, lse = update_out_and_lse(
    current_out,
    current_lse,
    block_out,
    block_lse
)
```

**数值稳定性技巧**：
- 使用 sigmoid 和 logsigmoid 进行稳定的指数更新
- 避免大数减法导致的精度丧失

**更新公式**：

$$\text{new\_out} = \text{old\_out} \cdot \alpha + \text{block\_out} \cdot (1-\alpha)$$

其中 $\alpha = \text{sigmoid}(\text{old\_lse} - \text{new\_lse})$

##### update_rope_for_context_parallel(cos, sin)

为上下文并行更新 RoPE（旋转位置编码）。

```python
cos_cp, sin_cp = update_rope_for_context_parallel(cos, sin)
```

**功能**：
- 将 RoPE 张量分区给各个 CP 秩
- 每个秩获得序列的一个连续段
- 用于处理长序列时的位置编码

**参数**：
- `cos` (Tensor): RoPE 余弦分量，形状 (seq_len, head_dim)
- `sin` (Tensor): RoPE 正弦分量，形状 (seq_len, head_dim)

**返回值**：
- 当前秩的 RoPE 分区

---

### 3.2 通信原语 (cp_comms.py)

#### 3.2.1 ContextCommunicate 类

管理 Ring 拓扑中的点对点通信。

```python
from scaletorch.parallel.context_parallel.cp_comms import ContextCommunicate

comm = ContextCommunicate('ring_attention_forward')
```

**初始化参数**：
- `name` (str): 通信操作的名称（用于日志和调试）

**设计目标**：
- 高效管理 Ring 拓扑下的点对点通信
- 支持批量异步操作
- 减少通信启动延迟

#### 3.2.2 核心方法

##### send_recv(tensor_to_send, recv_tensor=None)

发送数据到下一个秩，从上一个秩接收数据。

```python
# 异步模式
recv_k = comm.send_recv(send_k)
comm.commit()  # 开始实际传输
comm.wait()    # 等待完成
current_k = recv_k
```

**特点**：
- 异步操作，需要配合 `commit()` 和 `wait()` 使用
- 支持多组张量批处理
- 返回接收句柄

**参数**：
- `tensor_to_send` (Tensor): 要发送的张量
- `recv_tensor` (Tensor, optional): 接收缓冲区，如果为 None 自动分配

**返回值**：
- 接收到的张量句柄

##### commit()

提交所有待处理的 send_recv 操作。

```python
comm.send_recv(k)
comm.send_recv(v)
comm.commit()  # 开始实际传输
```

**功能**：
- 批量提交所有待处理操作
- 异步启动通信
- 减少通信启动次数

##### wait()

等待所有待处理的操作完成。

```python
comm.wait()  # 阻塞直到所有通信完成
```

**功能**：
- 阻塞等待所有通信完成
- 进行 CUDA 同步
- 确保数据有效性

#### 3.2.3 批量通信机制

**特点**：
- 支持多组张量（如 K/V）一次性批量通信
- 减少通信启动延迟
- 内部自动管理通信请求与资源回收

**使用模式**：

```python
# 单步通信
comm = ContextCommunicate('batch_comm')

# 批量提交多组张量
comm.send_recv(k)      # 张量 1
comm.send_recv(v)      # 张量 2
comm.send_recv(mask)   # 张量 3

# 一次性启动所有通信
comm.commit()

# 计算部分工作（可与通信重叠）
output = compute_attention(q, current_k, current_v)

# 等待所有通信完成
comm.wait()

# 获取通信结果
next_k = next_k_handle
next_v = next_v_handle
next_mask = next_mask_handle
```

## 4. Ring Attention 算法详解

### 4.1 数学形式化

#### 标准注意力机制

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

#### 分布式序列分割

将序列分成 $p$ 个分区：

$$K = [K_0; K_1; \cdots; K_{p-1}]$$

$$V = [V_0; V_1; \cdots; V_{p-1}]$$

#### Ring Attention 计算

GPU $i$ 计算的完整注意力：

$$O_i = \sum_{j=0}^{p-1} \text{Attention}(Q_i, K_j, V_j)$$

通过 $p$ 步的 Ring 传递实现：

**Step $s$**：GPU $i$ 接收 GPU $(i-s) \bmod p$ 的 K 和 V

$$O_i^{(s)} = \text{Attention}(Q_i, K_{(i-s) \bmod p}, V_{(i-s) \bmod p})$$

最终结果通过数值稳定的方式累积：

$$O_i = \bigoplus_{s=0}^{p-1} O_i^{(s)}$$

其中 $\bigoplus$ 表示稳定的组合操作。



## 5. 通信机制与优化

### 5.1 Ring 拓扑

```
物理拓扑：

GPU-0 ←→ GPU-1 ←→ GPU-2 ←→ GPU-3
  ↓                           ↑
  └──────────────────────────┘

每个 GPU 只与相邻 GPU 通信：
- GPU-0: 接收自 GPU-3，发送至 GPU-1
- GPU-1: 接收自 GPU-0，发送至 GPU-2
- GPU-2: 接收自 GPU-1，发送至 GPU-3
- GPU-3: 接收自 GPU-2，发送至 GPU-0
```

### 5.2 通信模式

| 阶段 | 操作           | 张量   | 方向      |
| ---- | -------------- | ------ | --------- |
| 前向 | 传递 K/V       | K, V   | Ring 循环 |
| 前向 | 启动下一步通信 | K, V   | 异步前传  |
| 反向 | 传递 dK/dV     | dK, dV | Ring 循环 |
| 反向 | AllReduce dQ   | dQ     | 所有 GPU  |

### 5.3 通信与计算重叠

```
时间轴 ────────────→

Step 1:
├─ 计算 Attn(Q_i, K_i, V_i) ────────┤
├─ 异步发送 K_i 到 GPU(i+1)         │
└─ 异步接收 K_{i-1} 从 GPU(i-1)     │

Step 2:
├─ 等待通信完成                     ├─────┤
├─ 计算 Attn(Q_i, K_{i-1}, V_{i-1})    ────────┤
├─ 异步发送 K_{i-1} 到 GPU(i+1)    │
└─ 异步接收 K_{i-2} 从 GPU(i-1)    │

通信延迟被计算时间隐藏
```

### 5.4 通信优化技巧

1. **批量通信**：多张量一次性提交，减少启动延迟
2. **异步操作**：前向推进时启动下一步通信
3. **本地优先**：同一节点内的 GPU 使用更快的通路
4. **通信融合**：合并小张量为大张量，提升带宽利用率


## 6. 性能分析

### 6.1 内存占用对比

| 方案            | 内存占用             | 说明                             |
| --------------- | -------------------- | -------------------------------- |
| 标准 Attention  | $O(N^2)$             | 存储完整 $N \times N$ 注意力矩阵 |
| Flash Attention | $O(N)$               | 块级计算，无额外缓存             |
| Ring Attention  | $O(N)$               | 只存储一个 KV 块                 |
| Ring + 梯度累积 | $O(N + \frac{N}{g})$ | 加上梯度缓冲区                   |

**实际数值示例**（序列长度 N=32K, 隐层维度 D=4096, 头数 H=64）：

```
标准 Attention:
  注意力矩阵: 32K × 32K × 4 bytes = 4 GB
  总内存: ~6 GB

Ring Attention (4 GPU):
  每 GPU 序列: 8K × 4096 × 4 bytes = 128 MB
  KV 缓冲: 128 MB
  输出缓冲: 128 MB
  总内存/GPU: ~400 MB

内存节省: 6GB / 0.4GB ≈ 15x
```

### 6.2 通信复杂度

假设 CP 大小为 $p$，序列长度为 $N$：

| 操作              | 通信量         | 步数 | 总通信                 |
| ----------------- | -------------- | ---- | ---------------------- |
| Ring 前向 K/V     | $O(N \cdot D)$ | $p$  | $O(p \cdot N \cdot D)$ |
| Ring 前向异步启动 | 同上           | 1    | $O(N \cdot D)$         |
| 反向 dK/dV Ring   | $O(N \cdot D)$ | $p$  | $O(p \cdot N \cdot D)$ |
| AllReduce dQ      | $O(N \cdot D)$ | 1    | $O(N \cdot D)$         |

**相对于序列长度**：通信量 $O(N)$ 线性相关，而不是 $O(N^2)$

### 6.3 吞吐量分析

**单个 GPU 性能**（假设 A100 GPU）：

```
FP32 计算: ~312 TFLOPS
GPU 内存带宽: ~2 TB/s
网络带宽: 400 Gbps ≈ 50 GB/s

计算强度（token 处理）:
  标准 Attn: N² 操作 / N·D 数据 = N/D （很高）
  Ring Attn: N² 操作 / (p·N·D) 数据 = N/(p·D) （较低，但均衡）

瓶颈：
  标准 Attn：对大 N 内存瓶颈
  Ring Attn：通信瓶颈（网络 vs PCIe）
```

### 6.4 加速比

相对于单 GPU 标准 Attention：

```
假设:
  N = 32K (序列长度)
  p = 4 (CP 大小)
  网络延迟因子 λ = 1.1x（相对于本地）

计算加速: 接近线性（每 GPU 处理 N/p 序列）

实际加速比:
  理想: 4x
  实际: 3.5-3.8x （10-20% 通信开销）

关键因素:
  - 网络拓扑：同节点最优
  - 序列长度：越长优势越明显
  - 梯度累积：隐藏通信延迟
```

---

## 7. 最佳实践

### 7.1 环境配置

```python
import os
import torch.distributed as dist
from scaletorch.parallel.pg_manager import setup_process_group_manager
from scaletorch.parallel.context_parallel import apply_context_parallel

# 步骤 1: 初始化分布式环境
dist.init_process_group(
    backend='nccl',
    init_method='env://'
)

# 步骤 2: 设置进程组
pgm = setup_process_group_manager(
    tp_size=1,      # 无张量并行
    cp_size=4,      # 上下文并行大小
    pp_size=1,      # 无流水线并行
    dp_size=1       # 数据并行大小（如需要）
)

# 步骤 3: 配置环境变量
os.environ['VERBOSE'] = '1'  # 启用详细日志
os.environ['CONTEXT_PARALLEL'] = '1'

# 步骤 4: 创建和配置模型
model = create_model()
model = apply_context_parallel(model)
```

### 7.2 序列分割原则

```python
# ✓ 好的做法
batch_size = 8
seq_len = 32768      # 能被 CP 大小整除
cp_size = 4
seq_per_gpu = seq_len // cp_size  # = 8192

# ✗ 避免的做法
seq_len = 32000      # 不能被 4 整除
# 会导致分区不均，某些 GPU 更多数据

# 推荐配置
seq_lengths = [4096, 8192, 16384, 32768, 65536]
cp_sizes = [1, 2, 4, 8]
# 确保 seq_len % cp_size == 0
```

### 7.3 性能优化建议

1. **选择合适的 CP 大小**
   ```python
   # 原则：利用同一节点的高速通信
   # 同一节点 GPU 数最大值
   gpus_per_node = 8
   cp_size = 4  # 同一节点内

   # 跨节点时谨慎，可能引入显著延迟
   ```

2. **设置梯度累积**
   ```python
   # 隐藏通信延迟
   grad_accumulation_steps = 4
   micro_batch_size = batch_size // grad_accumulation_steps
   ```

3. **启用混合精度训练**
   ```python
   from torch.cuda.amp import autocast, GradScaler

   scaler = GradScaler()
   with autocast(dtype=torch.float16):
       output = model(input)
       loss = compute_loss(output)
   ```

4. **监控性能指标**
   ```python
   # 跟踪以下指标
   - tokens/sec（吞吐量）
   - GPU 利用率
   - 通信时间占比
   - 内存占用
   ```

### 7.4 调试技巧

```python
import os

# 启用详细日志
os.environ['VERBOSE'] = '1'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

# 验证通信拓扑
import torch.distributed as dist
print(f"Rank: {dist.get_rank()}")
print(f"World Size: {dist.get_world_size()}")

# 检查 Ring 拓扑
from scaletorch.parallel.pg_manager import get_process_group_manager
pgm = get_process_group_manager()
print(f"CP Rank: {pgm.cp_rank}")
print(f"CP World Size: {pgm.cp_world_size}")
print(f"CP Prev Rank: {pgm.cp_prev_rank}")
print(f"CP Next Rank: {pgm.cp_next_rank}")

# 验证张量形状和数据类型
q, k, v = prepare_qkv()
print(f"Q shape: {q.shape}, dtype: {q.dtype}, device: {q.device}")
print(f"K shape: {k.shape}, dtype: {k.dtype}, device: {k.device}")
```

## 8. 常见问题

### Q1: Ring Attention 何时比标准 Attention 更优？

**A**: 当满足以下条件时：

```python
# 1. 序列长度足够长
seq_len >= 1024  # 最少 1024，推荐 > 4096

# 2. 内存是主要瓶颈
# 标准 Attention 内存占用 O(N²) 会导致 OOM
# Ring Attention O(N) 可以处理

# 3. 计算能力充足
# 计算时间 >> 通信时间
# 或能有效重叠通信与计算

# 实际对标
if seq_len > 4096:
    use_ring_attention = True
else:
    use_ring_attention = False  # 使用标准 Attention 或 Flash Attention
```

### Q2: CP 大小应该如何选择？

**A**: 考虑以下因素：

```python
def recommend_cp_size(total_gpus, seq_len, node_gpus=8):
    """推荐 CP 大小"""

    # 原则 1: 同一节点内优先
    if total_gpus <= node_gpus:
        return total_gpus

    # 原则 2: 同节点子集
    if total_gpus >= node_gpus:
        return node_gpus  # 或 4/8

    # 原则 3: 序列长度整除
    for cp_size in range(min(total_gpus, 8), 0, -1):
        if seq_len % cp_size == 0:
            return cp_size

    return 1  # 不使用 CP

# 示例
print(recommend_cp_size(16, 32768, 8))  # 8
print(recommend_cp_size(8, 16384, 8))   # 8
print(recommend_cp_size(4, 32000, 8))   # 4
```

### Q3: Ring Attention 能否与其他并行策略组合？

**A**: 可以，推荐的组合：

```python
# CP + DP: 长序列训练
combine_cp_dp = {
    'cp_size': 4,      # 同节点 Ring
    'dp_size': 4,      # 跨节点数据并行
    'tp_size': 1,
    'pp_size': 1
}

# CP + TP: 需要两个并行维度
combine_cp_tp = {
    'cp_size': 2,      # 序列长度
    'tp_size': 4,      # 隐层维度
    'dp_size': 1,
    'pp_size': 1
}

# CP + DP + TP: 完整混合并行
combine_all = {
    'cp_size': 2,      # 序列
    'tp_size': 2,      # 隐层
    'dp_size': 2,      # 批次
    'pp_size': 1       # 不使用流水线
}

# ⚠️ 不推荐
bad_combine = {
    'cp_size': 8,      # Ring 过大
    'tp_size': 8,      # 张量并行过大
    'pp_size': 4       # 流水线 + CP 通信复杂
}
```

### Q4: 如何处理梯度同步问题？

**A**:

```python
import torch.distributed as dist

# 问题: CP 环形拓扑中不能直接 AllReduce dQ

# 解决方案: 先按 CP 秩同步，再全局同步
def sync_gradients(model, pgm):
    """正确的梯度同步流程"""

    for param in model.parameters():
        if param.grad is not None:
            # 步骤 1: CP 内部同步（Ring 上的点对点）
            # 这在 Ring Attention 反向中已做

            # 步骤 2: 全局 AllReduce（所有 GPU）
            dist.all_reduce(param.grad)
            param.grad /= dist.get_world_size()

# 或使用梯度累积避免频繁同步
accumulation_steps = 4
for step in range(num_steps):
    loss = model(batch)
    loss.backward()

    if (step + 1) % accumulation_steps == 0:
        # 仅在积累完成时同步
        sync_gradients(model, pgm)
        optimizer.step()
        optimizer.zero_grad()
```

### Q5: 如何监控 Ring Attention 的通信开销？

**A**:

```python
import time
import torch.distributed as dist

class CommProfiler:
    def __init__(self):
        self.comm_times = []
        self.compute_times = []

    def profile_ring_attention(self, q, k, v):
        """分析通信开销"""

        # 通信时间
        torch.cuda.synchronize()
        start_comm = time.time()

        output = ring_attention(q, k, v)

        torch.cuda.synchronize()
        comm_time = time.time() - start_comm
        self.comm_times.append(comm_time)

        # 计算比例
        comm_ratio = comm_time / (comm_time + compute_time)

        return comm_ratio

profiler = CommProfiler()
for batch in dataloader:
    q, k, v = prepare_qkv(batch)
    comm_ratio = profiler.profile_ring_attention(q, k, v)
    print(f"Communication ratio: {comm_ratio:.2%}")
```

### Q6: 能否在 CPU 上使用 Ring Attention？

**A**: 不推荐，原因如下：

```python
# Ring Attention 优化了 GPU 的异步通信
# CPU 上：
# - 通信带宽远低于 GPU（PCIe vs NVLink）
# - 计算速度更慢
# - 通信开销比例太大

# 相反，应该用：
- CPU 上: 标准 Attention 或优化的 CPU 实现
- GPU 上: Ring Attention（尤其是超长序列）
```

---

## 参考文献

1. [Ring Attention with Blockwise Transformers for Context-Aware Generation](https://arxiv.org/abs/2310.01889)
   - 原始 Ring Attention 论文，提出了高效的分布式注意力算法

2. [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
   - 关于注意力内存优化的经典工作

3. [Flash Attention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
   - 块级注意力计算参考

4. PyTorch Distributed 官方文档
   - https://pytorch.org/docs/stable/distributed.html
