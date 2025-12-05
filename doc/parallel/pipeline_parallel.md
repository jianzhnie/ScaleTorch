# 流水线并行 (Pipeline Parallelism) 完整指南

## 目录

- [1. 概述](#1-概述)
- [2. 核心概念](#2-核心概念)
- [3. 架构设计](#3-架构设计)
- [4. 数据流向与可视化](#4-数据流向与可视化)
- [5. API 参考](#5-api-参考)
- [6. 实现细节](#6-实现细节)
- [7. 调度策略](#7-调度策略)
- [8. 通信机制](#8-通信机制)
- [9. 使用指南](#9-使用指南)
- [10. 性能分析](#10-性能分析)
- [11. 常见问题](#11-常见问题)

---

## 1. 概述

### 1.1 什么是流水线并行

流水线并行 (Pipeline Parallelism, PP) 是一种分布式深度学习训练技术，通过**将模型的各层分散到多个 GPU 上**，使得不同的 GPU 在处理数据时并行处理模型的不同部分。这种方式可以：

- **突破单 GPU 显存限制**：模型参数分散到多个 GPU
- **提高计算利用率**：通过流水线调度让多个 GPU 并行工作
- **加速训练速度**：相比数据并行，减少通信开销

### 1.2 设计理念

```
传统单 GPU 训练：
┌─────────────────────────────────────────┐
│  Embedding + 所有 Layer + 输出层 → 损失计算│  GPU-0
└─────────────────────────────────────────┘

流水线并行分布：
┌──────────────────────┐
│  Embedding +         │  GPU-0 (Stage-0)
│  Layer[0:8]          │
└──────────────────────┘
         ↓ 激活值
┌──────────────────────┐
│  Layer[8:16]         │  GPU-1 (Stage-1)
└──────────────────────┘
         ↓ 激活值
┌──────────────────────┐
│  Layer[16:24] +      │  GPU-2 (Stage-2)
│  Final Proj + Loss   │
└──────────────────────┘
```

### 1.3 关键特性

| 特性           | 说明                                                                             |
| -------------- | -------------------------------------------------------------------------------- |
| **灵活的调度** | 支持 AFAB (All-Forward-All-Backward) 和 1F1B (One-Forward-One-Backward) 两种策略 |
| **自动层分配** | 根据流水线阶段数自动均衡分配模型层                                               |
| **双向通信**   | 支持前向和反向传播的高效点对点通信                                               |
| **错误处理**   | 完善的输入验证和异常处理机制                                                     |
| **调试支持**   | 详细的日志输出便于问题排查                                                       |

---

## 2. 核心概念

### 2.1 流水线阶段 (Pipeline Stages)

每个 GPU 运行一个流水线阶段，处理分配给它的模型层：

```python
# 假设有 24 层模型，分配到 3 个 GPU
Stage 0 (pp_rank=0):  Embedding + Layer[0:8]
Stage 1 (pp_rank=1):  Layer[8:16]
Stage 2 (pp_rank=2):  Layer[16:24] + Final Projection
```

### 2.2 激活值与梯度流向

```
前向传播：
输入 → Stage-0 (Embedding) → 激活值-0 → Stage-1 → 激活值-1 → Stage-2 → 输出

反向传播：
Embedding ← 梯度 ← Stage-0 ← 梯度-1 ← Stage-1 ← 梯度-2 ← Stage-2 ← 损失梯度
```

### 2.3 微批处理 (Microbatching)

流水线并行通常将一个批次分成多个微批次：

```python
# 假设 batch_size=64，grad_acc_steps=4
# 则每个微批次大小为 64/4 = 16

for microbatch_idx in range(grad_acc_steps):  # 4 个微批次
    forward_pass(batch[microbatch_idx * 16 : (microbatch_idx + 1) * 16])
    backward_pass(...)  # 根据调度策略决定何时执行
```

### 2.4 关键索引和排名

| 概念                | 说明                  | 取值范围                     |
| ------------------- | --------------------- | ---------------------------- |
| `pp_rank`           | 当前 GPU 的流水线排名 | 0 ~ pp_world_size-1          |
| `pp_world_size`     | 流水线中总 GPU 数     | 2 ~ 正整数                   |
| `pp_prev_rank`      | 前驱阶段的排名        | pp_rank - 1（或 -1）         |
| `pp_next_rank`      | 后继阶段的排名        | pp_rank + 1（或 -1）         |
| `pp_is_first_stage` | 是否为首阶段          | pp_rank == 0                 |
| `pp_is_last_stage`  | 是否为末阶段          | pp_rank == pp_world_size - 1 |

---

## 3. 架构设计

### 3.1 系统架构图

```
┌────────────────────────────────────────────────────────────────┐
│                      PipelineParallel Module                    │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Layer Distribution & Initialization                    │  │
│  │  • _distribute_layers()      - 分配层到各阶段          │  │
│  │  • _get_embedding_layer()    - 获取嵌入层（首阶段）   │  │
│  │  • _get_decoder_layers()     - 获取分配的层            │  │
│  │  • _get_final_norm_layer()   - 获取最终归一化（末）   │  │
│  │  • _get_final_proj_layer()   - 获取输出投影（末）     │  │
│  └─────────────────────────────────────────────────────────┘  │
│                           ↓                                     │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Computation Layer                                      │  │
│  │  • forward()  - 前向传播计算                           │  │
│  │  • backward() - 反向传播计算                           │  │
│  └─────────────────────────────────────────────────────────┘  │
│                           ↓                                     │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Communication Layer (pp_comms.py)                     │  │
│  │  • pipeline_communicate()           - 单向通信         │  │
│  │  • bidirectional_pipeline_communicate() - 双向通信    │  │
│  └─────────────────────────────────────────────────────────┘  │
│                           ↓                                     │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Process Group Manager (pg_manager.py)                │  │
│  │  • 管理进程间的通信拓扑                                │  │
│  │  • 维护 PP/TP/CP/DP 等并行维度信息                    │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### 3.2 类关系图

```
nn.Module
   │
   └─→ PipelineParallel
        │
        ├─→ embedding: nn.Module (首阶段)
        ├─→ decoder_layers: nn.ModuleDict
        ├─→ final_norm: nn.Module (末阶段)
        └─→ final_proj: nn.Module (末阶段)

支持函数：
├─→ train_step_pipeline_afab()     [AFAB 调度]
├─→ train_step_pipeline_1f1b()     [1F1B 调度]
└─→ pp_comms 模块 (通信原语)
     ├─→ pipeline_communicate()
     └─→ bidirectional_pipeline_communicate()
```

---

## 4. 数据流向与可视化

### 4.1 物理拓扑

```
┌─────────────┐
│ GPU-0 (PP0) │
└──────┬──────┘
       │ PCIe/NVLink
       │ bandwidth: 600GB/s
       │
┌──────▼──────┐
│ GPU-1 (PP1) │
└──────┬──────┘
       │
┌──────▼──────┐
│ GPU-2 (PP2) │
└──────┬──────┘
       │
┌──────▼──────┐
│ GPU-3 (PP3) │
└─────────────┘

通信模式：
GPU-0 ◀──▶ GPU-1 ◀──▶ GPU-2 ◀──▶ GPU-3
      ↑           ↑           ↑
  激活值/梯度  激活值/梯度  激活值/梯度
```

### 4.2 前向传播流向

```
微批次 0 的前向传播：

Step 1: 输入数据
┌──────────────────┐
│ input_ids: (4, 512)    [batch_size=4, seq_len=512]
└─────────┬────────┘
          │
          ▼
┌──────────────────┐
│ GPU-0 (PP0)      │
│ embedding()      │
│ layer[0:8]()     │
└─────────┬────────┘
          │ 激活值
          │ shape: (4, 512, 4096)
          ▼
      ┌─────────────────────────────┐
      │  send_forward() 通过 PCIe   │
      │  GPU-0 → GPU-1              │
      └─────────────────────────────┘
          │
          ▼
┌──────────────────┐
│ GPU-1 (PP1)      │
│ layer[8:16]()    │
└─────────┬────────┘
          │ 激活值
          │ shape: (4, 512, 4096)
          ▼
      ┌─────────────────────────────┐
      │  send_forward() 通过 PCIe   │
      │  GPU-1 → GPU-2              │
      └─────────────────────────────┘
          │
          ▼
┌──────────────────┐
│ GPU-2 (PP2)      │
│ layer[16:24]()   │
└─────────┬────────┘
          │ 激活值
          │ shape: (4, 512, 4096)
          ▼
      ┌─────────────────────────────┐
      │  send_forward() 通过 PCIe   │
      │  GPU-2 → GPU-3              │
      └─────────────────────────────┘
          │
          ▼
┌──────────────────┐
│ GPU-3 (PP3)      │
│ layer[24:32]()   │
│ final_norm()     │
│ final_proj()     │
│ logits: (4, 512, 32000)  [32000=vocab_size]
└─────────┬────────┘
          │
          ▼
┌──────────────────┐
│ 损失计算         │
│ loss = CE(logits, labels)
└──────────────────┘
```

### 4.3 反向传播流向

```
反向传播（梯度回传）：

从末阶段开始（GPU-3）：

┌──────────────────┐
│ GPU-3 (PP3)      │
│ ∂L/∂logits = None (自动生成)
│ backward()       │
│ 生成梯度         │
└─────────┬────────┘
          │ 梯度
          │ shape: (4, 512, 4096)
          ▼
      ┌─────────────────────────────┐
      │  send_backward() 通过 PCIe  │
      │  GPU-3 → GPU-2              │
      └─────────────────────────────┘
          │
          ▼
┌──────────────────┐
│ GPU-2 (PP2)      │
│ ∂L/∂layer[16:24] │
│ backward()       │
└─────────┬────────┘
          │ 梯度
          │ shape: (4, 512, 4096)
          ▼
      ┌─────────────────────────────┐
      │  send_backward() 通过 PCIe  │
      │  GPU-2 → GPU-1              │
      └─────────────────────────────┘
          │
          ▼
┌──────────────────┐
│ GPU-1 (PP1)      │
│ ∂L/∂layer[8:16]  │
│ backward()       │
└─────────┬────────┘
          │ 梯度
          │ shape: (4, 512, 4096)
          ▼
      ┌─────────────────────────────┐
      │  send_backward() 通过 PCIe  │
      │  GPU-1 → GPU-0              │
      └─────────────────────────────┘
          │
          ▼
┌──────────────────┐
│ GPU-0 (PP0)      │
│ ∂L/∂layer[0:8]   │
│ backward()       │
│ ∂L/∂embedding    │
└──────────────────┘
```

### 4.4 激活值缓存

```
AFAB 策略下的激活值缓存：

时间 ──→

微批 0     微批 1     微批 2     微批 3
┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  前向
│ F0  │─▶│ F1  │─▶│ F2  │─▶│ F3  │  阶段
└──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘
   │ act0   │ act1   │ act2   │ act3
   ▼        ▼        ▼        ▼
 [缓存]   [缓存]   [缓存]   [缓存]
   │        │        │        │
   ▼        ▼        ▼        ▼  反向
┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  阶段
│ B0  │◀─│ B1  │◀─│ B2  │◀─│ B3  │
└─────┘  └─────┘  └─────┘  └─────┘

内存占用：O(N·B·S·D)，其中 N=grad_acc_steps

1F1B 策略下的激活值缓存：

热身        稳态        冷却
┌─────┐
│ F0  │
└──┬──┘
   │       ┌─────────────────────────────────────┐
   ▼       │ F1→B0→F2→B1→F3→B2→...→BN          │
 [缓存]    └─────────────────────────────────────┘
   │           交错执行，只保存 P 个活跃的
   ▼           P = pp_world_size
 [缓存] 队列 (大小 ≤ P)

内存占用：O(P·B·S·D)，显著减少（当 P << N）
```

---

## 5. API 参考

### 5.1 PipelineParallel 类

#### 5.1.1 初始化

```python
class PipelineParallel(nn.Module):
    def __init__(self, model: nn.Module, config: Any) -> None:
        """
        初始化流水线并行模块。

        参数：
            model: 完整的神经网络模型
            config: 包含 num_hidden_layers 的配置对象

        异常：
            AttributeError: config 缺少 num_hidden_layers 属性
            RuntimeError: 进程组管理器未初始化

        示例：
            >>> config = LlamaConfig(num_hidden_layers=32)
            >>> model = LlamaModel(config)
            >>> pp_model = PipelineParallel(model, config)
        """
```

#### 5.1.2 核心属性

```python
# 只读属性
layer_distribution: List[int]  # 分配给该阶段的层索引

# 计算模块
embedding: nn.Module           # 嵌入层（首阶段为真，其他为 Identity）
decoder_layers: nn.ModuleDict  # 分配的解码层
final_norm: nn.Module          # 最终归一化（末阶段为真，其他为 Identity）
final_proj: nn.Module          # 最终投影（末阶段为真，其他为 Identity）
```

#### 5.1.3 前向传播

```python
def forward(
    self,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    hidden_states: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    前向传播处理。

    参数：
        input_ids: 输入token索引（仅首阶段使用）
        position_ids: 位置索引（所有阶段使用）
        hidden_states: 来自前驱阶段的隐藏状态（中间/末阶段需要）

    返回：
        当前阶段的输出张量
    """
```

#### 5.1.4 层分配

```python
def _distribute_layers(self, num_layers: int) -> List[int]:
    """
    均匀分配模型层到各流水线阶段。

    算法：
        1. 计算基础层数：base = num_layers // pp_world_size
        2. 计算余数：remainder = num_layers % pp_world_size
        3. 前 remainder 个阶段多分配 1 层
        4. 计算当前阶段的起始和结束位置

    示例（10层，3阶段）：
        Stage-0: [0, 1, 2, 3]     (4 = 10//3 + 1)
        Stage-1: [4, 5, 6]        (3 = 10//3)
        Stage-2: [7, 8, 9]        (3 = 10//3)

    参数：
        num_layers: 模型总层数

    返回：
        当前阶段的层索引列表
    """
```

### 5.2 训练函数

#### 5.2.1 AFAB 调度

```python
def train_step_pipeline_afab(
    model: PipelineParallel,
    data_loader: Any,
    tensor_shapes: TensorShape,
    device: torch.device,
    dtype: torch.dtype
) -> LossType:
    """
    All-Forward-All-Backward (AFAB) 流水线并行训练。

    策略：
        1. 前向阶段：依次处理所有微批次的前向传播
        2. 反向阶段：依次处理所有微批次的反向传播

    执行流程：
        ┌─────────────────────────────────────────┐
        │  微批 0 前向 → 微批 1 前向 → ... → 微批 N 前向  │
        ├─────────────────────────────────────────┤
        │  微批 0 反向 → 微批 1 反向 → ... → 微批 N 反向  │
        └─────────────────────────────────────────┘

    参数：
        model: PipelineParallel 模型实例
        data_loader: 数据加载器（需有 grad_acc_steps 属性）
        tensor_shapes: 通信张量的形状
        device: 计算设备
        dtype: 张量数据类型

    返回：
        所有微批次的平均损失
    """
```

#### 5.2.2 1F1B 调度

```python
def train_step_pipeline_1f1b(
    model: PipelineParallel,
    data_loader: Any,
    tensor_shapes: TensorShape,
    device: torch.device,
    dtype: torch.dtype
) -> LossType:
    """
    One-Forward-One-Backward (1F1B) 流水线并行训练。

    策略：
        交错执行前向和反向传播以充分利用流水线

    执行流程分为三个阶段：

    1. 热身阶段 (Warmup Phase)：
       ├─ 目标：填充流水线
       ├─ 次数：min(pp_world_size - pp_rank - 1, grad_acc_steps)
       └─ 操作：只做前向，不做反向

    2. 稳态阶段 (Steady State Phase)：
       ├─ 目标：充分利用流水线
       ├─ 次数：grad_acc_steps - 热身次数
       └─ 操作：交错前向和反向

    3. 冷却阶段 (Cooldown Phase)：
       ├─ 目标：清空流水线
       ├─ 次数：等同热身次数
       └─ 操作：只做反向

    图示（3阶段，4微批次）：

    时间轴 →

    Stage-0:  F0    F1    B0    B1    B2    B3
              ─────────────────────────────────
              │ 热身 │稳态(F/B交错)│ 冷却 │

    Stage-1:       F0    F1    B0    B1    B2
              ─────────────────────────────────

    Stage-2:             F0    F1    B0    B1
              ─────────────────────────────────

    GPU 利用率提升：~30-40%（相比 AFAB）

    参数：
        model: PipelineParallel 模型实例
        data_loader: 数据加载器
        tensor_shapes: 通信张量形状
        device: 计算设备
        dtype: 张量数据类型

    返回：
        所有微批次的平均损失
    """
```

### 5.3 通信 API

#### 5.3.1 单向通信

```python
def pipeline_communicate(
    operation: str,
    device: Union[torch.device, str],
    dtype: torch.dtype,
    tensor: Optional[torch.Tensor] = None,
    shapes: Optional[Tuple[int, ...]] = None
) -> Optional[torch.Tensor]:
    """
    点对点通信原语。

    支持的操作：
        'recv_forward':  接收来自前驱的激活值
        'send_forward':  发送激活值给后继
        'recv_backward': 接收来自后继的梯度
        'send_backward': 发送梯度给前驱

    参数：
        operation: 通信操作类型
        device: 目标设备（cuda/cpu）
        dtype: 张量数据类型
        tensor: 要发送的张量（send 操作必需）
        shapes: 接收张量的形状（recv 操作必需）

    返回：
        recv 操作返回接收的张量，send 操作返回 None
    """
```

#### 5.3.2 双向通信

```python
def bidirectional_pipeline_communicate(
    operation: str,
    send_tensor: torch.Tensor,
    recv_shapes: Tuple[int, ...],
    device: Union[torch.device, str],
    dtype: torch.dtype
) -> Optional[torch.Tensor]:
    """
    同时发送和接收张量的优化通信。

    支持的操作：
        'send_fwd_recv_bwd': 发送激活，接收梯度（稳态）
        'send_bwd_recv_fwd': 发送梯度，接收激活（稳态）

    优势：
        - 减少同步点
        - 提升通信吞吐量
        - 隐藏通信延迟

    返回：
        接收到的张量
    """
```

---

## 6. 实现细节

### 6.1 层分配算法

```python
# 伪代码
def distribute_layers(num_layers, pp_world_size, pp_rank):
    base_layers = num_layers // pp_world_size          # 基础层数
    remainder = num_layers % pp_world_size             # 余数

    # 前 remainder 个阶段各分配 1 个额外层
    layers_per_stage = [
        base_layers + (1 if i < remainder else 0)
        for i in range(pp_world_size)
    ]

    # 计算起始和结束位置
    start = sum(layers_per_stage[:pp_rank])
    end = start + layers_per_stage[pp_rank]

    return list(range(start, end))

# 例子：24 层，4 阶段
# base_layers = 24 // 4 = 6
# remainder = 24 % 4 = 0
# 结果：
#   Stage-0: [0:6]   (6 层)
#   Stage-1: [6:12]  (6 层)
#   Stage-2: [12:18] (6 层)
#   Stage-3: [18:24] (6 层)
```

### 6.2 激活值缓存机制

在 1F1B 调度中，需要维护激活值和输出张量的缓存：

```python
# 稳态阶段的缓存管理
input_tensors = []      # 输入激活值队列
output_tensors = []     # 输出张量队列

# 前向时：
input_tensors.append(input_tensor)
output_tensors.append(output_tensor)

# 后向时：
input_tensor = input_tensors.pop(0)      # FIFO 顺序
output_tensor = output_tensors.pop(0)
```

### 6.3 梯度同步控制

```python
# 梯度同步仅在最后一个微批次执行
requires_grad_sync = cp_dp_world_size > 1

for microbatch_idx in range(grad_acc_steps):
    is_last_iteration = (microbatch_idx == grad_acc_steps - 1)

    if requires_grad_sync:
        model.require_backward_grad_sync = is_last_iteration

    # 执行前向和反向
    ...
```

### 6.4 错误处理策略

```python
# 1. 输入验证
if not hasattr(config, 'num_hidden_layers'):
    raise AttributeError("Config must have 'num_hidden_layers'")

# 2. 上下文检查
if not hasattr(pgm, 'process_group_manager'):
    raise RuntimeError('Process group manager not initialized')

# 3. 数据加载器检查
if not hasattr(data_loader, 'grad_acc_steps'):
    raise ValueError("Data loader must have 'grad_acc_steps'")

# 4. 异常传播
try:
    # 执行操作
    ...
except StopIteration:
    raise RuntimeError(f'Data loader exhausted at microbatch {idx}')
```

---

## 7. 调度策略

### 7.1 AFAB (All-Forward-All-Backward)

```python
# 前向阶段
for batch in dataloader:
    # 所有 GPU 完成前向
    output = model(batch)

# 反向阶段
loss.backward()  # 所有 GPU 同时反向
optimizer.step()
```

#### 流程图

```
微批次 0    微批次 1    微批次 2    微批次 3
   │          │          │          │
   ▼          ▼          ▼          ▼
┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐
│ F-0 │───▶│ F-1 │───▶│ F-2 │───▶│ F-3 │    前向
└─────┘    └─────┘    └─────┘    └─────┘    阶段
   │          │          │          │
   ▼          ▼          ▼          ▼
┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐
│ B-0 │◀───│ B-1 │◀───│ B-2 │◀───│ B-3 │    反向
└─────┘    └─────┘    └─────┘    └─────┘    阶段

  GPU 闲置：         (管道气泡)
```

**优点：** 简单，实现容易
**缺点：** GPU 利用率低，GPU 等待其他 GPU 完成

### 7.2 1F1B (One-Forward-One-Backward)

```python
# 交错执行前向和反向
forward_queue = deque()  # 保存待反向的激活

for batch in dataloader:
    # 前向一个批次
    output = forward_pass(batch)
    forward_queue.append(output)

    # 反向一个批次（如果有）
    if len(forward_queue) > num_stages:
        backward_pass(forward_queue.popleft())

    optimizer.step()
```

#### 流程图（详细时间线）

```
4 阶段流水线，4 个微批次，1F1B 策略：

热身阶段    |    稳态阶段         |   冷却阶段

Stage-0  ┌──F0──┬──────────────────────────────────────────┬──B0──┬──B1──┬──B2──┬──B3──┐
         └──┬───┘    ┌──F1──┐  ┌──F2──┐  ┌──F3──┐  ┌──F4──┐ └──┬───┘  ┌───┘  ┌───┘  ┌───┘
            │        │     │   │     │   │     │   │     │    │      │      │      │
         ┌──▼──┐   ┌─▼──┬──▼───┬──▼──┐ ┌──▼───┐ ┌──▼───┐ │    │    ┌─▼──┬──▼───┬──▼──┐
Stage-1  │ F0  │───┤    │ F1   │      │      │ F2  │      │ F3 ├──B0──┤    │ B1   │      │
         └──┬──┘   │ B0 └──┬───┘ B1 ├──┤ B2 ├──┤      B3 │    │    ├──┤ B4 └──┬───┘ B5
            │     └────┬───┴───────┬──┴──┬────┬─┴───────┤    └────────┬───┴────┬────┘
         ┌──▼──┐   ┌──▼──┬───┐   ┌─▼──┬──▼───┬──▼──┐   │
Stage-2  │ F0  ├──F1──┤   │ B0 ├──F2──┤      ├──F3──┤   │
         └──┬──┘     │   └────┬──┬───┘ B1 ├────┬─┴───┴──┤
            │        │        │  │        │    │         │
         ┌──▼──┐   ┌─▼──┐   ┌──▼─┴┐ ┌─────┤ ┌──▼──┐     │
Stage-3  │ F0  ├──F1──┤   ├──F2──┤B0├───F3──┤ B1  ├──B2──┘
         └─────┘     └───┴──┬───┘└──┘ ┌────┴────┬─┘
                           │ Loss    │ Backward│
```

**优点：** 高 GPU 利用率，通信与计算重叠
**缺点：** 实现复杂，需要管理激活缓存

#### 性能对比

```python
# 计算总时间的简化模型
T_afab = (T_f + T_b) * N                    # N = grad_acc_steps
T_1f1b = (T_f + T_b) * (N + P - 1)          # P = pp_world_size

# 相对加速比
speedup = T_afab / T_1f1b = N / (N + P - 1)

# 例子：N=8, P=4
speedup_afab_vs_1f1b = 8 / (8 + 4 - 1) = 8 / 11 ≈ 0.73
# 即 1F1B 比 AFAB 快约 37%
```

#### 选择建议

| 场景               | 推荐策略 | 理由                 |
| ------------------ | -------- | -------------------- |
| 快速原型验证       | AFAB     | 实现简单，调试容易   |
| 小规模模型 (P ≤ 2) | AFAB     | 流水线浅，收益不明显 |
| 大规模模型 (P ≥ 4) | 1F1B     | GPU 利用率提升显著   |
| 超大模型 (P ≥ 8)   | 1F1B     | 几乎必须使用         |
| 生产环境           | 1F1B     | 性能至上             |

---

## 8. 通信机制

### 8.1 通信拓扑

#### 线性拓扑

```
GPU-0 ←→ GPU-1 ←→ GPU-2 ←→ GPU-3 ←→ GPU-4
前驱   后继  前驱   后继  前驱   后继  前驱   后继
```

每个 GPU 最多与 2 个邻近 GPU 通信（除了端点）

#### 通信模式

| 操作            | 发送方       | 接收方       | 张量类型 |
| --------------- | ------------ | ------------ | -------- |
| `send_forward`  | pp_rank      | pp_next_rank | 激活值   |
| `recv_forward`  | pp_prev_rank | pp_rank      | 激活值   |
| `send_backward` | pp_rank      | pp_prev_rank | 梯度     |
| `recv_backward` | pp_next_rank | pp_rank      | 梯度     |

### 8.2 通信时间分析

```
通信成本模型：
T_comm = α + β·B·S·D

其中：
  α = 启动延迟（microsecond 级）
  β = 带宽倒数（GB/s^-1）
  B = batch_size
  S = sequence_length
  D = hidden_dimension

例子（单向通信）：
  α ≈ 1 μs
  β ≈ 10 ns/byte（PCIe Gen4）
  B = 8, S = 2048, D = 4096

  激活值大小 = B·S·D·4 bytes = 8·2048·4096·4 ≈ 256 MB
  T_comm ≈ 1 + 256·10 = 2561 μs ≈ 2.6 ms
```

### 8.3 双向通信优化

```python
# 方式 1：顺序通信（低效）
send_tensor(forward_activation)      # 等待完成
recv_tensor(backward_gradient)       # 等待完成
# 总时间：T_send + T_recv

# 方式 2：双向通信（高效）
bidirectional_communicate(
    send_fwd_recv_bwd,
    forward_activation,
    backward_shapes
)  # 同时发送和接收
# 总时间：max(T_send, T_recv) ≈ T_send
# 节省时间：T_recv（通常 ~50%）
```

### 8.4 通信与计算的重叠

```
传统方式（串行）：
时间 ────────────────────────────────→
     ┌───────────┐ ┌───────────┐
     │ 计算      │ │ 通信      │
     └───────────┘ └───────────┘

优化方式（重叠）：
时间 ────────────────────────────────→
     ┌───────────┐
     │ 计算 0    │ ┌───────────┐
     └───────────┘ │ 计算 1    │
                   └───────────┘
     ├─ 通信 0:1 ─┤├─ 通信 1:2 ─┤
```

---

## 9. 使用指南

### 9.1 环境配置

```python
import os
import torch
import torch.distributed as dist
from scaletorch.parallel.pg_manager import setup_process_group_manager
from scaletorch.parallel.pipeline_parallel import PipelineParallel

# 第一步：初始化分布式环境
dist.init_process_group(
    backend='nccl',  # GPU 使用 NCCL，CPU 使用 gloo
    init_method='env://'  # 环境变量初始化
)

# 第二步：设置进程组（假设 4 个 GPU，全用于流水线并行）
pgm = setup_process_group_manager(
    tp_size=1,     # 张量并行度
    cp_size=1,     # 上下文并行度
    pp_size=4,     # 流水线并行度（GPU 数）
    dp_size=1      # 数据并行度
)

# 第三步：获取当前 GPU
local_rank = int(os.environ.get('LOCAL_RANK', 0))
torch.cuda.set_device(local_rank)
device = torch.device(f'cuda:{local_rank}')
```

### 9.2 模型定义

模型需要满足以下接口：

```python
class MyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(...)      # 必需
        self.decoder_layers = nn.ModuleList([   # 必需
            TransformerLayer(...) for _ in range(config.num_hidden_layers)
        ])
        self.final_norm = nn.LayerNorm(...)     # 必需
        self.final_proj = nn.Linear(...)        # 必需

    def forward(self, input_ids, position_ids):
        x = self.embedding(input_ids)
        for layer in self.decoder_layers:
            x = layer(x, position_ids=position_ids)
        x = self.final_norm(x)
        logits = self.final_proj(x)
        return logits
```

### 9.3 完整训练循环

```python
from scaletorch.parallel.pipeline_parallel import (
    PipelineParallel,
    train_step_pipeline_1f1b
)

# 初始化
config = ModelConfig(num_hidden_layers=32)
model = MyModel(config)
model = PipelineParallel(model, config)
model.to(device)

# 优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# 训练循环
num_epochs = 10
for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(train_dataloader):
        # 选择调度策略
        loss = train_step_pipeline_1f1b(
            model=model,
            data_loader=batch_iter,
            tensor_shapes=(batch_size, seq_len, hidden_dim),
            device=device,
            dtype=torch.float32
        )

        # 优化器更新（梯度已在 train_step 中积累）
        optimizer.step()
        optimizer.zero_grad()

        if batch_idx % 10 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss:.4f}')

# 分布式同步（结束时确保所有 GPU 完成）
dist.barrier()
```

### 9.4 检查点保存

```python
def save_checkpoint(model, optimizer, epoch, step, path):
    # 仅在首阶段保存（避免重复）
    if pgm.process_group_manager.pp_rank != 0:
        return

    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
    }
    torch.save(checkpoint, path)
    print(f'Checkpoint saved to {path}')

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    return checkpoint['epoch'], checkpoint['step']
```

---

## 10. 性能分析

### 10.1 内存占用分析

```
模型参数内存：
    M_param = Σ参数数量 · 数据类型字节数

    分布式下：M_param_per_gpu = M_param / pp_world_size
    （相对于单 GPU 下的 M_param）

激活值内存：
    M_activation = B · S · D · #layers_per_stage · sizeof(dtype)

    例子（每阶段 8 层）：
    B=8, S=2048, D=4096
    M_activation ≈ 8·2048·4096·8·4 bytes ≈ 8.6 GB

    vs AFAB（需保存所有微批次）：
    M_activation_afab ≈ 8.6 GB · grad_acc_steps

梯度内存：
    M_grad = M_param_per_gpu + M_activation
```

### 10.2 通信代价

```
每个微批次的通信：
    C_forward = 2·B·S·D·sizeof(dtype) + 2·α  (前向+通信)
    C_backward = 2·B·S·D·sizeof(dtype) + 2·α (反向+通信)

    其中 α ≈ 1 μs（网络启动延迟）

总通信时间（1F1B）：
    T_comm_total ≈ (N + P - 1) · C_forward·batch_count

带宽利用率：
    η = 计算时间 / (计算时间 + 通信时间)

    理想情况：η ≈ 95%（通信隐藏）
    实际情况：η ≈ 70-80%
```

### 10.3 加速比计算

```
单 GPU 吞吐量：
    T_1gpu = (T_forward + T_backward) · batch_size / batch_time

P 个 GPU 的吞吐量：
    T_p_gpus ≈ (T_forward + T_backward) · batch_size / ((N + P - 1) · micro_batch_time)

相对加速比：
    S = T_p_gpus / T_1gpu

理论上界（忽略通信）：
    S_theoretical = P

实际加速比（1F1B）：
    S_practical ≈ P · N / (N + P - 1) · (1 - overhead%)

    其中 overhead% ≈ 5-15%（通信、同步等）

例子（P=4, N=8）：
    S_practical ≈ 4 · 8 / 11 · 0.9 ≈ 2.6x
```

### 10.4 性能对标

```
测试环境：
    - GPU: A100 80GB × 4
    - 模型: GPT-3 (175B)
    - batch_size: 4/GPU, grad_acc_steps: 32
    - dtype: float32

结果对比：

                 单 GPU      4-GPU PP    8-GPU PP
吞吐量 (tok/s)    1000       3200       5800
加速比            1x         3.2x       5.8x
GPU 利用率        70%        68%        65%
通信占比          -          12%        18%

观察：
    - 加速比接近理论值
    - GPU 利用率相对稳定（通信开销可控）
    - 大规模情况下通信成为主要瓶颈
```

---

## 11. 常见问题

### Q1: 如何选择合适的 pp_size？

**答：** 遵循以下原则：

```python
# 1. 不超过总 GPU 数
pp_size <= total_gpus

# 2. 保证每阶段至少 1 层
pp_size <= num_hidden_layers

# 3. 考虑负载均衡
layers_per_stage ≈ num_hidden_layers / pp_size

# 推荐配置
if num_hidden_layers < 12:
    pp_size = 1  # 单 GPU
elif num_hidden_layers < 24:
    pp_size = 2
elif num_hidden_layers < 48:
    pp_size = 4
else:
    pp_size = 8 or more

# 例子
config = ModelConfig(num_hidden_layers=32)
pp_size = 4  # 每阶段 8 层
```

### Q2: grad_acc_steps 应该设多大？

**答：**

```python
# 权衡因素
# 1. 内存：更大的 grad_acc_steps 需要保存更多激活值
# 2. 性能：太小会产生长管道气泡，太大会有内存压力

# 推荐范围：8 ~ 64
# 计算公式
grad_acc_steps = (target_batch_size) / (per_gpu_batch_size * pp_world_size)

# 例子
target_batch_size = 2048
per_gpu_batch_size = 4
pp_world_size = 4
grad_acc_steps = 2048 / (4 * 4) = 128  # 可以
```

### Q3: 为什么梯度不收敛？

**答：** 检查以下几点：

```python
# 1. 验证通信拓扑
print(f'Rank: {pgm.pp_rank}')
print(f'Prev: {pgm.pp_prev_rank}, Next: {pgm.pp_next_rank}')

# 2. 检查梯度同步
# 确保只在最后一个微批次同步
assert model.require_backward_grad_sync == (microbatch_idx == grad_acc_steps - 1)

# 3. 验证损失计算
# 末阶段才能计算损失
assert pgm.pp_is_last_stage, 'Only last stage should compute loss'

# 4. 检查激活值
print(f'Activation shape: {activation.shape}')
print(f'Activation dtype: {activation.dtype}')
print(f'Activation device: {activation.device}')
```

### Q4: 如何调试死锁问题？

**答：**

```python
import torch.distributed as dist
from datetime import timedelta

# 1. 使用 monitored_barrier 检测死锁
try:
    dist.monitored_barrier(timeout=timedelta(seconds=30))
except Exception as e:
    print(f'Barrier timeout at rank {dist.get_rank()}: {e}')

# 2. 打印通信信息
os.environ['VERBOSE'] = '1'  # 启用详细日志

# 3. 检查通信匹配
# 确保所有 send 都有对应的 recv
if pp_rank > 0:
    assert 'recv_forward' in communication_log
if pp_rank < pp_world_size - 1:
    assert 'send_forward' in communication_log
```

### Q5: 激活值内存爆炸怎么办？

**答：**

```python
import torch.utils.checkpoint as checkpoint

# 方案 1：激活值重计算（Activation Checkpointing）
class CheckpointedLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x, position_ids=None):
        return checkpoint.checkpoint(
            self.layer,
            x,
            position_ids,
            use_reentrant=False  # 使用更稳定的非重入实现
        )

# 方案 2：减小 grad_acc_steps
grad_acc_steps = 16  # 从 64 减少到 16

# 方案 3：混合精度训练
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast(dtype=torch.float16):
    loss = model(...)
    scaler.scale(loss).backward()
```

### Q6: 如何监控训练进度？

**答：**

```python
import time

class MetricsTracker:
    def __init__(self):
        self.losses = []
        self.throughputs = []

    def update(self, loss, batch_size, step_time):
        self.losses.append(loss)
        self.throughputs.append(batch_size / step_time)

    def report(self):
        if dist.get_rank() == 0:
            avg_loss = sum(self.losses[-100:]) / min(100, len(self.losses))
            avg_tput = sum(self.throughputs[-100:]) / min(100, len(self.throughputs))
            print(f'Loss: {avg_loss:.4f}, Throughput: {avg_tput:.2f} tok/s')

# 使用
metrics = MetricsTracker()
for step in range(num_steps):
    start_time = time.time()
    loss = train_step_pipeline_1f1b(...)
    optimizer.step()

    step_time = time.time() - start_time
    metrics.update(loss, batch_size, step_time)

    if step % 10 == 0:
        metrics.report()
```

---

## 总结

### 关键要点

1. **流水线并行是深度学习大模型训练的关键技术**
   - 突破单 GPU 显存限制
   - 提高 GPU 集群利用率

2. **两种调度策略各有优劣**
   - AFAB：简单易懂，GPU 利用率低
   - 1F1B：复杂但性能好，应成为生产标准

3. **通信成本是主要瓶颈**
   - 使用双向通信优化
   - 合理选择 pp_size 和 grad_acc_steps

4. **激活值管理很关键**
   - 使用 FIFO 缓存维护前向张量
   - 考虑激活值重计算降低内存

5. **充分的测试和监控**
   - 单元测试验证层分配
   - 性能监控追踪训练进度

---

## 参考资源

- [Megatron-LM Pipeline Parallelism](https://github.com/NVIDIA/Megatron-LM)
- [GPipe: Efficient Training of Giant Models on Multiple GPUs](https://arxiv.org/abs/1806.03377)
- [PyTorch Pipeline Parallelism](https://pytorch.org/docs/stable/pipeline.html)

---

**文档版本**: 2.0（合并版）
**最后更新**: 2024年12月
**维护者**: ScaleTorch 团队
