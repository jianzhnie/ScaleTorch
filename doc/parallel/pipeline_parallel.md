# 流水线并行 (Pipeline Parallelism)

## 概述

流水线并行通过将模型层分布到多个 GPU 上实现。每个 GPU 处理一个模型的子集，数据流以流水线方式流经各个阶段，从而提高 GPU 利用率和训练效率。

### 核心思想

**传统方式（AFAB）：** 所有 GPU 的前向传播完成 → 所有 GPU 的反向传播完成

**优化方式（1F1B）：** 前向和反向传播交错执行，提高 GPU 利用率

## 基本概念

### 管道阶段 (Pipeline Stages)

每个 GPU 上运行一个管道阶段，处理分配给该阶段的模型层。

```text
GPU-0 (Stage-0):    Embedding + Layers[0:4]
                           |
GPU-1 (Stage-1):    Layers[4:8]
                           |
GPU-2 (Stage-2):    Layers[8:12] + Final Projection
```

### 通信模式

- **前向传播：** 激活值从前一阶段发送到后一阶段
- **反向传播：** 梯度从后一阶段发送回前一阶段

## PipelineParallel 类

```python
class PipelineParallel(nn.Module):
    """
    实现流水线并行的核心类。

    将模型层分布到不同 GPU 上，每个 GPU 处理一个子集。
    支持 AFAB 和 1F1B 两种调度策略。
    """
```

### 关键属性

```python
layer_distribution    # 分配给该阶段的层索引列表
embedding            # 嵌入层（仅首阶段）
decoder_layers       # 分配给该阶段的解码层
final_norm           # 最终归一化（仅末阶段）
final_proj           # 最终投影（仅末阶段）
```

### 层分配策略

```python
def _distribute_layers(self, num_layers: int) -> List[int]:
    """
    均匀分配模型层到各流水线阶段。

    示例（10层，3个阶段）：
    - Stage-0: [0, 1, 2, 3]  (4层)
    - Stage-1: [4, 5, 6]     (3层)
    - Stage-2: [7, 8, 9]     (3层)
    """
```

## 前向传播

### 首阶段 (First Stage)

```python
# 输入来自数据加载器
x = batch['input_ids']

# 嵌入和处理
embeddings = self.embedding(x)

# 通过分配的层
for layer_idx in self.layer_distribution:
    embeddings = self.decoder_layers[str(layer_idx)](embeddings)

# 发送到下一阶段
pipeline_communicate(embeddings, self.pp_next_rank)
```

### 中间阶段 (Intermediate Stage)

```python
# 接收来自前一阶段的激活
x = pipeline_communicate(None, self.pp_prev_rank)

# 处理分配的层
for layer_idx in self.layer_distribution:
    x = self.decoder_layers[str(layer_idx)](x)

# 转发到下一阶段
pipeline_communicate(x, self.pp_next_rank)
```

### 末阶段 (Last Stage)

```python
# 接收来自前一阶段的激活
x = pipeline_communicate(None, self.pp_prev_rank)

# 处理分配的层
for layer_idx in self.layer_distribution:
    x = self.decoder_layers[str(layer_idx)](x)

# 最终处理
x = self.final_norm(x)
logits = self.final_proj(x)

# 计算损失
loss = compute_loss(logits, labels)
```

## 通信原语

### pipeline_communicate

```python
def pipeline_communicate(
    tensor: Optional[torch.Tensor],
    peer_rank: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    与相邻管道阶段通信激活值或梯度。

    参数：
        tensor: 要发送的张量（None 表示接收）
        peer_rank: 通信对象秩
        dtype: 数据类型

    返回：
        接收到的张量
    """
```

### bidirectional_pipeline_communicate

```python
def bidirectional_pipeline_communicate(
    send_tensor: Optional[torch.Tensor],
    recv_peer_rank: int,
    send_peer_rank: int,
) -> torch.Tensor:
    """
    同时发送和接收张量（优化通信）。

    在 1F1B 调度中用于同时发送梯度和接收新激活。
    """
```

## 调度策略

### AFAB (All-Forward-All-Backward)

```python
# 前向阶段
for batch in dataloader:
    # 所有 GPU 完成前向
    output = model(batch)

# 反向阶段
loss.backward()  # 所有 GPU 同时反向
optimizer.step()
```

**优点：** 简单，实现容易
**缺点：** GPU 利用率低，GPU 等待其他 GPU 完成

### 1F1B (One-Forward-One-Backward)

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

**优点：** 高 GPU 利用率，通信与计算重叠
**缺点：** 实现复杂，需要管理激活缓存

## 使用示例

### 示例 1: 基本流水线并行

```python
import torch
import torch.distributed as dist
from scaletorch.parallel.pg_manager import setup_process_group_manager
from scaletorch.parallel.pipeline_parallel import PipelineParallel
from scaletorch.model.model_llama import LlamaModel

# 初始化分布式环境
dist.init_process_group(backend='nccl')

# 设置进程组（4 个 GPU，2 个流水线阶段）
pgm = setup_process_group_manager(
    tp_size=1, cp_size=1, pp_size=4, dp_size=1
)

# 创建模型
config = LlamaConfig(num_hidden_layers=32)
model = LlamaModel(config)
model.to('cuda')

# 应用流水线并行
model = PipelineParallel(model, config)

# 训练循环
optimizer = torch.optim.AdamW(model.parameters())
for batch in dataloader:
    output = model(batch)
    loss = compute_loss(output, batch['labels'])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### 示例 2: 检查阶段分配

```python
pgm = setup_process_group_manager(pp_size=4)
model = PipelineParallel(model, config)

# 打印该阶段分配的层
print(f"Stage {pgm.pp_rank}: layers {model.layer_distribution}")

# 检查是否为首/末阶段
if pgm.pp_is_first_stage:
    print("This is the first stage (handles embedding)")
if pgm.pp_is_last_stage:
    print("This is the last stage (handles output projection and loss)")
```

### 示例 3: 检查通信拓扑

```python
pgm = setup_process_group_manager(pp_size=4)

# 前驱和后继阶段
if not pgm.pp_is_first_stage:
    print(f"Previous stage: rank {pgm.pp_prev_rank}")
if not pgm.pp_is_last_stage:
    print(f"Next stage: rank {pgm.pp_next_rank}")

# 线性拓扑验证
if pgm.pp_rank > 0 and pgm.pp_rank < pgm.pp_world_size - 1:
    print("This is an intermediate stage")
    print(f"Communication: {pgm.pp_prev_rank} -> {pgm.pp_rank} -> {pgm.pp_next_rank}")
```

## 性能特性

### 内存效率

```
单 GPU 内存：                完整模型大小
PP 分布到 P 个 GPU：        完整模型大小 / P + 激活缓存
```

### 通信开销

| 操作     | 通信量   | 说明                    |
| -------- | -------- | ----------------------- |
| 前向激活 | O(B·S·D) | B=batch, S=seq, D=dim   |
| 反向梯度 | O(B·S·D) | 同前向                  |
| 减速比   | ~20-30%  | 相对于单 GPU 的计算时间 |

### GPU 利用率

- **AFAB：** 50-60%（等待其他 GPU）
- **1F1B：** 80-90%（通信与计算重叠）

## 最佳实践

1. **层分配平衡：**

   ```python
   # 确保各阶段分配的层数相近
   layers_per_stage = num_layers / pp_size
   print(f"Expected: ~{layers_per_stage} layers per stage")
   print(f"Actual: {model.layer_distribution}")
   ```

2. **激活重计算（Activation Checkpointing）：**

   ```python
   import torch.utils.checkpoint as checkpoint

   # 仅保存部分激活，其他重计算
   for layer in model.decoder_layers:
       output = checkpoint.checkpoint(layer, input)
   ```

3. **张量大小对齐：**

   ```python
   # 确保张量形状一致性
   assert embeddings.shape == (batch_size, seq_len, hidden_dim)
   ```

4. **通信优化：**

   ```python
   # 使用双向通信减少同步
   from scaletorch.parallel.pipeline_parallel.pp_comms import (
       bidirectional_pipeline_communicate
   )
   ```

## 常见问题

### Q1: 前后阶段通信的延迟如何优化？

使用双向通信和异步操作，在一阶段反向传播时同时接收下一批数据的激活。

### Q2: 如何处理不均匀的层分配？

某些阶段可能分配到更多层。可以：
- 调整流水线大小
- 使用不同大小的批次
- 实现动态负载均衡

### Q3: 激活内存消耗很大怎么办？

```python
# 使用激活重计算
from torch.utils.checkpoint import checkpoint

# 在反向传播时重计算激活，减少内存占用
```

### Q4: 梯度不同步如何调试？

```python
# 验证通信拓扑
print(f"Rank: {pgm.pp_rank}")
print(f"Prev: {pgm.pp_prev_rank}, Next: {pgm.pp_next_rank}")

# 检查通信是否阻塞
dist.monitored_barrier(timeout=timedelta(seconds=30))
```

## 参考资源

- [Megatron-LM Pipeline Parallelism](https://github.com/NVIDIA/Megatron-LM)
- [GPipe: Efficient Training of Giant Models on Multiple GPUs](https://arxiv.org/abs/1806.03377)
- [PyTorch Pipeline Parallelism](https://pytorch.org/docs/stable/pipeline.html)
