# 通信原语 (Communication Primitives)

## 概述

ScaleTorch 通过一组底层通信原语支持各种并行策略。这些原语包括 AllReduce、AllGather、Reduce-Scatter、Send/Receive 等集体操作，为张量并行、流水线并行、数据并行等并行策略提供基础。

## 核心通信操作

### AllReduce

所有秩上的张量求和并将结果返回到所有秩。

```python
def all_reduce(tensor: torch.Tensor,
               op: dist.ReduceOp = dist.ReduceOp.SUM,
               group: Optional[dist.ProcessGroup] = None) -> torch.Tensor:
    """
    对所有秩的张量进行 AllReduce 操作。

    参数：
        tensor: 待操作张量
        op: 操作类型 (SUM, PROD, MAX, MIN)
        group: 通信组

    返回：
        AllReduce 结果张量
    """
```

**使用场景：**
- 梯度同步（数据并行）
- 列并行线性层输出合并
- 模型参数验证

**示例：**

```python
# 梯度同步
dist.all_reduce(gradient, op=dist.ReduceOp.SUM, group=dp_group)
gradient /= world_size
```

### AllGather

收集所有秩的张量到所有秩。

```python
def all_gather(tensor: torch.Tensor,
               group: Optional[dist.ProcessGroup] = None) -> List[torch.Tensor]:
    """
    从所有秩收集张量。

    参数：
        tensor: 本地张量
        group: 通信组

    返回：
        所有秩的张量列表
    """
```

**使用场景：**
- 张量并行权重收集
- 分布式梯度聚合
- 模型检查点

**示例：**

```python
# 收集分割的权重
tensor_list = [torch.empty_like(local_tensor) for _ in range(world_size)]
dist.all_gather(tensor_list, local_tensor, group=tp_group)
full_weight = torch.cat(tensor_list, dim=-1)
```

### Reduce-Scatter

对所有秩的张量求和，然后将结果分散到各秩。

```python
def reduce_scatter(output: torch.Tensor,
                   input_list: List[torch.Tensor],
                   op: dist.ReduceOp = dist.ReduceOp.SUM,
                   group: Optional[dist.ProcessGroup] = None):
    """
    AllReduce + Scatter 操作。

    参数：
        output: 输出张量
        input_list: 输入张量列表（仅用于本地排列）
        op: 操作类型
        group: 通信组
    """
```

**使用场景：**
- 行并行线性层梯度同步
- 高效的分布式操作

### Broadcast

从一个秩广播张量到所有秩。

```python
def broadcast(tensor: torch.Tensor,
              src: int,
              group: Optional[dist.ProcessGroup] = None):
    """
    从指定秩广播张量。

    参数：
        tensor: 张量（源秩上为发送数据）
        src: 源秩
        group: 通信组
    """
```

**使用场景：**
- 模型权重初始化同步
- 配置参数分发
- 随机数生成器状态同步

### Send/Receive

点对点通信。

```python
def send(tensor: torch.Tensor, dst: int) -> None:
    """发送张量到目标秩"""

def recv(tensor: torch.Tensor, src: int) -> None:
    """从源秩接收张量"""

def isend(tensor: torch.Tensor, dst: int) -> dist.Work:
    """异步发送"""

def irecv(tensor: torch.Tensor, src: int) -> dist.Work:
    """异步接收"""
```

**使用场景：**
- 流水线并行激活传递
- 上下文并行 Ring 通信
- 灵活的点对点通信

**示例：**

```python
# 流水线并行前向传播
if not pp_is_last_stage:
    dist.send(activation, dst=pp_next_rank)
else:
    dist.recv(activation, src=pp_prev_rank)
```

## 异步通信

### 异步 AllReduce

```python
# 启动异步操作
handle = dist.all_reduce(grad, async_op=True)

# 进行计算
next_loss = model(next_batch)
next_loss.backward()

# 等待完成
handle.wait()
```

**优势：**
- 通信与计算重叠
- 减少总体训练时间
- 提高 GPU 利用率

### 异步 Send/Receive

```python
# 异步发送和接收
send_handle = dist.isend(send_tensor, dst)
recv_handle = dist.irecv(recv_tensor, src)

# 执行计算
output = layer(input)

# 等待通信完成
send_handle.wait()
recv_handle.wait()
```

## 通信模式

### All-to-All

每个秩发送数据到所有其他秩。

```python
def all_to_all(send_list: List[torch.Tensor],
               recv_list: List[torch.Tensor],
               group: Optional[dist.ProcessGroup] = None):
    """
    All-to-All 集体操作。

    使用场景：
    - 序列长度并行的令牌交换
    - 完整洗牌操作
    """
```

**示例：**

```python
# CP 中的令牌交换
send_list = [local_seq[i::cp_size] for i in range(cp_size)]
recv_list = [torch.empty_like(send_list[0]) for _ in range(cp_size)]
dist.all_to_all(send_list, recv_list, group=cp_group)
```

### Tree Reduction

分层 Reduce 操作，减少通信轮次。

```
       秩0
      /  \
    秩1  秩2
    / \  / \
秩3 秩4 秩5 秩6
```

## 进程组管理

### 创建通信组

```python
from scaletorch.parallel.pg_manager import ProcessGroupManager

pgm = ProcessGroupManager(tp_size=4, cp_size=2, pp_size=2, dp_size=1)

# 张量并行组
tp_group = pgm.tp_group

# 数据并行组
dp_group = pgm.dp_group

# 自定义组
custom_group = dist.new_group([0, 2, 4, 6])
```

### 组的特性

| 组    | 维度保持一致 | 用途             |
| ----- | ------------ | ---------------- |
| TP    | DP, PP, CP   | 张量并行通信     |
| CP    | DP, PP, TP   | 上下文并行 Ring  |
| PP    | DP, CP, TP   | 流水线并行通信   |
| DP    | PP, CP, TP   | 数据并行梯度同步 |
| CP-DP | PP, TP       | 联合 CP+DP       |
| PP-DP | CP, TP       | 联合 PP+DP       |

## 通信复杂度分析

### Bandwidth Scaling

```
N 个秩的 AllReduce 通信量：
- 合并（Reduce）：log₂(N) 轮
- 广播（Broadcast）：log₂(N) 轮
- 总计：2·log₂(N) 轮

对于 N = 8：
- 单元素 AllReduce：6 轮
- 张量大小 P：总通信 6·P
```

### Latency vs. Bandwidth

```
通信时间 = α·log(N) + β·(P/BW)

其中：
- α: 通信延迟（每跳）
- log(N): 通信轮次
- β: 字节数
- BW: 网络带宽

优化策略：
- 减少通信轮次：使用树形拓扑
- 增加数据量：梯度积累、分组通信
- 隐藏延迟：异步通信、通信与计算重叠
```

## 最佳实践

1. **选择合适的通信操作：**

   ```python
   # 不好：多个 AllReduce
   for param in model.parameters():
       dist.all_reduce(param.grad)

   # 好：梯度堆叠后一次 AllReduce
   grads = torch.cat([p.grad.flatten() for p in model.parameters()])
   dist.all_reduce(grads)
   ```

2. **使用异步通信：**

   ```python
   # 启动异步操作
   handle = dist.all_reduce(grad, async_op=True)

   # 同时进行计算
   output = next_layer(input)

   # 等待完成
   handle.wait()
   ```

3. **通信与计算重叠：**

   ```python
   # 分层通信
   for i, (name, param) in enumerate(model.named_parameters()):
       # 开始通信前一层的梯度
       if i > 0:
           handles[i-1].wait()

       # 计算当前层梯度
       param.grad.backward()

       # 异步通信当前层梯度
       handles[i] = dist.all_reduce(param.grad, async_op=True)
   ```

4. **通信友好的数据排列：**

   ```python
   # 确保张量连续
   tensor = tensor.contiguous()
   dist.all_reduce(tensor)

   # 合并多个张量减少通信次数
   all_grads = torch.cat([p.grad for p in model.parameters()])
   ```

## 调试和监控

### 通信调试

```python
import os
os.environ['NCCL_DEBUG'] = 'INFO'  # 启用 NCCL 调试输出
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'  # PyTorch 分布式调试
```

### 性能监控

```python
import time
import torch.distributed as dist

# 测量 AllReduce 延迟
torch.cuda.synchronize()
start = time.time()

dist.all_reduce(tensor)

torch.cuda.synchronize()
elapsed = time.time() - start

print(f"AllReduce latency: {elapsed*1000:.2f} ms")
print(f"Throughput: {tensor.numel() * 4 / elapsed / 1e9:.2f} GB/s")
```

## 参考资源

- [PyTorch Distributed API](https://pytorch.org/docs/stable/distributed.html)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/)
- [Ring AllReduce Algorithm](https://tech.preferred.jp/en/blog/technologies/broadcast-allreduce/)
