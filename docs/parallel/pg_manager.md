# 进程组管理器 (Process Group Manager)

## 概述

`ProcessGroupManager` 是 ScaleTorch 的核心组件，负责在分布式训练中管理不同并行策略的进程组。通过统一的 4D 网格结构，它自动创建和管理所有必要的进程组。

## 4D 网格结构

### 定义

```text
Grid[DP_rank, PP_rank, CP_rank, TP_rank]
```

其中每个维度范围：

- DP_rank: 0 ~ DP_size-1
- PP_rank: 0 ~ PP_size-1
- CP_rank: 0 ~ CP_size-1
- TP_rank: 0 ~ TP_size-1

### 进程组类型

系统自动创建 7 个进程组：

| 进程组 | 维度保持一致    | 用途             |
| ------ | --------------- | ---------------- |
| TP     | DP, PP, CP 相同 | 张量并行通信     |
| CP     | DP, PP, TP 相同 | 上下文并行通信   |
| PP     | DP, CP, TP 相同 | 流水线并行通信   |
| DP     | PP, CP, TP 相同 | 数据并行梯度同步 |
| CP-DP  | PP, TP 相同     | 联合 CP+DP 操作  |
| PP-DP  | CP, TP 相同     | 联合 PP+DP 操作  |
| WORLD  | 所有秩          | 全局操作         |

## 初始化和属性

### 初始化

```python
from scaletorch.parallel.pg_manager import ProcessGroupManager

pgm = ProcessGroupManager(
    tp_size=4,   # 张量并行大小
    cp_size=2,   # 上下文并行大小
    pp_size=2,   # 流水线并行大小
    dp_size=2    # 数据并行大小
)
```

### 主要属性

**进程坐标：**

```python
pgm.global_rank        # 全局秩 (0 ~ world_size-1)
pgm.world_size         # 进程总数
pgm.local_rank         # 节点内秩
pgm.dp_rank, pgm.pp_rank, pgm.cp_rank, pgm.tp_rank
```

**张量并行属性：**

```python
pgm.tp_group           # 进程组对象
pgm.tp_world_size      # 组大小
pgm.tp_first_rank      # 组内首秩
pgm.tp_last_rank       # 组内末秩
pgm.tp_group_ids       # 组内所有秩
```

**上下文并行属性 (Ring)：**

```python
pgm.cp_group
pgm.cp_world_size
pgm.cp_send_rank       # 发送给这个秩
pgm.cp_recv_rank       # 从这个秩接收
pgm.cp_first_rank, pgm.cp_last_rank
```

**流水线并行属性：**

```python
pgm.pp_group
pgm.pp_is_first_stage  # 是否为首阶段
pgm.pp_is_last_stage   # 是否为末阶段
pgm.pp_prev_rank       # 前驱秩 (None 如果为首)
pgm.pp_next_rank       # 后继秩 (None 如果为末)
pgm.pp_first_rank, pgm.pp_last_rank
```

**数据并行属性：**

```python
pgm.dp_group
pgm.dp_world_size
pgm.dp_first_rank, pgm.dp_last_rank
pgm.dp_group_ids
```

## 使用示例

### 示例 1: 基本初始化

```python
import torch.distributed as dist
from scaletorch.parallel.pg_manager import setup_process_group_manager

# 初始化分布式
dist.init_process_group(backend='nccl')

# 设置进程组管理器
# 需要 4*1*2*2=16 个进程
pgm = setup_process_group_manager(
    tp_size=4, cp_size=1, pp_size=2, dp_size=2
)

# 获取配置信息
print(pgm.get_info())
```

### 示例 2: 流水线并行通信

```python
import torch

# 准备张量
batch_size, seq_len, hidden_dim = 2, 1024, 768
tensor = torch.randn(batch_size, seq_len, hidden_dim, device='cuda')

# 流水线通信
if pgm.pp_is_first_stage:
    # 发送给下一阶段
    dist.send(tensor, pgm.pp_next_rank)
else:
    # 接收来自前一阶段的数据
    dist.recv(tensor, pgm.pp_prev_rank)
```

### 示例 3: Ring 拓扑

```python
# 在 CP Ring 中进行通信
from scaletorch.parallel.context_parallel.cp_comms import ContextCommunicate

comm = ContextCommunicate('ring_operation')

for step in range(pgm.cp_world_size):
    if step > 0:
        # 从上一个秩接收
        dist.recv(data, pgm.cp_recv_rank)

    # 处理数据
    process_data(data)

    # 发送给下一个秩
    if step < pgm.cp_world_size - 1:
        dist.send(data, pgm.cp_send_rank)
```

## 全局函数

### setup_process_group_manager()

设置全局进程组管理器实例。

```python
from scaletorch.parallel.pg_manager import setup_process_group_manager

pgm = setup_process_group_manager(tp_size=4, cp_size=1, pp_size=2, dp_size=1)
```

### get_process_group_manager()

获取全局进程组管理器实例。

```python
from scaletorch.parallel.pg_manager import get_process_group_manager

pgm = get_process_group_manager()
if pgm is not None:
    print(f"Current rank: {pgm.global_rank}")
```

## 最佳实践

1. **配置约束验证：**

   ```python
   assert pgm.tp_size * pgm.cp_size * pgm.pp_size * pgm.dp_size == pgm.world_size
   ```

2. **秩信息调试：**

   ```python
   print(f"Global rank: {pgm.global_rank}")
   print(f"Grid position: DP={pgm.dp_rank}, PP={pgm.pp_rank}, "
         f"CP={pgm.cp_rank}, TP={pgm.tp_rank}")
   ```

3. **通信验证：**

   ```python
   # 验证 Ring 拓扑
   print(f"Ring: {pgm.cp_recv_rank} -> {pgm.global_rank} -> {pgm.cp_send_rank}")

   # 验证流水线拓扑
   if not pgm.pp_is_first_stage:
       print(f"Prev stage: {pgm.pp_prev_rank}")
   if not pgm.pp_is_last_stage:
       print(f"Next stage: {pgm.pp_next_rank}")
   ```

## 常见问题

### Q1: 进程组大小不匹配错误？

```text
ValueError: World size (32) != TP (4) * CP (2) * PP (2) * DP (2) = 32
```

**解决方案：** 确保总进程数等于四个并行大小的乘积。

### Q2: 如何调试通信死锁？

```python
# 添加日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 验证秩的一致性
print(f"Rank {pgm.global_rank}: TP={pgm.tp_rank}, CP={pgm.cp_rank}")
```

### Q3: 跨节点通信性能差？

```python
# 优先使用节点内进程
# 设置 TP 和 CP 在同一节点内
if pgm.local_rank < 4:
    # 这些秩在同一节点
    pass
```
