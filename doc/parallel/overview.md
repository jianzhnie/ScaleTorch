# ScaleTorch 并行计算框架文档

## 概述

ScaleTorch 的并行模块提供了一套完整的分布式训练支持，包括四种主要的并行策略：
- **张量并行 (Tensor Parallelism, TP)**: 在模型的权重维度上进行并行化
- **上下文并行 (Context Parallelism, CP)**: 在序列长度维度上进行并行化
- **流水线并行 (Pipeline Parallelism, PP)**: 在模型深度维度上进行并行化
- **数据并行 (Data Parallelism, DP)**: 在批次维度上进行并行化

## 架构设计

### 4D 网格管理

所有并行策略通过一个 **4D 网格** 来管理，结构如下：

```bash
Grid[DP_rank, PP_rank, CP_rank, TP_rank]
```

其中每个维度代表不同的并行类型。系统自动为每个并行类型创建独立的进程组，实现精细化的通信管理。

### 核心组件

```bash
scaletorch/parallel/
├── pg_manager.py           # 进程组管理器 (核心)
├── context_parallel/       # 上下文并行
│   ├── context_parallel.py # Ring Attention 实现
│   └── cp_comms.py        # 通信原语
├── tensor_parallel/        # 张量并行
│   ├── tensor_parallel.py
│   └── tp_comms.py        # 通信原语
├── pipeline_parallel/      # 流水线并行
│   ├── pipeline_parallel.py
│   └── pp_comms.py        # 通信原语
└── data_parallel/          # 数据并行
    ├── data_parallel.py
    └── bucket.py          # 梯度分桶
```

## 快速开始

### 1. 初始化进程组管理器

```python
import torch.distributed as dist
from scaletorch.parallel.pg_manager import setup_process_group_manager

# 初始化分布式训练环境
dist.init_process_group(backend='nccl')

# 创建进程组管理器
# tp_size=2: 张量并行大小
# cp_size=2: 上下文并行大小
# pp_size=2: 流水线并行大小
# dp_size=1: 数据并行大小
pg_manager = setup_process_group_manager(tp_size=2, cp_size=2, pp_size=2, dp_size=1)

# 获取当前进程的配置信息
print(pg_manager.get_info())
```

### 2. 应用特定的并行策略

具体的应用方式请参考各模块的详细文档：

- [分布式通信原语 (Distribute Communication)](./communication.md)
- [张量并行 (Tensor Parallel)](./tensor_parallel.md)
- [上下文并行 (Context Parallel)](./context_parallel.md)
- [流水线并行 (Pipeline Parallel)](./pipeline_parallel.md)
- [数据并行 (Data Parallel)](./data_parallel.md)
- [进程组管理 (Process Group Manager)](./pg_manager.md)
- [梯度分桶 （Gradient Bucket）](./gradient_bucket.md)

## 并行策略对比

| 特性           | 张量并行 | 上下文并行 | 流水线并行 | 数据并行 |
| -------------- | -------- | ---------- | ---------- | -------- |
| **并行维度**   | 权重     | 序列长度   | 模型深度   | 批次     |
| **通信量**     | 中等     | 低         | 低         | 高       |
| **内存占用**   | 低       | 中等       | 低         | 高       |
| **适用场景**   | 大模型   | 长序列     | 超大模型   | 标准训练 |
| **实现复杂度** | 高       | 中等       | 高         | 低       |

## 设计理念

### 1. **模块化设计**
   - 每种并行策略相互独立，可灵活组合
   - 通过进程组管理器统一协调

### 2. **零碎通信优化**
   - Ring Attention 用于上下文并行，减少通信
   - 异步通信重叠计算和通信

### 3. **数值稳定性**
   - 使用 Online Softmax 保证数值精度
   - 使用 Stable Update 避免溢出

### 4. **易于扩展**
   - 清晰的接口定义
   - 支持组合多种并行策略

## 配置示例

### 场景 1: 超大模型 (1.7T 参数)

```python
# 使用多种并行策略组合
pg_manager = setup_process_group_manager(
    tp_size=8,   # 8x 张量并行 (水平分割)
    cp_size=4,   # 4x 上下文并行 (序列分割)
    pp_size=16,  # 16x 流水线并行 (垂直分割)
    dp_size=2    # 2x 数据并行
)
# 总GPU数: 8 * 4 * 16 * 2 = 1024
```

### 场景 2: 长序列模型

```python
# 优先使用上下文并行处理长序列
pg_manager = setup_process_group_manager(
    tp_size=2,   # 2x 张量并行
    cp_size=16,  # 16x 上下文并行 (处理长序列)
    pp_size=1,   # 无流水线并行
    dp_size=4    # 4x 数据并行
)
```

## 常见问题

### Q1: 如何选择合适的并行配置?
**A:**
- 模型尺寸很大 → 优先使用流水线并行
- 序列长度很长 → 使用上下文并行
- 模型适中但GPU数多 → 优先使用张量并行 + 数据并行

### Q2: 多种并行策略能否混合使用?
**A:** 可以，ScaleTorch 完全支持混合并行。四种策略之间完全兼容。

### Q3: 通信开销会不会很大?
**A:**
- 上下文并行通信最少 (Ring 拓扑)
- 张量并行通信适中 (AllReduce)
- 流水线并行通信最少 (点对点)
- 数据并行可视为baseline

## 参考资源

- [论文: Megatron-LM](https://arxiv.org/abs/1909.08053) - 张量并行和流水线并行
- [论文: Ring Attention](https://arxiv.org/abs/2310.01889) - 上下文并行
- [PyTorch Distributed](https://pytorch.org/docs/stable/distributed.html)

## 许可证

MIT License
