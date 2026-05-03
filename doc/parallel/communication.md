# 通信原语 (Communication Primitives)

## 概述

ScaleTorch 通过 `scaletorch.dist` 模块提供统一的分布式通信原语接口。所有函数在非分布式环境下安全降级（直接返回或跳过），支持多种后端（NCCL、Gloo、HCCL 等）。核心文件：

- `scaletorch/dist/dist.py` — 集体通信操作
- `scaletorch/dist/utils.py` — 环境初始化、秩查询、设备工具

## 快速使用

```python
from scaletorch.dist import all_reduce, all_gather, broadcast, barrier

# 所有函数在非分布式环境自动降级
all_reduce(tensor, op='sum', group=dp_group)
result_list = all_gather(tensor, group=tp_group)
broadcast(tensor, src=0)
barrier()
```

## 集体通信操作

### AllReduce

所有秩上的张量规约，结果返回到所有秩。

```python
from scaletorch.dist import all_reduce

all_reduce(data: torch.Tensor,
           op: str = 'sum',
           group: Optional[ProcessGroup] = None) -> None
```

**参数：**
- `data`: 待规约张量（原地修改）
- `op`: 规约操作，支持 `'sum'`、`'product'`、`'min'`、`'max'`、`'band'`、`'bor'`、`'bxor'`、`'mean'`
- `group`: 通信组，默认全局组

**示例：**

```python
from scaletorch.dist import all_reduce
from scaletorch.parallel.pg_manager import process_group_manager as pgm

# 梯度同步
all_reduce(grad, op='mean', group=pgm.dp_group)
```

### AllGather

收集所有秩的张量到列表。

```python
from scaletorch.dist import all_gather

all_gather(data: torch.Tensor,
           group: Optional[ProcessGroup] = None) -> List[torch.Tensor]
```

**返回值：** 所有秩张量的列表，长度等于 `world_size`

**示例：**

```python
# 收集 TP 分割的权重
parts = all_gather(local_weight, group=pgm.tp_group)
full_weight = torch.cat(parts, dim=-1)
```

### Gather

收集所有秩的张量到目标秩。

```python
from scaletorch.dist import gather

gather(data: torch.Tensor,
       dst: int = 0,
       group: Optional[ProcessGroup] = None) -> List[Optional[torch.Tensor]]
```

**返回值：** 目标秩返回完整列表，其他秩返回 `None` 列表

### Broadcast

从源秩广播张量到所有秩。

```python
from scaletorch.dist import broadcast

broadcast(data: torch.Tensor,
          src: int = 0,
          group: Optional[ProcessGroup] = None) -> None
```

**使用场景：** 模型权重初始化同步、配置参数分发

### Scatter

从源秩分散张量到所有秩。

```python
from scaletorch.dist import scatter

scatter(tensor_out: torch.Tensor,
        scatter_list: Optional[List[torch.Tensor]] = None,
        src: int = 0,
        group: Optional[ProcessGroup] = None) -> None
```

### Reduce

规约张量到目标秩。

```python
from scaletorch.dist import reduce

reduce(data: torch.Tensor,
       dst: int = 0,
       op: str = 'sum',
       group: Optional[ProcessGroup] = None) -> None
```

### Reduce-Scatter

规约后分散结果到所有秩。

```python
from scaletorch.dist import reduce_scatter

reduce_scatter(tensor_out: torch.Tensor,
               input_list: Optional[List[torch.Tensor]] = None,
               op: str = 'sum',
               group: Optional[ProcessGroup] = None) -> None
```

### All-to-All

每个秩发送不同数据到不同目标。

```python
from scaletorch.dist import all_to_all

all_to_all(output_tensor_list: List[torch.Tensor],
           input_tensor_list: List[torch.Tensor],
           group: Optional[ProcessGroup] = None) -> List[torch.Tensor]
```

## 对象通信

用于非张量数据（如配置字典、字符串）的通信操作。

### broadcast_object_list

```python
from scaletorch.dist import broadcast_object_list

broadcast_object_list(data: List[Any],
                      src: int = 0,
                      group: Optional[ProcessGroup] = None) -> None
```

广播可序列化对象列表。常用于广播 tokenizer。

### all_gather_object

```python
from scaletorch.dist import all_gather_object

all_gather_object(data: Any,
                  group: Optional[ProcessGroup] = None) -> List[Any]
```

收集所有秩的可序列化对象。

### gather_object

```python
from scaletorch.dist import gather_object

gather_object(data: Any,
              dst: int = 0,
              group: Optional[ProcessGroup] = None) -> Optional[List[Any]]
```

收集可序列化对象到目标秩。

## 专用操作

### all_reduce_dict

对字典中所有张量分别执行 AllReduce。

```python
from scaletorch.dist import all_reduce_dict

all_reduce_dict(data: Dict[str, torch.Tensor],
                op: str = 'sum',
                group: Optional[ProcessGroup] = None) -> None
```

### all_reduce_params

对参数列表执行分桶 AllReduce，支持合并和分桶优化。

```python
from scaletorch.dist import all_reduce_params

all_reduce_params(params: Union[List, Generator],
                  coalesce: bool = True,
                  bucket_size_mb: int = -1,
                  op: str = 'sum',
                  group: Optional[ProcessGroup] = None) -> None
```

### sync_random_seed

同步随机种子到所有进程。

```python
from scaletorch.dist import sync_random_seed

seed = sync_random_seed(group=None) -> int
```

### collect_results

分布式结果收集，自动根据设备类型选择策略。

```python
from scaletorch.dist import collect_results

collect_results(results: list,
                size: int,
                device: str = 'cpu',
                tmpdir: Optional[str] = None) -> Optional[list]
```

## 环境工具 (`scaletorch.dist.utils`)

### 分布式初始化

```python
from scaletorch.dist import init_dist, infer_launcher

# 自动检测启动方式
launcher = infer_launcher()  # 'pytorch' | 'slurm' | 'mpi' | 'none'

# 初始化分布式环境
init_dist(launcher, backend='nccl')
```

### 秩查询

```python
from scaletorch.dist import (
    get_rank, get_world_size, get_local_rank,
    is_main_process, is_distributed, barrier
)

if is_distributed():
    rank = get_rank()
    world_size = get_world_size()
    local_rank = get_local_rank()

if is_main_process():
    print("Only rank 0 executes this")
```

### 设备工具

```python
from scaletorch.dist import get_comm_device, cast_data_device

# 获取通信设备（NCCL→CUDA, HCCL→NPU）
device = get_comm_device()

# 递归转换数据中的张量设备
data = cast_data_device(data, target_device)
```

## 操作字符串速查

`op` 参数支持以下字符串值：

| 字符串 | 对应操作 |
|--------|----------|
| `'sum'` | 求和（默认） |
| `'mean'` | 求平均 |
| `'product'` | 求积 |
| `'min'` | 取最小 |
| `'max'` | 取最大 |
| `'band'` | 按位与 |
| `'bor'` | 按位或 |
| `'bxor'` | 按位异或 |

## 通信复杂度

| 操作 | 通信量 | 说明 |
|------|--------|------|
| AllReduce | O(P) | P = 张量元素数 |
| AllGather | O(P × N) | N = world_size |
| Broadcast | O(P) | 仅源发送 |
| Reduce | O(P) | 仅目标接收 |
| Reduce-Scatter | O(P) | AllReduce + 分散 |
| All-to-All | O(P) | 每对收发 P/N |
| Send/Recv | O(P) | 点对点 |

## 参考资源

- [PyTorch Distributed API](https://pytorch.org/docs/stable/distributed.html)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/)
