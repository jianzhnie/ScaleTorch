# 深入理解 PyTorch DTensor：从设计哲学到工程实践

> 目标读者：具备 PyTorch 分布式训练基础，希望深入理解 DTensor 设计原理与内部机制的工程师。

---

## 引言

在大模型训练时代，分布式并行策略已从"可选项"变为"必选项"。数据并行（DP）、张量并行（TP）、流水线并行（PP）、参数分片（FSDP/ZeRO）……每种策略都解决了特定维度的扩展问题，却也带来了新的复杂度：通信操作与计算代码深度耦合，不同并行方案之间难以组合，用户被迫在"手写通信逻辑"与"接受框架限制"之间做取舍。

**DTensor（Distributed Tensor）** 是 PyTorch 对这一困境的系统级回应。它的核心思想极其简洁——将分布式训练中的所有并行策略，统一抽象为**"一个逻辑张量 + 它在一组设备上的分布信息"**。用户继续以单卡视角编写模型代码，DTensor 则在幕后通过 `__torch_dispatch__` 机制自动推导并插入必要的集合通信操作。

这种抽象的价值不仅在于简化代码。更重要的是，它让不同并行策略从"互斥的代码路径"变成了"可组合的配置选项"：你可以在 DeviceMesh 的一个维度上做数据并行，另一个维度上做张量并行，而无需重写任何模型代码。

本文将从设计动机、核心抽象、内部调度机制到在 torchtitan 中的工程实践，系统性地介绍 PyTorch DTensor。阅读后，你将理解：DTensor 为何如此设计，它如何在 Eager 模式下实现自动通信推导，以及如何在实际项目中用好这一抽象。

---

## 一、设计动机：为什么需要 DTensor？

### 1.1 并行策略的组合困局

当前主流分布式训练策略可归纳为五类：

| 并行方式 | 核心思想 | 典型实现 | 主要痛点 |
| :--- | :--- | :--- | :--- |
| **数据并行（DP）** | 复制模型参数，按 batch 维度切分输入数据 | DDP、FSDP | 大模型参数无法单机容纳 |
| **张量并行（TP）** | 在 hidden/vocab 维度切分权重矩阵 | Megatron-LM | 通信与计算强耦合，代码侵入性强 |
| **序列并行（SP）** | 在序列维度切分激活值 | — | 需与 TP 配合，手动管理通信 |
| **流水线并行（PP）** | 在层（layer）维度切分模型 | PiPPy、DeepSpeed | 气泡（bubble）开销，调度复杂 |
| **参数分片（FSDP/ZeRO）** | 按参数/优化器状态维度切分 | FSDP、DeepSpeed ZeRO | 与 TP/PP 组合时检查点逻辑复杂 |

当训练千亿甚至万亿参数模型时，单一并行策略往往无法满足需求。业界普遍采用**3D 并行**（DP + TP + PP）或更复杂的组合策略。然而，每种策略都有独立的实现框架和 API：DDP 用 `DistributedDataParallel` 包装模型，Megatron-LM 在 `nn.Module` 中硬编码通信操作，FSDP 有自己的参数分片与检查点逻辑……这些方案之间缺乏互操作性，组合使用意味着维护多套代码路径，并手动保证它们之间的通信正确性。

以 Megatron-LM 为例，其张量并行的核心逻辑直接在 `nn.Linear` 的 forward 中插入 `all_reduce` 和 `all_gather`：

```python
# Megatron-LM 风格的列并行 Linear（简化示意）
class ColumnParallelLinear(nn.Module):
    def forward(self, input):
        output = F.linear(input, self.weight)  # 本地计算
        output = all_gather(output)           # 手动插入通信
        return output
```

这种"计算+通信"的强耦合模式带来两个问题：
1. **可维护性差**：换一个并行策略就要重写通信逻辑；
2. **可组合性弱**：DP 与 TP 的通信操作可能冲突，需要人工协调。

**根本症结在于：业界缺乏一种跨并行策略的通用数据分布抽象。**

### 1.2 DTensor 的设计目标

受 [GSPMD](https://arxiv.org/pdf/2105.04663.pdf)、[OneFlow](https://arxiv.org/pdf/2110.15032.pdf) 和 [TensorFlow DTensor](https://www.tensorflow.org/guide/dtensor_overview) 的启发，PyTorch 提出 DTensor 作为 ShardedTensor 的下一代演进。其设计目标可概括为三点：

**目标一：统一分布式数据表示**

无论采用何种并行策略，模型参数和激活值的分布式状态都通过同一套 `DeviceMesh + Placement` 机制描述。这意味着：
- 检查点（checkpoint）的保存/加载逻辑只需实现一次，即可覆盖 DP、TP、FSDP 及其任意组合；
- 不同并行策略之间的张量转换，由 DTensor 自动推导通信操作，无需用户手写。

**目标二：Eager 模式下的原生张量并行**

DTensor 在 Eager 模式下即可工作，不依赖图编译。用户可以在 Python 交互式环境中逐行调试分布式代码，像操作普通 `torch.Tensor` 一样操作 DTensor——这大大降低了分布式训练的调试门槛。

**目标三：作为编译器优化的基础构建模块**

DTensor 的显式分布信息（`DTensorSpec`）为编译器提供了精确的优化锚点。`torch.compile` 可以在编译时融合 Placement 检查、消除运行时调度开销，并进一步做通信融合与计算-通信重叠。

**一个极简示例**：三行代码将大张量分片到多设备——

```python
import os
import torch
from torch.distributed.tensor import init_device_mesh, Shard, distribute_tensor

mesh = init_device_mesh("cuda", (int(os.environ["WORLD_SIZE"]),))
big_tensor = torch.randn(100000, 88)
my_dtensor = distribute_tensor(big_tensor, mesh, [Shard(dim=0)])
```

### 1.3 与相关工作的对比

DTensor 并非首创概念。下表对比了 PyTorch DTensor 与业界同类方案的核心术语：

| PyTorch DTensor | OneFlow SBP | GSPMD | 语义 |
| :---: | :---: | :---: | :--- |
| `Shard(dim)` | `split` | `tiled` | 按维度分片 |
| `Replicate()` | `broadcast` | `replicated` | 全设备复制 |
| `Partial(reduce_op)` | `partial` | `partially tiled` | 待规约的部分值 |

**关键差异**：GSPMD 处于完全编译器模式，可在图层面自动插入 `all_reduce` / `reduce_scatter`，因此不需要显式的 `Partial` 状态。PyTorch DTensor 选择在 Eager 模式下将 `Partial` 作为显式中间状态暴露，这让调度系统可以在运行时根据上下文决定最优规约策略，但也意味着 DTensor 需要自行管理状态转换。

**相关工作详述：**

**GSPMD** 是 JAX/TensorFlow 分布式训练的基础设施，通过 XLA 编译器实现分片传播（Sharding Propagation）和算子融合。PyTorch XLA 的 `mark_sharding` API 即基于 GSPMD Partitioner，在 TPU 上实现 SPMD 训练。GSPMD 的优势在于编译器全局优化能力；劣势在于调试困难，且与 PyTorch Eager 生态的兼容性有限。

**OneFlow GlobalTensor** 提供了与 DTensor 非常接近的抽象：三种张量类型（split、broadcast、partial sum）对应三种 Placement。OneFlow 的独特之处在于将 `SBP`（Split-Broadcast-Partial）作为一等公民，贯穿于算子、张量和模块的设计中。其 "Boxing" 机制自动处理不同 SBP 状态之间的转换，与 DTensor 的 redistribute 概念异曲同工。

**TensorFlow DTensor** 是 TensorFlow 同步分布式训练的扩展，支持 n 维 mesh 上的分片与复制，并基于 MLIR 实现编译 Pass。DTensor 的 API 设计（`distribute_tensor`、`distribute_module`）在很大程度上借鉴了 TF DTensor 的经验。

**PyTorch ShardedTensor** 是 DTensor 的前身，目前处于维护模式。它仅支持张量分片（Shard），不支持复制（Replicate）和部分值（Partial），因此无法表达数据并行或混合并行策略。DTensor 是 ShardedTensor 的通用化替代，详见 [8.2 节](#82-pytorch-shardedtensor--dtensor)。

---

## 二、核心抽象：DeviceMesh、Placement 与 DTensorSpec

DTensor 体系由三个紧密协作的概念构成：

- **DeviceMesh**：描述"有哪些设备参与"以及它们的拓扑结构；
- **Placement**：描述"张量在每个 mesh 维度上如何分布"；
- **DTensorSpec**：将前两者绑定，附加全局元信息，构成 DTensor 的"分布身份证"。

理解这三个概念的关系，是掌握 DTensor 的关键。

### 2.1 DeviceMesh：设备的 N 维网格

在 PyTorch 2.3 之前，管理分布式训练设备拓扑意味着手动创建和维护多个 `ProcessGroup`：一个用于数据并行，一个用于张量并行，可能还有一个用于流水线并行。代码中充斥着 `dist.new_group()` 和 `dist.all_reduce(..., group=tp_group)` 的调用，极易出错。

DeviceMesh 将这组设备抽象为一个 **N 维网格**，每一维对应一种并行策略。例如，一个 `(2, 4)` 的 mesh 可以表示 2 路数据并行 × 4 路张量并行，共 8 个 rank。

#### 2.1.1 创建与索引

推荐使用 `init_device_mesh()` 创建 mesh（而非直接构造 `DeviceMesh` 对象）：

```python
from torch.distributed.device_mesh import init_device_mesh

# 一维：纯数据并行（world_size=4）
dp_mesh = init_device_mesh("cuda", mesh_shape=(4,), mesh_dim_names=("dp",))

# 二维：2路DP × 4路TP（world_size=8）
mesh_2d = init_device_mesh(
    "cuda", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp")
)
# 内部 rank 矩阵：[[0, 1, 2, 3],
#                  [4, 5, 6, 7]]
# "dp" 维（axis=0）的通信发生在 [0,4]、[1,5]、[2,6]、[3,7] 之间
# "tp" 维（axis=1）的通信发生在 [0,1,2,3] 和 [4,5,6,7] 之间

# 三维：PP × DP × TP
mesh_3d = init_device_mesh(
    "cuda", mesh_shape=(2, 2, 4), mesh_dim_names=("pp", "dp", "tp")
)
```

**子 mesh 索引**是 DeviceMesh 的核心能力之一。通过维度名获取子 mesh，不同 rank 看到的子 mesh 内容不同——这正是 SPMD 编程的精髓：

```python
tp_mesh = mesh_2d["tp"]
# rank 0~3 上: tp_mesh.mesh == tensor([0, 1, 2, 3])
# rank 4~7 上: tp_mesh.mesh == tensor([4, 5, 6, 7])

dp_mesh = mesh_2d["dp"]
# rank 0,4 上: dp_mesh.mesh == tensor([0, 4])
# rank 1,5 上: dp_mesh.mesh == tensor([1, 5])
```

多维子 mesh 索引支持传入维度名元组，返回顺序由传入顺序决定：

```python
mesh_3d = init_device_mesh("cuda", (2, 2, 2), mesh_dim_names=("dp", "pp", "cp"))

dp_cp_mesh = mesh_3d["dp", "cp"]  # 先 dp 再 cp
cp_dp_mesh = mesh_3d["cp", "dp"]  # 先 cp 再 dp（维度顺序不同，结果不同）
```

**扁平化**将多维子 mesh 合并为单一逻辑维度，上层代码无需感知底层多维结构：

```python
dp_like = world_mesh[("dp_replicate", "dp_shard")]
dp_like._flatten(mesh_dim_name="dp")  # 合成一维，命名为 "dp"
```

**进程信息查询**：

```python
dp_degree = dp_mesh.size()            # 该维度的进程数
dp_rank   = dp_mesh.get_local_rank()  # 当前进程在该维度的局部 rank
```

DeviceMesh 的设计是**设备无关**的：通过 `local_device_id = rank % num_gpus_per_node` 推导本地设备，同时适用于 GPU 和 CPU。

#### 2.1.2 在 torchtitan 中的实践

torchtitan 将并行度配置与设备拓扑解耦，形成清晰的两层架构：

- **`ParallelDims`**（配置层）：声明各并行维度的大小（`dp_replicate`、`dp_shard`、`cp`、`tp`、`pp`、`ep` 等），并校验与 `WORLD_SIZE` 的一致性。
- **`DeviceMesh`**（执行层）：根据 `ParallelDims` 构建物理/逻辑设备拓扑。

`build_mesh()` 的核心流程：

1. 筛选出并行度大于 1 的维度，组成 `dims` 和 `names`；
2. 调用 `init_device_mesh(device_type, dims, mesh_dim_names=names)` 创建世界 mesh；
3. 预先构造常用逻辑子 mesh 并扁平化：
   - `"dp"`：数据并行维度，用于数据加载和梯度同步；
   - `"dp_shard_cp"`：FSDP + Context Parallel 的联合维度；
   - `"dp_cp"`：loss all-reduce 的通信维度。

在 Trainer 中的典型使用：

```python
parallel_dims = ParallelDims(dp_replicate=2, dp_shard=1, tp=4, pp=1, ...)
world_mesh = parallel_dims.world_mesh

if parallel_dims.dp_enabled:
    dp_mesh = world_mesh["dp"]
    dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
    # dp_degree 传给 DataLoader 做数据切分
    # dp_rank 用于确定当前进程处理的数据分片
```

### 2.2 Placement：张量在 Mesh 上的分布方式

Placement 回答的核心问题是：**逻辑张量的每个元素，实际存放在哪个设备的显存里？**

DTensor 提供三种 Placement 类型：

| Placement | 语义 | 数据特征 | 典型场景 |
| :--- | :--- | :--- | :--- |
| `Replicate()` | 每个 rank 持有完整副本 | 各 rank `local_tensor` 相同 | 数据并行的模型参数 |
| `Shard(dim)` | 按张量维度 `dim` 切成 N 份，每个 rank 一份 | 各 rank `local_tensor` 是全局张量的子集 | 张量并行的权重、FSDP 的参数分片 |
| `Partial(reduce_op)` | 每个 rank 持有待规约的部分值 | 各 rank `local_tensor` 形状相同，但值不完整 | 矩阵乘法后的中间结果 |

#### 2.2.1 Placement 状态转换与通信原语

三种 Placement 之间通过集合通信操作相互转换，构成 DTensor 的"状态机"：

```
    Shard(dim)  ──all_gather───►  Replicate
    Replicate   ──local_chunk──►  Shard(dim)      (无通信)
    Partial(op) ──all_reduce───►  Replicate
    Partial(op) ──reduce_scatter──►  Shard(dim)
    Shard(X)    ──all_to_all───►  Shard(Y)
```

以 2 个 rank、全局值为 `[a, b, c, d]` 的 4 元素张量为例，各转换的具体效果：

| 转换 | 通信操作 | 转换前 | 转换后 |
| :--- | :--- | :--- | :--- |
| Replicate → Shard | 本地切片（`torch.chunk`） | rank0: `[a,b,c,d]`, rank1: `[a,b,c,d]` | rank0: `[a,b]`, rank1: `[c,d]` |
| Shard → Replicate | `all_gather` | rank0: `[a,b]`, rank1: `[c,d]` | rank0: `[a,b,c,d]`, rank1: `[a,b,c,d]` |
| Partial → Replicate | `all_reduce` | rank0: `[1,2,3,4]`, rank1: `[5,6,7,8]` | rank0: `[6,8,10,12]`, rank1: `[6,8,10,12]` |
| Partial → Shard | `reduce_scatter` | rank0: `[1,2,3,4]`, rank1: `[5,6,7,8]` | rank0: `[6,8]`, rank1: `[10,12]` |
| Shard(X) → Shard(Y) | `all_to_all` | rank0: `[a,b]`, rank1: `[c,d]`（按行） | rank0: `[a,c]`, rank1: `[b,d]`（按列） |

> **约束**：一个 DTensor 不能同时包含不同规约类型的 `Partial`。例如 `[Partial("sum"), Partial("max")]` 是非法的，因为 `sum`（线性）与 `max`（非线性）不满足交换律，重分布时无法确定规约顺序。以下代码将抛出 `ValueError`：
>
> ```python
> # 非法：混合 partial 类型
> placements = [Partial("sum"), Partial("max"), Shard(0)]  # ValueError!
> ```

#### 2.2.2 代码示例

**Replicate 与 Shard 的对比**：

```python
mesh = init_device_mesh("cuda", mesh_shape=(4,), mesh_dim_names=("dp",))
global_x = torch.arange(8, dtype=torch.float32)  # [0., 1., ..., 7.]

# Replicate：每个 rank 都持有完整张量
dt_rep = distribute_tensor(global_x, mesh, placements=[Replicate()])
# 所有 rank 上 dt_rep.to_local() == tensor([0., 1., ..., 7.])

# Shard：按 dim=0 切成 4 份
dt_shard = distribute_tensor(global_x, mesh, placements=[Shard(0)])
# rank 0: tensor([0., 1.])
# rank 1: tensor([2., 3.])
# rank 2: tensor([4., 5.])
# rank 3: tensor([6., 7.])
```

同一个 DeviceMesh，仅改变 Placement，数据分布方式即从"全员完整副本"变为"每人一块切片"。

**Shard → Replicate 的重分布**：

```python
dt_replicate = dt_shard.redistribute(mesh, placements=[Replicate()])
# DTensor 自动插入 all-gather，每个 rank 恢复完整张量
```

**2D Mesh 上的组合 Placement**（HSDP 场景）：

```python
# 2D mesh: dp=2 (节点间), tp=2 (节点内)
mesh = init_device_mesh("cuda", mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))

# 在 dp 维复制（数据并行），在 tp 维分片（张量并行）
placements = [Replicate(), Shard(0)]
dt = distribute_tensor(big_tensor, mesh, placements)
# 结果：节点间复制，节点内分片
```

### 2.3 DTensor：分布式张量本体

DTensor 是 `torch.Tensor` 的子类，对外表现为普通张量，内部由两部分组成：

```
DTensor = local_tensor + DTensorSpec
```

- **`local_tensor`**（`torch.Tensor`）：当前 rank 上真实存放在显存中的数据块。可能是完整副本（Replicate），也可能只是全局张量的一个分片（Shard）。
- **`DTensorSpec`**：分布规格，包含：
  - `mesh`：该 DTensor 关联的 DeviceMesh；
  - `placements`：在 mesh 的每个维度上是 `Shard` / `Replicate` / `Partial` 中的哪一种；
  - `tensor_meta`：全局 shape、stride、dtype 等元信息。

所有与"分布"有关的决策——算子调度、重分布、checkpoint——都依赖 `DTensorSpec`。可以说，**DTensorSpec 是 DTensor 的"分布身份证"**。

### 2.4 PlacementSpec 的形式化定义

从实现角度，Placement 和 DeviceMesh 的形式化定义如下：

```python
from dataclasses import dataclass
from typing import List
import torch.distributed.distributed_c10d as distributed_c10d

@dataclass
class Placement:
    pass

@dataclass
class Shard(Placement):
    dim: int  # 被分片的张量维度

@dataclass
class Replicate(Placement):
    pass

@dataclass
class _Partial(Placement):
    # 禁止从构造函数直接创建，仅由算子产生
    reduce_op: distributed_c10d.ReduceOp

@dataclass
class DeviceMesh:
    mesh: torch.Tensor           # n 维 rank 矩阵
    _pgs: List[ProcessGroup]     # 各维度的进程组
```

- `device_mesh` 是一个 n 维数组，指定张量放置到哪些 rank/设备上；
- `placements` 是一个与 `device_mesh` 同秩的数组，描述 DTensor 数据在 `device_mesh` 第 i 维上的分布方式。

通过这两个结构，我们可以精确建模任意分片、复制及混合策略。

---

## 三、DTensor API：创建、转换与模块分布

### 3.1 创建 DTensor

DTensor 提供三种创建方式，适用于不同场景：

**方式一：`distribute_tensor()`（推荐，自动分发数据）**

将已有的完整张量按指定 Placement 自动分发到 mesh 各 rank：

```python
from torch.distributed.tensor import distribute_tensor, Shard

local_tensor = torch.randn(8, 16, device=f"cuda:{rank}")
dtensor_shard = distribute_tensor(
    local_tensor, device_mesh=mesh, placements=[Shard(0)]
)
# 底层自动执行 scatter，将 local_tensor 的切片分发到各 rank
```

**方式二：`DTensor.from_local()`（零拷贝，用户保证数据正确性）**

从每个 rank 上已有的本地张量构建 DTensor，**不触发任何数据搬运**。这要求用户已自行保证各 rank 上的 `local_tensor` 内容正确（例如，已从分布式文件系统加载了对应分片）：

```python
from torch.distributed.tensor import DTensor

local_shard = torch.load(f"model_shard_{rank}.pt")  # 用户自行加载分片
dtensor = DTensor.from_local(
    local_shard, device_mesh=mesh, placements=[Shard(0)]
)
```

**方式三：工厂函数（直接创建，无需预分配）**

```python
from torch.distributed.tensor import DTensor

dtensor_ones = DTensor.ones(
    (8, 4), device_mesh=mesh, placements=[Shard(1)]
)
```

创建后，DTensor 可像普通张量一样参与运算，必要时自动插入通信。

### 3.2 转换操作

```python
local = dt.to_local()            # 获取当前 rank 持有的本地张量（零开销）
full  = dt.full_tensor()         # 全局 all-gather，恢复完整张量（高通信开销）
dt2   = dt.redistribute(mesh, [Replicate()])  # 切换 Placement，自动插入通信
```

> ⚠️ `full_tensor()` 会触发全量通信，仅用于调试、检查点保存或数值验证等**非热路径**场景。

### 3.3 模块级 API：`distribute_module`

对于已有的 `nn.Module`，`distribute_module` 提供声明式的方式将参数转为分布式：

```python
from torch.distributed.tensor import distribute_module
import torch.nn as nn

def shard_linear_params(mod_name, mod, mesh):
    """将 Linear 层的参数按行分片"""
    if isinstance(mod, nn.Linear):
        for name, param in mod.named_parameters():
            dist_param = nn.Parameter(
                distribute_tensor(param, mesh, [Shard(0)])
            )
            mod.register_parameter(name, dist_param)

sharded_model = distribute_module(model, mesh, partition_fn=shard_linear_params)
```

通过 `input_fn` 和 `output_fn` 可进一步控制模块输入/输出的分布方式（分别安装为 `forward_pre_hook` 和 `forward_hook`），实现模型代码与并行策略的完全解耦：

```python
def DDP(user_model, mesh):
    def input_fn(mod, inputs):
        # 将输入在 batch 维切分为 DTensor
        x = inputs[0]
        if not isinstance(x, DTensor):
            x = DTensor.from_local(x, mesh, [Shard(0)])
        return (x,)
    
    # 默认 Replicate 所有参数，input_fn 切分输入
    distribute_module(user_model, device_mesh=mesh, input_fn=input_fn)
    return user_model
```

### 3.4 分片 + 复制混合策略

DTensor 的独特优势在于能同时表达分片和复制。以下示例展示 2 节点（每节点 2 GPU）场景下，节点间复制（数据并行）、节点内分片（张量并行）的混合策略：

```python
# mesh: 2 节点 × 2 GPU = 4 rank
# axis 0 (dp): 节点间通信
# axis 1 (tp): 节点内通信
mesh = init_device_mesh("cuda", mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))

# 第一维 Replicate（数据并行），第二维 Shard（张量并行）
placements = [Replicate(), Shard(0)]
dt = distribute_tensor(big_tensor, mesh, placements)
```

这种混合策略在 HSDP（Hybrid Sharded Data Parallel）中非常常见，DTensor 用一行 Placement 配置即可表达，无需手写任何通信逻辑。

---

## 四、内部机制：DTensor 如何自动计算与通信

### 4.1 五步调度流程

当代码中出现 `y = torch.matmul(x, w)` 且 `x` 或 `w` 为 DTensor 时，底层执行以下五步：

**Step 1：拦截（Interception）**

PyTorch 的 dispatcher 检测到输入包含实现了 `__torch_dispatch__` 的对象，将调用路由至 DTensor 的自定义处理逻辑：

```python
def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
    return DTensor._op_dispatcher.dispatch(func, args, kwargs or {})
```

`__torch_dispatch__` 的优先级高于 CUDA 等 backend key，确保 DTensor 在实际 kernel 执行前获得控制权。其底层机制是：PyTorch 检查输入对象的 `__torch_dispatch__` 属性，若存在且非禁用状态，则设置 `Python dispatch key`——该 key 的优先级高于所有 dense backend key。

**Step 2：拆包（Unpacking）**

`OpDispatcher` 遍历参数，将每个 DTensor 拆解为 `local_tensor` + `DTensorSpec`，记录关联的 DeviceMesh，组装成 `OpInfo` 结构。

**Step 3：Sharding 推导（Propagation）**

`ShardingPropagator` 根据算子语义和输入 spec 决定：
- 输出的 Placement 应该是什么；
- 当前输入布局是否"够用"；
- 若不够用，需要哪些通信来重分布。

推导遵循严格的优先级：

1. 自定义 `custom_op_handler`（最高优先级，用户显式注册）；
2. `@register_op_strategy` 注册的 strategy（计算通信代价，用于复杂算子）；
3. `@register_prop_rule` 注册的 rule（基于规则推导，覆盖常见算子）；
4. 回退到 `composite implicit autograd decomposition`（兜底，将算子分解为已支持的原子操作）。

**Step 4：通信 + 本地计算**

若需要重分布，`OpDispatcher` 先调用 `DeviceMesh` 关联的进程组执行集合通信（`all_gather`、`all_reduce`、`reduce_scatter` 等），调整各 rank 的 `local_tensor` 布局；然后在当前 rank 上直接调用底层 ATen 算子（如 `aten.mm`）。

**Step 5：重包装（Repacking）**

本地算子输出的普通 `torch.Tensor`，根据 Step 3 推导的输出 spec，重新包装为 DTensor。后续算子继续以 DTensor 方式调度。

> **核心洞察**：计算只发生在 `local_tensor` 上，通信只在需要调整布局时触发。用户看到的是"一个逻辑张量在多卡上运算"，DTensor 在幕后决定何时、在哪些 mesh 维度上做 `all_gather` / `all_reduce` / `reduce_scatter`。

### 4.2 分布式算子调度：签名、传播与重分片

#### 4.2.1 输入/输出签名集

DTensor 为每个支持的算子维护一组合法的输入/输出 Placement 组合（签名）。以 `matmul` 为例：

| input1 | input2 | 合法输出 | 说明 |
| :--- | :--- | :--- | :--- |
| `Shard(0)` | `Shard(1)` | `Shard(0)` 或 `Shard(1)` | 行分片 × 列分片，输出可保持行或列分片 |
| `Shard(1)` | `Shard(0)` | `Partial` | 列分片 × 行分片，输出为待规约的部分值 |
| `Replicate` | `Shard(1)` | `Shard(1)` | 复制 × 列分片，输出继承列分片 |
| `Shard(0)` | `Replicate` | `Shard(0)` | 行分片 × 复制，输出继承行分片 |

当输入匹配某个签名时，DTensor 直接按该签名的规则执行；当输入不匹配任何签名时，进入**自动重分片**流程。

#### 4.2.2 分片传播的三级回退策略

DTensor 的算子调度采用优雅的三级回退机制，确保对 PyTorch 算子集的广泛覆盖：

1. **规则匹配**：算子存在分片传播规则，且输入 Placement 匹配已知签名 → 直接推导输出 Placement 并执行；
2. **自动重分片**：算子存在规则，但输入 Placement 不匹配 → 通过简单代价模型（如最小通信量）自动将输入重分片为合法输入，再执行；
3. **默认回退**：算子不存在任何分片传播规则 → 默认将所有输入重分布为 `Replicate`（通过 `all_gather`），执行本地计算。

第三级回退默认关闭，可通过配置开启，用于调试或接受性能折损的场景。

#### 4.2.3 歧义操作与代价模型

同一分布式操作可能有多种合法的输出分布。例如 `matmul` 的两个输入都是 `Shard(0)` 时，输出可以是 `Shard(0)`（本地计算），也可以是 `Replicate`（需要通信）。选择哪种方案需要权衡：

- **通信量**：是否引入额外的 `all_gather` 或 `all_reduce`；
- **关键路径**：通信是否与计算重叠，是否阻塞后续操作；
- **内存占用**：输出 Placement 是否导致显存峰值。

DTensor 在 Eager 模式下采用简单代价模型（优先选择最小通信量），更复杂的全局优化留给编译器层。这种"本地最优 + 全局编译器优化"的分层策略，兼顾了灵活性与性能。

#### 4.2.4 自动重分片（Auto-Redistribute）

当输入 Placement 不匹配算子的任何签名时，DTensor 需要将其转换为合法输入。这类似于 [OneFlow 的 Boxing](https://docs.oneflow.org/en/master/parallelism/03_consistent_tensor.html) 机制：

- **即时重分布**：在计算前立即对不匹配输入执行 `redistribute`；
- **代价驱动**：基于最小距离算法或启发式代价模型，选择通信开销最小的重分片路径。

### 4.3 N 维 DeviceMesh 的算子实现

1 维和 2 维 DeviceMesh 的算子实现相对直接，但扩展到 N 维时复杂度急剧上升。DTensor 的策略是：

- **优先覆盖 1D/2D**：这已覆盖几乎所有现有生产用例（DP+TP、DP+FSDP 等）；
- **允许用户构建 N 维 mesh**：不限制 mesh 维度，但复杂算子可能回退到 `all_gather + 本地计算`；
- **依赖编译器扩展**：N 维场景的完整优化，有待基于 `prim ops` 和编译器全局分析实现。

### 4.4 反向传播与自动微分

DTensor 的自动微分无需额外实现。`redistribute` 本身注册为 `torch.autograd.Function`，其反向规则自动派生：

| 前向操作 | 通信 | 反向操作 | 通信 |
| :--- | :--- | :--- | :--- |
| `Shard → Replicate` | `all_gather` | `Replicate → Shard` | 丢弃非所属分片（无通信） |
| `Replicate → Shard` | 丢弃分片（无通信） | `Shard → Replicate` | `all_gather` |
| `Partial → Replicate` | `all_reduce` | `Replicate → Partial` | 隐式生成（由梯度推导） |
| `Partial → Shard` | `reduce_scatter` | `Shard → Partial` | 隐式生成（由梯度推导） |

这意味着用户只需编写前向的 Placement 逻辑，反向通信由 autograd 引擎自动推导。以 DDP 和 FSDP 为例：

**DDP 的反向推导**：

前向：`matmul(input: Shard[0], param: Replicate) → output: Shard[0]`（无通信）

反向：输入梯度 `matmul(input_grad: Shard[0], param: Replicate) → input_grad: Shard[0]`（无通信）；参数梯度 `matmul(input: Shard[0], output_grad: Shard[0]) → param_grad: Partial` → 自动触发 `all_reduce` → `Replicate`。

**FSDP 的反向推导**：

前向：`redistribute(param: Shard → Replicate)`（`all_gather`）→ `matmul` → `redistribute(param: Replicate → Shard)`（丢弃分片，无通信）。

反向：参数梯度为 `Partial`，经 `all_reduce` 后得到 `Replicate`；再经 `redistribute(Replicate → Shard)` 丢弃分片，等价于 **`reduce_scatter`**——这与现有 FSDP 的反向行为完全一致，但由 DTensor 自动完成，无需用户手写。

---

## 五、用 DTensor 表达经典并行策略

DTensor 的真正威力在于：**经典并行策略不再是独立的代码框架，而是同一套抽象的不同配置**。

### 5.1 数据并行（DDP）

DDP 的本质：参数 `Replicate`，输入在 batch 维 `Shard`，前向无通信，反向自动 `all_reduce`。

```python
from torch.distributed.tensor import distribute_module, DTensor, Shard

def make_ddp(model, mesh):
    def input_fn(mod, inputs):
        x = inputs[0]
        if not isinstance(x, DTensor):
            x = DTensor.from_local(x, mesh, [Shard(0)])
        return (x,)
    
    # 默认 Replicate 所有参数；input_fn 将输入在 batch 维切分
    distribute_module(model, device_mesh=mesh, input_fn=input_fn)
    return model
```

### 5.2 参数完全分片（FSDP）

FSDP 的本质：参数初始 `Shard`，前向时 `all_gather` 为 `Replicate`，计算后丢弃非所属分片恢复 `Shard`；反向时 `all_reduce` 的结果自动 `reduce_scatter` 为 `Shard`。

```python
from torch.distributed.tensor import distribute_module, DTensor, Shard, Replicate

def make_fsdp(model, mesh):
    # 初始状态：所有参数 Shard(0)
    def shard_all_params(mod_name, mod, mesh):
        for name, param in mod.named_parameters():
            mod.register_parameter(
                name, 
                nn.Parameter(distribute_tensor(param, mesh, [Shard(0)]))
            )
    
    sharded_model = distribute_module(
        model, mesh, partition_fn=shard_all_params
    )
    
    # 前向时：通过 hook 将 Shard 参数 all_gather 为 Replicate
    def pre_forward_hook(mod, inputs):
        for param in mod.parameters():
            if isinstance(param, DTensor) and param.placements[0].is_shard():
                param.data = param.redistribute(mesh, [Replicate()]).to_local()
        return inputs
    
    # 注册 hook（实际 FSDP 实现更复杂，此处示意核心逻辑）
    sharded_model.register_forward_pre_hook(pre_forward_hook)
    return sharded_model
```

> 注：实际 PyTorch FSDP2 的 `fully_shard` 实现基于 DTensor，但做了大量工程优化（如 `FlatParameter`、通信与计算重叠等）。上述代码展示的是 DTensor 层面的语义等价性，而非生产级实现。

---

## 六、torchtitan 中的工程实践

[torchtitan](https://github.com/pytorch/torchtitan) 是 PyTorch 官方的大模型训练参考实现，全面采用 DTensor 作为并行基础。理解 torchtitan 的用法，是掌握 DTensor 工程实践的最佳途径。

### 6.1 ParallelDims + DeviceMesh：配置与拓扑分离

torchtitan 将"要做什么并行"（配置）与"设备怎么连"（拓扑）严格分离：

```python
from torchtitan.parallelisms import ParallelDims

# 声明并行维度：2路DP复制 × 1路DP分片 × 4路TP × 1路PP
parallel_dims = ParallelDims(
    dp_replicate=2, dp_shard=1, tp=4, pp=1, cp=1, ep=1
)

# 自动构建 world_mesh，校验 WORLD_SIZE=2×1×4×1×1×1=8
world_mesh = parallel_dims.world_mesh

# 按需取出子 mesh
tp_mesh = world_mesh["tp"]   # 张量并行组
dp_mesh = world_mesh["dp"]   # 数据并行组（含 dp_replicate + dp_shard 的扁平化）
```

`build_mesh()` 的幕后工作：
1. 筛选并行度 > 1 的维度；
2. 调用 `init_device_mesh` 创建世界 mesh；
3. 预计算常用逻辑子 mesh（如 `"dp"`、`"dp_shard_cp"`、`"dp_cp"`）并扁平化，避免运行时重复计算。

### 6.2 parallelize_module：声明式并行化

torchtitan 使用 `parallelize_module` 配合并行风格（parallel style）声明式地配置模型并行：

```python
from torch.distributed.tensor.parallel import (
    parallelize_module, ColwiseParallel, RowwiseParallel, SequenceParallel
)

# 在 tp_mesh 上配置各层的并行策略
parallelize_module(
    model,
    world_mesh["tp"],
    parallelize_plan={
        "layers.*.attention.wq": ColwiseParallel(),   # Q 投影：列分片
        "layers.*.attention.wk": ColwiseParallel(),   # K 投影：列分片
        "layers.*.attention.wv": ColwiseParallel(),   # V 投影：列分片
        "layers.*.attention.wo": RowwiseParallel(),   # O 投影：行分片
        "layers.*.feed_forward": SequenceParallel(),  # 前馈：序列并行
    }
)
```

这些并行风格的本质是为每个子模块指定：
- **输入期望的 Placement**（如 `ColwiseParallel` 期望输入为 `Replicate`）；
- **输出产生的 Placement**（如 `ColwiseParallel` 输出为 `Shard(1)`）；
- **必要的重分布操作**（如 `RowwiseParallel` 的输出为 `Partial`，框架自动插入 `reduce_scatter` 转为 `Shard`）。

`Partial` 作为中间态由框架内部管理，用户通常无需直接操作。

### 6.3 DTensor 在 torchtitan 中的三大价值

1. **统一抽象，消除重复代码**：DP / TP / SP / FSDP / HSDP 都通过 `DeviceMesh + Placement` 描述，torchtitan 只需配置不同 mesh/placement，无需为每种并行重写通信逻辑。

2. **计算与通信解耦**：模型代码（`nn.Module` 的 forward）只关心"算什么"，"怎么通信"由 DTensor + OpDispatcher 自动决策。修改模型结构（如换激活函数、加残差连接）不会牵一发而动全身。

3. **高层特性可组合**：`loss_parallel`（为 `log_softmax` / `nll_loss` 注册自定义 handler）、`FSDP2 fully_shard`（在适当时机将普通 Tensor 转为 DTensor）、`TP + CP` 组合——不同并行能力叠加使用而非互斥。

---

## 七、编译器方向：从 Eager 到编译优化

DTensor 为张量并行提供了高效的 Eager 模式实现，但在纯数据并行场景（频繁使用 `Replicate`）下，与高度优化的 DDP/FSDP 相比仍有性能差距。核心原因在于：

- **DDP/FSDP 拥有全局视图**：它们了解整个模型架构，可以做梯度桶化（Gradient Bucketing）、计算-通信重叠（Compute-Communication Overlap）等跨层优化；
- **DTensor 只看到局部**：作为类张量对象，它只知道当前算子的输入输出，不了解后续操作，无法做跨算子的通信融合。

**PyTorch 的解决方案是：将 DTensor 作为编译器优化的基础层。**

`torch.compile` 可以从用户程序中提取完整计算图，在图层面做全局优化：
- 将多个小粒度的 `all_reduce` 融合为大的 `all_reduce`；
- 将 Placement 检查从运行时移到编译时，消除调度开销；
- 做通信与计算的重叠调度，隐藏通信延迟。

长远来看，DTensor 的显式分布信息（`DTensorSpec`）是编译器进行全局优化的关键输入。Eager 模式保证灵活性和可调试性，编译器模式追求极致性能——两者互补，而非替代。

---

## 八、互操作与生态演进

### 8.1 XLAShardedTensor ↔ DTensor

PyTorch XLA 团队正在与 DTensor 团队合作，实现 XLAShardedTensor 与 DTensor 之间的双向转换。目标包括：

1. 提供 `to_xla_sharded()` / `from_xla_sharded()` 转换器，让同一套分布式代码可在 GPU 和 TPU 上运行；
2. 使高层分布式 API（如 `distribute_module`）设备无关，自动适配 XLA 后端。

### 8.2 PyTorch ShardedTensor ↔ DTensor

ShardedTensor 是 DTensor 的前身，目前处于维护模式。两者的核心差异：

| 特性 | ShardedTensor | DTensor |
| :--- | :--- | :--- |
| 支持的分布类型 | 仅 `Shard` | `Shard` + `Replicate` + `Partial` |
| 每 rank 分片数 | 允许多个 | 仅一个（基于 chunk） |
| 分片模式 | 灵活（`EnumerableShardingSpec`） | 简洁（基于 chunk） |
| 自动通信 | 不支持 | 通过 `__torch_dispatch__` 自动推导 |
| 推荐状态 | 维护中，建议迁移 | 活跃开发，推荐用于新项目 |

PyTorch 团队优先将 DTensor 作为张量分片及高级用例的默认方案。对于 ShardedTensor 的遗留用例（如多本地分片），将在评估需求后决定是否提供迁移工具。

### 8.3 分布式检查点（Distributed Checkpoint）

分布式检查点已利用 ShardedTensor 实现高效的保存/加载。DTensor 的检查点支持带来三项改进：

1. **更具表达力的状态保存**：同时包含分片 + 复制的单个参数，可通过 DTensor 表示统一保存，无需拆分为多个 ShardedTensor；
2. **跨 world size 加载**：基于 DTensor 检查点的重分片，支持在不同规模集群上加载检查点（如从 8 卡训练的检查点加载到 16 卡继续训练）；
3. **向后兼容**：提供 `to_sharded_tensor()` 工具函数，将 DTensor 转换为 ShardedTensor，衔接已有检查点生态。

---

## 九、总结

DTensor 的核心设计可概括为一张表、一个公式、一个流程：

**一张表**：

| 概念 | 解决的问题 | 关键属性/方法 |
| :--- | :--- | :--- |
| **DeviceMesh** | 设备怎么连？ | `mesh_shape`, `mesh_dim_names`, `size()`, `get_local_rank()` |
| **Placement** | 张量怎么放？ | `Shard(dim)`, `Replicate()`, `Partial(reduce_op)` |
| **DTensorSpec** | 分布信息怎么存？ | `mesh`, `placements`, `tensor_meta` |
| **DTensor** | 用户怎么无感知使用？ | `to_local()`, `redistribute()`, `full_tensor()` |

**一个公式**：

```
DTensor = local_tensor + DTensorSpec
        = 本地数据块 + 全局分布身份证
```

**一个流程**：

```
用户代码: y = torch.matmul(x, w)
          ↓
DTensor:  ① 拦截 → ② 拆包 → ③ 推导 Sharding → ④ 通信+本地计算 → ⑤ 重包装
          ↓
用户看到: y 仍是 DTensor，无需关心通信细节
```

DTensor 的价值不在于引入了新的分布式技术，而在于**将已有技术统一为可组合的配置**。它让"配置并行策略"从"到处散落的进程组调用"变成"一处集中声明、处处按 mesh 使用"——这正是大模型训练框架从"工程堆叠"走向"系统设计"的关键一步。

---

## 参考与延伸阅读

- **RFC Issue**: [pytorch/pytorch#88838](https://github.com/pytorch/pytorch/issues/88838)
- **设计文档**: [PyTorch DistributedTensor Full Design Doc](https://docs.google.com/document/d/1nFeJ8NSFNhNlCkNgWK31ZGRqm1L9rd0i_XN_RprphaI/edit)
- **dev-discuss 讨论**: [RFC: PyTorch DistributedTensor](https://dev-discuss.pytorch.org/t/rfc-pytorch-distributedtensor/740)
- **GSPMD**: [GSPMD: General and Scalable Parallelization for ML Computation Graphs](https://arxiv.org/pdf/2105.04663.pdf)
- **OneFlow**: [OneFlow: Redesign the Distributed Deep Learning Framework from Scratch](https://arxiv.org/pdf/2110.15032.pdf)
- **TensorFlow DTensor**: [DTensor Concepts](https://www.tensorflow.org/guide/dtensor_overview)
- **Megatron-LM**: [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/pdf/1909.08053.pdf)
- **DeepSpeed**: [https://github.com/deepspeedai/DeepSpeed](https://github.com/deepspeedai/DeepSpeed)

