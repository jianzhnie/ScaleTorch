# Device Mesh — 多维进程组拓扑

> **官方参考**：[Getting Started with DeviceMesh](https://docs.pytorch.org/tutorials/recipes/distributed_device_mesh.html)
> (PyTorch ≥ 2.2)

## 什么是 DeviceMesh

DeviceMesh 是 PyTorch 提供的高级分布式抽象，管理底层的 `ProcessGroup`（NCCL/HCCL 通信域）。
它允许用户通过描述设备在多维网格中的 **布局** 来创建节点间和节点内的进程组，
而无需手动计算每个 rank 应该属于哪个子组。

```mermaid
graph TB
    subgraph manual["❌ 手动 new_group"]
        A1["计算 shard rank 分组"]
        A2["调用 new_group × (2+N/2) 次"]
        A3["rank in 判断归属"]
        A1 --> A2 --> A3
    end
    subgraph api["✅ DeviceMesh API"]
        B1["init_device_mesh(device_type, (2, 4),<br/>mesh_dim_names=('replicate', 'shard'))"]
        B2["mesh.get_group('shard')"]
        B1 --> B2
    end
    manual -->|"一行替代 20+ 行"| api
```

本项目的演示脚本位于 `examples/device_mesh/`，对应官方教程的各个阶段：

| 脚本 | 教程章节 / 策略 | Mesh 维度 | 方式 |
|---|---|---|---|
| `manual_process_group.py` | "without DeviceMesh" | 2D (2, N/2) | 手动 `dist.new_group()` — 理解底层 |
| `device_mesh_api.py` | "Simplify with DeviceMesh" | 2D (2, N/2) | `init_device_mesh()` — 生产推荐 |
| `fsdp_dp_demo.py` | "DeviceMesh with HSDP" | 2D (2, N/2) | FSDP + DP 混合分片 |
| `tensor_parallel_demo.py` | Megatron-LM TP | 1D (N,) | Colwise/Rowwise 权重分片 |
| `sequence_parallel_demo.py` | Megatron-LM SP | 1D (N,) | TP + 序列维度分片 |
| `fsdp_tp_demo.py` | 2D: FSDP + TP | 2D (dp, tp) | Llama 模型上的 TP + FSDP 组合 |

***

## 1. 后端检测

所有脚本共享同一套 NPU/CUDA 自动检测逻辑：

```python
try:
    import torch_npu
except ImportError:
    device_mod, backend, device_type = torch.cuda, "nccl", "cuda"
else:
    if torch.npu.is_available():
        device_mod, backend, device_type = torch.npu, "hccl", "npu"
    else:
        sys.exit("[2d_setup] torch_npu found but NPU is not available.")
```

| 场景                       | `device_mod` | `backend` | `device_type` |
| ------------------------ | ------------ | --------- | ------------- |
| `torch_npu` 未安装          | `torch.cuda` | `nccl`    | `cuda`        |
| `torch_npu` 已安装且 NPU 可用  | `torch.npu`  | `hccl`    | `npu`         |
| `torch_npu` 已安装但 NPU 不可用 | 退出并报错        | —         | —             |

三个变量的用途：

| 变量            | 用途                                                      |
| ------------- | ------------------------------------------------------- |
| `device_mod`  | 统一设备操作：`set_device` / `current_device` / `device_count` |
| `backend`     | `dist.init_process_group(backend)` 的通信后端                |
| `device_type` | `init_device_mesh(device_type, ...)` 的设备类型字符串           |

***

## 2. 分布式引导

```python
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
local_rank = int(os.environ.get("LOCAL_RANK", os.environ["RANK"]))

dist.init_process_group(backend)
device_mod.set_device(local_rank)
num_devices = device_mod.device_count()
```

- `RANK` / `WORLD_SIZE` / `LOCAL_RANK` 由 `torchrun` 自动注入。
- `init_process_group(backend)` 建立全局默认通信域（所有 rank 参与）。
- `set_device(local_rank)` 将进程绑定到物理 NPU/GPU。

> **注意**：此阶段仅创建 1 个全局组。二维拓扑需要在此基础上创建子组。

***

## 3. 手动方式（`2d_setup.py`）— 理解底层原理

8 卡场景下，目标拓扑如下：

```mermaid
graph LR
    subgraph shard_0["🧩 shard_0 (TP / EP)"]
        direction TB
        r0["rank 0"]
        r1["rank 1"]
        r2["rank 2"]
        r3["rank 3"]
    end
    subgraph shard_1["🧩 shard_1 (TP / EP)"]
        direction TB
        r4["rank 4"]
        r5["rank 5"]
        r6["rank 6"]
        r7["rank 7"]
    end

    r0 == "📦 DP" === r4
    r1 == "📦 DP" === r5
    r2 == "📦 DP" === r6
    r3 == "📦 DP" === r7
```

| 维度        | 组                                            | 策略                   |
| --------- | -------------------------------------------- | -------------------- |
| 实线框       | `shard_0` `[0,1,2,3]`, `shard_1` `[4,5,6,7]` | TP / EP（张量并行 / 专家并行） |
| 粗虚线 `===` | `(0,4)`, `(1,5)`, `(2,6)`, `(3,7)`           | DP / FSDP（数据并行）      |

### 3.1 Shard 组（连续半区）

```python
shard_rank_lists = (
    list(range(0, num_devices // 2)),           # [0, 1, 2, 3]
    list(range(num_devices // 2, num_devices)),  # [4, 5, 6, 7]
)
shard_groups = (
    dist.new_group(shard_rank_lists[0]),
    dist.new_group(shard_rank_lists[1]),
)
current_shard_group = (
    shard_groups[0] if rank in shard_rank_lists[0] else shard_groups[1]
)
```

- 设备按 **连续一半** 切成两个 shard 组。
- `dist.new_group(ranks)` 是**集合操作**：所有 8 个 rank 都必须调用，即使自己不加入该组。
- 每个 rank 通过 `rank in shard_rank_lists[0]` 判断归属并拿到自己的 group handle。

**用途**：Shard 组内做 **张量并行（Tensor Parallelism）** 或 **专家并行（Expert Parallelism）**——
模型的一层被切分到组内的 4 张卡上，组间互不干扰。

### 3.2 Replicate 组（交叉配对）

```python
urrent_replicate_group = None
current_replicate_ranks = None
shard_factor = len(shard_rank_lists[0])  # = 4
for i in range(num_devices // 2):        # i = 0, 1, 2, 3
    replicate_group_ranks = list(range(i, num_devices, shard_factor))
    replicate_group = dist.new_group(replicate_group_ranks)
    if rank in replicate_group_ranks:
        current_replicate_group = replicate_group
        current_replicate_ranks = replicate_group_ranks
```

`range(i, num_devices, shard_factor)` 以 `shard_factor=4` 为步长生成配对：

| i | `range(i, 8, 4)` | 配对              |
| - | ---------------- | --------------- |
| 0 | `[0, 4]`         | rank 0 ↔ rank 4 |
| 1 | `[1, 5]`         | rank 1 ↔ rank 5 |
| 2 | `[2, 6]`         | rank 2 ↔ rank 6 |
| 3 | `[3, 7]`         | rank 3 ↔ rank 7 |

**用途**：Replicate 组内做 **数据并行（Data Parallelism / FSDP）**——
组内两个 rank 持有相同的模型分片，处理不同 batch 数据，梯度在组内 all-reduce。


### 3.3 拓扑性质

| 维度        | 组大小                | 组数                 | 通信范围                       |
| --------- | ------------------ | ------------------ | -------------------------- |
| Shard     | `num_devices // 2` | 2                  | 组内 all-reduce / all-gather |
| Replicate | 2                  | `num_devices // 2` | 组内 all-reduce 梯度           |

两个维度的组 **正交**：任意两个 rank 恰好在一个维度上属于同一组，在另一个维度上属于不同组。

***

## 4. Smoke 测试

```python
tensor = torch.ones(1, device=device_mod.current_device()) * (rank + 1)
dist.all_reduce(tensor, group=current_shard_group)
expected = float(sum(r + 1 for r in my_shard_ranks))
assert abs(tensor.item() - expected) < 0.5
```

每个 rank 创建值为 `rank + 1` 的标量张量，在 **shard 组内** 做 all-reduce 求和：

| Shard 组  | Ranks      | 计算            | 期望值      |
| -------- | ---------- | ------------- | -------- |
| shard\_0 | 0, 1, 2, 3 | 1 + 2 + 3 + 4 | **10.0** |
| shard\_1 | 4, 5, 6, 7 | 5 + 6 + 7 + 8 | **26.0** |

输出示例（8 × NPU）：

```
[rank=0] shard_group=[0, 1, 2, 3] replicate_group=[0, 4] all_reduce=10.0 ✓
[rank=4] shard_group=[4, 5, 6, 7] replicate_group=[0, 4] all_reduce=26.0 ✓
...
```
验证了两点：
1. **Shard 组内通信正常**：同一半区的 rank 能正确 all-reduce。
2. **Shard 组间隔离正确**：跨半区的 rank 不参与对方的 all-reduce。

***

## 5. DeviceMesh API 方式

手动 `new_group` 需要管理 2 个 shard 组 + `N/2` 个 replicate 组的创建与匹配。
PyTorch 的 `init_device_mesh` 将这些细节封装为一行调用。

### 5.1 核心代码

```python
from torch.distributed.device_mesh import init_device_mesh

shard_size = num_devices // 2

mesh_2d = init_device_mesh(
    device_type,                              # "npu" or "cuda"
    mesh_shape=(2, shard_size),               # (replicate, shard)
    mesh_dim_names=("replicate", "shard"),
)

shard_group = mesh_2d.get_group(mesh_dim="shard")
replicate_group = mesh_2d.get_group(mesh_dim="replicate")
```

**一行** **`init_device_mesh`** **替代了第 3 节中 20+ 行的手动进程组创建逻辑。**

### 5.2 与手动方式对照

| 操作                 | 手动 `new_group`                                        | DeviceMesh                       |
| ------------------ | ----------------------------------------------------- | -------------------------------- |
| 创建 shard 组         | `dist.new_group([0,1,2,3])` × 2                       | `init_device_mesh(...)` 内部自动     |
| 创建 replicate 组     | `for i in ...: dist.new_group(...)` × 4               | `init_device_mesh(...)` 内部自动     |
| 获取 shard group     | `shard_groups[0] if rank in ... else shard_groups[1]` | `mesh_2d.get_group("shard")`     |
| 获取 replicate group | `if rank in ...: current_replicate_group = ...`       | `mesh_2d.get_group("replicate")` |
| 代码行数               | \~20                                                  | \~5                              |

### 5.3 内部原理

`init_device_mesh(mesh_shape=(2, 4), mesh_dim_names=("replicate", "shard"))` 在内部：

1. 按 `mesh_shape` 将 8 个 rank 排列为 2×4 网格。
2. 沿每个维度自动调用 `dist.new_group()`，为每行/每列创建子通信域。
3. 通过 `get_group(mesh_dim=...)` 暴露对应维度的 ProcessGroup。

等价于在 torchtitan 训练配置中：

```python
mesh = init_device_mesh(
    device_type,
    mesh_shape=(dp_size, tp_size),
    mesh_dim_names=("dp", "tp"),
)
```

二维 Mesh 支持将不同并行策略绑定到不同维度，由 PyTorch 的 `DTensor` 自动推导通信模式，无需手动管理 `new_group()` 调用。

### 5.4 Smoke 测试输出（8 × NPU）

与手动方式输出格式完全一致：

```
[rank=0] shard_group=[0, 1, 2, 3] replicate_group=[0, 4] all_reduce=10.0 ✓
[rank=4] shard_group=[4, 5, 6, 7] replicate_group=[0, 4] all_reduce=26.0 ✓
...
```

***

## 6. HSDP（`examples/device_mesh/fsdp_dp_demo.py`）

HSDP（Hybrid Sharding Data Parallel）将 FSDP 参数分片与数据并行复制结合，
通过二维 DeviceMesh 同时降低显存和跨节点通信。

### 6.1 拓扑语义

```
mesh_shape = (2, 4)
mesh_dim_names = ("dp_replicate", "dp_shard")
```

```mermaid
graph LR
    subgraph replica_0["🔄 dp_replicate group 0"]
        direction LR
        subgraph shard_A["📦 dp_shard [0,1,2,3]"]
            r0["rank 0"]
            r1["rank 1"]
            r2["rank 2"]
            r3["rank 3"]
        end
    end
    subgraph replica_1["🔄 dp_replicate group 1"]
        direction LR
        subgraph shard_B["📦 dp_shard [4,5,6,7]"]
            r4["rank 4"]
            r5["rank 5"]
            r6["rank 6"]
            r7["rank 7"]
        end
    end

    r0 == "grad all-reduce" === r4
    r1 == "grad all-reduce" === r5
    r2 == "grad all-reduce" === r6
    r3 == "grad all-reduce" === r7
```

| 维度             | 组大小 | 通信模式                             | 含义                  |
| -------------- | --- | -------------------------------- | ------------------- |
| `dp_shard`     | 4   | FSDP all-gather / reduce-scatter | 参数分片到组内 4 张卡，降低单卡显存 |
| `dp_replicate` | 2   | DP gradient all-reduce           | 跨复制组同步梯度            |

### 6.2 HSDP vs FSDP vs DDP

| 策略       | Mesh                      | 显存            | 跨节点通信                             |
| -------- | ------------------------- | ------------- | --------------------------------- |
| DDP      | 无分片                       | 最高（每卡完整模型）    | all-reduce 梯度                     |
| FSDP     | 1D shard                  | 最低（分片到所有卡）    | all-gather 参数（跨所有节点）              |
| **HSDP** | **2D (replicate, shard)** | **中等（分片到组内）** | **参数通信局限 shard 组，跨 replica 仅传梯度** |

HSDP 的关键优势：参数 all-gather 只发生在 `dp_shard` 组内（通常同节点 NVLink/HCCS），
跨节点的 `dp_replicate` 只传输梯度。

### 6.3 核心代码

```python
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

replicate_size = 2
shard_size = num_devices // replicate_size

mesh_2d = init_device_mesh(
    device_type,
    mesh_shape=(replicate_size, shard_size),
    mesh_dim_names=("dp_replicate", "dp_shard"),
)

model = ToyModel().to(device_mod.current_device())
fsdp_model = fully_shard(model, mesh=mesh_2d)
```

### 6.4 Smoke 测试输出（8 × NPU）

```
[rank=0] HSDP mesh: 2×4 (replicate=[0, 4] shard=[0, 1, 2, 3])
[rank=4] HSDP mesh: 2×4 (replicate=[0, 4] shard=[4, 5, 6, 7])
...
[rank=0] HSDP smoke test passed — loss=-1.1440 grad_norm=25.0927 ✓
[rank=3] HSDP smoke test passed — loss=-1.5912 grad_norm=3.6528 ✓
```

> **注**：loss 和 grad\_norm 在不同 rank 上可能不同。每个 rank 通过 `torch.randn` 生成了不同的输入数据，
> 且 FSDP 分片下 `grad_norm` 反映的是当前 rank 持有参数分片的局部梯度范数。
> 若需一致性验证，需固定随机种子。

### 6.5 维度语义对比

| 维度    | 2d\_setup / DeviceMesh  | HSDP                                       |
| ----- | ----------------------- | ------------------------------------------ |
| 第 0 维 | `replicate` → DP（数据并行）  | `dp_replicate` → DP 梯度同步                   |
| 第 1 维 | `shard` → TP / EP（模型并行） | `dp_shard` → FSDP 参数分片                     |
| 核心操作  | shard 组内 all-reduce 激活  | shard 组内 all-gather 参数 + reduce-scatter 梯度 |

两者构建了相同的二维 rank 排列 `(2, N/2)`，但上层语义完全不同。

***

## 7. Tensor Parallel vs Sequence Parallel

TP 和 SP 都使用 1D DeviceMesh 对模型进行列切分（Colwise）+ 行切分（Rowwise）。
核心差异在于**输入数据的分布方式**和**激活值的通信模式**。

### 7.1 Tensor Parallel（TP）

```python
# tensor_parallel_demo.py
"in_proj": ColwiseParallel()             # 输入复制，权重按列切分
"out_proj": RowwiseParallel()            # 权重按行切分，输出 all-reduce
```

> **模型定义**：2-rank TP 下各层权重形状变化：
>
> ```python
> class ToyModel(nn.Module):
>     def __init__(self):
>         self.in_proj  = nn.Linear(10, 32)   # W₁: (32, 10) → 列切 → 每 rank (16, 10)
>         self.relu     = nn.ReLU()
>         self.out_proj = nn.Linear(32, 5)    # W₂: (5, 32)  → 行切 → 每 rank (5, 16)
> ```
>
> 输入 `[20, 10]`，32 通道平分 → 每 rank 16 通道，因此图中切片维度均为 16。

```mermaid
graph TB
    input["相同输入 [20, 10]<br/>（所有 rank 相同）"]

    subgraph rank0["Rank 0"]
        w1_0["W₁[:16, :]<br/>列切分 (32→16)"]
        relu0["ReLU"]
        w2_0["W₂[:, :16]<br/>行切分 (32→16)"]
    end
    subgraph rank1["Rank 1"]
        w1_1["W₁[16:, :]<br/>列切分 (32→16)"]
        relu1["ReLU"]
        w2_1["W₂[:, 16:]<br/>行切分 (32→16)"]
    end

    ar["🔃 all-reduce"]

    output["完整输出 [20, 5]<br/>（所有 rank 相同）"]

    input --> w1_0
    input --> w1_1
    w1_0 --> relu0 --> w2_0
    w1_1 --> relu1 --> w2_1
    w2_0 --> ar
    w2_1 --> ar
    ar --> output
```

- **输入**：所有 rank 相同（`torch.manual_seed` 固定 seed）
- **通信**：仅 `RowwiseParallel` 末尾一次 **all-reduce**
- **激活显存**：完整（每 rank 持有完整激活张量）

### 7.2 Sequence Parallel（SP）

```python
# sequence_parallel_demo.py
"in_proj": ColwiseParallel(input_layouts=Shard(0))   # 输入按序列维度分片
"out_proj": RowwiseParallel(output_layouts=Shard(0))  # 输出按序列维度分片
```

```mermaid
graph TB
    inp0["输入分片 [10, 10]<br/>序列 [0:10]"]
    inp1["输入分片 [10, 10]<br/>序列 [10:20]"]

    ag["🔃 all-gather<br/>收集完整序列"]

    subgraph rank0["Rank 0"]
        w1_0s["W₁[:16, :]"]
        relu0s["ReLU"]
        w2_0s["W₂[:, :16]"]
    end
    subgraph rank1["Rank 1"]
        w1_1s["W₁[16:, :]"]
        relu1s["ReLU"]
        w2_1s["W₂[:, 16:]"]
    end

    rs["🔃 reduce-scatter<br/>按序列维度分散"]

    out0["输出分片 [10, 5]<br/>序列 [0:10]"]
    out1["输出分片 [10, 5]<br/>序列 [10:20]"]

    inp0 --> ag
    inp1 --> ag
    ag --> w1_0s
    ag --> w1_1s
    w1_0s --> relu0s --> w2_0s
    w1_1s --> relu1s --> w2_1s
    w2_0s --> rs
    w2_1s --> rs
    rs --> out0
    rs --> out1
```

- **输入**：每个 rank 持有序列的不同分片（`Shard(0)`），无需固定 seed
- **通信**：前端 **all-gather**（收集完整输入）+ 后端 **reduce-scatter**（分散输出）
- **激活显存**：按序列分片（大幅降低长序列场景的激活显存）

### 7.3 对比

| | TP | SP |
|---|---|---|
| 输入数据 | **相同**（`manual_seed` 固定） | **不同**（序列维度分片） |
| `input_layouts` | 默认 `Replicate()` | `Shard(0)` — 沿序列维度切分 |
| `output_layouts` | 默认 `Replicate()` | `Shard(0)` — 沿序列维度切分 |
| 前置通信 | 无 | **all-gather** |
| 后置通信 | **all-reduce** | **reduce-scatter** |
| 通信量 | 1× all-reduce（输出） | 1× all-gather + 1× reduce-scatter |
| 激活显存 | 完整（与 TP 组大小无关） | **1/N**（随 TP 组大小线性降低） |
| 适用场景 | 常规序列（<8K tokens） | **长序列**（>8K tokens），激活显存是瓶颈 |

### 7.4 代码对照

```python
# TP: 输入相同，输出汇总
"in_proj": ColwiseParallel(),                        # 无 input_layouts → Replicate
"out_proj": RowwiseParallel(),                       # 无 output_layouts → Replicate

# SP: 输入按序列分片，输出按序列分片
"in_proj": ColwiseParallel(input_layouts=Shard(0)),  # Shard(0) → 沿 dim 0 分片
"out_proj": RowwiseParallel(output_layouts=Shard(0)), # Shard(0) → 沿 dim 0 分片
```

> **本质**：SP = TP + 序列维度分片。在 TP 权重分片的基础上，将激活值也按序列维度分片，
> 代价是额外通信（all-gather + reduce-scatter），换来激活显存的线性下降。

### 7.5 2D 组合：FSDP + TP

`fsdp_tp_demo.py` 将 TP 和 FSDP 组合在二维 DeviceMesh 上，应用于 Llama 风格 transformer：

```python
# dp × tp = (world_size // tp_size, tp_size)
mesh_2d = init_device_mesh(device_type, (dp_size, tp_size), mesh_dim_names=("dp", "tp"))

# TP: 在 tp_mesh 上并行化每个 transformer block
tp_mesh = mesh_2d["tp"]
parallelize_module(transformer_block, device_mesh=tp_mesh, parallelize_plan={
    "attention.wq": ColwiseParallel(),
    "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
    ...
})

# FSDP: 在 dp_mesh 上包裹整个模型
dp_mesh = mesh_2d["dp"]
sharded_model = fully_shard(model, mesh=dp_mesh)
```

| 维度 | 策略 | 作用 |
|---|---|---|
| `tp` | Tensor Parallel | 权重/激活分片（节点内高带宽 HCCS） |
| `dp` | FSDP | 参数/梯度分片 + 数据并行（跨节点） |

***

## 8. 三维 Mesh 与子 Mesh 切片

当训练需要组合更多并行策略时（如 TP + DP + PP），可以使用三维 DeviceMesh
并通过切片语法复用父 Mesh 的通信域。

这是官方教程中 "Custom Parallel Solutions" 的 NPU 适配模式。

### 7.1 创建 3D Mesh 并切片

```python
from torch.distributed.device_mesh import init_device_mesh

# 3-D: 2 replicate × 2 shard × 2 tp = 8 devices
mesh_3d = init_device_mesh(
    device_type,
    mesh_shape=(2, 2, 2),
    mesh_dim_names=("replicate", "shard", "tp"),
)

# 子 Mesh 切片——复用父 Mesh 的 NCCL/HCCL 通信域，无额外 new_group 开销
hsdp_mesh = mesh_3d["replicate", "shard"]   # 2-D submesh for HSDP
tp_mesh = mesh_3d["tp"]                      # 1-D submesh for Tensor Parallel

# 从子 Mesh 中获取 ProcessGroup
replicate_group = hsdp_mesh["replicate"].get_group()
shard_group = hsdp_mesh["shard"].get_group()
tp_group = tp_mesh.get_group()
```

### 7.2 子 Mesh 切片原理

```mermaid
graph TB
    subgraph mesh_3d["mesh_3d (2, 2, 2)"]
        subgraph tp0["tp = 0"]
            r00["r0"]
            r01["r1"]
            r10["r2"]
            r11["r3"]
        end
        subgraph tp1["tp = 1"]
            r20["r4"]
            r21["r5"]
            r30["r6"]
            r31["r7"]
        end
    end

    hsdp["hsdp_mesh = mesh_3d['replicate', 'shard']<br/>→ 2-D 视图 (reuse comm)"]
    tpm["tp_mesh = mesh_3d['tp']<br/>→ 1-D 视图 (reuse comm)"]

    mesh_3d --> hsdp
    mesh_3d --> tpm
```

> 3D Mesh 将 8 个 rank 排列为 `(replicate=2, shard=2, tp=2)` 的三维网格。
> - `hsdp_mesh = mesh_3d["replicate", "shard"]` 提取前两维，形成 2×2=4 组的 HSDP 视图
> - `tp_mesh = mesh_3d["tp"]` 提取第三维，形成 2 组的 TP 视图
> - 切片操作复用父 Mesh 已建立的 NCCL/HCCL 通信域，**零额外开销**。

### 7.3 典型 3D 并行组合

| 维度          | 策略              | 通信模式                                |
| ----------- | --------------- | ----------------------------------- |
| `replicate` | DP              | 梯度 all-reduce（跨节点）                  |
| `shard`     | FSDP            | 参数 all-gather + reduce-scatter（节点内） |
| `tp`        | Tensor Parallel | 激活 all-reduce（节点内，最高带宽）             |

***

## 9. 最佳实践

### 始终命名 mesh 维度

```python
# ✅ 推荐：命名维度，子 Mesh 切片语义清晰
mesh = init_device_mesh("npu", (2, 4), mesh_dim_names=("replicate", "shard"))
hsdp = mesh["replicate", "shard"]

# ❌ 避免：未命名，仅能通过索引访问
mesh = init_device_mesh("npu", (2, 4))
```

### 动态匹配设备数

```python
# ✅ 推荐：动态计算 mesh_shape
num_devices = device_mod.device_count()
if num_devices % 2 != 0:
    sys.exit("Need even device count")
mesh = init_device_mesh(device_type, (2, num_devices // 2), ...)

# ❌ 避免：硬编码，在其他设备数下不可用
mesh = init_device_mesh("cuda", (2, 4), ...)
```

### 优先使用 `init_device_mesh` 而非 `DeviceMesh` 直接构造

```python
# ✅ 推荐：init_device_mesh 从 shape + 名称自动构建 mesh tensor
mesh = init_device_mesh("npu", (2, 4), mesh_dim_names=("dp", "tp"))

# ❌ 避免：手动构造 mesh tensor 容易出错，且需要额外 import torch
mesh = DeviceMesh("npu", torch.arange(8).reshape(2, 4))
```

### 使用 `get_group` 而非手动管理 group handle

```python
# ✅ 推荐：按名称获取
shard_group = mesh_2d.get_group(mesh_dim="shard")

# ❌ 避免：手动跟踪哪个 group 变量对应哪个维度
```

### `eager_init` 控制子组创建时机

```python
# 默认 False：子组延迟创建（首次使用时才初始化）
mesh = init_device_mesh("npu", (2, 4), mesh_dim_names=("dp", "tp"))

# True：立即通过 HCCL/NCCL comm split 创建所有子组
mesh = init_device_mesh(
    "npu", (2, 4), mesh_dim_names=("dp", "tp"), eager_init=True
)
```

### 运行方式

所有示例脚本通过 `torchrun` 启动：

```bash
# 单节点 8 卡
torchrun --nproc_per_node=8 examples/device_mesh/manual_process_group.py
torchrun --nproc_per_node=8 examples/device_mesh/device_mesh_api.py
torchrun --nproc_per_node=8 examples/device_mesh/fsdp_dp_demo.py

# 多节点 (以 2 节点 × 8 卡为例)
torchrun --nnodes=2 --nproc_per_node=8 \
    --rdzv_id=100 --rdzv_endpoint=<master_host>:29400 \
    examples/device_mesh/fsdp_dp_demo.py
```

***

## 10. 脚本速查

| 脚本 | 用途 | 运行命令 |
|---|---|---|
| `manual_process_group.py` | 手动 `new_group` 理解底层 | `torchrun --nproc_per_node=8 examples/device_mesh/manual_process_group.py` |
| `device_mesh_api.py` | `init_device_mesh` 简化写法 | `torchrun --nproc_per_node=8 examples/device_mesh/device_mesh_api.py` |
| `fsdp_dp_demo.py` | HSDP 混合分片数据并行 | `torchrun --nproc_per_node=8 examples/device_mesh/fsdp_dp_demo.py` |
| `tensor_parallel_demo.py` | Tensor Parallel (Megatron-LM) | `torchrun --nproc_per_node=8 examples/device_mesh/tensor_parallel_demo.py` |
| `sequence_parallel_demo.py` | Sequence Parallel | `torchrun --nproc_per_node=8 examples/device_mesh/sequence_parallel_demo.py` |
| `fsdp_tp_demo.py` | 2D: FSDP + TP (Llama) | `torchrun --nproc_per_node=8 examples/device_mesh/fsdp_tp_demo.py` |

