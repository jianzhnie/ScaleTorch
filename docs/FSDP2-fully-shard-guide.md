# PyTorch FSDP2 完全分片数据并行：从原理到实践

> **文档状态**：Prototype（核心 API 已稳定，细节可能微调）
>
> **适用版本**：PyTorch 2.12+
>
> **前置知识**：了解数据并行（Data Parallelism）基本概念

---

## 一、为什么需要 FSDP2

**完全分片数据并行（Fully Sharded Data Parallelism, FSDP）** 的核心思想是：将模型的参数、梯度和优化器状态分片到多个数据并行 Worker 上，以通信开销换取内存节省。然而，FSDP1（`FullyShardedDataParallel`）在工程实践中暴露出三个关键痛点：

1. **参数表示复杂**：FSDP1 将一组参数展平、拼接后再分片。开发者难以判断每个 Worker 上实际存储了哪些数据，reshard 到其他并行策略时也十分复杂。
2. **内存行为不可预测**：FSDP1 依赖 `torch.Tensor.record_stream` 处理多流场景，导致内存使用非确定性。`limit_all_gathers=True` 时还会阻塞 CPU。
3. **API 僵化**：缺乏对预取（prefetching）和集合通信调度的手动控制，高级用户难以针对特定硬件拓扑做优化。

FSDP2 正是为了解决这些问题而设计的。如果你是 FSDP 的新用户，**建议直接从 FSDP2 开始**。如果你正在使用 FSDP1，建议评估下文差异后再决定是否迁移。

---

## 二、FSDP2 vs FSDP1：关键差异

| 维度 | FSDP1 | FSDP2 |
|------|-------|-------|
| **分片方式** | Flat-parameter sharding（展平后分片） | **Per-parameter dim-0 sharding**（基于 DTensor） |
| **参数表示** | 展平后的张量，难以直观理解 | 每个参数在 dim-0 上被 `torch.chunk(dim=0)` 切分，清晰直观 |
| **内存管理** | 使用 `record_stream`，行为非确定性 | **避免 `record_stream`**，内存使用可预测，不阻塞 CPU |
| **预取控制** | 自动实现，用户不可干预 | **暴露手动 API**（`set_modules_to_forward_prefetch` 等） |
| **State Dict** | 直接支持 full state dict | 仅支持 sharded state dict，需通过 DTensor API 自行转换 |
| **冻结参数** | 约束较多 | 约束更宽松，更灵活 |

### 2.1 分片表示：为什么 DTensor 更优

FSDP2 使用 **DTensor** 作为分片参数的底层表示。每个参数在 dim-0 上被均匀切分：

```python
# 假设 mesh 世界大小为 4，参数 shape 为 [8, 16]
# FSDP2 后，每个 Worker 持有的分片参数 shape 为 [2, 16]
# 即原参数在 dim-0 上被 chunk 为 4 份
```

这种表示的优势在于：

- **直观**：开发者可以直接查看 `module.weight` 的 shape，立刻知道分片情况。
- **灵活**：reshard 到其他并行维度（如 Tensor Parallelism）时，DTensor 的 Placement 抽象让转换更直接。
- **高效**：Sharded State Dict 可直接保存，无需像 FSDP1 那样先 all-gather 再保存。

> **补充说明**：DTensor（Distributed Tensor）是 PyTorch 的分布式张量抽象，通过 `Placement`（如 `Shard(0)`、`Replicate()`）描述张量在不同设备上的分布方式。FSDP2 中，分片参数的 Placement 为 `(Shard(0),)`。

---

## 三、核心 API：`fully_shard`

### 3.1 用户契约

调用 `fully_shard(model)` 后，框架与用户的契约如下：

- **初始化阶段**：`fully_shard` 将 `model.parameters()` 从普通 `torch.Tensor` 原地转换为 DTensor。参数会根据 device mesh 移动到相应设备。
- **前向/反向之前**：pre-forward/backward hook 负责 all-gather 参数，并将 `model.parameters()` 从 DTensor 转换为普通 `torch.Tensor`。
- **前向/反向之后**：post-forward/backward hook 释放 unsharded 参数（无需通信），并将 `model.parameters()` 从普通 `torch.Tensor` 转回 DTensor。
- **优化器**：必须使用 DTensor 形式的 `model.parameters()` 初始化，优化器步直接在 DTensor 参数上执行。
- **调用方式**：使用 `model(input)` 而非 `model.forward(input)`，以触发 pre-forward hook 进行参数 all-gather。若必须使用 `model.forward(input)`，需显式调用 `model.unshard()` 或使用 `register_fsdp_forward_method(model, "forward")` 注册。

### 3.2 通信分组与调度

每次调用 `fully_shard` 会创建一个通信组。该组包含模块中所有尚未被分配到子模块组的参数。每个组的参数在前向之前通过一次 all-gather 集体通信聚合，梯度在反向之后通过一次 reduce-scatter 集体通信分片。与 DDP 不同，FSDP2 **没有 `bucket_cap_mb` 参数**——通信边界完全由你对哪些模块应用 `fully_shard` 决定。

考虑一个包含四个子模块的模型，其中 $a$、$b$、$c$、$d$ 分别表示各子模块的参数数量：

```
model[ m1[a] -> m2[b] -> m3[c] -> m4[d] ]
```

**仅对根模块调用 `fully_shard(model)`**：所有参数都在一个组中。前向和反向的通信流程如下：

```
all-gather(a+b+c+d) -> forward(m1,m2,m3,m4) -> backward(m4,m3,m2,m1) -> reduce-scatter(a+b+c+d)
```

所有通信表现为两次大型阻塞操作，与计算没有任何重叠。**这几乎从来不是你想要的结果。**

**对每个子模块分别调用 `fully_shard`**：例如依次调用 `fully_shard(m2)`、`fully_shard(m3)`，然后 `fully_shard(model)`。剩余参数 $a$ 和 $d$ 组成根组，$m2$ 和 $m3$ 各自拥有独立组。在前向中，all-gather 在独立的 CUDA 流上运行，因此下一个模块的 all-gather 可以与当前模块的前向计算重叠：

```
              time ──────────────────────────────────────────────►

compute:      [wait] [ fwd(m1)   | fwd(m2)    | fwd(m3,m4)     ]
AG stream:    [AG(a,d)]  [AG(b)  |    AG(c)   ]
```

当 `fwd(m1)` 在计算流上运行时，CPU 提前触发 $m2$ 的 pre-forward hook，在 AG 流上发起 `AG(b)`。为了让这种重叠更稳健（例如当 CPU 侧开销降低提前量时），可以使用 `set_modules_to_forward_prefetch` 更早地发起下一个 all-gather——在当前模块的 pre-forward hook 内部而非等待下一个模块的 hook 触发。

在反向中，FSDP2 同样显式预取下一个模块的 all-gather，并在独立的 CUDA 流上运行 reduce-scatter，无需额外配置：

```
              time ──────────────────────────────────────────────►

compute:      [ bwd(m4,m3)     | bwd(m2)        | bwd(m1)       ]
AG stream:    [AG(c)] [ AG(b)  |   AG(a,d)      ]
RS stream:                     |[RS(c)]  [ RS(b)|     RS(a,d)   ]
```

当 `bwd(m4,m3)` 在计算流上运行时，$m2$ 所需参数的 all-gather 在 AG 流上预取。当 `bwd(m2)` 运行时，`AG(a,d)` 和 `RS(c)` 同时与计算重叠。这种流水线机制正是推荐**自底向上逐层应用 `fully_shard`** 的原因。

**控制通信组大小**：选择对哪些模块进行包裹。更细粒度的包裹产生更小、更易重叠的组（类似于更小的 DDP bucket）；更粗粒度的包裹产生更大的组。没有自动分桶——分组是显式的，由模块结构决定。

### 3.3 基本用法

```python
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed import DeviceMesh

# 假设已初始化分布式环境
mesh = DeviceMesh("cuda", list(range(world_size)))

# 对模型逐层应用 FSDP2（推荐 bottom-up 调用）
for layer in model.layers:
    fully_shard(layer, mesh=mesh)

# 最后对根模块调用
fully_shard(model, mesh=mesh)
```

**关键规则**：`fully_shard` 应该 **自底向上（bottom-up）** 调用。先对子模块调用，再对父模块调用。每个调用会构造一个**通信组**，包含该模块中尚未被分配到其他组的参数。如果只对根模块调用一次，所有参数会被分到同一个组，失去层间通信与计算重叠的机会。

### 3.4 参数详解

#### `module`（`nn.Module` 或 `List[nn.Module]`）

要应用 FSDP 的模块。传入多个模块时，会将它们的参数分到**同一个通信组**，共享一次 all-gather 和 reduce-scatter。

#### `mesh`（`DeviceMesh`，可选）

定义数据并行维度和设备。

- **1D Mesh**：纯 FSDP，参数在唯一维度上分片，Placement 为 `(Shard(0),)`。
- **2D Mesh**：HSDP（Hybrid Sharding），参数在第 1 维分片、第 0 维复制，Placement 为 `(Replicate(), Shard(0))`。

Mesh 的 device type 决定通信后端。若为 CUDA 或类 CUDA 设备，使用当前设备。

#### `reshard_after_forward`（`Optional[Union[bool, int]]`，默认 `None`）

控制前向传播后参数的行为，是**内存与通信的核心权衡点**：

| 值 | 行为 | 内存占用 | 通信量 |
|----|------|----------|--------|
| `True` | 前向结束后立即 reshard，反向时重新 all-gather | **最低** | 较高（多一次 all-gather） |
| `False` | 保持 unsharded 状态，反向无需 all-gather | 较高 | **最低** |
| `None` | 非根模块设为 `True`，根模块设为 `False`（默认行为） | 中等 | 中等 |
| `int`（如 `8`） | reshard 到更小 mesh（如 intra-node 大小） | 中等 | 中等 |

> **注意**：根模块的 `reshard_after_forward` 通常设为 `False`。因为根模块参数通常会在反向开始时立即被 all-gather。若设为 `None`，框架会自动对非根模块使用 `True`，对根模块使用 `False`。

**数学直觉**：设数据并行世界大小为 $N$，参数张量为 $\mathbf{W} \in \mathbb{R}^{D_0 \times D_1}$。

- 分片后每个 Worker 持有的参数为 $\mathbf{W}_i \in \mathbb{R}^{(D_0/N) \times D_1}$，满足 $\mathbf{W} = \text{Concat}([\mathbf{W}_0, \mathbf{W}_1, \dots, \mathbf{W}_{N-1}], \text{dim}=0)$。
- 当 `reshard_after_forward=True` 时，前向中 all-gather 的通信量为 $O(D_0 \times D_1 \times (N-1)/N)$，近似为参数大小。

#### `shard_placement_fn`（`Callable`，可选）

自定义分片维度和/或 mesh。可以返回：

- `None`：使用默认分片（`Shard(0)`），在传入 `fully_shard` 的 mesh 上分片。
- `Shard`：在指定维度上分片，使用传入 `fully_shard` 的 mesh。
- `ShardPlacementResult`：同时指定分片 Placement 和自定义 `FSDPMeshInfo`。这允许不同参数在不同进程组上分片，支持 Mixture of Experts 等场景（专家参数使用与常规参数不同的 mesh）。

> **约束**：非零维度分片时，该维度大小必须能被 FSDP shard mesh 大小整除（即要求 even sharding）。

#### `mp_policy`（`MixedPrecisionPolicy`，默认全 `None`）

模块级混合精度策略。与 `torch.autocast` 不同，FSDP2 的混合精度在**模块边界**做精度转换，而非算子级别。

```python
from torch.distributed.fsdp import MixedPrecisionPolicy

mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,    # 前向/反向计算精度
    reduce_dtype=torch.float32,      # 梯度规约精度
    output_dtype=torch.bfloat16,   # 输出转换精度
    cast_forward_inputs=True,        # 是否转换输入精度
)
```

**关键优势**：FSDP 本来就需要保存高精度分片参数用于优化器更新。因此模块级混合精度**不额外增加内存开销**。

#### `offload_policy`（`OffloadPolicy`，默认不卸载）

参数/梯度/优化器状态的 CPU 卸载策略。

```python
from torch.distributed.fsdp import CPUOffloadPolicy

offload_policy = CPUOffloadPolicy(pin_memory=True)
# pin_memory=True：加速 H2D/D2H 拷贝并支持计算重叠
# pin_memory=False：减少 CPU 内存占用，适合内存紧张场景
```

#### `ignored_params`（`Set[nn.Parameter] | None`，默认 `None`）

不参与 FSDP 分片的参数集合。这些参数不会被分片，不会在初始化时被移动到设备，也不会在反向中参与梯度规约。

#### `dp_mesh_dims`（`Optional[DataParallelMeshDims]`，可选）

提供时，`mesh` 被视为完整的 SPMD mesh，参数应为该 mesh 上的 DTensor，所有 DP 维度使用 `Replicate()`。`shard` 字段指定 FSDP 分片的维度（多个维度会被展平）；`replicate` 字段指定 HSDP 复制维度（多个维度会被展平）。

---

## 四、FSDPModule：动态类与扩展方法

调用 `fully_shard(module)` 后，`module` 的类型会被动态替换为一个新类（如 `FSDPLinear`），它继承自原类型和 `FSDPModule`。这种设计保持了：

- **模块结构不变**：子模块层级、参数命名（`named_parameters`）均不变。
- **扩展能力**：通过 `FSDPModule` 提供 FSDP 专用方法。

### 4.1 手动 Unshard / Reshard

```python
# 手动分配内存并 all-gather 参数
handle = module.unshard(async_op=True)
# ... 其他计算 ...
handle.wait()  # 等待 unshard 完成

# 手动释放 unsharded 参数，恢复分片状态
module.reshard()
```

> **补充说明**：`unshard` 遵循 `MixedPrecisionPolicy`。若设置了 `param_dtype`，all-gather 后的参数将为该精度。`async_op=True` 时返回 `UnshardHandle`，FSDP 会在该模块的 pre-forward 中自动等待，用户通常无需手动调用 `wait()`。

### 4.2 预取控制：重叠通信与计算

FSDP2 默认的预取策略基于前向/反向的模块遍历顺序，但高级用户可以手动覆盖：

```python
# 前向预取：当前模块 all-gather 完成后，提前开始下一个模块的 all-gather
module_a.set_modules_to_forward_prefetch([module_b, module_c])

# 反向预取：当前模块反向开始前，提前 all-gather 前一个模块的参数
module_c.set_modules_to_backward_prefetch([module_b, module_a])
```

- **单元素列表**（如 `[next_module]`）：与默认行为相同的重叠度。
- **多元素列表**（长度 $\geq 2$）：更激进的预取，用更多预留内存换取更大的通信/计算重叠。

### 4.3 梯度同步控制

```python
# 实现梯度累积（无通信版）
module.set_requires_gradient_sync(False, recurse=True)
# 等价于 FSDP1 的 no_sync() 上下文

# 仅 HSDP 场景：只做 reduce-scatter，不做 all-reduce
module.set_requires_all_reduce(False, recurse=True)
```

### 4.4 运行时配置方法

| 方法 | 用途 |
|------|------|
| `set_reshard_after_forward(bool, recurse=True)` | 运行时修改 `reshard_after_forward`，可用于评估时设为 `False`，训练时恢复 `True` |
| `set_reshard_after_backward(bool, recurse=True)` | 反向结束后是否 reshard，用于梯度累积时减少通信 |
| `set_unshard_in_backward(False)` | 若模块参数不参与反向计算（如 Embedding 冻结），避免冗余 all-gather |
| `set_gradient_divide_factor(factor)` | 自定义梯度归约的除数因子，可能使用 NCCL `PreMulSum` |
| `set_post_optim_event(event)` | 自定义优化器步后的同步事件，避免 false dependency |
| `set_is_last_backward(bool)` | 标记是否为最后一次反向，用于 microbatching 场景 |

### 4.5 高级通信定制

| 方法 | 用途 |
|------|------|
| `set_custom_all_gather(comm)` | 覆盖默认 all-gather 行为，精细控制通信和内存 |
| `set_custom_reduce_scatter(comm)` | 覆盖默认 reduce-scatter 行为，精细控制通信和内存 |
| `set_allocate_memory_from_process_group_for_comm(enable)` | 使用 ProcessGroup 提供的优化分配器分配临时 staging buffer，可能支持零拷贝传输（如 NCCL SHARP） |
| `set_force_sum_reduction_for_comms(enable)` | 强制底层集合通信仅使用 "sum" 类型规约，即使需要额外的前/后缩放操作。NCCL 零拷贝传输目前仅支持此类 collective |
| `set_symm_mem_for_comm(backend='NCCL')` | 使用对称内存（symm_mem）后端分配 all-gather 的 staging buffer，允许 NCCL 使用优化的 all-gather 实现（如单节点 Copy Engine All-Gather、多节点 Symmetric Kernel All-Gather） |

> **补充说明**：`set_allocate_memory_from_process_group_for_comm` 和 `set_symm_mem_for_comm` 不能与 `set_custom_all_gather` 或 `set_custom_reduce_scatter` 同时使用。启用 Copy Engine All-Gather 需要设置 NCCL process group 的 zero-CTA policy：`opts.config.cta_policy = dist.ProcessGroupNCCL.NCCL_CTA_POLICY_ZERO`，或设置环境变量 `NCCL_CTA_POLICY=2`。

---

## 五、混合精度深度解析

FSDP2 的 `MixedPrecisionPolicy` 在模块边界做精度管理，其状态机如下：

$$
\text{Sharded Param} \xrightarrow{\text{all-gather}} \text{Unsharded Param (param\_dtype)} \xrightarrow{\text{forward}} \text{Output} \xrightarrow{\text{cast}} \text{Output (output\_dtype)}
$$

- **`param_dtype`**：控制 all-gather 后的参数精度，也是前向/反向计算精度。
- **`reduce_dtype`**：控制梯度 reduce-scatter/all-reduce 的精度。若设为 `float32`，可在保持低精度计算的同时，用全精度累积梯度，减少精度损失。
- **`cast_forward_inputs`**：若为 `True`，FSDP 会自动将模块输入的浮点张量转换到 `param_dtype`。

> **补充说明**：当 `reduce_dtype` 为 `None` 但 `param_dtype` 不为 `None` 时，梯度规约使用计算精度（即 `param_dtype`）。若同时关闭了梯度同步（`set_requires_gradient_sync(False)`），梯度将直接以 `reduce_dtype` 累积。

---

## 六、注册自定义前向方法

默认情况下，FSDP 只会在 `nn.Module.forward()` 前后插入 all-gather / free 的 hook。如果你的模块有其他方法被当作前向传播使用（如 `generate`、`encode`），需要显式注册：

```python
from torch.distributed.fsdp import register_fsdp_forward_method

register_fsdp_forward_method(module, "generate")
register_fsdp_forward_method(module, "encode")
```

若 `module` 不是 `FSDPModule`，此调用为无操作（no-op）。

---

## 七、共享通信上下文

对于流水线并行（Pipeline Parallelism），每个模型片段都是一个 FSDP 根模块。为了避免流间内存碎片，可以共享所有 FSDP 模块的 CUDA 流：

```python
from torch.distributed.fsdp import share_comm_ctx

share_comm_ctx([fsdp_model_1, fsdp_model_2, ...])
```

这会共享 all-gather、reduce-scatter 和 all-reduce 的 CUDA 流，避免分配跨流内存碎片。

---

## 八、最佳实践与常见陷阱

### 8.1 调用顺序：Bottom-Up 是必须的

```python
# ✅ 正确：先子模块，后父模块
for layer in model.layers:
    fully_shard(layer, mesh=mesh)
fully_shard(model, mesh=mesh)

# ❌ 错误：只对根模块调用，所有参数在同一个通信组
fully_shard(model, mesh=mesh)  # 失去层间重叠机会
```

### 8.2 State Dict：保存与加载

FSDP2 不直接支持 full state dict。推荐做法：

```python
# 保存：直接获取 sharded state dict（包含 DTensor）
state_dict = model.state_dict()
# 使用 torch.distributed.checkpoint 保存
torch.distributed.checkpoint.save_state_dict(state_dict, checkpoint_id="path/to/checkpoint")

# 加载：sharded state dict 可直接加载
# 若需要 full state dict，手动转换：
full_param = param.full_tensor()  # DTensor API，会触发 all-gather
```

### 8.3 内存调优决策树

```
GPU 内存是否充足？
├── 是 → reshard_after_forward=False（避免反向 all-gather）
└── 否 → 是否有节点内高速互联（NVLink/IB）？
    ├── 是 → reshard_after_forward=intra_node_size（如 8）
    └── 否 → reshard_after_forward=True（最小内存占用）
```

### 8.4 CPU 卸载的取舍

`CPUOffloadPolicy` 将参数、梯度、优化器状态卸载到 CPU，适合**单卡显存极小但 CPU 内存充裕**的场景。注意：

- `pin_memory=True`：加速拷贝并支持重叠，但会**固定 CPU 内存**，其他进程无法使用。
- 若 CPU 内存紧张，设为 `False`。

---

## 九、总结

FSDP2 是 PyTorch 分布式训练的推荐选择，其核心改进可概括为：

1. **DTensor 分片表示**：参数分布直观，reshard 灵活，State Dict 高效。
2. **确定性内存管理**：消除 `record_stream` 的不可预测性，CPU 不阻塞。
3. **可扩展的预取 API**：从默认自动策略到手动精细控制，覆盖不同层次的用户需求。
4. **简化的 API 表面**：移除冗余参数，通过 DTensor 和分布式 checkpoint 解耦状态管理。
5. **高级通信定制**：支持自定义 all-gather/reduce-scatter、对称内存优化、零拷贝传输等。

对于新用户，FSDP2 的学习曲线更平缓。对于 FSDP1 的老用户，迁移的收益在于**更可控的内存行为和更灵活的优化空间**。在 PyTorch 2.12 及后续版本中，FSDP2 将是完全分片数据并行的主力实现。

---

## 参考链接

- [PyTorch FSDP2 官方文档](https://docs.pytorch.org/docs/2.12/distributed.fsdp.fully_shard.html)
- [FSDP2 入门教程](https://docs.pytorch.org/tutorials/intermediate/FSDP2_tutorial.html)
- [FSDP1 迁移指南](https://docs.pytorch.org/tutorials/intermediate/FSDP_migration.html)
- [PyTorch Distributed Checkpoint](https://docs.pytorch.org/distributed.checkpoint.html)
- [DTensor 深入理解](https://docs.pytorch.org/docs/stable/distributed.tensor.html)
- [对称内存与 Copy Engine Collectives](https://docs.pytorch.org/docs/2.11/symmetric_memory.html)