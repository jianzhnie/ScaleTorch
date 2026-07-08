# PyTorch FSDP2 完全指南：基于 Llama 2 的全分片数据并行实战

> **文档状态**：Prototype（核心 API 已稳定，细节可能随 PyTorch 版本微调）
>
> **适用版本**：PyTorch 2.4+
>
> **前置知识**：了解数据并行（Data Parallelism）基本概念

---

## 1. FSDP 工作原理

在 **DistributedDataParallel（DDP）** 训练中，每个 rank（即进程）都保存一份完整的模型副本。每个 rank 处理一个独立的数据批次(Batch)，最后通过 **all-reduce** 聚合梯度。

DDP 实现简单，但会浪费 GPU 显存。因为模型权重和优化器状态在每个 rank 上都是完整副本。

为减少这种冗余，可采用**全参数分片（full parameter sharding）**。与 DDP 不同，**FSDP（Fully Sharded Data Parallel）** 会对模型参数、梯度与优化器状进行**分片 (Sharding)**，显著降低了显存占用。即使在显存受限的场景下，也能训练超大模型。

FSDP 的参数生命周期如下图所示，可分为五个阶段：

<img src="https://docs.pytorch.org/tutorials/_images/fsdp_workflow.png" alt="FSDP workflow" style="zoom: 20%;" />

1. **Fully Sharded（静止态）**：在 forward 与 backward 计算之外，参数保持完全分片。每张卡只保存 $1/N$ 的参数分片。
2. **All-Gather（准备态）**：在 forward 与 backward 开始前，分片参数通过 all-gather 聚合成完整参数。
3. **Compute（计算态）**：使用完整参数进行计算。
4. **Reduce-Scatter（同步态）**：在 backward 内部，完整梯度被立即归约并切分为分片梯度。
5. **Update（更新态）**：优化器使用分片梯度更新分片参数，优化器状态也保持分片。

如下图所示，FSDP 把 DDP 的 all-reduce 分解为 reduce-scatter 与 all-gather 两个操作：

<img src="https://docs.pytorch.org/tutorials/_images/fsdp_sharding.png" alt="FSDP all-gather and reduce-scatter" style="zoom: 50%;" />

> 标准的 all-reduce 操作用于聚合梯度，可以分解为 reduce-scatter 与 all-gather 两个阶段。
>
> 在 reduce-scatter 阶段，梯度按照 rank 索引在每个 GPU 上分块求和。在 all-gather 阶段，每个 GPU 上已聚合的梯度分片会被广播给所有 GPU。

重新排列这两个操作后，每个 rank 只需保存一份参数与优化器状态的分片。

```text
                      ┌─────────────────┐
                      │  training data  │
                      └────────┬────────┘
            ┌──────────────────┼──────────────────┐
            │ shard 0          │ shard 1          │ shard 2
            ▼                  ▼                  ▼
     ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
     │  param      │    │  param      │    │  param      │
     │  shard 0    │    │  shard 1    │    │  shard 2    │
     │  GPU 0      │    │  GPU 1      │    │  GPU 2      │
     └──────┬──────┘    └──────┬──────┘    └──────┬──────┘
            │                  │                  │
            └──────── all-gather (params) ────────┘
                               │
                    full params on each GPU
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                  ▼
         forward             forward             forward
            │                  │                  │
            └───── reduce-scatter (grads) ────────┘
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                  ▼
     grad shard 0       grad shard 1       grad shard 2
     optim shard 0      optim shard 1      optim shard 2
        step               step               step
```

### 1.1 FSDP 与 DDP 的对比

下图对比了标准 DDP 与 FSDP 的执行方式：

- **标准数据并行**：每张 GPU 保存完整模型副本。前向与反向只处理数据分片。本地计算完成后，各 GPU 通过 all-reduce 同步梯度，再各自更新完整权重。
- **FSDP**：每张 GPU 只保存模型分片。本地计算前，通过 all-gather 从其他 GPU 收集完整权重，完成前向传播。反向传播前再次收集权重。反向完成后，本地梯度经平均并通过 reduce-scatter 分片到各 GPU。每张 GPU 只更新自己的权重分片。

<img src="https://engineering.fb.com/wp-content/uploads/2021/07/FSDP-Graph-2.png?w=907" alt="Full Sharded Data Parallel graph" style="zoom:80%;" />

### 1.2 FSDP 的典型执行流程

FSDP 的执行流程可概括为以下几步：

1. 多个 rank 同时运行，可能分布在多台机器上。每个 rank（即每张 GPU）只保存模型的一个分片。
2. 分片后，每层权重以 **DTensor**（分布式张量）形式存在，而不是普通 Tensor。因此没有任何一个 rank 能独立完成某个模块的计算。
3. 每次运算前，FSDP 发起一次 all-gather。各 rank 交换该模块的分片，临时恢复出完整模块。
4. 每个 rank 用自己的微批次（micro-batch）在该临时模块上执行前向传播，然后丢弃临时模块，继续处理下一层。
5. 反向传播时，FSDP 同样需要对每层 all-gather 解除分片，再计算梯度。
6. 由于每个 rank 处理的微批次不同，各自算出的梯度也不同。FSDP 通过 reduce-scatter 交换并平均梯度，得到全局梯度，再更新各自的分片。

因此，FSDP 的通信和流程比普通数据并行更复杂。但模型被分散到多张 GPU 后，训练超大模型所需的单卡显存显著下降。这正是使用 FSDP 的核心动机。

为了最大限度提高内存效率，可以在每一层前向传播结束后丢弃全部权重。这样能为后续层节省显存。实现方法是把 FSDP 包装器应用于网络的每一层，并设置 `reshard_after_forward=True`。

伪代码如下：

```python
# FSDP 前向传播
for layer_i in layers:
    all-gather layer_i 的完整权重
    执行 layer_i 的前向传播
    丢弃 layer_i 的完整权重

# FSDP 反向传播
for layer_i in layers:
    all-gather layer_i 的完整权重
    执行 layer_i 的反向传播
    丢弃 layer_i 的完整权重
    reduce-scatter layer_i 的梯度
```

## 2. 为什么需要 FSDP2？

FSDP2 是 PyTorch 分布式并行的下一代方案。它解决了 FSDP1（基于 `FlatParameter` 的包装器模式）在灵活性、组合性与可控性上的痛点。FSDP2 不再通过 Python 类包装模型，而是通过 **`torch.distributed.fsdp.fully_shard`** API 对模型进行原地（in-place）并行化。

如果你是 FSDP 新用户，**建议直接从 FSDP2 开始**；如果你正在使用 FSDP1，建议评估下文差异后再决定是否迁移。

### 2.1 FSDP1 的局限性

PyTorch 现有的 `FullyShardedDataParallel`（FSDP1）把多层参数“扁平化”为单个 `FlatParameter` 来实现分片。这种设计在训练大模型时有效，但随着规模与技术发展，逐渐暴露出几个问题：

- **参数表示复杂**：FSDP1 将一组参数展平、拼接后再分片。开发者难以判断每个 Worker 上实际存储了哪些数据，reshard 到其他并行策略时也很麻烦。
- **内存行为不可预测**：FSDP1 依赖 `torch.Tensor.record_stream` 处理多流场景，内存使用非确定性。`limit_all_gathers=True` 时还会阻塞 CPU。
- **API 僵化**：缺乏对预取（prefetching）和集合通信调度的手动控制，高级用户难以针对特定硬件拓扑做优化。
- **FP8 支持受限**：FP8 权重与非 FP8 参数无法在同一个 all-gather 中混合处理。
- **冻结参数处理复杂**：冻结参数与非冻结参数难以在同一个通信组中灵活共存。
- **检查点保存开销大**：训练表示与状态字典表示不一致，需要额外的通信转换。
- **编译器优化困难**：`torch.compile` 等图编译器难以对扁平化参数进行通信优化。

### 2.2 核心设计：逐参数分片

FSDP2 的核心思想非常简洁：**将每个参数独立在第 0 维进行分片**，而不是把多个参数扁平化后统一分片。这种“逐参数分片”（per-parameter sharding）带来四个关键优势：

| 特性 | FSDP1（扁平参数） | FSDP2（逐参数分片） |
| --- | --- | --- |
| 参数表示 | 展平后的张量，难以直观理解 | 每个参数在 dim-0 上切分，清晰直观 |
| 内存管理 | 使用 `record_stream`，行为非确定性 | 避免 `record_stream`，内存可预测，不阻塞 CPU |
| 预取控制 | 自动实现，用户不可干预 | 暴露手动 API（`set_modules_to_forward_prefetch` 等） |
| FP8 混合 all-gather | ❌ 不支持 | ✅ 灵活混合 |
| 冻结参数处理 | ❌ 需要额外内存 | ✅ 同一通信组无额外内存 |
| 检查点保存 | ❌ 需要 all-gather 转换 | ✅ 无需通信（sharded state dict） |
| 编译器优化 | ❌ 难以优化 | ✅ 可调整通信组 |

FSDP2 使用 **DTensor** 作为分片参数的底层表示，每个参数在 dim-0 上被均匀切分：

```python
# 假设 mesh word_size 为 4，参数 shape 为 [8, 16]
# FSDP2 后，每个 Worker 持有的分片参数 shape 为 [2, 16]
# 即原参数在 dim-0 上被 chunk 为 4 份
```

这种表示的优势在于：

- **直观**：开发者可以直接查看 `module.weight` 的 shape，立刻知道分片情况。
- **灵活**：reshard 到其他并行维度（如 Tensor Parallelism）时，DTensor 的 Placement 抽象让转换更直接。
- **高效**：Sharded State Dict 可直接保存，无需像 FSDP1 那样先 all-gather 再保存。

> **补充说明**：DTensor（Distributed Tensor）是 PyTorch 的分布式张量抽象，通过 `Placement`（如 `Shard(0)`、`Replicate()`）描述张量在不同设备上的分布方式。FSDP2 中，分片参数的 Placement 为 `(Shard(0),)`。

### 2.3 FSDP2 的优势

与 [FSDP1](https://docs.pytorch.org/docs/stable/fsdp.html) 相比，FSDP2 具备以下优势：

- FSDP1 把多个参数打平为一个巨大的 `FlatParameter`，再切分这个一维向量。这破坏了原始模型的参数结构，导致自定义初始化、特定层微调等操作变得复杂。
- FSDP2 保持模型原始参数结构。每个 `nn.Parameter` 被单独切分并管理。因此 FSDP2 具有极高的组合性，可轻松与张量并行（TP）或激活检查点（activation checkpointing）结合。
- 改进内存管理，避免使用 `recordStream`（[文档](https://dev-discuss.pytorch.org/t/fsdp-cudacachingallocator-an-outsider-newb-perspective/1486)），实现更低且确定性的 GPU 内存占用，且无需 CPU 同步。
- 提供张量子类扩展点，可自定义 all-gather。例如为 float8 线性层启用 float8 all-gather（[文档](https://dev-discuss.pytorch.org/t/enabling-float8-all-gather-in-fsdp2/2359)），以及为 QLoRA 支持 NF4（[文档](https://github.com/pytorch/torchtune/blob/main/README.md)）。
- 冻结参数与非冻结参数可在同一通信组中混合，且不占用额外内存。

## 3. API 设计：从“包装器”到“函数式”

### 3.1 全新的 `fully_shard` API

FSDP2 摒弃了原有的类包装器模式，采用更简洁的函数式 API：

```python
@contract(state_cls=FSDPState)
def fully_shard(
    module: nn.Module,
    *,
    mesh: Optional[DeviceMesh] = None,
    reshard_after_forward: Union[bool, int] = True,
    mp_policy: MixedPrecisionPolicy = MixedPrecisionPolicy(),
    offload_policy: OffloadPolicy = OffloadPolicy(),
) -> nn.Module:
```

使用：

```python
# PyTorch 2.4-2.6 使用 torch.distributed._composable.fsdp
# PyTorch 2.7+ 推荐从 torch.distributed.fsdp 导入
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed import DeviceMesh

# 假设已初始化分布式环境
mesh = DeviceMesh("cuda", list(range(world_size)))

# 对模型逐层应用 FSDP2（推荐 bottom-up 调用）
for layer in model.layers:
    fully_shard(layer, mesh=mesh)

# 最后对根模块调用
fully_shard(model, mesh=mesh, reshard_after_forward=True)
```

**关键规则**：`fully_shard` 应该 **自底向上（bottom-up）** 调用。对子模块先调用，再对父模块调用。每个调用会构造一个**通信组**，包含该模块中尚未被分配到其他组的参数。如果只对根模块调用一次，所有参数会被分到同一个组，失去层间通信与计算重叠的机会。

### 3.2 关键参数解析

#### `mesh`：定义数据并行维度

**DeviceMesh** 描述了设备的拓扑结构。FSDP2 依赖它实现多维并行（例如 2D FSDP，或 FSDP + TP）。只需定义不同的 Mesh 维度即可。

- **1D Mesh**：纯 FSDP，参数在唯一维度上分片，Placement 为 `(Shard(0),)`。
- **2D Mesh**：HSDP（Hybrid Sharding），参数在第 1 维分片、第 0 维复制，Placement 为 `(Replicate(), Shard(0))`。

Mesh 的 device type 决定通信后端；若为 CUDA 或类 CUDA 设备，使用当前设备。

#### `reshard_after_forward`：内存与通信的权衡

`fully_shard()` 的 `reshard_after_forward` 参数控制前向结束后是否释放 all-gather 得到的完整参数。这是 FSDP2 最灵活的参数之一，其取值可与 DeepSpeed ZeRO 的不同阶段类比：

```python
# ZeRO-3 模式：前向后释放参数，反向时重新 all-gather
fully_shard(module, reshard_after_forward=True)

# ZeRO-2 模式：前向保持参数，反向无需 all-gather
fully_shard(module, reshard_after_forward=False)

# ZeRO++ hpZ 模式：前向后部分重分片（如节点内大小）
fully_shard(module, reshard_after_forward=8)  # 8 卡节点内保持
```

> 注：`reshard_after_forward` 与 DeepSpeed ZeRO 阶段的对应关系是一种工程近似类比，并非严格等价。具体行为还受 DeviceMesh、通信组划分等因素影响。

| 值 | 对应 ZeRO | 内存占用 | 反向通信 |
| --- | --- | --- | --- |
| `True` | ZeRO-3 | 最低 | 需要 all-gather |
| `False` | ZeRO-2 | 较高 | 无需 all-gather |
| `int` | ZeRO++ hpZ | 中等 | 部分 all-gather |

> **补充说明**：
>
> - 在 PyTorch FSDP2 中，`reshard_after_forward` 默认为 `True`，即前向结束后释放参数。
> - 设为 `False` 可让对应模块在前后向之间保持未分片，节省一次 all-gather，但会增加峰值显存。
> - 根模块的 `reshard_after_forward` 会被内部启发式地设为 `False`，因为根模块参数通常会在反向开始时立即被 all-gather。

**数学直觉**：设数据并行`Word Size`为 $N$，参数张量为 $\mathbf{W} \in \mathbb{R}^{D_0 \times D_1}$。

- 分片后每个 Worker 持有的参数为 $\mathbf{W}_i \in \mathbb{R}^{(D_0/N) \times D_1}$，满足 $\mathbf{W} = \text{Concat}([\mathbf{W}_0, \mathbf{W}_1, \dots, \mathbf{W}_{N-1}], \text{dim}=0)$。
- 当 `reshard_after_forward=True` 时，前向中 all-gather 的单卡接收通信量为 $O(D_0 \times D_1 \times (N-1)/N)$，近似为单卡接收完整参数的数据量。

理解该参数有助于设计模型结构。例如在 `LlamaForPretraining` 中，根模块只包含预测头。若把嵌入层也移到根模块，前向结束后嵌入层会一直驻留显存，可能造成浪费。

#### `mp_policy`：细粒度混合精度控制

```python
@dataclass
class MixedPrecisionPolicy:
    param_dtype: Optional[torch.dtype] = None      # 参数 / 计算精度
    reduce_dtype: Optional[torch.dtype] = None     # 梯度归约精度
    output_dtype: Optional[torch.dtype] = None     # 输出精度
    cast_forward_inputs: bool = True               # 是否转换输入
```

典型配置示例：

```python
# BF16 训练 + FP32 梯度归约（常用配置）
mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32
)
```

FSDP2 的 `MixedPrecisionPolicy` 在模块边界做精度管理，其状态机如下：

$$
\text{Sharded Param} \xrightarrow{\text{all-gather}} \text{Unsharded Param (param\_dtype)} \xrightarrow{\text{forward}} \text{Output} \xrightarrow{\text{cast}} \text{Output (output\_dtype)}
$$

- **`param_dtype`**：控制 all-gather 后的参数精度，也是前向 / 反向计算精度。
- **`reduce_dtype`**：控制梯度 reduce-scatter / all-reduce 的精度。若设为 `float32`，可在保持低精度计算的同时，用全精度累积梯度，减少精度损失。
- **`cast_forward_inputs`**：若为 `True`，FSDP 会自动将模块输入的浮点张量转换到 `param_dtype`。

> **补充说明**：当 `reduce_dtype` 为 `None` 但 `param_dtype` 不为 `None` 时，梯度归约使用计算精度（即 `param_dtype`）。若同时关闭了梯度同步（`set_requires_gradient_sync(False)`），梯度将直接以 `reduce_dtype` 累积。

对于 Transformer 架构，推荐对每个 Transformer 块和根模块分别应用：

```python
for module in model.modules():
    if isinstance(module, TransformerBlock):
        fully_shard(module, mesh=mesh, mp_policy=mp_policy)
fully_shard(model, mesh=mesh, mp_policy=mp_policy)
```

这种“分块 + 根”的模式构造了多个通信组，是实现通信与计算重叠的关键。

#### `offload_policy`：CPU 卸载策略

参数 / 梯度 / 优化器状态的 CPU 卸载策略。

```python
from torch.distributed.fsdp import CPUOffloadPolicy

offload_policy = CPUOffloadPolicy(pin_memory=True)
# pin_memory=True：加速 H2D/D2H 拷贝并支持计算重叠
# pin_memory=False：减少 CPU 内存占用，适合内存紧张场景
```

#### `shard_placement_fn`：自定义分片维度

自定义分片维度。默认在 dim-0 分片，若返回 `Shard(1)` 则在 dim-1 分片。

> **约束**：非零维度分片时，该维度大小必须能被 FSDP shard mesh 大小整除（即要求 even sharding）。

#### `ignored_params`：不参与分片的参数

不参与 FSDP 分片的参数集合。这些参数将保持原始状态，不会进行 all-gather / reduce-scatter。

### 3.3 FSDPModule：动态类与扩展方法

调用 `fully_shard(module)` 后，`module` 的类型会被动态替换为一个新类（如 `FSDPLinear`），它继承自原类型和 `FSDPModule`。这种设计保持了：

- **模块结构不变**：子模块层级、参数命名（`named_parameters`）均不变。
- **扩展能力**：通过 `FSDPModule` 提供 FSDP 专用方法。

#### 手动 Unshard / Reshard

```python
# 手动分配内存并 all-gather 参数
handle = module.unshard(async_op=True)
# ... 其他计算 ...
handle.wait()  # 等待 unshard 完成

# 手动释放 unsharded 参数，恢复分片状态
module.reshard()
```

> **补充说明**：`unshard` 遵循 `MixedPrecisionPolicy`，若设置了 `param_dtype`，all-gather 后的参数将为该精度。`async_op=True` 时返回 `UnshardHandle`，FSDP 会在该模块的 pre-forward 中自动等待，用户通常无需手动调用 `wait()`。

#### 预取控制：重叠通信与计算

FSDP2 默认的预取策略基于前向 / 反向的模块遍历顺序，但高级用户可以手动覆盖：

```python
# 前向预取：当前模块 all-gather 完成后，提前开始下一个模块的 all-gather
module_a.set_modules_to_forward_prefetch([module_b, module_c])

# 反向预取：当前模块反向开始前，提前 all-gather 前一个模块的参数
module_c.set_modules_to_backward_prefetch([module_b, module_a])
```

- **单元素列表**（如 `[next_module]`）：与默认行为相同的重叠度。
- **多元素列表**（长度 $\geq 2$）：更激进的预取，用更多预留内存换取更大的通信 / 计算重叠。

#### 梯度同步控制

```python
# 实现梯度累积（无通信版）
module.set_requires_gradient_sync(False, recurse=True)
# 等价于 FSDP1 的 no_sync() 上下文

# 仅 HSDP 场景：只做 reduce-scatter，不做 all-reduce
module.set_requires_all_reduce(False, recurse=True)
```

#### 其他实用方法

| 方法 | 用途 |
|------|------|
| `set_reshard_after_backward(bool)` | 反向结束后是否 reshard，用于梯度累积时减少通信 |
| `set_unshard_in_backward(False)` | 若模块参数不参与反向计算（如 Embedding 冻结），避免冗余 all-gather |
| `set_reduce_scatter_divide_factor(factor)` | 自定义梯度归约的除数因子，使用 NCCL `PreMulSum` |
| `set_post_optim_event(event)` | 自定义优化器步后的同步事件，避免 false dependency |
| `set_is_last_backward(bool)` | 标记是否为最后一次反向，用于 microbatching 场景 |

### 3.4 注册自定义前向方法

默认情况下，FSDP 只会在 `nn.Module.forward()` 前后插入 all-gather / free 的 hook。如果你的模块有其他方法被当作前向传播使用（如 `generate`、`encode`），需要显式注册：

```python
from torch.distributed.fsdp import register_fsdp_forward_method

register_fsdp_forward_method(module, "generate")
register_fsdp_forward_method(module, "encode")
```

若 `module` 不是 `FSDPModule`，此调用为无操作（no-op）。

## 4. FSDP2 的使用方法

后文代码片段均取自项目中的 `examples/FSDP2/fsdp2_llama2_main.py`。该脚本是一个可在 NPU / CUDA 上直接运行的 Llama 2 预训练示例，涵盖进程组初始化、DeviceMesh、模型切分、训练循环、混合精度、CPU 卸载、激活检查点、显式预取和分布式 Checkpoint。完整源码与辅助模块（`checkpoint.py`、`data.py`、`llama2.py`）位于同一目录。

启动方式示例：

```bash
# 8 卡 BF16 训练
torchrun --nproc_per_node=8 examples/FSDP2/fsdp2_llama2_main.py \
    --model-size 7B --mixed-precision

# CPU 卸载
torchrun --nproc_per_node=8 examples/FSDP2/fsdp2_llama2_main.py \
    --model-size 1B --mixed-precision --cpu-offload

# 显式预取 + 激活检查点
torchrun --nproc_per_node=8 examples/FSDP2/fsdp2_llama2_main.py \
    --model-size 7B --mixed-precision \
    --explicit-prefetching 2 --activation-checkpointing

# 合成数据快速验证
torchrun --nproc_per_node=8 examples/FSDP2/fsdp2_llama2_main.py \
    --model-size debug --use-synthetic-data --epochs 1
```

PyTorch FSDP2 的 `fully_shard` 支持多种切分粒度。最细粒度可按层切分，最大限度降低峰值显存。粒度越粗，通信开销通常越低。需要根据实际场景选择。

除切分粒度外，FSDP2 还提供多种配置，用于优化性能并缓解 OOM。

### 4.1 DTensor 基础

FSDP2 的底层基石是 **DTensor（`torch.distributed.tensor.DTensor`）**。

**逻辑视图与物理视图分离**：

- **逻辑上**：参数看起来仍是完整 Tensor（例如 `[4096, 4096]`），保持与单卡训练一致的编程体验。
- **物理上**：参数实际被切分并分布在 `DeviceMesh` 定义的设备组中。每张卡只持有本地张量（Local Tensor）。

> **补充说明**：以 8 卡数据并行、参数形状 `[4096, 4096]` 为例，每张卡的本地张量为 `[512, 4096]`。实际本地形状取决于 world size 与切分维度。

### 4.2 DeviceMesh 配置

**DeviceMesh** 描述了设备的拓扑结构。FSDP2 依赖它实现多维并行（例如 2D FSDP，或 FSDP + TP）。只需定义不同的 Mesh 维度即可。

`fsdp2_llama2_main.py` 使用一维 DeviceMesh 实现纯数据并行。其初始化逻辑在进程组启动后完成：

```python
from torch.distributed.device_mesh import init_device_mesh

mesh = init_device_mesh(
    device_type,
    mesh_shape=(world_size,),
    mesh_dim_names=("dp",),
)
```

该 `mesh` 会传入所有 `fully_shard()` 调用。`DeviceMesh` 也支持多维并行，如张量并行、流水线并行。组合多种并行方式通常能进一步提升训练性能。

更多信息可参考 [PyTorch 设备网格文档](https://docs.pytorch.org/tutorials/recipes/distributed_device_mesh.html)。

### 4.3 初始化进程组

多进程训练需要通过 `torchrun` 启动。每个 rank 会获得 world size、rank 和 local rank。`fsdp2_llama2_main.py` 的启动代码同时兼容 NPU（HCCL）与 CUDA（NCCL）：

```python
import os
import sys
import torch
import torch.distributed as dist

# 自动检测 NPU，否则回退到 CUDA
try:
    import torch_npu  # noqa: F401
except ImportError:
    device_mod, backend, device_type = torch.cuda, "nccl", "cuda"
else:
    if torch.npu.is_available():
        device_mod, backend, device_type = torch.npu, "hccl", "npu"
    else:
        sys.exit("[fsdp2_llama2] torch_npu found but NPU is not available.")

rank = int(os.environ["RANK"])
local_rank = int(os.environ.get("LOCAL_RANK", os.environ["RANK"]))
world_size = int(os.environ["WORLD_SIZE"])

dist.init_process_group(backend)
device_mod.set_device(local_rank)
device = torch.device(f"{device_type}:{local_rank}")
```

之后即可用该 `device` 和 `world_size` 创建一维 DeviceMesh（见 4.2 节）。

### 4.4 模型初始化

**对子模块应用 `fully_shard`**：与 DDP 不同，FSDP2 需要对子模块以及根模型都应用 `fully_shard`。`fully_shard()` 会原地把普通 Tensor 参数替换为 DTensor，并修改模块使其在计算前自动 all-gather。

`fsdp2_llama2_main.py` 中的 `_build_model` 展示了完整初始化流程：

```python
from llama2 import LlamaConfig, LlamaForPretraining

# PyTorch 2.4-2.6 与 2.7+ 的导入路径不同
try:
    from torch.distributed._composable.fsdp import (
        CPUOffloadPolicy, FSDPModule, MixedPrecisionPolicy, fully_shard,
    )
except ImportError:
    from torch.distributed.fsdp import (
        CPUOffloadPolicy, FSDPModule, MixedPrecisionPolicy, fully_shard,
    )

def _build_model(config: LlamaConfig, args, device, rank):
    with torch.device("meta"):
        model = LlamaForPretraining(config)

    fsdp_kwargs = {"mesh": args.mesh, "reshard_after_forward": True}
    if args.mixed_precision:
        fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
    if args.cpu_offload:
        fsdp_kwargs["offload_policy"] = CPUOffloadPolicy()

    # 自底向上切分：先每个 decoder layer，再 base_model，最后根 model
    for layer in model.base_model.layers:
        fully_shard(layer, **fsdp_kwargs)
    fully_shard(model.base_model, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)

    model.to_empty(device="cpu" if args.cpu_offload else device)
    model.reset_parameters()

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[fsdp2_llama2] {total_params/1e6:.1f}M params")
    return model
```

该函数在 meta 设备上构建模型，按层切分，最后通过 `to_empty` 与 `reset_parameters` 实例化权重。CPU 卸载会把目标设备设为 `"cpu"`。

通过 `print(model)` 可以查看嵌套包装结构。所有 FSDP2 公共 API 均通过 `FSDPModule` 暴露。例如，用户可以调用 `model.unshard()` 手动控制 all-gather 调度。详见下文“显式预取”部分。

注意这里对 `model`、`model.base_model` 以及每个 Transformer 层都调用了 `fully_shard()`。切分顺序必须**自底向上**，顶层最后切分。每个被切分的模块都是一次 all-gather 的单元。

> **补充说明**：顶层调用 `fully_shard()` 是为了包装根节点；前提是子模块已先切分，避免顶层 all-gather 展开整个模型。

以 Llama 结构为例：输入先进入 `base_model` 的嵌入层，再依次经过各 Transformer block。由于每个 block 已单独切分，FSDP 会分别为它们 all-gather。`base_model` 中的嵌入层和 RMSNorm 随 `base_model` 一起展开。顶层 `model` 主要包含预测头。每张 GPU 上只需驻留一个完整 block，以及嵌入层、RMSNorm 和预测头。

你还可以进一步把 `embed_tokens` 单独切分，或把每个 block 拆成注意力与前馈两个子模块再切分，以进一步降低显存。

```python
from torch.distributed.tensor import DTensor

for param in model.parameters():
    assert isinstance(param, DTensor)
    assert param.placements == (Shard(0),)
    # 通过 param.to_local() 查看分片参数

optim = torch.optim.Adam(model.parameters(), lr=1e-2)
```

注意，优化器在应用 `fully_shard` 之后构建。模型和优化器的状态字典均使用 DTensor 表示。

DTensor 简化了优化器、梯度裁剪和检查点操作：

- `torch.optim.Adam` 和 `torch.nn.utils.clip_grad_norm_` 对 DTensor 参数开箱即用。单设备训练和分布式训练的代码保持一致。
- 可使用 DTensor 和 DCP API 操作参数以获取完整状态字典。详见下文“状态字典”部分。对于分布式状态字典，可无需额外通信直接保存 / 加载检查点（[文档](https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html)）。

切分成功的标志是 `isinstance(model, FSDPModule)`。每个自定义模块需实现 `reset_parameters()`，否则 `model.reset_parameters()` 无法正确初始化权重。若从 checkpoint 加载，则跳过随机初始化。

### 4.5 内存调优决策树

```
GPU 内存是否充足？
├── 是 → reshard_after_forward=False（避免反向 all-gather）
└── 否 → 是否有节点内高速互联（NVLink/IB）？
    ├── 是 → reshard_after_forward=intra_node_size（如 8）
    └── 否 → reshard_after_forward=True（最小内存占用）
```

## 5. FSDP 训练循环

`fsdp2_llama2_main.py` 的训练循环与普通模型几乎一致。主要区别在于：使用 `DistributedSampler`、在 forward 前调用 `model.unshard()`、以及根据 CPU 卸载等开关调整梯度裁剪和 `zero_grad`。

```python
import torch.nn.functional as F

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.lr,
    betas=(0.9, 0.99),
    eps=1e-8,
    weight_decay=0.1,
)

num_training_steps = len(dataloader) * args.epochs
# warmup / cosine scheduler 省略
global_step = 0

for epoch in range(args.epochs):
    pbar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}") \
        if rank == 0 else dataloader

    for batch_id, batch in enumerate(pbar):
        # 定期保存 checkpoint
        if global_step > 0 and global_step % args.save_every == 0:
            save_checkpoint_dcp(model, optimizer, scheduler, args.checkpoint_dir)

        # 显式预取：提前触发第 1 个 all-gather
        model.unshard()

        input_ids, target_ids = batch
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        attn_mask = create_causal_mask(input_ids)
        if padding_token_id >= 0:
            attn_mask = attn_mask + create_padding_mask(input_ids, padding_token_id)

        logits = model(input_ids, attn_mask)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1),
            ignore_index=padding_token_id if padding_token_id >= 0 else -100,
        )

        optimizer.zero_grad(set_to_none=False if args.cpu_offload else True)
        loss.backward()
        if not args.cpu_offload:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        global_step += 1
```

> 注：示例中 `model.unshard()` 直接调用以提前触发 all-gather。FSDP2 的 `unshard()` 返回上下文管理器，若需要保持未分片状态，建议使用 `with model.unshard():`。请结合官方文档与具体需求确认用法。

即使不手动调用 `model.unshard()`，前向内部也会自动触发 all-gather。

FSDP 兼具数据并行特点。`fsdp2_llama2_main.py` 中的 `_build_dataloader` 为每个 rank 配置 `DistributedSampler`，并支持真实数据集与合成数据回退：

```python
from torch.utils.data.distributed import DistributedSampler
from data import PretrainingDataset, create_causal_mask, create_padding_mask

class _SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, vocab_size: int, seq_length: int, num_samples: int = 1000):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randint(0, self.vocab_size, (self.seq_length,), dtype=torch.int64)
        y = torch.randint(0, self.vocab_size, (self.seq_length,), dtype=torch.int64)
        return x, y

def _build_dataloader(args, config, rank, world_size):
    if args.use_synthetic_data:
        dataset = _SyntheticDataset(config.vocab_size, args.seq_len)
    else:
        import datasets, tokenizers
        tokenizer = tokenizers.Tokenizer.from_file(args.tokenizer_path)
        dataset = datasets.load_dataset(args.dataset_name, args.dataset_split, split="train")
        dataset = PretrainingDataset(dataset, tokenizer, args.seq_len)

    sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=2,
        prefetch_factor=2,
    )

    padding_token_id = getattr(
        getattr(dataset, "pad", None), "token_to_id", lambda x: -1
    )("[PAD]") if hasattr(dataset, "pad") else -1

    return dataloader, padding_token_id
```

这与普通数据并行中的 DataLoader 设置一致。

## 6. 性能优化手段

### 6.1 混合精度

启用混合精度可加速训练并降低显存，同时对精度影响极小。FSDP2 通过 `MixedPrecisionPolicy` 灵活区分**存储精度**、**计算精度**和**通信精度**。

在 `fsdp2_llama2_main.py` 中，开启 `--mixed-precision` 时，`_build_model` 会构造如下策略并作为 `fsdp_kwargs` 传入每一层 `fully_shard`：

```python
if args.mixed_precision:
    fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )
```

- **Param Dtype（计算）**：在 forward / backward 计算前，FSDP2 自动将参数 cast 为低精度（如 `bfloat16`）。
- **Reduce Dtype（通信）**：在梯度同步（reduce-scatter）阶段，为保证数值稳定性，通常将梯度 cast 为高精度（如 `float32`）进行累加。
- **Buffer 精度**：FSDP2 不对 buffer 进行分片，因此 `MixedPrecisionPolicy` 未单独提供 `buffer_dtype`。如需控制 buffer 精度，可通过其他方式设置。

```python
# FSDP2 混合精度转换流程
mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
# Forward: Parameters (FP32 storage) -> Cast to BF16 -> Compute
# Backward: Gradients (BF16) -> Cast to FP32 -> Reduce-Scatter
```

`param_dtype` 控制参数精度，`reduce_dtype` 控制梯度归约精度。也可通过 `output_dtype` 和 `cast_forward_inputs` 控制前向输入输出类型。由于 `fully_shard()` 按模块调用，不同模块可使用不同策略。

也可全局设置默认数据类型：

```python
torch.set_default_dtype(torch.bfloat16)
```

这会改变后续所有 DTensor 的默认类型。

与 [torch.amp](https://docs.pytorch.org/docs/stable/amp.html) 相比，FSDP2 混合精度具备以下优势：

- **高效且灵活的参数转换**：`FSDPModule` 内的所有参数在模块边界处统一转换（前向 / 反向前后）。可为每层设置不同的混合精度策略。例如，前几层使用 float32，剩余层使用 bfloat16。
- **float32 梯度归约（reduce-scatter）**：不同 rank 的梯度差异可能很大。使用 float32 归约梯度对数值稳定性至关重要。

### 6.2 CPU 卸载

由于 all-gather 前参数并不完整，而进程间通信数据量大、速度较慢，可以把分片模型在不用时放在 CPU 内存中。这就是 **CPU 卸载（CPU offloading）**。它把不用的分片模型状态放在 CPU 内存，以降低 GPU 显存占用。代价是计算时 CPU 与 GPU 之间需要搬运数据。

在 `fsdp2_llama2_main.py` 中，开启 `--cpu-offload` 时，`_build_model` 会把 `offload_policy` 加入 `fsdp_kwargs`，并把 `to_empty` 的目标设备设为 `"cpu"`：

```python
if args.cpu_offload:
    fsdp_kwargs["offload_policy"] = CPUOffloadPolicy()

for layer in model.base_model.layers:
    fully_shard(layer, **fsdp_kwargs)
fully_shard(model.base_model, **fsdp_kwargs)
fully_shard(model, **fsdp_kwargs)

model.to_empty(device="cpu" if args.cpu_offload else device)
model.reset_parameters()
```

CPU 卸载开启后，训练循环会把 `zero_grad(set_to_none=False)`，并跳过梯度裁剪（避免在 CPU 上产生额外同步）：

```python
optimizer.zero_grad(set_to_none=False if args.cpu_offload else True)
loss.backward()
if not args.cpu_offload:
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
```

**CPU 卸载的作用：**

- 将分片参数、梯度和优化器状态保存在 CPU。
- 前向 / 反向时将分片参数拷贝至 GPU，使用后释放。
- 将梯度拷贝回 CPU，并在 CPU 上执行优化器步骤。

**适用场景：**

- GPU 显存受限。
- 模型无法放入单卡显存。

**不适用场景：**

- CPU 内存有限。
- 训练速度优先于显存。

CPU 卸载通常会降低训练速度。启用后建议保留已分配的梯度张量，避免反复分配：

```python
optimizer.zero_grad(set_to_none=False)
```

### 6.3 激活检查点

FSDP 已比普通数据并行更省显存。如需进一步降低显存，可结合**激活检查点（activation checkpointing）**：

```python
import functools

if args.activation_checkpointing:
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        apply_activation_checkpointing,
        checkpoint_wrapper,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from llama2 import LlamaDecoderLayer

    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={LlamaDecoderLayer, nn.Embedding},
    )
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=checkpoint_wrapper,
        auto_wrap_policy=wrap_policy,
    )
```

`wrap_policy` 判断模块是否属于指定类别，若是则对其应用激活检查点。前向时丢弃内部激活值，反向时重新计算，以时间换空间。

> **补充说明**：上述 `apply_activation_checkpointing` 与 `checkpoint_wrapper` 来自 `torch.distributed.algorithms._checkpoint.checkpoint_wrapper`，主要面向 FSDP1。在 FSDP2 中，更常见的做法是直接使用 `torch.utils.checkpoint.checkpoint_wrapper` 或 `torch.utils.checkpoint.checkpoint` 包裹子模块。

### 6.4 torch.compile()

如果模型支持编译，可在切分后用 `torch.compile()` 编译 FSDP 模型。这样编译图引用的是 DTensor，而非普通 Tensor。

```python
# 先完成分片
fully_shard(model)
model.to_empty(device=device)
model.reset_parameters()

# 创建 DataLoader、优化器、学习率调度器、损失函数 ...

# 再编译
model = torch.compile(model)
loss_fn = torch.compile(loss_fn)

for epoch in range(epochs):
    for batch in dataloader:
        ...
        logits = model(input_ids, attn_mask)
        loss = loss_fn(logits.view(-1, logits.size(-1)), target_ids.view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
```

### 6.5 通信与计算重叠

为了极致的训练效率，FSDP2 实现了高度优化的**通信与计算重叠（communication-computation overlap）**机制，即**预取（prefetching）**。为了降低每步延迟，PyTorch 会在当前模块计算的同时预取下一模块的分片。这种预取让通信与计算重叠。部分 FSDP 配置会占用更多显存以换取更大重叠，需要在显存与速度之间权衡。

`fsdp2_llama2_main.py` 通过 `--explicit-prefetching N` 启用显式预取。$N$ 表示每层额外预取的层数：

```python
if args.explicit_prefetching > 0:
    modules = list(model.base_model.layers)
    num_pf = args.explicit_prefetching
    for i, m in enumerate(modules):
        if i < len(modules) - 1:
            m.set_modules_to_forward_prefetch(
                modules[i + 1 : i + 1 + num_pf]
            )
        if i > 0:
            m.set_modules_to_backward_prefetch(
                modules[max(0, i - num_pf) : i]
            )
```

同时，训练循环在 forward 前调用 `model.unshard()`，把第 1 个 all-gather 与数据搬运等前置计算重叠：

```python
model.unshard()
input_ids, target_ids = batch
input_ids = input_ids.to(device)
...
logits = model(input_ids, attn_mask)
```

`modules` 必须按执行顺序排列。预取顺序错误会降低性能。同时，不能指定未被切分的模块（如 `model.lm_head`），否则 FSDP 无法为它发起 all-gather。

`fully_shard` 注册前向 / 反向钩子，在计算前 all-gather 参数，在计算后重新分片参数。为了将 all-gather 与计算重叠，FSDP2 提供开箱即用的**隐式预取**，以及供高级用户手动控制 all-gather 调度的**显式预取**。

**隐式预取**：CPU 线程在层 $i$ 计算前发出层 $i$ 的 all-gather。All-gather 被排入独立的 CUDA stream，而层 $i$ 的计算在默认 stream 中执行。对于非 CPU 密集型工作负载（如大 batch size 的 Transformer），层 $i+1$ 的 all-gather 可与层 $i$ 的计算重叠。隐式预取在反向传播中的工作方式类似，只是 all-gather 按前向传播顺序的逆序发出。

[![FSDP 隐式预取](https://docs.pytorch.org/tutorials/_images/fsdp_implicit.png)](https://docs.pytorch.org/tutorials/_images/fsdp_implicit.png)

建议用户从隐式预取开始，了解开箱即用的性能表现。

**显式预取**：用户可通过 `set_modules_to_forward_prefetch` 指定前向顺序，通过 `set_modules_to_backward_prefetch` 指定反向顺序。如下代码所示，CPU 线程在层 $i$ 处发出层 $i+1$ 和 $i+2$ 的 all-gather。

显式预取在以下场景表现良好：

- **CPU 密集型工作负载**：若使用隐式预取，CPU 线程在层 $i$ 的 kernel 执行时太慢，无法为层 $i+1$ 发出 all-gather。必须显式在层 $i$ 前向运行前发出层 $i+1$ 的 all-gather。
- **预取 2 层以上**：隐式预取每次仅 all-gather 下一层，以保持内存占用最小。显式预取可一次 all-gather 多层，以可能获得更好的性能，代价是增加内存。详见代码中的 `layers_to_prefetch`。
- **提前发出第 1 个 all-gather**：隐式预取在调用 `model(x)` 时发生。第 1 个 all-gather 被暴露。可提前显式调用 `model.unshard()` 以提前发出第 1 个 all-gather。

### 6.6 梯度裁剪与 DTensor 优化器

```python
# 基于 DTensor 模型参数构建优化器
optim = torch.optim.Adam(model.parameters(), lr=1e-2)
for _ in range(epochs):
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    loss = model(x).sum()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
    optim.step()
    optim.zero_grad()
```

优化器在模型上应用 `fully_shard` 后初始化，持有对 DTensor `model.parameters()` 的引用。对于梯度裁剪，`torch.nn.utils.clip_grad_norm_` 对 DTensor 参数开箱即用。DTensor 内部会正确分发张量操作，以跨 rank 通信部分张量，保持单设备语义。

## 7. 分布式 Checkpoint

FSDP2 的 `.state_dict()` 默认返回 **Sharded State Dict**（值为 DTensor）。这种表示可直接保存 / 加载，无需额外通信。如果需要完整状态字典（full state dict），可通过 DTensor 的 `full_tensor()` 手动转换，或在 DCP 中设置 `full_state_dict=True`。

本节展示两种工作流：

- 第 1 次运行：创建模型和优化器的检查点。
- 第 2 次运行：从上一个检查点加载以恢复训练。

### 7.1 手动保存与加载

FSDP 模型本质仍是 PyTorch 模型，只是权重被替换为 DTensor。

```python
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import Shard

assert isinstance(model, FSDPModule)
assert isinstance(model, LlamaForPretraining)

for param in model.parameters():
    # DTensor 应该有 placement
    assert param.placements == (Shard(0),)
    # DTensor 与原张量 dtype 相同
    assert param.dtype == torch.float32
    # 查看当前分片内容
    print(param.get_local_tensor())
```

手动保存时，应只让 rank 0 将完整状态写入文件。

**保存状态字典**：`model.state_dict()` 返回 DTensor 状态字典。可通过调用 `full_tensor()` 将 DTensor 转换为普通 `torch.Tensor`。内部会发起跨 rank 的 all-gather 以获取未分片参数。对于 rank 0，`full_param.cpu()` 逐个将张量卸载到 CPU，避免未分片参数峰值占用 GPU 内存。

```python
# 仅在 rank 0 保存
if torch.distributed.get_rank() == 0:
    sharded_state_dict = model.state_dict()     # 值为 DTensor
    full_state_dict = {}                        # 值为普通 CPU Tensor
    for param_name, sharded_param in sharded_state_dict.items():
        full_param = sharded_param.full_tensor()
        full_state_dict[param_name] = full_param.cpu()
    torch.save(full_state_dict, "model.pth")
```

**加载状态字典**：在 meta device 下初始化模型并调用 `fully_shard`，将 `model.parameters()` 从普通 `torch.Tensor` 转换为 DTensor。从 `torch.load` 读取完整状态字典后，可调用 `distribute_tensor` 将普通 `torch.Tensor` 转换为 DTensor，使用与 `model.state_dict()` 相同的 placements 和 device mesh。最后可调用 `model.load_state_dict` 将 DTensor 状态字典加载到模型。

```python
# 所有进程一起加载
from torch.distributed.tensor import distribute_tensor

dist.barrier()
full_state_dict = torch.load("model.pth", map_location="cpu", mmap=True)
meta_sharded_state_dict = model.state_dict()    # meta 设备上的 FSDPModule 状态字典
sharded_state_dict = {}

for param_name, full_tensor in full_state_dict.items():
    sharded_meta_param = meta_sharded_state_dict.get(param_name)
    dtensor = distribute_tensor(
        full_tensor,
        sharded_meta_param.device_mesh,
        sharded_meta_param.placements,
    )
    sharded_state_dict[param_name] = nn.Parameter(dtensor)

# 必须使用 assign=True，以将 meta 设备上的张量替换为实际 DTensor
model.load_state_dict(sharded_state_dict, strict=False, assign=True)
dist.barrier()
```

优化器状态字典的工作方式类似（[代码](https://github.com/pytorch/examples/blob/70922969e70218458d2a945bf86fd8cc967fc6ea/distributed/FSDP2/checkpoint.py#L156)）。用户可自定义上述 DTensor 脚本以适配第三方检查点。

若无需自定义，可直接使用 [DCP API](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html) 以支持单节点和多节点训练。

### 7.2 使用 Stateful 包装器

为简化分布式 checkpoint 管理，可使用 PyTorch 的 `Stateful` API：

```python
# PyTorch 分布式检查点（DCP）导入
from torch.distributed.checkpoint.state_dict import (
    get_state_dict,
    set_state_dict,
    get_model_state_dict,
    StateDictOptions
)
from torch.distributed.checkpoint.stateful import Stateful


class AppState(Stateful):
    """用于检查点应用状态的包装器。由于该对象符合 Stateful 协议，
    PyTorch DCP 在 dcp.save/load API 中自动调用 state_dict/load_state_dict。

    注意：该包装器用于处理模型和优化器上的分布式状态字典方法。
    """

    def __init__(self, model, optimizer=None, epoch=None):
        self.model = model
        self.optimizer = optimizer
        self.epoch = epoch

    def state_dict(self):
        # 自动管理 FSDP2 的 FQN，并将默认状态字典类型设置为 SHARDED_STATE_DICT
        model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizer)
        return {
            "model": model_state_dict,
            "optim": optimizer_state_dict,
            "epoch": self.epoch
        }

    def load_state_dict(self, state_dict):
        # 加载完成后设置模型和优化器的状态字典
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"],
        )
        # 加载轮次信息（如果可用）
        if "epoch" in state_dict:
            self.epoch = state_dict["epoch"]
```

### 7.3 使用分布式 Checkpoint API

`fsdp2_llama2_main.py` 直接调用同一目录下 `checkpoint.py` 中的 `save_checkpoint_dcp` 与 `load_checkpoint_dcp`：

```python
from checkpoint import load_checkpoint_dcp, save_checkpoint_dcp

# 训练前：若存在 checkpoint 则恢复
if os.path.exists(args.checkpoint_dir):
    load_checkpoint_dcp(model, optimizer, scheduler, args.checkpoint_dir)

# 训练循环中定期保存
if global_step > 0 and global_step % args.save_every == 0:
    save_checkpoint_dcp(model, optimizer, scheduler, args.checkpoint_dir)

# 训练结束后再保存一次
save_checkpoint_dcp(model, optimizer, scheduler, args.checkpoint_dir)
```

`checkpoint.py` 中 DCP 辅助函数的核心逻辑如下：

```python
import torch.distributed as dist
from torch.distributed.checkpoint import load, save
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions, get_state_dict, set_state_dict,
)

def load_checkpoint_dcp(model, optimizer, scheduler=None, checkpoint_dir="checkpoint-dist"):
    dist.barrier()
    model_state, optimizer_state = get_state_dict(
        model, optimizer,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True),
    )
    load(
        {"model": model_state, "optimizer": optimizer_state},
        checkpoint_id=checkpoint_dir,
    )
    set_state_dict(
        model, optimizer,
        model_state_dict=model_state,
        optim_state_dict=optimizer_state,
        options=StateDictOptions(
            broadcast_from_rank0=True, full_state_dict=True, cpu_offload=True
        ),
    )
    if scheduler is not None and os.path.exists(f"{checkpoint_dir}/lrscheduler.pt"):
        scheduler.load_state_dict(
            torch.load(
                f"{checkpoint_dir}/lrscheduler.pt",
                map_location="cpu",
                weights_only=True,
            )
        )
    dist.barrier()

def save_checkpoint_dcp(model, optimizer, scheduler=None, checkpoint_dir="checkpoint-dist"):
    dist.barrier()
    model_state, optimizer_state = get_state_dict(
        model, optimizer,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True),
    )
    save(
        {"model": model_state, "optimizer": optimizer_state},
        checkpoint_id=checkpoint_dir,
    )
    if scheduler is not None and dist.get_rank() == 0:
        torch.save(scheduler.state_dict(), f"{checkpoint_dir}/lrscheduler.pt")
    dist.barrier()
```

> **补充说明**：`StateDictOptions` 中的 `cpu_offload` 仅控制保存 / 加载状态字典时是否先卸载到 CPU，与 FSDP 是否启用 `CPUOffloadPolicy` 相互独立。即使未启用 CPU 卸载，也可以保留该选项以减少显存峰值。

关于手动 DTensor 方式以及优化器状态字典的加载 / 保存细节，请参阅 `examples/FSDP2/checkpoint.py`。

## 8. FSDP1 迁移至 FSDP2 指南

以下对比 FSDP1 与等效的 `fully_shard` 用法，并重点说明关键差异及迁移步骤。完整可运行示例参见 `examples/FSDP2/fsdp2_llama2_main.py`。

**原始 FSDP() 用法**：

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

with torch.device("meta"):
    model = Transformer()
policy = ModuleWrapPolicy({TransformerBlock})
model = FSDP(model, auto_wrap_policy=policy)

def param_init_fn(module: nn.Module) -> None: ...

model = FSDP(model, auto_wrap_policy=policy, param_init_fn=param_init_fn)
```

**新的 fully_shard() 用法**：

```python
with torch.device("meta"):
    model = Transformer()

for module in model.modules():
    if isinstance(module, TransformerBlock):
        fully_shard(module)
fully_shard(model)

for tensor in itertools.chain(model.parameters(), model.buffers()):
    assert tensor.device == torch.device("meta")

# 分片后初始化模型
model.to_empty(device="cuda")
model.reset_parameters()
```

### 8.1 迁移步骤

1. 替换导入语句。
2. 直接实现“策略”：对目标子层应用 `fully_shard`。
3. 用 `fully_shard` 包装根模型，替代 `FSDP`。
4. 移除 `param_init_fn`，手动调用 `model.reset_parameters()`。
5. 替换其他 FSDP1 参数（见下表）。

### 8.2 参数映射表

| FSDP1 参数 | FSDP2 等效配置 |
| --- | --- |
| `sharding_strategy` | |
| - `FULL_SHARD` | `reshard_after_forward=True` |
| - `SHARD_GRAD_OP` | `reshard_after_forward=False` |
| - `HYBRID_SHARD` | `reshard_after_forward=True` + 2D device mesh |
| - `_HYBRID_SHARD_ZERO2` | `reshard_after_forward=False` + 2D device mesh |
| `cpu_offload` | |
| - `CPUOffload.offload_params=False` | `offload_policy=None` |
| - `CPUOffload.offload_params=True` | `offload_policy=CPUOffloadPolicy()` |
| `backward_prefetch` | |
| - `BACKWARD_PRE` | 始终使用 |
| - `BACKWARD_POST` | 不支持 |
| `mixed_precision` | |
| - `buffer_dtype` | 已省略，因为 `fully_shard` 不分片 buffer |
| - `cast_forward_inputs` | `fully_shard` 的 `cast_forward_inputs` 同时映射到 FSDP1 的 `cast_forward_inputs` 和 `cast_root_forward_inputs` |
| - `output_dtype` | `fully_shard` 的新配置 |
| `device_id` | 从 device_mesh 的 device 推断 |
| `sync_module_states` | 移至 DCP。用户可使用 `set_model_state_dict` 配合 `broadcast_from_rank0=True` 从 rank 0 广播状态字典 |
| `forward_prefetch` | 手动控制预取： |
| | - 手动调用 `fsdp_module.unshard()` |
| | - 使用 `set_modules_to_forward_prefetch` 和 `set_modules_to_backward_prefetch` 控制自动预取 |
| `limit_all_gathers` | 不再需要，因为 `fully_shard` 已移除 CPU 同步 |
| `use_orig_params` | 始终使用原始参数（不再有扁平参数） |
| `no_sync()` | `set_requires_gradient_sync` |
| `ignored_params` / `ignored_states` | `ignored_params` |

## 9. 总结

本文介绍了如何使用 PyTorch FSDP2 在多 GPU 上训练大模型：

- FSDP 通过分片参数降低单卡显存，代价是更多的 all-gather 与 reduce-scatter 通信。
- 使用 `fully_shard()` 自底向上切分子模块和顶层模块。
- 训练循环几乎与普通模型相同，只需注意 `DistributedSampler` 和显式 `model.unshard()`。
- 可以通过混合精度、CPU 卸载、`reshard_after_forward`、激活检查点、`torch.compile()` 和预取等手段在显存与速度之间权衡。
- 模型保存既可以用 DTensor 手动合并，也可以使用 PyTorch 分布式 checkpoint API。

理解这些机制后，你就可以根据模型规模和硬件条件，灵活调整 FSDP 配置，高效完成大模型训练。

## 10. 参考链接

- [PyTorch FSDP2 官方文档](https://docs.pytorch.org/docs/2.7/distributed.fsdp.fully_shard.html)
- [PyTorch Distributed Checkpoint](https://docs.pytorch.org/distributed.checkpoint.html)
- [DTensor 官方文档](https://docs.pytorch.org/docs/stable/distributed.tensor.html)
- [FSDP1 官方文档](https://docs.pytorch.org/docs/stable/fsdp.html)
- [PyTorch 设备网格文档](https://docs.pytorch.org/tutorials/recipes/distributed_device_mesh.html)
- [分布式 Checkpoint 配方](https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html)
