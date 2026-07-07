# FSDP2 完全指南

## FSDP 工作原理

在 **DistributedDataParallel (DDP)** 训练中，每个 Rank 都拥有一个完整的模型副本，并处理一个独立的数据 Batch，最后通过 **All-Reduce** 在所有 Rank 间同步梯度。

DDP 虽然非常流行，却浪费了 GPU 显存，因为模型权重和优化器状态在所有 DDP worker 上都被完整复制。

减少冗余的一种方法是**全参数分片（full parameter sharding）**， 与 DDP 相比，**FSDP (Fully Sharded Data Parallel)** 通过对模型参数、梯度和优化器状态进行**切片 (Sharding)**，显著降低了显存占用。这使得在单卡显存受限的情况下训练超大模型成为可能。如下图所示，

FSDP 参数生命周期：

[![FSDP workflow](https://docs.pytorch.org/tutorials/_images/fsdp_workflow.png)](https://docs.pytorch.org/tutorials/_images/fsdp_workflow.png)



1. **Fully Sharded (静止态)**：在 Forward 和 Backward 计算之外，参数处于完全分片状态（每张卡只存 1/N）。
2. **All-Gather (准备态)**：在 Forward 和 Backward 开始前，分片参数通过 all-gather 聚合为完整的参数。
3. **Compute (计算态)**：使用完整参数进行计算。
4. **Reduce-Scatter (同步态)**：在 Backward 内部，计算出的完整梯度被立即归约并切分（Reduce-Scatter）为分片梯度。
5. **Update (更新态)**：优化器使用切片梯度更新切片参数，因此优化器状态也是切分的。



如下图所示，FSDP 将 DDP 的 All-Reduce 操作分解为 Reduce-Scatter 和 All-Gather：

[![FSDP all-gather and reduce-scatter](https://docs.pytorch.org/tutorials/_images/fsdp_sharding.png)](https://docs.pytorch.org/tutorials/_images/fsdp_sharding.png)



> 标准的 all-reduce 操作用于聚合梯度，可以分解为两个独立的阶段：reduce-scatter 和 all-gather。
>
> 在 reduce-scatter 阶段，梯度会根据其 rank 索引在每个 GPU 上按 rank 分块求和。在 all-gather 阶段，每个 GPU 上可用的聚合梯度的分片部分将提供给所有 GPU。

重新排列这两个操作后，每个 DDP worker 只需保存一份参数和优化器状态的分片。

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



## FSDP 工作流程

下图对比了标准 DDP 与 FSDP：

- **标准数据并行**：每张 GPU 上都有完整模型副本，前向与反向只处理数据分片。本地计算完成后，各 GPU 共享参数与优化器状态，以计算全局权重更新。
- **FSDP**：每张 GPU 上只保存模型分片。本地计算前，通过 all-gather 从其他 GPU 收集完整权重以完成前向传播；反向传播前再次收集权重。反向完成后，本地梯度经平均并通过 reduce-scatter 分片到各 GPU，每张 GPU 更新自己的权重分片。

<img src="https://engineering.fb.com/wp-content/uploads/2021/07/FSDP-Graph-2.png?w=907" alt="Full Sharded Data Parallel graph" style="zoom:80%;" />



FSDP 的典型执行流程如下：

1. 多个进程同时运行，可能分布在多台机器上。每个进程（也就是每张 GPU）只保存模型的一个分片。
2. 分片后，每层权重以 **DTensor**（分布式张量）形式存在，而不是普通 Tensor。因此没有任何一个进程能独立完成某个模块的计算。
3. 每次运算前，FSDP 发起一次 all-gather，让各进程交换该模块的分片，临时恢复出完整模块。
4. 每个进程用自己的微批次（micro-batch）在这个临时模块上执行前向传播，然后丢弃临时模块，继续处理下一层。
5. 反向传播时，FSDP 同样需要对每层 all-gather 解除分片，再计算梯度。
6. 由于每个进程处理的微批次不同，各自算出的梯度也不同。FSDP 通过 reduce-scatter 交换并平均梯度，得到全局梯度，再更新各自的分片。

由此可见，FSDP 的通信和流程比普通数据并行更复杂。但正因为模型被分散到多张 GPU，训练超大模型所需的单卡显存显著下降，这正是使用 FSDP 的动机。

为了最大限度地提高内存效率，我们可以在每一层前向传播结束后丢弃全部权重，从而为后续层节省内存。这可以通过将 FSDP 包装器应用于网络的每一层（并设置 reshard_after_forward=True ）来实现。

伪代码如下：

```python
FSDP forward pass:
    for layer_i in layers:
        all-gather full weights for layer_i
        forward pass for layer_i
        discard full weights for layer_i

FSDP backward pass:
    for layer_i in layers:
        all-gather full weights for layer_i
        backward pass for layer_i
        discard full weights for layer_i
        reduce-scatter gradients for layer_i
```

##  为什么需要 FSDP2？

FSDP2 是 PyTorch 分布式并行的下一代范式，旨在解决 FSDP1（`FlatParameter` 包装器模式）在灵活性与组合性上的痛点。它不再通过 Python 类包装模型，而是通过 **`torch.distributed.fsdp.fully_shard`** API 对模型进行原位（In-place）的并行化处理。

### 1.1 现有 FSDP 的局限性

PyTorch 现有的 `FullyShardedDataParallel`（FSDP）通过将多层参数"扁平化"（flatten）为单个 `FlatParameter` 来实现分片。这种设计在训练大模型时有效，但随着训练规模和技术的发展，逐渐暴露出几个关键问题：

- **FP8 支持受限**：FP8 权重和非 FP8 参数无法在同一个 all-gather 中混合处理
- **冻结参数处理复杂**：冻结参数和非冻结参数难以在同一个通信组中灵活共存
- **检查点保存开销大**：训练表示与状态字典表示不一致，需要额外的通信转换
- **编译器优化困难**：`torch.compile` 等图编译器难以对扁平化参数进行通信优化

### 1.2 核心设计思想：逐参数分片

FSDP2 的核心思想极其简洁——**将每个参数独立在第 0 维进行分片**，而非将多个参数扁平化后统一分片。这种"逐参数分片"（Per-Parameter-Sharding）的设计带来了四个关键优势：

| 特性                | FSDP1（扁平参数）      | FSDP2（逐参数分片）    |
| ------------------- | ---------------------- | ---------------------- |
| FP8 混合 all-gather | ❌ 不支持               | ✅ 灵活混合             |
| 冻结参数处理        | ❌ 需要额外内存         | ✅ 同一通信组无额外内存 |
| 检查点保存          | ❌ 需要 all-gather 转换 | ✅ 无需通信             |
| 编译器优化          | ❌ 难以优化             | ✅ 可调整通信组         |

###  1.3 FSDP2 的优势

与 [FSDP1](https://docs.pytorch.org/docs/stable/fsdp.html) 相比，FSDP2 具备以下优势：

- 与 FSDP1 将多个参数打平（Flatten）为一个巨大的 `FlatParameter` 不同，FSDP2 采用了 **Per-Parameter Sharding（逐参数分片）** 的策略。

  - **FSDP1 (Legacy)**：将层内的所有参数拉平拼接，切分这个巨大的 1D 向量。这破坏了原始模型的参数结构，导致对参数的某些操作（如自定义初始化、特定层微调）变得复杂。

  - **FSDP2 (New)**：保持模型原始的参数结构不变。每个参数（`nn.Parameter`）被单独切分并管理。这种设计使得 FSDP2 具有极高的组合性（Composability），可以轻松与 Tensor Parallel (TP) 或 Checkpointing 结合。

- 改进内存管理系统，通过避免使用 `recordStream`（[文档](https://dev-discuss.pytorch.org/t/fsdp-cudacachingallocator-an-outsider-newb-perspective/1486)）实现更低且确定性的 GPU 内存占用，且无需任何 CPU 同步。
- 提供张量子类扩展点以自定义 all-gather，例如为 float8 线性层启用 float8 all-gather（[文档](https://dev-discuss.pytorch.org/t/enabling-float8-all-gather-in-fsdp2/2359)），以及为 QLoRA 支持 NF4（[文档](https://github.com/pytorch/torchtune/blob/main/README.md)）。
- 冻结参数和非冻结参数可在同一通信组中混合，且不占用额外内存。

## API 设计：从"包装器"到"函数式"

### 2.1 全新的 `fully_shard` API

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

### 2.2 关键参数解析

#### `reshard_after_forward`：内存与通信的权衡艺术

`fully_shard()` 的 `reshard_after_forward` 参数控制前向结束后是否释放 all-gather 得到的完整参数。这是 FSDP2 最灵活的参数之一，直接对标 DeepSpeed ZeRO 的不同阶段：

```python
# ZeRO-3 模式：前向后释放参数，反向时重新 all-gather
fully_shard(module, reshard_after_forward=True)

# ZeRO-2 模式：前向保持参数，反向无需 all-gather
fully_shard(module, reshard_after_forward=False)

# ZeRO++ hpZ 模式：前向后部分重分片（如节点内大小）
fully_shard(module, reshard_after_forward=8)  # 8 卡节点内保持
```

| 值      | 对应 ZeRO  | 内存占用 | 反向通信        |
| ------- | ---------- | -------- | --------------- |
| `True`  | ZeRO-3     | 最低     | 需要 all-gather |
| `False` | ZeRO-2     | 较高     | 无需 all-gather |
| `int`   | ZeRO++ hpZ | 中等     | 部分 all-gather |

> **补充说明**：
>
> - 在 PyTorch FSDP2 中，`reshard_after_forward` 默认为 `True`，即前向结束后释放参数。
> - 设为 `False` 可让对应模块在前后向之间保持未分片，节省一次 all-gather，但会增加峰值显存。

理解该参数有助于模型结构设计。例如，在 `LlamaForPretraining` 中，根模块只包含预测头；若把嵌入层也移到根模块，前向结束后嵌入层会一直驻留显存，可能造成浪费。

#### `mp_policy`：细粒度混合精度控制

```python
@dataclass
class MixedPrecisionPolicy:
    param_dtype: Optional[torch.dtype] = None      # 参数/计算精度
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



对于 Transformer 架构，推荐对每个 Transformer 块和根模块分别应用：

```python
for module in model.modules():
    if isinstance(module, TransformerBlock):
        fully_shard(module, mesh=mesh, mp_policy=mp_policy)
fully_shard(model, mesh=mesh, mp_policy=mp_policy)
```

这种"分块+根"的模式构造了多个通信组，是实现通信与计算重叠的关键。



<img src="https://private-user-images.githubusercontent.com/31054793/416243005-96eb3640-d995-446c-8a47-457569fbb16e.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3ODM0MTI4MDQsIm5iZiI6MTc4MzQxMjUwNCwicGF0aCI6Ii8zMTA1NDc5My80MTYyNDMwMDUtOTZlYjM2NDAtZDk5NS00NDZjLThhNDctNDU3NTY5ZmJiMTZlLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjA3MDclMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwNzA3VDA4MjE0NFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWNmZWQ4MzI2Zjc0NjM3MjFjMjhhYzE4MGRhMTEyY2JhOGNlZGE1ODFkODQzYjgxMzllYmZiMjA4MWY3M2ViMzcmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JnJlc3BvbnNlLWNvbnRlbnQtdHlwZT1pbWFnZSUyRnBuZyJ9.S-ap7hUG-GF2ftTM2spLqJzLJmVZqsu93jhg6VvlqWQ" alt="Image" style="zoom:50%;" />

##  FSDP2 的使用方法

PyTorch FSDP2 的 `fully_shard` 支持多种切分粒度。最细粒度可按层切分，最大限度降低峰值显存；粒度越粗，通信开销通常越低。需要根据实际场景选择。

除切分粒度外，FSDP2 还提供多种配置优化性能并缓解 OOM。

### DTensor

FSDP2 的底层基石是 **DTensor (`torch.distributed.tensor.DTensor`)**。

**逻辑视图与物理视图分离**：

- **逻辑上**：参数看起来仍然是一个完整的 Tensor（例如 `[4096, 4096]`），保持了与单卡训练一致的编程体验。
- **物理上**：参数实际上被切分并分布在 `DeviceMesh` 定义的设备组中（例如每张卡只持有 `[512, 4096]` 的 `Local Tensor`）。

###  DeviceMesh 配置

**DeviceMesh**：FSDP2 依赖 `DeviceMesh` 来描述设备的拓扑结构。这使得它天然支持多维并行（例如 2D FSDP，或 FSDP + TP），只需定义不同的 Mesh 维度即可。

`init_device_mesh` 用于定义训练设备的拓扑结构。本示例使用一维 mesh 实现数据并行；`DeviceMesh` 也支持多维并行，如张量并行、流水线并行。组合多种并行方式通常能进一步提升训练性能。

更多信息可参考 [PyTorch 设备网格文档](https://docs.pytorch.org/tutorials/recipes/distributed_device_mesh.html)。

### 初始化进程组

多进程训练需要通过 `torchrun` 启动。每个进程会看到 world size、rank 和 local rank。脚本中需初始化进程组：

```python
import torch.distributed as dist

# 初始化分布式环境
dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
device = torch.device(f"cuda:{local_rank}")
rank = dist.get_rank()
world_size = dist.get_world_size()
print(f"World size {world_size}, rank {rank}, local rank {local_rank}. Using {device}")
```

###  模型初始化

**对子模块应用 `fully_shard`**：与 DDP 不同，我们需要对子模块以及根模型都应用 [fully_shard](https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html)。`fully_shard()` 会原地把普通 Tensor 参数替换为 DTensor，并修改模块使其在计算前自动 all-gather。

以下面的 Transformer 为例，我们先对每个层应用 `fully_shard`，再对根模型应用：

- 在计算 `layers[i]` 的前向传播时，其余层保持分片以降低内存占用。
- 在 `fully_shard(model)` 内部，FSDP2 排除来自 `model.layers` 的参数，并将剩余参数归类为一个参数组，以高效执行 all-gather 和 reduce-scatter。
- `fully_shard` 将分片模型移动到实际训练设备（如 `cuda`）。

下面基于 Llama 架构演示：

```python
from torch.distributed.fsdp import FSDPModule, fully_shard

with torch.device("meta"):
    model_config = LlamaConfig()
    model = LlamaForPretraining(model_config)

for layer in model.base_model.layers:
    fully_shard(layer)
fully_shard(model.base_model)
fully_shard(model)

print(model)

model.to_empty(device=device)
model.reset_parameters()

assert isinstance(model, FSDPModule), f"Expected FSDPModule, got {type(model)}"
```

通过 `print(model)` 可以查看嵌套包装结构。`FSDPTransformer` 是 [Transformer](https://github.com/pytorch/examples/blob/70922969e70218458d2a945bf86fd8cc967fc6ea/distributed/FSDP2/model.py#L100) 和 [FSDPModule](https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html#torch.distributed.fsdp.FSDPModule) 的联合类。`FSDPTransformerBlock` 同理。所有 FSDP2 公共 API 均通过 `FSDPModule` 暴露。例如，用户可以调用 `model.unshard()` 手动控制 all-gather 调度。详见下文"显式预取"部分。

**`model.parameters()` 作为 DTensor**：`fully_shard` 跨 rank 分片参数，并将 `model.parameters()` 从普通 `torch.Tensor` 转换为 DTensor 以表示分片参数。FSDP2 默认在 dim-0 上分片，因此 DTensor 的 placements 为 `Shard(dim=0)`。假设有  $N$ 个 rank，分片前某参数有 $N$ 行。分片后，每个 rank 持有该参数的 1 行。可通过 `param.to_local()` 查看分片参数。

注意这里对 `model`、`model.base_model` 以及每个 Transformer 层都调用了 `fully_shard()`。切分顺序必须**自底向上**，顶层最后切分。每个被切分的模块都是一次 all-gather 的单元。

> **补充说明**：顶层调用 `fully_shard()` 是为了包装根节点；前提是子模块已先切分，避免顶层 all-gather 展开整个模型。

以 Llama 结构为例：输入先进入 `base_model` 的嵌入层，再依次经过各 Transformer block。由于每个 block 已单独切分，FSDP 会分别为它们 all-gather；`base_model` 中的嵌入层和 RMSNorm 随 `base_model` 一起展开；顶层 `model` 主要包含预测头。每张 GPU 上只需驻留一个完整 block，以及嵌入层、RMSNorm 和预测头。

你可以进一步把 `embed_tokens` 单独切分，或把每个 block 拆成注意力与前馈两个子模块再切分，以进一步降低显存。



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

- `torch.optim.Adam` 和 `torch.nn.utils.clip_grad_norm_` 对 DTensor 参数开箱即用。这使得单设备训练和分布式训练的代码保持一致。
- 可使用 DTensor 和 DCP API 操作参数以获取完整状态字典。详见下文"状态字典"部分。对于分布式状态字典，可无需额外通信直接保存 / 加载检查点（[文档](https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html)）。

### 在 meta 设备上初始化

切分后，用 `model.to_empty(device=device)` 把模型从 meta 设备转到 GPU，再调用 `model.reset_parameters()` 初始化权重。若从 checkpoint 加载，则跳过随机初始化。

每个自定义模块需实现 `reset_parameters()`。切分成功的标志是 `isinstance(model, FSDPModule)`。之后可像普通模型一样创建优化器，PyTorch 优化器对 DTensor 和 Tensor 的更新方式相同。

## 3. FSDP 训练循环

使用 FSDP 后，训练循环几乎不变：

```python
optimizer = torch.optim.AdamW(
    model.parameters(), lr=learning_rate, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.1,
)
warmup_scheduler = lr_scheduler.LinearLR(
    optimizer,
    start_factor=0.1, end_factor=1.0, total_iters=num_warmup_steps,
)
cosine_scheduler = lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=num_training_steps - num_warmup_steps,
    eta_min=0,
)
scheduler = lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[num_warmup_steps],
)
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)

# 开始训练
for epoch in range(epochs):
    pbar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
    for batch_id, batch in enumerate(pbar):
        # 在前向之前主动解除分片，并作为上下文保持未分片状态
        with model.unshard():
            input_ids, target_ids = batch
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            attn_mask = create_causal_mask(input_ids) + create_padding_mask(input_ids, PAD_TOKEN_ID)

            logits = model(input_ids, attn_mask)
            loss = loss_fn(logits.view(-1, logits.size(-1)), target_ids.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
```

唯一变化是可用 `model.unshard()` 提前触发 all-gather。即使不调用，前向内部也会自动触发。

FSDP 兼具数据并行特点。与分布式数据并行一样，需为 DataLoader 配置 `DistributedSampler`，让每个 rank 处理不同微批次：

```python
dataset = PretrainingDataset(dataset, tokenizer, seq_length)
sampler = DistributedSampler(dataset, shuffle=False)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=sampler,
    pin_memory=True,
    shuffle=False,
    num_workers=2,
    prefetch_factor=2,
)
```

这与普通数据并行中的 DataLoader 设置一致。

## 4. 性能优化手段

### 4.1 混合精度

启用混合精度可加速训练并降低显存，同时对精度影响极小。FSDP2 通过 `MixedPrecisionPolicy` 提供了灵活的精度控制，严格区分 **存储精度**、**计算精度**和**通信精度**。

```python
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

with torch.device("meta"):
    model_config = LlamaConfig()
    model = LlamaForPretraining(model_config)

mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
)

for layer in model.base_model.layers:
    fully_shard(layer, mp_policy=mp_policy)
fully_shard(model.base_model, mp_policy=mp_policy)
fully_shard(model, mp_policy=mp_policy)

model.to_empty(device=device)
```

- **Param Dtype (计算)**：在 Forward/Backward 计算前，FSDP2 会自动将参数 Cast 为低精度（如 `bfloat16`）。
- **Reduce Dtype (通信)**：在梯度同步（Reduce-Scatter）阶段，为了保证数值稳定性，通常将梯度 Cast 为高精度（如 `float32`）进行累加。
- **Buffer Dtype**：独立控制 Buffer（如 BatchNorm 统计量）的精度，防止溢出。

```python
# FSDP2 混合精度转换流程
mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
# Forward: Parameters (FP32 storage) -> Cast to BF16 -> Compute
# Backward: Gradients (BF16) -> Cast to FP32 -> AllReduce
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

**FSDP2 混合精度的优势：**

- 降低激活值与中间计算的显存占用
- 在现代 GPU 上获得更快计算
- 通过选择性精度保持数值稳定性

### 4.2 CPU 卸载

由于 all-gather 前参数并不完整，而进程间通信较慢、数据量又大，可以把分片模型在不用时放在 CPU 内存中，这就是 **CPU 卸载（CPU Offloading）**。CPU 卸载把不用的分片模型状态放在 CPU 内存，以降低 GPU 显存占用。代价是计算时 CPU 与 GPU 之间需要搬运数据。

```python
from torch.distributed.fsdp import MixedPrecisionPolicy, CPUOffloadPolicy, fully_shard

with torch.device("meta"):
    model_config = LlamaConfig()
    model = LlamaForPretraining(model_config)

mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
)
offload_policy = CPUOffloadPolicy(pin_memory=True)

for layer in model.base_model.layers:
    fully_shard(layer, mp_policy=mp_policy, offload_policy=offload_policy)
fully_shard(model.base_model, mp_policy=mp_policy, offload_policy=offload_policy)
fully_shard(model, mp_policy=mp_policy, offload_policy=offload_policy)

model.to_empty(device="cpu")
```

**CPU 卸载的作用：**

- 将分片参数、梯度和优化器状态保存在 CPU
- 前向/反向时将分片参数拷贝至 GPU，使用后释放
- 将梯度拷贝回 CPU，并在 CPU 上执行优化器步骤

**适用场景：**

- GPU 显存受限
- 模型无法放入单卡显存

**不适用场景：**

- CPU 内存有限
- 训练速度优先于显存

CPU 卸载通常会降低训练速度。启用后建议保留已分配的梯度张量，避免反复分配：

```python
optimizer.zero_grad(set_to_none=False)
```

### 4.3 梯度检查点

FSDP 已比普通数据并行更省显存。如需进一步降低显存，可结合 **梯度检查点（Gradient Checkpointing）**：

```python
import functools
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

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

`wrap_policy` 判断模块是否属于指定类别，若是则对其应用梯度检查点。前向时丢弃内部激活值，反向时重新计算，以时间换空间。

> 注：上述 `apply_activation_checkpointing` 与 `checkpoint_wrapper` 来自 `torch.distributed.algorithms._checkpoint.checkpoint_wrapper`，主要面向 FSDP1。在 FSDP2 中，更常见的做法是直接使用 `torch.utils.checkpoint.checkpoint_wrapper` 或 `torch.utils.checkpoint.checkpoint` 包裹子模块。

### 4.4 torch.compile()

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

### 4.5 通信与计算掩盖

为了极致的训练效率，FSDP2 实现了高度优化的 **通信计算掩盖（Overlap）** 机制，即 **Prefetching**。为了降低每步延迟，PyTorch 会在当前模块计算的同时预取下一模块的分片。这种 **预取（prefetching）** 让通信与计算重叠。部分 FSDP 配置会占用更多显存以换取更大重叠，需要在显存与速度之间权衡。

```python
num_to_forward_prefetch = 2
for i, layer in enumerate(model.layers):
    if i >= len(model.layers) - num_to_forward_prefetch:
        break
    layers_to_prefetch = [
        model.layers[i + j] for j in range(1, num_to_forward_prefetch + 1)
    ]
    layer.set_modules_to_forward_prefetch(layers_to_prefetch)

num_to_backward_prefetch = 2
for i, layer in enumerate(model.layers):
    if i < num_to_backward_prefetch:
        continue
    layers_to_prefetch = [
        model.layers[i - j] for j in range(1, num_to_backward_prefetch + 1)
    ]
    layer.set_modules_to_backward_prefetch(layers_to_prefetch)

for _ in range(epochs):
    # 提前触发第 1 个 all-gather
    # 将 all-gather 与 model(x) 前的任何计算重叠
    model.unshard()
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    loss = model(x).sum()
    loss.backward()
    optim.step()
    optim.zero_grad()
```

`modules` 必须按执行顺序排列。预取顺序错误会降低性能。同时，不能指定未被切分的模块（如 `model.lm_head`），否则 FSDP 无法为它发起 all-gather。

`fully_shard` 注册前向 / 反向钩子，在计算前 all-gather 参数，在计算后重新分片参数。为了将 all-gather 与计算重叠，FSDP2 提供开箱即用的**隐式预取**，以及供高级用户手动控制 all-gather 调度的**显式预取**。

**隐式预取**：CPU 线程在层 $i$ 计算前发出层 $i$ 的 all-gather。All-gather 被排入其独立的 CUDA stream，而层 $i$ 的计算在默认 stream 中执行。对于非 CPU 密集型工作负载（如大 batch size 的 Transformer），层 $i+1$ 的 all-gather 可与层 $i$ 的计算重叠。隐式预取在反向传播中的工作方式类似，只是 all-gather 按前向传播顺序的逆序发出。

[![FSDP 隐式预取](https://docs.pytorch.org/tutorials/_images/fsdp_implicit.png)](https://docs.pytorch.org/tutorials/_images/fsdp_implicit.png)

建议用户从隐式预取开始，以了解开箱即用的性能表现。

**显式预取**：用户可通过 [set_modules_to_forward_prefetch](https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html#torch.distributed.fsdp.FSDPModule.set_modules_to_forward_prefetch) 指定前向顺序，通过 [set_modules_to_backward_prefetch](https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html#torch.distributed.fsdp.FSDPModule.set_modules_to_backward_prefetch) 指定反向顺序。如下代码所示，CPU 线程在层 $i$ 处发出层 $i+1$ 和 $i+2$ 的 all-gather。

显式预取在以下场景表现良好：

**CPU 密集型工作负载**：若使用隐式预取，CPU 线程在层 $i$ 的 kernel 执行时太慢，无法为层 $i+1$ 发出 all-gather。必须显式在层 $i$ 前向运行前发出层 $i+1$ 的 all-gather。

**预取 2 层以上**：隐式预取每次仅 all-gather 下一层，以保持内存占用最小。显式预取可一次 all-gather 多层，以可能获得更好的性能，代价是增加内存。详见代码中的 `layers_to_prefetch`。

**提前发出第 1 个 all-gather**：隐式预取在调用 `model(x)` 时发生。第 1 个 all-gather 被暴露。可提前显式调用 [model.unshard()](https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html#torch.distributed.fsdp.FSDPModule.unshard) 以提前发出第 1 个 all-gather。

### 4.6  梯度裁剪与基于 DTensor 的优化器

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

## 5. 分布式 Checkpoint

本节展示如何将完整状态字典转换为 DTensor 状态字典以加载，以及如何将其转回完整状态字典以保存。

- 第 1 次运行：创建模型和优化器的检查点
- 第 2 次运行：从上一个检查点加载以恢复训练

### 5.1 手动保存与加载

FSDP 模型本质仍是 PyTorch 模型，只是权重被替换为 DTensor。

```python
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import Shard

assert isinstance(model, FSDPModule)
assert isinstance(model, LlamaForPretraining)

rank = torch.distributed.get_rank()
for param in model.parameters():
    # DTensor 应该有 placement
    assert param.placements == (Shard(rank),)
    # DTensor 与原张量 dtype 相同
    assert param.dtype == torch.float32
    # 查看当前分片内容
    print(param.get_local_tensor())
```

> 注：上述 `assert param.placements == (Shard(rank),)` 在一般情况下并不严谨。若参数在第 0 维切分，placement 应为 `Shard(0)`，与进程 rank 无关；仅当 world size 恰好等于切分维度时才可能巧合成立。

手动保存时，应只让 rank 0 将完整状态写入文件：

**保存状态字典**：`model.state_dict()` 返回 DTensor 状态字典。可通过调用 [full_tensor()](https://docs.pytorch.org/docs/stable/distributed.tensor.html#torch.distributed.tensor.DTensor.full_tensor) 将 DTensor 转换为普通 `torch.Tensor`。内部会发起跨 rank 的 all-gather 以获取未分片参数。对于 rank 0，`full_param.cpu()` 逐个将张量卸载到 CPU，避免未分片参数峰值占用 GPU 内存。

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

**加载状态字典**：在 meta device 下初始化模型并调用 `fully_shard`，将 `model.parameters()` 从普通 `torch.Tensor` 转换为 DTensor。从 `torch.load` 读取完整状态字典后，可调用 [distribute_tensor](https://docs.pytorch.org/docs/stable/distributed.tensor.html#torch.distributed.tensor.distribute_tensor) 将普通 `torch.Tensor` 转换为 DTensor，使用与 `model.state_dict()` 相同的 placements 和 device mesh。最后可调用 [model.load_state_dict](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.load_state_dict) 将 DTensor 状态字典加载到模型。

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

### 5.2 使用 Stateful 包装器

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

### 5.3 使用分布式 Checkpoint API

更简洁的方式是使用 PyTorch DCP：

**保存状态字典**：[get_model_state_dict](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.get_model_state_dict) 配合 `full_state_dict=True` 和 `cpu_offload=True` 会 all-gather 张量并将其卸载到 CPU。工作方式与 DTensor API 类似。

**加载状态字典**：可使用 [set_model_state_dict](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.set_model_state_dict) 将完整状态字典加载到 FSDP2 模型。设置 `broadcast_from_rank0=True` 可仅在 rank 0 上加载完整状态字典，以避免峰值 CPU 内存占用。DCP 会将张量分片并广播到其他 rank。

```python
from torch.distributed.checkpoint import load, save
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict, StateDictOptions


def save_checkpoint(model, optimizer):
    dist.barrier()
    model_state, optimizer_state = get_state_dict(
        model, optimizer, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
    )
    save(
        {"model": model_state, "optimizer": optimizer_state},
        checkpoint_id="checkpoint-dist",  # 每个 rank 保存各自的文件
    )
    dist.barrier()


def load_checkpoint(model, optimizer):
    dist.barrier()
    model_state, optimizer_state = get_state_dict(
        model, optimizer, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
    )
    load(
        {"model": model_state, "optimizer": optimizer_state},
        checkpoint_id="checkpoint-dist"
    )
    set_state_dict(
        model, optimizer,
        model_state_dict=model_state, optim_state_dict=optimizer_state,
        options=StateDictOptions(broadcast_from_rank0=True, full_state_dict=True, cpu_offload=True)
    )
    dist.barrier()
```

> 注：如果不使用 CPU 卸载，就必须去掉 `cpu_offload` 选项”。实际上 `StateDictOptions` 中的 `cpu_offload` 仅控制保存/加载状态字典时是否先卸载到 CPU，与 FSDP 是否启用 `CPUOffloadPolicy` 相互独立。即使未启用 CPU 卸载，也可以保留该选项以减少显存峰值。

关于使用 [set_optimizer_state_dict](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.set_optimizer_state_dict) 和 [get_optimizer_state_dict](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.get_optimizer_state_dict) 加载和保存优化器状态字典，请参阅 [pytorch/examples](https://github.com/pytorch/examples/blob/main/distributed/FSDP2/checkpoint.py)。

## FSDP1 迁移至 FSDP2 指南

以下对比 FSDP 与等效的 `fully_shard` 用法，并重点说明关键差异及迁移步骤。

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

### 迁移步骤

1. 替换导入语句
2. 直接实现"策略"（对目标子层应用 `fully_shard`）
3. 用 `fully_shard` 包装根模型，替代 `FSDP`
4. 移除 `param_init_fn`，手动调用 `model.reset_parameters()`
5. 替换其他 FSDP1 参数（见下表）

### 参数映射表

| FSDP1 参数                          | FSDP2 等效配置                                               |
| ----------------------------------- | ------------------------------------------------------------ |
| `sharding_strategy`                 |                                                              |
| - `FULL_SHARD`                      | `reshard_after_forward=True`                                 |
| - `SHARD_GRAD_OP`                   | `reshard_after_forward=False`                                |
| - `HYBRID_SHARD`                    | `reshard_after_forward=True` + 2D device mesh                |
| - `_HYBRID_SHARD_ZERO2`             | `reshard_after_forward=False` + 2D device mesh               |
| `cpu_offload`                       |                                                              |
| - `CPUOffload.offload_params=False` | `offload_policy=None`                                        |
| - `CPUOffload.offload_params=True`  | `offload_policy=CPUOffloadPolicy()`                          |
| `backward_prefetch`                 |                                                              |
| - `BACKWARD_PRE`                    | 始终使用                                                     |
| - `BACKWARD_POST`                   | 不支持                                                       |
| `mixed_precision`                   |                                                              |
| - `buffer_dtype`                    | 已省略，因为 `fully_shard` 不分片 buffer                     |
| - `cast_forward_inputs`             | `fully_shard` 的 `cast_forward_inputs` 同时映射到 FSDP1 的 `cast_forward_inputs` 和 `cast_root_forward_inputs` |
| - `output_dtype`                    | `fully_shard` 的新配置                                       |
| `device_id`                         | 从 device_mesh 的 device 推断                                |
| `sync_module_states`                | 移至 DCP。用户可使用 [set_model_state_dict](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.set_model_state_dict) 配合 `broadcast_from_rank0=True` 从 rank 0 广播状态字典 |
| `forward_prefetch`                  | 手动控制预取：                                               |
|                                     | - 手动调用 [fsdp_module.unshard()](https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html#torch.distributed.fsdp.FSDPModule.unshard) |
|                                     | - 使用 [set_modules_to_forward_prefetch](https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html#torch.distributed.fsdp.FSDPModule.set_modules_to_forward_prefetch) 和 [set_modules_to_backward_prefetch](https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html#torch.distributed.fsdp.FSDPModule.set_modules_to_backward_prefetch) 控制自动预取 |
| `limit_all_gathers`                 | 不再需要，因为 `fully_shard` 已移除 CPU 同步                 |
| `use_orig_params`                   | 始终使用原始参数（不再有扁平参数）                           |
| `no_sync()`                         | [set_requires_gradient_sync](https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html#torch.distributed.fsdp.FSDPModule.set_requires_gradient_sync) |
| `ignored_params` / `ignored_states` | [ignored_params](https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html#torch.distributed.fsdp.fully_shard) |

## 7. 总结

本文介绍了如何使用 PyTorch FSDP2 在多 GPU 上训练大模型：

- FSDP 通过分片参数降低单卡显存，代价是更多的 all-gather 与 reduce-scatter 通信。
- 使用 `fully_shard()` 自底向上切分子模块和顶层模块。
- 训练循环几乎与普通模型相同，只需注意 `DistributedSampler` 和显式 `model.unshard()`。
- 可以通过混合精度、CPU 卸载、`reshard_after_forward`、梯度检查点、`torch.compile()` 和预取等手段在显存与速度之间权衡。
- 模型保存既可以用 DTensor 手动合并，也可以使用 PyTorch 分布式 checkpoint API。

理解这些机制后，你就可以根据模型规模和硬件条件，灵活调整 FSDP 配置，高效完成大模型训练。
