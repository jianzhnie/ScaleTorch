# FSDP2 实战指南：在 PyTorch 中使用全分片数据并行训练大模型

本文介绍 PyTorch FSDP2 的核心概念与实战用法，帮助你在多 GPU 环境下训练大模型。

## 1. FSDP 的基本概念

**分片（Sharding）** 最早来自数据库领域，指把数据拆成多个较小的单元（shard）以提升性能。在机器学习中，分片指把模型参数分散到多个设备上。

与流水线并行（Pipeline Parallelism）不同，FSDP 的分片不对应完整的网络层，而是把单个运算拆散。例如，`nn.Linear` 本质是一次矩阵乘法；分片后，每个 rank 只保存权重矩阵的一部分。计算时，FSDP 临时把各分片聚合为完整矩阵，运算完成后再释放，以回收显存。

普通数据并行（Data Parallelism，DP）会在每张 GPU 上保存完整模型副本，只同步数据和梯度。FSDP 不保存完整副本，每一步都需要按模块同步参数。

因此，FSDP 以更高的通信开销换取更低的显存占用。

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

> **补充说明**：前向/反向时按模块 all-gather 参数，反向结束时通过 reduce-scatter 聚合梯度；每个 rank 仍处理不同微批次，数据本身不在 rank 间同步。

### 1.1 工作流程

FSDP 的典型执行流程如下：

1. 多个进程同时运行，可能分布在多台机器上。每个进程（也就是每张 GPU）只保存模型的一个分片。
2. 分片后，每层权重以 **DTensor**（分布式张量）形式存在，而不是普通 Tensor。因此没有任何一个进程能独立完成某个模块的计算。
3. 每次运算前，FSDP 发起一次 all-gather，让各进程交换该模块的分片，临时恢复出完整模块。
4. 每个进程用自己的微批次（micro-batch）在这个临时模块上执行前向传播，然后丢弃临时模块，继续处理下一层。
5. 反向传播时，FSDP 同样需要对每层 all-gather 解除分片，再计算梯度。
6. 由于每个进程处理的微批次不同，各自算出的梯度也不同。FSDP 通过 reduce-scatter 交换并平均梯度，得到全局梯度，再更新各自的分片。

由此可见，FSDP 的通信和流程比普通数据并行更复杂。但正因为模型被分散到多张 GPU，训练超大模型所需的单卡显存显著下降，这正是使用 FSDP 的动机。

### 1.2 通信与计算重叠

为了降低每步延迟，PyTorch 会在当前模块计算的同时预取下一模块的分片。这种 **预取（prefetching）** 让通信与计算重叠。部分 FSDP 配置会占用更多显存以换取更大重叠，需要在显存与速度之间权衡。

## 2. FSDP2 模型分片

PyTorch FSDP2 的 `fully_shard` 支持多种切分粒度。最细粒度可按层切分，最大限度降低峰值显存；粒度越粗，通信开销通常越低。需要根据实际场景选择。

除切分粒度外，FSDP2 还提供多种配置优化性能并缓解 OOM。

### 2.1 DeviceMesh 配置

`init_device_mesh` 用于定义训练设备的拓扑结构。本示例使用一维 mesh 实现数据并行；`DeviceMesh` 也支持多维并行，如张量并行、流水线并行。组合多种并行方式通常能进一步提升训练性能。

更多信息可参考 [PyTorch 设备网格文档](https://docs.pytorch.org/tutorials/recipes/distributed_device_mesh.html)。

### 2.2 准备模型

当模型大到单卡放不下时，可先在 meta 设备上实例化模型，再切分并分发到各 GPU。

#### 初始化进程组

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

#### 创建并切分模型

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

model.to_empty(device=device)
model.reset_parameters()

assert isinstance(model, FSDPModule), f"Expected FSDPModule, got {type(model)}"
```

`fully_shard()` 会原地把普通 Tensor 参数替换为 DTensor，并修改模块使其在计算前自动 all-gather。

注意这里对 `model`、`model.base_model` 以及每个 Transformer 层都调用了 `fully_shard()`。切分顺序必须**自底向上**，顶层最后切分。每个被切分的模块都是一次 all-gather 的单元。

> **补充说明**：顶层调用 `fully_shard()` 是为了包装根节点；前提是子模块已先切分，避免顶层 all-gather 展开整个模型。

以 Llama 结构为例：输入先进入 `base_model` 的嵌入层，再依次经过各 Transformer block。由于每个 block 已单独切分，FSDP 会分别为它们 all-gather；`base_model` 中的嵌入层和 RMSNorm 随 `base_model` 一起展开；顶层 `model` 主要包含预测头。每张 GPU 上只需驻留一个完整 block，以及嵌入层、RMSNorm 和预测头。

你可以进一步把 `embed_tokens` 单独切分，或把每个 block 拆成注意力与前馈两个子模块再切分，以进一步降低显存。

#### 在 meta 设备上初始化

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

启用混合精度可加速训练并降低显存，同时对精度影响极小。通过 `MixedPrecisionPolicy` 在切分时启用：

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

`param_dtype` 控制参数精度，`reduce_dtype` 控制梯度归约精度。也可通过 `output_dtype` 和 `cast_forward_inputs` 控制前向输入输出类型。由于 `fully_shard()` 按模块调用，不同模块可使用不同策略。

也可全局设置默认数据类型：

```python
torch.set_default_dtype(torch.bfloat16)
```

这会改变后续所有 DTensor 的默认类型。

**FSDP2 混合精度的优势：**

- 降低激活值与中间计算的显存占用
- 在现代 GPU 上获得更快计算
- 通过选择性精度保持数值稳定性

### 4.2 CPU 卸载

CPU 卸载把不用的分片模型状态放在 CPU 内存，以降低 GPU 显存占用。代价是计算时 CPU 与 GPU 之间需要搬运数据。

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

### 4.3 `reshard_after_forward`

`fully_shard()` 的 `reshard_after_forward` 参数控制前向结束后是否释放 all-gather 得到的完整参数。

> **补充说明**：在 PyTorch FSDP2 中，`reshard_after_forward` 默认为 `True`，即前向结束后释放参数。设为 `False` 可让对应模块在前后向之间保持未分片，节省一次 all-gather，但会增加峰值显存。

理解该参数有助于模型结构设计。例如，在 `LlamaForPretraining` 中，根模块只包含预测头；若把嵌入层也移到根模块，前向结束后嵌入层会一直驻留显存，可能造成浪费。

### 4.4 梯度检查点

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

### 4.5 torch.compile()

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

### 4.6 All-Gather 预取

FSDP 通过预取下一个模块的分片来重叠通信与计算，降低每步延迟。你可以显式控制预取行为：

```python
num_prefetch = 2
modules = list(model.base_model.layers)

for i, module in enumerate(modules):
    if i == len(modules) - 1:
        break
    module.set_modules_to_forward_prefetch(modules[i+1:i+num_prefetch+1])

for i, module in enumerate(modules):
    if i == 0:
        continue
    module.set_modules_to_backward_prefetch(modules[max(0, i-num_prefetch):i])
```

`modules` 必须按执行顺序排列。预取顺序错误会降低性能。同时，不能指定未被切分的模块（如 `model.lm_head`），否则 FSDP 无法为它发起 all-gather。

## 5. 分布式 Checkpoint

本节介绍 FSDP 模型的保存与加载。

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

加载时，从完整状态重建 DTensor 并赋值给模型：

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

这两个函数需由所有进程同时调用。每个 rank 会保存各自的分片到 `checkpoint_id` 目录下，不要直接用 `torch.load()` 读取这些文件。

训练结束后，也可以从分片 checkpoint 恢复出一个未分片的普通模型：

```python
model = LlamaForPretraining(model_config)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
load_checkpoint(model, optimizer)
torch.save(model.state_dict(), "model.pth")
```

## 6. 运行训练任务

使用 `torchrun` 启动：

```bash
torchrun --standalone --nproc_per_node=4 fsdp_training.py
```

`nproc_per_node=4` 表示在当前节点 4 张 GPU 上启动 4 个进程。跨节点训练需配合相应启动参数。

## 7. 总结

本文介绍了如何使用 PyTorch FSDP2 在多 GPU 上训练大模型：

- FSDP 通过分片参数降低单卡显存，代价是更多的 all-gather 与 reduce-scatter 通信。
- 使用 `fully_shard()` 自底向上切分子模块和顶层模块。
- 训练循环几乎与普通模型相同，只需注意 `DistributedSampler` 和显式 `model.unshard()`。
- 可以通过混合精度、CPU 卸载、`reshard_after_forward`、梯度检查点、`torch.compile()` 和预取等手段在显存与速度之间权衡。
- 模型保存既可以用 DTensor 手动合并，也可以使用 PyTorch 分布式 checkpoint API。

理解这些机制后，你就可以根据模型规模和硬件条件，灵活调整 FSDP 配置，高效完成大模型训练。
