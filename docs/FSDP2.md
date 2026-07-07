# FSDP2

## FSDP 的基本概念

**分片（Sharding）** 一词最早来自数据库系统，指把数据库拆成多个较小的单元（shard）以提升性能。在机器学习中，分片指把模型参数分散到多个设备上。

与流水线并行不同，FSDP 的分片不对应完整的网络层，而是把单个运算拆散。例如，`nn.Linear` 本质上是一次矩阵乘法，它的分片版本只保存权重矩阵的一部分。当需要计算时，FSDP 会临时把各分片聚合起来，恢复成完整矩阵，完成运算后再释放，以回收显存。

使用 FSDP 时，所有模型参数都会被分片，每个进程只保存其中一份。数据并行（Data Parallelism）会在每张 GPU 上保存一份完整模型，只同步数据和梯度；FSDP 则不保存完整模型，每一步都需要同步模型参数。

因此，FSDP 以更高的通信开销换取更低的显存占用。``

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

> 前向/反向时按模块 all-gather 参数，反向结束时通过 reduce-scatter 聚合梯度；每个 rank 仍处理不同的微批次，数据本身并不在 rank 间同步。



### FSDP 的工作流程

FSDP 的典型执行流程如下：

1. 多个进程同时运行，可能分布在多台机器上。每个进程（也就是每张 GPU）只保存模型的一个分片。
2. 分片后，每层权重以 **DTensor**（分布式张量）形式存在，而不是普通 Tensor。因此没有任何一个进程能独立完成某个模块的计算。
3. 每次运算前，FSDP 发起一次 all-gather，让各进程交换该模块的分片，临时恢复出完整模块。
4. 每个进程用自己的微批次（micro-batch）在这个临时模块上执行前向传播，然后丢弃临时模块，继续处理下一层。
5. 反向传播时，FSDP 同样需要对每层 all-gather 解除分片，再计算梯度。
6. 由于每个进程处理的微批次不同，各自算出的梯度也不同。FSDP 通过 reduce-scatter 交换并平均梯度，得到全局梯度，再更新各自的分片。

由此可见，FSDP 的通信和流程比普通数据并行更复杂。但正因为模型被分散到多张 GPU，训练超大模型所需的单卡显存显著下降，这正是使用 FSDP 的动机。

## FSDP2 模型分片

PyTorch FSDP2 的 `fully_shard` 支持多种粒度的分片。在最细粒度下，可以对每一层进行分片以最小化峰值内存占用。

除分片粒度外，FSDP2 还提供多种配置选项以优化性能并缓解 OOM 错误：

### 设备网格配置

`init_device_mesh` 配置描述训练运行设备拓扑的 `DeviceMesh`。本示例使用简单的 1D 网格实现数据并行，但 `DeviceMesh` 也支持多维并行方式，包括张量并行和流水线并行。在多数情况下，组合多种并行方式可以进一步提升训练性能。

关于高级多维并行配置的更多信息，请参阅 [PyTorch 设备网格文档](https://docs.pytorch.org/tutorials/recipes/distributed_device_mesh.html)。

### 为 FSDP 训练准备模型

当模型大到单卡放不下时，一种常见做法是：先在 meta 设备上实例化模型，再切分并分发到各 GPU。

#### 初始化进程组

在 PyTorch 中，需要通过 `torchrun` 启动多进程脚本。每个进程会看到 world size、rank 和 local rank。脚本中需要初始化进程组：

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

下面基于前文介绍的 Llama 模型架构，演示如何切分：

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

在 PyTorch 中，`fully_shard()` 用于创建分片模型。它会原地把普通 Tensor 参数替换为 DTensor，并修改模块，使其在计算前自动执行 all-gather。

注意上面不仅对 `model` 调用了 `fully_shard()`，也对 `model.base_model` 以及每个 Transformer 层都做了调用。这需要仔细理解。

通常你不应只对顶层模型做切分，而应对子模块也分别切分。切分顺序必须**自底向上**，顶层最后切分。每个被切分的模块将成为一次 all-gather 的单元。

[编辑注：原文“不要切分顶层模型”的表述易引起误解。实际仍需对顶层模块调用 `fully_shard()` 以包装根节点；关键在于子模块必须先切分，避免顶层 all-gather 展开整个模型。]

以上述 Llama 结构为例：输入先进入 `base_model` 的嵌入层，然后依次经过各 Transformer block。由于每个 block 已单独切分，FSDP 会分别为它们 all-gather；`base_model` 中的输入嵌入层和最后的 RMSNorm 层则随 `base_model` 的顶层包装一起展开；顶层 `model` 主要包含预测头 `lm_head`。这样，每张 GPU 上只需驻留一个完整的 Transformer block，以及嵌入层、RMSNorm 层和预测头。

你可以进一步调整设计，例如把 `embed_tokens` 也单独切分，或者把每个 block 拆成注意力与前馈两个子模块再切分，以进一步降低显存需求。

#### 在 meta 设备上初始化

模型切分完成后，使用 `model.to_empty(device=device)` 把它从 meta 设备转移到实际 GPU，并调用 `model.reset_parameters()` 重新初始化权重。如果要从 checkpoint 加载，则跳过随机初始化。

为了让 `reset_parameters()` 生效，每个自定义模块都需要实现该方法。

模型切分成功的标志是 `isinstance(model, FSDPModule)`。之后可以像普通模型一样创建优化器。PyTorch 优化器对 DTensor 和 Tensor 的更新方式相同。

###  FSDP 训练循环

使用 FSDP 后，训练循环几乎无需改动：

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
        # 在前向之前主动解除分片
        model.unshard()

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

        pbar.set_postfix(loss=loss.item())
        pbar.update(1)
    pbar.close()
```

唯一明显的变化是调用 `model.unshard()` 在前向之前触发 all-gather。但这是可选的：即使不调用，`model(input_ids, attn_mask)` 内部也会自动触发 all-gather。显式调用可以在准备输入张量的同时提前开始通信。

FSDP 兼具数据并行的特点。与分布式数据并行一样，你需要为 DataLoader 配置 `DistributedSampler`，让每个 rank 处理不同的微批次。因为每个进程通过 all-gather 得到完整模块，所以可以用各自的微批次独立计算；反向时再把梯度聚合。

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

这与分布式数据并行中的 DataLoader 设置完全一致。

## 运行训练任务

运行完整脚本需要使用 `torchrun`：

```bash
torchrun --standalone --nproc_per_node=4 fsdp_training.py
```

其中 `nproc_per_node=4` 表示在当前节点的 4 张 GPU 上启动 4 个进程。跨节点训练时，需要配合相应的启动参数。

## 混合精度

启用混合精度可以加速训练并降低 GPU 内存使用，同时对精度影响极小。混合精度训练可以通过 `MixedPrecisionPolicy` 在切分时启用：

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

`mp_policy` 定义了混合精度的具体行为。上例中参数使用 bfloat16，而梯度 reduce-scatter 使用 float32。你也可以通过 `output_dtype` 和 `cast_forward_inputs` 控制前向输入/输出的数据类型。由于 `fully_shard()` 是对每个模块单独调用，不同模块甚至可以使用不同的混合精度策略。

当然，也可以全局设置默认数据类型：

```python
torch.set_default_dtype(torch.bfloat16)
```

这会改变后续所有 DTensor 的默认类型。

**FSDP2 混合精度的优势：**

- 降低激活值和中间计算的内存占用
- 在现代 GPU 上实现更快的计算
- 通过选择性精度保持数值稳定性



## CPU 卸载

由于 all-gather 前参数并不完整，而进程间通信较慢、数据量又大，可以把分片模型在不用时放在 CPU 内存中，这就是 **CPU 卸载（CPU Offloading）**。CPU 卸载通过将模型组件存储在 CPU 上来降低 GPU 内存占用。但这会带来计算期间 CPU 与 GPU 之间数据传输开销增加的权衡。

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

- 将分片参数、梯度和优化器状态存储在 CPU 上
- 在前向/反向计算期间将分片参数拷贝至 GPU，使用后释放
- 将计算得到的梯度拷贝至 CPU，由 PyTorch 在 CPU 上执行优化器步骤

**适用场景：**

- GPU 内存受限时
- 模型过大无法放入 GPU 内存时

**不适用场景：**

- CPU 内存有限时（可能导致 CPU 内存溢出崩溃）
- 训练速度优先于内存使用时



CPU 卸载通常会显著降低训练速度。启用后，建议把优化器的梯度清零方式改为保留已分配的梯度张量：

```python
optimizer.zero_grad(set_to_none=False)  # 保留已分配的梯度张量
```

因为 CPU 内存相对充裕，保留梯度张量可以避免反复分配带来的开销。



## `reshard_after_forward` 标志

`fully_shard()` 的第三个关键参数是 `reshard_after_forward`。

- `reshard_after_forward` 默认设置为 `None` ，表示模型在前向传播结束后，根模块会保留解除分片后的参数，反向时无需再次 all-gather；非根模块则默认丢弃。
- `reshard_after_forward=True` 时，前向传播后立即释放 all-gather 的模型权重。

> **补充说明**：`reshard_after_forward=True` 可以降低峰值显存，但反向传播需要重新 all-gather，会增加worker 节点的通信量。是否开启取决于模型大小与通信带宽的权衡。如果未分片模型参数能够完全放入每个工作节点且不构成内存瓶颈，则无需启用 `reshard_after_forward`。

理解该参数有助于模型结构设计。例如在上面的 `LlamaForPretraining` 中，根模块只包含预测头；如果把嵌入层也移到根模块，前向结束后嵌入层会一直驻留显存，可能造成显存浪费。



## 梯度检查点

FSDP 已经比单纯的 DDP (分布式数据并行)更省显存。如果仍需进一步降低显存，可以结合 **梯度检查点（Gradient Checkpointing）**。与普通模型不同，FSDP 中不是用 `torch.utils.checkpoint.checkpoint()` 直接包裹模块，而是设置策略并应用到分片模型：

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

`wrap_policy` 判断模块是否属于指定类别，若是则对其应用梯度检查点。前向时模块内部的激活值会被丢弃，反向时再重新计算，从而用时间换空间。

## All-Gather 预取

### 通信与计算重叠

为了提升效率，PyTorch 会在当前模块计算的同时，预取下一个模块的分片 (发起对下一个模块的 all-gather)，使通信与计算重叠。这种 **预取（prefetching）** 可以降低每步的延迟。部分 FSDP 配置会增加显存占用以换取更大的重叠，从而提高吞吐，需要根据硬件和模型大小权衡。

你可以显式控制预取行为：

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

默认情况下，FSDP 只预取下一个模块。上面的代码改为预取前后各两个模块。`modules` 列表必须按执行顺序排列。如果预取顺序错误，性能反而会下降。同时，不能指定未被切分的模块（例如 `model.lm_head`），否则 FSDP 无法为它发起 all-gather。



## 分布式检查点

本节设置分布式检查点，包括从检查点加载分布式模型、保存分布式模型检查点，以及保存用于推理的模型。

### 分布式检查点包装器

本节使用 PyTorch 的 `Stateful` API 创建检查点包装器，以简化分布式检查点管理。根据 PyTorch 文档，该基础包装器处理跨多个工作节点保存和加载 FSDP2 模型状态的复杂性。

```python
# PyTorch 分布式检查点（DCP）导入
from torch.distributed.checkpoint.state_dict import (
    get_state_dict,
    set_state_dict,
    get_model_state_dict,
    StateDictOptions
)
from torch.distributed.checkpoint.stateful import Stateful
```

```python
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
        # 此行自动管理 FSDP2 的 FQN（完全限定名），并将默认状态字典类型设置为 FSDP.SHARDED_STATE_DICT
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



## FSDP 模型的保存与加载

FSDP 模型本质仍是 PyTorch 模型，只是权重被替换为 DTensor。你可以像操作 Tensor 一样查看每个分片：

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

利用这一特性，可以实现手动的保存与加载。保存时必须只让单个进程写文件，避免相互覆盖：

```python
# 仅在 rank 0 保存
if torch.distributed.get_rank() == 0:
    sharded_state_dict = model.state_dict()     # 值为 DTensor
    full_state_dict = {}                        # 值为普通 CPU Tensor
    for param_name, sharded_param in sharded_state_dict.items():
        full_param = sharded_param.full_tensor()
        full_state_dict[param_name] = full_param.cpu()
    torch.save(full_state_dict, "model.pth")

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

更简单的方式是使用 **分布式 Checkpoint API**

### 从检查点加载分布式模型

使用 `dcp.load` 加载分布式检查点，当训练运行之间工作节点数量变化时，它会自动处理重新分片。这种灵活性允许使用不同的资源配置恢复训练。

```python
from torch.distributed.checkpoint import load, save
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict, StateDictOptions

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

> 注：原文称“如果不使用 CPU 卸载，就必须去掉 `cpu_offload` 选项”。实际上 `StateDictOptions` 中的 `cpu_offload` 仅控制保存/加载状态字典时是否先卸载到 CPU，与 FSDP 是否启用 `CPUOffloadPolicy` 相互独立。即使未启用 CPU 卸载，也可以保留该选项以减少显存峰值。

### 保存模型检查点

```python
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
```

这两个函数应由所有进程同时调用。每个 rank 会把各自的分片保存到 `checkpoint_id` 目录下的不同文件中，不要直接用 `torch.load()` 读取这些文件，因为格式不同。

训练结束后，也可以从分片 checkpoint 恢复出一个未分片的普通模型：

```python
model = LlamaForPretraining(model_config)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
load_checkpoint(model, optimizer)
torch.save(model.state_dict(), "model.pth")
```



### 保存模型用于推理

训练完成后，通常需要将分片检查点合并为单个文件，以便于共享或推理。与常规分布式检查点不同，该过程生成兼容 `torch.load` 的大型产物。为此，`get_model_state_dict` 函数将参数分片 all-gather 到 rank 0，重建完整状态字典，然后将合并后的检查点保存到集群存储。

> **注意**：该方法的关键限制是整个模型必须在 rank 0 上实例化。对于大模型，这可能超出可用 CPU 内存并导致内存溢出。在这种情况下，建议保持模型的分片格式，并依赖分布式模型加载进行推理。

```python
def save_model_for_inference(model: FSDPModule, world_rank: int) -> None:
    """保存完整的未分片模型用于推理。
    
    该函数将分布式模型权重合并为单个检查点文件，
    可用于无需 FSDP 的推理。
    
    Args:
        model: 待保存的 FSDP2 包装模型
        world_rank: 当前工作节点的 rank
    """
    logger.info("正在准备推理模型...")
    
    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
        save_file = os.path.join(temp_checkpoint_dir, "full-model.pt")

        # 步骤 1：跨所有 rank all-gather 模型状态
        # 从分布式分片重建完整模型
        model_state_dict = get_model_state_dict(
            model=model,
            options=StateDictOptions(
                full_state_dict=True,    # 重建完整模型
                cpu_offload=True,        # 移至 CPU 以节省 GPU 内存
            )
        )

        logger.info("成功获取完整模型状态字典")
        checkpoint = None

        # 步骤 2：保存完整模型（仅 rank 0）
        if world_rank == 0: 
            torch.save(model_state_dict, save_file)
            logger.info(f"已保存完整模型至 {save_file}")

            # 为共享存储创建检查点
            checkpoint = ray.train.Checkpoint.from_directory(temp_checkpoint_dir)

        # 步骤 3：向 Ray Train 报告最终检查点
        ray.train.report(
            {}, 
            checkpoint=checkpoint, 
            checkpoint_dir_name="full_model"
        )
```

## 使用 torch.compile()

如果模型支持编译，FSDP 模型也可以被 `torch.compile()` 编译。但必须在**切分之后**再编译，这样编译后的计算图引用的是 DTensor，而不是普通 Tensor。

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

## 总结

本文介绍了如何使用 PyTorch FSDP 在多 GPU 上训练大模型：

- FSDP 通过分片参数降低单卡显存，代价是更多的 all-gather 与 reduce-scatter 通信。
- 使用 `fully_shard()` 自底向上切分子模块和顶层模块。
- 训练循环几乎与普通模型相同，只需注意 `DistributedSampler` 和显式 `model.unshard()`。
- 可以通过混合精度、CPU 卸载、`reshard_after_forward`、梯度检查点和预取等手段在显存与速度之间权衡。
- 模型保存既可以用 DTensor 手动合并，也可以使用 PyTorch 分布式 checkpoint API。

理解这些机制后，你就可以根据模型规模和硬件条件，灵活调整 FSDP 配置，高效完成大模型训练。