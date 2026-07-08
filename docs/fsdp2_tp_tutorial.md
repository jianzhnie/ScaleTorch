# 使用张量并行（Tensor Parallel）训练大规模 Transformer 模型

> **创建时间**：2024-04-19 | **最后更新**：2025-07-18 | **最后验证**：2024-11-05
> **作者**：Wanchao Liang, Tianyu Liu

---

## 概述

本教程演示如何使用张量并行（Tensor Parallel, TP）和完全分片数据并行（Fully Sharded Data Parallel, FSDP）在数百到数千 GPU 上训练大规模 Transformer 模型。

### 前置要求

- PyTorch 2.3.0 或更高版本，安装 CUDA/Linux 支持
- 张量并行 API
- [DeviceMesh 入门指南](https://docs.pytorch.org/tutorials/recipes/distributed_device_mesh.html)
- [FSDP 入门指南](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)

---

## 张量并行的工作原理

张量并行（TP）最初在 [Megatron-LM 论文](https://arxiv.org/pdf/1909.08053.pdf) 中提出，是一种高效训练大规模 Transformer 模型的模型并行技术。本教程中提到的序列并行（Sequence Parallel, SP）是张量并行的一种变体，它在序列维度上对 `nn.LayerNorm` 或 `RMSNorm` 进行分片，以进一步节省训练过程中的激活内存。随着模型规模增大，激活内存成为瓶颈，因此在张量并行训练中通常会对 LayerNorm 或 RMSNorm 层应用序列并行。

<img src="https://docs.pytorch.org/tutorials/_images/megatron_lm.png" alt="Megatron-LM TP" style="zoom: 33%;" />

**图 1.** Transformer 模型 MLP 和自注意力层上的张量并行风格分片。注意力和 MLP 中的矩阵乘法通过分片计算完成（图片来源）。

### 高层视角

PyTorch 张量并行在高层按以下方式工作：

**分片初始化**

1. 确定对每一层应用哪种 `ParallelStyle`。
2. 通过调用 `parallelize_module` 对初始化后的模块进行分片。
3. 并行化后的模块将其模型参数替换为 DTensor，DTensor 负责使用分片计算运行并行化模块。

**运行时前向/反向**

1. 根据用户为每种 `ParallelStyle` 指定的输入/输出 DTensor 布局，运行适当的通信操作以转换输入/输出的 DTensor 布局（如 `all_reduce`、`all_gather` 和 `reduce_scatter`）。
2. 对并行化层运行分片计算以节省计算/内存（例如 `nn.Linear`、`nn.Embedding`）。

---

## 何时以及为何应用张量并行

PyTorch 的完全分片数据并行（FSDP）已经能够将模型训练扩展到特定数量的 GPU。然而，当需要进一步在模型大小和 GPU 数量方面扩展训练时，会出现许多额外挑战，这些挑战可能需要将张量并行与 FSDP 结合使用：

1. **通信延迟主导**：当 `World Size`（GPU 数量）变得非常大（超过 128/256 个 GPU）时，FSDP 的集合通信（如 `all_gather`）被环形延迟主导。通过在 FSDP 之上实现 TP/SP，FSDP 的 `World Size` 可以减少 8 倍（将 FSDP 仅应用于跨主机），从而将延迟成本降低相同的倍数。

2. **数据并行达到极限**：当无法将全局批次大小提高到超过 GPU 数量时（由于收敛性和 GPU 内存限制），张量/序列并行是已知的唯一方法来"调整"全局批次大小并继续使用更多 GPU 进行扩展。这意味着模型大小和 GPU 数量都可以继续扩展。

3. **FLOPS 优化**：对于某些类型的模型，当 `Local Rank` 的批次（Batch Size）大小变小时，TP/SP 可以产生更适合浮点运算（FLOPS）的矩阵乘法形状。 

### 预训练时何时会达到这些极限？

目前，使用数十亿或数万亿 token 预训练大型语言模型（LLM）可能需要数月时间，即使使用数千个 GPU。

- **极限 1**：在大规模训练 LLM 时总会达到。例如，Llama 2 70B 使用 2K GPU 训练了 35 天，在 2K 规模下需要多维并行。
- **极限 2**：当 Transformer 模型变大时（如 Llama 2 70B），也会很快达到。即使本地 `batch_size=1`，也无法单独使用 FSDP，因为内存和收敛限制。例如，Llama 2 的全局批次大小为 1K，因此在 2K GPU 上无法单独使用数据并行。

## 如何应用张量并行

PyTorch 张量并行 API 提供了一组模块级原语（`ParallelStyle`），用于配置模型每一层的分片，包括：

- **`ColwiseParallel` 和 `RowwiseParallel`**：按列或行方式分片 `nn.Linear` 和 `nn.Embedding`。
- **`SequenceParallel`**：对 `nn.LayerNorm`、`nn.Dropout`、`RMSNorm` 等执行分片计算。
- **`PrepareModuleInput` 和 `PrepareModuleOutput`**：配置模块输入/输出的分片布局，并执行适当的通信操作。

### 示例：Llama 2 模型

为了演示如何使用 PyTorch 原生张量并行 API，我们来看一个常见的 Transformer 模型。本教程使用最新的 Llama 2 模型作为参考 Transformer 模型实现，因为它在社区中也被广泛使用。

由于张量并行将单个张量分片到一组设备上，我们需要首先设置分布式环境（如 NCCL 通信器）。张量并行是一种单程序多数据（SPMD）分片算法，类似于 PyTorch DDP/FSDP，它在底层利用 PyTorch DTensor 执行分片。它还利用 DeviceMesh 抽象（底层管理 ProcessGroup）进行设备管理和分片。

要了解如何利用 DeviceMesh 设置多维并行，请参阅[此教程](https://docs.pytorch.org/tutorials/recipes/distributed_device_mesh.html)。张量并行通常在每台主机内工作，因此让我们首先初始化一个连接主机内 8 个 GPU 的 DeviceMesh：

```python
from torch.distributed.device_mesh import init_device_mesh

tp_mesh = init_device_mesh("cuda", (8,))
```

现在我们已经初始化了 DeviceMesh，让我们详细看看 Llama 2 模型架构，了解应该如何执行张量并行分片。

### TransformerBlock 的核心结构

这里我们关注核心的 `TransformerBlock`，Transformer 模型通过堆叠相同的 `TransformerBlock` 来扩展模型规模。核心 `TransformerBlock` 由一个注意力层和一个前馈层组成。我们先来看更简单的前馈层。

#### 前馈层（FeedForward）

前馈层包含三个线性层，执行 SwiGLU 风格的 MLP。查看其前向函数：

```python
# 前馈层中的前向传播
def forward(self, x):
    return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

它并发执行 `w1` 和 `w3` 的矩阵乘法，然后用 `w2` 的矩阵乘法与 `w1`/`w3` 线性投影结果的组合结果相乘。这意味着我们可以使用张量并行论文中的思想，将 `w1`/`w3` 线性层按列方式分片，将 `w2` 线性层按行方式分片，这样在所有三层结束时只需要一次 `all_reduce` 通信。

使用 PyTorch 原生张量并行，我们可以像下面这样为前馈层创建 `parallelize_plan`：

```python
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module

layer_tp_plan = {
    # 默认 ColwiseParallel 输入布局为 Replicate
    # 默认 RowwiseParallel 输出布局为 Replicate
    "feed_forward.w1": ColwiseParallel(),
    "feed_forward.w2": RowwiseParallel(),
    "feed_forward.w3": ColwiseParallel(),
}
```

这就是使用 PyTorch 张量并行 API 配置前馈层分片的方式。注意，用户只需要指定如何分片各个层，通信（例如 `all_reduce`）将在后台自动发生。

#### 注意力层（Attention）

注意力层包含 `wq`、`wk`、`wv` 线性层，将输入投影到 q/k/v，然后执行注意力计算，并通过 `wo` 线性层进行输出投影。这里的张量并行旨在对 q/k/v 投影执行列方向分片，对 `wo` 线性投影执行行方向分片。因此我们可以将注意力层的计划添加到刚刚起草的 `tp_plan` 中：

```python
layer_tp_plan = {
    # 默认 ColwiseParallel 输入布局为 Replicate
    # 默认 RowwiseParallel 输出布局为 Replicate
    "attention.wq": ColwiseParallel(use_local_output=False),
    "attention.wk": ColwiseParallel(use_local_output=False),
    "attention.wv": ColwiseParallel(use_local_output=False),
    "attention.wo": RowwiseParallel(),
    "feed_forward.w1": ColwiseParallel(),
    "feed_forward.w2": RowwiseParallel(),
    "feed_forward.w3": ColwiseParallel(),
}
```

这几乎就是应用张量并行到 `TransformerBlock` 所需的 `layer_tp_plan`。然而，我们需要注意的一点是，当按列方向分片线性层时，线性层的输出将在最后一个张量维度上分片，而行方向分片的线性层直接接受在最后一个维度上分片的输入。

如果在列方向线性和行方向线性之间有任何更多的张量操作（如 view 操作），我们需要调整相关的形状操作以适应分片形状。

对于 Llama 模型，在注意力层中有几个与形状相关的 view 操作。具体来说，对于 `wq`/`wk`/`wv` 线性层的列方向并行，激活张量在 `num_heads` 维度上分片。为了管理全局和局部 `num_heads` 之间的差异，我们应该设置 `use_local_output=False` 以确保输出是 DTensor。与普通张量不同，DTensor 了解并行计划，并将自动处理 `num_heads` 维度的变化。

最后，我们需要调用 `parallelize_module` API 使每个 `TransformerBlock` 的计划生效。在底层，它将注意力层和前馈层内的模型参数分发到 DTensor，并在必要时为模块输入和输出注册通信钩子（在每个模块前后分别注册）：

```python
for layer_id, transformer_block in enumerate(model.layers):
    layer_tp_plan = {...}  # 即我们刚刚生成的计划

    parallelize_module(
        module=transformer_block,
        device_mesh=tp_mesh,
        parallelize_plan=layer_tp_plan,
    )
```

#### 词嵌入和输出层

既然我们已经详细阐述了每个 `TransformerBlock` 的分片计划，通常第一层有 `nn.Embedding`，最后一层有 `nn.Linear` 投影层，用户可以选择对第一层 `nn.Embedding` 进行行方向或列方向分片，对最后一层 `nn.Linear` 投影层进行列方向分片，并指定适当的输入和输出布局。

以下是一个示例：

```python
model = parallelize_module(
    model,
    tp_mesh,
    {
        "tok_embeddings": RowwiseParallel(
            input_layouts=Replicate(),
        ),
        "output": ColwiseParallel(
            output_layouts=Replicate(),
        ),
    }
)
```

> **注意**：如果要分片的模型太大而无法放入 CPU 内存，可以使用 meta device 初始化（例如，首先在 meta device 上初始化模型，分片各层，然后实例化模型），或者在 Transformer 模型初始化期间逐层并行化 `TransformerBlock`。

---

## 对 LayerNorm/RMSNorm 层应用序列并行

序列并行在上述张量并行的基础上工作。与基本张量并行相比，基本张量并行仅在注意力模块和前馈模块内分片张量，并保持其模块输入和输出（即前向传播中的激活和反向传播中的梯度）为复制状态，序列并行将它们保持在序列维度上的分片状态。

在典型的 `TransformerBlock` 中，前向函数结合归一化层（LayerNorm 或 RMSNorm）、注意力层、前馈层和残差连接。例如：

```python
# TransformerBlock 中的前向传播
def forward(self, x):
    h = x + self.attention(self.attention_norm(x))
    out = h + self.feed_forward(self.ffn_norm(h))
    return out
```

在大多数用例中，激活（和梯度）在注意力和前馈模块外部的形状为 `[batch size, sequence length, hidden dimension]`。用 DTensor 的语言来说，序列并行对模块的前向/反向传播使用 `Shard(1)` 布局执行激活计算。

按照前面的代码示例，下面的代码演示如何对 `TransformerBlock` 内的 RMSNorm 层应用序列并行：

首先导入序列并行所需的依赖：

```python
from torch.distributed.tensor.parallel import (
    PrepareModuleInput,
    SequenceParallel,
)
```

接下来调整 `layer_tp_plan` 以在 RMSNorm 层上启用序列并行：

```python
layer_tp_plan = {
    # 现在 SequenceParallel 的输入和输出具有 Shard(1) 布局，
    # 表示输入/输出张量在序列维度上分片
    "attention_norm": SequenceParallel(),
    "attention": PrepareModuleInput(
        input_layouts=(Shard(1), Replicate()),
        desired_input_layouts=(Replicate(), Replicate()),
    ),
    "attention.wq": ColwiseParallel(use_local_output=False),
    "attention.wk": ColwiseParallel(use_local_output=False),
    "attention.wv": ColwiseParallel(use_local_output=False),
    "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
    "ffn_norm": SequenceParallel(),
    "feed_forward": PrepareModuleInput(
        input_layouts=(Shard(1),),
        desired_input_layouts=(Replicate(),),
    ),
    "feed_forward.w1": ColwiseParallel(),
    "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
    "feed_forward.w3": ColwiseParallel(),
}
```

可以看到，我们现在使用 `PrepareModuleInput` 将注意力层和前馈层的模块输入布局从 `Shard(1)` 修改为 `Replicate()`，并将其输出布局标记为 `Shard(1)`。

与张量并行一样，用户只需要指定输入和输出的张量分片布局，层之间的通信将自动发生。

注意，使用序列并行时，我们假设 `TransformerBlock` 的输入和输出始终在序列维度上分片，以便多个 `TransformerBlock` 可以无缝连接。

这可以通过显式指定开头 `nn.Embedding` 层的输出和最终 `nn.Linear` 投影层的输入为 `Shard(1)` 来实现：

```python
model = parallelize_module(
    model,
    tp_mesh,
    {
        "tok_embeddings": RowwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Shard(1),
        ),
        "norm": SequenceParallel(),
        "output": ColwiseParallel(
            input_layouts=Shard(1),
            output_layouts=Replicate()
        ),
    }
)
```

---

## 应用 Loss Parallel

Loss Parallel 是一种相关技术，用于在计算损失函数时节省内存和通信，因为模型输出通常非常大。在 Loss Parallel 中，当模型输出在（通常巨大的）词汇维度上分片时，可以高效地计算交叉熵损失，而无需将所有模型输出收集到每个 GPU 上。这不仅显著降低了内存消耗，还通过减少通信开销和并行执行分片计算来提高训练速度。

下图简要说明了 Loss Parallel 如何通过分片计算避免将所有模型输出收集到每个 GPU 上。

<img src="https://docs.pytorch.org/tutorials/_images/loss_parallel.png" alt="loss parallel" style="zoom: 25%;" />

**图 2.** 在一个 GPU 上使用 Loss Parallel 进行交叉熵损失前向计算。蓝色表示分片张量；绿色表示复制张量；黄色表示具有部分值的张量（待 all_reduce）。黑色箭头是本地计算；红色箭头是 GPU 间的函数式集合通信。

在 PyTorch 张量并行 API 中，Loss Parallel 可以通过上下文管理器 `loss_parallel` 启用，使用它可以直接使用 `torch.nn.functional.cross_entropy` 或 `torch.nn.CrossEntropyLoss`，无需修改其他代码。

要应用 Loss Parallel，模型预测（通常形状为 `[batch size, sequence length, vocabulary size]`）应该在词汇维度上分片。这可以通过标记最后一层线性投影层输出的输出布局来轻松实现：

```python
model = parallelize_module(
    model,
    tp_mesh,
    {
        "tok_embeddings": RowwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Shard(1),
        ),
        "norm": SequenceParallel(),
        "output": ColwiseParallel(
            input_layouts=Shard(1),
            # 使用 DTensor 作为输出
            use_local_output=False,
        ),
    },
)
```

在上面的代码中，我们还将序列并行应用于输出之前的归一化层。我们应用 `use_local_output=False` 让输出保持为 DTensor，以与 `loss_parallel` 上下文管理器配合工作。之后，可以简单地按如下方式调用 `cross_entropy` 损失函数。注意反向计算也需要在上下文内发生。

```python
import torch.nn.functional as F
from torch.distributed.tensor.parallel import loss_parallel

pred = model(input_ids)
with loss_parallel():
    # 假设 pred 和 labels 的形状为 [batch, seq, vocab]
    loss = F.cross_entropy(pred.flatten(0, 1), labels.flatten(0, 1))
    loss.backward()
```

---

## 将张量并行与完全分片数据并行结合

既然我们已经展示了如何对模型应用张量/序列并行，让我们也来看看张量并行和完全分片数据并行如何协同工作。

由于张量并行会产生阻塞计算的通信，我们希望确保它在快速通信通道内运行，例如 NVLink。

在实践中，我们通常**在每台主机内应用张量并行，跨主机应用完全分片数据并行**。

<img src="https://docs.pytorch.org/tutorials/_images/fsdp_tp.png" alt="fsdp + tp" style="zoom: 33%;" />

**图 3.** FSDP 和 TP 在不同的设备维度上工作，FSDP 通信发生在主机间，TP 通信发生在主机内。

这种 2-D 并行模式可以通过 2-D DeviceMesh 轻松表达，我们只需要将每个"子"DeviceMesh 传递给各个并行 API：

```python
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module
from torch.distributed.fsdp import fully_shard

# 即 2-D mesh 为 [dp, tp]，在 64 个 GPU 上训练，执行 8 路 DP 和 8 路 TP
mesh_2d = init_device_mesh("cuda", (8, 8))
tp_mesh = mesh_2d["tp"]  # 连接主机内设备的子 mesh
dp_mesh = mesh_2d["dp"]  # 连接跨主机设备的子 mesh

model = Model(...)

tp_plan = {...}

# 在 tp_mesh 上应用主机内张量并行
model_tp = parallelize_module(model, tp_mesh, tp_plan)
# 在 dp_mesh 上应用跨主机 FSDP
model_2d = fully_shard(model_tp, mesh=dp_mesh, ...)
```

这使我们能够轻松地在每台主机内（主机内）应用张量并行，跨主机（主机间）应用 FSDP，**对 Llama 模型代码零修改**。

张量（模型）并行和数据并行技术结合在一起，提供了继续使用大量 GPU 增加模型大小并高效训练的能力。

---

## 结论

本教程演示了如何使用张量并行结合完全分片数据并行，在数百到数千 GPU 上训练大规模 Transformer 模型。它解释了如何对模型的不同部分应用张量并行，而无需对模型本身进行任何代码修改。张量并行是一种高效的大规模训练模型并行技术。

要查看本教程中解释的完整端到端代码示例，请参阅 pytorch/examples 仓库中的[张量并行示例](https://github.com/pytorch/examples/blob/main/distributed/tensor_parallelism/tensor_parallel_example.py)。
