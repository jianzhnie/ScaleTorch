# PyTorch 全分片数据并行（FSDP）

## 动机

最近的研究表明，大规模模型训练将有助于提高模型质量。在过去三年中，模型规模从拥有 1.1 亿参数的 [BERT](https://arxiv.org/abs/1810.04805) 增长到了拥有一万亿参数的 [Megatron-2](https://arxiv.org/abs/2104.04473)，增长了 10,000 倍。随着机器学习 (ML) 模型的规模、大小和参数量的不断增加，ML 从业者发现在自己的硬件上训练甚至加载如此大的模型变得越来越难 \[1\]。

除了需要大量的计算和工程资源外，大多数像这样的扩展方法还会引入额外的通信成本，并要求工程师仔细评估内存使用和计算效率之间的权衡。例如，典型的数据并行训练需要在每个 GPU 上维护模型的冗余副本，而模型并行训练会引入额外的通信成本，以在不同 worker（GPU）之间移动激活值。

分布式训练是训练这些机器学习大模型的关键。 **大规模分布式训练** 领域最近取得了不少重大进展，我们将其中一些最突出的进展总结如下:

1. 使用 ZeRO 数据并行 - 零冗余优化器 \[2\]
   - 阶段 1: 跨数据并行进程 / GPU 对`优化器状态` 进行分片
   - 阶段 2: 跨数据并行进程/ GPU 对`优化器状态 + 梯度` 进行分片
   - 阶段 3: 跨数据并行进程 / GPU 对`优化器状态 + 梯度 + 模型参数` 进行分片
2. CPU offload: 进一步将 ZeRO 阶段 2 的`优化器状态 + 梯度` offload到 CPU 上 \[3\]
3. 张量并行 \[4\]: 模型并行的一种形式，通过对各层参数进行精巧的跨加速器 / GPU 分片，在实现并行计算的同时避免了昂贵的通信同步开销。
4. 流水线并行 \[5\]: 模型并行的另一种形式，其将模型的不同层放在不同的加速器 / GPU 上，并利用流水线来保持所有加速器同时运行。举个例子，在第 2 个加速器 / GPU 对第 1 个 micro batch 进行计算的同时，第 1 个加速器 / GPU 对第 2 个 micro batch 进行计算。
5. 3D 并行 \[3\]: 采用 `ZeRO 数据并行 + 张量并行 + 流水线并行` 的方式来训练数百亿参数的大模型。例如，BigScience 176B 语言模型就采用了该并行方式 \[6\]。

## FSDP 概念

本文我们主要关注 ZeRO 数据并行，更具体地讲是 PyTorch 最新的完全分片数据并行 (Fully Sharded Data Parallel，FSDP) 功能。 **[DeepSpeed](https://github.com/microsoft/deepspeed)** 和 **[FairScale](https://github.com/facebookresearch/fairscale/)** 实现了 ZeRO 论文的核心思想。DeepSpeed ZeRO 和 FairScale 的全分片数据并行（Fully Sharded Data Parallel, FSDP），通过将模型的参数、梯度和优化器状态分片到数据并行worker上，打破了这一限制，同时仍然保持了数据并行的简单性。PyTorch 已正式将 Fairscale FSDP 整合进其 Distributed 模块中，并增加了更多的优化。

FSDP 将 AI 模型的参数分片到数据并行 worker上，并可以选择将部分训练计算offload到 CPU 上。顾名思义，FSDP 是一种数据并行训练算法。尽管参数被分片到不同的 GPU上，但每个微批次数据的计算仍然在每个 GPU worker 本地进行。这种概念上的简单性使得 FSDP 更容易理解和适用于更广泛的使用场景（与层内并行和流水线并行相比）。

- 与优化器状态+梯度分片数据并行方法相比，FSDP 将模型参数、梯度和优化器状态更均匀地分片到 GPU 上来提高内存效率
- 并通过在训练期间分解通信并将其与前向和后向传播重叠来提高计算效率，实现更好的性能。

FSDP 产生的结果与标准分布式数据并行（DDP）训练相同，并且提供了一个易于使用的接口，作为 PyTorch 的 DistributedDataParallel 模块的即插即用替换。通过 FSDP，现在可以更高效地使用更少的 GPU 训练规模大得多的模型。

在 AWS 上的 PyTorch FSDP 扩展测试表明，它可以扩展到训练拥有 1 万亿参数的`dense`模型。在我们的实验中，GPT 1T 模型在 A100 GPU 上实现了每秒 84 万亿次浮点运算（TFLOPS），而 GPT 175B 模型在 A100 GPU 上实现了每秒 159 万亿次浮点运算（TFLOPS）。与 FairScale 的原始实现相比，启用 CPU offload 时，原生 FSDP 实现显著改善了模型初始化时间。

## FSDP 的工作原理

在标准的 DDP 训练中，每个worker处理一个单独的批次，并使用 [all-reduce 操作](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#allreduce) 在所有worker之间对梯度进行求和。虽然 DDP 已经变得非常流行，但它占用了比实际需求更多的 GPU 内存，因为模型权重和优化器状态在所有 DDP worker之间是重复的。

减少重复的一种方法是应用一种称为全参数分片的过程，其中只有本地只存储计算所需的模型参数、梯度和优化器的子集的一个分片。微软推广了这种实现方法，称为 ZeRO-3。

解锁全参数分片的关键见解是，我们可以将 DDP 中的 [all-reduce](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#allreduce) 操作分解为单独的 [reduce-scatter](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#reducescatter) 和 [all-gather](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#allgather) 操作：

![全分片数据并行图](https://engineering.fb.com/wp-content/uploads/2021/07/FSDP-graph-2a.png?w=1024)

> all-reduce 作为 reduce-scatter 和 all-gather 的组合。标准的 all-reduce 操作可以分解为两个独立的阶段：reduce-scatter 和 all-gather。
>
> - 在 reduce-scatter 阶段，梯度根据每个 GPU 的排名索引在 GPU 之间以相等的块进行求和。
> - 在 all-gather 阶段，每个 GPU 上可用的聚合梯度的分片部分被提供给所有 GPU。

我们可以重新排列 reduce-scatter 和 all-gather，使得每个 DDP worker只需要存储单个分片的参数和优化器状态。下图展示了标准 DDP 训练（顶部）和 FSDP 训练（底部）的对比：

![全分片数据并行图](https://engineering.fb.com/wp-content/uploads/2021/07/FSDP-Graph-2.png?w=907)

> 标准数据并行训练与全分片数据并行训练的对比

上述工作流概述了 DDP 和 FSDP 的幕后流程。我们先来了解一下 DDP 是如何工作的，然后再看 FSDP 是如何改进它的。

在标准数据并行 DDP  训练方法中，每个 worker (加速器 / GPU)  上都会保留一份模型的所有参数、梯度和优化器状态的副本，并且仅对数据的分片进行一系列前向和后向传播。每个 worker 会获取不同的数据，这些数据会经过前向传播，计算损失，然后再反向传播以生成梯度。接着，执行 all-reduce 操作，此时每个worker 从其余 worker 获取梯度并取平均。这样一轮下来，每个worker上的梯度都是相同的，且都是全局梯度，接着优化器再用这些梯度来更新模型参数。我们可以看到，每个 GPU 上都保留完整副本会消耗大量的显存，这限制了该方法所能支持的 batch size 以及模型尺寸。

在 FSDP 中， FSDP 将所有这些优化器状态、梯度和模型参数分片到数据并行 worker上， 每个 加速器/ GPU 上只存储一个模型的分片，进一步地，还可以通过将这些张量 offload 到 CPU 内存来支持那些 GPU 显存容纳不下的大模型。

然后，在具体运行时，与 DDP 类似，FSDP 的每个 worker 获取不同的数据。在前向传播过程中，如果启用了 CPU offload，则首先将本地分片的参数搬到 GPU/加速器。然后，每个 worker 对给定的 FSDP 包装模块/层执行 all-gather 操作其它 GPU 收集所需的参数，执行前向计算，然后释放/清空其他worker的参数分片。在对所有 FSDP 模块全部执行该操作后就是计算损失，然后是后向传播。在后向传播期间，再次执行 all-gather 操作以获取给定 FSDP 模块所需的所有参数，执行计算以获得局部梯度，然后再次释放其他worker的分片。最后，使用 reduce-scatter 操作对局部梯度进行平均并将相应分片给对应的worker，该操作使得每个worker都可以更新其本地分片的参数。如果启用了 CPU offload的话，梯度会传给 CPU，以便直接在 CPU 上更新参数。

为了最大化内存效率，我们可以在每层的前向传播之后丢弃完整权重，为后续层节省内存。这可以通过将 FSDP 包装器应用于网络中的每一层来实现（使用 reshard_after_forward=True）。

下图展示了 FSDP 在 2 个数据并行进程中的工作流程：

![img](https://pytorch.org/assets/images/fsdp_workflow.png)

> 图 3. FSDP 工作流程

通常，模型层以嵌套方式包装在 FSDP 中，因此只有单个 FSDP 实例中的层需要在正向或反向计算期间将完整参数收集到单个设备上。收集的完整参数将在计算后立即释放，释放的内存可用于下一层的计算。通过这种方式，可以节省峰值 GPU 内存，从而使训练能够扩展到使用更大的模型规模或更大的批量大小。为了进一步最大化内存效率，FSDP 可以在实例不活跃于计算时将参数、梯度和优化器状态 offload 到 CPU。

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

## 在 PyTorch 中使用 FSDP

在 PyTorch 中，有两种方法可以将模型包装在 FSDP 中。自动包装是 DDP 的即插即用替换；手动包装需要对模型定义代码进行最小的更改，并能够探索复杂的分片策略。

### 自动包装

模型层应以嵌套方式包装在 FSDP 中，以节省峰值内存并启用通信和计算重叠。最简单的方法是自动包装，它可以作为 DDP 的即插即用替换，而无需更改其余代码。

`fsdp_auto_wrap_policy` 参数允许指定一个可调用函数，以递归方式将层包装在 FSDP 中。PyTorch FSDP 提供的 `default_auto_wrap_policy` 函数递归地包装参数数量大于 1 亿的层。您可以根据需要提供自己的包装策略。自定义包装策略的示例显示在 [FSDP API 文档](https://pytorch.org/docs/stable/fsdp.html) 中。

此外，可以选择配置 `cpu_offload`，以在计算中不使用这些参数时将包装的参数offload 到 CPU。这可以进一步提高内存效率，但代价是主机和设备之间的数据传输开销。

下面的示例展示了如何使用自动包装来包装 FSDP。

```python
from torch.distributed.fsdp import (
   FullyShardedDataParallel,
   CPUOffload,
)
from torch.distributed.fsdp.wrap import (
   default_auto_wrap_policy,
)
import torch.nn as nn

class model(nn.Module):
   def __init__(self):
       super().__init__()
       self.layer1 = nn.Linear(8, 4)
       self.layer2 = nn.Linear(4, 16)
       self.layer3 = nn.Linear(16, 4)

model = DistributedDataParallel(model())
fsdp_model = FullyShardedDataParallel(
   model(),
   fsdp_auto_wrap_policy=default_auto_wrap_policy,
   cpu_offload=CPUOffload(offload_params=True),
)
```

### 手动包装

手动包装可用于通过有选择地对模型的某些部分应用 `wrap` 来探索复杂的分片策略。总体设置可以通过 `enable_wrap()` 上下文管理器传递。

```python
from torch.distributed.fsdp import (
   FullyShardedDataParallel,
   CPUOffload,
)
from torch.distributed.fsdp.wrap import (
   enable_wrap,
   wrap,
)
import torch.nn as nn
from typing import Dict


class model(nn.Module):
   def __init__(self):
       super().__init__()
       self.layer1 = wrap(nn.Linear(8, 4))
       self.layer2 = nn.Linear(4, 16)
       self.layer3 = wrap(nn.Linear(16, 4))

wrapper_kwargs = Dict(cpu_offload=CPUOffload(offload_params=True))
with enable_wrap(wrapper_cls=FullyShardedDataParallel, **wrapper_kwargs):
   fsdp_model = wrap(model())
```

使用上述两种方法之一将模型包装在 FSDP 中后，模型可以以类似于本地训练的方式进行训练，如下所示：

```python
optim = torch.optim.Adam(fsdp_model.parameters(), lr=0.0001)
for sample, label in next_batch():
  out = fsdp_model(input)
  loss = criterion(out, label)
  loss.backward()
  optim.step()
```

## FSDP 对比 DDP 实例

我们以基于 GPT-2 的 Large (762M) 和 XL (1.5B) 模型的因果语言建模任务为例。

以下是预训练 GPT-2 模型的代码。其与 [此处](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm_no_trainer.py) 的官方因果语言建模示例相似，仅增加了 2 个参数 `n_train` (2000) 和 `n_val` (500) 以防止对整个数据集进行预处理/训练，从而支持更快地进行概念验证。

[run_clm_no_trainer.py](https://huggingface.co/blog/assets/62_pytorch_fsdp/run_clm_no_trainer.py)

运行 `accelerate config` 命令后得到的 FSDP 配置示例如下:

```bash
compute_environment: LOCAL_MACHINE
deepspeed_config: {}
distributed_type: FSDP
fsdp_config:
  min_num_params: 2000
  offload_params: false
  sharding_strategy: 1
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
mixed_precision: 'no'
num_machines: 1
num_processes: 2
use_cpu: false
```

### 多 GPU FSDP

本文我们使用单节点多 GPU 上作为实验平台。我们比较了分布式数据并行 (DDP) 和 FSDP 在各种不同配置下的性能。我们可以看到，对 GPT-2 Large(762M) 模型而言，DDP 尚能够支持其中某些 batch size 而不会引起内存不足 (OOM) 错误。但当使用 GPT-2 XL (1.5B) 时，即使 batch size 为 1，DDP 也会失败并出现 OOM 错误。同时，我们看到，FSDP 可以支持以更大的 batch size 训练 GPT-2 Large 模型，同时它还可以使用较大的 batch size 训练 DDP 训练不了的 GPT-2 XL 模型。

**硬件配置**: 2 张 24GB 英伟达 Titan RTX GPU。

GPT-2 Large 模型 (762M 参数) 的训练命令如下:

```bash
export BS=#`try with different batch sizes till you don't get OOM error,
#i.e., start with larger batch size and go on decreasing till it fits on GPU`

time accelerate launch run_clm_no_trainer.py \
--model_name_or_path gpt2-large \
--dataset_name wikitext \
--dataset_config_name wikitext-2-raw-v1 \
--per_device_train_batch_size $BS
--per_device_eval_batch_size $BS
--num_train_epochs 1
--block_size 12
```

FSDP 运行截屏:

![FSDP 运行截屏](https://huggingface.co/blog/assets/62_pytorch_fsdp/sample_fsdp_run.png)

表 1: GPT-2 Large (762M) 模型 FSDP 训练性能基准测试

| 并行方法                                                    | 最大 Batch Size ($BS) | 大致训练时间 (分钟) |
| :---------------------------------------------------------- | :-------------------: | :-----------------: |
| DDP                                                         |           7           |         15          |
| DDP + FP16                                                  |           7           |          8          |
| FSDP (配置: SHARD_GRAD_OP)                                  |          11           |         11          |
| FSDP (配置: min_num_params = 1M + FULL_SHARD)               |          15           |         12          |
| FSDP (配置: min_num_params = 2K + FULL_SHARD)               |          15           |         13          |
| FSDP (配置: min_num_params = 1M + FULL_SHARD + CPU offload) |          20           |         23          |
| FSDP (配置: min_num_params = 2K + FULL_SHARD + CPU offload) |          22           |         24          |

从表 1 中我们可以看到，相对于 DDP 而言，FSDP **支持更大的 batch size**，在不使用和使用 CPU offload设置的情况下 FSDP 支持的最大 batch size 分别可达 DDP 的 **2 倍及 3 倍**。从训练时间来看，混合精度的 DDP 最快，其后是分别使用 ZeRO 阶段 2 和阶段 3 的 FSDP。由于因果语言建模的任务的上下文序列长度 ( `--block_size` ) 是固定的，因此 FSDP 在训练时间上加速还不是太高。对于动态 batch size 的应用而言，支持更大 batch size 的 FSDP 可能会在训练时间方面有更大的加速。目前，FSDP 的混合精度支持在 `transformers` 上还存在一些 [问题](https://github.com/pytorch/pytorch/issues/75676)。一旦问题解决，训练时间将会进一步显著缩短。

### 使用 CPU offload来支持放不进 GPU 显存的大模型训练

训练 GPT-2 XL (1.5B) 模型的命令如下:

```bash
export BS=#`try with different batch sizes till you don't get OOM error,
#i.e., start with larger batch size and go on decreasing till it fits on GPU`

time accelerate launch run_clm_no_trainer.py \
--model_name_or_path gpt2-xl \
--dataset_name wikitext \
--dataset_config_name wikitext-2-raw-v1 \
--per_device_train_batch_size $BS
--per_device_eval_batch_size $BS
--num_train_epochs 1
--block_size 12
```

> 表 2: GPT-2 XL (1.5B) 模型上的 FSDP 基准测试

|                    并行方法                    | 最大 Batch Size ($BS) | GPU 数 | 大致训练时间 (小时) |                                                                                              备注                                                                                              |
| :--------------------------------------------: | :-------------------: | :----: | :-----------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                      DDP                       |           1           |   1    |         NA          | OOM Error RuntimeError: CUDA out of memory. Tried to allocate 40.00 MiB (GPU 0; 23.65 GiB total capacity; 22.27 GiB already allocated; 20.31 MiB free; 22.76 GiB reserved in total by PyTorch) |
|                      DDP                       |           1           |   2    |         NA          | OOM Error RuntimeError: CUDA out of memory. Tried to allocate 40.00 MiB (GPU 0; 23.65 GiB total capacity; 22.27 GiB already allocated; 20.31 MiB free; 22.76 GiB reserved in total by PyTorch) |
|                   DDP + FP16                   |           1           |   1    |         NA          | OOM Error RuntimeError: CUDA out of memory. Tried to allocate 40.00 MiB (GPU 0; 23.65 GiB total capacity; 22.27 GiB already allocated; 20.31 MiB free; 22.76 GiB reserved in total by PyTorch) |
|        FSDP (配置: min_num_params = 2K)        |           5           |   2    |         0.6         |                                                                                                                                                                                                |
| FSDP (配置: min_num_params = 2K + CPU offload) |          10           |   1    |          3          |                                                                                                                                                                                                |
| FSDP (配置: min_num_params = 2K + CPU offload) |          14           |   2    |        1.16         |                                                                                                                                                                                                |

从表 2 中，我们可以观察到 DDP (带和不带 fp16) 甚至在 batch size 为 1 的情况下就会出现 CUDA OOM 错误，从而无法运行。而开启了 ZeRO- 阶段 3 的 FSDP 能够以 batch size 为 5 (总 batch size = 10 (5 × 2) ) 在 2 个 GPU 上运行。当使用 2 个 GPU 时，开启了 CPU offload的 FSDP 还能将最大 batch size 进一步增加到每 GPU 14。 **开启了 CPU offload的 FSDP 可以在单个 GPU 上训练 GPT-2 1.5B 模型，batch size 为 10**。这使得机器学习从业者能够用最少的计算资源来训练大模型，从而助力大模型训练民主化。

## 基准测试结果

我们在 AWS 集群上使用 PyTorch FSDP 对 175B 和 1T GPT 模型进行了广泛的扩展测试。每个集群节点是配备 8 个 [NVIDIA A100-SXM4-40GB](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf) GPU 的实例，节点间通过 AWS Elastic Fabric Adapter (EFA) 连接，网络带宽为 400 Gbps。

GPT 模型使用 [minGPT](https://github.com/karpathy/minGPT) 实现。随机生成的输入数据集用于基准测试目的。所有实验均使用 50K 词汇量、fp16 精度和 [SGD](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html) 优化器运行。

| 模型     | 层数 | 隐藏大小 | 注意力头 | 模型大小（十亿参数） |
| -------- | ---- | -------- | -------- | -------------------- |
| GPT 175B | 96   | 12288    | 96       | 175                  |
| GPT 1T   | 128  | 25600    | 160      | 1008                 |

除了在实验中使用带有参数 CPU offload 的 FSDP 外，还在测试中应用了 PyTorch 的 [激活检查点功能](https://pytorch.org/docs/stable/checkpoint.html)。

在 128 个 GPU 上，GPT 175B 模型的最大每 GPU 吞吐量为 159 万亿次浮点运算/秒（占 NVIDIA A100 峰值理论性能的 51%，即每 GPU 312 万亿次浮点运算/秒），批量大小为 20，序列长度为 512；进一步增加 GPU 数量会导致每 GPU 吞吐量下降，因为节点间的通信量增加。

对于 GPT 1T 模型，在 128 个 GPU 上，批量大小为 4，序列长度为 2048 时，最大每 GPU 吞吐量为 84 万亿次浮点运算/秒（占峰值吞吐量的 27%）。然而，进一步增加 GPU 数量对每 GPU 吞吐量的影响不大，因为我们观察到 1T 模型训练中的最大瓶颈不是来自通信，而是来自 CUDA 缓存分配器在峰值 GPU 内存接近极限时的缓慢。使用具有更大内存容量的 A100 80G GPU 将主要解决此问题，并有助于扩展批量大小以实现更大的吞吐量。

![img](https://pytorch.org/assets/images/175b_throught.png)

![img](https://pytorch.org/assets/images/1t_thought.png)

## Accelerate 的 FSDP 集成的功能和限制

下面，我们深入了解以下 Accelerate 对 FSDP 的集成中，支持了那些功能，有什么已知的限制。

**支持 FSDP 所需的 PyTorch 版本**: PyTorch Nightly 或 1.12.0 之后的版本。

**命令行支持的配置:**

1. **分片策略**: \[1\] FULL_SHARD, \[2\] SHARD_GRAD_OP
2. **Min Num Params**: FSDP 默认自动包装的最小参数量。
3. **Offload Params**: 是否将参数和梯度offload到 CPU。

如果想要对更多的控制参数进行配置，用户可以利用 `FullyShardedDataParallelPlugin` ，其可以指定 `auto_wrap_policy` 、 `backward_prefetch` 以及 `ignored_modules` 。

创建该类的实例后，用户可以在创建 Accelerator 对象时把该实例传进去。

有关这些选项的更多信息，请参阅 PyTorch [FullyShardedDataParallel](https://github.com/pytorch/pytorch/blob/0df2e863fbd5993a7b9e652910792bd21a516ff3/torch/distributed/fsdp/filled_sharded_data_parallel.py#L236) 代码。

接下来，我们体会下 `min_num_params` 配置的重要性。以下内容摘自 \[8\]，它详细说明了 FSDP 自动包装策略的重要性。

![FSDP 自动包装策略的重要性](https://huggingface.co/blog/assets/62_pytorch_fsdp/auto_wrap_importance.png)

(图源: [链接](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html))

当使用 `default_auto_wrap_policy` 时，如果该层的参数量超过 `min_num_params` ，则该层将被包装在一个 FSDP 模块中。官方有一个在 GLUE MRPC 任务上微调 BERT-Large (330M) 模型的示例代码，其完整地展示了如何正确使用 FSDP 功能，其中还包含了用于跟踪峰值内存使用情况的代码。

[fsdp_with_peak_mem_tracking.py](https://github.com/huggingface/accelerate/tree/main/examples/by_feature/fsdp_with_peak_mem_tracking.py)

我们利用 Accelerate 的跟踪功能来记录训练和评估期间的峰值内存使用情况以及模型准确率指标。下图展示了 wandb [实验台](https://wandb.ai/smangrul/FSDP-Test?workspace=user-smangrul) 页面的截图。

![wandb 实验台](https://huggingface.co/blog/assets/62_pytorch_fsdp/wandb_run.png)

我们可以看到，DDP 占用的内存是使用了自动模型包装功能的 FSDP 的两倍。不带自动模型包装的 FSDP 比带自动模型包装的 FSDP 的内存占用更多，但比 DDP 少得多。与 `min_num_params=1M` 时相比， `min_num_params=2k` 时带自动模型包装的 FSDP 占用的内存略少。这凸显了 FSDP 自动模型包装策略的重要性，用户应该调整 `min_num_params` 以找到能显著节省内存又不会导致大量通信开销的设置。如 \[8\] 中所述，PyTorch 团队也在为此开发自动配置调优工具。

### **需要注意的一些事项**

- PyTorch FSDP 会自动对模型子模块进行包装、将参数摊平并对其进行原位分片。因此，在模型包装之前创建的任何优化器都会被破坏并导致更多的内存占用。因此，强烈建议在对模型调用 `prepare` 方法后再创建优化器，这样效率会更高。对单模型而言，如果没有按照顺序调用的话， `Accelerate` 会抛出以下告警信息，并自动帮你包装模型并创建优化器。

  > FSDP Warning: When using FSDP, it is efficient and recommended to call prepare for the model before creating the optimizer

即使如此，我们还是推荐用户在使用 FSDP 时用以下方式显式准备模型和优化器:

```diff
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", return_dict=True)
+ model = accelerator.prepare(model)

optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr)

- model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(model,
- optimizer, train_dataloader, eval_dataloader, lr_scheduler
- )

+ optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
+ optimizer, train_dataloader, eval_dataloader, lr_scheduler
+ )
```

- 对单模型而言，如果你的模型有多组参数，而你想为它们设置不同优化器超参。此时，如果你对整个模型统一调用 `prepare` 方法，这些参数的组别信息会丢失，你会看到如下告警信息:

  > FSDP Warning: When using FSDP, several parameter groups will be conflated into a single one due to nested module wrapping and parameter flattening.

告警信息表明，在使用 FSDP 对模型进行包装后，之前创建的参数组信息丢失了。因为 FSDP 会将嵌套式的模块参数摊平为一维数组 (一个数组可能包含多个子模块的参数)。举个例子，下面是 GPU 0 上 FSDP 模型的有名称的参数 (当使用 2 个 GPU 时，FSDP 会把第一个分片的参数给 GPU 0， 因此其一维数组中大约会有 55M (110M / 2) 个参数)。此时，如果我们在 FSDP 包装前将 BERT-Base 模型的 \[bias, LayerNorm.weight\] 参数的权重衰减设为 0，则在模型包装后，该设置将无效。原因是，你可以看到下面这些字符串中均已不含这俩参数的名字，这俩参数已经被并入了其他层。想要了解更多细节，可参阅本 [问题](https://github.com/pytorch/pytorch/issues/76501) (其中写道: `原模型参数没有 .grads 属性意味着它们无法单独被优化器优化 (这就是我们为什么不能支持对多组参数设置不同的优化器超参)` )。

```
{
'_fsdp_wrapped_module.flat_param': torch.Size([494209]),

'_fsdp_wrapped_module._fpw_module.bert.embeddings.word_embeddings._fsdp_wrapped_module.flat_param': torch.Size([11720448]),

'_fsdp_wrapped_module._fpw_module.bert.encoder._fsdp_wrapped_module.flat_param': torch.Size([42527232])
}
```

- 如果是多模型情况，须在创建优化器之前调用模型 `prepare` 方法，否则会抛出错误。

## Reference

\[1\] [Train Large, Then Compress: Rethinking Model Size for Efficient Training and Inference of Transformers](http://nlp.cs.berkeley.edu/pubs/Li-Wallace-Shen-Lin-Keutzer-Klein-Gonzalez_2020_Transformers_paper.pdf)

\[2\] [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/pdf/1910.02054v3.pdf)

\[3\] [DeepSpeed: Extreme-scale model training for everyone - Microsoft Research](https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/)

\[4\] [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/pdf/1909.08053.pdf)

\[5\] [Introducing GPipe, an Open Source Library for Efficiently Training Large-scale Neural Network Models](https://ai.googleblog.com/2019/03/introducing-gpipe-open-source-library.html)

\[6\] [Which hardware do you need to train a 176B parameters model?](https://bigscience.huggingface.co/blog/which-hardware-to-train-a-176b-parameters-model)

\[7\] [Introducing PyTorch Fully Sharded Data Parallel (FSDP) API | PyTorch](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/)

- https://pytorch.org/docs/stable/fsdp.html

- https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/

- https://engineering.fb.com/2021/07/15/open-source/fsdp/

- https://pytorch.org/tutorials/intermediate/FSDP_advanced_tutorial.html?highlight=fsdp

- https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
