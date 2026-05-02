# PyTorch 全分片数据并行 (FSDP)

## 动机

模型规模从 BERT 的 1.1 亿参数增长到 Megatron-2 的万亿参数（10,000 倍增长），在单机上训练甚至加载大模型变得越来越困难。标准 DDP 在每个 GPU 上维护完整模型副本，内存利用率低；模型并行则引入额外通信开销。分布式训练的关键技术包括：

1. **ZeRO 数据并行** — 分阶段优化内存：阶段 1 分片优化器状态，阶段 2 分片优化器状态+梯度，阶段 3（全分片）分片优化器状态+梯度+模型参数
2. **CPU Offload** — 将 ZeRO 阶段 2 的优化器状态+梯度卸载到 CPU
3. **张量并行** — 层内参数跨 GPU 分片，并行计算且通信开销低
4. **流水线并行** — 不同层放在不同 GPU 上，流水线保持利用率
5. **3D 并行** — ZeRO + 张量并行 + 流水线并行组合（如 BigScience 176B）

FSDP 实现了 ZeRO 阶段 3 的核心思想。DeepSpeed 和 FairScale 率先实现，PyTorch 将 FairScale FSDP 整合进 `torch.distributed` 模块并做了进一步优化。

## FSDP 概念

FSDP 是一种数据并行算法：将模型参数、梯度和优化器状态分片到数据并行 worker 上，每个微批次的计算仍在本地 GPU 上进行。相比 DDP：

- **内存效率**：参数、梯度、优化器状态均匀分片，每个 GPU 只存一个分片
- **计算效率**：通信与前向/后向传播重叠
- **易用性**：作为 `DistributedDataParallel` 的即插即用替换，结果与 DDP 一致

## FSDP 工作原理

### all-reduce 的分解

核心洞察：DDP 的 all-reduce 可分解为 **reduce-scatter** + **all-gather**：

- **reduce-scatter**：梯度按 rank 分块求和，每个 GPU 只得到对应分片的聚合结果
- **all-gather**：各 GPU 交换聚合后的梯度分片，恢复完整梯度

![全分片数据并行图](https://engineering.fb.com/wp-content/uploads/2021/07/FSDP-graph-2a.png?w=1024)

### DDP vs FSDP 流程对比

**DDP**：每个 worker 保留完整模型副本 → 前向传播 → 反向传播 → all-reduce 同步梯度 → 优化器更新。简单但显存浪费严重。

**FSDP**：每个 worker 只保留参数的一个分片，运行时按需收集和释放：

```
构造函数:   分片模型参数，每个 rank 只保留自己的分片

前向传播:
    for layer in fsdp_units:
        all-gather → 收集该层完整参数
        forward   → 执行前向计算
        discard   → 释放非本地分片

反向传播:
    for layer in fsdp_units:
        all-gather     → 收集该层完整参数
        backward       → 执行反向计算
        discard        → 释放参数
        reduce-scatter → 同步梯度分片
```

![DDP vs FSDP 对比](https://engineering.fb.com/wp-content/uploads/2021/07/FSDP-Graph-2.png?w=907)

FSDP 通常以嵌套方式包装模型层，只有当前 FSDP 单元需要将完整参数收集到单个设备上，计算后立即释放。可进一步将参数、梯度、优化器状态 offload 到 CPU 以最大化内存效率。

![FSDP 工作流程](https://pytorch.org/assets/images/fsdp_workflow.png)

## 基础用法：MNIST 示例

> 要求：PyTorch >= 1.12。更早版本将 `size_based_auto_wrap_policy` 替换为 `default_auto_wrap_policy`，将 `auto_wrap_policy` 替换为 `fsdp_auto_wrap_policy`。

### 分布式环境初始化

```python
import os
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()
```

### 模型定义与训练/验证函数

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=None):
    model.train()
    ddp_loss = torch.zeros(2).to(rank)
    if sampler:
        sampler.set_epoch(epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(rank), target.to(rank)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target, reduction="sum")
        loss.backward()
        optimizer.step()
        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(data)
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        print(
            "Train Epoch: {} \tLoss: {:.6f}".format(
                epoch, ddp_loss[0] / ddp_loss[1]
            )
        )


def test(model, rank, world_size, test_loader):
    model.eval()
    ddp_loss = torch.zeros(3).to(rank)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(rank), target.to(rank)
            output = model(data)
            ddp_loss[0] += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            ddp_loss[1] += pred.eq(target.view_as(pred)).sum().item()
            ddp_loss[2] += len(data)
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        print(
            "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
                ddp_loss[0] / ddp_loss[2],
                int(ddp_loss[1]),
                int(ddp_loss[2]),
                100.0 * ddp_loss[1] / ddp_loss[2],
            )
        )
```

### 用 FSDP 包装模型

```python
def fsdp_main(rank, world_size, args):
    setup(rank, world_size)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset1 = datasets.MNIST(
        "../data", train=True, download=True, transform=transform
    )
    dataset2 = datasets.MNIST("../data", train=False, transform=transform)

    sampler1 = DistributedSampler(
        dataset1, rank=rank, num_replicas=world_size, shuffle=True
    )
    sampler2 = DistributedSampler(dataset2, rank=rank, num_replicas=world_size)

    train_loader = torch.utils.data.DataLoader(
        dataset1,
        batch_size=args.batch_size,
        sampler=sampler1,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset2,
        batch_size=args.test_batch_size,
        sampler=sampler2,
        num_workers=2,
        pin_memory=True,
    )

    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=20000
    )
    torch.cuda.set_device(rank)

    model = Net().to(rank)
    model = FSDP(model, auto_wrap_policy=my_auto_wrap_policy)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    for epoch in range(1, args.epochs + 1):
        train(
            args,
            model,
            rank,
            world_size,
            train_loader,
            optimizer,
            epoch,
            sampler=sampler1,
        )
        test(model, rank, world_size, test_loader)
        scheduler.step()

    if args.save_model:
        dist.barrier()
        states = model.state_dict()
        if rank == 0:
            torch.save(states, "mnist_cnn.pt")

    cleanup()
```

### 运行

```python
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch MNIST FSDP Example")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.7)
    parser.add_argument("--save-model", action="store_true", default=False)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(fsdp_main, args=(WORLD_SIZE, args), nprocs=WORLD_SIZE, join=True)
```

## 高级用法：HuggingFace T5 微调

本节以 HuggingFace T5 模型为例，展示 FSDP 的高级功能：

- Transformer 自动包装策略
- 混合精度训练
- 设备端模型初始化
- 分片策略选择（ZeRO-2 / ZeRO-3）
- 反向预取
- 流式 CPU 检查点保存

### Transformer 自动包装策略

对于 Transformer 编码器-解码器架构，嵌入表在编码器和解码器间共享，需要放在外层 FSDP 单元中。通过注册 Transformer 层类（如 `T5Block`），可以实现更通信高效的分片：

```python
from transformers.models.t5.modeling_t5 import T5Block
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

t5_auto_wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={T5Block},
)
```

### 混合精度

FSDP 支持对参数、梯度和缓冲区分别设置不同精度：

```python
from torch.distributed.fsdp import MixedPrecision

# BFloat16（需要 Ampere+ GPU，CUDA >= 11.0，NCCL >= 2.10）
bfSixteen = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)

# FP16
fpSixteen = MixedPrecision(
    param_dtype=torch.float16,
    reduce_dtype=torch.float16,
    buffer_dtype=torch.float16,
)

# 仅降低梯度通信精度
grad_bf16 = MixedPrecision(reduce_dtype=torch.bfloat16)
```

实验表明 BFloat16 训练速度提升约 4 倍，内存减少约 30%。

### 设备端初始化

当模型过大无法在单个 GPU 上初始化时，使用 `device_id` 参数让 FSDP 逐单元将 CPU 模型移到 GPU：

```python
model = FSDP(
    model,
    auto_wrap_policy=t5_auto_wrap_policy,
    mixed_precision=bfSixteen,
    device_id=torch.cuda.current_device(),
)
```

### 分片策略

- `ShardingStrategy.FULL_SHARD`（默认，ZeRO-3）：分片参数+梯度+优化器状态
- `ShardingStrategy.SHARD_GRAD_OP`（ZeRO-2）：仅分片梯度+优化器状态，保留完整参数

```python
from torch.distributed.fsdp import ShardingStrategy

# ZeRO-2：通信更少（省去 backward 的 all-gather），但内存占用更高
model = FSDP(
    model,
    auto_wrap_policy=t5_auto_wrap_policy,
    sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
)
```

### 反向预取

控制何时预取下一个 FSDP 单元的参数，重叠 all_gather 通信与梯度计算：

- `BACKWARD_PRE`：当前单元计算开始前就请求下一单元参数，通信与计算重叠，速度提升 2-10%，内存略增
- `BACKWARD_POST`（默认）：当前单元处理完后才请求，内存开销最小

```python
from torch.distributed.fsdp.fully_sharded_data_parallel import BackwardPrefetch

model = FSDP(
    model,
    auto_wrap_policy=t5_auto_wrap_policy,
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
)
```

### 检查点保存（流式 CPU 卸载）

大模型保存时，使用 `FullStateDictConfig` 在 rank 0 上逐个 allgather 参数并卸载到 CPU，避免 GPU OOM：

```python
from torch.distributed.fsdp import FullStateDictConfig, StateDictType

save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
    cpu_state = model.state_dict()
if rank == 0:
    torch.save(cpu_state, "model.pt")
```

### 完整 T5 微调训练函数

```python
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing_wrapper,
)


def fsdp_main(args):
    model, tokenizer = setup_model("t5-base")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # 数据集准备（略）
    train_loader = ...
    val_loader = ...

    setup()

    t5_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={T5Block},
    )
    torch.cuda.set_device(local_rank)

    model = FSDP(
        model,
        auto_wrap_policy=t5_auto_wrap_policy,
        mixed_precision=bfSixteen,
        device_id=torch.cuda.current_device(),
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    for epoch in range(1, args.epochs + 1):
        train_accuracy = train(
            args, model, rank, world_size, train_loader, optimizer, epoch
        )
        if args.run_validation:
            curr_val_loss = validation(model, rank, world_size, val_loader)
        scheduler.step()

        if args.save_model and curr_val_loss < best_val_loss:
            save_policy = FullStateDictConfig(
                offload_to_cpu=True, rank0_only=True
            )
            with FSDP.state_dict_type(
                model, StateDictType.FULL_STATE_DICT, save_policy
            ):
                cpu_state = model.state_dict()
            if rank == 0:
                torch.save(cpu_state, save_name)

    dist.barrier()
    cleanup()
```

用 torchrun 启动：

```bash
torchrun --nnodes 1 --nproc_per_node 4 T5_training.py
```

## CPU Offload

模型过大即使 FSDP 也无法放入 GPU 时，可将参数和梯度卸载到 CPU。注意这会显著降低训练速度：

```python
model = FSDP(
    model,
    auto_wrap_policy=my_auto_wrap_policy,
    cpu_offload=CPUOffload(offload_params=True),
)
```

## auto_wrap_policy 的重要性

如果不设置 `auto_wrap_policy`，FSDP 将整个模型放在一个单元中，all-gather 会收集所有参数，无法节省内存，也无法实现通信与计算重叠。

`auto_wrap_policy` 按条件（如参数数量）自动将层包装成多个 FSDP 单元。每个单元独立收集/释放参数，峰值内存大幅降低。

```python
# 按参数数量自动包装
my_auto_wrap_policy = functools.partial(
    size_based_auto_wrap_policy, min_num_params=20000
)
model = FSDP(model, auto_wrap_policy=my_auto_wrap_policy)
```

调优 `min_num_params` 很重要：值太小导致通信开销增大，值太大则分片效果不明显。需通过实验找到平衡点。

## 性能基准

### GPT-2 Large (762M) — 2 × Titan RTX 24GB

| 配置 | 最大 Batch Size | 训练时间 (分钟) |
|------|:---:|:---:|
| DDP | 7 | 15 |
| DDP + FP16 | 7 | 8 |
| FSDP (SHARD_GRAD_OP) | 11 | 11 |
| FSDP (min=1M + FULL_SHARD) | 15 | 12 |
| FSDP (min=2K + FULL_SHARD) | 15 | 13 |
| FSDP (min=1M + FULL_SHARD + CPU offload) | 20 | 23 |
| FSDP (min=2K + FULL_SHARD + CPU offload) | 22 | 24 |

FSDP 支持的最大 batch size 可达 DDP 的 2-3 倍。

### GPT-2 XL (1.5B) — 2 × Titan RTX 24GB

| 配置 | 最大 Batch Size | GPU 数 | 训练时间 | 备注 |
|------|:---:|:---:|:---:|------|
| DDP | 1 | 1 | — | OOM |
| DDP | 1 | 2 | — | OOM |
| DDP + FP16 | 1 | 1 | — | OOM |
| FSDP (min=2K) | 5 | 2 | 0.6h | |
| FSDP (min=2K + CPU offload) | 10 | 1 | 3h | 单 GPU 可训练 |
| FSDP (min=2K + CPU offload) | 14 | 2 | 1.16h | |

DDP 在任何配置下均 OOM。FSDP + CPU offload 可在单 GPU 上训练 1.5B 模型。

### 大规模扩展：GPT 175B / 1T

**环境**：AWS 集群，每节点 8 × A100-SXM4-40GB，EFA 400Gbps 互联。minGPT 实现，50K 词表，fp16，SGD 优化器。

| 模型 | 层数 | 隐藏维度 | 注意力头 | 参数量 |
|------|------|---------|---------|--------|
| GPT 175B | 96 | 12288 | 96 | 175B |
| GPT 1T | 128 | 25600 | 160 | 1008B |

- **GPT 175B**：128 GPU，batch=20，seq=512 → 159 TFLOPS/GPU（峰值 51%）
- **GPT 1T**：128 GPU，batch=4，seq=2048 → 84 TFLOPS/GPU（峰值 27%），瓶颈在 CUDA 缓存分配器而非通信

## Accelerate FSDP 集成注意事项

1. **优化器创建顺序**：必须在模型 `prepare` 之后再创建优化器。FSDP 会对子模块包装、参数摊平并分片，之前创建的优化器会被破坏
2. **参数组丢失**：FSDP 将嵌套模块参数摊平为一维数组，原有的参数组设置（如不同权重衰减）会失效
3. **多模型场景**：必须在创建优化器前调用每个模型的 `prepare` 方法

```python
# 正确顺序
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased")
model = accelerator.prepare(model)  # 先包装模型
optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr)
optimizer, train_loader, eval_loader, lr_scheduler = accelerator.prepare(
    optimizer, train_loader, eval_loader, lr_scheduler
)
```

## 参考资料

- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/pdf/1910.02054v3.pdf)
- [Introducing PyTorch FSDP API](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/)
- [PyTorch FSDP 文档](https://pytorch.org/docs/stable/fsdp.html)
- [PyTorch FSDP 高级教程](https://pytorch.org/tutorials/intermediate/FSDP_advanced_tutorial.html)
- [PyTorch FSDP 基础教程](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [FSDP Notes](https://pytorch.org/docs/stable/notes/fsdp.html)
- [DeepSpeed: Extreme-scale model training for everyone](https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/)
- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/pdf/1909.08053.pdf)
