# 使用全分片数据并行 (FSDP) 进行高级模型训练

在本教程中，我们以 HuggingFace (HF) T5 模型为例，使用 FSDP 进行文本摘要的微调。

该示例使用 Wikihow 数据集，为了简单起见，我们将在单节点 P4dn 实例上展示训练过程，该实例配备 8 个 A100 GPU。我们现在已经有了几篇关于多节点集群上大规模 FSDP 训练的博客文章（[链接1](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/)，[链接2](https://engineering.fb.com/2021/07/15/open-source/fsdp/)）和一篇 [论文](https://arxiv.org/abs/2304.11277)。

FSDP 的主要优势之一是减少每个 GPU 的内存占用。这使得与 DDP 相比，可以在更低的总内存下训练更大的模型，并利用计算和通信的重叠来高效地训练模型。这种减少的内存压力可以用于训练更大的模型或增加批量大小，从而潜在地提高整体训练吞吐量。你可以在 [这里](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/) 阅读更多关于 PyTorch FSDP 的信息。

## 本教程中的 FSDP 功能

- Transformer 自动包装策略
- 混合精度
- 在设备上初始化 FSDP 模型
- 分片策略
- 反向预取
- 通过流式传输到 CPU 保存模型检查点

## FSDP 工作原理回顾

在高层次上，FSDP 的工作原理如下：

*在构造函数中*

- 分片模型参数，每个 rank 只保留其自己的分片

*在前向传播中*

- 运行 all_gather 以从所有 rank 收集所有分片，以恢复该 FSDP 单元的完整参数并进行前向计算
- 丢弃刚刚收集的非拥有的参数分片以释放内存

*在反向传播中*

- 运行 all_gather 以从所有 rank 收集所有分片，以恢复该 FSDP 单元的完整参数并进行反向计算
- 丢弃非拥有的参数以释放内存
- 运行 reduce_scatter 以同步梯度

## 微调 HF T5

HF T5 预训练模型有四种不同的大小，从 6000 万个参数的小模型到 110 亿参数的 XXL 模型。在本教程中，我们演示了使用 FSDP 对 T5 3B 进行文本摘要微调的过程，使用的是 WikiHow 数据集。本教程的主要重点是突出 FSDP 中可用于训练超过 30 亿参数的大规模模型的不同功能。此外，我们还涵盖了针对 Transformer 模型的特定功能。本教程的代码可在 [Pytorch 示例](https://github.com/pytorch/examples/tree/main/distributed/FSDP/) 中找到。

*设置*

1.1 安装最新的 PyTorch

```bash
pip3 install torch torchvision torchaudio
```

1.2 数据集设置

请创建一个数据文件夹，从 [wikihowAll.csv](https://ucsb.app.box.com/s/ap23l8gafpezf4tq3wapr6u8241zz358) 和 [wikihowSep.cs](https://ucsb.app.box.com/s/7yq601ijl1lzvlfu4rjdbbxforzd2oag) 下载 WikiHow 数据集，并将它们放在数据文件夹中。我们将使用 [summarization_dataset](https://github.com/pytorch/examples/blob/main/distributed/FSDP/summarization_dataset.py) 中的 wikihow 数据集。

接下来，我们将以下代码片段添加到 Python 脚本“T5_training.py”中。

注意

本教程的完整源代码可在 [PyTorch 示例](https://github.com/pytorch/examples/tree/main/distributed/FSDP/) 中找到。

1.3 导入必要的包：

```python
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer, GPT2TokenizerFast
from transformers import T5Tokenizer, T5ForConditionalGeneration
import functools
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers.models.t5.modeling_t5 import T5Block

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
 checkpoint_wrapper,
 CheckpointImpl,
 apply_activation_checkpointing_wrapper)

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from functools import partial
from torch.utils.data import DataLoader
from pathlib import Path
from summarization_dataset import *
from transformers.models.t5.modeling_t5 import T5Block
from typing import Type
import time
import tqdm
from datetime import datetime
```

1.4 分布式训练设置。这里我们使用两个辅助函数来初始化分布式训练的进程，并在训练完成后进行清理。在本教程中，我们将使用 torch elastic，使用 [torchrun](https://pytorch.org/docs/stable/elastic/run.html)，它会自动设置 worker RANK 和 WORLD_SIZE。

```python
def setup():
    # 初始化进程组
    dist.init_process_group("nccl")

def cleanup():
    dist.destroy_process_group()
```

2.1 设置 HuggingFace T5 模型：

```python
def setup_model(model_name):
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer =  T5Tokenizer.from_pretrained(model_name)
    return model, tokenizer
```

我们还添加了几个辅助函数来获取运行日期和格式化内存指标。

```python
def get_date_of_run():
    """创建用于文件保存唯一性的日期和时间
    示例: 2022-05-07-08:31:12_PM'
    """
    date_of_run = datetime.now().strftime("%Y-%m-%d-%I:%M:%S_%p")
    print(f"--> 当前运行日期和时间 = {date_of_run}")
    return date_of_run

def format_metrics_to_gb(item):
    """快速函数将数字格式化为千兆字节并保留 4 位小数精度"""
    metric_num = item / g_gigabyte
    metric_num = round(metric_num, ndigits=4)
    return metric_num
```

2.2 定义训练函数：

```python
def train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=None):
    model.train()
    local_rank = int(os.environ['LOCAL_RANK'])
    fsdp_loss = torch.zeros(2).to(local_rank)

    if sampler:
        sampler.set_epoch(epoch)
    if rank==0:
        inner_pbar = tqdm.tqdm(
            range(len(train_loader)), colour="blue", desc="r0 训练 Epoch"
        )
    for batch in train_loader:
        for key in batch.keys():
            batch[key] = batch[key].to(local_rank)
        optimizer.zero_grad()
        output = model(input_ids=batch["source_ids"],attention_mask=batch["source_mask"],labels=batch["target_ids"] )
        loss = output["loss"]
        loss.backward()
        optimizer.step()
        fsdp_loss[0] += loss.item()
        fsdp_loss[1] += len(batch)
        if rank==0:
            inner_pbar.update(1)

    dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)
    train_accuracy = fsdp_loss[0] / fsdp_loss[1]


    if rank == 0:
        inner_pbar.close()
        print(
                f"训练 Epoch: \t{epoch}, 损失: \t{train_accuracy:.4f}"
            )
    return train_accuracy
```

2.3 定义验证函数：

```python
def validation(model, rank, world_size, val_loader):
    model.eval()
    correct = 0
    local_rank = int(os.environ['LOCAL_RANK'])
    fsdp_loss = torch.zeros(3).to(local_rank)
    if rank == 0:
        inner_pbar = tqdm.tqdm(
            range(len(val_loader)), colour="green", desc="验证 Epoch"
        )
    with torch.no_grad():
        for batch in val_loader:
            for key in batch.keys():
                batch[key] = batch[key].to(local_rank)
            output = model(input_ids=batch["source_ids"],attention_mask=batch["source_mask"],labels=batch["target_ids"])
            fsdp_loss[0] += output["loss"].item()  # 累加批量损失
            fsdp_loss[1] += len(batch)

            if rank==0:
                inner_pbar.update(1)

    dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)
    val_loss = fsdp_loss[0] / fsdp_loss[1]
    if rank == 0:
        inner_pbar.close()
        print(f"验证损失: {val_loss:.4f}")
    return val_loss
```

2.4 定义一个分布式训练函数，将模型包装在 FSDP 中：

```python
def fsdp_main(args):

    model, tokenizer = setup_model("t5-base")

    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])


    dataset = load_dataset('wikihow', 'all', data_dir='data/')
    print(dataset.keys())
    print("训练数据集大小: ", dataset['train'].shape)
    print("验证数据集大小: ", dataset['validation'].shape)


    #wikihow(tokenizer, type_path, num_samples, input_length, output_length, print_text=False)
    train_dataset = wikihow(tokenizer, 'train', 1500, 512, 150, False)
    val_dataset = wikihow(tokenizer, 'validation', 300, 512, 150, False)

    sampler1 = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=True)
    sampler2 = DistributedSampler(val_dataset, rank=rank, num_replicas=world_size)

    setup()


    train_kwargs = {'batch_size': args.batch_size, 'sampler': sampler1}
    test_kwargs = {'batch_size': args.test_batch_size, 'sampler': sampler2}
    cuda_kwargs = {'num_workers': 2,
                    'pin_memory': True,
                    'shuffle': False}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, **test_kwargs)

    t5_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            T5Block,
        },
    )
    sharding_strategy: ShardingStrategy = ShardingStrategy.SHARD_GRAD_OP #for Zero2 and FULL_SHARD for Zero3
    torch.cuda.set_device(local_rank)


    #init_start_event = torch.cuda.Event(enable_timing=True)
    #init_end_event = torch.cuda.Event(enable_timing=True)

    #init_start_event.record()

    bf16_ready = (
    torch.version.cuda
    and torch.cuda.is_bf16_supported()
    and LooseVersion(torch.version.cuda) >= "11.0"
    and dist.is_nccl_available()
    and nccl.version() >= (2, 10)
    )

    if bf16_ready:
        mp_policy = bfSixteen
    else:
        mp_policy = None # defaults to fp32

    # 模型在输入到 FSDP 之前位于 CPU 上
    model = FSDP(model,
        auto_wrap_policy=t5_auto_wrap_policy,
        mixed_precision=mp_policy,
        #sharding_strategy=sharding_strategy,
        device_id=torch.cuda.current_device())

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    best_val_loss = float("inf")
    curr_val_loss = float("inf")
    file_save_name = "T5-model-"

    if rank == 0:
        time_of_run = get_date_of_run()
        dur = []
        train_acc_tracking = []
        val_acc_tracking = []
        training_start_time = time.time()

    if rank == 0 and args.track_memory:
        mem_alloc_tracker = []
        mem_reserved_tracker = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_accuracy = train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=sampler1)
        if args.run_validation:
            curr_val_loss = validation(model, rank, world_size, val_loader)
        scheduler.step()

        if rank == 0:

            print(f"--> epoch {epoch} 完成...进入保存和统计区域")

            dur.append(time.time() - t0)
            train_acc_tracking.append(train_accuracy.item())

            if args.run_validation:
                val_acc_tracking.append(curr_val_loss.item())

            if args.track_memory:
                mem_alloc_tracker.append(
                    format_metrics_to_gb(torch.cuda.memory_allocated())
                )
                mem_reserved_tracker.append(
                    format_metrics_to_gb(torch.cuda.memory_reserved())
                )
            print(f"完成保存和统计区域...")

        if args.save_model and curr_val_loss < best_val_loss:

            # 保存
            if rank == 0:
                print(f"--> 进入保存模型状态")

            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(
                model, StateDictType.FULL_STATE_DICT, save_policy
            ):
                cpu_state = model.state_dict()
            #print(f"保存过程: rank {rank} 完成 state_dict")


            if rank == 0:
                print(f"--> 保存模型 ...")
                currEpoch = (
                    "-" + str(epoch) + "-" + str(round(curr_val_loss.item(), 4)) + ".pt"
                )
                print(f"--> 尝试保存模型前缀 {currEpoch}")
                save_name = file_save_name + "-" + time_of_run + "-" + currEpoch
                print(f"--> 保存为模型名称 {save_name}")

                torch.save(cpu_state, save_name)

        if curr_val_loss < best_val_loss:

            best_val_loss = curr_val_loss
            if rank==0:
                print(f"-->>>> 新的验证损失记录: {best_val_loss}")

    dist.barrier()
    cleanup()
```

2.5 解析参数并设置主函数：

```python
if __name__ == '__main__':
    # 训练设置
    parser = argparse.ArgumentParser(description='PyTorch T5 FSDP 示例')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='训练输入批量大小 (默认: 64)')
    parser.add_argument('--test-batch-size', type=int, default=4, metavar='N',
                        help='测试输入批量大小 (默认: 1000)')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='训练的 epoch 数量 (默认: 3)')
    parser.add_argument('--lr', type=float, default=.002, metavar='LR',
                        help='学习率 (默认: .002)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='学习率步长 gamma (默认: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='禁用 CUDA 训练')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='随机种子 (默认: 1)')
    parser.add_argument('--track_memory', action='store_false', default=True,
                        help='跟踪 GPU 内存')
    parser.add_argument('--run_validation', action='store_false', default=True,
                        help='运行验证')
    parser.add_argument('--save-model', action='store_false', default=True,
                        help='保存当前模型')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    fsdp_main(args)
```

使用 torchrun 运行训练：

```bash
torchrun --nnodes 1 --nproc_per_node 4  T5_training.py
```

## Transformer 包装策略

正如在 [之前的教程](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) 中讨论的那样，auto_wrap_policy 是 FSDP 的一个功能，它使得自动分片给定模型并将模型、优化器和梯度分片放入不同的 FSDP 单元变得容易。

对于某些架构（如 Transformer 编码器-解码器），模型的某些部分（如嵌入表）与编码器和解码器共享。在这种情况下，我们需要将嵌入表放在外部 FSDP 单元中，以便编码器和解码器都可以访问它。此外，通过为 Transformer 注册层类，分片计划可以变得更加通信高效。在 PyTorch 1.12 中，FSDP 添加了对此的支持，现在我们有了一个针对 Transformer 的包装策略。

它可以创建如下，其中 T5Block 表示 T5 Transformer 层类（包含 MHSA 和 FFN）。

```python
t5_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            T5Block,
        },
    )
torch.cuda.set_device(local_rank)


model = FSDP(model,
    auto_wrap_policy=t5_auto_wrap_policy)
```

要查看包装后的模型，可以轻松打印模型并直观检查分片和 FSDP 单元。

## 混合精度

FSDP 支持灵活的混合精度训练，允许使用任意降低的精度类型（如 fp16 或 bfloat16）。目前，BFloat16 仅在 Ampere GPU 上可用，因此在使用之前需要确认本机支持。例如，在 V100 上，BFloat16 仍然可以运行，但由于它不是本机运行，可能会导致显著的减速。

要检查 BFloat16 是否本机支持，可以使用以下代码：

```python
bf16_ready = (
    torch.version.cuda
    and torch.cuda.is_bf16_supported()
    and LooseVersion(torch.version.cuda) >= "11.0"
    and dist.is_nccl_available()
    and nccl.version() >= (2, 10)
)
```

FSDP 混合精度的优势之一是提供了对参数、梯度和缓冲区的不同精度级别的精细控制，如下所示：

```python
fpSixteen = MixedPrecision(
    param_dtype=torch.float16,
    # 梯度通信精度
    reduce_dtype=torch.float16,
    # 缓冲区精度
    buffer_dtype=torch.float16,
)

bfSixteen = MixedPrecision(
    param_dtype=torch.bfloat16,
    # 梯度通信精度
    reduce_dtype=torch.bfloat16,
    # 缓冲区精度
    buffer_dtype=torch.bfloat16,
)

fp32_policy = MixedPrecision(
    param_dtype=torch.float32,
    # 梯度通信精度
    reduce_dtype=torch.float32,
    # 缓冲区精度
    buffer_dtype=torch.float32,
)
```

注意，如果未指定某种类型（参数、reduce、buffer），它们将不会被转换。

这种灵活性允许用户进行精细控制，例如仅将梯度通信设置为降低精度，而所有参数/缓冲区计算以全精度进行。这在节点内通信是主要瓶颈且参数/缓冲区必须以全精度以避免精度问题的情况下可能有用。这可以通过以下策略实现：

```python
grad_bf16 = MixedPrecision(reduce_dtype=torch.bfloat16)
```

在 2.4 中，我们只需将相关的混合精度策略添加到 FSDP 包装器中：

```python
model = FSDP(model,
       auto_wrap_policy=t5_auto_wrap_policy,
       mixed_precision=bfSixteen)
```

在我们的实验中，我们观察到通过使用 BFloat16 进行训练，训练速度提高了 4 倍，并且在某些实验中内存减少了约 30%，可以用于增加批量大小。

## 在设备上初始化 FSDP 模型

在 1.12 中，FSDP 支持 device_id 参数，旨在在由 device_id 指定的设备上初始化输入 CPU 模块。当整个模型不适合单个 GPU 但适合主机的 CPU 内存时，这很有用。当指定 device_id 时，FSDP 将模型按 FSDP 单元逐个移动到指定设备，避免 GPU OOM 问题，同时比基于 CPU 的初始化快几倍：

```python
torch.cuda.set_device(local_rank)

 model = FSDP(model,
        auto_wrap_policy=t5_auto_wrap_policy,
        mixed_precision=bfSixteen,
        device_id=torch.cuda.current_device())
```

## 分片策略

FSDP 分片策略默认设置为完全分片模型参数、梯度和优化器状态，这些状态在所有 rank 之间分片（也称为 Zero3 分片）。如果你对 Zero2 分片策略感兴趣，即仅分片优化器状态和梯度，FSDP 通过使用“ShardingStrategy.SHARD_GRAD_OP”而不是“ShardingStrategy.FULL_SHARD”来支持此功能，如下所示：

```python
torch.cuda.set_device(local_rank)

 model = FSDP(model,
        auto_wrap_policy=t5_auto_wrap_policy,
        mixed_precision=bfSixteen,
        device_id=torch.cuda.current_device(),
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP # ZERO2)
```

这将减少 FSDP 中的通信开销，在这种情况下，它在 forward 和 backward 过程中持有完整的参数。

这节省了 backward 中的 all_gather，因此通信更少，代价是内存占用更高。注意，full model params 在 backward 结束时被释放，all_gather 将在下一次 forward 中发生。

## 反向预取

反向预取设置控制何时请求下一个 FSDP 单元的参数。通过将其设置为 BACKWARD_PRE，可以在当前单元计算开始之前请求并更早地到达下一个 FSDP 单元的参数。这重叠了 all_gather 通信和梯度计算，可以提高训练速度，代价是稍微增加内存消耗。它可以在 2.4 中的 FSDP 包装器中使用，如下所示：

```python
torch.cuda.set_device(local_rank)

 model = FSDP(model,
        auto_wrap_policy=t5_auto_wrap_policy,
        mixed_precision=bfSixteen,
        device_id=torch.cuda.current_device(),
        backward_prefetch = BackwardPrefetch.BACKWARD_PRE)
```

backward_prefetch 有两种模式，BACKWARD_PRE 和 BACKWARD_POST。BACKWARD_POST 意味着直到当前 FSDP 单元处理完成时才会请求下一个 FSDP 单元的参数，从而最小化内存开销。在某些情况下，使用 BACKWARD_PRE 可以将模型训练速度提高 2-10%，对于更大的模型，速度提升甚至更高。

## 通过流式传输到 Rank0 CPU 保存模型检查点

要使用 FULL_STATE_DICT 保存模型检查点，即以与本地模型相同的方式保存模型，PyTorch 1.12 提供了几个实用工具来支持保存更大的模型。

首先，可以指定 FullStateDictConfig，允许 state_dict 仅在 rank 0 上填充并卸载到 CPU。

使用此配置时，FSDP 将在 rank 0 上逐个 allgather 模型参数，并将它们卸载到 CPU。当最终保存 state_dict 时，它仅在 rank 0 上填充并包含 CPU 张量。这避免了单个 GPU 内存中模型过大的潜在 OOM 问题，并允许用户检查点模型，其大小大约为用户机器上可用 CPU RAM 的大小。

此功能可以运行如下：

```python
save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
with FSDP.state_dict_type(
            model, StateDictType.FULL_STATE_DICT, save_policy
        ):
            cpu_state = model.state_dict()
if rank == 0:
 save_name = file_save_name + "-" + time_of_run + "-" + currEpoch
 torch.save(cpu_state, save_name)
```

## 总结

在本教程中，我们介绍了 Pytorch 1.12 中 FSDP 的许多新功能，并使用 HF T5 作为运行示例。使用适当的包装策略（尤其是针对 Transformer 模型），结合混合精度和反向预取，应该可以加快你的训练运行。此外，在设备上初始化模型和通过流式传输到 CPU 保存检查点等功能应该有助于避免处理大模型时的 OOM 错误。
