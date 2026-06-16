# 开始使用 Fully Sharded Data Parallel (FSDP)

**作者**: [Hamid Shojanazeri](https://github.com/HamidShojanazeri), [Yanli Zhao](https://github.com/zhaojuanmao), [Shen Li](https://mrshenli.github.io/)

> **注意**: 在 [GitHub](https://github.com/pytorch/tutorials/blob/main/intermediate_source/FSDP_tutorial.rst) 上查看和编辑此教程。

大规模训练 AI 模型是一项具有挑战性的任务，需要大量的计算能力和资源。同时，处理这些超大规模模型的训练也带来了相当大的工程复杂性。`PyTorch FSDP` 在 PyTorch 1.11 中发布，使得这一过程变得更加简单。

在本教程中，我们将展示如何使用 `FSDP API` 来训练简单的 MNIST 模型，这些 API 和逻辑也可以扩展到其他更大的模型，例如 [HuggingFace BERT 模型](https://huggingface.co/blog/zero-deepspeed-fairscale) 和 [GPT-3 模型（最多 1T 参数）](https://pytorch.medium.com/training-a-1-trillion-parameter-model-with-pytorch-fully-sharded-data-parallel-on-aws-3ac13aa96cff)。示例代码参考了 [这里](https://github.com/yqhu/mnist_examples) 的 DDP MNIST 代码。

## FSDP 的工作原理

在 `DistributedDataParallel` (DDP) 训练中，每个进程/工作节点拥有模型的副本，并处理一批数据，最后通过 all-reduce 汇总不同工作节点的梯度。在 DDP 中，模型的权重和优化器状态会在所有工作节点之间复制。FSDP 是一种数据并行方式，它将模型参数、优化器状态和梯度分片到 DDP 的各个 rank 中。

使用 FSDP 进行训练时，GPU 内存占用比使用 DDP 时更小。这使得训练一些非常大的模型成为可能，因为它允许更大的模型或批次大小适应设备。不过，这会带来通信量的增加。通过内部优化（如通信和计算的重叠）可以减少通信开销。

<img src="https://pytorch.org/tutorials/_images/fsdp_workflow.png" alt="FSDP workflow" style="zoom: 30%;" />

FSDP 的工作流程大致如下：

- 在构造函数中

  - 分片模型参数，每个 rank 只保留自己的分片。

- 在前向过程中

  - 运行 all_gather 从所有 rank 收集所有分片，以恢复该 FSDP 单元的完整参数。

  - 运行前向计算。

  - 丢弃刚刚收集的参数分片。

- 在反向过程中

  - 运行 all_gather 从所有 rank 收集所有分片，以恢复该 FSDP 单元的完整参数。

  - 运行反向计算。

  - 运行 reduce_scatter 同步梯度。

  - 丢弃参数。

FSDP 的分片可以看作是将 DDP 的梯度 all-reduce 分解为 reduce-scatter 和 all-gather。具体来说，在反向传播过程中，FSDP 减少并分散梯度，确保每个 rank 拥有梯度的分片。然后在优化器步骤中更新相应参数的分片。最后，在后续的前向传播中，执行 all-gather 操作以收集和组合更新的参数分片。

<img src="https://pytorch.org/tutorials/_images/fsdp_sharding.png" alt="FSDP allreduce" style="zoom:50%;" />

## 如何使用 FSDP

这里我们使用一个简单的模型来演示如何在 MNIST 数据集上进行训练。API 和逻辑也可以应用于训练更大的模型。

### 设置

- 1.1 安装安装PyTorch 和 Torchvision

请参考 [Get Started 指南](https://pytorch.org/get-started/locally/) 获取安装信息。

我们将以下代码片段添加到 Python 脚本“FSDP_mnist.py”中。

- 1.2 导入必要的包

> **注意**: 本教程适用于 PyTorch 1.12 及更高版本。如果您使用的是早期版本，请将所有 `size_based_auto_wrap_policy` 替换为 `default_auto_wrap_policy`，并将 `fsdp_auto_wrap_policy` 替换为 `auto_wrap_policy`。

```python
# 基于: https://github.com/pytorch/examples/blob/master/mnist/main.py
import os
import argparse
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
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
```

1.3 分布式训练设置。

如前所述，FSDP 是一种数据并行方式，需要分布式训练环境，因此我们使用两个辅助函数来初始化分布式训练的进程并清理。

```python
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
```

2.1 定义用于手写数字分类的简单模型。

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
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
```

2.2 定义训练函数

```python
def train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=None):
    model.train()
    ddp_loss = torch.zeros(2).to(rank)
    if sampler:
        sampler.set_epoch(epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(rank), target.to(rank)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target, reduction='sum')
        loss.backward()
        optimizer.step()
        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(data)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, ddp_loss[0] / ddp_loss[1]))
```

2.3 定义验证函数

```python
def test(model, rank, world_size, test_loader):
    model.eval()
    correct = 0
    ddp_loss = torch.zeros(3).to(rank)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(rank), target.to(rank)
            output = model(data)
            ddp_loss[0] += F.nll_loss(output, target, reduction='sum').item()  # 汇总批次损失
            pred = output.argmax(dim=1, keepdim=True)  # 获取最大对数概率的索引
            ddp_loss[1] += pred.eq(target.view_as(pred)).sum().item()
            ddp_loss[2] += len(data)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    if rank == 0:
        test_loss = ddp_loss[0] / ddp_loss[2]
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, int(ddp_loss[1]), int(ddp_loss[2]),
            100. * ddp_loss[1] / ddp_loss[2]))
```

2.4 定义一个分布式训练函数，将模型包装在 FSDP 中

> **注意**: 要保存 FSDP 模型，我们需要在每个 rank 上调用 state_dict，然后在 Rank 0 上保存整体状态。

```python
def fsdp_main(rank, world_size, args):
    setup(rank, world_size)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset1 = datasets.MNIST('../data', train=True, download=True,
                        transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                        transform=transform)

    sampler1 = DistributedSampler(dataset1, rank=rank, num_replicas=world_size, shuffle=True)
    sampler2 = DistributedSampler(dataset2, rank=rank, num_replicas=world_size)

    train_kwargs = {'batch_size': args.batch_size, 'sampler': sampler1}
    test_kwargs = {'batch_size': args.test_batch_size, 'sampler': sampler2}
    cuda_kwargs = {'num_workers': 2,
                    'pin_memory': True,
                    'shuffle': False}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=100
    )
    torch.cuda.set_device(rank)

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    model = Net().to(rank)

    model = FSDP(model)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    init_start_event.record()
    for epoch in range(1, args.epochs + 1):
        train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=sampler1)
        test(model, rank, world_size, test_loader)
        scheduler.step()

    init_end_event.record()

    if rank == 0:
        print(f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
        print(f"{model}")

    if args.save_model:
        # 使用 barrier 确保所有 rank 上的训练已完成
        dist.barrier()
        states = model.state_dict()
        if rank == 0:
            torch.save(states, "mnist_cnn.pt")

    cleanup()
```

2.5 最后，解析参数并设置主函数

```python
if __name__ == '__main__':
    # 训练设置
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='输入批次大小进行训练（默认: 64）')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='输入批次大小进行测试（默认: 1000）')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='训练的 epoch 数量（默认: 14）')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='学习率（默认: 1.0）')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='学习率步长 gamma（默认: 0.7）')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='禁用 CUDA 训练')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='随机种子（默认: 1）')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='保存当前模型')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(fsdp_main,
        args=(WORLD_SIZE, args),
        nprocs=WORLD_SIZE,
        join=True)
```

我们记录了 CUDA 事件以测量 FSDP 模型的时间。CUDA 事件时间为 110.85 秒。

```bash
python FSDP_mnist.py

CUDA event elapsed time on training loop 40.67462890625sec
```

将模型包装在 FSDP 中，模型的结构如下所示，我们可以看到模型已经被包装在一个 FSDP 单元中。或者，我们将在下一步中添加 `auto_wrap_policy`，并讨论它们之间的区别。

```bash
FullyShardedDataParallel(
  (_fsdp_wrapped_module): FlattenParamsWrapper(
    (_fpw_module): Net(
      (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
      (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
      (dropout1): Dropout(p=0.25, inplace=False)
      (dropout2): Dropout(p=0.5, inplace=False)
      (fc1): Linear(in_features=9216, out_features=128, bias=True)
      (fc2): Linear(in_features=128, out_features=10, bias=True)
    )
  )
)
```

以下是 FSDP MNIST 训练在 AWS EC2 g4dn.12xlarge 实例（4 个 GPU）上的峰值内存使用情况，由 PyTorch Profiler 捕获。

![FSDP 峰值内存使用](/_static/img/distributed/FSDP_memory.gif)

应用 `auto_wrap_policy`，否则，FSDP 会将整个模型放在一个 FSDP 单元中，这会降低计算效率和内存效率。其工作原理是，假设您的模型包含 100 个线性层。如果您执行 `FSDP(model)`，将只有一个 FSDP 单元包装整个模型。在这种情况下，allgather 将收集所有 100 个线性层的完整参数，因此不会节省 CUDA 内存用于参数分片。此外，对于所有 100 个线性层，只有一个阻塞的 allgather 调用，因此不会在层之间进行通信和计算的重叠。

为了避免这种情况，您可以传入一个 `auto_wrap_policy`，当满足指定条件（例如大小限制）时，它会自动封装当前的 FSDP 单元并启动一个新的单元。这样，您将拥有多个 FSDP 单元，并且每次只需要一个 FSDP 单元收集完整参数。例如，假设您有 5 个 FSDP 单元，每个单元包装 20 个线性层。然后，在前向传播中，第 1 个 FSDP 单元将 allgather 前 20 个线性层的参数，进行计算，丢弃参数，然后继续处理下一个 20 个线性层。因此，在任何时间点，每个 rank 只实例化 20 个线性层的参数/梯度，而不是 100 个。

在 2.4 中，我们定义了 `auto_wrap_policy` 并将其传递给 FSDP 包装器。在以下示例中，`my_auto_wrap_policy` 定义了一个层如果参数数量大于 100，则可以被 FSDP 封装或分片。如果参数数量小于 100，它将与 FSDP 封装的其他小层一起封装。

找到最佳的 `auto_wrap_policy` 是具有挑战性的，PyTorch 将在未来添加自动调优功能。在没有自动调优工具的情况下，建议通过实验使用不同的 `auto_wrap_policy` 来分析工作流程并找到最佳配置。

```python
my_auto_wrap_policy = functools.partial(
    size_based_auto_wrap_policy, min_num_params=20000
)
torch.cuda.set_device(rank)
model = Net().to(rank)

model = FSDP(model,
    auto_wrap_policy=my_auto_wrap_policy)
```

应用 `auto_wrap_policy` 后，模型的结构如下：

```bash
FullyShardedDataParallel(
  (_fsdp_wrapped_module): FlattenParamsWrapper(
    (_fpw_module): Net(
      (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
      (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
      (dropout1): Dropout(p=0.25, inplace=False)
      (dropout2): Dropout(p=0.5, inplace=False)
      (fc1): FullyShardedDataParallel(
        (_fsdp_wrapped_module): FlattenParamsWrapper(
          (_fpw_module): Linear(in_features=9216, out_features=128, bias=True)
        )
      )
      (fc2): Linear(in_features=128, out_features=10, bias=True)
    )
  )
)
```

```bash
python FSDP_mnist.py

CUDA event elapsed time on training loop 41.89130859375sec
```

以下是 FSDP 使用 `auto_wrap_policy` 的 MNIST 训练在 AWS EC2 g4dn.12xlarge 实例（4 个 GPU）上的峰值内存使用情况，由 PyTorch Profiler 捕获。可以观察到，与未应用 `auto_wrap_policy` 的 FSDP 相比，每个设备的峰值内存使用量更小，从 ~75 MB 减少到 66 MB。

![FSDP 使用 Auto_wrap_policy 的峰值内存使用](/_static/img/distributed/FSDP_autowrap.gif)

## CPU 卸载

如果模型非常大，即使使用 FSDP 也无法适应 GPU，那么 CPU 卸载可能会有所帮助。

目前仅支持参数和梯度的 CPU 卸载。可以通过传入 `cpu_offload=CPUOffload(offload_params=True)` 来启用。

请注意，这目前隐式启用了梯度卸载到 CPU，以便参数和梯度位于同一设备上以与优化器一起工作。此 API 可能会发生变化。默认值为 `None`，在这种情况下不会进行卸载。

使用此功能可能会显著减慢训练速度，因为频繁地在主机和设备之间复制张量，但它可以帮助提高内存效率并训练更大规模的模型。

在 2.4 中，我们只需将其添加到 FSDP 包装器中：

```python
model = FSDP(model,
    auto_wrap_policy=my_auto_wrap_policy,
    cpu_offload=CPUOffload(offload_params=True))
```

## 与 DDP 的比较

如果我们在 2.4 中正常地将模型包装在 DDP 中，并将更改保存到“DDP_mnist.py”中。

```python
model = Net().to(rank)
model = DDP(model)
```

```bash
python DDP_mnist.py

CUDA event elapsed time on training loop 39.77766015625sec
```

以下是 DDP MNIST 训练在 AWS EC2 g4dn.12xlarge 实例（4 个 GPU）上的峰值内存使用情况，由 PyTorch Profiler 捕获。

![DDP 峰值内存使用](/_static/img/distributed/DDP_memory.gif)

考虑到我们在这里定义的简单示例和微小的 MNIST 模型，我们可以观察到 DDP 和 FSDP 之间的峰值内存使用差异。在 DDP 中，每个进程都持有一个模型的副本，因此内存占用比 FSDP 更高，FSDP 将模型参数、优化器状态和梯度分片到 DDP 的各个 rank 中。使用 `auto_wrap_policy` 的 FSDP 的峰值内存使用量最低，其次是 FSDP 和 DDP。

此外，从时间上看，考虑到小模型并在单台机器上运行训练，FSDP 无论是否使用 `auto_wrap_policy` 都几乎与 DDP 一样快。此示例并不代表大多数实际应用，有关 DDP 和 FSDP 的详细分析和比较，请参阅 [此博客文章](https://pytorch.medium.com/6c8da2be180d)。
