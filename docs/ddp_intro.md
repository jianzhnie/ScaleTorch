# 分布式数据并行入门

## 先决条件

- [PyTorch 分布式概述](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [DistributedDataParallel API 文档](https://pytorch.org/docs/master/generated/torch.nn.parallel.DistributedDataParallel.html)

在本教程中，我们将从一个基本的 DDP 用例开始，然后演示更多高级用例，包括模型检查点和将 DDP 与模型并行结合。

**注意**: 本教程中的代码在一个 8-GPU 服务器上运行，但可以轻松推广到其他环境。

## 基本用例

要创建一个 DDP 模块，你必须首先正确设置进程组。更多细节可以在 [使用 PyTorch 编写分布式应用程序](https://pytorch.org/tutorials/intermediate/dist_tuto.html) 中找到。

```python
import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

# 在 Windows 平台上，torch.distributed 包仅支持 Gloo 后端、FileStore 和 TcpStore。
# 对于 FileStore，在 init_process_group 中设置 init_method 参数为本地文件。示例如下：
# init_method="file:///f:/libtmp/some_file"
# dist.init_process_group(
#    "gloo",
#    rank=rank,
#    init_method=init_method,
#    world_size=world_size)
# 对于 TcpStore，与在 Linux 上的方式相同。

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # 初始化进程组
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
```

现在，让我们创建一个 Toy 模块，用 DDP 包装它，并给它一些虚拟输入数据。请注意，由于 DDP 在构造函数中将模型状态从 rank 0 进程广播到所有其他进程，因此你不需要担心不同的 DDP 进程从不同的初始模型参数值开始。

```python
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # 创建模型并将其移动到带有 id rank 的 GPU 上
    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)
    loss_fn(outputs, labels).backward()
    optimizer.step()

    cleanup()
    print(f"Finished running basic DDP example on rank {rank}.")

def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)
```

正如你所见，DDP 包装了较低级别的分布式通信细节，并提供了一个干净的 API，就像它是一个本地模型一样。梯度同步通信在反向传播期间进行，并与反向计算重叠。当 `backward()` 返回时，`param.grad` 已经包含了同步的梯度张量。对于基本用例，DDP 只需要几行额外的代码来设置进程组。当将 DDP 应用于更高级的用例时，一些注意事项需要谨慎。

## 处理速度不均衡

在 DDP 中，构造函数、前向传播和反向传播是分布式同步点。不同的进程预计会启动相同数量的同步，并以相同的顺序到达这些同步点，并且大致同时进入每个同步点。否则，较快的进程可能会提前到达并在等待掉队者时超时。因此，用户负责平衡跨进程的工作负载分布。有时，由于网络延迟、资源争用或不可预测的工作负载峰值等原因，处理速度不均衡是不可避免的。为了避免在这些情况下超时，请确保在调用 [init_process_group](https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group) 时传递一个足够大的 `timeout` 值。

## 保存和加载 `checkppoint`

在训练期间使用 `torch.save` 和 `torch.load` 检查点模块并从中恢复是很常见的。有关更多详细信息，请参阅 [保存和加载模型](https://pytorch.org/tutorials/beginner/saving_loading_models.html)。在使用 DDP 时，一个优化是在一个进程中保存模型，然后在所有进程中加载它，以减少写入开销。这之所以有效，是因为所有进程都从相同的参数开始，并且在反向传播中同步梯度，因此优化器应该保持将参数设置为相同的值。如果你使用这种优化（即在一个进程中保存但在所有进程中恢复），请确保在保存完成之前没有进程开始加载。此外，在加载模块时，你需要提供一个适当的 `map_location` 参数，以防止进程进入其他设备。如果缺少 `map_location`，`torch.load` 会首先将模块加载到 CPU，然后将每个参数复制到保存它的地方，这会导致同一台机器上的所有进程使用相同的一组设备。有关更高级的故障恢复和弹性支持，请参阅 [TorchElastic](https://pytorch.org/elastic)。

```python
def demo_checkpoint(rank, world_size):
    print(f"Running DDP checkpoint example on rank {rank}.")
    setup(rank, world_size)

    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"
    if rank == 0:
        # 所有进程应该看到相同的参数，因为它们都从相同的随机参数开始，并且在反向传播中同步梯度。
        # 因此，在一个进程中保存它是足够的。
        torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)

    # 使用 barrier() 确保进程 1 在进程 0 保存模型后加载模型。
    dist.barrier()
    # 配置 map_location 正确
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    ddp_model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=map_location, weights_only=True))

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)

    loss_fn(outputs, labels).backward()
    optimizer.step()

    # 不需要使用 dist.barrier() 来保护下面的文件删除操作
    # 因为 DDP 中的 AllReduce 操作已经在反向传播中起到了同步作用。

    if rank == 0:
        os.remove(CHECKPOINT_PATH)

    cleanup()
    print(f"Finished running DDP checkpoint example on rank {rank}.")
```

## 将 DDP 与模型并行结合

DDP 也适用于多 GPU 模型。当训练具有大量数据的庞大模型时，DDP 包装多 GPU 模型尤其有用。

```python
class ToyMpModel(nn.Module):
    def __init__(self, dev0, dev1):
        super(ToyMpModel, self).__init__()
        self.dev0 = dev0
        self.dev1 = dev1
        self.net1 = torch.nn.Linear(10, 10).to(dev0)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5).to(dev1)

    def forward(self, x):
        x = x.to(self.dev0)
        x = self.relu(self.net1(x))
        x = x.to(self.dev1)
        return self.net2(x)
```

当将多 GPU 模型传递给 DDP 时，`device_ids` 和 `output_device` 必须**不**设置。输入和输出数据将由应用程序或模型 `forward()` 方法放置在适当的设备上。

```python
def demo_model_parallel(rank, world_size):
    print(f"Running DDP with model parallel example on rank {rank}.")
    setup(rank, world_size)

    # 为这个进程设置 mp_model 和设备
    dev0 = rank * 2
    dev1 = rank * 2 + 1
    mp_model = ToyMpModel(dev0, dev1)
    ddp_mp_model = DDP(mp_model)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_mp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    # outputs 将在 dev1 上
    outputs = ddp_mp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(dev1)
    loss_fn(outputs, labels).backward()
    optimizer.step()

    cleanup()
    print(f"Finished running DDP with model parallel example on rank {rank}.")

if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    run_demo(demo_basic, world_size)
    run_demo(demo_checkpoint, world_size)
    world_size = n_gpus//2
    run_demo(demo_model_parallel, world_size)
```

## 使用 torchrun 初始化 DDP

我们可以利用 PyTorch Elastic 来简化 DDP 代码并更轻松地初始化作业。让我们仍然使用 Toymodel 示例并创建一个名为 `elastic_ddp.py` 的文件。

```python
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

def demo_basic():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(f"Start running basic DDP example on rank {rank}.")
    # 创建模型并将其移动到带有 id rank 的 GPU 上
    device_id = rank % torch.cuda.device_count()
    model = ToyModel().to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(device_id)
    loss_fn(outputs, labels).backward()
    optimizer.step()
    dist.destroy_process_group()
    print(f"Finished running basic DDP example on rank {rank}.")

if __name__ == "__main__":
    demo_basic()
```

然后可以在所有节点上运行 `torch elastic/torchrun` 命令来初始化上面创建的 DDP 作业：

```bash
torchrun --nnodes=2 --nproc_per_node=8 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29400 elastic_ddp.py
```

在上面的示例中，我们在两台主机上运行 DDP 脚本，每台主机上运行 8 个进程。也就是说，我们在 16 个 GPU 上运行这个作业。请注意，`$MASTER_ADDR` 必须在所有节点上相同。

这里 `torchrun` 将在启动它的节点上生成 8 个进程，并在每个进程上调用 `elastic_ddp.py`，但用户还需要应用集群管理工具（如 slurm）在 2 个节点上实际运行此命令。

例如，在启用了 SLURM 的集群上，我们可以编写一个脚本来运行上面的命令并设置 `MASTER_ADDR` 为：

```bash
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
```

然后我们可以使用 SLURM 命令 `srun --nodes=2 ./torchrun_script.sh` 运行这个脚本。

这只是一个示例；你可以选择自己的集群调度工具来启动 `torchrun` 作业。

有关 Elastic run 的更多信息，请参阅 [快速入门文档](https://pytorch.org/docs/stable/elastic/quickstart.html)。

## 分布式程序启动脚本

### mp.spawn 对比 torchrun

Multiprocessing-distributed 和Torchrun 的主要差异包括：

1. 启动方式

- Multiprocessing-distributed：需要手动编写多进程启动代码，通常使用 `torch.multiprocessing.spawn()` 创建多个进程
- Torchrun：是 PyTorch 官方提供的启动工具，可以直接通过命令行启动分布式训练，更加简洁

2. 进程管理

- Multiprocessing-distributed：需要自己管理进程的创建、通信和同步
- Torchrun：自动管理进程，处理进程的创建、通信和资源分配

3. 配置复杂度

- Multiprocessing-distributed：配置较为复杂，需要手动设置 `world_size`、`rank` 等参数
- Torchrun：配置更加简单，通过环境变量自动设置分布式训练参数

4. 代码侵入性

- Multiprocessing-distributed：需要在代码中添加大量分布式训练相关的初始化和同步逻辑
- Torchrun：对原始训练代码的侵入性较低，只需少量修改

5. 灵活性

- Multiprocessing-distributed：更加灵活，可以精细控制进程行为
- Torchrun：标准化程度高，适用性更广

下面我给你展示两种方式的简单代码示例：

### mp.spawn 示例：

```python
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

def main_worker(gpu, ngpus_per_node, args):
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://localhost:23456',
        world_size=ngpus_per_node,
        rank=gpu
    )

    model = MyModel()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

def main():
    ngpus_per_node = torch.cuda.device_count()
    mp.spawn(main_worker,
             nprocs=ngpus_per_node,
             args=(ngpus_per_node, args))

if __name__ == '__main__':
    main()
```

### Torchrun 示例：

```python
import torch
import torch.distributed as dist

def main():
    dist.init_process_group(backend='nccl')

    model = MyModel()
    model = torch.nn.parallel.DistributedDataParallel(model)

if __name__ == '__main__':
    main()
```

启动命令：

- Multiprocessing-distributed：`python train.py`
- Torchrun：`torchrun --nproc_per_node=4 train.py`

总的来说，torchrun 更加现代和便捷，推荐在新项目中使用。对于需要精细控制的场景，multiprocessing-distributed 仍然是一个好选择。

## Rank 和 Locl_Rank

在分布式训练中，`local_rank` 和 `rank` 是两个不同但都很重要的概念：

1. `rank`（全局进程编号）

- 表示在所有分布式训练进程中的全局唯一标识符
- 范围是 0 到 (world_size - 1)
- 在整个分布式训练集群中唯一标识一个进程
- 用于进程间通信和同步

2. `local_rank`（本地进程编号）

- 表示在单个机器/节点上的本地进程编号
- 通常用于选择当前进程使用的 GPU 设备
- 范围是 0 到 (local_world_size - 1)
- 主要用于设备分配

设置示例：

```python
import os
import torch.distributed as dist

# 通过环境变量获取
world_size = int(os.environ.get('WORLD_SIZE', 1))  # 总的进程数
rank = int(os.environ.get('RANK', 0))  # 全局进程编号
local_rank = int(os.environ.get('LOCAL_RANK', 0))  # 本地进程编号

# 初始化分布式环境
dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)

# 选择 GPU
device = torch.device(f'cuda:{local_rank}')
```

典型场景：

- 单机4卡机器：

  - `world_size` = 4
  - `rank` 可能是 0, 1, 2, 3
  - `local_rank` 也是 0, 1, 2, 3
  - 每个进程使用对应编号的 GPU

- 2台机器各4卡：

  - 第一台机器：`rank` 0-3, `local_rank` 0-3
  - 第二台机器：`rank` 4-7, `local_rank` 0-3
  - 确保每台机器的 `local_rank` 从 0 开始

Torchrun 启动命令会自动设置这些环境变量：

```bash
torchrun --nproc_per_node=4 train.py
```

这样可以方便地实现跨机器的分布式训练。

## 分布式训练中日志打印

在分布式训练中，为了避免日志信息重复和混乱，通常只在 rank 0 进程上打印日志。这是因为 rank 0 代表主进程，负责协调和汇总信息。

以下是在 `Trainer` 类中设置日志打印的推荐方法：

```python
import logging
import torch.distributed as dist

class Trainer:
    def __init__(
        self,
        args: argparse.Namespace,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        device: Optional[torch.device] = None,
    ) -> None:
        # 获取当前进程的 rank
        self.rank = dist.get_rank() if dist.is_initialized() else 0

        # 仅在 rank 0 进程配置日志
        if self.rank == 0:
            logging.basicConfig(level=logging.INFO, format='%(message)s')
            self.logger = logging.getLogger(__name__)
        else:
            # 其他进程使用空日志记录器
            self.logger = logging.getLogger(__name__)
            self.logger.disabled = True

    def train(self) -> None:
        for epoch in range(1, self.args.epochs + 1):
            # 训练逻辑
            epoch_loss = self.run_epoch(epoch)

            # 仅在 rank 0 进程记录日志
            if self.rank == 0:
                self.logger.info(f'Epoch {epoch}, Train Loss: {epoch_loss:.4f}')
                test_metrics = self.test()
                self.logger.info(f'Epoch {epoch}, Eval Metrics: {test_metrics}')

    def test(self) -> Dict[str, float]:
        # 测试逻辑

        # 同步测试结果
        test_loss = ...
        correct = ...

        # 仅在 rank 0 进程打印日志
        if self.rank == 0:
            self.logger.info(
                '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.
                format(test_loss, correct, total_samples, accuracy))

        return {'loss': test_loss, 'accuracy': accuracy}
```

关键点：

1. 使用 `dist.get_rank()` 获取当前进程的 rank
2. 仅在 rank 0 进程初始化和启用日志记录器
3. 在需要打印日志的方法中，使用 `if self.rank == 0:` 条件
4. 其他进程的日志记录器被禁用

这种方法确保：

- 只有一个进程（rank 0）打印日志
- 避免日志信息重复
- 集中显示训练和评估的关键信息

如果需要更复杂的日志记录（如写入文件），可以在 rank 0 进程中配置文件日志记录器。
