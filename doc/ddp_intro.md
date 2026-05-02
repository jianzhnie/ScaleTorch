# 分布式数据并行 (DDP) 入门

## 先决条件

- [PyTorch 分布式概述](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [DistributedDataParallel API 文档](https://pytorch.org/docs/master/generated/torch.nn.parallel.DistributedDataParallel.html)

本文从基本 DDP 用例开始，逐步介绍检查点保存、模型并行结合、torchrun 启动、Rank 概念和日志打印等高级用法。

## 基本用例

### 进程组初始化

DDP 要求先正确设置进程组。单机训练常用 `mp.spawn`，后端可选 `"gloo"`（CPU/GPU 通用）或 `"nccl"`（GPU 训练推荐）：

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


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()
```

### DDP 包装与训练

DDP 构造函数会将 rank 0 的模型状态广播到所有进程，无需手动同步初始参数。梯度同步在反向传播期间自动进行并与计算重叠——`backward()` 返回时 `param.grad` 已包含同步后的梯度。

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
    setup(rank, world_size)
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


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn, args=(world_size,), nprocs=world_size, join=True)
```

## 处理速度不均衡

DDP 的构造函数、前向传播和反向传播是分布式同步点。不同进程必须以相同顺序到达同步点，否则较快进程会等待超时。

应对方法：在 `init_process_group` 中设置足够大的 `timeout` 值，并尽量均衡各进程的工作负载。

## 保存和加载 Checkpoint

推荐**仅在 rank 0 保存**，所有进程再加载——因为所有进程参数相同（初始广播 + 梯度同步），无需重复写入。加载时注意提供 `map_location`，防止进程使用错误的设备。

```python
def demo_checkpoint(rank, world_size):
    setup(rank, world_size)
    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"
    if rank == 0:
        torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)

    # 确保保存完成后再加载
    dist.barrier()
    map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
    ddp_model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=map_location, weights_only=True)
    )

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)
    loss_fn(outputs, labels).backward()
    optimizer.step()

    # DDP 的 AllReduce 已在反向传播中同步，无需额外 barrier
    if rank == 0:
        os.remove(CHECKPOINT_PATH)

    cleanup()
```

更多故障恢复和弹性支持见 [TorchElastic](https://pytorch.org/elastic)。

## 将 DDP 与模型并行结合

DDP 也适用于多 GPU 模型（模型并行）。此时 `device_ids` 和 `output_device` **不要设置**——输入输出由模型 `forward()` 方法自行放置到正确设备。

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


def demo_model_parallel(rank, world_size):
    setup(rank, world_size)
    dev0 = rank * 2
    dev1 = rank * 2 + 1
    mp_model = ToyMpModel(dev0, dev1)
    ddp_mp_model = DDP(mp_model)  # 不设 device_ids

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_mp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_mp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(dev1)
    loss_fn(outputs, labels).backward()
    optimizer.step()

    cleanup()


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    run_demo(demo_basic, n_gpus)
    run_demo(demo_checkpoint, n_gpus)
    run_demo(demo_model_parallel, n_gpus // 2)
```

## mp.spawn vs torchrun

| 维度 | mp.spawn | torchrun |
|------|----------|----------|
| 启动方式 | 代码内 `mp.spawn()` 创建进程 | 命令行工具，自动管理进程 |
| 参数配置 | 手动设置 `world_size`、`rank` | 通过环境变量自动设置 |
| 代码侵入 | 需在 worker 函数中传入 `rank`、`world_size` | 代码直接读取环境变量，更简洁 |
| 灵活性 | 可精细控制进程行为 | 标准化程度高，适用性广 |

**mp.spawn** 示例：

```python
import torch
import torch.multiprocessing as mp
import torch.distributed as dist


def main_worker(gpu, ngpus_per_node, args):
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://localhost:23456",
        world_size=ngpus_per_node,
        rank=gpu,
    )
    model = MyModel()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    # ... 训练逻辑


if __name__ == "__main__":
    ngpus_per_node = torch.cuda.device_count()
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
```

**torchrun** 示例：

```python
import os
import torch
import torch.distributed as dist


def main():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    model = MyModel().to(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    # ... 训练逻辑
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

torchrun 启动：

```bash
# 单机 4 GPU
torchrun --nproc_per_node=4 train.py

# 双机各 8 GPU
torchrun --nnodes=2 --nproc_per_node=8 \
    --rdzv_id=100 --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:29400 train.py
```

SLURM 集群中设置 `MASTER_ADDR`：

```bash
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
srun --nodes=2 ./torchrun_script.sh
```

推荐新项目使用 torchrun。详见 [PyTorch Elastic 快速入门](https://pytorch.org/docs/stable/elastic/quickstart.html)。

## Rank 与 Local_Rank

| 概念 | 含义 | 范围 | 用途 |
|------|------|------|------|
| `rank` | 全局进程编号 | 0 ~ world_size-1 | 进程间通信和同步 |
| `local_rank` | 单节点内进程编号 | 0 ~ local_world_size-1 | GPU 设备分配 |

```python
import os
import torch
import torch.distributed as dist

world_size = int(os.environ.get("WORLD_SIZE", 1))
rank = int(os.environ.get("RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))

dist.init_process_group(backend="nccl")
device = torch.device(f"cuda:{local_rank}")
```

典型场景：

- **单机 4 卡**：`rank` = `local_rank` = {0, 1, 2, 3}，一一对应
- **双机各 4 卡**：节点 0 的 `rank` = {0,1,2,3}，节点 1 的 `rank` = {4,5,6,7}；两节点 `local_rank` 均为 {0,1,2,3}

torchrun 自动设置以上环境变量。

## 分布式训练日志打印

为避免所有进程重复打印，通常仅在 rank 0 输出日志：

```python
import logging
import torch.distributed as dist


class Trainer:
    def __init__(self, model, ...):
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        if self.rank == 0:
            logging.basicConfig(level=logging.INFO, format="%(message)s")
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.disabled = True

    def train(self):
        for epoch in range(1, self.epochs + 1):
            epoch_loss = self.run_epoch(epoch)
            if self.rank == 0:
                self.logger.info(f"Epoch {epoch}, Train Loss: {epoch_loss:.4f}")
                metrics = self.test()
                self.logger.info(f"Epoch {epoch}, Eval Metrics: {metrics}")
```

要点：
- 用 `dist.get_rank()` 获取当前进程 rank
- 仅 rank 0 启用日志记录器，其他进程禁用
- 写文件日志时同样只在 rank 0 操作
