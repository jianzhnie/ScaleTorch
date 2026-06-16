# 使用分布式 RPC 框架实现参数服务器

- [PyTorch 分布式概述](../beginner/dist_overview.html)
- [RPC API 文档](https://pytorch.org/docs/master/rpc.html)

本教程通过一个简单的示例，使用 PyTorch 的 [分布式 RPC 框架](https://pytorch.org/docs/stable/rpc.html) 实现参数服务器。参数服务器框架是一种范式，其中一组服务器存储参数，如大型嵌入表，并且多个训练器查询参数服务器以检索最新的参数。这些训练器可以在本地运行训练循环，并偶尔与参数服务器同步以获取最新的参数。有关参数服务器方法的更多阅读，请查看 [这篇论文](https://www.cs.cmu.edu/~muli/file/parameter_server_osdi14.pdf)。

使用分布式 RPC 框架，我们将构建一个示例，其中多个训练器使用 RPC 与同一个参数服务器通信，并使用 [RRef](https://pytorch.org/docs/stable/rpc.html#torch.distributed.rpc.RRef) 访问远程参数服务器实例上的状态。每个训练器将通过使用分布式 autograd 跨多个节点拼接 autograd 图来以分布式方式启动其专用的反向传播。

**注意**：本教程涵盖了分布式 RPC 框架的使用，这对于将模型拆分到多个机器上或实现参数服务器训练策略（其中网络训练器从不同机器上获取参数）非常有用。如果你正在寻找在许多 GPU 上复制模型的方法，请参阅 [分布式数据并行教程](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)。还有另一个 [RPC 教程](https://pytorch.org/tutorials/intermediate/rpc_tutorial.html) 涵盖了强化学习和 RNN 用例。

让我们从熟悉的开始：导入所需的模块并定义一个将在 MNIST 数据集上训练的简单 ConvNet。下面的网络主要采用了 [pytorch/examples 仓库](https://github.com/pytorch/examples/tree/master/mnist) 中的网络定义。

```python
import argparse
import os
import time
from threading import Lock

import torch
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.distributed.optim import DistributedOptimizer
from torchvision import datasets, transforms

# --------- MNIST 网络定义，来自 pytorch/examples -----

class Net(nn.Module):
    def __init__(self, num_gpus=0):
        super(Net, self).__init__()
        print(f"使用 {num_gpus} 个 GPU 进行训练")
        self.num_gpus = num_gpus
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() and self.num_gpus > 0 else "cpu")
        print(f"将前两个卷积层放在 {str(device)}")
        # 将卷积层放在第一个 cuda 设备上，如果没有 cuda 设备则放在 CPU 上
        self.conv1 = nn.Conv2d(1, 32, 3, 1).to(device)
        self.conv2 = nn.Conv2d(32, 64, 3, 1).to(device)
        # 如果有一个 cuda 设备，将网络的其余部分放在第二个 cuda 设备上
        if "cuda" in str(device) and num_gpus > 1:
            device = torch.device("cuda:1")

        print(f"将剩余的层放在 {str(device)}")
        self.dropout1 = nn.Dropout2d(0.25).to(device)
        self.dropout2 = nn.Dropout2d(0.5).to(device)
        self.fc1 = nn.Linear(9216, 128).to(device)
        self.fc2 = nn.Linear(128, 10).to(device)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)

        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        # 如果需要，将张量移动到下一个设备
        next_device = next(self.fc1.parameters()).device
        x = x.to(next_device)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
```

接下来，让我们定义一些对脚本的其余部分有用的辅助函数。以下使用 [rpc_sync](https://pytorch.org/docs/stable/rpc.html#torch.distributed.rpc.rpc_sync) 和 [RRef](https://pytorch.org/docs/stable/rpc.html#torch.distributed.rpc.RRef) 来定义一个函数，该函数在远程节点上存在的对象上调用给定的方法。下面，我们对远程对象的句柄由 `rref` 参数给出，我们将其运行在其所有者节点上：`rref.owner()`。在调用者节点上，我们通过使用 `rpc_sync` 同步运行此命令，这意味着我们将阻塞直到收到响应。

```python
# --------- 辅助方法 --------------------

# 在本地节点上，使用 RRef 持有的值作为第一个参数调用方法。
# 其他参数作为传递给被调用函数的参数。
# 适用于调用实例方法。method 可以是任何匹配的函数，包括类方法。
def call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)

# 给定一个 RRef，返回调用传递的方法在 RRef 持有的值上的结果。
# 此调用在拥有 RRef 的远程节点上运行，并传递给定的参数。
# 示例：如果 RRef 持有的值是 Foo 类型，那么
# remote_method(Foo.bar, rref, arg1, arg2) 等同于在远程节点上调用
# <foo_instance>.bar(arg1, arg2) 并获取结果。

def remote_method(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc.rpc_sync(rref.owner(), call_method, args=args, kwargs=kwargs)
```

现在，我们准备好定义我们的参数服务器。我们将子类化 `nn.Module` 并保存一个对上面定义的网络的句柄。我们还将保存一个输入设备，该设备将是我们输入在调用模型之前传输到的设备。

```python
# --------- 参数服务器 --------------------
class ParameterServer(nn.Module):
    def __init__(self, num_gpus=0):
        super().__init__()
        model = Net(num_gpus=num_gpus)
        self.model = model
        self.input_device = torch.device(
            "cuda:0" if torch.cuda.is_available() and num_gpus > 0 else "cpu")
```

接下来，我们将定义我们的前向传播。请注意，无论模型输出的设备如何，我们都将输出移动到 CPU，因为分布式 RPC 框架目前仅支持通过 RPC 发送 CPU 张量。我们有意禁用通过 RPC 发送 CUDA 张量，因为调用者/被调用者可能具有不同的设备（CPU/GPU），但我们可能会在未来的版本中支持这一点。

```python
class ParameterServer(nn.Module):
...
    def forward(self, inp):
        inp = inp.to(self.input_device)
        out = self.model(inp)
        # 此输出通过 RPC 转发，截至 1.5.0 版本仅接受 CPU 张量。
        # 由于这个原因，张量必须在 GPU 内存中移动进出。
        out = out.to("cpu")
        return out
```

接下来，我们将定义一些对训练和验证有用的杂项函数。第一个，`get_dist_gradients`，将接收一个分布式 Autograd 上下文 ID，并调用 `dist_autograd.get_gradients` API 以检索由分布式 autograd 计算的梯度。更多信息可以在 [分布式 autograd 文档](https://pytorch.org/docs/stable/rpc.html#distributed-autograd-framework) 中找到。请注意，我们还遍历结果字典并将每个张量转换为 CPU 张量，因为框架目前仅支持通过 RPC 发送张量。接下来，`get_param_rrefs` 将遍历我们的模型参数并将它们包装为（本地）[RRef](https://pytorch.org/docs/stable/rpc.html#torch.distributed.rpc.RRef)。此方法将通过 RPC 由训练器节点调用，并返回要优化的参数列表。这是作为输入传递给 [分布式优化器](https://pytorch.org/docs/stable/rpc.html#module-torch.distributed.optim) 所必需的，它要求所有必须优化的参数作为 `RRef` 列表。

```python
# 使用分布式 autograd 检索为此模型累积的梯度。
# 主要用于验证。
def get_dist_gradients(self, cid):
    grads = dist_autograd.get_gradients(cid)
    # 此输出通过 RPC 转发，截至 1.5.0 版本仅接受 CPU 张量。
    # 由于这个原因，张量必须在 GPU 内存中移动进出。
    cpu_grads = {}
    for k, v in grads.items():
        k_cpu, v_cpu = k.to("cpu"), v.to("cpu")
        cpu_grads[k_cpu] = v_cpu
    return cpu_grads

# 将本地参数包装在 RRef 中。构建分布式优化器时需要，
# 该优化器远程优化参数。
def get_param_rrefs(self):
    param_rrefs = [rpc.RRef(param) for param in self.model.parameters()]
    return param_rrefs
```

最后，我们将创建方法来初始化我们的参数服务器。请注意，所有进程中只有一个参数服务器实例，所有训练器都将与同一个参数服务器通信并更新相同的存储模型。正如在 `run_parameter_server` 中看到的，服务器本身不采取任何独立行动；它等待来自训练器（尚未定义）的请求，并通过运行请求的函数来响应它们。

```python
# 全局参数服务器实例。
param_server = None
# 确保我们只有一个参数服务器。
global_lock = Lock()

def get_parameter_server(num_gpus=0):
    """
    返回一个单例参数服务器给所有训练器进程
    """
    global param_server
    # 确保我们只获取一个 ParameterServer 的句柄。
    with global_lock:
        if not param_server:
            # 构造一次
            param_server = ParameterServer(num_gpus=num_gpus)
        return param_server

def run_parameter_server(rank, world_size):
    # 参数服务器仅充当模型的主机并响应来自训练器的请求。
    # rpc.shutdown() 默认会等待所有工作进程完成，
    # 在本例中意味着参数服务器将等待所有训练器完成，然后退出。
    print("PS 主节点初始化 RPC")
    rpc.init_rpc(name="parameter_server", rank=rank, world_size=world_size)
    print("RPC 初始化！运行参数服务器...")
    rpc.shutdown()
    print("RPC 在参数服务器上关闭。")
```

请注意，上面的 `rpc.shutdown()` 不会立即关闭参数服务器。相反，它将等待所有工作进程（本例中为训练器）也调用 `rpc.shutdown()`。这为我们提供了保证，即参数服务器不会在所有训练器（尚未定义）完成其训练过程之前离线。

接下来，我们将定义我们的 `TrainerNet` 类。这还将是 `nn.Module` 的子类，我们的 `__init__` 方法将使用 `rpc.remote` API 获取一个 RRef，或远程引用，到我们的参数服务器。请注意，这里我们没有将参数服务器复制到本地进程，相反，我们可以将 `self.param_server_rref` 视为一个分布式共享指针，指向在单独进程上运行的参数服务器。

```python
# --------- 训练器 --------------------

# 对应于此训练器训练的网络的 nn.Module。
# forward() 方法仅在给定的参数服务器上调用网络。
class TrainerNet(nn.Module):
    def __init__(self, num_gpus=0):
        super().__init__()
        self.num_gpus = num_gpus
        self.param_server_rref = rpc.remote(
            "parameter_server", get_parameter_server, args=(num_gpus,))
```

接下来，我们将定义一个名为 `get_global_param_rrefs` 的方法。为了激发这个方法的需求，值得阅读 [分布式优化器](https://pytorch.org/docs/stable/rpc.html#module-torch.distributed.optim) 的文档，特别是 API 签名。优化器必须传递一个 `RRef` 列表，对应于要优化的远程参数，因此在这里我们获取必要的 `RRef`。由于给定的 `TrainerNet` 与之交互的唯一远程工作进程是 `ParameterServer`，我们只需在 `ParameterServer` 上调用 `remote_method`。我们使用在 `ParameterServer` 类中定义的 `get_param_rrefs` 方法。此方法将返回一个 `RRef` 列表，其中包含需要优化的参数。请注意，在这种情况下，我们的 `TrainerNet` 没有定义自己的参数；如果有，我们还需要将每个参数包装在 `RRef` 中，并将其包含在我们的 `DistributedOptimizer` 输入中。

```python
class TrainerNet(nn.Module):
...
    def get_global_param_rrefs(self):
        remote_params = remote_method(
            ParameterServer.get_param_rrefs,
            self.param_server_rref)
        return remote_params
```

现在，我们准备好定义我们的 `forward` 方法，它将调用（同步）RPC 在 `ParameterServer` 上定义的网络上运行前向传播。请注意，我们将 `self.param_server_rref` 传递给我们的 RPC 调用，这是一个对我们的 `ParameterServer` 的远程句柄。此调用将向运行我们 `ParameterServer` 的节点发送一个 RPC，调用前向传播，并返回与模型输出对应的 `Tensor`。

```python
class TrainerNet(nn.Module):
...
    def forward(self, x):
        model_output = remote_method(
            ParameterServer.forward, self.param_server_rref, x)
        return model_output
```

随着我们的训练器完全定义，现在是时候编写我们的神经网络训练循环了，它将创建我们的网络和优化器，通过网络运行一些输入并计算损失。训练循环看起来很像本地训练程序的训练循环，但由于我们的网络分布在机器之间，因此有一些修改。

下面，我们初始化我们的 `TrainerNet` 并构建一个 `DistributedOptimizer`。请注意，如上所述，我们必须传递所有参与分布式训练的节点的全局参数，这些参数是我们想要优化的。此外，我们传递要使用的本地优化器，在本例中为 SGD。请注意，我们可以像创建本地优化器一样配置底层优化器算法 - 所有 `optimizer.SGD` 的参数都将被正确转发。作为一个示例，我们传递一个自定义学习率，该学习率将用作所有本地优化器的学习率。

```python
def run_training_loop(rank, num_gpus, train_loader, test_loader):
    # 以分布式方式运行典型的神经网络前向 + 反向 + 优化器步骤。
    net = TrainerNet(num_gpus=num_gpus)
    # 构建分布式优化器。
    param_rrefs = net.get_global_param_rrefs()
    opt = DistributedOptimizer(optim.SGD, param_rrefs, lr=0.03)
```

接下来，我们定义我们的主训练循环。我们遍历 PyTorch 的 [DataLoader](https://pytorch.org/docs/stable/data.html) 给出的可迭代对象。在编写典型的前向/反向/优化器循环之前，我们首先将逻辑包装在 [分布式 Autograd 上下文](https://pytorch.org/docs/stable/rpc.html#torch.distributed.autograd.context) 中。请注意，这是必需的，以记录模型前向传播中调用的 RPC，以便可以构建一个包含所有参与分布式工作进程的反向传播的适当图。分布式 autograd 上下文返回一个 `context_id`，作为标识符，用于累积和优化与特定迭代对应的梯度。

与调用典型的 `loss.backward()` 启动本地工作进程上的反向传播不同，我们调用 `dist_autograd.backward()` 并传递我们的 `context_id` 和 `loss`，这是我们希望反向传播开始的根。此外，我们将此 `context_id` 传递到我们的优化器调用中，这是必需的，以便能够在所有节点上查找由该特定反向传播计算的相应梯度。

```python
def run_training_loop(rank, num_gpus, train_loader, test_loader):
...
    for i, (data, target) in enumerate(train_loader):
        with dist_autograd.context() as cid:
            model_output = net(data)
            target = target.to(model_output.device)
            loss = F.nll_loss(model_output, target)
            if i % 5 == 0:
                print(f"Rank {rank} 训练批次 {i} 损失 {loss.item()}")
            dist_autograd.backward(cid, [loss])
            # 确保分布式 autograd 成功运行并且梯度已返回。
            assert remote_method(
                ParameterServer.get_dist_gradients,
                net.param_server_rref,
                cid) != {}
            opt.step(cid)

    print("训练完成！")
    print("获取准确率....")
    get_accuracy(test_loader, net)
```

以下内容仅在我们完成训练后计算模型的准确率，类似于传统的本地模型。但是，请注意，我们传递给此函数的 `net` 是 `TrainerNet` 的一个实例，因此前向传播以透明的方式调用 RPC。

```python
def get_accuracy(test_loader, model):
    model.eval()
    correct_sum = 0
    # 如果可能，使用 GPU 进行评估
    device = torch.device("cuda:0" if model.num_gpus > 0
        and torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            out = model(data, -1)
            pred = out.argmax(dim=1, keepdim=True)
            pred, target = pred.to(device), target.to(device)
            correct = pred.eq(target.view_as(pred)).sum().item()
            correct_sum += correct

    print(f"准确率 {correct_sum / len(test_loader.dataset)}")
```

接下来，类似于我们定义 `run_parameter_server` 作为我们的 `ParameterServer` 的主循环，负责初始化 RPC，让我们为我们的训练器定义一个类似的循环。不同之处在于，我们的训练器必须运行上面定义的训练循环：

```python
# 训练器的主循环。
def run_worker(rank, world_size, num_gpus, train_loader, test_loader):
    print(f"Worker rank {rank} 初始化 RPC")
    rpc.init_rpc(
        name=f"trainer_{rank}",
        rank=rank,
        world_size=world_size)

    print(f"Worker {rank} 完成初始化 RPC")

    run_training_loop(rank, num_gpus, train_loader, test_loader)
    rpc.shutdown()
```

请注意，类似于 `run_parameter_server`，`rpc.shutdown()` 默认会等待所有工作进程，包括训练器和参数服务器，调用 `rpc.shutdown()` 后此节点才会退出。这确保节点以有序的方式终止，并且没有节点在另一个节点期望其在线时离线。

我们已经完成了训练器和参数服务器特定的代码，剩下的就是添加代码来启动训练器和参数服务器。首先，我们必须接受适用于我们的参数服务器和训练器的各种参数。`world_size` 对应于将参与训练的总节点数，并且是所有训练器和参数服务器的总和。我们还必须为每个单独进程传递一个唯一的 `rank`，从 0（我们将运行单个参数服务器的地方）到 `world_size - 1`。`master_addr` 和 `master_port` 是可用于标识运行 rank 0 进程的位置的参数，并将由各个节点用于相互发现。要在本地测试此示例，只需传递 `localhost` 和所有实例启动时相同的 `master_port`。请注意，出于演示目的，此示例仅支持 0-2 个 GPU，尽管该模式可以扩展以利用更多 GPU。

```python
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="基于 RPC 的参数服务器训练")
    parser.add_argument(
        "--world_size",
        type=int,
        default=4,
        help="""参与进程总数。应该是主节点和所有训练节点的总和。""")
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="此进程的全局排名。传递 0 表示主节点。")
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=0,
        help="""用于训练的 GPU 数量，目前支持 0 到 2 个 GPU。
        请注意，此参数将传递给参数服务器。""")
    parser.add_argument(
        "--master_addr",
        type=str,
        default="localhost",
        help="""主节点的地址，如果未提供，将默认为 localhost。
        主节点必须能够在地址 + 端口上接受网络流量。""")
    parser.add_argument(
        "--master_port",
        type=str,
        default="29500",
        help="""主节点监听的端口，如果未提供，将默认为 29500。
        主节点必须能够在主机和端口上接受网络流量。""")

    args = parser.parse_args()
    assert args.rank is not None, "必须提供 rank 参数。"
    assert args.num_gpus <= 3, f"目前仅支持 0-2 个 GPU（得到 {args.num_gpus}）。"
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port
```

现在，我们将创建一个对应于参数服务器或训练器的进程，具体取决于我们的命令行参数。如果传递的 rank 为 0，我们将创建一个 `ParameterServer`，否则创建一个 `TrainerNet`。请注意，我们使用 `torch.multiprocessing` 来启动对应于我们想要执行的函数的子进程，并使用 `p.join()` 从主线程等待此进程的完成。在初始化我们的训练器的情况下，我们还使用 PyTorch 的 [dataloaders](https://pytorch.org/docs/stable/data.html) 来指定 MNIST 数据集上的训练和测试数据加载器。

```python
processes = []
world_size = args.world_size
if args.rank == 0:
    p = mp.Process(target=run_parameter_server, args=(0, world_size))
    p.start()
    processes.append(p)
else:
    # 获取训练数据
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=32, shuffle=True,)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            '../data',
            train=False,
            transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                        ])),
        batch_size=32,
        shuffle=True,
    )
    # 在此节点上启动训练器
    p = mp.Process(
        target=run_worker,
        args=(
            args.rank,
            world_size, args.num_gpus,
            train_loader,
            test_loader))
    p.start()
    processes.append(p)

for p in processes:
    p.join()
```

要在本地运行示例，请在单独的终端窗口中为服务器和每个要启动的工作进程运行以下命令：`python rpc_parameter_server.py --world_size=WORLD_SIZE --rank=RANK`。例如，对于 world size 为 2 的主节点，命令将是 `python rpc_parameter_server.py --world_size=2 --rank=0`。训练器可以在单独的窗口中使用命令 `python rpc_parameter_server.py --world_size=2 --rank=1` 启动，这将开始使用一个服务器和一个训练器进行训练。请注意，本教程假设训练使用 0 到 2 个 GPU，可以通过将 `--num_gpus=N` 传递到训练脚本来配置此参数。

你可以传递命令行参数 `--master_addr=ADDRESS` 和 `--master_port=PORT` 来指示主工作进程正在监听的地址和端口，例如，测试训练器和主节点在不同机器上运行的功能。
