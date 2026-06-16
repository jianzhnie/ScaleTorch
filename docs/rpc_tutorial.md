# 分布式RPC框架入门

## 先决条件

- [PyTorch分布式概述](../beginner/dist_overview.html)
- [RPC API文档](https://pytorch.org/docs/master/rpc.html)

本教程使用两个简单的示例来演示如何使用`torch.distributed.rpc`包构建分布式训练。该包最初在PyTorch v1.4中作为实验性功能引入。这两个示例的源代码可以在[PyTorch示例](https://github.com/pytorch/examples)中找到。

之前的教程，[使用分布式数据并行入门](ddp_tutorial.html)和[使用PyTorch编写分布式应用](dist_tuto.html)，描述了`DistributedDataParallel`，它支持一种特定的训练范式，即在多个进程中复制模型，每个进程处理输入数据的一部分。有时，你可能会遇到需要不同训练范式的场景。例如：

1. 在强化学习中，获取环境训练数据可能相对昂贵，而模型本身可能相当小。在这种情况下，生成多个并行运行的观察者并共享单个代理可能很有用。在这种情况下，代理在本地处理训练，但应用程序仍然需要在观察者和训练器之间发送和接收数据的库。

2. 你的模型可能太大，无法在单台机器的GPU上容纳，因此需要一个库来帮助将模型拆分到多台机器上。或者你可能正在实现一个[参数服务器](https://www.cs.cmu.edu/~muli/file/parameter_server_osdi14.pdf)训练框架，其中模型参数和训练器位于不同的机器上。

`torch.distributed.rpc`包可以帮助解决上述场景。

- 在情况1中，[RPC](https://pytorch.org/docs/stable/rpc.html#rpc)和[RRef](https://pytorch.org/docs/stable/rpc.html#rref)允许从一个工作进程向另一个工作进程发送数据，同时轻松引用远程数据对象。
- 在情况2中，[分布式自动梯度](https://pytorch.org/docs/stable/rpc.html#distributed-autograd-framework)和[分布式优化器](https://pytorch.org/docs/stable/rpc.html#module-torch.distributed.optim)使执行反向传播和优化器步骤就像是本地训练一样。

在接下来的两节中，我们将通过强化学习示例和语言模型示例来演示`torch.distributed.rpc`的API。请注意，本教程的目标不是构建最准确或最高效的模型来解决给定的问题，而是展示如何使用`torch.distributed.rpc`包构建分布式训练应用程序。

## 使用RPC和RRef的分布式强化学习

本节描述了使用RPC解决OpenAI Gym的CartPole-v1问题的分布式强化学习模型的步骤。策略代码大多借鉴了现有的单线程[示例](https://github.com/pytorch/examples/blob/master/reinforcement_learning)。我们将跳过`Policy`设计的细节，专注于RPC的使用。

```python
import torch.nn as nn
import torch.nn.functional as F

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)
```

我们准备介绍观察者。在这个示例中，每个观察者创建自己的环境，并等待代理的命令来运行一个回合。在每个回合中，一个观察者最多循环`n_steps`次迭代，在每次迭代中，它使用RPC将其环境状态传递给代理并获取一个动作。然后，它将该动作应用于其环境，并从环境获取奖励和下一个状态。之后，观察者使用另一个RPC向代理报告奖励。

请注意，这显然不是最有效的观察者实现。例如，一个简单的优化可能是将当前状态和上一个奖励打包到一个RPC中，以减少通信开销。然而，目标是演示RPC API，而不是为CartPole构建最佳求解器。因此，让我们保持逻辑简单，步骤明确。

```python
import argparse
import gym
import torch.distributed.rpc as rpc

parser = argparse.ArgumentParser(
    description="RPC Reinforcement Learning Example",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument('--world_size', default=2, type=int, metavar='W',
                    help='number of workers')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='interval between training status logs')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='how much to value future rewards')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed  for reproducibility')
args = parser.parse_args()

class Observer:
    def __init__(self):
        self.id = rpc.get_worker_info().id
        self.env = gym.make('CartPole-v1')
        self.env.seed(args.seed)

    def run_episode(self, agent_rref):
        state, ep_reward = self.env.reset(), 0
        for _ in range(10000):
            # send the state to the agent to get an action
            action = agent_rref.rpc_sync().select_action(self.id, state)

            # apply the action to the environment, and get the reward
            state, reward, done, _ = self.env.step(action)

            # report the reward to the agent for training purpose
            agent_rref.rpc_sync().report_reward(self.id, reward)

            # finishes after the number of self.env._max_episode_steps
            if done:
                break
```

代理的代码稍微复杂一些，我们将分几个部分来解释。在这个示例中，代理同时作为训练器和主控端，它向多个分布式观察者发送命令运行回合，并在本地记录所有动作和奖励，这些将在每个回合后的训练阶段使用。

下面的代码展示了`Agent`的构造函数，大部分代码行都在初始化各种组件。最后的循环在远程其他工作进程上初始化观察者，并在本地保存这些观察者的`RRefs`。代理稍后将使用这些观察者`RRefs`发送命令。应用程序不需要担心`RRefs`的生命周期。每个`RRef`的所有者维护一个引用计数映射来跟踪其生命周期，并保证只要有任何活跃用户，远程数据对象就不会被删除。请参考`RRef`的[设计文档](https://pytorch.org/docs/master/notes/rref.html)了解详情。

```python
import gym
import numpy as np

import torch
import torch.distributed.rpc as rpc
import torch.optim as optim
from torch.distributed.rpc import RRef, rpc_async, remote
from torch.distributions import Categorical

class Agent:
    def __init__(self, world_size):
        self.ob_rrefs = []
        self.agent_rref = RRef(self)
        self.rewards = {}
        self.saved_log_probs = {}
        self.policy = Policy()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-2)
        self.eps = np.finfo(np.float32).eps.item()
        self.running_reward = 0
        self.reward_threshold = gym.make('CartPole-v1').spec.reward_threshold
        for ob_rank in range(1, world_size):
            ob_info = rpc.get_worker_info(OBSERVER_NAME.format(ob_rank))
            self.ob_rrefs.append(remote(ob_info, Observer))
            self.rewards[ob_info.id] = []
            self.saved_log_probs[ob_info.id] = []
```

接下来，代理公开两个API供观察者选择动作和报告奖励。这些函数仅在代理本地运行，但将通过RPC由观察者触发。

```python
class Agent:
    ...
    def select_action(self, ob_id, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs[ob_id].append(m.log_prob(action))
        return action.item()

    def report_reward(self, ob_id, reward):
        self.rewards[ob_id].append(reward)
```

让我们添加一个`run_episode`函数到代理中，告诉所有观察者执行一个回合。在这个函数中，它首先创建一个列表来收集异步RPC的future，然后遍历所有观察者`RRefs`进行异步RPC。在这些RPC中，代理还传递了自身的`RRef`给观察者，以便观察者也可以调用代理的函数。如上所示，每个观察者都会向代理发起RPC，这些是嵌套的RPC。每个回合后，`saved_log_probs`和`rewards`将包含记录的动作概率和奖励。

```python
class Agent:
    ...
    def run_episode(self):
        futs = []
        for ob_rref in self.ob_rrefs:
            # make async RPC to kick off an episode on all observers
            futs.append(
                rpc_async(
                    ob_rref.owner(),
                    ob_rref.rpc_sync().run_episode,
                    args=(self.agent_rref,)
                )
            )

        # wait until all obervers have finished this episode
        for fut in futs:
            fut.wait()
```

最后，在一个回合结束后，代理需要训练模型，这在下面的`finish_episode`函数中实现。这个函数中没有RPC，主要借鉴了单线程[示例](https://github.com/pytorch/examples/blob/master/reinforcement_learning)。因此，我们跳过描述其内容。

```python
class Agent:
    ...
    def finish_episode(self):
        # joins probs and rewards from different observers into lists
        R, probs, rewards = 0, [], []
        for ob_id in self.rewards:
            probs.extend(self.saved_log_probs[ob_id])
            rewards.extend(self.rewards[ob_id])

        # use the minimum observer reward to calculate the running reward
        min_reward = min([sum(self.rewards[ob_id]) for ob_id in self.rewards])
        self.running_reward = 0.05 * min_reward + (1 - 0.05) * self.running_reward

        # clear saved probs and rewards
        for ob_id in self.rewards:
            self.rewards[ob_id] = []
            self.saved_log_probs[ob_id] = []

        policy_loss, returns = [], []
        for r in rewards[::-1]:
            R = r + args.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        for log_prob, R in zip(probs, returns):
            policy_loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        return min_reward
```

准备好了`Policy`、`Observer`和`Agent`类后，我们就可以启动多个进程来执行分布式训练。在这个示例中，所有进程运行相同的`run_worker`函数，并使用等级来区分其角色。等级0始终是代理，其他所有等级都是观察者。代理作为主控端，通过反复调用`run_episode`和`finish_episode`直到运行奖励超过环境指定的奖励阈值。所有观察者被动地等待来自代理的命令。代码被`rpc.init_rpc`和`rpc.shutdown`包裹，分别初始化和终止RPC实例。更多细节可在[API页面](https://pytorch.org/docs/stable/rpc.html)中找到。

```python
import os
from itertools import count

import torch.multiprocessing as mp

AGENT_NAME = "agent"
OBSERVER_NAME="obs{}"

def run_worker(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    if rank == 0:
        # rank0 is the agent
        rpc.init_rpc(AGENT_NAME, rank=rank, world_size=world_size)

        agent = Agent(world_size)
        print(f"This will run until reward threshold of {agent.reward_threshold}"
                " is reached. Ctrl+C to exit.")
        for i_episode in count(1):
            agent.run_episode()
            last_reward = agent.finish_episode()

            if i_episode % args.log_interval == 0:
                print(f"Episode {i_episode}\tLast reward: {last_reward:.2f}\tAverage reward: "
                    f"{agent.running_reward:.2f}")
            if agent.running_reward > agent.reward_threshold:
                print(f"Solved! Running reward is now {agent.running_reward}!")
                break
    else:
        # other ranks are the observer
        rpc.init_rpc(OBSERVER_NAME.format(rank), rank=rank, world_size=world_size)
        # observers passively waiting for instructions from the agent

    # block until all rpcs finish, and shutdown the RPC instance
    rpc.shutdown()


mp.spawn(
    run_worker,
    args=(args.world_size, ),
    nprocs=args.world_size,
    join=True
)
```

#### 训练输出示例

下面是使用 `world_size=2` 训练时的一些示例输出。

这将运行直到奖励阈值达到 475.0。按 Ctrl+C 退出。

```
Episode 10      Last reward: 26.00      Average reward: 10.01
Episode 20      Last reward: 16.00      Average reward: 11.27
Episode 30      Last reward: 49.00      Average reward: 18.62
Episode 40      Last reward: 45.00      Average reward: 26.09
Episode 50      Last reward: 44.00      Average reward: 30.03
Episode 60      Last reward: 111.00     Average reward: 42.23
Episode 70      Last reward: 131.00     Average reward: 70.11
Episode 80      Last reward: 87.00      Average reward: 76.51
Episode 90      Last reward: 86.00      Average reward: 95.93
Episode 100     Last reward: 13.00      Average reward: 123.93
Episode 110     Last reward: 33.00      Average reward: 91.39
Episode 120     Last reward: 73.00      Average reward: 76.38
Episode 130     Last reward: 137.00     Average reward: 88.08
Episode 140     Last reward: 89.00      Average reward: 104.96
Episode 150     Last reward: 97.00      Average reward: 98.74
Episode 160     Last reward: 150.00     Average reward: 100.87
Episode 170     Last reward: 126.00     Average reward: 104.38
Episode 180     Last reward: 500.00     Average reward: 213.74
Episode 190     Last reward: 322.00     Average reward: 300.22
Episode 200     Last reward: 165.00     Average reward: 272.71
Episode 210     Last reward: 168.00     Average reward: 233.11
Episode 220     Last reward: 184.00     Average reward: 195.02
Episode 230     Last reward: 284.00     Average reward: 208.32
Episode 240     Last reward: 395.00     Average reward: 247.37
Episode 250     Last reward: 500.00     Average reward: 335.42
Episode 260     Last reward: 500.00     Average reward: 386.30
Episode 270     Last reward: 500.00     Average reward: 405.29
Episode 280     Last reward: 500.00     Average reward: 443.29
Episode 290     Last reward: 500.00     Average reward: 464.65
Solved! Running reward is now 475.3163778435275!
```

在这个示例中，我们展示了如何使用 RPC 作为跨工作进程传递数据的通信工具，以及如何使用 RRef 引用远程对象。虽然您可以直接在 ProcessGroup 的发送和接收 API 或其他通信/RPC 库的基础上构建整个结构，但通过使用 `torch.distributed.rpc`，您可以获得原生支持并持续优化的性能。

## 使用分布式自动求导和分布式优化器的分布式 RNN

在本节中，我们使用 RNN 模型展示如何使用 RPC API 构建分布式模型并行训练。这个示例 RNN 模型非常小，可以轻松地放入单个 GPU，但我们仍然将其层分布在两个不同的工作进程上以演示这个想法。开发者可以应用类似的技术将更大的模型分布在多个设备和机器上。

RNN 模型设计借鉴了 PyTorch 示例仓库中的词语言模型，它包含三个主要组件：一个嵌入表、一个 LSTM 层和一个解码器。以下代码将嵌入表和解码器包装成子模块，以便它们的构造函数可以传递给 RPC API。在 `EmbeddingTable` 子模块中，我们有意将嵌入层放在 GPU 上以覆盖使用情况。在 v1.4 中，RPC 总是在目标工作进程上创建 CPU 张量参数或返回值。如果函数接收 GPU 张量，您需要显式地将其移动到适当的设备。

### EmbeddingTable 类

```python
class EmbeddingTable(nn.Module):
    """
    RNNModel 的编码层
    """
    def __init__(self, ntoken, ninp, dropout):
        super(EmbeddingTable, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp).cuda()
        self.encoder.weight.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        return self.drop(self.encoder(input.cuda()).cpu())
```

### Decoder 类

```python
class Decoder(nn.Module):
    def __init__(self, ntoken, nhid, dropout):
        super(Decoder, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.decoder = nn.Linear(nhid, ntoken)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-0.1, 0.1)

    def forward(self, output):
        return self.decoder(self.drop(output))
```

有了这些子模块，我们现在可以使用 RPC 将它们组合在一起创建一个 RNN 模型。在下面的代码中，`ps` 表示一个参数服务器，它托管嵌入表和解码器的参数。构造函数使用远程 API 在参数服务器上创建 `EmbeddingTable` 对象和 `Decoder` 对象，并在本地创建 LSTM 子模块。

在前向传播期间，训练器使用 `EmbeddingTable` RRef 找到远程子模块，并使用 RPC 将输入数据传递给 `EmbeddingTable` 并获取查找结果。然后，它通过本地 LSTM 层运行嵌入，最后使用另一个 RPC 将输出发送到 `Decoder` 子模块。

总的来说，要实现分布式模型并行训练，开发者可以将模型划分为子模块，调用 RPC 远程创建子模块实例，并在需要时使用 RRef 找到它们。正如您在下面的代码中看到的，它看起来非常类似于单机模型并行训练。主要区别是将 `Tensor.to(device)` 替换为 RPC 函数。

### RNNModel 类

```python
class RNNModel(nn.Module):
    def __init__(self, ps, ntoken, ninp, nhid, nlayers, dropout=0.5):
        super(RNNModel, self).__init__()

        # 远程设置嵌入表
        self.emb_table_rref = rpc.remote(ps, EmbeddingTable, args=(ntoken, ninp, dropout))
        # 本地设置 LSTM
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        # 远程设置解码器
        self.decoder_rref = rpc.remote(ps, Decoder, args=(ntoken, nhid, dropout))

    def forward(self, input, hidden):
        # 传递输入到远程嵌入表并获取嵌入张量
        emb = _remote_method(EmbeddingTable.forward, self.emb_table_rref, input)
        output, hidden = self.rnn(emb, hidden)
        # 传递输出到远程解码器并获取解码输出
        decoded = _remote_method(Decoder.forward, self.decoder_rref, output)
        return decoded, hidden
```

我将继续翻译剩余部分：

### 参数 RRefs 辅助函数

在引入分布式优化器之前，让我们添加一个辅助函数来生成模型参数的 RRefs 列表，这将被分布式优化器使用。在本地训练中，应用程序可以调用 `Module.parameters()` 来获取所有参数张量的引用，并将其传递给本地优化器进行后续更新。然而，在分布式训练场景中，一些参数位于远程机器上，因此相同的 API 不起作用。因此，与其传递参数张量列表，分布式优化器接收 RRefs 列表，每个模型参数对应一个 RRef，无论是本地还是远程模型参数。这个辅助函数非常简单，只需调用 `Module.parameters()` 并为每个参数创建一个本地 RRef。

```python
def _parameter_rrefs(module):
    param_rrefs = []
    for param in module.parameters():
        param_rrefs.append(RRef(param))
    return param_rrefs
```

由于 `RNNModel` 包含三个子模块，我们需要调用 `_parameter_rrefs` 三次，并将其封装到另一个辅助函数中。

```python
class RNNModel(nn.Module):
    ...
    def parameter_rrefs(self):
        remote_params = []
        # 获取嵌入表的 RRefs
        remote_params.extend(_remote_method(_parameter_rrefs, self.emb_table_rref))
        # 为本地参数创建 RRefs
        remote_params.extend(_parameter_rrefs(self.rnn))
        # 获取解码器的 RRefs
        remote_params.extend(_remote_method(_parameter_rrefs, self.decoder_rref))
        return remote_params
```

### 训练循环

现在，我们准备实现训练循环。在初始化模型参数之后，我们创建 `RNNModel` 和 `DistributedOptimizer`。分布式优化器将获取参数 RRefs 列表，找到所有不同的所有者工作进程，并在每个所有者工作进程上使用给定的参数（在本例中是 SGD）创建本地优化器（例如 `lr=0.05`）。

在训练循环中，它首先创建一个分布式自动求导上下文，这将帮助分布式自动求导引擎找到梯度和相关的 RPC 发送/接收函数。分布式自动求导引擎的设计细节可以在其设计说明中找到。然后，它像本地模型一样开始前向传播，并运行分布式反向传播。对于分布式反向传播，您只需指定根节点列表，在本例中是损失张量。分布式自动求导引擎将自动遍历分布式图并正确写入梯度。接下来，它在分布式优化器上运行步骤函数，这将联系所有相关的本地优化器以更新模型参数。

与本地训练相比，一个小的区别是您不需要运行 `zero_grad()`，因为每个自动求导上下文都有专用空间来存储梯度，而且我们每次迭代都会创建一个上下文，所以不同迭代的梯度不会累积到相同的张量集。

```python
def run_trainer():
    batch = 5
    ntoken = 10
    ninp = 2

    nhid = 3
    nindices = 3
    nlayers = 4
    hidden = (
        torch.randn(nlayers, nindices, nhid),
        torch.randn(nlayers, nindices, nhid)
    )

    model = rnn.RNNModel('ps', ntoken, ninp, nhid, nlayers)

    # 设置分布式优化器
    opt = DistributedOptimizer(
        optim.SGD,
        model.parameter_rrefs(),
        lr=0.05,
    )

    criterion = torch.nn.CrossEntropyLoss()

    def get_next_batch():
        for _ in range(5):
            data = torch.LongTensor(batch, nindices) % ntoken
            target = torch.LongTensor(batch, ntoken) % nindices
            yield data, target

    # 训练 10 个迭代
    for epoch in range(10):
        for data, target in get_next_batch():
            # 创建分布式自动求导上下文
            with dist_autograd.context() as context_id:
                hidden[0].detach_()
                hidden[1].detach_()
                output, hidden = model(data, hidden)
                loss = criterion(output, target)
                # 运行分布式反向传播
                dist_autograd.backward(context_id, [loss])
                # 运行分布式优化器
                opt.step(context_id)
                # 不需要清零梯度，因为梯度会累积到
                # 分布式自动求导上下文中，
                # 每次迭代都会重置。
        print("训练轮次 {}".format(epoch))
```

### 启动工作进程

最后，让我们添加一些胶水代码来启动参数服务器和训练器进程。

```python
def run_worker(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    if rank == 1:
        rpc.init_rpc("trainer", rank=rank, world_size=world_size)
        _run_trainer()
    else:
        rpc.init_rpc("ps", rank=rank, world_size=world_size)
        # 参数服务器不执行任何操作
        pass

    # 阻塞直到所有 RPC 完成
    rpc.shutdown()


if __name__=="__main__":
    world_size = 2
    mp.spawn(run_worker, args=(world_size, ), nprocs=world_size, join=True)
```
