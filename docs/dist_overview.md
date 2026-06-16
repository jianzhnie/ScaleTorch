# PyTorch 分布式概述

这是 `torch.distributed` 的概述页面。本页的目的是将文档分类到不同的主题，并简要描述每个主题。如果你是第一次使用 PyTorch 构建分布式训练应用程序，建议使用此文档导航到最适合你用例的技术。

## 介绍

PyTorch 分布式库包括一组并行模块、通信层以及用于启动和调试大型训练作业的基础设施。

## 并行 API

这些并行模块提供了高级功能，并与现有模型组合：

- [分布式数据并行 (DDP)](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
- [全分片数据并行训练 (FSDP)](https://pytorch.org/docs/stable/fsdp.html)
- [张量并行 (TP)](https://pytorch.org/docs/stable/distributed.tensor.parallel.html)
- [流水线并行 (PP)](https://pytorch.org/docs/main/distributed.pipelining.html)

## 通信 API

[PyTorch 分布式通信层 (C10D)](https://pytorch.org/docs/stable/distributed.html) 提供了集体通信 API（例如，[all_reduce](https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_reduce) 和 [all_gather](https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_gather)）以及点对点通信 API（例如，[send](https://pytorch.org/docs/stable/distributed.html#torch.distributed.send) 和 [isend](https://pytorch.org/docs/stable/distributed.html#torch.distributed.isend)），这些 API 在所有并行实现中都在底层使用。[使用 PyTorch 编写分布式应用程序](https://pytorch.org/tutorials/intermediate/dist_tuto.html)展示了使用 c10d 通信 API 的示例。

## 启动器

[torchrun](https://pytorch.org/docs/stable/elastic/run.html) 是一个广泛使用的启动脚本，用于在本地和远程机器上生成进程以运行分布式 PyTorch 程序。

## 应用并行技术扩展你的模型

数据并行是一种广泛采用的单程序多数据训练范式，其中模型在每个进程上复制，每个模型副本计算不同输入数据样本的局部梯度，在数据并行通信器组内平均梯度后进行每次优化器步骤。

当模型不适合 GPU 时，需要使用模型并行技术（或分片数据并行），并且可以组合在一起形成多维 (N-D) 并行技术。

在决定为你的模型选择哪种并行技术时，使用这些常见指南：

1. 如果你的模型适合单个 GPU，但你想使用多个 GPU 轻松扩展训练，请使用 [分布式数据并行 (DDP)](https://pytorch.org/docs/stable/notes/ddp.html)。

   - 如果你使用多个节点，请使用 [torchrun](https://pytorch.org/docs/stable/elastic/run.html) 启动多个 PyTorch 进程。
   - 另请参阅：[分布式数据并行入门](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

2. 当你的模型无法放入一个 GPU 时，请使用 [全分片数据并行 (FSDP)](https://pytorch.org/docs/stable/fsdp.html)。

   - 另请参阅：[FSDP 入门](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)

3. 如果你在使用 FSDP 时遇到扩展限制，请使用 [张量并行 (TP)](https://pytorch.org/docs/stable/distributed.tensor.parallel.html) 和/或 [流水线并行 (PP)](https://pytorch.org/docs/main/distributed.pipelining.html)。

   - 尝试我们的 [张量并行教程](https://pytorch.org/tutorials/intermediate/TP_tutorial.html)
   - 另请参阅：[TorchTitan 3D 并行的端到端示例](https://github.com/pytorch/torchtitan)

**注意**: 数据并行训练也适用于 [自动混合精度 (AMP)](https://pytorch.org/docs/stable/notes/amp_examples.html#working-with-multiple-gpus)。
