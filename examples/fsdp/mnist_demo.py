import argparse
from typing import Any, Dict

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR, _LRScheduler
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

from scaletorch.utils import (cleanup_distribute_environment,
                              setup_distributed_environment)
from scaletorch.utils.net_utils import LeNet


class FSDPTrainer:
    """Fully Sharded Data Parallel (FSDP) Trainer for MNIST classification.

    This trainer implements distributed training using PyTorch's FSDP strategy,
    which shards model parameters across multiple GPUs to enable training of large models.

    Attributes:
        rank (int): Current process rank
        world_size (int): Total number of processes
        args (argparse.Namespace): Training arguments and hyperparameters
        model (FSDP): The FSDP-wrapped model
        optimizer (Optimizer): The optimizer for training
        scheduler (_LRScheduler): Learning rate scheduler
        train_loader (DataLoader): DataLoader for training data
        test_loader (DataLoader): DataLoader for test data
    """

    def __init__(self, model: nn.Module, args: argparse.Namespace, rank: int,
                 world_size: int) -> None:
        """Initialize the FSDP trainer.

        Args:
            model: The neural network model to train
            args: Command line arguments containing training parameters
            rank: Current process rank
            world_size: Total number of processes
        """
        self.rank = rank
        self.world_size = world_size
        self.args = args

        # Set device for current process
        torch.cuda.set_device(rank)

        # Wrap model with FSDP
        self.model = FSDP(model.to(rank))

        # Initialize optimizer and learning rate scheduler
        self.optimizer: Optimizer = torch.optim.Adadelta(
            self.model.parameters(), lr=args.lr)
        self.scheduler: _LRScheduler = StepLR(self.optimizer,
                                              step_size=1,
                                              gamma=args.gamma)

        # Setup data loaders
        self._setup_data()

        # Initialize CUDA events for timing
        self.init_start_event = torch.cuda.Event(enable_timing=True)
        self.init_end_event = torch.cuda.Event(enable_timing=True)

    def _setup_data(self) -> None:
        """Set up datasets, samplers, and data loaders for training and
        testing."""
        # Define data transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ])

        # Create datasets
        train_dataset: Dataset = datasets.MNIST('./data',
                                                train=True,
                                                download=True,
                                                transform=transform)
        test_dataset: Dataset = datasets.MNIST('./data',
                                               train=False,
                                               transform=transform)

        # Initialize distributed samplers
        self.train_sampler = DistributedSampler(train_dataset,
                                                rank=self.rank,
                                                num_replicas=self.world_size,
                                                shuffle=True)
        self.test_sampler = DistributedSampler(test_dataset,
                                               rank=self.rank,
                                               num_replicas=self.world_size)

        # Configure dataloader parameters
        train_kwargs: Dict[str, Any] = {
            'batch_size': self.args.batch_size,
            'sampler': self.train_sampler,
            'num_workers': 2,
            'pin_memory': True,
            'shuffle': False,
        }
        test_kwargs: Dict[str, Any] = {
            'batch_size': self.args.test_batch_size,
            'sampler': self.test_sampler,
            'num_workers': 2,
            'pin_memory': True,
            'shuffle': False,
        }

        # Create data loaders
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, **train_kwargs)
        self.test_loader = torch.utils.data.DataLoader(test_dataset,
                                                       **test_kwargs)

    def train_epoch(self, epoch: int) -> None:
        """Train the model for one epoch.

        Args:
            epoch: Current epoch number
        """
        self.model.train()
        ddp_loss = torch.zeros(2).to(self.rank)
        self.train_sampler.set_epoch(epoch)

        for data, target in self.train_loader:
            # Move data to appropriate device
            data, target = data.to(self.rank), target.to(self.rank)

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target, reduction='sum')

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Accumulate statistics
            ddp_loss[0] += loss.item()  # Sum of losses
            ddp_loss[1] += len(data)  # Number of samples

        # Aggregate statistics across processes
        dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

        # Print training statistics (only on rank 0)
        if self.rank == 0:
            print('Train Epoch: {} \tLoss: {:.6f}'.format(
                epoch, ddp_loss[0] / ddp_loss[1]))

    def test(self) -> None:
        """Evaluate the model on the test dataset."""
        self.model.eval()
        ddp_loss = torch.zeros(3).to(self.rank)  # [sum_loss, correct, total]

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.rank), target.to(self.rank)
                output = self.model(data)

                # Accumulate test statistics
                ddp_loss[0] += F.nll_loss(output, target,
                                          reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                ddp_loss[1] += pred.eq(target.view_as(pred)).sum().item()
                ddp_loss[2] += len(data)

        # Aggregate statistics across processes
        dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

        # Print test statistics (only on rank 0)
        if self.rank == 0:
            test_loss = ddp_loss[0] / ddp_loss[2]
            print(
                'Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.
                format(
                    test_loss,
                    int(ddp_loss[1]),
                    int(ddp_loss[2]),
                    100.0 * ddp_loss[1] / ddp_loss[2],
                ))

    def train(self) -> None:
        """Execute the complete training loop for all epochs."""
        self.init_start_event.record()

        for epoch in range(1, self.args.epochs + 1):
            self.train_epoch(epoch)
            self.test()
            self.scheduler.step()

        self.init_end_event.record()

        if self.rank == 0:
            print(
                f'CUDA event elapsed time: {self.init_start_event.elapsed_time(self.init_end_event) / 1000}sec'
            )
            print(f'{self.model}')

    def save_model(self) -> None:
        """Save the trained model if save_model flag is set.

        Only rank 0 process saves the model after synchronization.
        """
        if self.args.save_model:
            dist.barrier()  # Synchronize processes
            states = self.model.state_dict()
            if self.rank == 0:
                torch.save(states, 'mnist_cnn.pt')


def fsdp_main(rank, world_size, args):
    setup_distributed_environment(rank, world_size)

    # 创建模型
    model = LeNet()

    # 初始化训练器
    trainer = FSDPTrainer(model, args, rank, world_size)

    # 开始训练
    trainer.train()

    # 保存模型
    trainer.save_model()

    cleanup_distribute_environment()


if __name__ == '__main__':
    # 保持原有的参数解析代码不变
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    # ... 参数设置代码 ...
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(fsdp_main, args=(WORLD_SIZE, args), nprocs=WORLD_SIZE, join=True)
