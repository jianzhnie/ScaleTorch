import argparse
import logging
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

from scaletorch.utils import (cleanup_distribute_environment, get_system_info,
                              setup_distributed_environment)
from scaletorch.utils.net_utils import LeNet

# Configure global logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


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

    def __init__(
        self,
        args: argparse.Namespace,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        rank: int,
        world_size: int,
    ) -> None:
        """Initialize the FSDP trainer.

        Args:
            model: The neural network model to train
            args: Command line arguments containing training parameters
            rank: Current process rank
            world_size: Total number of processes
        """
        self.args = args
        self.rank = rank
        self.world_size = world_size

        # Set device for current process
        torch.cuda.set_device(rank)
        # Wrap model with FSDP
        self.model = FSDP(model.to(rank))

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Initialize CUDA events for timing
        self.init_start_event = torch.cuda.Event(enable_timing=True)
        self.init_end_event = torch.cuda.Event(enable_timing=True)

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


def prepare_data(self) -> None:
    """Set up datasets, samplers, and data loaders for training and testing."""
    # Define data transformations
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307, ), (0.3081, ))])

    # Create datasets
    # Load MNIST datasets
    train_dataset = datasets.MNIST(root=args.data_path,
                                   train=True,
                                   download=True,
                                   transform=transform)
    test_dataset = datasets.MNIST(root=args.data_path,
                                  train=False,
                                  download=True,
                                  transform=transform)
    # Initialize distributed samplers
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    test_sampler = DistributedSampler(test_dataset, shuffle=False)

    # Create data loaders with enhanced configuration
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True,
    )
    return train_loader, test_loader


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for distributed training configuration.

    Returns:
        argparse.Namespace: Parsed and validated training arguments
    """
    parser = argparse.ArgumentParser(
        description='Distributed PyTorch MNIST Training')

    # Training hyperparameters
    parser.add_argument('--batch-size',
                        type=int,
                        default=64,
                        help='Training batch size')
    parser.add_argument('--test-batch-size',
                        type=int,
                        default=1000,
                        help='Test batch size')
    parser.add_argument('--epochs',
                        type=int,
                        default=5,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1.0, help='Learning rate')
    parser.add_argument('--gamma',
                        type=float,
                        default=0.7,
                        help='Learning rate decay factor')

    # Utility arguments
    parser.add_argument('--dry-run',
                        action='store_true',
                        help='Quick training verification')
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        help='Random seed for reproducibility')
    parser.add_argument('--log-interval',
                        type=int,
                        default=100,
                        help='Batch logging frequency')
    parser.add_argument('--save-model',
                        action='store_true',
                        help='Save trained model checkpoint')
    parser.add_argument('--data-path',
                        type=str,
                        default='./data',
                        help='Dataset download directory')

    return parser.parse_args()


def main(rank, world_size, args):
    get_system_info()
    # Parse command-line arguments
    args = parse_arguments()
    # Enable cuDNN benchmark for performance optimization
    torch.backends.cudnn.benchmark = True

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)

    # Validate GPU availability
    if torch.cuda.device_count() < 2:
        logger.error('Distributed training requires multiple GPUs')
        sys.exit(1)

    # Provide distributed launch guidance
    logger.info('Distributed launch command:')
    logger.info('torchrun --nproc_per_node=<num_gpus> script_name.py')

    try:
        setup_distributed_environment(rank, world_size)
        # Prepare data loaders
        train_loader, test_loader = prepare_data(args)
        # 创建模型
        model = LeNet()

        # Setup optimizer and learning rate scheduler
        optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

        # 初始化训练器
        trainer = FSDPTrainer(
            args=args,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            rank=None,
            world_size=None,
        )

        # 开始训练
        trainer.train()

        # 保存模型
        trainer.save_model()

    except Exception as e:
        logger.error(f'Training failed: {e}')
    finally:
        # Cleanup distributed resources
        cleanup_distribute_environment()


if __name__ == '__main__':
    # 保持原有的参数解析代码不变
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    # ... 参数设置代码 ...
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(main, args=(WORLD_SIZE, args), nprocs=WORLD_SIZE, join=True)
