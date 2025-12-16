from typing import Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import tyro
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

from scaletorch.trainer.config import TrainingArguments
from scaletorch.utils import cleanup_dist, get_system_info, init_dist_pytorch
from scaletorch.utils.lenet_model import LeNet
from scaletorch.utils.logger_utils import get_logger

logger = get_logger(__name__)


class FSDPTrainer:
    """Fully Sharded Data Parallel (FSDP) Trainer for MNIST classification.

    This trainer implements distributed training using PyTorch's FSDP strategy,
    which shards model parameters across multiple GPUs to enable training of large models.

    Attributes:
        args (TrainingArguments): Training arguments and hyperparameters.
        rank (int): Current process rank.
        world_size (int): Total number of processes.
        model (FSDP): The FSDP-wrapped model.
        optimizer (Optimizer): The optimizer for training.
        scheduler (_LRScheduler): Learning rate scheduler.
        train_loader (DataLoader): DataLoader for training data.
        test_loader (DataLoader): DataLoader for test data.
    """

    def __init__(
        self,
        args: TrainingArguments,
        rank: int,
        world_size: int,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
    ) -> None:
        """Initialize the FSDP trainer.

        Args:
            args (TrainingArguments): Command line arguments containing training parameters.
            model (nn.Module): The neural network model to train.
            train_loader (DataLoader): DataLoader for training data.
            test_loader (DataLoader): DataLoader for test data.
            optimizer (torch.optim.Optimizer): The optimizer for training.
            scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
            rank (int): Current process rank.
            world_size (int): Total number of processes.

        Raises:
            ValueError: If rank or world_size is None.
        """
        self.args = args
        self.rank = rank
        self.world_size = world_size

        # Configure device and model distribution
        self.device = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(self.device)
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
            epoch (int): Current epoch number.
        """
        self.model.train()
        ddp_loss = torch.zeros(2).to(self.device)

        # Set epoch for distributed sampler to ensure proper shuffling
        if hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(epoch)

        for data, target in self.train_loader:
            # Move data to appropriate device
            data, target = data.to(self.device), target.to(self.device)

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
            logger.info(
                f'Train Epoch: {epoch} \tLoss: {ddp_loss[0] / ddp_loss[1]:.6f}'
            )

    def test(self) -> None:
        """Evaluate the model on the test dataset."""
        self.model.eval()
        ddp_loss = torch.zeros(3).to(self.device)  # [sum_loss, correct, total]

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
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
            logger.info(f'Test set: Average loss: {test_loss:.4f}, '
                        f'Accuracy: {int(ddp_loss[1])}/{int(ddp_loss[2])} '
                        f'({100.0 * ddp_loss[1] / ddp_loss[2]:.2f}%)\n')

    def train(self) -> None:
        """Execute the complete training loop for all epochs."""
        self.init_start_event.record()

        for epoch in range(1, self.args.epochs + 1):
            self.train_epoch(epoch)
            self.test()
            self.scheduler.step()

        self.init_end_event.record()

        if self.rank == 0:
            logger.info(
                f'CUDA event elapsed time: {self.init_start_event.elapsed_time(self.init_end_event) / 1000}sec'
            )
            logger.info(f'{self.model}')

    def save_model(self) -> None:
        """Save the trained model if save_model flag is set.

        Only rank 0 process saves the model after synchronization.
        """
        if self.args.save_model:
            dist.barrier()  # Synchronize processes
            states = self.model.state_dict()
            if self.rank == 0:
                torch.save(states, 'mnist_cnn.pt')


def prepare_data(args: TrainingArguments, rank: int,
                 world_size: int) -> Tuple[DataLoader, DataLoader]:
    """Set up datasets, samplers, and data loaders for training and testing.

    Args:
        args (TrainingArguments): Command line arguments.
        rank (int): Current process rank.
        world_size (int): Total number of processes.

    Returns:
        Tuple[DataLoader, DataLoader]: A tuple containing (train_loader, test_loader).
    """
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307, ), (0.3081, ))])

    # Create datasets
    train_dataset = datasets.MNIST(root=args.data_path,
                                   train=True,
                                   download=True,
                                   transform=transform)
    test_dataset = datasets.MNIST(root=args.data_path,
                                  train=False,
                                  download=True,
                                  transform=transform)

    # Initialize samplers with rank/world_size
    train_sampler = DistributedSampler(train_dataset,
                                       num_replicas=world_size,
                                       rank=rank,
                                       shuffle=True)
    test_sampler = DistributedSampler(test_dataset,
                                      num_replicas=world_size,
                                      rank=rank,
                                      shuffle=False)

    # Create data loaders with enhanced configuration
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, test_loader


def main(rank: int, world_size: int, args: TrainingArguments) -> None:
    """Main training function for each distributed process.

    Args:
        rank (int): Current process rank.
        world_size (int): Total number of processes.
        args (TrainingArguments): Command line arguments.
    """
    try:
        get_system_info()
        setup_distributed_environment(rank=rank, world_size=world_size)

        # Prepare data loaders with rank/world_size
        train_loader, test_loader = prepare_data(args, rank, world_size)

        model = LeNet()
        optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

        # Initialize trainer with correct rank/world_size
        trainer = FSDPTrainer(
            args=args,
            rank=rank,
            world_size=world_size,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            scheduler=scheduler,
        )

        trainer.train()
        trainer.save_model()

    except Exception as e:
        logger.error(f'Training failed on rank {rank}: {str(e)}')
        raise  # Re-raise exception after logging
    finally:
        cleanup_distribute_environment()


if __name__ == '__main__':
    # Parse command-line arguments
    args: TrainingArguments = tyro.cli(TrainingArguments)
    torch.manual_seed(args.seed)
    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(main, args=(WORLD_SIZE, args), nprocs=WORLD_SIZE, join=True)
