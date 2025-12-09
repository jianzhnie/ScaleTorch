import os
import sys
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

sys.path.append(os.getcwd())
from scaletorch.trainer.config import TrainingArguments
from scaletorch.utils.env_utils import get_system_info
from scaletorch.utils.lenet_model import LeNet
from scaletorch.utils.logger_utils import get_logger
from scaletorch.utils.torch_dist import (cleanup_distribute_environment,
                                         setup_distributed_environment)

logger = get_logger(__name__)


class DistributedTrainer:
    """A distributed trainer class for PyTorch model training using
    DistributedDataParallel.

    This class handles distributed training across multiple GPUs, including:
    - Model distribution
    - Data parallelization
    - Metrics aggregation
    - Checkpoint management
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
        """Initialize the Distributed Trainer.

        Args:
            args (TrainingArguments): Command-line arguments.
            rank (int): Local rank of the current process.
            world_size (int): Total number of distributed processes.
            model (nn.Module): Neural network model.
            train_loader (DataLoader): Training data loader.
            test_loader (DataLoader): Test data loader.
            optimizer (torch.optim.Optimizer): Optimization algorithm.
            scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        """
        self.args = args
        self.rank = rank
        self.world_size = world_size

        # Configure device and model distribution
        self.device = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(self.device)

        # Move model to device and wrap with DDP for distributed training
        model = model.to(self.device)
        self.model = DDP(
            model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=
            False,  # Optimization flag for better performance
            broadcast_buffers=
            False,  # Disable buffer broadcasting when not needed
            gradient_as_bucket_view=True,  # Memory optimization
        )

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger

    def run_batch(self, source: torch.Tensor, targets: torch.Tensor) -> float:
        """Process a single training batch in a distributed setting.

        Args:
            source (torch.Tensor): Input data tensor.
            targets (torch.Tensor): Target labels tensor.

        Returns:
            float: Computed loss value for the batch.
        """
        # Reset gradients
        self.optimizer.zero_grad()

        # Forward pass
        output = self.model(source)

        # Compute loss
        loss = F.nll_loss(output, targets)

        # Backward pass and parameter update
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def run_epoch(self, epoch: int) -> float:
        """Train the model for one epoch in a distributed setting.

        Args:
            epoch (int): Current epoch number.

        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()

        # Set epoch for distributed sampler
        self.train_loader.sampler.set_epoch(epoch)

        total_loss = 0.0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # Move data to device
            data, target = data.to(self.device), target.to(self.device)

            # Process batch and accumulate loss
            batch_loss = self.run_batch(data, target)
            total_loss += batch_loss

            # Log progress at specified intervals
            if self.rank == 0 and batch_idx % self.args.log_interval == 0:
                self.logger.info(
                    f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(self.train_loader.dataset)} '
                    f'({100.0 * batch_idx / len(self.train_loader):.0f}%)]\tLoss: {batch_loss:.6f}'
                )
        return total_loss / len(self.train_loader)

    def test(self) -> Dict[str, float]:
        """Evaluate the model on the test dataset in a distributed setting.

        Returns:
            Dict[str, float]: Dictionary containing test loss and accuracy.
        """
        self.model.eval()
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)

                # Accumulate test loss
                test_loss += F.nll_loss(output, target, reduction='sum').item()

                # Count correct predictions
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        # Reduce metrics across all processes
        metrics = torch.tensor([test_loss, correct], device=self.device)
        dist.all_reduce(metrics)

        test_loss = metrics[0].item() / len(self.test_loader.dataset)
        correct = metrics[1].item()
        accuracy = 100.0 * correct / len(self.test_loader.dataset)

        if self.rank == 0:
            self.logger.info(
                f'\nTest set: Average loss: {test_loss:.4f}, '
                f'Accuracy: {correct}/{len(self.test_loader.dataset)} ({accuracy:.0f}%)\n'
            )

        return {'loss': test_loss, 'accuracy': accuracy}

    def train(self) -> None:
        """Execute the complete distributed model training process."""
        for epoch in range(1, self.args.epochs + 1):
            # Train for one epoch
            epoch_loss = self.run_epoch(epoch)

            # Log epoch loss on the primary process
            if self.rank == 0:
                self.logger.info(f'Epoch {epoch} Loss: {epoch_loss:.4f}')

            # Synchronize all processes
            dist.barrier()

            # Perform testing on the primary process
            test_metrics = self.test()
            if self.rank == 0:
                self.logger.info(
                    f'Epoch {epoch}, Eval Metrics: {test_metrics}')

            # Step learning rate scheduler
            self.scheduler.step()

        # Save trained model on the primary process
        if self.rank == 0 and self.args.save_model:
            self.save_checkpoint(self.args.epochs)

    def save_checkpoint(self,
                        epoch: int,
                        path: Optional[str] = None) -> Optional[str]:
        """Save model checkpoint with training state.

        Args:
            epoch (int): Current epoch number.
            path (Optional[str]): Custom checkpoint path.

        Returns:
            Optional[str]: Path where checkpoint was saved.
        """
        if self.rank != 0:
            return None

        # Use provided path or generate default
        checkpoint_path = path or f'checkpoint_epoch_{epoch}.pt'

        # Save comprehensive checkpoint
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': self.model.module.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
            },
            checkpoint_path,
        )

        self.logger.info(
            f'Epoch {epoch} | Checkpoint saved at {checkpoint_path}')
        return checkpoint_path


def prepare_data(args: TrainingArguments, rank: int,
                 world_size: int) -> Tuple[DataLoader, DataLoader]:
    """Prepare distributed datasets and data loaders.

    Args:
        args (TrainingArguments): Command-line arguments.
        rank (int): Local rank of the current process.
        world_size (int): Total number of distributed processes.

    Returns:
        Tuple[DataLoader, DataLoader]: A tuple containing (train_loader, test_loader).
    """
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307, ), (0.3081, ))])

    # Load datasets with error handling
    try:
        train_dataset = datasets.MNIST(root=args.data_path,
                                       train=True,
                                       download=True,
                                       transform=transform)
        test_dataset = datasets.MNIST(root=args.data_path,
                                      train=False,
                                      download=True,
                                      transform=transform)
    except Exception as e:
        raise RuntimeError(f'Failed to load MNIST dataset: {e}')

    # Create samplers and loaders with optimal settings
    train_sampler = DistributedSampler(train_dataset,
                                       num_replicas=world_size,
                                       rank=rank,
                                       shuffle=True,
                                       seed=args.seed)
    test_sampler = DistributedSampler(test_dataset, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    return train_loader, test_loader


def train_process(rank: int, world_size: int, args: TrainingArguments) -> None:
    """Training process for each distributed process.

    Args:
        rank (int): Local GPU rank.
        world_size (int): Total number of processes.
        args (TrainingArguments): Command-line arguments.
    """
    try:
        torch.cuda.set_device(rank)

        # Setup distributed environment
        setup_distributed_environment(rank=rank, world_size=world_size)

        # Prepare data loaders
        train_loader, test_loader = prepare_data(args, rank, world_size)

        # Initialize model
        model = LeNet()

        # Setup optimizer and scheduler
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

        # Create distributed trainer
        trainer = DistributedTrainer(
            args=args,
            rank=rank,
            world_size=world_size,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            scheduler=scheduler,
        )

        # Start training
        trainer.train()

    except Exception as e:
        logger.error(f'Process {rank} failed: {e}')
        raise
    finally:
        # Ensure cleanup of distributed environment
        cleanup_distribute_environment()


def setup_training_environment(args: TrainingArguments) -> None:
    """Configure training environment settings.

    Args:
        args (TrainingArguments): Command-line arguments containing configuration.
    """
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    logger.info(f'Set random seed to {args.seed}')


def validate_gpu_requirements() -> int:
    """Validate GPU requirements for distributed training.

    Returns:
        int: Number of available GPUs.

    Raises:
        RuntimeError: If fewer than 2 GPUs are available.
    """
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available on this system')

    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise RuntimeError(
            f'Distributed training requires at least 2 GPUs, but found {world_size}'
        )
    return world_size


def main() -> None:
    """Main function to launch distributed training.

    Sets up the distributed environment, initializes training components, and
    launches multiple training processes.
    """
    # Log system information
    get_system_info()

    # Initialize training configuration
    args: TrainingArguments = tyro.cli(TrainingArguments)
    setup_training_environment(args)

    # Validate GPU availability
    world_size = validate_gpu_requirements()

    # Launch distributed processes
    mp.spawn(train_process, args=(world_size, args), nprocs=world_size)


if __name__ == '__main__':
    main()
