import dataclasses
import json
import os
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
from transformers import HfArgumentParser

from scaletorch.trainer.config import ScaleTorchArguments
from scaletorch.utils import (LeNet, get_current_device, get_device_count,
                              get_logger, get_system_info, init_dist_pytorch)

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
        args: ScaleTorchArguments,
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
            args (ScaleTorchArguments): Command-line arguments.
            rank (int): Local rank of the current process.
            world_size (int): Total number of distributed processes.
            model (nn.Module): Neural network model.
            train_loader (DataLoader): Training data loader.
            test_loader (DataLoader): Test data loader.
            optimizer (torch.optim.Optimizer): Optimization algorithm.
            scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        """
        self.args = args
        # Determine process rank and world size
        self.rank = rank
        self.world_size = world_size
        # Setup device
        self.device = get_current_device()

        # Move model to device and wrap with DDP for distributed training
        model = model.to(self.device)

        # Handle DDP initialization with fallback for different backends
        if dist.is_initialized() and world_size > 1:
            try:
                self.model = DDP(
                    model,
                    device_ids=[rank],
                    output_device=rank,
                    find_unused_parameters=False,
                    broadcast_buffers=False,
                    gradient_as_bucket_view=True,
                )
            except Exception as e:
                logger.warning(
                    f'Could not create DDP with device_ids: {e}. Falling back to default DDP.'
                )
                self.model = DDP(model)
        else:
            self.model = model

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler

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
        if hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(epoch)

        total_loss = 0.0
        num_batches = len(self.train_loader)

        for batch_idx, (data, target) in enumerate(self.train_loader):
            # Move data to device
            data, target = data.to(self.device), target.to(self.device)

            # Process batch and accumulate loss
            batch_loss = self.run_batch(data, target)
            total_loss += batch_loss

            # Log progress at specified intervals
            if self.rank == 0 and batch_idx % self.args.log_interval == 0:
                logger.info(
                    f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(self.train_loader.dataset)} '
                    f'({100.0 * batch_idx / len(self.train_loader):.0f}%)]\tLoss: {batch_loss:.6f}'
                )

        return total_loss / num_batches if num_batches > 0 else 0.0

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

        # Aggregate metrics across all processes
        if dist.is_initialized() and dist.get_world_size() > 1:
            metrics = torch.tensor([test_loss, correct], device=self.device)
            dist.all_reduce(metrics)

            # Compute final metrics
            test_loss = metrics[0].item() / len(self.test_loader.dataset)
            correct = metrics[1].item()
        else:
            test_loss = test_loss / len(self.test_loader.dataset)
        accuracy = 100.0 * correct / len(self.test_loader.dataset)

        if self.rank == 0:
            logger.info(
                f'\nTest set: Average loss: {test_loss:.4f}, '
                f'Accuracy: {correct}/{len(self.test_loader.dataset)} ({accuracy:.0f}%)\n'
            )

        return {'loss': test_loss, 'accuracy': accuracy}

    def train(self) -> None:
        """Execute the complete distributed model training process."""
        try:
            for epoch in range(1, self.args.epochs + 1):
                # Train for one epoch
                epoch_loss = self.run_epoch(epoch)

                # Log epoch loss on the primary process
                if self.rank == 0:
                    logger.info(f'Epoch {epoch} Loss: {epoch_loss:.4f}')

                # Synchronize all processes if in distributed mode
                if dist.is_initialized() and dist.get_world_size() > 1:
                    dist.barrier()

                # Perform testing on the primary process
                test_metrics = self.test()
                if self.rank == 0:
                    logger.info(f'Epoch {epoch}, Eval Metrics: {test_metrics}')

                # Step learning rate scheduler
                self.scheduler.step()

            # Save trained model on the primary process
            if self.rank == 0 and self.args.save_model_checkpoint:
                self.save_checkpoint(self.args.epochs)

        except Exception as e:
            logger.error(f'Training failed: {e}')
            raise

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

        state_dict = {
            'epoch':
            epoch,
            'model_state_dict':
            self.model.module.state_dict()
            if isinstance(self.model, DDP) else self.model.state_dict(),
            'optimizer_state_dict':
            self.optimizer.state_dict(),
            'scheduler_state_dict':
            self.scheduler.state_dict(),
        }

        try:
            torch.save(state_dict, checkpoint_path)
            logger.info(
                f'Epoch {epoch} | Checkpoint saved at {checkpoint_path}')
        except Exception as e:
            logger.error(f'Failed to save checkpoint: {e}')
            return None

        return checkpoint_path


def prepare_data(args: ScaleTorchArguments) -> Tuple[DataLoader, DataLoader]:
    """Prepare distributed datasets and data loaders.

    Args:
        args (ScaleTorchArguments): Command-line arguments.

    Returns:
        Tuple[DataLoader, DataLoader]: A tuple containing (train_loader, test_loader).
    """
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307, ), (0.3081, ))])

    # Load MNIST datasets
    train_dataset = datasets.MNIST(root=args.data_path,
                                   train=True,
                                   download=True,
                                   transform=transform)
    test_dataset = datasets.MNIST(root=args.data_path,
                                  train=False,
                                  download=True,
                                  transform=transform)

    # Create distributed samplers if in distributed mode
    if dist.is_initialized() and dist.get_world_size() > 1:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
    else:
        # For non-distributed training
        from torch.utils.data import RandomSampler, SequentialSampler
        train_sampler = RandomSampler(train_dataset)
        test_sampler = SequentialSampler(test_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True,
    )

    return train_loader, test_loader


def train_process(rank: int, world_size: int,
                  args: ScaleTorchArguments) -> None:
    """Training process for each distributed process.

    Args:
        rank (int): Local GPU rank.
        args (ScaleTorchArguments): Command-line arguments.
    """
    # Log system information
    get_system_info()
    logger.info(f'Distributed training started on rank {rank}')
    try:
        # Setup distributed environment
        init_dist_pytorch()

        # Prepare data loaders
        train_loader, test_loader = prepare_data(args)

        # Initialize model
        model = LeNet()

        # Setup optimizer and learning rate scheduler
        optimizer = optim.Adadelta(model.parameters(), lr=args.learning_rate)
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
        logger.error(f'Training failed on rank {rank}: {e}')
        raise


def main() -> None:
    """Main function to launch distributed training.

    Sets up the distributed environment, initializes training components, and
    launches multiple training processes.
    """
    # Create parser for ScaleTorchArguments
    parser = HfArgumentParser(ScaleTorchArguments)

    # Parse command-line arguments into dataclass
    # This will automatically validate all arguments via __post_init__
    args, = parser.parse_args_into_dataclasses()

    # Log initialization with formatted argument display
    logger.info(
        'Initializing ScaleTorchArguments with parsed command line arguments...'
    )
    logger.info('\n--- Parsed Arguments ---')
    logger.info(json.dumps(dataclasses.asdict(args), indent=4))

    num_device = get_device_count()
    # 获取world size从环境变量或GPU数量
    world_size = int(os.environ.get('WORLD_SIZE', num_device))
    if world_size <= 0:
        world_size = 1  # 回退到单进程

    # Launch distributed processes
    mp.spawn(train_process,
             args=(world_size, args),
             nprocs=world_size,
             join=True)


if __name__ == '__main__':
    main()
