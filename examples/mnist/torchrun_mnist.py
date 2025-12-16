"""
Distributed MNIST Training Example using ScaleTorch.

This script demonstrates how to train a LeNet model on the MNIST dataset
using DistributedDataParallel (DDP) with ScaleTorch.
"""

import dataclasses
import json
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
from transformers import HfArgumentParser

from scaletorch.trainer.config import ScaleTorchArguments
from scaletorch.utils import (LeNet, cleanup_dist, get_current_device,
                              get_dist_info, get_logger, get_system_info,
                              init_dist_pytorch)

logger = get_logger(__name__)


class DistributedTrainer:
    """A comprehensive distributed trainer class for PyTorch model training
    using DistributedDataParallel (DDP) strategy.

    This class encapsulates the entire training workflow, including:
    - Distributed setup
    - Device management
    - Model training
    - Validation
    - Checkpointing
    """

    def __init__(
        self,
        args: ScaleTorchArguments,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
    ) -> None:
        """Initialize the Distributed Trainer with all necessary components.

        Args:
            args (ScaleTorchArguments): Parsed command-line arguments
            model (nn.Module): Neural network model to be trained
            train_loader (DataLoader): Training data loader
            test_loader (DataLoader): Validation/test data loader
            optimizer (torch.optim.Optimizer): Optimization algorithm
            scheduler (torch.optim.lr_scheduler.LRScheduler): Learning rate scheduler
        """
        self.args = args

        # Determine process rank and local rank
        self.rank, world_size, local_rank = get_dist_info()

        # Setup device
        self.device = get_current_device()

        # Wrap model with DistributedDataParallel
        self.model = model.to(self.device)
        if dist.is_initialized() and world_size > 1:
            self.model = DDP(model,
                             device_ids=[local_rank],
                             output_device=local_rank)

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler

    def run_batch(self, source: torch.Tensor, targets: torch.Tensor) -> float:
        """Process a single training batch in a distributed environment.

        Args:
            source (torch.Tensor): Input data tensor
            targets (torch.Tensor): Target labels tensor

        Returns:
            float: Computed loss value for the batch
        """
        # Zero out previous gradients
        self.optimizer.zero_grad()

        # Compute model output
        output = self.model(source)

        # Compute negative log-likelihood loss
        loss = F.nll_loss(output, targets)

        # Backpropagate and update parameters
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def run_epoch(self, epoch: int) -> float:
        """Train the model for a single epoch in a distributed setting.

        Args:
            epoch (int): Current training epoch number

        Returns:
            float: Average training loss for the epoch
        """
        self.model.train()

        # Set epoch for distributed sampler to ensure proper shuffling
        if hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(epoch)

        total_loss = 0.0
        num_batches = len(self.train_loader)

        for batch_idx, (data, target) in enumerate(self.train_loader):
            # Move data to selected device
            data, target = data.to(self.device), target.to(self.device)

            # Process batch and accumulate loss
            batch_loss = self.run_batch(data, target)
            total_loss += batch_loss

            # Periodic logging for primary process
            if self.rank == 0 and batch_idx % self.args.log_interval == 0:
                logger.info(
                    f'Train Epoch: {epoch} '
                    f'[{batch_idx * len(data)}/{len(self.train_loader.dataset)} '
                    f'({100.0 * batch_idx / len(self.train_loader):.0f}%)]\t'
                    f'Loss: {batch_loss:.6f}')

        return total_loss / num_batches if num_batches > 0 else 0.0

    def test(self) -> Dict[str, float]:
        """Evaluate model performance on test dataset across distributed
        processes.

        Returns:
            Dict[str, float]: Evaluation metrics including loss and accuracy
        """
        self.model.eval()
        test_loss, correct = 0.0, 0

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

        # Log results on primary process
        if self.rank == 0:
            logger.info(f'\nTest set: Average loss: {test_loss:.4f}, '
                        f'Accuracy: {correct}/{len(self.test_loader.dataset)} '
                        f'({accuracy:.0f}%)\n')

        return {'loss': test_loss, 'accuracy': accuracy}

    def train(self) -> None:
        """Execute the complete distributed training workflow.

        Manages epoch training, testing, and optional model checkpointing.
        """
        try:
            for epoch in range(1, self.args.epochs + 1):
                # Train for one epoch
                epoch_loss = self.run_epoch(epoch)

                # Log epoch loss on primary process
                if self.rank == 0:
                    logger.info(f'Epoch {epoch} Loss: {epoch_loss:.4f}')

                # Synchronize processes if in distributed mode
                if dist.is_initialized() and dist.get_world_size() > 1:
                    dist.barrier()

                # Perform testing
                test_metrics = self.test()

                if self.rank == 0:
                    logger.info(f'Epoch {epoch}, Eval Metrics: {test_metrics}')

                # Update learning rate
                self.scheduler.step()

            # Optional model saving
            if self.rank == 0 and self.args.save_model_checkpoint:
                self.save_checkpoint(self.args.epochs)

        except Exception as e:
            logger.error(f'Training failed: {e}')
            raise

    def save_checkpoint(self,
                        epoch: int,
                        path: Optional[str] = None) -> Optional[str]:
        """Save a comprehensive model checkpoint.

        Args:
            epoch (int): Current training epoch
            path (Optional[str], optional): Custom checkpoint path

        Returns:
            Optional[str]: Path where checkpoint was saved, or None
        """
        if self.rank != 0:
            return None

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
    """Prepare distributed datasets and data loaders for training.

    Args:
        args (ScaleTorchArguments): Parsed command-line arguments

    Returns:
        Tuple[DataLoader, DataLoader]: Training and testing data loaders
    """
    # Data normalization parameters for MNIST
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


def main(args: ScaleTorchArguments) -> None:
    """Main entry point for distributed training workflow.

    Manages distributed setup, training initialization, and execution.
    """
    # Log system information
    get_system_info()
    logger.info('Distributed training started')
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
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            scheduler=scheduler,
        )

        # Start training
        trainer.train()

    except Exception as e:
        logger.error(f'Training failed: {e}')
        raise
    finally:
        # Cleanup distributed resources
        cleanup_dist()


if __name__ == '__main__':
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

    main(args)
