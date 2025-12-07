import os
import sys
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

# Append current working directory to system path
sys.path.append(os.getcwd())

from scaletorch.trainer.config import TrainingArguments
# Import distributed utilities
from scaletorch.utils import (cleanup_distribute_environment, get_system_info,
                              setup_distributed_environment)
from scaletorch.utils.lenet_model import LeNet
from scaletorch.utils.logger_utils import get_logger

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
        args: TrainingArguments,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
    ) -> None:
        """Initialize the Distributed Trainer with all necessary components.

        Args:
            args (TrainingArguments): Parsed command-line arguments
            model (nn.Module): Neural network model to be trained
            train_loader (DataLoader): Training data loader
            test_loader (DataLoader): Validation/test data loader
            optimizer (torch.optim.Optimizer): Optimization algorithm
            scheduler (torch.optim.lr_scheduler.LRScheduler): Learning rate scheduler
        """
        self.args = args

        # Determine process rank and local rank
        self.local_rank = int(os.environ['LOCAL_RANK'])
        self.global_rank = int(os.environ['RANK'])

        # Setup device
        self.device = torch.device(f'cuda:{self.local_rank}')

        # Wrap model with DistributedDataParallel
        self.model = model.to(self.device)
        if dist.is_initialized():
            self.model = DDP(model,
                             device_ids=[self.local_rank],
                             output_device=self.local_rank)

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
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # Move data to selected device
            data, target = data.to(self.device), target.to(self.device)

            # Process batch and accumulate loss
            batch_loss = self.run_batch(data, target)
            total_loss += batch_loss

            # Periodic logging for primary process
            if self.global_rank == 0 and batch_idx % self.args.log_interval == 0:
                logger.info(
                    f'Train Epoch: {epoch} '
                    f'[{batch_idx * len(data)}/{len(self.train_loader.dataset)} '
                    f'({100.0 * batch_idx / len(self.train_loader):.0f}%)]\t'
                    f'Loss: {batch_loss:.6f}')

        return total_loss / len(self.train_loader)

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
        metrics = torch.tensor([test_loss, correct], device=self.device)
        dist.all_reduce(metrics)

        # Compute final metrics
        test_loss = metrics[0].item() / len(self.test_loader.dataset)
        correct = metrics[1].item()
        accuracy = 100.0 * correct / len(self.test_loader.dataset)

        # Log results on primary process
        if self.global_rank == 0:
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
                if self.global_rank == 0:
                    logger.info(f'Epoch {epoch} Loss: {epoch_loss:.4f}')

                # Synchronize processes
                dist.barrier()

                # Perform testing
                test_metrics = self.test()

                if self.global_rank == 0:
                    logger.info(f'Epoch {epoch}, Eval Metrics: {test_metrics}')

                # Update learning rate
                self.scheduler.step()

            # Optional model saving
            if self.global_rank == 0 and self.args.save_model:
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
        if self.global_rank != 0:
            return None

        checkpoint_path = path or f'checkpoint_epoch_{epoch}.pt'

        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
            },
            checkpoint_path,
        )

        logger.info(f'Epoch {epoch} | Checkpoint saved at {checkpoint_path}')
        return checkpoint_path


def prepare_data(args: TrainingArguments) -> Tuple[DataLoader, DataLoader]:
    """Prepare distributed datasets and data loaders for training.

    Args:
        args (TrainingArguments): Parsed command-line arguments

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

    # Create distributed samplers
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


def setup_training_environment(args: TrainingArguments) -> None:
    """Configure training environment settings.

    Args:
        args: Command-line arguments containing configuration

    Note:
        Sets random seed and CUDNN benchmark mode
    """
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    logger.info(f'Set random seed to {args.seed}')


def validate_gpu_requirements() -> int:
    """Validate GPU requirements for distributed training.

    Returns:
        int: Number of available GPUs

    Raises:
        RuntimeError: If fewer than 2 GPUs are available
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
    """Main entry point for distributed training workflow.

    Manages distributed setup, training initialization, and execution.
    """
    # Log system information
    get_system_info()

    # Initialize training configuration
    args: TrainingArguments = tyro.cli(TrainingArguments)

    setup_training_environment(args)

    # Validate GPU availability
    validate_gpu_requirements()

    try:
        # Setup distributed environment
        setup_distributed_environment()

        # Prepare data loaders
        train_loader, test_loader = prepare_data(args)

        # Initialize model (assuming Net is imported from lenet)
        model = LeNet()

        # Setup optimizer and learning rate scheduler
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
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
    finally:
        # Cleanup distributed resources
        cleanup_distribute_environment()


if __name__ == '__main__':
    main()
