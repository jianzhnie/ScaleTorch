from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from scaletorch.utils.arg_utils import TrainingArguments
from scaletorch.utils.logger_utils import get_logger
from scaletorch.utils.net_utils import LeNet

logger = get_logger(__name__)


class Trainer:
    """A comprehensive trainer class for PyTorch model training and evaluation.

    This class manages the entire training process, including:
    - Batch processing
    - Epoch training
    - Model testing
    - Checkpoint saving and loading
    - Metrics tracking

    Attributes:
        args (TrainingArguments): Training configuration parameters
        device (torch.device): Device to run training on (CPU/GPU)
        model (nn.Module): Neural network model
        train_loader (DataLoader): Training data loader
        test_loader (DataLoader): Test data loader
        optimizer (torch.optim.Optimizer): Optimization algorithm
        scheduler (torch.optim.lr_scheduler.LRScheduler): Learning rate scheduler
        logger (Logger): Logger instance for tracking progress
    """

    def __init__(
        self,
        args: TrainingArguments,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,  # Fixed type hint
        device: Optional[torch.device] = None,
    ) -> None:
        self.args = args
        self.device = device or torch.device('cpu')
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Configure logging
        self.logger = logger

    def run_batch(self, source: torch.Tensor, targets: torch.Tensor) -> float:
        """Process a single training batch.

        Args:
            source (torch.Tensor): Input data tensor
            targets (torch.Tensor): Target labels tensor

        Returns:
            float: Computed loss value for the batch
        """
        # Reset gradients
        self.optimizer.zero_grad()

        # Forward pass
        output = self.model(source)

        # Compute loss using Negative Log-Likelihood
        loss = F.nll_loss(output, targets)

        # Backward pass and parameter update
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def run_epoch(self, epoch: int) -> float:
        """Train the model for one epoch.

        Args:
            epoch (int): Current epoch number

        Returns:
            float: Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            # Move data to specified device
            data, target = data.to(self.device), target.to(self.device)

            # Process batch and accumulate loss
            batch_loss = self.run_batch(data, target)
            total_loss += batch_loss

            # Log progress at specified intervals
            if batch_idx % self.args.log_interval == 0:
                self.logger.info(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch,
                        batch_idx * len(data),
                        len(self.train_loader.dataset),
                        100.0 * batch_idx / len(self.train_loader),
                        batch_loss,
                    ))

                # Stop after first batch if dry run
                if self.args.dry_run:
                    break

        return total_loss / len(self.train_loader)

    def test(self) -> Dict[str, float]:
        """Evaluate the model on test dataset.

        Returns:
            Dict[str, float]: Dictionary containing test loss and accuracy
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

        # Compute average metrics
        test_loss /= len(self.test_loader.dataset)
        accuracy = 100.0 * correct / len(self.test_loader.dataset)

        self.logger.info(
            '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.
            format(test_loss, correct, len(self.test_loader.dataset),
                   accuracy))

        return {'loss': test_loss, 'accuracy': accuracy}

    def train(self) -> None:
        """Execute complete model training process.

        Runs for specified number of epochs, testing after each epoch.
        """
        for epoch in range(1, self.args.epochs + 1):
            # Train for one epoch and log loss
            epoch_loss = self.run_epoch(epoch)

            # Log epoch loss on primary process (optional)
            self.logger.info(f'Epoch {epoch}, Train Loss: {epoch_loss:.4f}')

            # Perform testing
            test_metrics = self.test()

            # Log epoch loss on primary process (optional)
            self.logger.info(f'Epoch {epoch}, Eval Metrics: {test_metrics}')

            # Step learning rate scheduler
            self.scheduler.step()

        # Optional: Save trained model
        if self.args.save_model:
            self.save_checkpoint(self.args.epochs)

    def save_checkpoint(self, epoch: int, path: Optional[str] = None) -> str:
        """Save model checkpoint with training state.

        Args:
            epoch (int): Current epoch number
            path (Optional[str]): Custom checkpoint path. If None, generates default path

        Returns:
            str: Path where checkpoint was saved

        Raises:
            IOError: If unable to save checkpoint to specified path
        """
        checkpoint_path = path or f'checkpoint_epoch_{epoch}.pt'

        try:
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'training_args': self.args,  # Added training configuration
                },
                checkpoint_path,
            )
        except IOError as e:
            self.logger.error(f'Failed to save checkpoint: {e}')
            raise

        self.logger.info(
            f'Epoch {epoch} | Checkpoint saved at {checkpoint_path}')
        return checkpoint_path


def setup_device(args: TrainingArguments) -> torch.device:
    """Configure and return the appropriate device for training.

    Args:
        args: Command-line arguments containing device preferences

    Returns:
        torch.device: Selected device (cuda/mps/cpu)
    """
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    if not args.no_cuda and not torch.cuda.is_available():
        logger.warning('CUDA requested but not available. Using CPU instead.')

    device = torch.device('cuda') if use_cuda else torch.device('cpu')

    return device


def get_data_loaders(args: TrainingArguments,
                     use_cuda: bool) -> Tuple[DataLoader, DataLoader]:
    """Create and configure data loaders for training and testing.

    Args:
        args: Command-line arguments containing data loading parameters
        use_cuda: Whether CUDA is being used (affects DataLoader configuration)

    Returns:
        Tuple[DataLoader, DataLoader]: Training and test data loaders
    """
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}

    if use_cuda:
        cuda_kwargs = {'num_workers': 1, 'pin_memory': True, 'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307, ), (0.3081, ))])

    train_dataset = datasets.MNIST(root=args.data_path,
                                   train=True,
                                   download=True,
                                   transform=transform)
    test_dataset = datasets.MNIST(root=args.data_path,
                                  train=False,
                                  transform=transform)

    train_loader = DataLoader(train_dataset, **train_kwargs)
    test_loader = DataLoader(test_dataset, **test_kwargs)
    return train_loader, test_loader


def main() -> None:
    """Main function to set up and execute model training."""
    # Parse and validate arguments
    args: TrainingArguments = tyro.cli(TrainingArguments)

    # Set up device
    device = setup_device(args)
    logger.info(f'Using device: {device}')

    # Set up data loaders
    train_loader, test_loader = get_data_loaders(args, device.type == 'cuda')

    # Initialize model components
    model = LeNet().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # Create trainer and start training
    trainer = Trainer(args, model, train_loader, test_loader, optimizer,
                      scheduler, device)
    trainer.train()


if __name__ == '__main__':
    main()
