import argparse
import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from scaletorch.utils.net_utils import LeNet


class Trainer:
    """A comprehensive trainer class for PyTorch model training and evaluation.

    This class manages the entire training process, including:
    - Batch processing
    - Epoch training
    - Model testing
    - Checkpoint saving
    """

    def __init__(
        self,
        args: argparse.Namespace,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize the Trainer with model, data, and training parameters.

        Args:
            args (argparse.Namespace): Command-line arguments and configurations
            model (nn.Module): Neural network model to train
            train_loader (DataLoader): DataLoader for training data
            test_loader (DataLoader): DataLoader for test data
            optimizer (torch.optim.Optimizer): Optimization algorithm
            scheduler (torch.optim.lr_scheduler.LRScheduler): Learning rate scheduler
            device (Optional[torch.device], optional): Training device. Defaults to None.
        """
        self.args = args
        self.device = device or torch.device('cpu')
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        self.logger = logging.getLogger(__name__)

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
            path (Optional[str], optional): Custom checkpoint path

        Returns:
            str: Path where checkpoint was saved
        """
        # Use provided path or generate default
        checkpoint_path = path or f'checkpoint_epoch_{epoch}.pt'

        # Save comprehensive checkpoint
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
            },
            checkpoint_path,
        )

        self.logger.info(
            f'Epoch {epoch} | Checkpoint saved at {checkpoint_path}')
        return checkpoint_path


def parse_arguments() -> argparse.Namespace:
    """Configure and parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description='PyTorch MNIST Training')

    # Training configuration arguments
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
                        default=14,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1.0, help='Learning rate')
    parser.add_argument('--gamma',
                        type=float,
                        default=0.7,
                        help='Learning rate decay')

    # Device selection arguments
    parser.add_argument('--no-cuda',
                        action='store_true',
                        help='Disable CUDA training')
    parser.add_argument('--no-mps',
                        action='store_true',
                        help='Disable macOS GPU training')

    # Utility arguments
    parser.add_argument('--dry-run',
                        action='store_true',
                        help='Quick training check')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--log-interval',
                        type=int,
                        default=100,
                        help='Logging frequency')
    parser.add_argument('--save-model',
                        action='store_true',
                        help='Save trained model')
    parser.add_argument('--data-path',
                        type=str,
                        default='./data',
                        help='Dataset download path')

    return parser.parse_args()


def main() -> None:
    """Main function to set up and execute model training.

    Handles device selection, data loading, and trainer initialization.
    """
    # Parse command-line arguments
    args = parse_arguments()

    # Determine training device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)

    # Select device
    device = (torch.device('cuda') if use_cuda else
              torch.device('mps') if use_mps else torch.device('cpu'))

    # Configure data loading
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}

    if use_cuda:
        cuda_kwargs = {'num_workers': 1, 'pin_memory': True, 'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # Prepare data transformations
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

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    # Initialize model, optimizer, and scheduler
    model = LeNet().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # Create trainer and start training
    trainer = Trainer(args, model, train_loader, test_loader, optimizer,
                      scheduler, device)
    trainer.train()


if __name__ == '__main__':
    main()
