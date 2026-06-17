"""Single-GPU/CPU MNIST Training Example using ScaleTorch."""

import dataclasses
import json
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import HfArgumentParser

from scaletorch.trainer.config import ScaleTorchArguments
from scaletorch.models.lenet import LeNet
from scaletorch.utils import get_device, get_logger

logger = get_logger(__name__)


class Trainer:
    """Trainer for single-device MNIST training."""

    def __init__(
        self,
        args: ScaleTorchArguments,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        device: Optional[torch.device] = None,
    ) -> None:
        self.args = args
        self.device = device or torch.device('cpu')
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler

    def run_batch(self, source: torch.Tensor, targets: torch.Tensor) -> float:
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.nll_loss(output, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def run_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            batch_loss = self.run_batch(data, target)
            total_loss += batch_loss

            if batch_idx % self.args.log_interval == 0:
                logger.info(
                    'Train Epoch: %d [%d/%d (%.0f%%)]\tLoss: %.6f', epoch,
                    batch_idx * len(data),
                    len(self.train_loader.dataset),
                    100.0 * batch_idx / len(self.train_loader), batch_loss)

        return total_loss / len(self.train_loader)

    def test(self) -> Dict[str, float]:
        self.model.eval()
        metrics = {
            'loss': 0.0,
            'correct_predictions': 0,
            'total_samples': len(self.test_loader.dataset),  # type: ignore
        }

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)

                metrics['loss'] += F.nll_loss(output, target,
                                              reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                metrics['correct_predictions'] += (pred.eq(
                    target.view_as(pred)).sum().item())

        metrics['loss'] /= metrics['total_samples']
        metrics['accuracy'] = (100.0 * metrics['correct_predictions'] /
                               metrics['total_samples'])

        logger.info(
            '\nTest set: Average loss: %.4f, Accuracy: %d/%d (%.1f%%)',
            metrics['loss'], metrics['correct_predictions'],
            metrics['total_samples'], metrics['accuracy'])

        return metrics

    def train(self) -> None:
        for epoch in range(1, self.args.epochs + 1):
            epoch_loss = self.run_epoch(epoch)
            logger.info('Epoch %d, Train Loss: %.4f', epoch, epoch_loss)

            test_metrics = self.test()
            logger.info('Epoch %d, Eval Metrics: %s', epoch, test_metrics)

            self.scheduler.step()

        if self.args.save_model_checkpoint:
            self.save_checkpoint(self.args.epochs)

    def save_checkpoint(self, epoch: int, path: Optional[str] = None) -> str:
        checkpoint_path = path or f'checkpoint_epoch_{epoch}.pt'

        try:
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                },
                checkpoint_path,
            )
        except IOError as e:
            logger.error('Failed to save checkpoint: %s', e)
            raise

        logger.info('Epoch %d | Checkpoint saved at %s', epoch,
                     checkpoint_path)
        return checkpoint_path


def get_data_loaders(
        args: ScaleTorchArguments) -> Tuple[DataLoader, DataLoader]:
    """Create data loaders for training and testing."""
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307, ), (0.3081, ))])

    train_dataset = datasets.MNIST(root=args.data_path,
                                   train=True,
                                   download=True,
                                   transform=transform)
    test_dataset = datasets.MNIST(root=args.data_path,
                                  train=False,
                                  download=True,
                                  transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size)
    return train_loader, test_loader


def main() -> None:
    parser = HfArgumentParser(ScaleTorchArguments)
    args, = parser.parse_args_into_dataclasses()

    logger.info(json.dumps(dataclasses.asdict(args), indent=4))

    device = get_device()
    logger.info('Using device: %s', device)

    train_loader, test_loader = get_data_loaders(args)

    model = LeNet()
    optimizer = optim.Adadelta(model.parameters(), lr=args.learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    trainer = Trainer(args, model, train_loader, test_loader, optimizer,
                      scheduler, device)
    trainer.train()


if __name__ == '__main__':
    main()
