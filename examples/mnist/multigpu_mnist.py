"""Multi-GPU MNIST Training Example using ScaleTorch with mp.spawn."""

import dataclasses
import json
import os

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

from scaletorch.models.lenet import LeNet
from scaletorch.trainer.config import ScaleTorchArguments
from scaletorch.utils import get_system_info
from scaletorch.utils.logger_utils import get_logger

logger = get_logger(__name__)


class DistributedTrainer:
    """Distributed trainer using DDP for MNIST classification."""

    def __init__(
        self,
        args: ScaleTorchArguments,
        rank: int,
        world_size: int,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
    ) -> None:
        self.args = args
        self.rank = rank
        self.world_size = world_size

        self.device = torch.device(f"cuda:{rank}")
        model = model.to(self.device)

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
                logger.warning("Could not create DDP with device_ids: %s", e)
                self.model = DDP(model)
        else:
            self.model = model

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

        if hasattr(self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(epoch)

        total_loss = 0.0
        num_batches = len(self.train_loader)

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            batch_loss = self.run_batch(data, target)
            total_loss += batch_loss

            if self.rank == 0 and batch_idx % self.args.log_interval == 0:
                logger.info(
                    "Train Epoch: %d [%d/%d (%.0f%%)]\tLoss: %.6f",
                    epoch,
                    batch_idx * len(data),
                    len(self.train_loader.dataset),
                    100.0 * batch_idx / len(self.train_loader),
                    batch_loss,
                )

        return total_loss / num_batches if num_batches > 0 else 0.0

    def test(self) -> dict[str, float]:
        self.model.eval()
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction="sum").item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        if dist.is_initialized() and dist.get_world_size() > 1:
            metrics = torch.tensor([test_loss, correct], device=self.device)
            dist.all_reduce(metrics)
            test_loss = metrics[0].item() / len(self.test_loader.dataset)
            correct = metrics[1].item()
        else:
            test_loss = test_loss / len(self.test_loader.dataset)

        accuracy = 100.0 * correct / len(self.test_loader.dataset)

        if self.rank == 0:
            logger.info(
                "\nTest set: Average loss: %.4f, Accuracy: %d/%d (%.0f%%)\n",
                test_loss,
                correct,
                len(self.test_loader.dataset),
                accuracy,
            )

        return {"loss": test_loss, "accuracy": accuracy}

    def train(self) -> None:
        try:
            for epoch in range(1, self.args.epochs + 1):
                epoch_loss = self.run_epoch(epoch)

                if self.rank == 0:
                    logger.info("Epoch %d Loss: %.4f", epoch, epoch_loss)

                if dist.is_initialized() and dist.get_world_size() > 1:
                    dist.barrier()

                test_metrics = self.test()
                if self.rank == 0:
                    logger.info("Epoch %d, Eval Metrics: %s", epoch, test_metrics)

                self.scheduler.step()

            if self.rank == 0 and self.args.save_model_checkpoint:
                self.save_checkpoint(self.args.epochs)

        except Exception as e:
            logger.error("Training failed: %s", e)
            raise

    def save_checkpoint(self, epoch: int, path: str | None = None) -> str | None:
        if self.rank != 0:
            return None

        checkpoint_path = path or f"checkpoint_epoch_{epoch}.pt"

        state_dict = {
            "epoch": epoch,
            "model_state_dict": self.model.module.state_dict()
            if isinstance(self.model, DDP)
            else self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }

        try:
            torch.save(state_dict, checkpoint_path)
            logger.info("Epoch %d | Checkpoint saved at %s", epoch, checkpoint_path)
        except Exception as e:
            logger.error("Failed to save checkpoint: %s", e)
            return None

        return checkpoint_path


def prepare_data(args: ScaleTorchArguments) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        root=args.data_path, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=args.data_path, train=False, download=True, transform=transform
    )

    if dist.is_initialized() and dist.get_world_size() > 1:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
    else:
        train_sampler = None
        test_sampler = None

    use_cuda = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=4,
        pin_memory=use_cuda,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        sampler=test_sampler,
        shuffle=False,
        num_workers=4,
        pin_memory=use_cuda,
    )

    return train_loader, test_loader


def train_process(rank: int, world_size: int, args: ScaleTorchArguments) -> None:
    get_system_info()
    logger.info("Distributed training started on rank %d", rank)
    try:
        # mp.spawn doesn't set env vars — set them before init
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank)

        from scaletorch.utils.env_utils import init_dist_pytorch

        init_dist_pytorch()

        train_loader, test_loader = prepare_data(args)

        model = LeNet()
        optimizer = optim.Adadelta(model.parameters(), lr=args.learning_rate)
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

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

        trainer.train()

    except Exception as e:
        logger.error("Training failed on rank %d: %s", rank, e)
        raise


def main() -> None:
    parser = HfArgumentParser(ScaleTorchArguments)
    (args,) = parser.parse_args_into_dataclasses()

    logger.info(json.dumps(dataclasses.asdict(args), indent=4))

    num_device = torch.cuda.device_count()
    world_size = int(os.environ.get("WORLD_SIZE", num_device))
    if world_size <= 0:
        world_size = 1

    mp.spawn(train_process, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
