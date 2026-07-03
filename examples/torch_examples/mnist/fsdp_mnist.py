"""FSDP MNIST Training Example using ScaleTorch."""

import dataclasses
import json

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
from transformers import HfArgumentParser

from scaletorch.models.lenet import LeNet
from scaletorch.trainer.config import ScaleTorchArguments
from scaletorch.utils import cleanup_dist, get_system_info, init_dist_pytorch
from scaletorch.utils.logger_utils import get_logger

logger = get_logger(__name__)


class FSDPTrainer:
    """Fully Sharded Data Parallel (FSDP) Trainer for MNIST classification."""

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
        torch.cuda.set_device(self.device)
        self.model = FSDP(model.to(self.device))

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.init_start_event = torch.cuda.Event(enable_timing=True)
        self.init_end_event = torch.cuda.Event(enable_timing=True)

    def train_epoch(self, epoch: int) -> None:
        self.model.train()
        ddp_loss = torch.zeros(2).to(self.device)

        if hasattr(self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(epoch)

        for data, target in self.train_loader:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target, reduction="sum")
            loss.backward()
            self.optimizer.step()

            ddp_loss[0] += loss.item()
            ddp_loss[1] += len(data)

        torch.distributed.all_reduce(ddp_loss, op=torch.distributed.ReduceOp.SUM)

        if self.rank == 0:
            logger.info(
                "Train Epoch: %d \tLoss: %.6f", epoch, ddp_loss[0] / ddp_loss[1]
            )

    def test(self) -> None:
        self.model.eval()
        ddp_loss = torch.zeros(3).to(self.device)

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)

                ddp_loss[0] += F.nll_loss(output, target, reduction="sum").item()
                pred = output.argmax(dim=1, keepdim=True)
                ddp_loss[1] += pred.eq(target.view_as(pred)).sum().item()
                ddp_loss[2] += len(data)

        torch.distributed.all_reduce(ddp_loss, op=torch.distributed.ReduceOp.SUM)

        if self.rank == 0:
            test_loss = ddp_loss[0] / ddp_loss[2]
            logger.info(
                "Test set: Average loss: %.4f, Accuracy: %d/%d (%.2f%%)",
                test_loss,
                int(ddp_loss[1]),
                int(ddp_loss[2]),
                100.0 * ddp_loss[1] / ddp_loss[2],
            )

    def train(self) -> None:
        self.init_start_event.record()

        for epoch in range(1, self.args.epochs + 1):
            self.train_epoch(epoch)
            self.test()
            self.scheduler.step()

        self.init_end_event.record()

        if self.rank == 0:
            logger.info(
                "CUDA event elapsed time: %.3fsec",
                self.init_start_event.elapsed_time(self.init_end_event) / 1000,
            )
            logger.info("%s", self.model)

    def save_model(self) -> None:
        if self.args.save_model_checkpoint:
            torch.distributed.barrier()
            states = self.model.state_dict()
            if self.rank == 0:
                torch.save(states, "mnist_cnn.pt")


def prepare_data(
    args: ScaleTorchArguments, rank: int, world_size: int
) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        root=args.data_path, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=args.data_path, train=False, download=True, transform=transform
    )

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    test_sampler = DistributedSampler(
        test_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )

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


def main(rank: int, world_size: int, args: ScaleTorchArguments) -> None:
    try:
        get_system_info()
        init_dist_pytorch()

        train_loader, test_loader = prepare_data(args, rank, world_size)

        model = LeNet()
        optimizer = torch.optim.Adadelta(model.parameters(), lr=args.learning_rate)
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

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
        logger.error("Training failed on rank %d: %s", rank, e)
        raise
    finally:
        cleanup_dist()


if __name__ == "__main__":
    parser = HfArgumentParser(ScaleTorchArguments)
    (args,) = parser.parse_args_into_dataclasses()
    logger.info(json.dumps(dataclasses.asdict(args), indent=4))

    torch.manual_seed(args.seed)
    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(main, args=(WORLD_SIZE, args), nprocs=WORLD_SIZE, join=True)
