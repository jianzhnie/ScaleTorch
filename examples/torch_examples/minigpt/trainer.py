import io
import os
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urlparse

import boto3
import fsspec
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from scaletorch.utils import get_current_device


@dataclass
class TrainerConfig:
    """Configuration class for distributed training parameters."""

    max_epochs: int | None = None
    batch_size: int | None = None
    data_loader_workers: int = 0
    grad_norm_clip: float = 1.0
    snapshot_path: str | None = "snapshot.pt"
    save_every: int = 1
    use_amp: bool = False


@dataclass
class Snapshot:
    """Represents a training snapshot for model resumption."""

    model_state: OrderedDict[str, torch.Tensor]
    optimizer_state: dict[str, Any]
    finished_epoch: int


class Trainer:
    """Distributed training utility with DDP, AMP, and snapshot support."""

    def __init__(
        self,
        trainer_config: TrainerConfig,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_dataset: Dataset,
        test_dataset: Dataset | None = None,
    ) -> None:
        if not all(key in os.environ for key in ["LOCAL_RANK", "RANK"]):
            raise RuntimeError("Distributed environment not initialized. Use torchrun.")

        self.config = trainer_config
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])

        self.train_dataset = train_dataset
        self.train_loader = self._prepare_dataloader(train_dataset)
        self.test_loader = (
            self._prepare_dataloader(test_dataset) if test_dataset else None
        )

        self.epochs_run = 0
        self.device = get_current_device()
        self.model = model.to(self.device)
        self.optimizer = optimizer

        if self.config.use_amp:
            device_type = "cuda" if self.device.type == "cuda" else "cpu"
            self.scaler = torch.amp.GradScaler(device_type)
        else:
            self.scaler = None

        self._load_snapshot()
        self.model = DDP(self.model, device_ids=[self.local_rank])

    def _prepare_dataloader(self, dataset: Dataset | None) -> DataLoader | None:
        if not dataset:
            return None

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size or 1,
            pin_memory=True,
            shuffle=False,
            num_workers=self.config.data_loader_workers,
            sampler=DistributedSampler(dataset),
        )

    def _load_snapshot(self) -> None:
        try:
            with fsspec.open(self.config.snapshot_path) as f:
                snapshot_data = torch.load(f, map_location="cpu", weights_only=False)
        except FileNotFoundError:
            print("No snapshot found. Starting training from scratch.")
            return

        try:
            snapshot = Snapshot(**snapshot_data)
            self.model.load_state_dict(snapshot.model_state)
            self.optimizer.load_state_dict(snapshot.optimizer_state)
            self.epochs_run = snapshot.finished_epoch
            print(f"Resumed training from snapshot at Epoch {self.epochs_run}")
        except Exception as e:
            print(f"Error loading snapshot: {e}. Starting from scratch.")

    def _run_batch(
        self, source: torch.Tensor, targets: torch.Tensor, train: bool = True
    ) -> float:
        device_type = self.device.type if self.device.type != "mps" else "cpu"
        with (
            torch.set_grad_enabled(train),
            torch.amp.autocast(
                device_type=device_type,
                dtype=torch.float16,
                enabled=bool(self.config.use_amp),
            ),
        ):
            _, loss = self.model(source, targets)

        if train:
            self.optimizer.zero_grad(set_to_none=True)

            if self.config.use_amp and self.scaler:
                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_norm_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_norm_clip
                )
                self.optimizer.step()

        return loss.item()

    def _run_epoch(
        self, epoch: int, dataloader: DataLoader, train: bool = True
    ) -> None:
        dataloader.sampler.set_epoch(epoch)

        for batch_idx, (source, targets) in enumerate(dataloader):
            step_type = "Train" if train else "Eval"
            source = source.to(self.device)
            targets = targets.to(self.device)

            batch_loss = self._run_batch(source, targets, train)

            if batch_idx % 100 == 0:
                print(
                    f"[GPU{self.global_rank}] "
                    f"Epoch {epoch} | Iter {batch_idx} | "
                    f"{step_type} Loss {batch_loss:.5f}"
                )

    def _save_snapshot(self, epoch: int) -> None:
        raw_model = self.model.module if hasattr(self.model, "module") else self.model

        snapshot = Snapshot(
            model_state=raw_model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            finished_epoch=epoch,
        )

        snapshot_dict = asdict(snapshot)
        try:
            if self.config.snapshot_path.startswith("s3://"):
                self._upload_to_s3(snapshot_dict, self.config.snapshot_path)
            else:
                torch.save(snapshot_dict, self.config.snapshot_path)
            print(f"Snapshot saved at epoch {epoch}")
        except Exception as e:
            print(f"Failed to save snapshot: {e}")

    def _upload_to_s3(self, obj: dict[str, Any], dst: str) -> None:
        buffer = io.BytesIO()
        torch.save(obj, buffer)
        buffer.seek(0)

        parsed_url = urlparse(dst, allow_fragments=False)
        boto3.client("s3").upload_fileobj(
            buffer, parsed_url.netloc, parsed_url.path.lstrip("/")
        )

    def train(self) -> None:
        for epoch in range(self.epochs_run, self.config.max_epochs or 0):
            epoch += 1

            self._run_epoch(epoch, self.train_loader, train=True)

            if self.local_rank == 0 and epoch % self.config.save_every == 0:
                self._save_snapshot(epoch)

            if self.test_loader:
                self._run_epoch(epoch, self.test_loader, train=False)
