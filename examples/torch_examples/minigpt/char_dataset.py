from dataclasses import dataclass

import fsspec
import torch
from torch.utils.data import Dataset

from scaletorch.utils.logger_utils import get_logger

logger = get_logger(__name__)


@dataclass
class DataConfig:
    """Configuration class for character-level dataset parameters."""

    path: str | None = None
    block_size: int | None = None
    train_split: float | None = None
    truncate: float = 1.0
    encoding: str = "utf-8"


class CharDataset(Dataset):
    """A PyTorch Dataset for character-level text processing."""

    def __init__(self, data_cfg: DataConfig) -> None:
        if not data_cfg.path:
            raise ValueError("Data path must be provided")
        if not data_cfg.block_size:
            raise ValueError("Block size must be specified")

        try:
            with fsspec.open(data_cfg.path, "r", encoding=data_cfg.encoding) as file:
                data = file.read()
        except Exception as e:
            raise OSError("Error reading file %s: %s" % (data_cfg.path, e))

        data = data[: int(len(data) * data_cfg.truncate)]

        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        logger.info("Data has %d characters, %d unique.", data_size, vocab_size)

        self.stoi: dict[str, int] = {ch: i for i, ch in enumerate(chars)}
        self.itos: dict[int, str] = {i: ch for i, ch in enumerate(chars)}

        self.block_size: int = data_cfg.block_size
        self.vocab_size: int = vocab_size
        self.data: str = data

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.block_size + 1]

        try:
            dix = [self.stoi[s] for s in chunk]
        except KeyError as e:
            raise ValueError("Unknown character encountered: %s" % e)

        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)

        return x, y
