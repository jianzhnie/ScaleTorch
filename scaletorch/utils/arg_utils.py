from dataclasses import asdict, dataclass, field
from typing import Any, Dict


@dataclass
class TrainingArguments:
    """Arguments pertaining to MNIST distributed training configuration."""

    batch_size: int = field(
        default=64,
        metadata={'help': 'Training batch size'},
    )
    test_batch_size: int = field(
        default=1000,
        metadata={'help': 'Test batch size'},
    )
    epochs: int = field(
        default=5,
        metadata={'help': 'Number of training epochs'},
    )
    lr: float = field(
        default=1.0,
        metadata={'help': 'Learning rate'},
    )
    gamma: float = field(
        default=0.7,
        metadata={'help': 'Learning rate decay factor'},
    )
    dry_run: bool = field(
        default=False,
        metadata={'help': 'Quick training verification'},
    )
    seed: int = field(
        default=1,
        metadata={'help': 'Random seed for reproducibility'},
    )
    log_interval: int = field(
        default=100,
        metadata={'help': 'Batch logging frequency'},
    )
    save_model: bool = field(
        default=False,
        metadata={'help': 'Save trained model checkpoint'},
    )
    data_path: str = field(
        default='./data',
        metadata={'help': 'Dataset download directory'},
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
