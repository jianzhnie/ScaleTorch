"""ScaleTorch argument dataclasses with HfArgumentParser CLI support."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Optional, Tuple

from transformers import AutoConfig, HfArgumentParser

from scaletorch.utils.logger_utils import get_logger

logger = get_logger(__name__)

_VALID_LR_SCHEDULERS = ('linear', 'cosine', 'polynomial', 'step', 'onecycle')
_VALID_OPTIMIZERS = ('adamw', 'sgd', 'adam', 'lamb')

__all__ = [
    'DataArguments',
    'ModelArguments',
    'ParallelArguments',
    'LrSchedulerArguments',
    'OptimizerArguments',
    'TrainingArguments',
    'CheckpointArguments',
    'LoggingArguments',
    'ScaleTorchArguments',
]


@dataclass
class DataArguments:
    """Data loading and processing arguments."""
    data_path: str = field(
        default='./data',
        metadata={'help': 'Dataset download directory'},
    )
    dataset_name: str = field(
        default='wikitext2',
        metadata={'help': 'Dataset name'},
    )
    tokenizer_name_or_path: str = field(
        default='facebook/opt-125m',
        metadata={'help': 'Tokenizer name or path'},
    )
    subset_name: Optional[str] = field(
        default=None,
        metadata={'help': 'Subset name'},
    )
    split: str = field(
        default='train',
        metadata={'help': 'Split name'},
    )
    num_proc: int = field(
        default=1,
        metadata={'help': 'Number of processes for data processing'},
    )
    num_workers: int = field(
        default=2,
        metadata={'help': 'Number of DataLoader workers'},
    )
    num_samples: Optional[int] = field(
        default=None,
        metadata={'help': 'Number of samples to use (None for full dataset)'},
    )
    pin_memory: bool = field(
        default=True,
        metadata={'help': 'Whether to pin memory'},
    )


@dataclass
class ModelArguments:
    """Model configuration arguments. None fields auto-populated from HuggingFace config."""

    model_name_or_path: str = field(
        default='facebook/opt-125m',
        metadata={'help': 'Model name or path'},
    )
    num_hidden_layers: Optional[int] = field(
        default=None,
        metadata={'help': 'Number of hidden layers'},
    )
    num_attention_heads: Optional[int] = field(
        default=None,
        metadata={'help': 'Number of attention heads'},
    )
    num_key_value_heads: Optional[int] = field(
        default=None,
        metadata={'help': 'Number of key-value heads'},
    )
    use_flash_attention: bool = field(
        default=True,
        metadata={'help': 'Whether to use flash attention'},
    )
    dtype: str = field(
        default='bfloat16',
        metadata={'help': 'Data type'},
    )

    def __post_init__(self) -> None:
        """Auto-fill None fields from HuggingFace model config."""
        try:
            hf_config = AutoConfig.from_pretrained(self.model_name_or_path)
            if self.num_hidden_layers is None:
                self.num_hidden_layers = getattr(hf_config, 'num_hidden_layers', None)
            if self.num_attention_heads is None:
                self.num_attention_heads = getattr(hf_config, 'num_attention_heads', None)
            if self.num_key_value_heads is None:
                self.num_key_value_heads = getattr(hf_config, 'num_key_value_heads', None)
        except Exception as e:
            logger.warning(
                "Could not load AutoConfig for '%s': %s", self.model_name_or_path, e
            )


@dataclass
class ParallelArguments:
    """Distributed parallelism configuration. Total GPUs = TP * PP * DP * CP."""
    tensor_parallel_size: int = field(
        default=1,
        metadata={'help': 'Size of tensor parallelism'},
    )
    pipeline_parallel_size: int = field(
        default=1,
        metadata={'help': 'Size of pipeline parallelism'},
    )
    data_parallel_size: int = field(
        default=1,
        metadata={'help': 'Size of data parallelism'},
    )
    context_parallel_size: int = field(
        default=1,
        metadata={'help': 'Size of context parallelism'},
    )
    expert_parallel_size: int = field(  # TODO: implement
        default=1,
        metadata={'help': 'Size of expert parallelism'},
    )
    pipeline_parallel_engine: str = field(
        default='1f1b',
        metadata={'help': 'Pipeline parallel engine type (1f1b or afab)'},
    )
    backend: str = field(
        default='nccl',
        metadata={'help': 'Distributed backend (nccl, gloo, etc.)'},
    )

    def __post_init__(self) -> None:
        """Validate parallelism sizes, engine, and backend."""
        for name, val in [
            ('data_parallel_size', self.data_parallel_size),
            ('tensor_parallel_size', self.tensor_parallel_size),
            ('pipeline_parallel_size', self.pipeline_parallel_size),
            ('context_parallel_size', self.context_parallel_size),
        ]:
            if val < 1:
                raise ValueError(f'{name} must be >= 1, got {val}')
        if self.pipeline_parallel_engine not in {'1f1b', 'afab'}:
            raise ValueError(
                f'pipeline_parallel_engine must be "1f1b" or "afab", '
                f'got {self.pipeline_parallel_engine}')
        if self.backend not in {'nccl', 'gloo', 'hccl'}:
            raise ValueError(f'backend must be one of {{nccl, gloo, hccl}}, '
                             f'got {self.backend}')


@dataclass
class LrSchedulerArguments:
    """LR scheduler config. Types: linear, cosine, polynomial, step, onecycle."""
    lr_scheduler_type: str = field(
        default='linear',
        metadata={'help': 'Type of learning rate scheduler'},
    )
    warmup_steps: int = field(
        default=0,
        metadata={'help': 'Number of warmup steps'},
    )
    T_max: Optional[int] = field(
        default=None,
        metadata={'help': 'Maximum number of iterations for cosine annealing'},
    )
    eta_min: float = field(
        default=0.0,
        metadata={'help': 'Minimum learning rate for cosine annealing'},
    )
    power: float = field(
        default=1.0,
        metadata={'help': 'Power factor for polynomial decay'},
    )
    step_size: int = field(
        default=1,
        metadata={'help': 'Step size for step decay'},
    )
    gamma: float = field(
        default=0.1,
        metadata={'help': 'Multiplicative factor for step decay'},
    )
    max_lr: Optional[float] = field(
        default=None,
        metadata={'help': 'Upper learning rate boundary for OneCycleLR'},
    )
    pct_start: float = field(
        default=0.3,
        metadata={
            'help':
            'Percentage of the cycle spent increasing LR for OneCycleLR'
        },
    )

    def __post_init__(self) -> None:
        """Validate scheduler type and parameters."""
        if self.lr_scheduler_type not in _VALID_LR_SCHEDULERS:
            raise ValueError(
                f'lr_scheduler_type must be one of {_VALID_LR_SCHEDULERS}, '
                f'got {self.lr_scheduler_type}')
        if self.warmup_steps < 0:
            raise ValueError(
                f'warmup_steps must be >= 0, got {self.warmup_steps}')

        t = self.lr_scheduler_type
        if t == 'cosine' and self.eta_min < 0:
            raise ValueError(f'eta_min must be >= 0, got {self.eta_min}')
        if t == 'polynomial' and self.power <= 0:
            raise ValueError(f'power must be > 0, got {self.power}')
        if t == 'step':
            if self.step_size <= 0:
                raise ValueError(
                    f'step_size must be > 0, got {self.step_size}')
            if self.gamma <= 0 or self.gamma > 1:
                raise ValueError(f'gamma must be in (0, 1], got {self.gamma}')
        if t == 'onecycle' and not 0 < self.pct_start < 1:
            raise ValueError(
                f'pct_start must be in (0, 1), got {self.pct_start}')


@dataclass
class OptimizerArguments:
    """Optimizer configuration. Supported: adamw, sgd, adam, lamb."""
    optimizer_type: str = field(
        default='adamw',
        metadata={'help': 'Type of optimizer'},
    )
    weight_decay: float = field(
        default=0.0,
        metadata={'help': 'Weight decay for optimizer'},
    )
    use_fused_adam: bool = field(
        default=True,
        metadata={'help': 'Whether to use fused AdamW if available'},
    )
    betas: Tuple[float, float] = field(
        default=(0.9, 0.999),
        metadata={'help': 'Betas for Adam optimizer'},
    )
    learning_rate: float = field(
        default=1e-3,
        metadata={'help': 'Learning rate for optimizer'},
    )

    def __post_init__(self) -> None:
        if self.optimizer_type not in _VALID_OPTIMIZERS:
            raise ValueError(
                f'optimizer_type must be one of {_VALID_OPTIMIZERS}, got {self.optimizer_type}'
            )


@dataclass
class TrainingArguments:
    """Training hyperparameters. Global batch = DP * micro_batch * grad_accum."""

    batch_size: int = field(
        default=64,
        metadata={'help': 'Training batch size'},
    )
    test_batch_size: int = field(  # TODO: implement
        default=1000,
        metadata={'help': 'Test batch size'},
    )
    micro_batch_size: Optional[int] = field(
        default=None,
        metadata={'help': 'Micro batch size'},
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={'help': 'Number of gradient accumulation steps'},
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={'help': 'Whether to use gradient checkpointing'},
    )
    max_grad_norm: Optional[float] = field(
        default=1.0,
        metadata={'help': 'Max gradient norm for clipping (None to disable)'},
    )
    epochs: int = field(
        default=5,
        metadata={'help': 'Number of training epochs'},
    )
    seed: int = field(
        default=1,
        metadata={'help': 'Random seed for reproducibility'},
    )
    sequence_length: Optional[int] = field(
        default=1024,
        metadata={'help': 'Sequence length for training'},
    )
    log_interval: int = field(
        default=100,
        metadata={'help': 'Batch logging frequency'},
    )
    max_tokens: Optional[int] = field(
        default=None,
        metadata={'help': 'Max tokens to train on (overrides epochs/steps)'},
    )
    total_train_steps: Optional[int] = field(
        default=None,
        metadata={'help': 'Total training steps (overrides epochs)'},
    )
    use_cpu: bool = field(
        default=False,
        metadata={'help': 'Whether to run training on CPU'},
    )

    def __post_init__(self) -> None:
        """Validate training parameters."""
        if self.gradient_accumulation_steps < 1:
            raise ValueError(f'gradient_accumulation_steps must be >= 1, '
                             f'got {self.gradient_accumulation_steps}')
        if self.micro_batch_size is not None and self.micro_batch_size < 1:
            raise ValueError(
                f'micro_batch_size must be >= 1, got {self.micro_batch_size}')
        if self.sequence_length is not None and self.sequence_length < 1:
            raise ValueError(
                f'sequence_length must be >= 1, got {self.sequence_length}')


@dataclass
class CheckpointArguments:
    """Checkpointing configuration."""
    work_dir: str = field(
        default='./work_dir',
        metadata={'help': 'Directory to save checkpoints'},
    )
    save_model_checkpoint: bool = field(
        default=True,
        metadata={'help': 'Whether to save model checkpoints'},
    )
    save_frequency: int = field(
        default=300,
        metadata={'help': 'Frequency of saving checkpoints (in steps)'},
    )
    resume_path: str = field(
        default='',
        metadata={'help': 'Path to resume checkpoint from'},
    )

    def __post_init__(self) -> None:
        if self.save_frequency < 0:
            raise ValueError(
                f'save_frequency must be >= 0, got {self.save_frequency}')


@dataclass
class LoggingArguments:
    """Experiment logging configuration (wandb)."""
    use_wandb: bool = field(
        default=False,
        metadata={'help': 'Whether to use Weights & Biases logging'},
    )
    project_name: str = field(
        default='scaletorch',
        metadata={'help': 'Project name for logging'},
    )
    experiment_name: Optional[str] = field(
        default=None,
        metadata={'help': 'Experiment name for logging'},
    )


@dataclass
class ScaleTorchArguments(
        DataArguments,
        ModelArguments,
        ParallelArguments,
        LrSchedulerArguments,
        OptimizerArguments,
        TrainingArguments,
        CheckpointArguments,
        LoggingArguments,
):
    """Aggregated arguments for ScaleTorch distributed training."""

    # Computed attributes (not parsed from command line)
    global_batch_size: int = field(init=False, default=0)
    global_batch_size_token: Optional[int] = field(init=False, default=None)

    def __post_init__(self) -> None:
        """Validate all parent args and compute global batch size."""
        ModelArguments.__post_init__(self)
        ParallelArguments.__post_init__(self)
        LrSchedulerArguments.__post_init__(self)
        OptimizerArguments.__post_init__(self)
        TrainingArguments.__post_init__(self)
        CheckpointArguments.__post_init__(self)

        if self.micro_batch_size is None:
            self.micro_batch_size = self.batch_size
            logger.info(
                'micro_batch_size not provided, using batch_size: %d', self.batch_size
            )

        if (self.sequence_length and self.context_parallel_size
                and self.sequence_length % self.context_parallel_size != 0):
            raise ValueError(
                f'sequence_length ({self.sequence_length}) must be divisible by '
                f'context_parallel_size ({self.context_parallel_size})')

        self.global_batch_size = (self.data_parallel_size *
                                  self.micro_batch_size *
                                  self.gradient_accumulation_steps)

        if self.sequence_length is not None:
            self.global_batch_size_token = (self.global_batch_size *
                                            self.sequence_length)

    def validate_world_size(self, world_size: int) -> None:
        """Check world_size matches TP*PP*DP*CP. Call after dist init."""
        expected = (self.tensor_parallel_size * self.pipeline_parallel_size *
                    self.data_parallel_size * self.context_parallel_size)
        if world_size != expected:
            raise ValueError(
                f'world_size ({world_size}) != TP({self.tensor_parallel_size}) * '
                f'PP({self.pipeline_parallel_size}) * '
                f'DP({self.data_parallel_size}) * '
                f'CP({self.context_parallel_size}) = {expected}')
        logger.info('Configuration validation passed')


def main() -> None:
    """Demo: parse ScaleTorchArguments from CLI and log them."""
    parser = HfArgumentParser(ScaleTorchArguments)
    args, = parser.parse_args_into_dataclasses()
    logger.info(json.dumps(dataclasses.asdict(args), indent=4))
    logger.info('Global batch size: %d', args.global_batch_size)
    if args.global_batch_size_token is not None:
        logger.info('Global batch size (tokens): %d', args.global_batch_size_token)


if __name__ == '__main__':
    main()
