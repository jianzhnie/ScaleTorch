"""
Argument parsing and validation utilities for ScaleTorch.

This module provides dataclass-based argument definitions for ScaleTorch training,
including data loading, model configuration, parallelism settings, training parameters,
checkpointing, logging. It uses HuggingFace's HfArgumentParser
for command-line argument parsing with validation.

Example:
    ```python
    from scaletorch.configs.arg_utils import ScaleTorchArguments
    from transformers import HfArgumentParser

    parser = HfArgumentParser(ScaleTorchArguments)
    args, = parser.parse_args_into_dataclasses()
    ```
"""

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Optional

from transformers import AutoConfig, HfArgumentParser

from scaletorch.utils.logger_utils import get_logger

logger = get_logger(__name__)

__all__ = [
    'DataArguments',
    'ModelArguments',
    'LrSchedulerArguments',
    'ParallelArguments',
    'TrainingArguments',
    'CheckpointArguments',
    'LoggingArguments',
    'ScaleTorchArguments',
    'validate_parallelism_sizes',
    'validate_pipeline_engine',
    'validate_training_parameters',
]


def validate_parallelism_sizes(
    data_parallel_size: int,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    context_parallel_size: int,
) -> None:
    """
    Validate that all parallelism sizes are positive integers.

    Args:
        data_parallel_size: Size of data parallelism.
        tensor_parallel_size: Size of tensor parallelism.
        pipeline_parallel_size: Size of pipeline parallelism.
        context_parallel_size: Size of context parallelism.

    Raises:
        ValueError: If any parallelism size is less than 1.
    """
    if data_parallel_size < 1:
        raise ValueError(
            f'data_parallel_size must be >= 1, got {data_parallel_size}')
    if tensor_parallel_size < 1:
        raise ValueError(
            f'tensor_parallel_size must be >= 1, got {tensor_parallel_size}')
    if pipeline_parallel_size < 1:
        raise ValueError(
            f'pipeline_parallel_size must be >= 1, got {pipeline_parallel_size}'
        )
    if context_parallel_size < 1:
        raise ValueError(
            f'context_parallel_size must be >= 1, got {context_parallel_size}')


def validate_pipeline_engine(pipeline_parallel_engine: str) -> None:
    """
    Validate that the pipeline parallel engine is supported.

    Args:
        pipeline_parallel_engine: Pipeline parallel engine type.

    Raises:
        ValueError: If the engine is not one of the supported types.
    """
    supported_engines = {'1f1b', 'afab'}
    if pipeline_parallel_engine not in supported_engines:
        raise ValueError(
            f'pipeline_parallel_engine must be one of {supported_engines}, '
            f'got {pipeline_parallel_engine}')


def validate_training_parameters(
    gradient_accumulation_steps: int,
    micro_batch_size: Optional[int],
    sequence_length: Optional[int],
) -> None:
    """
    Validate training parameters.

    Args:
        gradient_accumulation_steps: Number of gradient accumulation steps.
        micro_batch_size: Micro batch size (optional).
        sequence_length: Sequence length (optional).

    Raises:
        ValueError: If any parameter is invalid.
    """
    if gradient_accumulation_steps < 1:
        raise ValueError(f'gradient_accumulation_steps must be >= 1, '
                         f'got {gradient_accumulation_steps}')
    if micro_batch_size is not None and micro_batch_size < 1:
        raise ValueError(
            f'micro_batch_size must be >= 1, got {micro_batch_size}')
    if sequence_length is not None and sequence_length < 1:
        raise ValueError(
            f'sequence_length must be >= 1, got {sequence_length}')


@dataclass
class DataArguments:
    """
    Arguments pertaining to data loading and processing.

    Attributes:
        data_path: Directory path where datasets are stored or downloaded.
        dataset_name: Name of the dataset to use (e.g., 'wikitext2', 'c4').
        tokenizer_name: Name or path of the tokenizer to use.
        subset_name: Optional subset name for datasets with multiple subsets.
        split: Dataset split to use ('train', 'validation', 'test').
        num_proc: Number of processes to use for data processing.
        pin_memory: Whether to pin memory in DataLoader for faster GPU transfer.
    """
    data_path: str = field(
        default='./data',
        metadata={'help': 'Dataset download directory'},
    )
    dataset_name: str = field(
        default='wikitext2',
        metadata={'help': 'Dataset name'},
    )
    tokenizer_name_or_path: str = field(
        default='openai/gpt2',
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
    pin_memory: bool = field(
        default=True,
        metadata={'help': 'Whether to pin memory'},
    )

    def __post_init__(self) -> None:
        """
        Validate data arguments.
        """
        pass


@dataclass
class ModelArguments:
    """
    Arguments pertaining to the model configuration.

    Attributes:
        model_name_or_path: HuggingFace model name or local path to model.
        num_hidden_layers: Number of hidden layers (auto-loaded if None).
        num_attention_heads: Number of attention heads (auto-loaded if None).
        num_key_value_heads: Number of key-value heads for GQA/MQA (auto-loaded if None).
        use_flash_attention: Whether to use Flash Attention for efficiency.
        dtype: Data type for model parameters ('float32', 'float16', 'bfloat16').

    Note:
        If model_name_or_path is a valid HuggingFace model, num_hidden_layers,
        num_attention_heads, and num_key_value_heads will be automatically loaded
        from the model config if not explicitly provided.
    """

    model_name_or_path: str = field(
        default='gpt2',
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
        """
        Load model configuration from HuggingFace if available.

        Attempts to load the model configuration and populate missing parameters.
        If loading fails, uses the provided values or keeps them as None.
        """
        # Store original values before attempting to load from HuggingFace
        original_num_hidden_layers = self.num_hidden_layers
        original_num_attention_heads = self.num_attention_heads
        original_num_key_value_heads = self.num_key_value_heads

        try:
            hf_config = AutoConfig.from_pretrained(self.model_name_or_path)
            logger.info(
                f'Successfully loaded config for model: {self.model_name_or_path}'
            )
            # Only override if not explicitly provided
            if self.num_hidden_layers is None:
                self.num_hidden_layers = getattr(hf_config,
                                                 'num_hidden_layers', None)
            if self.num_attention_heads is None:
                self.num_attention_heads = getattr(hf_config,
                                                   'num_attention_heads', None)
            if self.num_key_value_heads is None:
                self.num_key_value_heads = getattr(hf_config,
                                                   'num_key_value_heads', None)
        except Exception as e:
            logger.warning(
                f"Could not load AutoConfig for '{self.model_name_or_path}': {e}. "
                'Using provided values or None for model parameters.')
            # Keep original values (no need to reassign, but explicit for clarity)
            self.num_hidden_layers = original_num_hidden_layers
            self.num_attention_heads = original_num_attention_heads
            self.num_key_value_heads = original_num_key_value_heads


@dataclass
class LrSchedulerArguments:
    """
    Configuration arguments for learning rate scheduler.

    Attributes:
        lr_scheduler_type: Type of learning rate scheduler.
            Options include: 'linear', 'cosine', 'polynomial', 'step', 'onecycle'
        warmup_steps: Number of warmup steps for the scheduler.

        Additional scheduler-specific attributes that can be provided:
        - T_max: Maximum number of iterations for cosine annealing
        - eta_min: Minimum learning rate for cosine annealing
        - power: Power factor for polynomial decay
        - step_size: Step size for step decay
        - gamma: Multiplicative factor for step decay
        - max_lr: Maximum learning rate for OneCycleLR
        - pct_start: Percentage of steps for increasing LR in OneCycleLR
    """
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
        """
        Validate learning rate scheduler arguments.

        Raises:
            ValueError: If validation fails for any scheduler parameter.
        """
        supported_schedulers = [
            'linear',
            'cosine',
            'polynomial',
            'step',
            'onecycle',
        ]

        # Validate scheduler type
        if self.lr_scheduler_type not in supported_schedulers:
            raise ValueError(
                f'lr_scheduler_type must be one of {supported_schedulers}, got {self.lr_scheduler_type}'
            )

        # Validate common parameters
        if self.warmup_steps < 0:
            raise ValueError(
                f'warmup_steps must be >= 0, got {self.warmup_steps}')

        # Validate scheduler-specific parameters
        scheduler_type = self.lr_scheduler_type.lower()

        if scheduler_type == 'cosine':
            # Cosine scheduler specific validation
            if self.eta_min < 0:
                raise ValueError(f'eta_min must be >= 0, got {self.eta_min}')

        elif scheduler_type == 'polynomial':
            # Polynomial scheduler specific validation
            if self.power <= 0:
                raise ValueError(f'power must be > 0, got {self.power}')

        elif scheduler_type == 'step':
            # Step scheduler specific validation
            if self.step_size <= 0:
                raise ValueError(
                    f'step_size must be > 0, got {self.step_size}')
            if self.gamma <= 0 or self.gamma > 1:
                raise ValueError(f'gamma must be in (0, 1], got {self.gamma}')

        elif scheduler_type == 'onecycle':
            # OneCycleLR specific validation
            if self.pct_start <= 0 or self.pct_start >= 1:
                raise ValueError(
                    f'pct_start must be in (0, 1), got {self.pct_start}')


@dataclass
class ParallelArguments:
    """
    Arguments pertaining to distributed parallelism configuration.

    Attributes:
        tensor_parallel_size: Number of GPUs for tensor parallelism (splits model weights).
        pipeline_parallel_size: Number of GPUs for pipeline parallelism (splits model layers).
        data_parallel_size: Number of GPUs for data parallelism (splits batches).
        context_parallel_size: Number of GPUs for context parallelism (splits sequence length).
        expert_parallel_size: Number of GPUs for expert parallelism (for MoE models).
        pipeline_parallel_engine: Pipeline scheduling engine ('1f1b' or 'afab').
        backend: Distributed communication backend ('nccl' for GPU, 'gloo' for CPU).

    Note:
        Total number of GPUs = tensor_parallel_size * pipeline_parallel_size *
        data_parallel_size * context_parallel_size * expert_parallel_size
    """
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
    expert_parallel_size: int = field(
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
        """
        Validate all parallelism arguments.

        Raises:
            ValueError: If any parallelism size is invalid or engine/backend is unsupported.
        """
        # Validate parallelism sizes
        validate_parallelism_sizes(self.data_parallel_size,
                                   self.tensor_parallel_size,
                                   self.pipeline_parallel_size,
                                   self.context_parallel_size)

        # Validate pipeline parallel engine
        validate_pipeline_engine(self.pipeline_parallel_engine)

        # Validate backend
        supported_backends = {'nccl', 'gloo'}
        if self.backend not in supported_backends:
            raise ValueError(
                f'backend must be one of {supported_backends}, got {self.backend}'
            )


@dataclass
class TrainingArguments:
    """
    Arguments pertaining to training configuration.

    Attributes:
        batch_size: Global training batch size.
        test_batch_size: Batch size for evaluation/testing.
        micro_batch_size: Micro batch size per GPU (optional, defaults to batch_size).
        gradient_accumulation_steps: Number of micro-batches to accumulate before optimizer step.
        epochs: Number of complete passes through the training dataset.
        learning_rate: Learning rate for optimizer (optional, may be set by scheduler).
        seed: Random seed for reproducibility.
        sequence_length: Maximum sequence length for training (optional).
        log_interval: Number of steps between logging training metrics.

    Note:
        Global batch size = data_parallel_size * micro_batch_size * gradient_accumulation_steps
    """

    batch_size: int = field(
        default=64,
        metadata={'help': 'Training batch size'},
    )
    test_batch_size: int = field(
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
    epochs: int = field(
        default=5,
        metadata={'help': 'Number of training epochs'},
    )
    learning_rate: Optional[float] = field(
        default=1e-3,
        metadata={'help': 'Learning rate (alias for lr)'},
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

    def __post_init__(self) -> None:
        """
        Validate training arguments.

        Raises:
            ValueError: If any training parameter is invalid.
        """
        validate_training_parameters(
            self.gradient_accumulation_steps,
            self.micro_batch_size,
            self.sequence_length,
        )


@dataclass
class CheckpointArguments:
    """
    Arguments pertaining to checkpointing and model saving.

    Attributes:
        work_dir: Directory path where checkpoints and outputs are saved.
        save_frequency: Number of training steps between checkpoint saves.
        resume_path: Path to a checkpoint file to resume training from (empty string if not resuming).
    """
    work_dir: str = field(
        default='./work_dir',
        metadata={'help': 'Directory to save checkpoints'},
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
        """
        Validate checkpoint arguments.

        Raises:
            ValueError: If save_frequency is invalid.
        """
        if self.save_frequency < 1:
            raise ValueError(
                f'save_frequency must be >= 1, got {self.save_frequency}')


@dataclass
class LoggingArguments:
    """
    Arguments pertaining to experiment logging and tracking.

    Attributes:
        use_wandb: Whether to enable Weights & Biases (wandb) logging.
        project_name: Name of the wandb project for organizing experiments.
        experiment_name: Optional name for this specific experiment run.
    """
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

    def __post_init__(self) -> None:
        """Validate logging arguments (currently no validation needed)."""
        pass


@dataclass
class ScaleTorchArguments(
        DataArguments,
        ModelArguments,
        LrSchedulerArguments,
        ParallelArguments,
        TrainingArguments,
        CheckpointArguments,
        LoggingArguments,
):
    """
    Comprehensive arguments class for ScaleTorch distributed training.

    This class combines all argument categories needed for ScaleTorch training:
    - Data loading and processing
    - Model configuration
    - Parallelism settings
    - Training hyperparameters
    - Checkpointing
    - Logging

    Attributes:
        global_batch_size: Computed global batch size (data_parallel_size *
            micro_batch_size * gradient_accumulation_steps).
        global_batch_size_token: Computed global batch size in tokens
            (global_batch_size * sequence_length). None if sequence_length is None.

    Example:
        ```python
        from scaletorch.configs.arg_utils import ScaleTorchArguments
        from transformers import HfArgumentParser

        parser = HfArgumentParser(ScaleTorchArguments)
        args, = parser.parse_args_into_dataclasses()
        print(f"Global batch size: {args.global_batch_size}")
        ```
    """

    # Computed attributes (not parsed from command line)
    global_batch_size: int = field(init=False, default=0)
    global_batch_size_token: Optional[int] = field(init=False, default=None)

    def __post_init__(self) -> None:
        """
        Validate all inherited arguments and compute derived values.

        This method:
        1. Calls __post_init__ on all parent classes to validate their arguments
        2. Computes global_batch_size and global_batch_size_token

        Raises:
            ValueError: If any validation fails or required values are None.
        """
        # Validate all parent class arguments
        DataArguments.__post_init__(self)
        LrSchedulerArguments.__post_init__(self)
        ModelArguments.__post_init__(self)
        ParallelArguments.__post_init__(self)
        TrainingArguments.__post_init__(self)
        CheckpointArguments.__post_init__(self)
        LoggingArguments.__post_init__(self)

        # Compute global batch size (requires micro_batch_size to be set)
        if self.micro_batch_size is None:
            # Fallback to batch_size if micro_batch_size is not provided
            self.micro_batch_size = self.batch_size
            logger.info(
                f'micro_batch_size not provided, using batch_size: {self.batch_size}'
            )

        self.global_batch_size = (self.data_parallel_size *
                                  self.micro_batch_size *
                                  self.gradient_accumulation_steps)

        # Compute global batch size in tokens (requires sequence_length)
        if self.sequence_length is None:
            logger.warning(
                'sequence_length is None, cannot compute global_batch_size_token'
            )
            self.global_batch_size_token = None
        else:
            self.global_batch_size_token = (self.global_batch_size *
                                            self.sequence_length)


def main() -> None:
    """
    Example function demonstrating argument parsing with ScaleTorchArguments.

    This function shows how to:
    1. Create an HfArgumentParser for ScaleTorchArguments
    2. Parse command-line arguments into a strongly-typed dataclass
    3. Validate all configuration parameters automatically
    4. Log the parsed and validated arguments

    The parser automatically handles:
    - Command-line argument parsing
    - Type conversion and validation
    - Help message generation (via --help flag)

    Raises:
        ImportError: If transformers library is not installed.
        ValueError: If required arguments are missing or invalid.
        SystemExit: If --help flag is used (standard argparse behavior).

    Example:
        Run from command line:
        ```bash
        python -m scaletorch.configs.arg_utils --model_name_or_path gpt2 \\
            --batch_size 32 --tensor_parallel_size 2
        ```
    """
    # Create parser for ScaleTorchArguments
    parser = HfArgumentParser(ScaleTorchArguments)

    # Parse command-line arguments into dataclass
    # This will automatically validate all arguments via __post_init__
    scaletorch_args, = parser.parse_args_into_dataclasses()

    # Log the parsed and validated arguments
    logger.info('Initializing with parsed command line arguments...')
    logger.info('\n=== ScaleTorch Arguments ===')
    logger.info(json.dumps(dataclasses.asdict(scaletorch_args), indent=2))
    logger.info(f'\nGlobal batch size: {scaletorch_args.global_batch_size}')
    if scaletorch_args.global_batch_size_token is not None:
        logger.info(
            f'Global batch size (tokens): {scaletorch_args.global_batch_size_token}'
        )


if __name__ == '__main__':
    main()
