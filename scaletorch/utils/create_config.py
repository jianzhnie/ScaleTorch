"""
Configuration creation utility for ScaleTorch experiments.

This module provides functionality to create experiment configuration files
for distributed training setups with various parallelism strategies.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from transformers import AutoConfig

from scaletorch.utils.logger_utils import get_logger

__all__ = ['create_single_config']

logger = get_logger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration creation fails."""
    pass


class TemplateNotFoundError(FileNotFoundError):
    """Raised when template file cannot be found."""
    pass


def _find_template_file(template_dir: str = 'template',
                        template_filename: str = 'base_config.json') -> Path:
    """
    Locate the template configuration file.

    Searches for 'template/base_config.json' in the following order:
    1. Directory adjacent to this module: `../template/base_config.json`
    2. Current module directory: `./template/base_config.json`
    3. Current working directory: `./template/base_config.json`

    Returns
    -------
    Path
        Absolute path to the template file.

    Raises
    ------
    TemplateNotFoundError
        If the template file cannot be found in any of the searched locations.
    """
    module_dir = Path(__file__).resolve().parent
    candidates: List[Path] = [
        module_dir.parent / template_dir / template_filename,
        module_dir / template_dir / template_filename,
        Path.cwd() / template_dir / template_filename,
    ]

    for candidate in candidates:
        if candidate.exists():
            logger.debug(f'Found template file at: {candidate}')
            return candidate

    searched_paths = [str(p) for p in candidates]
    raise TemplateNotFoundError(
        f"Could not find '{template_dir}/{template_filename}'. "
        f"Searched locations: {', '.join(searched_paths)}")


def _validate_parallelism_sizes(
    data_parallel_size: int,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    context_parallel_size: int,
) -> None:
    """
    Validate parallelism size parameters.

    Parameters
    ----------
    data_parallel_size : int
        Number of data parallelism workers.
    tensor_parallel_size : int
        Number of tensor parallelism workers.
    pipeline_parallel_size : int
        Number of pipeline parallelism workers.
    context_parallel_size : int
        Number of context parallelism workers.

    Raises
    ------
    ValueError
        If any parallelism size is less than 1.
    """
    parallelism_params = {
        'data_parallel_size': data_parallel_size,
        'tensor_parallel_size': tensor_parallel_size,
        'pipeline_parallel_size': pipeline_parallel_size,
        'context_parallel_size': context_parallel_size,
    }

    for param_name, value in parallelism_params.items():
        if value < 1:
            raise ValueError(f'{param_name} must be >= 1, got {value}')


def _validate_pipeline_engine(engine: str) -> None:
    """
    Validate pipeline parallel engine parameter.

    Parameters
    ----------
    engine : str
        Pipeline parallel engine strategy.

    Raises
    ------
    ValueError
        If engine is not supported.
    """
    supported_engines = {'1f1b', 'interleaved'}  # Example supported engines
    if engine not in supported_engines:
        raise ValueError(f"Unsupported pipeline engine '{engine}'. "
                         f'Supported engines: {supported_engines}')


def _validate_training_parameters(
    grad_accumulation_steps: int,
    micro_batch_size: int,
    sequence_length: int,
) -> None:
    """
    Validate training-related parameters.

    Parameters
    ----------
    grad_accumulation_steps : int
        Number of gradient accumulation steps.
    micro_batch_size : int
        Micro batch size.
    sequence_length : int
        Sequence length.

    Raises
    ------
    ValueError
        If any training parameter is invalid.
    """
    if grad_accumulation_steps < 1:
        raise ValueError(
            f'grad_accumulation_steps must be >= 1, got {grad_accumulation_steps}'
        )
    if micro_batch_size < 1:
        raise ValueError(
            f'micro_batch_size must be >= 1, got {micro_batch_size}')
    if sequence_length < 1:
        raise ValueError(
            f'sequence_length must be >= 1, got {sequence_length}')


def _load_model_configuration(
    model_name_or_path: str,
    num_hidden_layers: Optional[int],
    num_attention_heads: Optional[int],
    num_key_value_heads: Optional[int],
) -> Dict[str, Any]:
    """
    Load model configuration from HuggingFace or use provided values.

    Parameters
    ----------
    model_name_or_path : str
        Model name or path.
    num_hidden_layers : Optional[int]
        Override for number of hidden layers.
    num_attention_heads : Optional[int]
        Override for number of attention heads.
    num_key_value_heads : Optional[int]
        Override for number of key-value heads.

    Returns
    -------
    Dict[str, Any]
        Model configuration parameters.
    """
    model_config = {}

    try:
        hf_config = AutoConfig.from_pretrained(model_name_or_path)
        logger.info(
            f'Successfully loaded config for model: {model_name_or_path}')

        # Extract model parameters, using provided values as overrides
        model_config['num_hidden_layers'] = (
            num_hidden_layers if num_hidden_layers is not None else getattr(
                hf_config, 'num_hidden_layers', None))
        model_config['num_attention_heads'] = (
            num_attention_heads if num_attention_heads is not None else
            getattr(hf_config, 'num_attention_heads', None))
        model_config['num_key_value_heads'] = (
            num_key_value_heads if num_key_value_heads is not None else
            getattr(hf_config, 'num_key_value_heads', None))

    except Exception as exc:
        logger.warning(
            f"Could not load AutoConfig for '{model_name_or_path}': {exc}. "
            'Using provided values or None for model parameters.')
        # Use provided values or None if not available
        model_config['num_hidden_layers'] = num_hidden_layers
        model_config['num_attention_heads'] = num_attention_heads
        model_config['num_key_value_heads'] = num_key_value_heads

    return model_config


def _calculate_batch_sizes(
    data_parallel_size: int,
    micro_batch_size: int,
    grad_accumulation_steps: int,
    sequence_length: int,
) -> Tuple[int, int]:
    """
    Calculate global batch sizes for logging purposes.

    Parameters
    ----------
    data_parallel_size : int
        Number of data parallelism workers.
    micro_batch_size : int
        Micro batch size per worker.
    grad_accumulation_steps : int
        Number of gradient accumulation steps.
    sequence_length : int
        Sequence length in tokens.

    Returns
    -------
    Tuple[int, int]
        Global batch size and global batch size in tokens.
    """
    global_batch_size = data_parallel_size * micro_batch_size * grad_accumulation_steps
    global_batch_size_token = global_batch_size * sequence_length
    return global_batch_size, global_batch_size_token


def create_single_config(
    template_dir: str,
    data_parallel_size: int,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    context_parallel_size: int,
    pipeline_parallel_engine: str,
    model_name_or_path: str,
    num_hidden_layers: Optional[int] = None,
    num_attention_heads: Optional[int] = None,
    num_key_value_heads: Optional[int] = None,
    grad_accumulation_steps: int = 1,
    micro_batch_size: int = 1,
    sequence_length: int = 1024,
    use_cpu: bool = False,
    use_fused_adam: bool = False,
    subset_name: Optional[str] = None,
    experiment_name: str = 'scaletorch_experiment',
    use_wandb: bool = False,
    output_dir: Union[str, Path] = '.',
) -> Path:
    """
    Create a configuration JSON file for a single experiment run.

    This function creates a complete experiment configuration by combining
    a base template with user-specified parameters for distributed training
    setup, model configuration, and training hyperparameters.

    Parameters
    ----------
    data_parallel_size : int
        Number of data parallelism workers. Must be >= 1.
    tensor_parallel_size : int
        Number of tensor parallelism workers. Must be >= 1.
    pipeline_parallel_size : int
        Number of pipeline parallelism workers. Must be >= 1.
    context_parallel_size : int
        Number of context parallelism workers. Must be >= 1.
    pipeline_parallel_engine : str
        Pipeline parallel engine strategy (e.g., '1f1b').
    model_name_or_path : str
        Model name or path to create configuration for.
    num_hidden_layers : Optional[int], optional
        Number of hidden layers. If None, will attempt to load from model config.
    num_attention_heads : Optional[int], optional
        Number of attention heads. If None, will attempt to load from model config.
    num_key_value_heads : Optional[int], optional
        Number of key-value heads. If None, will attempt to load from model config.
    grad_accumulation_steps : int, optional
        Number of gradient accumulation steps. Must be >= 1.
    micro_batch_size : int, optional
        Micro batch size per worker. Must be >= 1.
    sequence_length : int, optional
        Sequence length in tokens. Must be >= 1.
    use_cpu : bool, optional
        Whether to use CPU for training. Defaults to False.
    use_fused_adam : bool, optional
        Whether to use fused Adam optimizer. Defaults to False.
    subset_name : Optional[str], optional
        Name of the dataset subset to use.
    experiment_name : str, optional
        Name of the experiment. Defaults to 'scaletorch_experiment'.
    use_wandb : bool, optional
        Whether to use Weights & Biases for logging. Defaults to False.
    output_dir : Union[str, Path], optional
        Output directory for the configuration. Defaults to current directory.

    Returns
    -------
    Path
        Absolute path to the experiment run directory containing the config file.

    Raises
    ------
    TemplateNotFoundError
        If the base template file cannot be located.
    ConfigurationError
        If the configuration cannot be written or validated.
    ValueError
        If any input parameters are invalid.

    Notes
    -----
    The function will overwrite any existing experiment directory with the same name.
    Global batch size is calculated as: data_parallel_size * micro_batch_size * grad_accumulation_steps.
    """
    # Validate input parameters
    _validate_parallelism_sizes(data_parallel_size, tensor_parallel_size,
                                pipeline_parallel_size, context_parallel_size)
    _validate_training_parameters(grad_accumulation_steps, micro_batch_size,
                                  sequence_length)

    # Resolve and create output directory
    output_dir = Path(output_dir).resolve()
    run_path = output_dir / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f'Creating experiment configuration: {experiment_name}')
    logger.info(f'Output directory: {run_path}')

    # Load base template
    try:
        template_path = _find_template_file(template_dir=template_dir)
        logger.debug(f'Using template file: {template_path}')

        with open(template_path, 'r', encoding='utf-8') as f:
            base_config: Dict[str, Any] = json.load(f)
    except TemplateNotFoundError:
        raise
    except Exception as exc:
        raise ConfigurationError(
            f'Failed to load template configuration: {exc}') from exc

    # Create deep copy to avoid modifying the template
    config_content = deepcopy(base_config)

    # Configure training parameters
    training_config = config_content.setdefault('training', {})
    training_config['sequence_length'] = sequence_length
    training_config['gradient_accumulation_steps'] = grad_accumulation_steps
    training_config['micro_batch_size'] = micro_batch_size

    # Configure checkpoint settings
    checkpoint_config = config_content.setdefault('checkpoint', {})
    checkpoint_config['save_dir'] = str(run_path)

    # Configure dataset settings
    dataset_config = config_content.setdefault('dataset', {})
    if subset_name is not None:
        dataset_config['subset_name'] = subset_name

    # Configure model settings
    model_config = _load_model_configuration(model_name_or_path,
                                             num_hidden_layers,
                                             num_attention_heads,
                                             num_key_value_heads)
    model_config['model_name_or_path'] = model_name_or_path
    model_config['use_fused_adam'] = use_fused_adam
    config_content['model'] = model_config

    # Configure distributed training settings
    distributed_config = config_content.setdefault('distributed', {})
    distributed_config.update({
        'tensor_parallel_size': tensor_parallel_size,
        'context_parallel_size': context_parallel_size,
        'data_parallel_size': data_parallel_size,
        'pipeline_parallel_size': pipeline_parallel_size,
        'pipeline_parallel_engine': pipeline_parallel_engine,
        'use_cpu': use_cpu,
    })

    if use_cpu:
        environment_config = config_content.setdefault('environment', {})
        environment_config['FLASH_ATTEN'] = '0'
        distributed_config['backend'] = 'gloo'

    # Configure logging settings
    logging_config = config_content.setdefault('logging', {})
    logging_config['use_wandb'] = use_wandb
    logging_config['run_name'] = experiment_name

    # Calculate and log batch size information
    global_batch_size, global_batch_size_token = _calculate_batch_sizes(
        data_parallel_size, micro_batch_size, grad_accumulation_steps,
        sequence_length)

    logger.info(f'Batch size configuration - '
                f'Global: {global_batch_size:,} samples, '
                f'Global tokens: {global_batch_size_token:,}, '
                f'Data parallel size: {data_parallel_size}, '
                f'Sequence length: {sequence_length}, '
                f'Gradient accumulation: {grad_accumulation_steps}, '
                f'Micro batch size: {micro_batch_size}')

    # Clean up existing run directory if it exists
    if run_path.exists():
        try:
            logger.warning(
                f'Removing existing experiment directory: {run_path}')
            shutil.rmtree(run_path)
        except Exception as exc:
            raise ConfigurationError(
                f"Failed to remove existing run directory '{run_path}': {exc}"
            ) from exc

    # Create fresh run directory
    try:
        run_path.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise ConfigurationError(
            f"Failed to create run directory '{run_path}': {exc}") from exc

    # Write configuration file
    config_file_path = run_path / 'config.json'
    try:
        with open(config_file_path, 'w', encoding='utf-8') as f:
            json.dump(config_content, f, indent=2, ensure_ascii=False)

        logger.info(
            f'Configuration successfully written to: {config_file_path}')

    except Exception as exc:
        raise ConfigurationError(
            f"Failed to write configuration file '{config_file_path}': {exc}"
        ) from exc

    return run_path


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command line arguments for the configuration creation script.

    Parameters
    ----------
    argv : Optional[List[str]], optional
        Command line arguments. If None, uses sys.argv.

    Returns
    -------
    argparse.Namespace
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description=
        'Create experiment configuration files for ScaleTorch distributed training.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Parallelism configuration
    parallelism_group = parser.add_argument_group('Parallelism Configuration')
    parallelism_group.add_argument(
        '--data_parallel_size',
        type=int,
        default=1,
        help='Number of data parallelism workers',
    )
    parallelism_group.add_argument(
        '--tensor_parallel_size',
        type=int,
        default=1,
        help='Number of tensor parallelism workers',
    )
    parallelism_group.add_argument(
        '--pipeline_parallel_size',
        type=int,
        default=1,
        help='Number of pipeline parallelism workers',
    )
    parallelism_group.add_argument(
        '--context_parallel_size',
        type=int,
        default=1,
        help='Number of context parallelism workers',
    )
    parallelism_group.add_argument(
        '--pipeline_parallel_engine',
        type=str,
        default='1f1b',
        help='Pipeline parallel engine strategy',
    )

    # Model configuration
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument(
        '--model_name_or_path',
        type=str,
        default='Qwen/Qwen8-8B',
        help='Model name or path for configuration',
    )
    model_group.add_argument(
        '--num_hidden_layers',
        type=int,
        default=None,
        help='Number of hidden layers (overrides model config)',
    )
    model_group.add_argument(
        '--num_attention_heads',
        type=int,
        default=None,
        help='Number of attention heads (overrides model config)',
    )
    model_group.add_argument(
        '--num_key_value_heads',
        type=int,
        default=None,
        help='Number of key-value heads (overrides model config)',
    )

    # Training configuration
    training_group = parser.add_argument_group('Training Configuration')
    training_group.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help='Number of gradient accumulation steps',
    )
    training_group.add_argument(
        '--micro_batch_size',
        type=int,
        default=1,
        help='Micro batch size per worker',
    )
    training_group.add_argument(
        '--sequence_length',
        type=int,
        default=1024,
        help='Sequence length in tokens',
    )
    training_group.add_argument(
        '--subset_name',
        type=str,
        default=None,
        help='Dataset subset name',
    )

    # Experiment configuration
    experiment_group = parser.add_argument_group('Experiment Configuration')
    experiment_group.add_argument(
        '--template_dir',
        type=str,
        default='template',
        help='Directory containing the configuration templates',
    )
    experiment_group.add_argument(
        '--experiment_name',
        type=str,
        default='scaletorch_experiment',
        help='Experiment name',
    )
    experiment_group.add_argument(
        '--output_dir',
        type=str,
        default='.',
        help='Output directory for configuration files',
    )

    # Feature flags
    flags_group = parser.add_argument_group('Feature Flags')
    flags_group.add_argument(
        '--use_wandb',
        action='store_true',
        help='Enable Weights & Biases logging',
    )
    flags_group.add_argument(
        '--use_cpu',
        action='store_true',
        help='Use CPU for training',
    )
    flags_group.add_argument(
        '--use_fused_adam',
        action='store_true',
        help='Use fused Adam optimizer',
    )

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """
    Main entry point for the configuration creation script.

    Parameters
    ----------
    argv : Optional[List[str]], optional
        Command line arguments. If None, uses sys.argv.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure).
    """
    try:
        args = _parse_args(argv)

        run_dir = create_single_config(
            template_dir=args.template_dir,
            data_parallel_size=args.data_parallel_size,
            tensor_parallel_size=args.tensor_parallel_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
            context_parallel_size=args.context_parallel_size,
            pipeline_parallel_engine=args.pipeline_parallel_engine,
            model_name_or_path=args.model_name_or_path,
            num_hidden_layers=args.num_hidden_layers,
            num_attention_heads=args.num_attention_heads,
            num_key_value_heads=args.num_key_value_heads,
            grad_accumulation_steps=args.gradient_accumulation_steps,
            micro_batch_size=args.micro_batch_size,
            sequence_length=args.sequence_length,
            subset_name=args.subset_name,
            experiment_name=args.experiment_name,
            use_wandb=args.use_wandb,
            use_cpu=args.use_cpu,
            use_fused_adam=args.use_fused_adam,
            output_dir=args.output_dir,
        )

        logger.info(f'✅ Configuration created successfully at: {run_dir}')
        return 0

    except Exception as exc:
        logger.error(f'❌ Failed to create configuration: {exc}')
        logger.debug('Full exception details:', exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
