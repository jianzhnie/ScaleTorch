"""
Utility script for creating experiment configuration files.

Example usage:
    python create_config.py --out_dir tmp --exp_name test_2_node --tp 2 --cp 2 --pp 2 --dp 2 --model_name HuggingFaceTB/SmolLM-360M-Instruct --num_attention_heads 16 --num_key_value_heads 4 --grad_acc_steps 1 --mbs 32 --seq_len 4096 --use_wandb
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from copy import deepcopy
from typing import Any, Dict, List, Optional

from transformers import AutoConfig

from scaletorch.utils.logger_utils import get_logger

__all__ = ['create_single_config']

logger = get_logger(__name__)


def _find_template_file() -> str:
    """Locate `template/base_config.json`.

    Search order:
    1. directory adjacent to this module: `.../template/base_config.json`
    2. current working directory: `./template/base_config.json`

    Returns
    -------
    str
        Absolute path to the template file.

    Raises
    ------
    FileNotFoundError
        If the template file cannot be found in either location.
    """
    module_dir = os.path.dirname(os.path.abspath(__file__))
    candidates: List[str] = [
        os.path.join(module_dir, '..', 'template', 'base_config.json'),
        os.path.join(module_dir, 'template', 'base_config.json'),
        os.path.join(os.getcwd(), 'template', 'base_config.json'),
    ]

    for path in candidates:
        path = os.path.abspath(path)
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        "Could not find 'template/base_config.json'. Checked: " +
        ', '.join(candidates))


def create_single_config(
    data_parallel_size: int,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    context_parallel_size: int,
    pipeline_parallel_engine: str,
    model_name_or_path: str,
    num_hidden_layers: Optional[int],
    num_attention_heads: Optional[int],
    num_key_value_heads: Optional[int],
    grad_accumulation_steps: int,
    micro_batch_size: int,
    sequence_length: int,
    use_cpu: bool = False,
    use_fused_adam: bool = False,
    subset_name: Optional[str] = None,
    experiment_name: str = 'scaletorch_experiment',
    use_wandb: bool = False,
    out_dir: str = '.',
) -> str:
    """Create a config JSON for a single experiment run and return the run path.

    Parameters
    ----------
    data_parallel_size : int
        Number of data parallelism.
    tensor_parallel_size : int
        Number of tensor parallelism.
    pipeline_parallel_size : int
        Number of pipeline parallelism.
    context_parallel_size : int
        Number of context parallelism.
    pipeline_parallel_engine : str
        Pipeline parallel engine.
    model_name_or_path : str
        Model name or path to create configs for.
    num_hidden_layers : Optional[int]
        Number of hidden layers.
    num_attention_heads : Optional[int]
        Number of attention heads.
    num_key_value_heads : Optional[int]
        Number of key value heads.
    grad_accumulation_steps : int
        Gradient accumulation steps.
    micro_batch_size : int
        Micro batch size.
    sequence_length : int
        Sequence length.
    use_cpu : bool, optional
        Use CPU for training, by default False.
    use_fused_adam : bool, optional
        Use fused adam, by default False.
    subset_name : Optional[str]
        Subset name.
    experiment_name : str
        Experiment name.
    use_wandb : bool, optional
        Use wandb for logging, by default False.
    out_dir : str, optional
        Output directory to store the configs, by default '.'.

    Returns
    -------
    str
        Absolute path to the run directory.

    Raises
    ------
    FileNotFoundError
        When the base template cannot be located.
    ValueError
        When the generated config cannot be written for some reason.
    """
    # Ensure output directory exists
    out_dir = os.path.abspath(out_dir)
    run_path = os.path.join(out_dir, experiment_name)
    os.makedirs(out_dir, exist_ok=True)

    template_path = _find_template_file()
    logger.debug('Using template at: %s', template_path)

    base_config: Dict[str, Any] = {}
    with open(template_path, 'r', encoding='utf-8') as f:
        base_config = json.load(f)

    config_content = deepcopy(base_config)
    # Environment and run-specific metadata
    config_content.setdefault('training', {})['seq_length'] = sequence_length
    config_content.setdefault('checkpoint', {})['save_dir'] = run_path
    config_content.setdefault('dataset', {})['subset_name'] = subset_name

    # Model settings
    config_content.setdefault('model', {})['name'] = model_name_or_path

    try:
        hf_model_config = AutoConfig.from_pretrained(model_name_or_path)
    except Exception as exc:  # Keep generic to avoid failing on any HF error
        logger.warning("Could not load AutoConfig for '%s': %s",
                       model_name_or_path, exc)
        hf_model_config = None

    if hf_model_config is not None:
        # Some AutoConfig implementations might not expose all attributes
        if num_hidden_layers is None:
            config_content['model']['num_hidden_layers'] = getattr(
                hf_model_config, 'num_hidden_layers', None)
        else:
            config_content['model']['num_hidden_layers'] = num_hidden_layers

        if num_attention_heads is None:
            config_content['model']['num_attention_heads'] = getattr(
                hf_model_config, 'num_attention_heads', None)
        else:
            config_content['model'][
                'num_attention_heads'] = num_attention_heads

        if num_key_value_heads is None:
            # Not all configs expose num_key_value_heads
            config_content['model']['num_key_value_heads'] = getattr(
                hf_model_config, 'num_key_value_heads', None)
        else:
            config_content['model'][
                'num_key_value_heads'] = num_key_value_heads

    # Explicitly capture fused adam preference
    config_content['model']['use_fused_adam'] = use_fused_adam

    # Distributed config
    dist = config_content.setdefault('distributed', {})
    dist['tp_size'] = tensor_parallel_size
    dist['cp_size'] = context_parallel_size
    dist['dp_size'] = data_parallel_size
    dist['pp_size'] = pipeline_parallel_size
    dist['pp_engine'] = pipeline_parallel_engine
    dist['use_cpu'] = use_cpu
    if use_cpu:
        config_content.setdefault('environment', {})['FLASH_ATTEN'] = '0'
        dist['backend'] = 'gloo'

    # Logging
    log_cfg = config_content.setdefault('logging', {})
    log_cfg['use_wandb'] = use_wandb
    log_cfg['run_name'] = experiment_name

    # Batch size calculations for debug
    global_batch_size = data_parallel_size * micro_batch_size * grad_accumulation_steps
    global_batch_size_token = global_batch_size * sequence_length
    logger.info(
        'Global_Batch_size_token: %s, Global_Batch_size: %s, data_parallel_size: %s, sequence_length: %s, grad_accumulation_steps: %s, micro_batch_size: %s',
        f'{global_batch_size_token:,}',
        global_batch_size,
        data_parallel_size,
        sequence_length,
        grad_accumulation_steps,
        micro_batch_size,
    )

    # Training params
    training = config_content.setdefault('training', {})
    training['gradient_accumulation_steps'] = grad_accumulation_steps
    training['micro_batch_size'] = micro_batch_size

    # Write run config, ensuring a clean run directory
    if os.path.exists(run_path):
        try:
            shutil.rmtree(run_path)
        except Exception as exc:
            logger.error("Failed to remove existing run directory '%s': %s",
                         run_path, exc)
            raise

    os.makedirs(run_path, exist_ok=True)
    config_file_path = os.path.join(run_path, 'config.json')
    try:
        with open(config_file_path, 'w', encoding='utf-8') as new_config:
            json.dump(config_content, new_config, indent=4)
    except Exception as exc:
        logger.error("Failed to write config to '%s': %s", config_file_path,
                     exc)
        raise ValueError('Unable to write config file') from exc

    logger.info('Config written to %s', config_file_path)
    return run_path


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments.

    Parameters
    ----------
    argv : Optional[List[str]], optional
        Command line arguments, by default None

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Create experiment configuration files.')
    parser.add_argument('--out_dir',
                        type=str,
                        help='Output directory to store the configs',
                        default='tmp')
    parser.add_argument('--tensor_parallel_size',
                        type=int,
                        help='Number of tensor parallelism',
                        default=1)
    parser.add_argument('--context_parallel_size',
                        type=int,
                        help='Number of context parallelism',
                        default=1)
    parser.add_argument('--data_parallel_size',
                        type=int,
                        help='Number of data parallelism',
                        default=1)
    parser.add_argument('--pipeline_parallel_size',
                        type=int,
                        help='Number of pipeline parallelism',
                        default=1)
    parser.add_argument('--pipeline_parallel_engine',
                        type=str,
                        help='Pipeline parallel engine',
                        default='1f1b')
    parser.add_argument('--model_name_or_path',
                        type=str,
                        help='Model name to create configs for',
                        default='HuggingFaceTB/SmolLM-360M-Instruct')
    parser.add_argument('--num_hidden_layers',
                        type=int,
                        help='Number of hidden layers',
                        default=None)
    parser.add_argument('--num_attention_heads',
                        type=int,
                        help='Number of attention heads',
                        default=None)
    parser.add_argument('--num_key_value_heads',
                        type=int,
                        help='Number of key value heads',
                        default=None)
    parser.add_argument('--grad_acc_steps',
                        type=int,
                        help='Gradient accumulation steps',
                        default=1)
    parser.add_argument('--micro_batch_size',
                        type=int,
                        help='Micro batch size',
                        default=1)
    parser.add_argument('--sequence_length',
                        type=int,
                        help='Sequence length',
                        default=1024)
    parser.add_argument('--subset_name',
                        type=str,
                        help='Subset name',
                        default=None)
    parser.add_argument('--experiment_name',
                        type=str,
                        help='Experiment name',
                        default='dummy_exp')
    parser.add_argument('--use_wandb',
                        action='store_true',
                        help='Use wandb for logging')
    parser.add_argument('--use_cpu',
                        action='store_true',
                        help='Use CPU for training')
    parser.add_argument('--use_fused_adam',
                        action='store_true',
                        help='Use fused adam')
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for the script.

    Parameters
    ----------
    argv : Optional[List[str]], optional
        Command line arguments, by default None

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure).
    """
    try:
        args = _parse_args(argv)
        run_dir = create_single_config(
            out_dir=args.out_dir,
            tensor_parallel_size=args.tensor_parallel_size,
            context_parallel_size=args.context_parallel_size,
            data_parallel_size=args.data_parallel_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
            pipeline_parallel_engine=args.pipeline_parallel_engine,
            model_name_or_path=args.model_name_or_path,
            num_hidden_layers=args.num_hidden_layers,
            num_attention_heads=args.num_attention_heads,
            num_key_value_heads=args.num_key_value_heads,
            grad_accumulation_steps=args.grad_acc_steps,
            micro_batch_size=args.micro_batch_size,
            sequence_length=args.sequence_length,
            subset_name=args.subset_name,
            experiment_name=args.experiment_name,
            use_wandb=args.use_wandb,
            use_cpu=args.use_cpu,
            use_fused_adam=args.use_fused_adam,
        )
        logger.info('Configs created successfully at %s ✅', run_dir)
        return 0
    except Exception as exc:
        logger.exception('Failed to create config: %s', exc)
        return 1


if __name__ == '__main__':
    sys.exit(main())
