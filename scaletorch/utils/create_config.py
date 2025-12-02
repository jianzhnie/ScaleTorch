"""
python create_config.py --out_dir tmp --exp_name test_2_node --tp 2 --cp 2 --pp 2 --dp 2 --model_name HuggingFaceTB/SmolLM-360M-Instruct --num_attention_heads 16 --num_key_value_heads 4 --grad_acc_steps 1 --mbs 32 --seq_len 4096 --use_wandb
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from copy import deepcopy
from typing import Any, Dict, Optional

from picotron.utils import download_model
from transformers import AutoConfig

__all__ = ['create_single_config']

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def _find_template_file() -> str:
    """Locate `template/base_config.json`.

    Search order:
    1. directory adjacent to this module: `.../template/base_config.json`
    2. current working directory: `./template/base_config.json`

    Raises
    ------
    FileNotFoundError
        If the template file cannot be found in either location.
    """
    module_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
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
    out_dir: str,
    tp: int,
    cp: int,
    dp: int,
    pp: int,
    pp_engine: str,
    model_name: str,
    num_hidden_layers: Optional[int],
    num_attention_heads: Optional[int],
    num_key_value_heads: Optional[int],
    grad_acc_steps: int,
    mbs: int,
    seq_len: int,
    subset_name: Optional[str],
    exp_name: str,
    use_wandb: bool = False,
    use_cpu: bool = False,
    use_fused_adam: bool = False,
    hf_token: Optional[str] = None,
) -> str:
    """Create a config JSON for a single experiment run and return the run path.

    Parameters mirror the original script's CLI options. The function writes
    a `config.json` into ``<out_dir>/<exp_name>/config.json`` and returns the
    absolute run directory path.

    Raises
    ------
    FileNotFoundError
        When the base template cannot be located.
    ValueError
        When the generated config cannot be written for some reason.
    """
    # Ensure output directory exists
    out_dir = os.path.abspath(out_dir)
    run_path = os.path.join(out_dir, exp_name)
    os.makedirs(out_dir, exist_ok=True)

    template_path = _find_template_file()
    logger.debug('Using template at: %s', template_path)

    with open(template_path, 'r', encoding='utf-8') as f:
        base_config: Dict[str, Any] = json.load(f)

    config_content = deepcopy(base_config)
    # Environment and run-specific metadata
    config_content.setdefault('environment', {})['HF_TOKEN'] = hf_token
    config_content.setdefault('training', {})['seq_length'] = seq_len
    config_content.setdefault('checkpoint', {})['save_dir'] = run_path
    config_content.setdefault('dataset', {})['subset_name'] = subset_name

    # Model settings
    config_content.setdefault('model', {})['name'] = model_name

    try:
        hf_model_config = AutoConfig.from_pretrained(model_name)
    except Exception as exc:  # Keep generic to avoid failing on any HF error
        logger.warning("Could not load AutoConfig for '%s': %s", model_name,
                       exc)
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
    dist['tp_size'] = tp
    dist['cp_size'] = cp
    dist['dp_size'] = dp
    dist['pp_size'] = pp
    dist['pp_engine'] = pp_engine
    dist['use_cpu'] = use_cpu
    if use_cpu:
        config_content.setdefault('environment', {})['FLASH_ATTEN'] = '0'
        dist['backend'] = 'gloo'

    # Logging
    log_cfg = config_content.setdefault('logging', {})
    log_cfg['use_wandb'] = use_wandb
    log_cfg['run_name'] = exp_name

    # Batch size calculations for debug
    gbs = dp * mbs * grad_acc_steps
    gbs_token = gbs * seq_len
    logger.info(
        'Gbs_token: %s, Gbs: %s, dp: %s, seq_len: %s, grad_acc_steps: %s, mbs: %s',
        f'{gbs_token:,}',
        gbs,
        dp,
        seq_len,
        grad_acc_steps,
        mbs,
    )

    # Training params
    training = config_content.setdefault('training', {})
    training['gradient_accumulation_steps'] = grad_acc_steps
    training['micro_batch_size'] = mbs

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

    def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument('--out_dir',
                            type=str,
                            help='Output directory to store the configs',
                            default='tmp')
        parser.add_argument('--tp',
                            type=int,
                            help='number of tensor parallelism',
                            default=1)
        parser.add_argument('--cp',
                            type=int,
                            help='number of context parallelism',
                            default=1)
        parser.add_argument('--dp',
                            type=int,
                            help='number of data parallelism',
                            default=1)
        parser.add_argument('--pp',
                            type=int,
                            help='number of pipeline parallelism',
                            default=1)
        parser.add_argument('--pp_engine',
                            type=str,
                            help='pipeline parallel engine',
                            default='1f1b')
        parser.add_argument('--model_name',
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
                            help='grad accumulation',
                            default=1)
        parser.add_argument('--mbs',
                            type=int,
                            help='micro batch size',
                            default=1)
        parser.add_argument('--seq_len',
                            type=int,
                            help='Sequence length',
                            default=1024)
        parser.add_argument('--subset_name',
                            type=str,
                            help='Subset name',
                            default=None)
        parser.add_argument('--exp_name',
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
        parser.add_argument('--hf_token',
                            type=str,
                            help='HF token',
                            default=None)
        return parser.parse_args(argv)

    if __name__ == '__main__':
        args = _parse_args()
        try:
            run_dir = create_single_config(
                out_dir=args.out_dir,
                tp=args.tp,
                cp=args.cp,
                dp=args.dp,
                pp=args.pp,
                pp_engine=args.pp_engine,
                model_name=args.model_name,
                num_hidden_layers=args.num_hidden_layers,
                num_attention_heads=args.num_attention_heads,
                num_key_value_heads=args.num_key_value_heads,
                grad_acc_steps=args.grad_acc_steps,
                mbs=args.mbs,
                seq_len=args.seq_len,
                subset_name=args.subset_name,
                exp_name=args.exp_name,
                use_wandb=args.use_wandb,
                use_cpu=args.use_cpu,
                use_fused_adam=args.use_fused_adam,
                hf_token=args.hf_token,
            )
            logger.info('Configs created successfully at %s ✅', run_dir)
        except Exception as exc:
            logger.exception('Failed to create config: %s', exc)
            raise

        # Attempt model download, but don't fail the script if the download fails.
        try:
            download_model(args.model_name, args.hf_token)
            logger.info('Model files downloaded successfully ✅')
        except Exception as exc:
            logger.warning('Model download failed: %s', exc)
