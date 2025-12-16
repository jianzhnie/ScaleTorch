import os
from typing import Tuple

import hydra
import torch
from mingpt.char_dataset import CharDataset, DataConfig
from mingpt.model import GPT, GPTConfig, OptimizerConfig, create_optimizer
from mingpt.trainer import Trainer, TrainerConfig
from omegaconf import DictConfig
from torch.distributed import destroy_process_group, init_process_group
from torch.utils.data import Dataset, random_split


def ddp_setup() -> None:
    """Set up the Distributed Data Parallel (DDP) environment.

    This function:
    - Initializes the process group using NCCL backend
    - Sets the current CUDA device based on LOCAL_RANK

    Raises:
        RuntimeError: If distributed environment variables are not set
    """
    try:
        # Initialize distributed process group
        init_process_group(backend='nccl')

        # Set CUDA device based on local rank
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    except KeyError:
        raise RuntimeError(
            'Distributed environment not set up. '
            "Ensure you're using torchrun or torch.distributed.launch")
    except Exception as e:
        raise RuntimeError(f'Error setting up distributed environment: {e}')


def get_train_objs(
    gpt_cfg: GPTConfig, opt_cfg: OptimizerConfig, data_cfg: DataConfig
) -> Tuple[GPT, torch.optim.Optimizer, Dataset, Dataset]:
    """Prepare training objects for distributed training.

    Args:
        gpt_cfg (GPTConfig): Configuration for GPT model
        opt_cfg (OptimizerConfig): Configuration for optimizer
        data_cfg (DataConfig): Configuration for dataset

    Returns:
        Tuple containing:
        - Initialized GPT model
        - Optimizer
        - Training dataset
        - Test/Validation dataset
    """
    # Create character-level dataset
    dataset = CharDataset(data_cfg)

    # Split dataset into training and testing
    train_len = int(len(dataset) * data_cfg.train_split)
    train_set, test_set = random_split(
        dataset, [train_len, len(dataset) - train_len])

    # Update model configuration based on dataset
    gpt_cfg.vocab_size = dataset.vocab_size
    gpt_cfg.block_size = dataset.block_size

    # Create model and optimizer
    model = GPT(gpt_cfg)
    optimizer = create_optimizer(model, opt_cfg)

    return model, optimizer, train_set, test_set


@hydra.main(version_base=None, config_path='.', config_name='gpt2_train_cfg')
def main(cfg: DictConfig) -> None:
    """Main training script for distributed GPT training.

    Args:
        cfg (DictConfig): Hydra configuration dictionary

    The script:
    1. Sets up distributed training environment
    2. Loads configurations from Hydra config
    3. Prepares training objects
    4. Runs the trainer
    5. Cleans up distributed environment
    """
    try:
        # Set up distributed training environment
        ddp_setup()

        # Extract configurations from Hydra config
        gpt_cfg = GPTConfig(**cfg['gpt_config'])
        opt_cfg = OptimizerConfig(**cfg['optimizer_config'])
        data_cfg = DataConfig(**cfg['data_config'])
        trainer_cfg = TrainerConfig(**cfg['trainer_config'])

        # Prepare training objects
        model, optimizer, train_data, test_data = get_train_objs(
            gpt_cfg, opt_cfg, data_cfg)

        # Initialize and run trainer
        trainer = Trainer(trainer_cfg, model, optimizer, train_data, test_data)
        trainer.train()

    except Exception as e:
        print(f'Training failed with error: {e}')
        raise
    finally:
        # Ensure process group is destroyed
        destroy_process_group()


def cli_entry() -> None:
    """Entry point for the training script.

    Provides a clean separation for potential future modifications or
    additional CLI handling.
    """
    main()


if __name__ == '__main__':
    cli_entry()
