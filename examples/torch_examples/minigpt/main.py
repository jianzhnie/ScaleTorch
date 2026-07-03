import os

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset, random_split

from minigpt.char_dataset import CharDataset, DataConfig
from minigpt.model import GPT, GPTConfig, OptimizerConfig, create_optimizer
from minigpt.trainer import Trainer, TrainerConfig
from scaletorch.utils import cleanup_dist, init_dist_pytorch


def ddp_setup() -> None:
    """Set up the DDP environment using scaletorch."""
    try:
        init_dist_pytorch()

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    except KeyError:
        raise RuntimeError(
            "Distributed environment not set up. "
            "Ensure you're using torchrun or torch.distributed.launch"
        )
    except Exception as e:
        raise RuntimeError("Error setting up distributed environment: %s" % e)


def get_train_objs(
    gpt_cfg: GPTConfig, opt_cfg: OptimizerConfig, data_cfg: DataConfig
) -> tuple[GPT, torch.optim.Optimizer, Dataset, Dataset]:
    """Prepare training objects for distributed training."""
    dataset = CharDataset(data_cfg)

    train_len = int(len(dataset) * data_cfg.train_split)
    train_set, test_set = random_split(dataset, [train_len, len(dataset) - train_len])

    gpt_cfg.vocab_size = dataset.vocab_size
    gpt_cfg.block_size = dataset.block_size

    model = GPT(gpt_cfg)
    optimizer = create_optimizer(model, opt_cfg)

    return model, optimizer, train_set, test_set


@hydra.main(version_base=None, config_path=".", config_name="gpt2_train_cfg")
def main(cfg: DictConfig) -> None:
    try:
        ddp_setup()

        gpt_cfg = GPTConfig(**cfg["gpt_config"])
        opt_cfg = OptimizerConfig(**cfg["optimizer_config"])
        data_cfg = DataConfig(**cfg["data_config"])
        trainer_cfg = TrainerConfig(**cfg["trainer_config"])

        model, optimizer, train_data, test_data = get_train_objs(
            gpt_cfg, opt_cfg, data_cfg
        )

        trainer = Trainer(trainer_cfg, model, optimizer, train_data, test_data)
        trainer.train()

    except Exception as e:
        print("Training failed with error: %s" % e)
        raise
    finally:
        cleanup_dist()


def cli_entry() -> None:
    main()


if __name__ == "__main__":
    cli_entry()
