from pathlib import Path
from typing import Dict

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from scaletorch.utils.logger_utils import get_logger

logger = get_logger(__name__)


def load_custom_dataset(data_path: str,
                        split: str = 'train',
                        cache_dir: str = '') -> Dataset:
    """
    Load a dataset from either a local JSON/JSONL file or from the Hugging Face Hub.

    Args:
        data_path (str): A string path to a local JSON/JSONL file or a Hugging Face
                         dataset name (e.g., 'gemini/math_qa_datasets').
        split (str, optional): The split of the dataset to load (default: 'train').
        cache_dir (str, optional): The directory to cache the dataset (default: None).

    Returns:
        Dataset: A loaded Hugging Face `Dataset` object, specifically the 'train' split.

    Raises:
        FileNotFoundError: If the local data file does not exist.
        ValueError: If the file format is not supported or if the dataset from
                    the Hub cannot be loaded.
    """
    data_path_obj = Path(data_path)

    try:
        if data_path_obj.suffix in ['.json', '.jsonl']:
            logger.info(
                f'🔍 Detected local file format: {data_path_obj.suffix}, using JSON loader.'
            )
            # The 'json' loader supports both .json and .jsonl formats.
            dataset = load_dataset('json',
                                   data_files=data_path,
                                   split=split,
                                   cache_dir=cache_dir)
        else:
            logger.info(
                '🌐 Detected dataset name, loading from Hugging Face Hub.')
            dataset = load_dataset(data_path, split=split, cache_dir=cache_dir)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f'Failed to find the data file at: {data_path}') from e
    except Exception as e:
        # Catch any other loading-related errors.
        raise ValueError(
            f'Failed to load dataset from {data_path}: {e}') from e

    logger.info(f'✅ Successfully loaded dataset with {len(dataset)} samples.')
    return dataset


class PretrainDataset(Dataset):
    """

    Args:
        data_path (str): Path to the training data.
        tokenizer (PreTrainedTokenizer): The tokenizer to use.
        split (str): The split to use from the training data.
        max_length (int): The maximum length of the input sequences (default: 550).
    """

    def __init__(self,
                 data_path: str,
                 split: str,
                 cache_dir: str = None,
                 tokenizer: PreTrainedTokenizer = None,
                 max_length: int = 1024) -> None:

        self.dataset = load_custom_dataset(data_path, split, cache_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Retrieves and preprocesses a single data sample.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing:
                - input_ids: Tokenized and padded input sequence
                - attention_mask: Mask indicating non-padded tokens
                - token_type_ids: (if using BERT-like models)
        """
        # Get text sample from dataset
        sample = self.dataset[idx]
        text: str = sample['text']

        # Tokenize and pad the text
        encodings_input = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'  # Return PyTorch tensors directly
        )

        # Remove the batch dimension added by return_tensors='pt'
        return {key: val.squeeze(0) for key, val in encodings_input.items()}
