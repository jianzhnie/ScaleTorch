"""
Distributed data loader for ScaleTorch with micro-batching support.

This module provides a custom DataLoader implementation that supports:
- Micro-batching for gradient accumulation
- Context parallelism for sequence length splitting
- Distributed sampling across data parallel ranks
- Efficient tokenization and chunking of text data
"""

from functools import partial
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
from datasets import Dataset, Features, Sequence, Value, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from scaletorch.utils.utils import print


class DatasetProcessor:
    """
    Handles dataset loading, tokenization, and preprocessing.

    This class encapsulates all dataset-related operations including:
    - Loading datasets from HuggingFace datasets
    - Initializing and managing tokenizers
    - Tokenizing and chunking text data into sequences

    Attributes:
        tokenizer: The tokenizer instance used for text processing
        pgm: Process group manager for distributed operations
    """

    def __init__(self,
                 tokenizer_name: str,
                 device: Union[str, torch.device],
                 pgm: Optional[Any] = None) -> None:
        """
        Initialize the DatasetProcessor.

        Args:
            tokenizer_name: Name or path of the tokenizer
            device: Device for distributed operations
            pgm: Optional process group manager for distributed mode

        Raises:
            RuntimeError: If tokenizer initialization fails
        """
        self.pgm = pgm
        self._initialize_tokenizer(tokenizer_name, device)

    def _initialize_tokenizer(self, tokenizer_name: str,
                              device: Union[str, torch.device]) -> None:
        """Initialize tokenizer with distributed broadcasting."""
        if self.pgm is None or self.pgm.global_rank == 0:
            print(
                f'Rank {0 if self.pgm is None else self.pgm.global_rank}: Creating tokenizer'
            )
            try:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                objects = [tokenizer]
            except Exception as e:
                raise RuntimeError(
                    f"Failed to create tokenizer '{tokenizer_name}': {e}")
        else:
            objects = [None]

        if self.pgm is not None:
            print(
                f'Rank {self.pgm.global_rank}: Broadcasting tokenizer to all ranks',
                is_print_rank=self.pgm.global_rank == 0)

            try:
                dist.broadcast_object_list(objects, src=0, device=device)
                self.tokenizer = objects[0]
                if self.tokenizer is None:
                    raise RuntimeError(
                        'Failed to receive tokenizer from rank 0')
            except Exception as e:
                raise RuntimeError(f'Failed to broadcast tokenizer: {e}')
        else:
            # Single process mode
            self.tokenizer = objects[0]

    def load_dataset(self, dataset_name: str, split: str,
                     subset_name: Optional[str],
                     num_samples: Optional[int]) -> Dataset:
        """
        Load and optionally subset the dataset.

        Args:
            dataset_name: Name of the dataset to load
            split: Dataset split to use ('train', 'validation', etc.)
            subset_name: Optional subset name for the dataset
            num_samples: Optional limit on number of samples to use

        Returns:
            Loaded dataset

        Raises:
            RuntimeError: If dataset loading fails
        """
        try:
            dataset = load_dataset(dataset_name, split=split, name=subset_name)

            if num_samples is not None and num_samples > 0:
                actual_samples = min(num_samples, len(dataset))
                dataset = dataset.select(range(actual_samples))
                print(f'Using {actual_samples} samples from dataset')

            return dataset
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset '{dataset_name}': {e}")

    @staticmethod
    def tokenizer_group_text(
            examples: List[str], tokenizer: PreTrainedTokenizer,
            sequence_length: int) -> Dict[str, List[List[int]]]:
        """
        Tokenize a list of texts and group them in chunks of sequence_length + 1.

        Args:
            examples: List of text strings to tokenize
            tokenizer: Tokenizer to use for encoding
            sequence_length: Target sequence length

        Returns:
            Dictionary with 'input_ids' key containing list of token sequences

        Raises:
            RuntimeError: If tokenization fails
        """
        try:
            # Tokenize the batch of texts
            tokenized_text_batch = tokenizer.batch_encode_plus(
                examples,
                return_attention_mask=False,
                return_token_type_ids=False,
                return_tensors='np')

            # Concatenate all tokenized texts
            concatenated_tokens = {
                'input_ids': np.concatenate(tokenized_text_batch['input_ids'])
            }

            total_length = len(concatenated_tokens['input_ids'])

            # Adjust total length to be divisible by sequence_length
            if total_length >= sequence_length + 1:
                total_length = ((total_length - 1) //
                                sequence_length) * sequence_length + 1

            # Create chunks of sequence_length + 1 tokens
            result = {
                'input_ids': [
                    concatenated_tokens['input_ids'][i:i + sequence_length +
                                                     1].tolist()
                    for i in range(0, total_length -
                                   sequence_length, sequence_length)
                ]
            }

            return result

        except Exception as e:
            raise RuntimeError(f'Error during tokenization: {e}')

    def tokenize_dataset(self, dataset: Dataset, text_column_name: str,
                         sequence_length: int, num_proc: int) -> Dataset:
        """
        Tokenize the dataset and group texts in chunks of sequence_length + 1.

        Args:
            dataset: Dataset to tokenize
            text_column_name: Name of the text column
            sequence_length: Target sequence length
            num_proc: Number of processes for parallel tokenization

        Returns:
            Tokenized dataset with chunked sequences

        Raises:
            RuntimeError: If tokenization fails
        """
        try:
            # Create a partial function with fixed arguments
            tokenizer_func = partial(self.tokenizer_group_text,
                                     tokenizer=self.tokenizer,
                                     sequence_length=sequence_length)

            tokenized_dataset = dataset.map(
                tokenizer_func,
                input_columns=text_column_name,
                remove_columns=dataset.column_names,
                features=Features({
                    'input_ids':
                    Sequence(feature=Value(dtype='int64'),
                             length=sequence_length + 1)
                }),
                batched=True,
                num_proc=num_proc,
                load_from_cache_file=True,
                desc=f'Grouping texts in chunks of {sequence_length + 1}')

            return tokenized_dataset

        except Exception as e:
            raise RuntimeError(f'Error tokenizing dataset: {e}')
