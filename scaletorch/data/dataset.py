"""
Distributed data loader for ScaleTorch with micro-batching support.

This module provides a custom DataLoader implementation that supports:
- Micro-batching for gradient accumulation
- Context parallelism for sequence length splitting
- Distributed sampling across data parallel ranks
- Efficient tokenization and chunking of text data

The DatasetProcessor class handles all dataset-related operations including:
- Loading datasets from HuggingFace datasets library
- Initializing and broadcasting tokenizers across distributed ranks
- Tokenizing and chunking text data into fixed-length sequences
"""

from functools import partial
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.distribute as dist
from datasets import Dataset, Features, Sequence, Value, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from scaletorch.parallel.pg_manager import process_group_manager as pgm
from scaletorch.utils import get_logger

logger = get_logger(__name__)


class DatasetProcessor:
    """
    Handles dataset loading, tokenization, and preprocessing.

    This class encapsulates all dataset-related operations including:
    - Loading datasets from HuggingFace datasets
    - Initializing and managing tokenizers
    - Tokenizing and chunking text data into sequences
    - Broadcasting tokenizers across distributed ranks

    The processor supports both single-process and distributed training modes.
    In distributed mode, the tokenizer is created on rank 0 and broadcast to
    all other ranks to ensure consistency.

    Attributes:
        tokenizer: The tokenizer instance used for text processing.
            Initialized on rank 0 and broadcast to all ranks in distributed mode.
        pgm: Process group manager for distributed operations.
            None in single-process mode.
    """

    def __init__(self, tokenizer_name_or_path: str,
                 device: Union[str, torch.device]) -> None:
        """
        Initialize the DatasetProcessor.

        Args:
            tokenizer_name_or_path: Name or path of the tokenizer (e.g., 'gpt2',
                'bert-base-uncased', or a local path).
            device: Device for distributed operations. Used for broadcasting
                objects in distributed mode. Can be a string ('cpu', 'cuda:0')
                or a torch.device object.

        Raises:
            RuntimeError: If tokenizer initialization or broadcasting fails.
            ValueError: If tokenizer_name is empty or invalid.
        """
        if not tokenizer_name_or_path or not isinstance(
                tokenizer_name_or_path, str):
            raise ValueError(
                f'tokenizer_name_or_path must be a non-empty string, got {tokenizer_name_or_path}'
            )

        self.pgm = pgm
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self._initialize_tokenizer(tokenizer_name_or_path, device)

    def _initialize_tokenizer(self, tokenizer_name_or_path: str,
                              device: Union[str, torch.device]) -> None:
        """
        Initialize tokenizer with distributed broadcasting.

        In distributed mode, the tokenizer is created on rank 0 and broadcast
        to all other ranks to ensure consistency. In single-process mode,
        the tokenizer is created directly.

        Args:
            tokenizer_name_or_path: Name or path of the tokenizer to load.
            device: Device for distributed broadcasting operations.

        Raises:
            RuntimeError: If tokenizer creation or broadcasting fails.
        """
        # Create tokenizer on rank 0 (or in single-process mode)
        if self.pgm is None or self.pgm.global_rank == 0:
            rank_str = '0' if self.pgm is None else str(self.pgm.global_rank)
            logger.info(
                f'Rank {rank_str}: Creating tokenizer from {tokenizer_name_or_path}'
            )

            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_name_or_path)
                objects: List[Optional[PreTrainedTokenizer]] = [tokenizer]
            except FileNotFoundError as e:
                raise RuntimeError(
                    f"Tokenizer '{tokenizer_name_or_path}' not found. "
                    f'Please check the path or model name: {e}')
            except Exception as e:
                raise RuntimeError(
                    f"Failed to create tokenizer '{tokenizer_name_or_path}': {e}"
                )
        else:
            # Other ranks wait to receive the tokenizer
            objects = [None]

        # Broadcast tokenizer to all ranks in distributed mode
        if self.pgm is not None:
            logger.info(
                f'Rank {self.pgm.global_rank}: Broadcasting tokenizer to all ranks'
            )

            try:
                # Note: device parameter may not be supported in all PyTorch versions
                # but is kept for consistency with the codebase
                dist.broadcast_object_list(objects, src=0, device=device)
                self.tokenizer = objects[0]

                if self.tokenizer is None:
                    raise RuntimeError(
                        'Failed to receive tokenizer from rank 0. '
                        'The broadcast may have failed or rank 0 did not send the tokenizer.'
                    )
            except RuntimeError:
                # Re-raise RuntimeError as-is
                raise
            except Exception as e:
                raise RuntimeError(
                    f'Failed to broadcast tokenizer from rank 0: {e}')
        else:
            # Single process mode - use the tokenizer directly
            self.tokenizer = objects[0]

        # Validate tokenizer was successfully initialized
        if self.tokenizer is None:
            raise RuntimeError(
                'Tokenizer initialization failed: tokenizer is None')

    def load_dataset(self, dataset_name: str, split: str,
                     subset_name: Optional[str],
                     num_samples: Optional[int]) -> Dataset:
        """
        Load and optionally subset the dataset from HuggingFace datasets.

        This method loads a dataset from the HuggingFace datasets library.
        If num_samples is specified, it will limit the dataset to that number
        of samples. The dataset is loaded on all ranks independently.

        Args:
            dataset_name: Name of the dataset to load (e.g., 'wikitext',
                'openwebtext', or a local path).
            split: Dataset split to use ('train', 'validation', 'test', etc.).
            subset_name: Optional subset name for the dataset (e.g., 'wikitext-2'
                for the wikitext dataset). None if not applicable.
            num_samples: Optional limit on number of samples to use.
                If None, uses all available samples. Must be positive if specified.

        Returns:
            Loaded Dataset object from HuggingFace datasets library.

        Raises:
            RuntimeError: If dataset loading fails.
            ValueError: If num_samples is specified but is non-positive.
        """
        if num_samples is not None and num_samples <= 0:
            raise ValueError(
                f'num_samples must be positive if specified, got {num_samples}'
            )

        try:
            # Load dataset from HuggingFace datasets
            dataset = load_dataset(dataset_name, split=split, name=subset_name)

            # Optionally limit the number of samples
            if num_samples is not None and num_samples > 0:
                original_length = len(dataset)
                actual_samples = min(num_samples, original_length)
                dataset = dataset.select(range(actual_samples))
                logger.info(
                    f'Using {actual_samples} samples from dataset '
                    f'(requested: {num_samples}, available: {original_length})'
                )

            return dataset
        except FileNotFoundError as e:
            raise RuntimeError(f"Dataset '{dataset_name}' not found. "
                               f'Please check the dataset name or path: {e}')
        except Exception as e:
            raise RuntimeError(
                f"Failed to load dataset '{dataset_name}' with split '{split}': {e}"
            )

    @staticmethod
    def tokenizer_group_text(
            examples: List[str], tokenizer: PreTrainedTokenizer,
            sequence_length: int) -> Dict[str, List[List[int]]]:
        """
        Tokenize a list of texts and group them into fixed-length chunks.

        This method tokenizes a batch of text strings, concatenates all tokens,
        and then splits them into chunks of size (sequence_length + 1). The extra
        token is used for creating input-target pairs where targets are shifted
        by one position for language modeling.

        The method ensures that:
        - All texts are tokenized and concatenated into a single sequence
        - The total length is adjusted to be divisible by sequence_length
        - Chunks are created with overlap of 1 token (for input/target pairs)

        Args:
            examples: List of text strings to tokenize. Each string will be
                tokenized and all tokens will be concatenated together.
            tokenizer: Tokenizer instance to use for encoding text to tokens.
                Must have a `batch_encode_plus` method.
            sequence_length: Target sequence length for each chunk. Each chunk
                will have size (sequence_length + 1) to support input-target
                pairs for language modeling.

        Returns:
            Dictionary with a single key 'input_ids' containing a list of lists.
            Each inner list is a token sequence of length (sequence_length + 1).
            Format: {'input_ids': [[token1, token2, ...], [token1, token2, ...], ...]}

        Raises:
            RuntimeError: If tokenization fails or tokenizer is invalid.
            ValueError: If sequence_length is non-positive or examples is empty.
        """
        if sequence_length <= 0:
            raise ValueError(
                f'sequence_length must be positive, got {sequence_length}')
        if not examples:
            raise ValueError('examples list cannot be empty')

        try:
            # Tokenize the batch of texts
            # Using numpy arrays for memory efficiency with large datasets
            tokenized_text_batch = tokenizer.batch_encode_plus(
                examples,
                return_attention_mask=False,
                return_token_type_ids=False,
                return_tensors='np'  # Return as numpy arrays for efficiency
            )

            # Concatenate all tokenized texts into a single sequence
            # This allows us to split long documents across multiple chunks
            concatenated_tokens = {
                'input_ids': np.concatenate(tokenized_text_batch['input_ids'])
            }

            total_length = len(concatenated_tokens['input_ids'])

            # Adjust total length to be divisible by sequence_length
            # This ensures we can create complete chunks without remainder
            # The +1 accounts for the target token in each chunk
            if total_length >= sequence_length + 1:
                # Calculate how many complete chunks we can make
                # Subtract 1 to account for the target token, then round down
                total_length = ((total_length - 1) //
                                sequence_length) * sequence_length + 1
            else:
                # If total length is less than sequence_length + 1,
                # we can't create any chunks, so return empty result
                return {'input_ids': []}

            # Create chunks of sequence_length + 1 tokens
            # Each chunk overlaps by 1 token with the next chunk
            # This allows us to create input-target pairs for language modeling
            result = {
                'input_ids': [
                    concatenated_tokens['input_ids'][i:i + sequence_length +
                                                     1].tolist()
                    for i in range(0, total_length -
                                   sequence_length, sequence_length)
                ]
            }

            return result

        except AttributeError as e:
            raise RuntimeError(
                f'Tokenizer does not support batch_encode_plus: {e}')
        except Exception as e:
            raise RuntimeError(
                f'Error during tokenization of {len(examples)} examples: {e}')

    def tokenize_dataset(self, dataset: Dataset, text_column_name: str,
                         sequence_length: int, num_proc: int) -> Dataset:
        """
        Tokenize the dataset and group texts into fixed-length chunks.

        This method applies tokenization and chunking to the entire dataset.
        It uses the HuggingFace datasets library's `map` function to process
        the dataset in parallel across multiple processes. The resulting
        dataset will have a single column 'input_ids' containing token sequences
        of length (sequence_length + 1).

        The method:
        1. Tokenizes all texts in the specified column
        2. Concatenates and chunks tokens into sequences of (sequence_length + 1)
        3. Removes all original columns and keeps only 'input_ids'
        4. Uses caching to speed up subsequent runs

        Args:
            dataset: HuggingFace Dataset object to tokenize. Must contain
                a column with the name specified in text_column_name.
            text_column_name: Name of the column containing text data to tokenize.
                This column will be removed from the output dataset.
            sequence_length: Target sequence length for each chunk. Each resulting
                sequence will have length (sequence_length + 1) to support
                input-target pairs for language modeling.
            num_proc: Number of processes to use for parallel tokenization.
                Set to 1 for single-process mode. Higher values can speed up
                processing but use more memory.

        Returns:
            New Dataset object with tokenized and chunked sequences.
            The dataset will have:
            - A single column 'input_ids' of type Sequence[int64]
            - Each sequence has length (sequence_length + 1)
            - All original columns are removed

        Raises:
            RuntimeError: If tokenization fails, column not found, or dataset
                processing encounters an error.
            ValueError: If sequence_length is non-positive, num_proc is invalid,
                or text_column_name is not in the dataset.
            AttributeError: If tokenizer is not initialized.
        """
        if self.tokenizer is None:
            raise AttributeError(
                'Tokenizer not initialized. Call _initialize_tokenizer first.')

        if sequence_length <= 0:
            raise ValueError(
                f'sequence_length must be positive, got {sequence_length}')
        if num_proc < 1:
            raise ValueError(f'num_proc must be at least 1, got {num_proc}')

        # Validate that the text column exists in the dataset
        if text_column_name not in dataset.column_names:
            raise ValueError(
                f"Column '{text_column_name}' not found in dataset. "
                f'Available columns: {dataset.column_names}')

        try:
            # Create a partial function with fixed arguments
            # This allows us to pass the tokenizer and sequence_length to the
            # mapping function while keeping the examples parameter flexible
            tokenizer_func = partial(self.tokenizer_group_text,
                                     tokenizer=self.tokenizer,
                                     sequence_length=sequence_length)

            # Apply tokenization and chunking to the dataset
            # The map function processes the dataset in parallel across num_proc processes
            tokenized_dataset = dataset.map(
                tokenizer_func,
                input_columns=text_column_name,
                remove_columns=dataset.column_names,
                features=Features({
                    'input_ids':
                    Sequence(feature=Value(dtype='int64'),
                             length=sequence_length + 1)
                }),
                batched=True,  # Process in batches for efficiency
                num_proc=num_proc,  # Parallel processing
                load_from_cache_file=True,  # Use cache if available
                desc=
                f'Tokenizing and grouping texts in chunks of {sequence_length + 1}'
            )

            return tokenized_dataset

        except KeyError as e:
            raise RuntimeError(
                f"Error accessing dataset column '{text_column_name}': {e}")
        except Exception as e:
            raise RuntimeError(
                f'Error tokenizing dataset with {len(dataset)} samples: {e}')
