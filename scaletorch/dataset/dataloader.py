"""
Distributed data loader for ScaleTorch with micro-batching support.

This module provides a custom DataLoader implementation that supports:
- Micro-batching for gradient accumulation
- Context parallelism for sequence length splitting
- Distributed sampling across data parallel ranks
- Efficient tokenization and chunking of text data
"""

from functools import partial
from typing import Any, Dict, Iterator, List, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
from datasets import Dataset, Features, Sequence, Value, load_dataset
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, PreTrainedTokenizer

import scaletorch.parallel.pg_manager as pgm
from scaletorch.utils.utils import print


class MicroBatchDataLoader(DataLoader):
    """
    Custom DataLoader for distributed training with micro-batching and context parallelism.

    This loader supports:
    - Micro-batching for gradient accumulation
    - Context parallelism for splitting sequences across GPUs
    - Distributed sampling for data parallelism
    - Efficient tokenization and chunking of text data

    Attributes:
        micro_batch_size: Size of each micro-batch
        seq_length: Total sequence length
        seq_length_per_gpu: Sequence length per GPU after context parallelism split
        global_batch_size: Effective batch size across all data parallel ranks
        grad_acc_steps: Number of gradient accumulation steps
    """

    def __init__(self,
                 micro_batch_size: int,
                 seq_length: int,
                 dataset_name: str,
                 tokenizer_name: str,
                 num_workers: int,
                 num_proc: int,
                 grad_acc_steps: int,
                 device: Union[str, torch.device],
                 subset_name: Optional[str] = None,
                 split: str = 'train',
                 num_samples: Optional[int] = None,
                 pin_memory: bool = True,
                 text_column_name: str = 'text') -> None:
        """
        Initialize the MicroBatchDataLoader.

        Args:
            micro_batch_size: Number of samples per micro-batch
            seq_length: Length of input sequences
            dataset_name: Name of the dataset to load
            tokenizer_name: Name or path of the tokenizer
            num_workers: Number of worker processes for data loading
            num_proc: Number of processes for dataset processing
            grad_acc_steps: Number of gradient accumulation steps
            device: Device for distributed operations
            subset_name: Optional subset name for the dataset
            split: Dataset split to use ('train', 'validation', etc.)
            num_samples: Optional limit on number of samples to use
            pin_memory: Whether to pin memory for faster GPU transfer
            text_column_name: Name of the text column in the dataset

        Raises:
            RuntimeError: If distributed setup fails
            ValueError: If configuration parameters are invalid
        """
        super().__init__([])  # Initialize parent with empty dataset

        # Validate input parameters
        if micro_batch_size <= 0:
            raise ValueError(
                f'micro_batch_size must be positive, got {micro_batch_size}')
        if seq_length <= 0:
            raise ValueError(f'seq_length must be positive, got {seq_length}')
        if grad_acc_steps <= 0:
            raise ValueError(
                f'grad_acc_steps must be positive, got {grad_acc_steps}')

        self.micro_batch_size = micro_batch_size
        self.seq_length = seq_length
        self.grad_acc_steps = grad_acc_steps
        self.text_column_name = text_column_name

        # Calculate distributed batch sizes
        self.global_batch_size = micro_batch_size * grad_acc_steps * pgm.process_group_manager.dp_world_size
        self.num_global_micro_batches = self.global_batch_size // micro_batch_size

        # Calculate sequence length per GPU for context parallelism
        if pgm.process_group_manager.cp_world_size <= 0:
            raise ValueError(
                f'Invalid cp_world_size: {pgm.process_group_manager.cp_world_size}'
            )

        if seq_length % pgm.process_group_manager.cp_world_size != 0:
            raise ValueError(
                f'seq_length ({seq_length}) must be divisible by cp_world_size '
                f'({pgm.process_group_manager.cp_world_size})')

        self.seq_length_per_gpu = seq_length // pgm.process_group_manager.cp_world_size

        # Load and prepare dataset
        self._load_dataset(dataset_name, split, subset_name, num_samples)

        # Initialize tokenizer with distributed broadcasting
        self._initialize_tokenizer(tokenizer_name, device)

        # Tokenize and chunk the dataset
        self.tokenized_dataset = self.tokenize_dataset(self.dataset,
                                                       text_column_name,
                                                       self.seq_length,
                                                       num_proc)

        # Setup distributed sampler
        self._setup_distributed_sampler()

        # Initialize DataLoader
        super().__init__(self.tokenized_dataset,
                         batch_size=micro_batch_size,
                         collate_fn=self.collate_batch,
                         pin_memory=pin_memory,
                         num_workers=num_workers,
                         sampler=self.sampler,
                         shuffle=False)

        # Initialize iterator state
        self._iterator: Optional[Iterator] = None

    def _load_dataset(self, dataset_name: str, split: str,
                      subset_name: Optional[str],
                      num_samples: Optional[int]) -> None:
        """Load and optionally subset the dataset."""
        try:
            self.dataset = load_dataset(dataset_name,
                                        split=split,
                                        name=subset_name)

            if num_samples is not None and num_samples > 0:
                actual_samples = min(num_samples, len(self.dataset))
                self.dataset = self.dataset.select(range(actual_samples))
                print(f'Using {actual_samples} samples from dataset')

        except Exception as e:
            raise RuntimeError(f"Failed to load dataset '{dataset_name}': {e}")

    def _initialize_tokenizer(self, tokenizer_name: str,
                              device: Union[str, torch.device]) -> None:
        """Initialize tokenizer with distributed broadcasting."""
        if pgm.process_group_manager.global_rank == 0:
            print(
                f'Rank {pgm.process_group_manager.global_rank}: Creating tokenizer'
            )
            try:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                objects = [tokenizer]
            except Exception as e:
                raise RuntimeError(
                    f"Failed to create tokenizer '{tokenizer_name}': {e}")
        else:
            objects = [None]

        print(
            f'Rank {pgm.process_group_manager.global_rank}: Broadcasting tokenizer to all ranks',
            is_print_rank=pgm.process_group_manager.global_rank == 0)

        try:
            dist.broadcast_object_list(objects, src=0, device=device)
            self.tokenizer = objects[0]
            if self.tokenizer is None:
                raise RuntimeError('Failed to receive tokenizer from rank 0')
        except Exception as e:
            raise RuntimeError(f'Failed to broadcast tokenizer: {e}')

    def _setup_distributed_sampler(self) -> None:
        """Setup distributed sampler for data parallelism."""
        try:
            self.sampler = DistributedSampler(
                self.tokenized_dataset,
                num_replicas=pgm.process_group_manager.dp_world_size,
                rank=pgm.process_group_manager.dp_rank,
                shuffle=False)
        except Exception as e:
            raise RuntimeError(f'Failed to setup distributed sampler: {e}')

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

    def collate_batch(self, batch: List[Dict[str,
                                             Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of samples into tensors with context parallelism support.

        Args:
            batch: List of samples with 'input_ids' key

        Returns:
            Dictionary containing:
                - input_ids: Input token sequences
                - target_ids: Target token sequences (shifted by 1)
                - position_ids: Position IDs for the sequence segment
                - hidden_states: None (placeholder for model compatibility)
        """
        try:
            # Stack input IDs into a batch tensor
            batch_input_ids = torch.stack([
                torch.tensor(item['input_ids'], dtype=torch.long)
                for item in batch
            ])

            batch_size = batch_input_ids.size(0)

            # Calculate indices for context parallelism
            start_idx = pgm.process_group_manager.cp_rank * self.seq_length_per_gpu
            end_idx = start_idx + self.seq_length_per_gpu

            # Extract the sequence segment for this GPU
            input_ids = batch_input_ids[:, start_idx:end_idx].contiguous()
            target_ids = batch_input_ids[:, start_idx + 1:end_idx +
                                         1].contiguous()

            # Generate position IDs for this segment
            position_ids = torch.arange(start_idx, end_idx,
                                        dtype=torch.long).unsqueeze(0).expand(
                                            batch_size, -1).contiguous()

            return {
                'input_ids': input_ids,
                'target_ids': target_ids,
                'position_ids': position_ids,
                'hidden_states': None  # Placeholder for model compatibility
            }

        except Exception as e:
            raise RuntimeError(f'Error collating batch: {e}')

    def __iter__(self) -> Iterator:
        """Initialize or return the iterator."""
        if self._iterator is None:
            self._iterator = super().__iter__()
        return self

    def __next__(self) -> Dict[str, torch.Tensor]:
        """
        Get the next batch, handling iterator reinitialization on exhaustion.

        Returns:
            Next batch of data

        Raises:
            StopIteration: When no more batches are available
        """
        if self._iterator is None:
            self._iterator = super().__iter__()

        try:
            batch = next(self._iterator)
            return batch
        except StopIteration:
            # Reinitialize the sampler and iterator for the next epoch
            try:
                if hasattr(self.sampler, 'set_epoch'):
                    current_epoch = getattr(self.sampler, 'epoch', 0)
                    self.sampler.set_epoch(current_epoch + 1)

                self._iterator = super().__iter__()
                return next(self._iterator)
            except StopIteration:
                # No data available even after reinitialization
                self._iterator = None
                raise StopIteration
