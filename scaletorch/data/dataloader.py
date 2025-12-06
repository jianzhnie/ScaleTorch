"""
Distributed data loader for ScaleTorch with micro-batching support.

This module provides a custom DataLoader implementation that supports:
- Micro-batching for gradient accumulation
- Context parallelism for sequence length splitting
- Distributed sampling across data parallel ranks
- Efficient tokenization and chunking of text data
"""

from typing import Any, Dict, Iterator, List, Optional, Union

import torch
from torch.utils.data import DataLoader, DistributedSampler

from scaletorch.data.dataset import DatasetProcessor
from scaletorch.parallel.pg_manager import process_group_manager as pgm


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
        sequence_length: Total sequence length
        sequence_length_per_gpu: Sequence length per GPU after context parallelism split
        global_batch_size: Effective batch size across all data parallel ranks
        gradient_accumulation_steps: Number of gradient accumulation steps
    """

    def __init__(self,
                 micro_batch_size: int,
                 sequence_length: int,
                 dataset_name: str,
                 tokenizer_name: str,
                 num_workers: int,
                 num_proc: int,
                 gradient_accumulation_steps: int,
                 device: Union[str, torch.device],
                 subset_name: Optional[str] = None,
                 split: str = 'train',
                 num_samples: Optional[int] = None,
                 pin_memory: bool = True,
                 text_column_name: str = 'text',
                 prefetch_factor: int = 2) -> None:
        """
        Initialize the MicroBatchDataLoader.

        Args:
            micro_batch_size: Number of samples per micro-batch
            sequence_length: Length of input sequences
            dataset_name: Name of the dataset to load
            tokenizer_name: Name or path of the tokenizer
            num_workers: Number of worker processes for data loading
            num_proc: Number of processes for dataset processing
            gradient_accumulation_steps: Number of gradient accumulation steps
            device: Device for distributed operations
            subset_name: Optional subset name for the dataset
            split: Dataset split to use ('train', 'validation', etc.)
            num_samples: Optional limit on number of samples to use
            pin_memory: Whether to pin memory for faster GPU transfer
            text_column_name: Name of the text column in the dataset
            prefetch_factor: Number of batches to prefetch per worker

        Raises:
            RuntimeError: If distributed setup fails
            ValueError: If configuration parameters are invalid
        """
        super().__init__([])  # Initialize parent with empty dataset

        # Validate input parameters
        if micro_batch_size <= 0:
            raise ValueError(
                f'micro_batch_size must be positive, got {micro_batch_size}')
        if sequence_length <= 0:
            raise ValueError(
                f'sequence_length must be positive, got {sequence_length}')
        if gradient_accumulation_steps <= 0:
            raise ValueError(
                f'gradient_accumulation_steps must be positive, got {gradient_accumulation_steps}'
            )
        if prefetch_factor < 1:
            raise ValueError(
                f'prefetch_factor must be at least 1, got {prefetch_factor}')

        self.micro_batch_size = micro_batch_size
        self.sequence_length = sequence_length
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.text_column_name = text_column_name
        self.prefetch_factor = prefetch_factor

        # Calculate distributed batch sizes
        self.pgm = pgm
        if self.pgm is None:
            # Single process mode
            self.global_batch_size = micro_batch_size * gradient_accumulation_steps
            self.cp_world_size = 1
            self.dp_world_size = 1
        else:
            # Distributed mode
            self.global_batch_size = micro_batch_size * gradient_accumulation_steps * self.pgm.dp_world_size
            self.cp_world_size = self.pgm.cp_world_size
            self.dp_world_size = self.pgm.dp_world_size

        self.num_global_micro_batches = self.global_batch_size // micro_batch_size

        # Calculate sequence length per GPU for context parallelism
        if self.cp_world_size <= 0:
            raise ValueError(f'Invalid cp_world_size: {self.cp_world_size}')

        if sequence_length % self.cp_world_size != 0:
            raise ValueError(
                f'sequence_length ({sequence_length}) must be divisible by cp_world_size '
                f'({self.cp_world_size})')

        self.sequence_length_per_gpu = sequence_length // self.cp_world_size

        # Calculate tokens per step for memory management
        self.tokens_per_step = micro_batch_size * self.sequence_length_per_gpu

        # Initialize dataset processor
        self.dataset_processor = DatasetProcessor(tokenizer_name, device,
                                                  self.pgm)

        # Load and prepare dataset
        dataset = self.dataset_processor.load_dataset(dataset_name, split,
                                                      subset_name, num_samples)

        # Tokenize and chunk the dataset
        self.tokenized_dataset = self.dataset_processor.tokenize_dataset(
            dataset, text_column_name, self.sequence_length, num_proc)

        # Setup distributed sampler if in distributed mode
        if self.pgm is not None:
            self._setup_distributed_sampler()

        # Initialize DataLoader with improved settings
        # Use persistent_workers only if num_workers > 0 to avoid issues
        super().__init__(
            self.tokenized_dataset,
            batch_size=micro_batch_size,
            collate_fn=self.collate_batch,
            pin_memory=pin_memory
            and torch.cuda.is_available(),  # Only pin if CUDA available
            num_workers=num_workers,
            sampler=self.sampler,
            shuffle=False,
            prefetch_factor=prefetch_factor
            if num_workers > 0 else 2,  # Enable multi-worker prefetching
            persistent_workers=num_workers >
            0)  # Keep workers alive between epochs only if using workers

        # Initialize iterator and prefetch state
        self._iterator: Optional[Iterator] = None
        self._prefetched_batch: Optional[Dict[str, torch.Tensor]] = None
        self._batch_available: bool = False

    def _setup_distributed_sampler(self) -> None:
        """Setup distributed sampler for data parallelism with improved shuffling."""
        if self.pgm is None:
            # Single process mode, no need for distributed sampler
            return

        try:
            # Use DistributedSampler with proper shuffling for better data distribution
            self.sampler = DistributedSampler(
                self.tokenized_dataset,
                num_replicas=self.pgm.dp_world_size,
                rank=self.pgm.dp_rank,
                shuffle=True,  # Enable shuffling for better data diversity
                drop_last=
                True  # Drop last batch to ensure consistent batch sizes
            )
        except Exception as e:
            raise RuntimeError(f'Failed to setup distributed sampler: {e}')

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
            # Optimize memory usage by pre-allocating tensors
            batch_size = len(batch)

            # Extract input_ids directly from batch items without intermediate list
            # Use stack instead of tensor for better memory efficiency with existing tensors
            input_ids_list = [item['input_ids'] for item in batch]

            # If items are already tensors, use stack; otherwise create new tensor
            if isinstance(input_ids_list[0], torch.Tensor):
                batch_input_ids = torch.stack(input_ids_list, dim=0)
            else:
                batch_input_ids = torch.tensor(input_ids_list,
                                               dtype=torch.long)

            # Calculate indices for context parallelism
            if self.pgm is None:
                # Single process mode, use entire sequence segment
                start_idx = 0
                end_idx = self.sequence_length_per_gpu
            else:
                start_idx = self.pgm.cp_rank * self.sequence_length_per_gpu
                end_idx = start_idx + self.sequence_length_per_gpu

            # Extract the sequence segment for this GPU with minimal copying
            input_ids = batch_input_ids[:, start_idx:end_idx]
            target_ids = batch_input_ids[:, start_idx + 1:end_idx + 1]

            # Generate position IDs for this segment - use expand for memory efficiency (no copy)
            position_ids = torch.arange(start_idx,
                                        end_idx,
                                        dtype=torch.long,
                                        device=batch_input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(
                batch_size, -1)  # More memory efficient than repeat

            return {
                'input_ids': input_ids,
                'target_ids': target_ids,
                'position_ids': position_ids,
                'hidden_states': None  # Placeholder for model compatibility
            }

        except Exception as e:
            raise RuntimeError(f'Error collating batch: {e}')

    def __iter__(self) -> Iterator:
        """
        Initialize or return the iterator with prefetch support.

        Returns:
            Iterator object for data loading
        """
        # Always reinitialize the iterator to handle epoch boundaries properly
        self._iterator = super().__iter__()
        # Start prefetching the first batch
        self._prefetch_next()
        return self

    def _prefetch_next(self) -> None:
        """
        Prefetch the next batch asynchronously.
        """
        if self._iterator is None:
            self._iterator = super().__iter__()

        try:
            self._prefetched_batch = next(self._iterator)
            self._batch_available = True
        except StopIteration:
            # Try to continue with next epoch
            try:
                if hasattr(self.sampler, 'set_epoch'):
                    # Increment epoch for the sampler
                    current_epoch = getattr(self.sampler, 'epoch', 0)
                    self.sampler.set_epoch(current_epoch + 1)

                # Reinitialize iterator for next epoch
                self._iterator = super().__iter__()

                # Try to get first batch from next epoch
                self._prefetched_batch = next(self._iterator)
                self._batch_available = True
            except StopIteration:
                # Dataset is completely exhausted
                self._batch_available = False
                self._iterator = None
            except Exception as e:
                raise RuntimeError(f'Error during epoch transition: {e}')

    def __next__(self) -> Dict[str, torch.Tensor]:
        """
        Get the next batch with prefetching support.

        Returns:
            Next batch of data

        Raises:
            StopIteration: When no more batches are available after exhausting all retries
            RuntimeError: If batch format is invalid or other errors occur
        """
        if not hasattr(self, '_batch_available') or not self._batch_available:
            raise StopIteration('Data loader exhausted all available samples')

        # Get the prefetched batch
        batch = self._prefetched_batch

        # Prefetch the next batch immediately
        self._prefetch_next()

        return batch
