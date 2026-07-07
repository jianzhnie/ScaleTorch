"""Data utilities for Llama FSDP2 pre-training.

Provides causal/padding mask helpers and a ``PretrainingDataset`` that
tokenises raw text corpora into fixed-length sequences with [BOT]/[EOT]
delimiters.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import Dataset


def create_causal_mask(batch: Tensor, dtype: torch.dtype = torch.float32) -> Tensor:
    """Build a causal (upper-triangular) attention mask.

    Args:
        batch: Input token ids of shape ``(batch_size, seq_len)``.
        dtype: Data type of the mask (default ``float32``).

    Returns:
        Mask of shape ``(seq_len, seq_len)`` where ``mask[i, j] = -inf`` for
        ``j > i`` (future positions) and ``0`` otherwise.
    """
    _batch_size, seq_len = batch.shape
    mask = torch.full(
        (seq_len, seq_len), float("-inf"), device=batch.device, dtype=dtype
    ).triu(diagonal=1)
    return mask


def create_padding_mask(
    batch: Tensor, padding_token_id: int, dtype: torch.dtype = torch.float32
) -> Tensor:
    """Build a padding mask that hides ``padding_token_id`` positions.

    Args:
        batch: Input token ids of shape ``(batch_size, seq_len)``.
        padding_token_id: The id reserved for padding.
        dtype: Data type of the mask (default ``float32``).

    Returns:
        Mask of shape ``(batch_size, 1, seq_len, seq_len)`` where padded
        positions are ``-inf`` and valid positions are ``0``.
    """
    padded = torch.zeros_like(batch, device=batch.device, dtype=dtype).masked_fill(
        batch == padding_token_id, float("-inf")
    )
    # Broadcast: (B, 1, S, S) — additive mask compatible with SDPA.
    mask = padded[:, :, None] + padded[:, None, :]
    return mask[:, None, :, :]


class PretrainingDataset(Dataset):
    """Wrap a HuggingFace text dataset for causal language modelling.

    Each item is a pair ``(input_ids, target_ids)`` where ``target_ids`` is
    ``input_ids`` shifted right by one position.  Sequences are clipped /
    padded to ``seq_length`` tokens.  Special tokens ``[BOT]``, ``[EOT]``,
    and ``[PAD]`` are automatically prepended / appended.

    Args:
        dataset: A HuggingFace ``Dataset`` with a ``"text"`` column.
        tokenizer: A ``tokenizers.Tokenizer`` instance.
        seq_length: Fixed sequence length for training.
    """

    def __init__(
        self,
        dataset: "datasets.Dataset",  # noqa: F821
        tokenizer: "tokenizers.Tokenizer",  # noqa: F821
        seq_length: int,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.bot = tokenizer.token_to_id("[BOT]")
        self.eot = tokenizer.token_to_id("[EOT]")
        self.pad = tokenizer.token_to_id("[PAD]")

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        seq = self.dataset[index]["text"]  # type: ignore[index]
        tokens: list[int] = [self.bot] + self.tokenizer.encode(seq).ids + [self.eot]
        toklen = len(tokens)
        if toklen < self.seq_length + 1:
            pad_length = self.seq_length + 1 - toklen
            tokens += [self.pad] * pad_length
        x = torch.tensor(tokens[: self.seq_length], dtype=torch.int64)
        y = torch.tensor(tokens[1 : self.seq_length + 1], dtype=torch.int64)
        return x, y
