"""Tensor parallelism: column/row parallel linear and vocab parallel embedding."""

from __future__ import annotations

import math
import warnings
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from scaletorch.parallel.process_group import process_group_manager as pgm
from scaletorch.parallel.tensor_parallel.tp_comms import (
    GatherFromModelParallelRegion,
    ReduceFromModelParallelRegion,
    linear_with_all_reduce,
    linear_with_async_all_reduce,
)


def apply_tensor_parallel(
    model: torch.nn.Module,
    enable_sequence_parallel: bool = False,
    module_mapping: list[tuple[str, str, str]] | None = None,
) -> torch.nn.Module:
    """
    Apply tensor parallelism to a transformer model by replacing standard linear layers
    with their tensor parallel equivalents.

    This function modifies the model in-place by replacing attention and MLP linear layers
    with column-parallel or row-parallel versions based on their role in the computation.

    Args:
        model: The transformer model to apply tensor parallelism to.
        enable_sequence_parallel: If True, configure TP layers for sequence parallelism
            (skip internal all-reduce since SP handles communication).
        module_mapping: Optional custom mapping of (sub_module_name, linear_name, style)
            tuples. If None, uses the default mapping for decoder-only transformers.

    Returns:
        The modified model with tensor parallel layers.

    Raises:
        AttributeError: If the model doesn't have the expected structure.
        ValueError: If tensor parallel world size is not properly configured.
    """
    if not hasattr(model, "decoder_layers"):
        raise AttributeError("Model must have 'decoder_layers' attribute")

    if pgm.tp_world_size <= 1:
        return model  # No tensor parallelism needed

    def _replace_module(
        module: torch.nn.Module,
        linear_proj_name: str,
        style: str,
        args: dict[str, Any] | None = None,
    ) -> None:
        """
        Replace a standard linear layer with its tensor parallel equivalent.

        Args:
            module: The parent module containing the linear layer.
            linear_proj_name: Name of the linear layer attribute to replace.
            style: The tensor parallel style ('column', 'row', or 'vocab').
            args: Additional arguments for the replacement layer.

        Raises:
            ValueError: If the style is invalid or module structure is unexpected.
        """
        if args is None:
            args = {}

        if style not in ["column", "row", "vocab"]:
            raise ValueError(f"Invalid tensor parallel style: {style}")

        if not hasattr(module, linear_proj_name):
            raise AttributeError(
                f"Module {module} does not have attribute {linear_proj_name}"
            )

        linear_layer = getattr(module, linear_proj_name)

        if style == "column":
            new_linear_layer = ColumnParallelLinear(
                in_features=linear_layer.in_features,
                out_features=linear_layer.out_features,
                bias=linear_layer.bias is not None,
                gather_output=args.get("gather_output", False),
                sequence_parallel=enable_sequence_parallel,
            )
        elif style == "row":
            new_linear_layer = RowParallelLinear(
                in_features=linear_layer.in_features,
                out_features=linear_layer.out_features,
                bias=linear_layer.bias is not None,
                sequence_parallel=enable_sequence_parallel,
            )
        else:  # vocab
            new_linear_layer = VocabParallelEmbedding(
                num_embeddings=linear_layer.num_embeddings,
                embedding_dim=linear_layer.embedding_dim,
            )
        setattr(module, linear_proj_name, new_linear_layer)

    # Default TP mapping for decoder-only transformer architectures (Llama, Qwen, etc.).
    # For new model architectures, pass a custom mapping via the module_mapping parameter.
    default_module_linear_name_style_mapping = [
        ("attention", "q_proj", "column"),
        ("attention", "k_proj", "column"),
        ("attention", "v_proj", "column"),
        ("attention", "out_proj", "row"),
        ("mlp", "up_proj", "column"),
        ("mlp", "gate_proj", "column"),
        ("mlp", "down_proj", "row"),
    ]

    mapping = (
        module_mapping
        if module_mapping is not None
        else default_module_linear_name_style_mapping
    )

    # Apply tensor parallelism to decoder layers
    for layer in model.decoder_layers:
        for (
            module_name,
            linear_proj_name,
            style,
        ) in mapping:
            if hasattr(layer, module_name):
                _replace_module(getattr(layer, module_name), linear_proj_name, style)
            else:
                # Log warning for missing modules but continue processing
                warnings.warn(
                    f"Layer {layer} does not have module {module_name}", stacklevel=2
                )

    # Apply tensor parallelism to embedding and final projection layers
    _replace_module(model, "embedding", "vocab")
    _replace_module(model, "final_proj", "column", args={"gather_output": True})

    return model


class ColumnParallelLinear(nn.Module):
    """
    Column Parallel Linear layer for tensor parallelism.

    Implements Y = XW + b where the weight matrix W is parallelized along its
    second dimension (column-wise). Each GPU holds a subset of the columns:
    W = [W_1, W_2, ..., W_p] where p is the tensor parallel world size.

    The forward pass computes Y_i = XW_i + b_i on each GPU, where Y_i
    represents a subset of the output columns.

    Args:
        in_features: Input feature dimension (first dimension of W).
        out_features: Output feature dimension (second dimension of W).
        bias: Whether to include bias term.
        gather_output: Whether to gather outputs from all GPUs (used for final layers).
        async_all_reduce: Whether to use asynchronous all-reduce operations.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        gather_output: bool = False,
        async_all_reduce: bool = True,
        sequence_parallel: bool = False,
    ) -> None:
        super().__init__()

        self.tp_world_size = pgm.tp_world_size
        self.tp_rank = pgm.tp_rank

        self.in_features = in_features
        self.out_features = out_features
        self.sequence_parallel = sequence_parallel

        # Ensure output features are divisible by tensor parallel world size
        if out_features % self.tp_world_size != 0:
            raise ValueError(
                f"Output features ({out_features}) must be divisible by "
                f"tensor parallel world size ({self.tp_world_size})"
            )

        self.output_size_per_partition = out_features // self.tp_world_size
        self.gather_output = gather_output
        self.async_all_reduce = async_all_reduce

        # Initialize weight parameter (note: transposed for F.linear)
        self.weight = nn.Parameter(
            torch.empty(
                self.output_size_per_partition, in_features, dtype=torch.float32
            )
        )

        # Initialize bias if requested
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.output_size_per_partition))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters using uniform distribution adapted for tensor parallelism.

        Uses direct partition initialization to avoid allocating the full
        unsharded weight tensor, saving significant memory for large models.
        """
        bound = math.sqrt(1.0 / self.in_features)
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of column parallel linear layer.

        Args:
            x: Input tensor of shape (batch_size, seq_len, in_features).

        Returns:
            Output tensor of shape (batch_size, seq_len, output_size_per_partition)
            or (batch_size, seq_len, out_features) if gather_output is True.

        Raises:
            RuntimeError: If input tensor has incompatible shape.
        """
        if x.size(-1) != self.in_features:
            raise RuntimeError(
                f"Input tensor last dimension ({x.size(-1)}) must match "
                f"in_features ({self.in_features})"
            )

        # Apply linear transformation
        if self.sequence_parallel:
            # SP handles communication — just do local linear
            output = F.linear(x, self.weight, self.bias)
        elif self.async_all_reduce:
            output = linear_with_async_all_reduce(x, self.weight, self.bias)
        else:
            output = linear_with_all_reduce(x, self.weight, self.bias)

        # Optionally gather outputs from all tensor parallel ranks
        if self.gather_output:
            output = GatherFromModelParallelRegion.apply(output)

        return output

    def extra_repr(self) -> str:
        """String representation of the layer."""
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"tp_world_size={self.tp_world_size}, "
            f"tp_rank={self.tp_rank}, "
            f"gather_output={self.gather_output}"
        )


class RowParallelLinear(nn.Module):
    """
    Row Parallel Linear layer for tensor parallelism.

    Implements Y = XW + b where the weight matrix W is parallelized along its
    first dimension (row-wise) and input X is parallelized along its second dimension:

    W = [W_1; W_2; ...; W_p] (stacked vertically)
    X = [X_1, X_2, ..., X_p] (concatenated horizontally)

    The forward pass computes Y = sum(X_i * W_i) + b across all GPUs.

    Args:
        in_features: Input feature dimension (first dimension of W).
        out_features: Output feature dimension (second dimension of W).
        bias: Whether to include bias term.
        async_all_reduce: Whether to use asynchronous all-reduce operations.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        async_all_reduce: bool = True,
        sequence_parallel: bool = False,
    ) -> None:
        super().__init__()

        self.tp_world_size = pgm.tp_world_size
        self.tp_rank = pgm.tp_rank
        self.async_all_reduce = async_all_reduce
        self.sequence_parallel = sequence_parallel

        self.in_features = in_features
        self.out_features = out_features

        # Ensure input features are divisible by tensor parallel world size
        if in_features % self.tp_world_size != 0:
            raise ValueError(
                f"Input features ({in_features}) must be divisible by "
                f"tensor parallel world size ({self.tp_world_size})"
            )

        self.input_size_per_partition = in_features // self.tp_world_size

        # Initialize weight parameter
        self.weight = nn.Parameter(
            torch.empty(
                out_features, self.input_size_per_partition, dtype=torch.float32
            )
        )

        # Initialize bias if requested
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters using uniform distribution adapted for tensor parallelism.

        Uses direct partition initialization to avoid allocating the full
        unsharded weight tensor, saving significant memory for large models.
        """
        bound = math.sqrt(1.0 / self.in_features)
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of row parallel linear layer.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size_per_partition).

        Returns:
            Output tensor of shape (batch_size, seq_len, out_features).

        Raises:
            RuntimeError: If input tensor has incompatible shape.
        """
        if x.size(-1) != self.input_size_per_partition:
            raise RuntimeError(
                f"Input tensor last dimension ({x.size(-1)}) must match "
                f"input_size_per_partition ({self.input_size_per_partition})"
            )

        # Apply local linear transformation
        output_parallel = F.linear(x, self.weight)

        if self.sequence_parallel:
            # SP handles communication via reduce-scatter — skip all-reduce
            output = output_parallel
        else:
            # All-reduce across tensor parallel ranks to sum partial results
            output = ReduceFromModelParallelRegion.apply(output_parallel)

        # Add bias if present
        return output if self.bias is None else output + self.bias

    def extra_repr(self) -> str:
        """String representation of the layer."""
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"tp_world_size={self.tp_world_size}, "
            f"tp_rank={self.tp_rank}"
        )


class VocabParallelEmbedding(nn.Module):
    """
    Vocabulary Parallel Embedding layer for tensor parallelism.

    Distributes the embedding matrix across tensor parallel ranks along the
    vocabulary dimension. Each GPU holds a subset of the embedding vectors.

    Args:
        num_embeddings: Total number of embeddings in the vocabulary.
        embedding_dim: Dimension of each embedding vector.
        padding_idx: Index of padding token (optional).
        max_norm: Maximum norm for embedding vectors (optional).
        norm_type: Type of norm to use for max_norm.
        scale_grad_by_freq: Whether to scale gradients by token frequency.
        sparse: Whether to use sparse gradients.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
    ) -> None:
        super().__init__()

        self.tp_world_size = pgm.tp_world_size
        self.tp_rank = pgm.tp_rank

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

        # Ensure vocabulary size is divisible by tensor parallel world size
        if num_embeddings % self.tp_world_size != 0:
            raise ValueError(
                f"Number of embeddings ({num_embeddings}) must be divisible by "
                f"tensor parallel world size ({self.tp_world_size})"
            )

        # Calculate vocabulary range for this tensor parallel rank
        self.vocab_start_index, self.vocab_end_index = self._calculate_vocab_range(
            num_embeddings, self.tp_rank, self.tp_world_size
        )
        self.num_embeddings_per_partition = (
            self.vocab_end_index - self.vocab_start_index
        )

        # Initialize embedding weight parameter
        self.weight = nn.Parameter(
            torch.empty(
                self.num_embeddings_per_partition, embedding_dim, dtype=torch.float32
            )
        )

        self.reset_parameters()

    def _calculate_vocab_range(
        self, global_vocab_size: int, rank: int, world_size: int
    ) -> tuple[int, int]:
        """
        Calculate the vocabulary range assigned to a specific tensor parallel rank.

        Args:
            global_vocab_size: Total vocabulary size.
            rank: Tensor parallel rank.
            world_size: Tensor parallel world size.

        Returns:
            Tuple of (start_index, end_index) for the vocabulary range.
        """
        per_partition_vocab_size = global_vocab_size // world_size
        start_index = rank * per_partition_vocab_size
        end_index = start_index + per_partition_vocab_size
        return start_index, end_index

    def reset_parameters(self) -> None:
        """Initialize embedding parameters using normal distribution.

        Initializes directly on the partition to avoid allocating the full
        vocabulary embedding matrix on every rank.
        """
        nn.init.normal_(
            self.weight, mean=0.0, std=math.sqrt(1.0 / self.embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of vocabulary parallel embedding layer.

        Handles out-of-vocabulary tokens by masking and setting their embeddings to zero.

        Args:
            x: Input token indices of shape (batch_size, sequence_length).

        Returns:
            Embedded tokens of shape (batch_size, sequence_length, embedding_dim).

        Raises:
            RuntimeError: If input tensor has incompatible dtype or shape.
        """
        if x.dtype not in (torch.long, torch.int):
            raise RuntimeError(f"Input must be integer type, got {x.dtype}")

        # Create mask for out-of-vocabulary tokens
        input_mask = (x < self.vocab_start_index) | (x >= self.vocab_end_index)

        # Adjust token indices to local vocabulary range
        masked_input = x - self.vocab_start_index
        masked_input[input_mask] = 0  # Use index 0 for OOV tokens

        # Perform embedding lookup
        output_parallel = F.embedding(
            masked_input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

        # Zero out embeddings for out-of-vocabulary tokens
        output_parallel = output_parallel.masked_fill(input_mask.unsqueeze(-1), 0.0)

        # All-reduce across tensor parallel ranks
        output = ReduceFromModelParallelRegion.apply(output_parallel)

        return output

    def extra_repr(self) -> str:
        """String representation of the layer."""
        return (
            f"num_embeddings={self.num_embeddings}, "
            f"embedding_dim={self.embedding_dim}, "
            f"tp_world_size={self.tp_world_size}, "
            f"tp_rank={self.tp_rank}, "
            f"vocab_range=[{self.vocab_start_index}, {self.vocab_end_index})"
        )
