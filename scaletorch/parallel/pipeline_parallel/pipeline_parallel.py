"""
Pipeline Parallel Implementation for Distributed Training.

This module implements pipeline parallelism by distributing model layers across multiple GPUs.
It supports two main scheduling strategies:
1. All-Forward-All-Backward (AFAB) - Sequential forward and backward passes
2. 1F1B (One-Forward-One-Backward) - Interleaved forward and backward passes for better GPU utilization
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from scaletorch.parallel.pipeline_parallel.pp_comms import (
    bidirectional_pipeline_communicate,
    pipeline_communicate,
)
from scaletorch.parallel.process_group import process_group_manager as pgm
from scaletorch.utils.logger_utils import get_logger

# Configure logging
logger = get_logger(__name__)


class PipelineParallel(nn.Module):
    """
    Implements pipeline parallelism by distributing model layers across multiple GPUs.

    Each GPU processes a subset of the model's layers in a pipeline fashion, where
    activations are passed between stages. The first stage handles input embeddings,
    intermediate stages process transformer layers, and the final stage handles output
    projection and loss computation.

    Attributes:
        layer_distribution: List of layer indices assigned to this pipeline stage
        embedding: Embedding layer (first stage only) or Identity
        decoder_layers: ModuleDict of transformer layers assigned to this stage
        final_norm: Final layer normalization (last stage only) or Identity
        final_proj: Final projection layer (last stage only) or Identity
    """

    def __init__(self, model: nn.Module, config: Any) -> None:
        """
        Initialize the PipelineParallel module.

        Args:
            model: The complete model to be distributed across pipeline stages
            config: Configuration object containing model parameters including num_hidden_layers

        Raises:
            AttributeError: If config doesn't have required attributes
            RuntimeError: If process group manager is not properly initialized
        """
        super().__init__()

        # Validate inputs
        if not hasattr(config, "num_hidden_layers"):
            raise AttributeError("Config must have 'num_hidden_layers' attribute")

        # Determine layer distribution for this pipeline stage
        self.layer_distribution = self._distribute_layers(config.num_hidden_layers)

        # Configure stage-specific components
        self.embedding = self._get_embedding_layer(model)
        self.decoder_layers = self._get_decoder_layers(model)
        self.final_norm = self._get_final_norm_layer(model)
        self.final_proj = self._get_final_proj_layer(model)

        # Initialize parameters
        self.reset_parameters()

        logger.info(
            "Pipeline stage %d initialized with layers %s",
            pgm.pp_rank,
            self.layer_distribution,
        )

    def _distribute_layers(
        self, num_layers: int, custom_distribution: list[list[int]] | None = None
    ) -> list[int]:
        """
        Distribute model layers across pipeline stages as evenly as possible.

        Uses a greedy algorithm to assign layers to stages, ensuring that earlier
        stages get an extra layer when the total number of layers is not evenly
        divisible by the number of pipeline stages.

        Args:
            num_layers: Total number of layers in the model
            custom_distribution: Optional explicit per-stage layer index lists.
                If provided, must have length == pp_world_size and this stage's
                list is returned directly.

        Returns:
            List of layer indices assigned to this pipeline stage

        Example:
            For num_layers=10 and pp_world_size=3:
            - Stage 0: [0, 1, 2, 3] (4 layers)
            - Stage 1: [4, 5, 6] (3 layers)
            - Stage 2: [7, 8, 9] (3 layers)
        """
        pp_world_size = pgm.pp_world_size
        pp_rank = pgm.pp_rank

        if custom_distribution is not None:
            if len(custom_distribution) != pp_world_size:
                raise ValueError(
                    f"custom_distribution must have {pp_world_size} entries "
                    f"(one per PP stage), got {len(custom_distribution)}"
                )
            return custom_distribution[pp_rank]

        # Calculate base layers per stage and remainder
        base_layers_per_stage = num_layers // pp_world_size
        remainder = num_layers % pp_world_size

        # Distribute remainder layers to first 'remainder' stages
        layers_per_stage = [
            base_layers_per_stage + (1 if i < remainder else 0)
            for i in range(pp_world_size)
        ]

        # Calculate starting layer index for this stage
        start_layer = sum(layers_per_stage[:pp_rank])
        end_layer = start_layer + layers_per_stage[pp_rank]

        return list(range(start_layer, end_layer))

    def _get_embedding_layer(self, model: nn.Module) -> nn.Module:
        """Get embedding layer for first stage, Identity for others."""
        if pgm.pp_is_first_stage:
            if not hasattr(model, "embedding"):
                raise AttributeError(
                    "Model must have 'embedding' layer for first stage"
                )
            return model.embedding
        return nn.Identity()

    def _get_decoder_layers(self, model: nn.Module) -> nn.ModuleDict:
        """Get decoder layers assigned to this pipeline stage."""
        if not hasattr(model, "decoder_layers"):
            raise AttributeError("Model must have 'decoder_layers' attribute")

        decoder_layers = nn.ModuleDict()
        for layer_idx in self.layer_distribution:
            if layer_idx >= len(model.decoder_layers):
                raise IndexError(
                    f"Layer index {layer_idx} exceeds model decoder layers"
                )
            decoder_layers[str(layer_idx)] = model.decoder_layers[layer_idx]

        return decoder_layers

    def _get_final_norm_layer(self, model: nn.Module) -> nn.Module:
        """Get final normalization layer for last stage, Identity for others."""
        if pgm.pp_is_last_stage:
            if not hasattr(model, "final_norm"):
                raise AttributeError(
                    "Model must have 'final_norm' layer for last stage"
                )
            return model.final_norm
        return nn.Identity()

    def _get_final_proj_layer(self, model: nn.Module) -> nn.Module:
        """Get final projection layer for last stage, Identity for others."""
        if pgm.pp_is_last_stage:
            if not hasattr(model, "final_proj"):
                raise AttributeError(
                    "Model must have 'final_proj' layer for last stage"
                )
            return model.final_proj
        return nn.Identity()

    def reset_parameters(self) -> None:
        """
        Initialize or reset all model parameters for this pipeline stage.

        Only resets parameters for components that exist in this stage
        (embedding for first stage, decoder layers for all stages,
        final layers for last stage).
        """
        if pgm.pp_is_first_stage:
            self.embedding.reset_parameters()

        for layer in self.decoder_layers.values():
            layer.input_layernorm.reset_parameters()
            layer.attention.reset_parameters()
            layer.post_attention_layernorm.reset_parameters()
            layer.mlp.reset_parameters()

        if pgm.pp_is_last_stage:
            self.final_norm.reset_parameters()
            self.final_proj.reset_parameters()

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass for this pipeline stage.

        Processes input through assigned layers and produces output for next stage.

        Args:
            input_ids: Input token indices (first stage only)
            position_ids: Position indices for positional embeddings
            hidden_states: Hidden states from previous stage (intermediate/last stages)

        Returns:
            Output tensor to be passed to next pipeline stage

        Raises:
            ValueError: If input validation fails
        """
        # Validate inputs
        if hidden_states is None and not pgm.pp_is_first_stage:
            raise ValueError("Hidden states required for non-first stages")

        # Determine input tensor
        x = hidden_states if hidden_states is not None else input_ids

        # Process through embedding (first stage only)
        x = self.embedding(x)

        # Process through assigned decoder layers
        for layer in self.decoder_layers.values():
            x = layer(x, position_ids=position_ids)

        # Apply final normalization and projection (last stage only)
        x = self.final_norm(x)
        return self.final_proj(x)

    def backward(
        self,
        input_tensor: torch.Tensor | None,
        output_tensor: torch.Tensor,
        output_tensor_grad: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """
        Backward pass for this pipeline stage.

        Computes gradients for assigned layers using gradient from next stage.

        Args:
            input_tensor: Input tensor from forward pass (for gradient computation)
            output_tensor: Output tensor from forward pass
            output_tensor_grad: Gradient tensor from next pipeline stage

        Returns:
            Gradient tensor for input (to be passed to previous stage)

        Raises:
            ValueError: If output_tensor_grad is None and not last stage
        """
        # Retain input gradient if input exists
        if input_tensor is not None:
            input_tensor.retain_grad()

        # Handle gradient for last stage (backward from stored loss)
        if output_tensor_grad is None:
            if not pgm.pp_is_last_stage:
                raise ValueError(
                    "output_tensor_grad cannot be None for non-last stages"
                )
            if not hasattr(self, "_pipeline_loss") or self._pipeline_loss is None:
                raise RuntimeError(
                    "No loss stored for backward. "
                    "Ensure forward step computes loss on last stage."
                )
            self._pipeline_loss.backward(retain_graph=False, create_graph=False)
            self._pipeline_loss = None
            return input_tensor.grad if input_tensor is not None else None

        # Compute backward pass
        torch.autograd.backward(
            output_tensor,
            grad_tensors=output_tensor_grad,
            retain_graph=False,
            create_graph=False,
        )

        return input_tensor.grad if input_tensor is not None else None


def _forward_step(
    model: PipelineParallel,
    data_loader: Any,
    input_tensor: torch.Tensor | None,
    device: torch.device,
    logging_loss_ref: list,
) -> torch.Tensor:
    """
    Perform a single forward step for pipeline parallel training.

    Args:
        model: Pipeline parallel model instance
        data_loader: Data loader providing training batches
        input_tensor: Input activation from previous stage (None for first stage)
        device: Target device for computations
        logging_loss_ref: Mutable [float] for accumulating loss

    Returns:
        Output activation for next stage

    Raises:
        RuntimeError: If data loader is exhausted
    """
    try:
        batch = next(data_loader)
    except StopIteration:
        raise RuntimeError("Data loader exhausted during forward step") from None

    batch["hidden_states"] = (
        input_tensor.to(device) if input_tensor is not None else input_tensor
    )

    output_tensor = model.forward(
        input_ids=batch["input_ids"].to(device),
        position_ids=batch["position_ids"].to(device),
        hidden_states=batch["hidden_states"],
    )

    if pgm.pp_is_last_stage:
        loss = F.cross_entropy(
            output_tensor.flatten(0, 1),
            batch["target_ids"].to(device).flatten(),
            reduction="mean",
        )
        logging_loss_ref[0] += loss.item()
        model._pipeline_loss = loss

    return output_tensor


def train_step_pipeline_afab(
    model: PipelineParallel,
    data_loader: Any,
    tensor_shapes: list[int] | tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    """
    Implements All-Forward-All-Backward (AFAB) pipeline parallel training.

    This strategy first performs all forward passes sequentially, then all backward passes.
    Simple to implement but creates pipeline bubbles where GPUs are idle.

    Pipeline execution:
    1. Forward phase: All microbatches processed forward through pipeline
    2. Backward phase: All microbatches processed backward through pipeline

    Args:
        model: Pipeline parallel model instance
        data_loader: Data loader providing training batches with gradient_accumulation_steps attribute
        tensor_shapes: Expected shapes of tensors for inter-stage communication
        device: Target device for computations
        dtype: Data type for tensors

    Returns:
        Average loss across all microbatches

    Raises:
        RuntimeError: If data loader is exhausted or communication fails
        ValueError: If input validation fails
    """
    # Validate inputs
    if not hasattr(data_loader, "gradient_accumulation_steps"):
        raise ValueError(
            "Data loader must have 'gradient_accumulation_steps' attribute"
        )

    if data_loader.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")

    logging_loss_ref: list = [0.0]
    input_tensors: list[torch.Tensor | None] = []
    output_tensors: list[torch.Tensor] = []
    requires_grad_sync = pgm.cp_dp_world_size > 1
    num_microbatches = data_loader.gradient_accumulation_steps
    phase: str = "init"

    logger.debug(f"Starting AFAB training with {num_microbatches} microbatches")

    try:
        # === Forward Phase ===
        phase = "forward"
        for microbatch_idx in range(num_microbatches):
            logger.debug("Forward microbatch %d", microbatch_idx)

            input_tensor = pipeline_communicate(
                operation="recv_forward",
                shapes=tensor_shapes,
                device=device,
                dtype=dtype,
            )
            input_tensors.append(input_tensor)

            output_tensor = _forward_step(
                model, data_loader, input_tensor, device, logging_loss_ref
            )
            output_tensors.append(output_tensor)

            pipeline_communicate(
                operation="send_forward",
                tensor=output_tensor,
                device=device,
                dtype=dtype,
            )

        # === Backward Phase ===
        phase = "backward"
        for microbatch_idx in range(num_microbatches):
            logger.debug("Backward microbatch %d", microbatch_idx)

            if requires_grad_sync:
                # sync deferred to after all PP comms
                pass

            output_tensor_grad = pipeline_communicate(
                operation="recv_backward",
                shapes=tensor_shapes,
                device=device,
                dtype=dtype,
            )

            # Retrieve saved tensors (FIFO order)
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            # Backward pass
            input_tensor_grad = model.backward(
                input_tensor, output_tensor, output_tensor_grad
            )

            # Send gradient to previous stage
            pipeline_communicate(
                operation="send_backward",
                tensor=input_tensor_grad,
                device=device,
                dtype=dtype,
            )

    except Exception as e:
        logger.error(
            "AFAB training failed during %s phase: %s", phase, e, exc_info=True
        )
        raise RuntimeError(f"AFAB training failed: {e}") from e

    # After all PP comms are done, trigger DP grad sync
    if requires_grad_sync and hasattr(model, "sync_grads_manually"):
        model.sync_grads_manually()

    logging_loss = logging_loss_ref[0] / num_microbatches
    logger.debug("AFAB training completed with loss: %s", logging_loss)
    return logging_loss


def train_step_pipeline_1f1b(
    model: PipelineParallel,
    data_loader: Any,
    tensor_shapes: list[int] | tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    """
    Implements 1F1B (one-forward-one-backward) pipeline parallel training.

    This strategy interleaves forward and backward passes to reduce pipeline bubbles
    and improve GPU utilization. More complex but provides better performance.

    Pipeline execution:
    1. Warmup phase: Forward passes to fill pipeline
    2. Steady state: Alternating forward and backward passes
    3. Cooldown phase: Remaining backward passes

    Args:
        model: Pipeline parallel model instance
        data_loader: Data loader providing training batches with gradient_accumulation_steps attribute
        tensor_shapes: Expected shapes of tensors for inter-stage communication
        device: Target device for computations
        dtype: Data type for tensors

    Returns:
        Average loss across all microbatches

    Raises:
        RuntimeError: If data loader is exhausted or communication fails
        ValueError: If input validation fails
    """
    # Validate inputs
    if not hasattr(data_loader, "gradient_accumulation_steps"):
        raise ValueError(
            "Data loader must have 'gradient_accumulation_steps' attribute"
        )

    if data_loader.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")

    # Calculate pipeline scheduling parameters
    pp_rank = pgm.pp_rank
    pp_world_size = pgm.pp_world_size
    gradient_accumulation_steps = data_loader.gradient_accumulation_steps

    num_warmup_microbatches = min(
        pp_world_size - pp_rank - 1, gradient_accumulation_steps
    )
    num_microbatches_remaining = gradient_accumulation_steps - num_warmup_microbatches

    logging_loss_ref: list = [0.0]
    input_tensors: list[torch.Tensor | None] = []
    output_tensors: list[torch.Tensor] = []
    requires_grad_sync = pgm.cp_dp_world_size > 1
    phase: str = "init"

    logger.debug(
        f"Starting 1F1B training with {gradient_accumulation_steps} microbatches, "
        f"{num_warmup_microbatches} warmup, {num_microbatches_remaining} steady"
    )

    try:
        # === Warmup Phase ===
        phase = "warmup"
        for warmup_idx in range(num_warmup_microbatches):
            logger.debug("Warmup forward %d", warmup_idx)

            input_tensor = pipeline_communicate(
                operation="recv_forward",
                shapes=tensor_shapes,
                device=device,
                dtype=dtype,
            )

            output_tensor = _forward_step(
                model, data_loader, input_tensor, device, logging_loss_ref
            )

            pipeline_communicate(
                operation="send_forward",
                tensor=output_tensor,
                device=device,
                dtype=dtype,
            )

            # Store tensors for later backward passes
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)

        # === Steady State Phase ===
        phase = "steady"
        if num_microbatches_remaining > 0:
            input_tensor = pipeline_communicate(
                operation="recv_forward",
                shapes=tensor_shapes,
                device=device,
                dtype=dtype,
            )

        # Disable gradient sync during steady state
        if requires_grad_sync:
            model.require_backward_grad_sync = False

        for steady_idx in range(num_microbatches_remaining):
            logger.debug("Steady state %d", steady_idx)

            is_last_iteration = steady_idx == num_microbatches_remaining - 1

            # Forward step
            output_tensor = _forward_step(
                model, data_loader, input_tensor, device, logging_loss_ref
            )

            # Bidirectional communication: send forward, receive backward
            output_tensor_grad = bidirectional_pipeline_communicate(
                operation="send_fwd_recv_bwd",
                send_tensor=output_tensor,
                recv_shapes=tensor_shapes,
                device=device,
                dtype=dtype,
            )

            # Store current tensors
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)

            # Retrieve oldest tensors for backward
            input_tensor, output_tensor = input_tensors.pop(0), output_tensors.pop(0)

            # Enable gradient sync for last iteration when appropriate
            if num_warmup_microbatches == 0 and is_last_iteration:
                pass  # sync deferred to after all PP comms

            # Backward step
            input_tensor_grad = model.backward(
                input_tensor, output_tensor, output_tensor_grad
            )

            # Communication for next iteration
            if is_last_iteration:
                input_tensor = None
                pipeline_communicate(
                    operation="send_backward",
                    tensor=input_tensor_grad,
                    device=device,
                    dtype=dtype,
                )
            else:
                input_tensor = bidirectional_pipeline_communicate(
                    operation="send_bwd_recv_fwd",
                    send_tensor=input_tensor_grad,
                    recv_shapes=tensor_shapes,
                    device=device,
                    dtype=dtype,
                )

        # === Cooldown Phase ===
        phase = "cooldown"
        for cooldown_idx in range(num_warmup_microbatches):
            logger.debug("Cooldown backward %d", cooldown_idx)

            # Configure gradient synchronization
            if requires_grad_sync:
                is_last_iteration = cooldown_idx == num_warmup_microbatches - 1
                pass  # sync deferred to after all PP comms

            # Retrieve stored tensors
            input_tensor, output_tensor = input_tensors.pop(0), output_tensors.pop(0)

            # Receive gradient from next stage
            output_tensor_grad = pipeline_communicate(
                operation="recv_backward",
                shapes=tensor_shapes,
                device=device,
                dtype=dtype,
            )

            # Backward step
            input_tensor_grad = model.backward(
                input_tensor, output_tensor, output_tensor_grad
            )

            # Send gradient to previous stage
            pipeline_communicate(
                operation="send_backward",
                tensor=input_tensor_grad,
                device=device,
                dtype=dtype,
            )

    except Exception as e:
        logger.error(
            "1F1B training failed during %s phase: %s", phase, e, exc_info=True
        )
        raise RuntimeError(f"1F1B training failed: {e}") from e

    # After all PP microbatches and P2P comms are complete, trigger DP grad sync.
    # This avoids concurrent HCCL P2P (PP) + allreduce (DP) on Ascend NPU.
    if requires_grad_sync and hasattr(model, "sync_grads_manually"):
        model.sync_grads_manually()

    logging_loss = logging_loss_ref[0] / gradient_accumulation_steps
    logger.debug("1F1B training completed with loss: %s", logging_loss)
    return logging_loss
