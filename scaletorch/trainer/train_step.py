"""Core training step and gradient utilities."""

from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from scaletorch.data.dataloader import MicroBatchDataLoader


def train_step(
    model: torch.nn.Module,
    data_loader: MicroBatchDataLoader,
    device: torch.device,
    dtype: torch.dtype,
    scaler: GradScaler | None = None,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = False,
    use_no_sync: bool = True,
) -> float:
    """Perform a single training step with gradient accumulation and mixed precision.

    Returns:
        Accumulated loss across all gradient accumulation steps.

    Raises:
        RuntimeError: If data loading or forward/backward pass fails.
        ValueError: If input dimensions are invalid.
    """
    if not model.training:
        model.train()

    accumulation_loss = 0.0

    sync_context = (
        model.no_sync()
        if (
            use_no_sync
            and hasattr(model, "no_sync")
            and gradient_accumulation_steps > 1
        )
        else contextlib.nullcontext()
    )

    with sync_context:
        for i in range(gradient_accumulation_steps):
            try:
                batch = next(data_loader)
            except StopIteration:
                raise RuntimeError(
                    f"Data loader exhausted after {i} gradient accumulation steps. "
                    f"Expected {gradient_accumulation_steps} steps. "
                    "Check your dataset size and gradient accumulation configuration."
                ) from None

            if (
                not isinstance(batch, dict)
                or "input_ids" not in batch
                or "target_ids" not in batch
            ):
                raise ValueError(
                    f"Invalid batch format. Expected dict with input_ids and target_ids, got {type(batch)}"
                )

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            target_ids = batch["target_ids"].to(device, non_blocking=True)

            if input_ids.ndim != 2:
                raise ValueError(
                    f"Expected input_ids to be 2D, got shape {input_ids.shape}"
                )
            if target_ids.ndim not in [1, 2]:
                raise ValueError(
                    f"Expected target_ids to be 1D or 2D, got shape {target_ids.shape}"
                )

            try:
                forward_context = autocast(device_type=device.type, dtype=dtype)

                with forward_context:
                    outputs = model(
                        input_ids=input_ids,
                        gradient_checkpointing=gradient_checkpointing,
                    )

                    batch_size, seq_len = input_ids.shape
                    target_ids_flat = target_ids.reshape(-1)
                    outputs_reshaped = outputs.view(seq_len * batch_size, -1)

                    if outputs_reshaped.size(0) != target_ids_flat.size(0):
                        raise ValueError(
                            f"Shape mismatch: outputs {outputs_reshaped.shape} vs targets {target_ids_flat.shape}"
                        )

                    loss = (
                        F.cross_entropy(
                            outputs_reshaped, target_ids_flat, reduction="mean"
                        )
                        / gradient_accumulation_steps
                    )
            except Exception as e:
                raise RuntimeError(f"Model forward pass failed: {e}") from e

            try:
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            except Exception as e:
                raise RuntimeError(f"Backward pass failed: {e}") from e

            accumulation_loss += loss.detach().item()

            del outputs, outputs_reshaped, target_ids_flat, input_ids, target_ids

    return accumulation_loss


def clip_gradients(model: torch.nn.Module, max_norm: float) -> float:
    """Clip gradients using clip_grad_norm_.

    Returns:
        The gradient norm before clipping, or 0.0 if clipping was skipped.
    """
    if max_norm <= 0:
        return 0.0

    parameters = [p for p in model.parameters() if p.grad is not None]
    if not parameters:
        return 0.0

    total_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm)
    return total_norm.item()
