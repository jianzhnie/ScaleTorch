"""
Context parallel implementation for distributed attention computation.

This module provides ring attention implementation for efficient distributed attention
computation across multiple GPUs/processes. It supports both forward and backward
passes with causal masking capabilities.

Key features:
- Ring-based attention computation for memory efficiency
- Support for causal masking
- Efficient communication with overlap of computation and communication
- Compatible with PyTorch autograd
"""

import logging
import os
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

import scaletorch.parallel.pg_manager as pgm
from scaletorch.parallel.context_parallel.cp_comms import ContextCommunicate

# Configure logging
logger = logging.getLogger(__name__)

# Global configuration constants
CONTEXT_PARALLEL_ENV_VAR: str = 'CONTEXT_PARALLEL'


def apply_context_parallel(model: torch.nn.Module) -> torch.nn.Module:
    """
    Apply context parallel configuration to a model.

    Sets the CONTEXT_PARALLEL environment variable based on the current
    context parallel world size. This can be used by other components
    to detect if context parallel is enabled.

    Args:
        model (torch.nn.Module): The model to apply context parallel to

    Returns:
        torch.nn.Module: The same model (for chaining)

    Raises:
        RuntimeError: If process group manager is not properly initialized
    """
    if not hasattr(pgm, 'process_group_manager'):
        raise RuntimeError('Process group manager not initialized')

    cp_enabled = pgm.process_group_manager.cp_world_size > 1
    os.environ[CONTEXT_PARALLEL_ENV_VAR] = '1' if cp_enabled else '0'

    logger.info(f"Context parallel {'enabled' if cp_enabled else 'disabled'} "
                f'(world_size={pgm.process_group_manager.cp_world_size})')

    return model


def ring_attention(q: Tensor, k: Tensor, v: Tensor, sm_scale: float,
                   is_causal: bool) -> Tensor:
    """
    Compute ring attention using the custom autograd function.

    This is the main entry point for ring attention computation. It handles
    both forward and backward passes automatically through PyTorch's autograd.

    Args:
        q (Tensor): Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
        k (Tensor): Key tensor of shape (batch_size, num_heads, seq_len, head_dim)
        v (Tensor): Value tensor of shape (batch_size, num_heads, seq_len, head_dim)
        sm_scale (float): Softmax scaling factor (typically 1.0 / sqrt(head_dim))
        is_causal (bool): Whether to apply causal masking

    Returns:
        Tensor: Output attention tensor of shape (batch_size, num_heads, seq_len, head_dim)

    Raises:
        ValueError: If input tensors have incompatible shapes or dtypes
    """
    # Input validation
    _validate_attention_inputs(q, k, v, sm_scale)

    return RingAttentionFunc.apply(q, k, v, sm_scale, is_causal)


class RingAttentionFunc(torch.autograd.Function):
    """
    Custom autograd function for ring attention computation.

    This function implements both forward and backward passes for ring attention,
    enabling efficient distributed attention computation with memory efficiency.
    """

    @staticmethod
    def forward(ctx: Any, q: Tensor, k: Tensor, v: Tensor, sm_scale: float,
                is_causal: bool) -> Tensor:
        """
        Forward pass of ring attention.

        Implements the ring attention algorithm where each rank processes a portion
        of the sequence and communicates with neighboring ranks in a ring topology.

        Args:
            ctx: PyTorch autograd context
            q: Query tensor
            k: Key tensor
            v: Value tensor
            sm_scale: Softmax scaling factor
            is_causal: Whether to apply causal masking

        Returns:
            Tensor: Attention output tensor
        """
        comm = ContextCommunicate('ring_attention_forward')

        # Clone original tensors for backward pass
        # TODO: Find a more memory-efficient way to save these tensors
        k_og = k.clone()
        v_og = v.clone()

        out: Optional[Tensor] = None
        lse: Optional[Tensor] = None
        next_k: Optional[Tensor] = None
        next_v: Optional[Tensor] = None

        # Ring attention loop
        for step in range(comm.world_size):
            # Initiate communication for next iteration (if not last step)
            if step + 1 != comm.world_size:
                next_k = comm.send_recv(k)
                next_v = comm.send_recv(v)
                comm.commit()

            # Compute attention for current step
            if not is_causal or step <= comm.rank:
                block_out, block_lse = ring_attention_forward(
                    q, k, v, sm_scale, is_causal and step == 0)
                out, lse = update_out_and_lse(out, lse, block_out, block_lse)

            # Wait for communication to complete (if not last step)
            if step + 1 != comm.world_size:
                comm.wait()
                k = next_k
                v = next_v

        # Convert output to input dtype and save for backward
        out = out.to(q.dtype)
        ctx.save_for_backward(q, k_og, v_og, out, lse.squeeze(-1))
        ctx.sm_scale = sm_scale
        ctx.is_causal = is_causal

        logger.debug(
            f'Ring attention forward completed | Output shape: {out.shape}')

        return out

    @staticmethod
    def backward(ctx: Any, dout: Tensor,
                 *args) -> Tuple[Optional[Tensor], ...]:
        """
        Backward pass of ring attention.

        Computes gradients with respect to query, key, and value tensors using
        the ring attention algorithm.

        Args:
            ctx: PyTorch autograd context
            dout: Gradient of output tensor
            *args: Additional arguments (unused)

        Returns:
            Tuple of gradients (dq, dk, dv, None, None)
        """
        q, k, v, out, softmax_lse = ctx.saved_tensors
        sm_scale = ctx.sm_scale
        is_causal = ctx.is_causal

        # Initialize communication handlers
        kv_comm = ContextCommunicate('ring_attention_backward_kv')
        d_kv_comm = ContextCommunicate('ring_attention_backward_grad')

        dq: Optional[Tensor] = None
        dk: Optional[Tensor] = None
        dv: Optional[Tensor] = None
        next_dk: Optional[Tensor] = None
        next_dv: Optional[Tensor] = None

        # Pre-allocate gradient buffers
        block_dq_buffer = torch.empty_like(q, dtype=torch.float32)
        block_dk_buffer = torch.empty_like(k, dtype=torch.float32)
        block_dv_buffer = torch.empty_like(v, dtype=torch.float32)

        # Backward ring attention loop
        for step in range(kv_comm.world_size):
            # Initiate communication for next iteration (if not last step)
            if step + 1 != kv_comm.world_size:
                next_k = kv_comm.send_recv(k)
                next_v = kv_comm.send_recv(v)
                kv_comm.commit()

            # Compute gradients for current step
            if step <= kv_comm.rank or not is_causal:
                bwd_causal = is_causal and step == 0

                block_dq_buffer, block_dk_buffer, block_dv_buffer = ring_attention_backward(
                    dout, q, k, v, out, softmax_lse, sm_scale, bwd_causal)

                # Accumulate gradients
                if dq is None:
                    dq = block_dq_buffer
                    dk = block_dk_buffer
                    dv = block_dv_buffer
                else:
                    dq += block_dq_buffer
                    d_kv_comm.wait()
                    dk = block_dk_buffer + next_dk
                    dv = block_dv_buffer + next_dv
            elif step != 0:
                d_kv_comm.wait()
                dk = next_dk
                dv = next_dv

            # Wait for communication to complete (if not last step)
            if step + 1 != kv_comm.world_size:
                kv_comm.wait()
                k = next_k
                v = next_v

            # Send gradients to next rank
            next_dk = d_kv_comm.send_recv(dk)
            next_dv = d_kv_comm.send_recv(dv)
            d_kv_comm.commit()

        d_kv_comm.wait()

        logger.debug('Ring attention backward completed')

        return dq, next_dk, next_dv, None, None


def ring_attention_forward(q: Tensor, k: Tensor, v: Tensor, sm_scale: float,
                           is_causal: bool) -> Tuple[Tensor, Tensor]:
    """
    Compute standard attention forward pass for a single ring step.

    Args:
        q: Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
        k: Key tensor of shape (batch_size, num_heads, seq_len, head_dim)
        v: Value tensor of shape (batch_size, num_heads, seq_len, head_dim)
        sm_scale: Softmax scaling factor
        is_causal: Whether to apply causal masking

    Returns:
        Tuple[Tensor, Tensor]: Output tensor and log-sum-exp for numerical stability
    """
    batch_size, num_heads, seq_len, head_dim = q.shape

    # Compute attention scores
    scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale

    # Apply causal masking if requested
    if is_causal:
        causal_mask = torch.triu(torch.ones(seq_len,
                                            seq_len,
                                            device=q.device,
                                            dtype=torch.bool),
                                 diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1).expand(
            batch_size, num_heads, seq_len, seq_len)
        scores.masked_fill_(causal_mask, float('-inf'))

    # Online softmax computation for numerical stability
    scores_max = torch.max(scores, dim=-1, keepdim=True)[0]
    exp_scores = torch.exp(scores - scores_max)
    exp_sum = torch.sum(exp_scores, dim=-1, keepdim=True)
    log_sum_exp = torch.log(exp_sum) + scores_max

    # Compute attention probabilities and output
    attention_probs = exp_scores / exp_sum
    output = torch.matmul(attention_probs, v)

    return output, log_sum_exp.squeeze(-1)


def ring_attention_backward(dO: Tensor, Q: Tensor, K: Tensor, V: Tensor,
                            output: Tensor, softmax_lse: Tensor,
                            sm_scale: float,
                            is_causal: bool) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Compute attention backward pass for a single ring step.

    Args:
        dO: Gradient of output tensor
        Q: Query tensor
        K: Key tensor
        V: Value tensor
        output: Output tensor from forward pass
        softmax_lse: Log-sum-exp from forward pass
        sm_scale: Softmax scaling factor
        is_causal: Whether to apply causal masking was applied

    Returns:
        Tuple[Tensor, Tensor, Tensor]: Gradients with respect to Q, K, V
    """
    batch_size, num_heads, seq_len, head_dim = Q.shape

    # Recreate attention scores and probabilities
    scores = torch.matmul(Q, K.transpose(-2, -1)) * sm_scale
    if is_causal:
        causal_mask = torch.triu(torch.ones(seq_len,
                                            seq_len,
                                            device=Q.device,
                                            dtype=torch.bool),
                                 diagonal=1)
        scores = scores.masked_fill(
            causal_mask.unsqueeze(0).unsqueeze(1), float('-inf'))

    attention_probs = torch.exp(scores - softmax_lse.unsqueeze(-1))

    # Compute gradients step by step
    # Step 1: Gradient with respect to V
    dV = torch.matmul(attention_probs.transpose(-2, -1), dO)

    # Step 2: Gradient with respect to attention probabilities
    dAttention = torch.matmul(dO, V.transpose(-2, -1))

    # Step 3: Normalization term
    normalization = torch.sum(dO * output, dim=-1, keepdim=True)

    # Step 4: Gradient with respect to scores
    dScores = attention_probs * (dAttention - normalization)

    # Apply causal masking if requested
    if is_causal:
        dScores = dScores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(1), 0)

    # Step 5: Gradient with respect to Q
    dQ = torch.matmul(dScores, K) * sm_scale

    # Step 6: Gradient with respect to K
    dK = torch.matmul(dScores.transpose(-2, -1), Q) * sm_scale

    return dQ, dK, dV


def update_out_and_lse(
        out: Optional[Tensor],
        lse: Optional[Tensor],
        block_out: Tensor,
        block_lse: Tensor,
        slice_: Optional[slice] = None) -> Tuple[Tensor, Tensor]:
    """
    Update output and log-sum-exp using stable numerical techniques.

    This function implements a numerically stable way to accumulate attention
    outputs and log-sum-exp values across multiple ring steps.

    Args:
        out: Current accumulated output (None for first update)
        lse: Current accumulated log-sum-exp (None for first update)
        block_out: New output block to incorporate
        block_lse: New log-sum-exp block to incorporate
        slice_: Optional slice for partial updates

    Returns:
        Tuple[Tensor, Tensor]: Updated output and log-sum-exp

    Raises:
        RuntimeError: If slice_ is provided on first update
    """

    def _update(current_out: Tensor,
                current_lse: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Numerically stable update using sigmoid and logsigmoid.

        Reference: https://github.com/zhuzilin/ring-flash-attention/pull/34#issuecomment-2076126795
        """
        current_out = current_out - F.sigmoid(block_lse - current_lse) * (
            current_out - block_out)
        current_lse = current_lse - F.logsigmoid(current_lse - block_lse)
        return current_out, current_lse

    # Convert to appropriate dtypes
    block_out = block_out.to(torch.float32)
    block_lse = block_lse.unsqueeze(dim=-1)

    # Handle first update
    if out is None:
        if slice_ is not None:
            raise RuntimeError(
                'First update_out_and_lse should not pass slice_ args')
        return block_out, block_lse

    # Apply update (with optional slicing)
    if slice_ is not None:
        out[slice_], lse[slice_] = _update(out[slice_], lse[slice_])
    else:
        out, lse = _update(out, lse)

    return out, lse


def update_rope_for_context_parallel(cos: Tensor,
                                     sin: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Update RoPE (Rotary Position Embedding) tensors for context parallel processing.

    This function partitions the RoPE tensors across context parallel ranks,
    ensuring each rank gets the appropriate portion for its sequence segment.

    Args:
        cos: Cosine component of RoPE
        sin: Sine component of RoPE

    Returns:
        Tuple[Tensor, Tensor]: Partitioned cos and sin tensors for current rank

    Raises:
        ValueError: If sequence length is not divisible by context parallel world size
        RuntimeError: If process group manager is not properly initialized
    """
    if not hasattr(pgm, 'process_group_manager'):
        raise RuntimeError('Process group manager not initialized')

    seq_len, _ = cos.size()
    cp_rank = pgm.process_group_manager.cp_rank
    cp_world_size = pgm.process_group_manager.cp_world_size

    # Validate divisibility
    if seq_len % cp_world_size != 0:
        raise ValueError(
            f'Input sequence length ({seq_len}) must be divisible by '
            f'context parallel world size ({cp_world_size})')

    # Calculate partition boundaries
    size_per_partition = seq_len // cp_world_size
    start_idx = cp_rank * size_per_partition
    end_idx = (cp_rank + 1) * size_per_partition

    logger.debug(f'RoPE update | Rank {cp_rank}/{cp_world_size} | '
                 f'Partition: [{start_idx}, {end_idx}) / {seq_len}')

    return cos[start_idx:end_idx], sin[start_idx:end_idx]


def _validate_attention_inputs(q: Tensor, k: Tensor, v: Tensor,
                               sm_scale: float) -> None:
    """
    Validate input tensors for attention computation.

    Args:
        q: Query tensor
        k: Key tensor
        v: Value tensor
        sm_scale: Softmax scaling factor

    Raises:
        ValueError: If inputs are invalid
    """
    # Check tensor types
    if not all(isinstance(tensor, torch.Tensor) for tensor in [q, k, v]):
        raise ValueError('All inputs must be torch.Tensor instances')

    # Check tensor dtypes
    if not (q.dtype == k.dtype == v.dtype):
        raise ValueError('All tensors must have the same dtype')

    # Check tensor shapes
    if not (q.shape == k.shape == v.shape):
        raise ValueError('All tensors must have the same shape')

    if len(q.shape) != 4:
        raise ValueError(
            'Tensors must be 4D: (batch_size, num_heads, seq_len, head_dim)')

    # Check scaling factor
    if sm_scale <= 0:
        raise ValueError('sm_scale must be positive')

    # Check device consistency
    if not (q.device == k.device == v.device):
        raise ValueError('All tensors must be on the same device')
