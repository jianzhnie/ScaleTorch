"""
Pipeline Parallel Communication Module

This module provides communication primitives for pipeline parallelism in distributed training.
It handles point-to-point tensor transfers between pipeline stages during forward and backward passes.
"""

import os
from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist

import scaletorch.parallel.pg_manager as pgm

# Global state for debugging and monitoring
_STEP: int = 0
_VERBOSE: bool = os.environ.get('VERBOSE', '0') == '1'

# Valid operations for pipeline communication
VALID_OPERATIONS = {
    'recv_forward', 'send_forward', 'recv_backward', 'send_backward'
}

BIDIRECTIONAL_OPERATIONS = {'send_fwd_recv_bwd', 'send_bwd_recv_fwd'}


class PipelineCommunicationError(Exception):
    """Custom exception for pipeline communication errors."""
    pass


def _validate_operation(operation: str, valid_operations: set) -> None:
    """Validate that the operation is supported.

    Args:
        operation: The operation to validate
        valid_operations: Set of valid operation names

    Raises:
        PipelineCommunicationError: If operation is not supported
    """
    if operation not in valid_operations:
        raise PipelineCommunicationError(
            f'Invalid operation: {operation}. '
            f'Valid operations are: {valid_operations}')


def _log_communication(operation: str,
                       is_send: bool,
                       peer_rank: int,
                       direction: str = 'forward') -> None:
    """Log communication operation if verbose mode is enabled.

    Args:
        operation: Name of the communication operation
        is_send: Whether this is a send operation
        peer_rank: Rank of the peer process
        direction: Direction of communication (forward/backward)
    """
    global _STEP, _VERBOSE

    if not _VERBOSE:
        return

    current_rank = pgm.process_group_manager.pp_rank
    arrow = '→' if is_send else '←'
    action = 'sending' if is_send else 'receiving'

    print(
        f'{operation} | {action} {direction} | '
        f'Rank {current_rank} {arrow} {peer_rank} | '
        f'Step: {_STEP} | Rank: {current_rank}',
        flush=True)


def pipeline_communicate(
        operation: str,
        device: Union[torch.device, str],
        dtype: torch.dtype,
        tensor: Optional[torch.Tensor] = None,
        shapes: Optional[Tuple[int, ...]] = None) -> Optional[torch.Tensor]:
    """
    Perform unidirectional pipeline communication operation.

    This function handles point-to-point communication between pipeline stages for
    forward and backward passes. It supports four operations:
    - recv_forward: Receive activations from previous stage
    - send_forward: Send activations to next stage
    - recv_backward: Receive gradients from next stage
    - send_backward: Send gradients to previous stage

    Args:
        operation: Type of communication operation ('recv_forward', 'send_forward',
                   'recv_backward', 'send_backward')
        device: Target device for tensor operations
        dtype: Data type for tensors
        tensor: Input tensor for send operations (optional for recv operations)
        shapes: Shape tuple for creating receive tensors (required for recv operations)

    Returns:
        Received tensor for recv operations, None for send operations

    Raises:
        PipelineCommunicationError: If operation is invalid or parameters are missing

    Example:
        >>> # Receive forward activations
        >>> recv_tensor = pipeline_communicate('recv_forward', 'cuda', torch.float32, shapes=(1024, 512))
        >>> # Send forward activations
        >>> pipeline_communicate('send_forward', 'cuda', torch.float32, tensor=activations)
    """
    global _STEP, _VERBOSE

    # Validate operation
    _validate_operation(operation, VALID_OPERATIONS)

    # Initialize variables
    src: Optional[int] = None
    dest: Optional[int] = None
    result_tensor: Optional[torch.Tensor] = None

    # Handle different operation types
    if operation == 'recv_forward':
        if pgm.process_group_manager.pp_is_first_stage:
            return None
        if shapes is None:
            raise PipelineCommunicationError(
                'shapes must be provided for recv operations')
        result_tensor = torch.empty(shapes,
                                    requires_grad=True,
                                    device=device,
                                    dtype=dtype)
        src = pgm.process_group_manager.pp_prev_rank

    elif operation == 'send_forward':
        if pgm.process_group_manager.pp_is_last_stage:
            return None
        if tensor is None:
            raise PipelineCommunicationError(
                'tensor must be provided for send operations')
        dest = pgm.process_group_manager.pp_next_rank

    elif operation == 'recv_backward':
        if pgm.process_group_manager.pp_is_last_stage:
            return None
        if shapes is None:
            raise PipelineCommunicationError(
                'shapes must be provided for recv operations')
        result_tensor = torch.empty(shapes,
                                    requires_grad=True,
                                    device=device,
                                    dtype=dtype)
        src = pgm.process_group_manager.pp_next_rank

    elif operation == 'send_backward':
        if pgm.process_group_manager.pp_is_first_stage:
            return None
        if tensor is None:
            raise PipelineCommunicationError(
                'tensor must be provided for send operations')
        dest = pgm.process_group_manager.pp_prev_rank

    # Determine if this is a send operation and get peer rank
    is_send = operation.startswith('send')
    peer_rank = dest if is_send else src

    # Log the communication operation
    direction = 'forward' if 'forward' in operation else 'backward'
    _log_communication(operation, is_send, peer_rank, direction)

    # Create and execute the communication operation
    comm_op = dist.P2POp(dist.isend if is_send else dist.irecv,
                         tensor if is_send else result_tensor, peer_rank)

    # Execute communication and wait for completion
    requests = dist.batch_isend_irecv([comm_op])
    for req in requests:
        req.wait()

    # Synchronize CUDA operations
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Update step counter
    if _VERBOSE:
        _STEP += 1

    return result_tensor


def bidirectional_pipeline_communicate(
        operation: str, send_tensor: torch.Tensor, recv_shapes: Tuple[int,
                                                                      ...],
        device: Union[torch.device,
                      str], dtype: torch.dtype) -> Optional[torch.Tensor]:
    """
    Perform bidirectional pipeline communication operation.

    This function handles simultaneous send and receive operations between pipeline stages.
    It supports two operations:
    - send_fwd_recv_bwd: Send forward activations and receive backward gradients
    - send_bwd_recv_fwd: Send backward gradients and receive forward activations

    Args:
        operation: Type of bidirectional operation ('send_fwd_recv_bwd' or 'send_bwd_recv_fwd')
        send_tensor: Tensor to send to peer
        recv_shapes: Shape tuple for the tensor to receive
        device: Target device for tensor operations
        dtype: Data type for tensors

    Returns:
        Received tensor, or None if operation is not applicable to this stage

    Raises:
        PipelineCommunicationError: If operation is invalid

    Example:
        >>> # Send forward activations and receive backward gradients
        >>> recv_grads = bidirectional_pipeline_communicate(
        ...     'send_fwd_recv_bwd', activations, (1024, 512), 'cuda', torch.float32
        ... )
    """
    global _STEP, _VERBOSE

    # Validate operation
    _validate_operation(operation, BIDIRECTIONAL_OPERATIONS)

    # Determine operation direction
    is_forward_send = (operation == 'send_fwd_recv_bwd')

    # Check if operation is applicable to this stage
    if (is_forward_send and pgm.process_group_manager.pp_is_last_stage) or \
       (not is_forward_send and pgm.process_group_manager.pp_is_first_stage):
        return None

    # Determine peer rank
    peer_rank = (pgm.process_group_manager.pp_next_rank if is_forward_send else
                 pgm.process_group_manager.pp_prev_rank)

    # Create receive tensor
    recv_tensor = torch.empty(recv_shapes,
                              requires_grad=True,
                              device=device,
                              dtype=dtype)

    # Log the bidirectional communication
    if _VERBOSE:
        direction = 'next' if is_forward_send else 'prev'
        current_rank = pgm.process_group_manager.pp_rank
        print(
            f'{operation} | sending {direction} {current_rank} -> {peer_rank} | '
            f'receiving {direction} {peer_rank} -> {current_rank} | '
            f'Step: {_STEP} | Rank: {current_rank}',
            flush=True)

    # Create and execute bidirectional communication operations
    send_op = dist.P2POp(dist.isend, send_tensor, peer_rank)
    recv_op = dist.P2POp(dist.irecv, recv_tensor, peer_rank)

    # Execute both operations and wait for completion
    requests = dist.batch_isend_irecv([send_op, recv_op])
    for req in requests:
        req.wait()

    # Synchronize CUDA operations
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Update step counter
    if _VERBOSE:
        _STEP += 1

    return recv_tensor


def get_communication_stats() -> dict:
    """
    Get current communication statistics.

    Returns:
        Dictionary containing communication statistics
    """
    return {'step': _STEP, 'verbose': _VERBOSE}


def reset_communication_stats() -> None:
    """Reset communication statistics."""
    global _STEP
    _STEP = 0
