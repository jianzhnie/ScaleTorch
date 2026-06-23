"""Pipeline parallel point-to-point communication primitives."""

from __future__ import annotations

import os

import torch
import torch.distributed as torch_dist

from scaletorch.env import ENV_VERBOSE
from scaletorch.parallel.process_group import process_group_manager as pgm
from scaletorch.utils.logger_utils import get_logger

logger = get_logger(__name__)

# Communication state — encapsulated to avoid bare module-level mutables
class _CommState:
    """Encapsulates mutable communication state."""

    def __init__(self) -> None:
        self.step: int = 0
        self.verbose: bool = os.environ.get(ENV_VERBOSE, "0") == "1"

_comm_state = _CommState()

# Valid operations for pipeline communication
VALID_OPERATIONS = {"recv_forward", "send_forward", "recv_backward", "send_backward"}

BIDIRECTIONAL_OPERATIONS = {"send_fwd_recv_bwd", "send_bwd_recv_fwd"}


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
            f"Invalid operation: {operation}. Valid operations are: {valid_operations}"
        )


def _log_communication(
    operation: str, is_send: bool, peer_rank: int, direction: str = "forward"
) -> None:
    """Log communication operation if verbose mode is enabled.

    Args:
        operation: Name of the communication operation
        is_send: Whether this is a send operation
        peer_rank: Rank of the peer process
        direction: Direction of communication (forward/backward)
    """
    if not _comm_state.verbose:
        return

    current_rank = pgm.pp_rank
    arrow = "→" if is_send else "←"
    action = "sending" if is_send else "receiving"

    logger.debug(
        "%s | %s %s | Rank %s %s %s | Step: %d",
        operation, action, direction,
        current_rank, arrow, peer_rank,
        _comm_state.step,
    )


def pipeline_communicate(
    operation: str,
    device: torch.device | str,
    dtype: torch.dtype,
    tensor: torch.Tensor | None = None,
    shapes: tuple[int, ...] | None = None,
) -> torch.Tensor | None:
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
    _validate_operation(operation, VALID_OPERATIONS)

    # Initialize variables
    src: int | None = None
    dest: int | None = None
    result_tensor: torch.Tensor | None = None

    # Handle different operation types
    if operation == "recv_forward":
        if pgm.pp_is_first_stage:
            return None
        if shapes is None:
            raise PipelineCommunicationError(
                "shapes must be provided for recv operations"
            )
        result_tensor = torch.empty(
            shapes, requires_grad=True, device=device, dtype=dtype
        )
        src = pgm.pp_prev_rank

    elif operation == "send_forward":
        if pgm.pp_is_last_stage:
            return None
        if tensor is None:
            raise PipelineCommunicationError(
                "tensor must be provided for send operations"
            )
        dest = pgm.pp_next_rank

    elif operation == "recv_backward":
        if pgm.pp_is_last_stage:
            return None
        if shapes is None:
            raise PipelineCommunicationError(
                "shapes must be provided for recv operations"
            )
        result_tensor = torch.empty(
            shapes, requires_grad=True, device=device, dtype=dtype
        )
        src = pgm.pp_next_rank

    elif operation == "send_backward":
        if pgm.pp_is_first_stage:
            return None
        if tensor is None:
            raise PipelineCommunicationError(
                "tensor must be provided for send operations"
            )
        dest = pgm.pp_prev_rank

    # Determine if this is a send operation and get peer rank
    is_send = operation.startswith("send")
    peer_rank = dest if is_send else src

    # Log the communication operation
    direction = "forward" if "forward" in operation else "backward"
    _log_communication(operation, is_send, peer_rank, direction)

    # Execute communication using direct send/recv (more compatible with HCCL)
    if is_send:
        torch_dist.send(tensor, peer_rank, group=pgm.pp_group)
    else:
        torch_dist.recv(result_tensor, peer_rank, group=pgm.pp_group)

    _comm_state.step += 1

    return result_tensor


def bidirectional_pipeline_communicate(
    operation: str,
    send_tensor: torch.Tensor,
    recv_shapes: tuple[int, ...],
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor | None:
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
    _validate_operation(operation, BIDIRECTIONAL_OPERATIONS)

    # Determine operation direction
    is_forward_send = operation == "send_fwd_recv_bwd"

    # Check if operation is applicable to this stage
    if (is_forward_send and pgm.pp_is_last_stage) or (
        not is_forward_send and pgm.pp_is_first_stage
    ):
        return None

    # Determine peer rank
    peer_rank = pgm.pp_next_rank if is_forward_send else pgm.pp_prev_rank

    # Create receive tensor
    recv_tensor = torch.empty(
        recv_shapes, requires_grad=True, device=device, dtype=dtype
    )

    # Log the bidirectional communication
    if _comm_state.verbose:
        direction = "next" if is_forward_send else "prev"
        current_rank = pgm.pp_rank
        logger.debug(
            "%s | sending %s %s -> %s | receiving %s %s -> %s | Step: %d",
            operation, direction, current_rank, peer_rank,
            direction, peer_rank, current_rank,
            _comm_state.step,
        )

    # Create and execute bidirectional communication operations.
    # Use non-blocking isend + blocking recv to avoid deadlock on HCCL.
    send_req = torch_dist.isend(send_tensor, peer_rank, group=pgm.pp_group)
    torch_dist.recv(recv_tensor, peer_rank, group=pgm.pp_group)
    send_req.wait()

    _comm_state.step += 1

    return recv_tensor


def get_communication_stats() -> dict:
    """
    Get current communication statistics.

    Returns:
        Dictionary containing communication statistics
    """
    return {"step": _comm_state.step, "verbose": _comm_state.verbose}


def reset_communication_stats() -> None:
    """Reset communication statistics."""
    _comm_state.step = 0
