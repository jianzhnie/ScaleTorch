"""
Context parallel communication utilities for distributed attention computation.

This module provides efficient communication primitives for context parallel processing,
including ring-based send/receive operations with batch processing capabilities.
"""

import logging
import os
from typing import List, Optional

import torch
from torch import distributed as dist

import scaletorch.parallel.pg_manager as pgm

# Configure logging
logger = logging.getLogger(__name__)

# Global state variables (consider removing in future versions)
STEP: int = 0
VERBOSE: bool = os.environ.get('VERBOSE', '0') == '1'


class ContextCommunicate:
    """
    Context parallel communication handler for ring-based distributed operations.

    This class manages point-to-point communication operations in a ring topology
    for context parallel processing. It supports batching multiple send/receive
    operations for improved performance.

    Attributes:
        rank (int): Current process rank in context parallel group
        world_size (int): Total number of processes in context parallel group
        send_rank (int): Rank to send data to in the ring
        recv_rank (int): Rank to receive data from in the ring
    """

    def __init__(self, msg: str = '') -> None:
        """
        Initialize context parallel communication handler.

        Args:
            msg (str): Optional message for debugging/identification purposes

        Raises:
            RuntimeError: If process group manager is not properly initialized
        """
        global STEP, VERBOSE

        self._pending_operations: List[dist.P2POp] = []
        self._active_requests: Optional[List[dist.Work]] = None

        # Validate process group manager availability
        if not hasattr(pgm, 'process_group_manager'):
            raise RuntimeError('Process group manager not initialized')

        self.rank: int = pgm.process_group_manager.cp_rank
        self.world_size: int = pgm.process_group_manager.cp_world_size
        self.send_rank: int = pgm.process_group_manager.cp_send_rank
        self.recv_rank: int = pgm.process_group_manager.cp_recv_rank

        if VERBOSE:
            logger.info(
                f'ContextCommunicate ({msg}) initialized | '
                f'RANK: {self.rank} | WORLD_SIZE: {self.world_size} | '
                f'SEND_RANK: {self.send_rank} | RECV_RANK: {self.recv_rank}')

    def send_recv(self,
                  tensor_to_send: torch.Tensor,
                  recv_tensor: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Perform asynchronous send and receive operations in the ring topology.

        Args:
            tensor_to_send (torch.Tensor): Tensor to send to the next rank
            recv_tensor (torch.Tensor, optional): Pre-allocated tensor for receiving data.
                If None, a new tensor will be created with the same shape and dtype as tensor_to_send.

        Returns:
            torch.Tensor: The received tensor

        Raises:
            ValueError: If tensor shapes or dtypes are incompatible
            RuntimeError: If communication operations fail
        """
        global STEP, VERBOSE

        # Input validation
        if not isinstance(tensor_to_send, torch.Tensor):
            raise ValueError('tensor_to_send must be a torch.Tensor')

        if tensor_to_send.numel() == 0:
            raise ValueError('Cannot send empty tensor')

        # Create result tensor if not provided
        if recv_tensor is None:
            result_tensor = torch.zeros_like(tensor_to_send)
        else:
            # Validate compatibility
            if recv_tensor.shape != tensor_to_send.shape:
                raise ValueError(
                    f'Shape mismatch: send {tensor_to_send.shape} vs recv {recv_tensor.shape}'
                )
            if recv_tensor.dtype != tensor_to_send.dtype:
                raise ValueError(
                    f'Dtype mismatch: send {tensor_to_send.dtype} vs recv {recv_tensor.dtype}'
                )
            if recv_tensor.device != tensor_to_send.device:
                raise ValueError(
                    f'Device mismatch: send {tensor_to_send.device} vs recv {recv_tensor.device}'
                )
            result_tensor = recv_tensor

        try:
            # Create send operation
            send_operation = dist.P2POp(
                dist.isend,
                tensor_to_send,
                self.send_rank,
                group=pgm.process_group_manager.cp_group)

            # Create receive operation
            recv_operation = dist.P2POp(
                dist.irecv,
                result_tensor,
                self.recv_rank,
                group=pgm.process_group_manager.cp_group)

            # Add operations to pending list
            self._pending_operations.extend([send_operation, recv_operation])

            if VERBOSE:
                logger.debug(
                    f'ContextCommunicate | send_recv | STEP: {STEP} | RANK: {self.rank} | '
                    f'Sending to rank {self.send_rank}, receiving from rank {self.recv_rank} | '
                    f'Tensor shape: {tensor_to_send.shape}, dtype: {tensor_to_send.dtype}'
                )

        except Exception as e:
            raise RuntimeError(
                f'Failed to create send/recv operations: {e}') from e

        return result_tensor

    def commit(self) -> None:
        """
        Commit all pending communication operations for batch execution.

        This method batches all pending send/receive operations and initiates
        them asynchronously. Must be called before wait().

        Raises:
            RuntimeError: If called twice without wait() or if no operations are pending
        """
        global STEP, VERBOSE

        if self._active_requests is not None:
            raise RuntimeError('Commit called twice without wait()')

        if not self._pending_operations:
            raise RuntimeError('No pending operations to commit')

        try:
            self._active_requests = dist.batch_isend_irecv(
                self._pending_operations)

            if VERBOSE:
                logger.debug(
                    f'ContextCommunicate | commit | STEP: {STEP} | RANK: {self.rank} | '
                    f'Committed {len(self._pending_operations) // 2} send/recv pairs'
                )

        except Exception as e:
            raise RuntimeError(
                f'Failed to commit communication operations: {e}') from e

    def wait(self) -> None:
        """
        Wait for all committed communication operations to complete.

        This method blocks until all pending operations initiated by commit()
        are complete. It also performs CUDA synchronization and cleans up
        internal state.

        Raises:
            RuntimeError: If called before commit() or if operation completion fails
        """
        global STEP, VERBOSE

        if self._active_requests is None:
            raise RuntimeError('Wait called before commit()')

        try:
            # Wait for all operations to complete
            for i, request in enumerate(self._active_requests):
                request.wait()

                if VERBOSE:
                    operation_type = 'send' if i % 2 == 0 else 'receive'
                    peer_rank = self.send_rank if operation_type == 'send' else self.recv_rank
                    logger.debug(
                        f'ContextCommunicate | wait | STEP: {STEP} | RANK: {self.rank} | '
                        f'Completed {operation_type} with rank {peer_rank}')

            # Synchronize CUDA operations
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Clean up state
            self._active_requests = None
            self._pending_operations = []

            if VERBOSE:
                logger.debug(
                    f'ContextCommunicate | wait | STEP: {STEP} | RANK: {self.rank} | '
                    'All operations completed successfully')

        except Exception as e:
            raise RuntimeError(f'Failed to wait for operations: {e}') from e

    def __del__(self) -> None:
        """Cleanup any remaining operations on destruction."""
        if self._active_requests is not None:
            try:
                for request in self._active_requests:
                    if hasattr(request, 'wait'):
                        request.wait()
            except Exception:
                pass  # Best effort cleanup
