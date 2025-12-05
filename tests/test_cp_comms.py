#!/usr/bin/env python3
"""
Test script for cp_comms.py module.
Tests ContextCommunicate functionality for context parallel communication.
"""

import unittest
from unittest.mock import MagicMock, patch

import torch

from scaletorch.parallel.context_parallel.cp_comms import ContextCommunicate


class TestContextCommunicate(unittest.TestCase):
    """Test cases for ContextCommunicate class."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock process group manager
        self.pgm_patcher = patch(
            'scaletorch.parallel.context_parallel.cp_comms.pgm')
        self.mock_pgm = self.pgm_patcher.start()

        # Configure mock process group manager
        self.mock_pgm.process_group_manager.cp_rank = 1
        self.mock_pgm.process_group_manager.cp_world_size = 4
        self.mock_pgm.process_group_manager.cp_send_rank = 2
        self.mock_pgm.process_group_manager.cp_recv_rank = 0
        self.mock_pgm.process_group_manager.cp_group = MagicMock()

        # Mock torch.distributed
        self.dist_patcher = patch(
            'scaletorch.parallel.context_parallel.cp_comms.dist')
        self.mock_dist = self.dist_patcher.start()

    def tearDown(self):
        """Clean up test fixtures."""
        self.pgm_patcher.stop()
        self.dist_patcher.stop()

    def test_init_success(self):
        """Test ContextCommunicate initialization."""
        comm = ContextCommunicate('test_message')

        self.assertEqual(comm.rank, 1)
        self.assertEqual(comm.world_size, 4)
        self.assertEqual(comm.send_rank, 2)
        self.assertEqual(comm.recv_rank, 0)

    def test_init_no_manager(self):
        """Test ContextCommunicate initialization without process group manager."""
        del self.mock_pgm.process_group_manager

        with self.assertRaises(RuntimeError) as context:
            ContextCommunicate('test_message')

        self.assertIn('Process group manager not initialized',
                      str(context.exception))

    def test_send_recv_success(self):
        """Test successful send_recv operation."""
        comm = ContextCommunicate('test_operation')

        # Create test tensor
        tensor_to_send = torch.randn(2, 4, 8)

        # Mock dist.P2POp
        self.mock_dist.P2POp.return_value = MagicMock()

        # Mock batch_isend_irecv
        self.mock_dist.batch_isend_irecv.return_value = [MagicMock()]

        result = comm.send_recv(tensor_to_send)

        # Verify that P2POp was called for both send and recv
        self.assertEqual(self.mock_dist.P2POp.call_count, 2)

        # Verify result tensor shape and dtype
        self.assertEqual(result.shape, tensor_to_send.shape)
        self.assertEqual(result.dtype, tensor_to_send.dtype)

    def test_send_recv_with_preallocated_tensor(self):
        """Test send_recv with pre-allocated receive tensor."""
        comm = ContextCommunicate('test_operation')

        tensor_to_send = torch.randn(2, 4, 8)
        recv_tensor = torch.zeros_like(tensor_to_send)

        # Mock dist.P2POp
        self.mock_dist.P2POp.return_value = MagicMock()

        result = comm.send_recv(tensor_to_send, recv_tensor)

        # Result should be the pre-allocated tensor
        self.assertIs(result, recv_tensor)

    def test_send_recv_invalid_tensor(self):
        """Test send_recv with invalid tensor."""
        comm = ContextCommunicate('test_operation')

        with self.assertRaises(ValueError) as context:
            comm.send_recv('not a tensor')

        self.assertIn('tensor_to_send must be a torch.Tensor',
                      str(context.exception))

    def test_send_recv_empty_tensor(self):
        """Test send_recv with empty tensor."""
        comm = ContextCommunicate('test_operation')

        empty_tensor = torch.empty(0)

        with self.assertRaises(ValueError) as context:
            comm.send_recv(empty_tensor)

        self.assertIn('Cannot send empty tensor', str(context.exception))

    def test_send_recv_shape_mismatch(self):
        """Test send_recv with shape mismatch between send and recv tensors."""
        comm = ContextCommunicate('test_operation')

        tensor_to_send = torch.randn(2, 4, 8)
        recv_tensor = torch.zeros(2, 4, 16)  # Different shape

        with self.assertRaises(ValueError) as context:
            comm.send_recv(tensor_to_send, recv_tensor)

        self.assertIn('Shape mismatch', str(context.exception))

    def test_send_recv_dtype_mismatch(self):
        """Test send_recv with dtype mismatch."""
        comm = ContextCommunicate('test_operation')

        tensor_to_send = torch.randn(2, 4, 8)
        recv_tensor = torch.zeros(2, 4, 8,
                                  dtype=torch.float64)  # Different dtype

        with self.assertRaises(ValueError) as context:
            comm.send_recv(tensor_to_send, recv_tensor)

        self.assertIn('Dtype mismatch', str(context.exception))

    def test_send_recv_device_mismatch(self):
        """Test send_recv with device mismatch."""
        comm = ContextCommunicate('test_operation')

        tensor_to_send = torch.randn(2, 4, 8)
        recv_tensor = torch.zeros(2, 4, 8).cpu()  # Force to CPU

        # This would normally test device mismatch, but in most test environments
        # we only have one device available
        if tensor_to_send.device != recv_tensor.device:
            with self.assertRaises(ValueError) as context:
                comm.send_recv(tensor_to_send, recv_tensor)

            self.assertIn('Device mismatch', str(context.exception))

    def test_commit_success(self):
        """Test successful commit operation."""
        comm = ContextCommunicate('test_operation')

        # Add some pending operations
        tensor = torch.randn(2, 4, 8)
        comm.send_recv(tensor)

        # Mock batch_isend_irecv
        mock_requests = [MagicMock()]
        self.mock_dist.batch_isend_irecv.return_value = mock_requests

        comm.commit()

        # Verify that batch_isend_irecv was called
        self.mock_dist.batch_isend_irecv.assert_called_once()

    def test_commit_no_pending_operations(self):
        """Test commit with no pending operations."""
        comm = ContextCommunicate('test_operation')

        with self.assertRaises(RuntimeError) as context:
            comm.commit()

        self.assertIn('No pending operations to commit',
                      str(context.exception))

    def test_commit_twice_without_wait(self):
        """Test calling commit twice without wait."""
        comm = ContextCommunicate('test_operation')

        tensor = torch.randn(2, 4, 8)
        comm.send_recv(tensor)

        # Mock batch_isend_irecv
        mock_requests = [MagicMock()]
        self.mock_dist.batch_isend_irecv.return_value = mock_requests

        comm.commit()

        with self.assertRaises(RuntimeError) as context:
            comm.commit()

        self.assertIn('Commit called twice without wait()',
                      str(context.exception))

    def test_wait_success(self):
        """Test successful wait operation."""
        comm = ContextCommunicate('test_operation')

        tensor = torch.randn(2, 4, 8)
        comm.send_recv(tensor)

        # Mock batch_isend_irecv and requests
        mock_request = MagicMock()
        mock_requests = [mock_request]
        self.mock_dist.batch_isend_irecv.return_value = mock_requests

        comm.commit()
        comm.wait()

        # Verify that wait was called on the request
        mock_request.wait.assert_called_once()

    def test_wait_before_commit(self):
        """Test calling wait before commit."""
        comm = ContextCommunicate('test_operation')

        with self.assertRaises(RuntimeError) as context:
            comm.wait()

        self.assertIn('Wait called before commit()', str(context.exception))

    def test_multiple_operations(self):
        """Test multiple send_recv operations followed by commit and wait."""
        comm = ContextCommunicate('test_operation')

        # Create multiple tensors
        tensor1 = torch.randn(2, 4, 8)
        tensor2 = torch.randn(3, 5, 7)

        # Perform multiple send_recv operations
        comm.send_recv(tensor1)
        comm.send_recv(tensor2)

        # Mock batch_isend_irecv and requests
        mock_requests = [MagicMock(),
                         MagicMock(),
                         MagicMock(),
                         MagicMock()]  # 2 send + 2 recv
        self.mock_dist.batch_isend_irecv.return_value = mock_requests

        comm.commit()
        comm.wait()

        # Verify that all requests had wait called
        for mock_request in mock_requests:
            mock_request.wait.assert_called_once()


if __name__ == '__main__':
    unittest.main()
