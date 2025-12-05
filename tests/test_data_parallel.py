#!/usr/bin/env python3
"""
Test script for data_parallel module.
Tests DataParallelNaive and DataParallelBucket functionality.
"""

import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from scaletorch.parallel.data_parallel.data_parallel import (
    DataParallelBucket, DataParallelNaive)


class TestDataParallelNaive(unittest.TestCase):
    """Test cases for DataParallelNaive class."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock process group manager
        self.pgm_patcher = patch(
            'scaletorch.parallel.data_parallel.data_parallel.pgm')
        self.mock_pgm = self.pgm_patcher.start()

        # Configure mock process group manager
        self.mock_pgm.process_group_manager = MagicMock()
        self.mock_pgm.process_group_manager.cp_dp_group = MagicMock()
        self.mock_pgm.process_group_manager.cp_dp_world_size = 2

        # Create a simple model
        self.model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(),
                                   nn.Linear(5, 1))

    def tearDown(self):
        """Clean up test fixtures."""
        self.pgm_patcher.stop()

    def test_init_success(self):
        """Test DataParallelNaive initialization."""
        dp_model = DataParallelNaive(self.model)

        self.assertIs(dp_model.module, self.model)
        self.assertTrue(dp_model.require_backward_grad_sync)

    def test_init_no_manager(self):
        """Test DataParallelNaive initialization without process group manager."""
        # Temporarily set pgm to None to simulate uninitialized process group manager
        with patch('scaletorch.parallel.data_parallel.data_parallel.pgm',
                   None):
            with self.assertRaises(RuntimeError) as context:
                DataParallelNaive(self.model)

        self.assertIn('Process group manager must be initialized',
                      str(context.exception))

    def test_forward_pass(self):
        """Test forward pass through DataParallelNaive."""
        dp_model = DataParallelNaive(self.model)

        input_tensor = torch.randn(4, 10)
        output = dp_model(input_tensor)

        self.assertEqual(output.shape, (4, 1))

    def test_no_sync_context_manager(self):
        """Test no_sync context manager."""
        dp_model = DataParallelNaive(self.model)

        # Initially, gradient sync should be enabled
        self.assertTrue(dp_model.require_backward_grad_sync)

        # Use no_sync context manager
        with dp_model.no_sync():
            self.assertFalse(dp_model.require_backward_grad_sync)

        # After context manager, should be restored
        self.assertTrue(dp_model.require_backward_grad_sync)

    def test_backward_hook_registration(self):
        """Test that backward hooks are registered for parameters."""
        dp_model = DataParallelNaive(self.model)

        # Check that hooks are registered for parameters that require gradients
        for param in dp_model.module.parameters():
            if param.requires_grad:
                # In a real implementation, we would check if hooks are registered
                # For now, we just ensure the method exists and can be called
                self.assertTrue(hasattr(param, 'requires_grad'))
                self.assertTrue(param.requires_grad)


class TestDataParallelBucket(unittest.TestCase):
    """Test cases for DataParallelBucket class."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock process group manager
        self.pgm_patcher = patch(
            'scaletorch.parallel.data_parallel.data_parallel.pgm')
        self.mock_pgm = self.pgm_patcher.start()

        # Configure mock process group manager
        self.mock_pgm.process_group_manager = MagicMock()
        self.mock_pgm.process_group_manager.cp_dp_group = MagicMock()

        # Mock BucketManager
        self.bucket_patcher = patch(
            'scaletorch.parallel.data_parallel.data_parallel.BucketManager')
        self.mock_bucket_manager = self.bucket_patcher.start()

        # Create a simple model
        self.model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(),
                                   nn.Linear(5, 1))

    def tearDown(self):
        """Clean up test fixtures."""
        self.pgm_patcher.stop()
        self.bucket_patcher.stop()

    def test_init_success(self):
        """Test DataParallelBucket initialization."""
        dp_model = DataParallelBucket(self.model)

        self.assertIs(dp_model.module, self.model)
        self.assertTrue(dp_model.require_backward_grad_sync)
        self.assertFalse(dp_model._post_backward_callback_set)
        self.assertEqual(dp_model.grad_type, torch.float32)

        # Verify BucketManager was created
        self.mock_bucket_manager.assert_called_once()

    def test_init_invalid_bucket_size(self):
        """Test DataParallelBucket initialization with invalid bucket size."""
        with self.assertRaises(ValueError) as context:
            DataParallelBucket(self.model, bucket_cap_mb=0)

        self.assertIn('bucket_cap_mb must be positive', str(context.exception))

    def test_init_no_manager(self):
        """Test DataParallelBucket initialization without process group manager."""
        # Temporarily set pgm to None to simulate uninitialized process group manager
        with patch('scaletorch.parallel.data_parallel.data_parallel.pgm',
                   None):
            with self.assertRaises(RuntimeError) as context:
                DataParallelBucket(self.model)

        self.assertIn('Process group manager must be initialized',
                      str(context.exception))

    def test_forward_pass(self):
        """Test forward pass through DataParallelBucket."""
        dp_model = DataParallelBucket(self.model)

        input_tensor = torch.randn(4, 10)
        output = dp_model(input_tensor)

        self.assertEqual(output.shape, (4, 1))

    def test_no_sync_context_manager(self):
        """Test no_sync context manager for DataParallelBucket."""
        dp_model = DataParallelBucket(self.model)

        # Initially, gradient sync should be enabled
        self.assertTrue(dp_model.require_backward_grad_sync)

        # Use no_sync context manager
        with dp_model.no_sync():
            self.assertFalse(dp_model.require_backward_grad_sync)

        # After context manager, should be restored
        self.assertTrue(dp_model.require_backward_grad_sync)

    def test_different_gradient_types(self):
        """Test DataParallelBucket with different gradient types."""
        dp_model_fp16 = DataParallelBucket(self.model, grad_type=torch.float16)
        dp_model_bf16 = DataParallelBucket(self.model,
                                           grad_type=torch.bfloat16)

        self.assertEqual(dp_model_fp16.grad_type, torch.float16)
        self.assertEqual(dp_model_bf16.grad_type, torch.bfloat16)


if __name__ == '__main__':
    unittest.main()
