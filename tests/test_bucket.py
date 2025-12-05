#!/usr/bin/env python3
"""
Test script for bucket.py module.
Tests Bucket and BucketManager functionality for gradient synchronization.
"""

import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from scaletorch.parallel.data_parallel.bucket import Bucket, BucketManager


class TestBucket(unittest.TestCase):
    """Test cases for Bucket class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock process group
        self.mock_process_group = MagicMock()
        self.mock_process_group_size = 4
        self.mock_process_group = MagicMock()

        # Mock dist.get_world_size
        self.dist_patcher = patch(
            'scaletorch.parallel.data_parallel.bucket.dist')
        self.mock_dist = self.dist_patcher.start()
        self.mock_dist.get_world_size.return_value = self.mock_process_group_size

        # Create test parameters
        self.param1 = nn.Parameter(torch.randn(10, 5))
        self.param2 = nn.Parameter(torch.randn(5, 3))
        self.params = [self.param1, self.param2]

        # Create gradient data tensor
        total_elements = 10 * 5 + 5 * 3  # Total number of elements
        self.grad_data = torch.zeros(total_elements)

    def tearDown(self):
        """Clean up test fixtures."""
        self.dist_patcher.stop()

    def test_init_success(self):
        """Test Bucket initialization."""
        bucket = Bucket(self.params, self.grad_data, self.mock_process_group)

        self.assertEqual(bucket.params, set(self.params))
        torch.testing.assert_close(bucket.grad_data, self.grad_data)
        self.assertEqual(bucket.process_group, self.mock_process_group)
        self.assertEqual(bucket.process_group_size,
                         self.mock_process_group_size)
        self.assertIsNone(bucket.handle)

    def test_init_empty_params(self):
        """Test Bucket initialization with empty parameters."""
        with self.assertRaises(ValueError) as context:
            Bucket([], self.grad_data, self.mock_process_group)

        self.assertIn('Parameter list cannot be empty', str(context.exception))

    def test_init_invalid_grad_data(self):
        """Test Bucket initialization with invalid grad_data."""
        with self.assertRaises(ValueError) as context:
            Bucket(self.params, 'not a tensor', self.mock_process_group)

        self.assertIn('grad_data must be a torch.Tensor',
                      str(context.exception))

    def test_sync_gradient_success(self):
        """Test successful gradient synchronization."""
        bucket = Bucket(self.params, self.grad_data, self.mock_process_group)

        # Add some data to grad_data
        self.grad_data.normal_()

        # Mock dist.all_reduce to simulate gradient averaging in-place
        # and return a mock handle for async_op
        def mock_all_reduce(tensor, group=None, async_op=False):
            if async_op:
                # Return a mock handle for async operation
                mock_handle = MagicMock()
                return mock_handle
            else:
                tensor.div_(self.mock_process_group_size)
                return None

        self.mock_dist.all_reduce.side_effect = mock_all_reduce

        bucket.sync_gradient()

        # Verify that all_reduce was called with async_op=True
        call_args = self.mock_dist.all_reduce.call_args
        self.assertTrue(call_args.kwargs.get('async_op', False))

        # Verify that handle was set
        self.assertIsNotNone(bucket.handle)

    def test_sync_gradient_already_in_progress(self):
        """Test sync_gradient when synchronization is already in progress."""
        bucket = Bucket(self.params, self.grad_data, self.mock_process_group)

        # Set a handle to simulate in-progress synchronization
        bucket.handle = MagicMock()

        with self.assertRaises(RuntimeError) as context:
            bucket.sync_gradient()

        self.assertIn(
            'Cannot start new synchronization while previous one is in progress',
            str(context.exception))

    def test_wait_success(self):
        """Test successful wait operation."""
        bucket = Bucket(self.params, self.grad_data, self.mock_process_group)

        # Set a mock handle
        mock_handle = MagicMock()
        bucket.handle = mock_handle

        bucket.wait()

        # Verify that wait was called on the handle
        mock_handle.wait.assert_called_once()

        # Verify that handle was cleared
        self.assertIsNone(bucket.handle)

    def test_wait_no_synchronization(self):
        """Test wait when no synchronization is in progress."""
        bucket = Bucket(self.params, self.grad_data, self.mock_process_group)

        with self.assertRaises(RuntimeError) as context:
            bucket.wait()

        self.assertIn('No synchronization operation in progress',
                      str(context.exception))

    def test_reset(self):
        """Test bucket reset functionality."""
        bucket = Bucket(self.params, self.grad_data, self.mock_process_group)

        # Add some data and mark parameters as ready
        self.grad_data.normal_()
        bucket.params_with_grad_ready.add(self.param1)
        bucket.handle = MagicMock()

        bucket.reset()

        # Verify that everything was reset
        self.assertEqual(len(bucket.params_with_grad_ready), 0)
        torch.testing.assert_close(bucket.grad_data,
                                   torch.zeros_like(self.grad_data))
        self.assertIsNone(bucket.handle)

    def test_is_synchronization_complete_false(self):
        """Test is_synchronization_complete when not all parameters are ready."""
        bucket = Bucket(self.params, self.grad_data, self.mock_process_group)

        # Mark only one parameter as ready
        bucket.mark_param_as_ready(self.param1)

        self.assertFalse(bucket.is_synchronization_complete())

    def test_is_synchronization_complete_true(self):
        """Test is_synchronization_complete when all parameters are ready."""
        bucket = Bucket(self.params, self.grad_data, self.mock_process_group)

        # Mark all parameters as ready
        bucket.mark_param_as_ready(self.param1)
        bucket.mark_param_as_ready(self.param2)

        self.assertTrue(bucket.is_synchronization_complete())

    def test_mark_param_as_ready_success(self):
        """Test successful parameter marking."""
        bucket = Bucket(self.params, self.grad_data, self.mock_process_group)

        # Mock sync_gradient to verify it's called
        bucket.sync_gradient = MagicMock()

        # Mark first parameter as ready (should not trigger sync)
        bucket.mark_param_as_ready(self.param1)
        self.assertIn(self.param1, bucket.params_with_grad_ready)
        bucket.sync_gradient.assert_not_called()

        # Mark second parameter as ready (should trigger sync)
        bucket.mark_param_as_ready(self.param2)
        self.assertIn(self.param2, bucket.params_with_grad_ready)
        bucket.sync_gradient.assert_called_once()

    def test_mark_param_as_ready_invalid_param(self):
        """Test mark_param_as_ready with parameter not in bucket."""
        bucket = Bucket(self.params, self.grad_data, self.mock_process_group)

        other_param = nn.Parameter(torch.randn(3, 3))  # Not in the bucket

        with self.assertRaises(ValueError) as context:
            bucket.mark_param_as_ready(other_param)

        self.assertIn('is not in this bucket', str(context.exception))

    def test_mark_param_as_ready_already_ready(self):
        """Test mark_param_as_ready when parameter is already marked as ready."""
        bucket = Bucket(self.params, self.grad_data, self.mock_process_group)

        bucket.mark_param_as_ready(self.param1)

        with self.assertRaises(ValueError) as context:
            bucket.mark_param_as_ready(self.param1)

        self.assertIn('is already marked as ready', str(context.exception))


class TestBucketManager(unittest.TestCase):
    """Test cases for BucketManager class."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock dist.get_world_size
        self.dist_patcher = patch(
            'scaletorch.parallel.data_parallel.bucket.dist')
        self.mock_dist = self.dist_patcher.start()
        self.mock_dist.get_world_size.return_value = 4

        # Create mock process group
        self.mock_process_group = MagicMock()

        # Create test parameters with different sizes
        self.param1 = nn.Parameter(torch.randn(100, 50))  # 5000 elements
        self.param2 = nn.Parameter(torch.randn(50, 30))  # 1500 elements
        self.param3 = nn.Parameter(torch.randn(30, 20))  # 600 elements
        self.params = [self.param1, self.param2, self.param3]

        # Set parameters to require gradients
        for param in self.params:
            param.requires_grad = True

    def tearDown(self):
        """Clean up test fixtures."""
        self.dist_patcher.stop()

    def test_init_success(self):
        """Test BucketManager initialization."""
        bucket_size = 2000  # Elements per bucket
        bucket_manager = BucketManager(self.params, self.mock_process_group,
                                       bucket_size)

        self.assertEqual(bucket_manager.params, self.params)
        self.assertEqual(bucket_manager.process_group, self.mock_process_group)
        self.assertEqual(bucket_manager.bucket_size, bucket_size)
        self.assertEqual(bucket_manager.grad_type, torch.float32)

        # Verify buckets were created
        self.assertGreater(len(bucket_manager.buckets), 0)
        self.assertEqual(len(bucket_manager.buckets),
                         len(bucket_manager.grad_data_list))
