# tests/test_base.py
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn


class BaseTestCase(unittest.TestCase):
    """Base test case with common functionality."""

    def setUp(self):
        """Set up common test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.mock_patches = []

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

        # Stop all mock patches
        for patcher in self.mock_patches:
            patcher.stop()

    def create_mock_process_group(self, world_size=4):
        """Create a mock process group."""
        mock_pg = MagicMock()
        return mock_pg

    def create_mock_dist(self, world_size=4, rank=0):
        """Create mock distributed environment."""
        dist_patcher = patch('torch.distributed')
        mock_dist = dist_patcher.start()
        self.mock_patches.append(dist_patcher)

        mock_dist.is_initialized.return_value = True
        mock_dist.get_world_size.return_value = world_size
        mock_dist.get_rank.return_value = rank
        mock_dist.all_reduce = MagicMock()

        return mock_dist

    def create_simple_model(self, input_size=10, hidden_size=5, output_size=1):
        """Create a simple neural network for testing."""
        return nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(),
                             nn.Linear(hidden_size, output_size))

    def assert_tensors_equal(self, tensor1, tensor2, msg=None):
        """Assert that two tensors are equal."""
        self.assertTrue(torch.allclose(tensor1, tensor2, rtol=1e-5, atol=1e-5),
                        msg or f'Tensors not equal: {tensor1} vs {tensor2}')
