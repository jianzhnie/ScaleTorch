"""Base test utilities for ScaleTorch tests."""

import tempfile
import unittest
from unittest.mock import MagicMock

import torch
import torch.nn as nn


class BaseTestCase(unittest.TestCase):
    """Base test case with common distributed training test utilities."""

    def setUp(self):
        """Set up common test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.mock_patches = []

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

        for patcher in self.mock_patches:
            patcher.stop()

    def create_mock_process_group(self, world_size=4):
        """Create a mock process group."""
        mock_pg = MagicMock()
        mock_pg.size.return_value = world_size
        return mock_pg

    def create_mock_pgm(self, tp_size=1, pp_size=1, dp_size=1, cp_size=1, rank=0):
        """Create a mock process group manager with common attributes."""
        mock = MagicMock()
        mock.tp_world_size = tp_size
        mock.pp_world_size = pp_size
        mock.dp_world_size = dp_size
        mock.cp_world_size = cp_size
        mock.tp_rank = rank % tp_size
        mock.pp_rank = rank % pp_size
        mock.dp_rank = rank % dp_size
        mock.cp_rank = rank % cp_size
        mock.global_rank = rank
        mock.world_size = tp_size * pp_size * dp_size * cp_size
        mock.pp_is_first_stage = rank % pp_size == 0
        mock.pp_is_last_stage = rank % pp_size == pp_size - 1
        mock.cp_dp_world_size = cp_size * dp_size
        return mock

    def create_simple_model(self, input_size=10, hidden_size=5, output_size=1):
        """Create a simple neural network for testing."""
        return nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def assert_tensors_equal(self, tensor1, tensor2, msg=None, atol=1e-5):
        """Assert that two tensors are approximately equal."""
        self.assertTrue(
            torch.allclose(tensor1, tensor2, rtol=1e-5, atol=atol),
            msg or f"Tensors not equal:\n{tensor1}\nvs\n{tensor2}",
        )

    def assert_tensor_shape(self, tensor, expected_shape, msg=None):
        """Assert tensor has expected shape."""
        self.assertEqual(
            tensor.shape,
            torch.Size(expected_shape),
            msg or f"Expected shape {expected_shape}, got {tensor.shape}",
        )
