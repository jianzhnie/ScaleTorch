"""Shared pytest fixtures and configuration for ScaleTorch tests."""
import unittest
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn


@pytest.fixture(autouse=True)
def reset_pgm():
    """Reset the process group manager proxy before each test."""
    import scaletorch.parallel.pg_manager as pgm_module
    old_instance = pgm_module.process_group_manager._instance
    yield
    pgm_module.process_group_manager._instance = old_instance


@pytest.fixture
def mock_dist():
    """Mock torch.distributed for unit tests."""
    with patch('scaletorch.parallel.pg_manager.dist') as mock:
        mock.is_initialized.return_value = True
        mock.get_rank.return_value = 0
        mock.get_world_size.return_value = 8
        mock.new_group.return_value = MagicMock()
        yield mock


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1),
    )


@pytest.fixture
def device():
    """Get the test device (CPU for unit tests)."""
    return torch.device('cpu')
