"""Shared pytest fixtures and configuration for ScaleTorch tests."""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn


@pytest.fixture(autouse=True)
def reset_pgm():
    """Reset the process group manager proxy before each test."""
    import scaletorch.parallel.process_group as pgm_module

    old_instance = pgm_module.process_group_manager._instance
    yield
    pgm_module.process_group_manager._instance = old_instance


@pytest.fixture
def mock_dist():
    """Mock distributed functions used by process_group module."""
    with (
        patch("scaletorch.dist.is_distributed") as mock_is_distributed,
        patch("scaletorch.dist.get_rank") as mock_get_rank,
        patch("scaletorch.dist.get_world_size") as mock_get_world_size,
        patch("scaletorch.dist.new_group") as mock_new_group,
    ):
        mock_is_distributed.return_value = True
        mock_get_rank.return_value = 0
        mock_get_world_size.return_value = 8
        mock_new_group.return_value = MagicMock()
        yield {
            "is_distributed": mock_is_distributed,
            "get_rank": mock_get_rank,
            "get_world_size": mock_get_world_size,
            "new_group": mock_new_group,
        }


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
    return torch.device("cpu")
