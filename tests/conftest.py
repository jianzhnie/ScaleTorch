"""Shared pytest fixtures and configuration for ScaleTorch tests."""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Core fixtures
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Helpers migrated from tests/test_base.py (BaseTestCase)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_process_group():
    """Factory fixture: returns a function that creates a mock process group."""

    def _create(world_size=4):
        mock_pg = MagicMock()
        mock_pg.size.return_value = world_size
        return mock_pg

    return _create


@pytest.fixture
def mock_pgm():
    """Factory fixture: returns a function that creates a mock ProcessGroupManager."""

    def _create(tp_size=1, pp_size=1, dp_size=1, cp_size=1, ep_size=1, rank=0):
        mock = MagicMock()
        mock.tp_world_size = tp_size
        mock.pp_world_size = pp_size
        mock.dp_world_size = dp_size
        mock.cp_world_size = cp_size
        mock.ep_world_size = ep_size
        mock.tp_rank = rank % tp_size
        mock.pp_rank = rank % pp_size
        mock.dp_rank = rank % dp_size
        mock.cp_rank = rank % cp_size
        mock.ep_rank = rank % ep_size
        mock.global_rank = rank
        mock.world_size = tp_size * pp_size * dp_size * cp_size
        mock.pp_is_first_stage = rank % pp_size == 0
        mock.pp_is_last_stage = rank % pp_size == pp_size - 1
        mock.cp_dp_world_size = cp_size * dp_size
        mock.__bool__ = lambda self: True
        return mock

    return _create
