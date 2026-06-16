"""Tests for scaletorch.trainer.lr_scheduler_config module."""
import pytest
import torch
from torch.nn import Linear
from torch.optim import Adam

from scaletorch.trainer.lr_scheduler import create_lr_scheduler


@pytest.fixture
def optimizer():
    """Create a simple optimizer for testing."""
    model = Linear(10, 1)
    return Adam(model.parameters(), lr=1e-3)


class MockConfig:
    """Mock config for scheduler tests."""

    def __init__(self, scheduler_type='linear', **kwargs):
        self.lr_scheduler_type = scheduler_type
        self.warmup_steps = kwargs.get('warmup_steps', 0)
        self.T_max = kwargs.get('T_max', None)
        self.eta_min = kwargs.get('eta_min', 0.0)
        self.power = kwargs.get('power', 1.0)
        self.step_size = kwargs.get('step_size', 1)
        self.gamma = kwargs.get('gamma', 0.1)
        self.max_lr = kwargs.get('max_lr', None)
        self.pct_start = kwargs.get('pct_start', 0.3)


class TestLinearScheduler:
    def test_creation(self, optimizer):
        config = MockConfig('linear', warmup_steps=100)
        scheduler = create_lr_scheduler(optimizer, config, num_training_steps=1000)
        assert scheduler is not None

    def test_warmup_increases_lr(self, optimizer):
        config = MockConfig('linear', warmup_steps=10)
        scheduler = create_lr_scheduler(optimizer, config, num_training_steps=100)

        initial_lr = optimizer.param_groups[0]['lr']
        for _ in range(5):
            optimizer.step()
            scheduler.step()
        warmup_lr = optimizer.param_groups[0]['lr']
        assert warmup_lr >= initial_lr or config.warmup_steps == 0


class TestCosineScheduler:
    def test_creation(self, optimizer):
        config = MockConfig('cosine', warmup_steps=100)
        scheduler = create_lr_scheduler(optimizer, config, num_training_steps=1000)
        assert scheduler is not None

    def test_lr_decreases_after_warmup(self, optimizer):
        config = MockConfig('cosine', warmup_steps=5)
        scheduler = create_lr_scheduler(optimizer, config, num_training_steps=100)

        for _ in range(10):
            optimizer.step()
            scheduler.step()
        mid_lr = optimizer.param_groups[0]['lr']

        for _ in range(80):
            optimizer.step()
            scheduler.step()
        late_lr = optimizer.param_groups[0]['lr']

        assert late_lr <= mid_lr


class TestPolynomialScheduler:
    def test_creation(self, optimizer):
        config = MockConfig('polynomial', warmup_steps=100, power=2.0)
        scheduler = create_lr_scheduler(optimizer, config, num_training_steps=1000)
        assert scheduler is not None


class TestStepScheduler:
    def test_creation(self, optimizer):
        config = MockConfig('step', step_size=100, gamma=0.5)
        scheduler = create_lr_scheduler(optimizer, config, num_training_steps=1000)
        assert scheduler is not None

    def test_lr_drops_at_step(self, optimizer):
        config = MockConfig('step', step_size=5, gamma=0.5)
        scheduler = create_lr_scheduler(optimizer, config, num_training_steps=100)

        initial_lr = optimizer.param_groups[0]['lr']
        for _ in range(5):
            optimizer.step()
            scheduler.step()
        after_step_lr = optimizer.param_groups[0]['lr']
        assert abs(after_step_lr - initial_lr * 0.5) < 1e-7


class TestOneCycleScheduler:
    def test_creation(self, optimizer):
        config = MockConfig('onecycle', max_lr=1e-2, pct_start=0.3)
        scheduler = create_lr_scheduler(optimizer, config, num_training_steps=1000)
        assert scheduler is not None
