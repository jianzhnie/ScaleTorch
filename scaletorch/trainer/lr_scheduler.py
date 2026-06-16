"""
Learning rate scheduler configuration for ScaleTorch.

This module provides configuration classes and factory functions for creating
various types of learning rate schedulers used in training workflows.
"""

from typing import Callable, Dict, Optional

import numpy as np
from torch.optim.lr_scheduler import (LambdaLR, OneCycleLR, PolynomialLR,
                                      StepLR, _LRScheduler)
from torch.optim.optimizer import Optimizer as OptimizerBase

from scaletorch.trainer.config import LrSchedulerArguments
from scaletorch.utils.logger_utils import get_logger

logger = get_logger(__name__)


def _get_warmup_factor(step: int, warmup_steps: int) -> float:
    """
    Calculate the warmup factor based on the current step.

    Args:
        step: Current training step
        warmup_steps: Number of warmup steps

    Returns:
        Warmup factor (0.0 to 1.0)
    """
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, step / warmup_steps)


def create_lr_scheduler(
        optimizer: OptimizerBase,
        config: LrSchedulerArguments,
        num_training_steps: Optional[int] = 1000) -> Optional[_LRScheduler]:
    """
    Create and configure the learning rate scheduler based on the provided arguments.

    Args:
        optimizer: PyTorch optimizer to wrap with the scheduler
        config: Configuration object containing scheduler settings
        num_training_steps: Total number of training steps (required for schedulers
            that need to know the total training duration)

    Returns:
        The configured learning rate scheduler instance, or None if creation failed
        (e.g., missing required parameters)

    Raises:
        ValueError: If validation fails for any scheduler parameter (raised from LrSchedulerArguments)

    Example:
        >>> from torch.optim import Adam
        >>> from scaletorch.trainer.lr_scheduler_config import (
        ...     LrSchedulerArguments, create_lr_scheduler
        ... )
        >>> optimizer = Adam(model.parameters(), lr=1e-3)
        >>> config = LrSchedulerArguments(
        ...     lr_scheduler_type='linear',
        ...     warmup_steps=1000,
        ... )
        >>> scheduler = create_lr_scheduler(
        ...     optimizer, config, num_training_steps=10000
        ... )
    """
    scheduler_type = config.lr_scheduler_type.lower()

    # Define scheduler creation functions
    def create_linear_scheduler() -> Optional[_LRScheduler]:
        """Create a linear warmup + linear decay scheduler."""
        warmup_steps = config.warmup_steps
        if num_training_steps is None:
            logger.warning(
                'num_training_steps is None, skipping linear scheduler creation'
            )
            return None

        def lr_lambda(step: int) -> float:
            """Lambda function for linear scheduler."""
            # Apply warmup
            warmup_factor = _get_warmup_factor(step, warmup_steps)
            if step < warmup_steps:
                return warmup_factor

            # Linear decay phase
            remaining_steps = num_training_steps - step
            total_decay_steps = num_training_steps - warmup_steps
            return max(0.0, remaining_steps /
                       total_decay_steps) if total_decay_steps > 0 else 0.0

        return LambdaLR(optimizer, lr_lambda=lr_lambda)

    def create_cosine_scheduler() -> Optional[_LRScheduler]:
        """Create a cosine annealing scheduler with optional warmup."""
        T_max = config.T_max if config.T_max is not None else num_training_steps
        eta_min = config.eta_min
        warmup_steps = config.warmup_steps

        if num_training_steps is None or T_max is None:
            logger.warning(
                'num_training_steps or T_max is None, skipping cosine scheduler creation'
            )
            return None

        def lr_lambda(step: int) -> float:
            """Lambda function for cosine scheduler."""
            # Apply warmup
            warmup_factor = _get_warmup_factor(step, warmup_steps)
            if step < warmup_steps:
                return warmup_factor

            # Cosine decay phase
            if T_max <= warmup_steps:
                return eta_min

            progress = (step - warmup_steps) / (T_max - warmup_steps)
            return eta_min + (1.0 - eta_min) * 0.5 * (1 +
                                                      np.cos(np.pi * progress))

        return LambdaLR(optimizer, lr_lambda=lr_lambda)

    def create_polynomial_scheduler() -> Optional[_LRScheduler]:
        """Create a polynomial decay scheduler with optional warmup."""
        power = config.power
        warmup_steps = config.warmup_steps
        if num_training_steps is None:
            logger.warning(
                'num_training_steps is None, skipping polynomial scheduler creation'
            )
            return None

        # If warmup is needed, wrap PolynomialLR with LambdaLR
        if warmup_steps <= 0:
            return PolynomialLR(optimizer,
                                total_iters=num_training_steps,
                                power=power)
        else:

            def lr_lambda(step: int) -> float:
                """Lambda function for polynomial scheduler with warmup."""
                warmup_factor = _get_warmup_factor(step, warmup_steps)
                if step < warmup_steps:
                    return warmup_factor

                # Adjust step for polynomial decay
                adjusted_step = step - warmup_steps
                # Get the polynomial decay factor
                base_factor = (1.0 - adjusted_step /
                               (num_training_steps - warmup_steps))**power
                return base_factor

            return LambdaLR(optimizer, lr_lambda=lr_lambda)

    def create_step_scheduler() -> StepLR:
        """Create a step decay scheduler."""
        return StepLR(optimizer,
                      step_size=config.step_size,
                      gamma=config.gamma)

    def create_onecycle_scheduler() -> Optional[_LRScheduler]:
        """Create a OneCycleLR scheduler."""
        if num_training_steps is None:
            logger.warning(
                'num_training_steps is None, skipping OneCycleLR scheduler creation'
            )
            return None

        # Get max_lr from config or use optimizer's default learning rate
        max_lr = config.max_lr if config.max_lr is not None else optimizer.defaults[
            'lr']
        return OneCycleLR(optimizer,
                          max_lr=max_lr,
                          total_steps=num_training_steps,
                          pct_start=config.pct_start)

    # Map scheduler types to their creation functions
    scheduler_creation_map: Dict[str, Callable[[], Optional[_LRScheduler]]] = {
        'linear': create_linear_scheduler,
        'cosine': create_cosine_scheduler,
        'polynomial': create_polynomial_scheduler,
        'step': create_step_scheduler,
        'onecycle': create_onecycle_scheduler,
    }

    # Get the appropriate creation function and call it
    if scheduler_type in scheduler_creation_map:
        return scheduler_creation_map[scheduler_type]()
    else:
        logger.warning(
            f'Unknown scheduler type {scheduler_type}, using no scheduler')
        return None
