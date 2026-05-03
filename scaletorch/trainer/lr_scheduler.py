"""
Learning rate scheduler configuration for ScaleTorch.

This module provides configuration classes and factory functions for creating
various types of learning rate schedulers used in training workflows.
"""

import math
from typing import Callable, Dict, Optional

from torch.optim.lr_scheduler import (LRScheduler, LambdaLR, OneCycleLR,
                                      PolynomialLR, StepLR)
from torch.optim.optimizer import Optimizer as OptimizerBase

from scaletorch.trainer.config import LrSchedulerArguments
from scaletorch.utils.logger_utils import get_logger

logger = get_logger(__name__)


def _get_warmup_factor(step: int, warmup_steps: int) -> float:
    """Return warmup factor (0.0 → 1.0) for given step."""
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, step / warmup_steps)


def create_lr_scheduler(
        optimizer: OptimizerBase,
        config: LrSchedulerArguments,
        num_training_steps: Optional[int] = 1000) -> Optional[LRScheduler]:
    """Factory for LR schedulers. Returns None if required params missing."""
    scheduler_type = config.lr_scheduler_type.lower()

    def create_linear_scheduler() -> Optional[LRScheduler]:
        warmup_steps = config.warmup_steps
        if num_training_steps is None:
            logger.warning(
                'num_training_steps is None, skipping linear scheduler creation'
            )
            return None

        def lr_lambda(step: int) -> float:
            warmup_factor = _get_warmup_factor(step, warmup_steps)
            if step < warmup_steps:
                return warmup_factor
            remaining = num_training_steps - step
            decay_steps = num_training_steps - warmup_steps
            return max(0.0, remaining / decay_steps) if decay_steps > 0 else 0.0

        return LambdaLR(optimizer, lr_lambda=lr_lambda)

    def create_cosine_scheduler() -> Optional[LRScheduler]:
        T_max = config.T_max if config.T_max is not None else num_training_steps
        eta_min = config.eta_min
        warmup_steps = config.warmup_steps

        if num_training_steps is None or T_max is None:
            logger.warning(
                'num_training_steps or T_max is None, skipping cosine scheduler creation'
            )
            return None

        def lr_lambda(step: int) -> float:
            warmup_factor = _get_warmup_factor(step, warmup_steps)
            if step < warmup_steps:
                return warmup_factor
            if T_max <= warmup_steps:
                return eta_min
            progress = (step - warmup_steps) / (T_max - warmup_steps)
            return eta_min + (1.0 - eta_min) * 0.5 * (1 +
                                                      math.cos(
                                                          math.pi * progress))

        return LambdaLR(optimizer, lr_lambda=lr_lambda)

    def create_polynomial_scheduler() -> Optional[LRScheduler]:
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
                warmup_factor = _get_warmup_factor(step, warmup_steps)
                if step < warmup_steps:
                    return warmup_factor
                adjusted = step - warmup_steps
                return (1.0 - adjusted /
                        (num_training_steps - warmup_steps))**power

            return LambdaLR(optimizer, lr_lambda=lr_lambda)

    def create_step_scheduler() -> Optional[LRScheduler]:
        warmup_steps = config.warmup_steps
        if warmup_steps <= 0:
            return StepLR(optimizer,
                          step_size=config.step_size,
                          gamma=config.gamma)

        def lr_lambda(step: int) -> float:
            warmup_factor = _get_warmup_factor(step, warmup_steps)
            if step < warmup_steps:
                return warmup_factor
            # Apply step decay relative to warmup-adjusted step
            decay_step = (step - warmup_steps) // config.step_size
            return config.gamma**decay_step

        return LambdaLR(optimizer, lr_lambda=lr_lambda)

    def create_onecycle_scheduler() -> Optional[LRScheduler]:
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
    scheduler_creation_map: Dict[str, Callable[[], Optional[LRScheduler]]] = {
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
