#!/usr/bin/env python3
"""
Test script for learning rate scheduler configuration module.
"""

from torch.nn import Linear
from torch.optim import Adam

from scaletorch.trainer.lr_scheduler_config import (LrSchedulerArguments,
                                                    create_lr_scheduler)


def test_scheduler_creation():
    """Test creation of various learning rate schedulers."""
    # Create a simple model and optimizer
    model = Linear(10, 1)
    optimizer = Adam(model.parameters(), lr=1e-3)

    # Test configurations
    num_training_steps = 10000

    # Test linear scheduler
    print('Testing linear scheduler...')
    linear_config = LrSchedulerArguments(lr_scheduler_type='linear',
                                         warmup_steps=1000)
    linear_scheduler = create_lr_scheduler(
        optimizer, linear_config, num_training_steps=num_training_steps)
    assert linear_scheduler is not None
    print('✓ Linear scheduler created successfully')

    # Test cosine scheduler
    print('\nTesting cosine scheduler...')
    cosine_config = LrSchedulerArguments(lr_scheduler_type='cosine',
                                         warmup_steps=1000)
    cosine_scheduler = create_lr_scheduler(
        optimizer, cosine_config, num_training_steps=num_training_steps)
    assert cosine_scheduler is not None
    print('✓ Cosine scheduler created successfully')

    # Test polynomial scheduler
    print('\nTesting polynomial scheduler...')
    poly_config = LrSchedulerArguments(lr_scheduler_type='polynomial',
                                       warmup_steps=1000,
                                       power=2.0)
    poly_scheduler = create_lr_scheduler(optimizer,
                                         poly_config,
                                         num_training_steps=num_training_steps)
    assert poly_scheduler is not None
    print('✓ Polynomial scheduler created successfully')

    # Test step scheduler
    print('\nTesting step scheduler...')
    step_config = LrSchedulerArguments(lr_scheduler_type='step',
                                       step_size=1000,
                                       gamma=0.5)
    step_scheduler = create_lr_scheduler(optimizer,
                                         step_config,
                                         num_training_steps=num_training_steps)
    assert step_scheduler is not None
    print('✓ Step scheduler created successfully')

    # Test onecycle scheduler
    print('\nTesting onecycle scheduler...')
    onecycle_config = LrSchedulerArguments(lr_scheduler_type='onecycle',
                                           max_lr=1e-2,
                                           pct_start=0.3)
    onecycle_scheduler = create_lr_scheduler(
        optimizer, onecycle_config, num_training_steps=num_training_steps)
    assert onecycle_scheduler is not None
    print('✓ OneCycle scheduler created successfully')

    # Test with warmup_steps explicitly set
    print('\nTesting warmup_steps parameter...')
    warmup_config = LrSchedulerArguments(lr_scheduler_type='linear',
                                         warmup_steps=1000)
    warmup_scheduler = create_lr_scheduler(
        optimizer, warmup_config, num_training_steps=num_training_steps)
    assert warmup_scheduler is not None
    print('✓ warmup_steps parameter works correctly')

    print('\nAll tests passed! ✅')


if __name__ == '__main__':
    test_scheduler_creation()
