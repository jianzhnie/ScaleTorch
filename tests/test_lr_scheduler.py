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
                                         num_warmup_steps=1000)
    linear_scheduler = create_lr_scheduler(
        optimizer, linear_config, num_training_steps=num_training_steps)
    assert linear_scheduler is not None
    print('✓ Linear scheduler created successfully')

    # Test cosine scheduler
    print('\nTesting cosine scheduler...')
    cosine_config = LrSchedulerArguments(lr_scheduler_type='cosine',
                                         num_warmup_steps=1000)
    cosine_scheduler = create_lr_scheduler(
        optimizer, cosine_config, num_training_steps=num_training_steps)
    assert cosine_scheduler is not None
    print('✓ Cosine scheduler created successfully')

    # Test polynomial scheduler
    print('\nTesting polynomial scheduler...')
    poly_config = LrSchedulerArguments(lr_scheduler_type='polynomial',
                                       num_warmup_steps=1000,
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

    # Test reduce_on_plateau scheduler
    print('\nTesting reduce_on_plateau scheduler...')
    plateau_config = LrSchedulerArguments(
        lr_scheduler_type='reduce_on_plateau', mode='min', patience=10)
    plateau_scheduler = create_lr_scheduler(
        optimizer, plateau_config, num_training_steps=num_training_steps)
    assert plateau_scheduler is not None
    print('✓ ReduceLROnPlateau scheduler created successfully')

    # Test with warmup_steps (backward compatibility)
    print('\nTesting backward compatibility with warmup_steps...')
    legacy_config = LrSchedulerArguments(lr_scheduler_type='linear', )
    # Add warmup_steps attribute dynamically
    legacy_config.warmup_steps = 1000
    legacy_scheduler = create_lr_scheduler(
        optimizer, legacy_config, num_training_steps=num_training_steps)
    assert legacy_scheduler is not None
    print('✓ Backward compatibility maintained')

    print('\nAll tests passed! ✅')


if __name__ == '__main__':
    test_scheduler_creation()
