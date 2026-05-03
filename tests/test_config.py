"""Tests for scaletorch.trainer.config module."""

import unittest
from unittest.mock import patch

from scaletorch.trainer.config import (
    CheckpointArguments,
    DataArguments,
    LoggingArguments,
    LrSchedulerArguments,
    ModelArguments,
    OptimizerArguments,
    ParallelArguments,
    ScaleTorchArguments,
    TrainingArguments,
    validate_parallelism_sizes,
    validate_pipeline_engine,
    validate_training_parameters,
)


class TestValidationFunctions(unittest.TestCase):

    def test_validate_parallelism_sizes_valid(self):
        validate_parallelism_sizes(1, 1, 1, 1)
        validate_parallelism_sizes(4, 8, 2, 1)

    def test_validate_parallelism_sizes_zero_raises(self):
        for args in [(0, 1, 1, 1), (1, 0, 1, 1), (1, 1, 0, 1), (1, 1, 1, 0)]:
            with self.assertRaises(ValueError):
                validate_parallelism_sizes(*args)

    def test_validate_pipeline_engine_valid(self):
        validate_pipeline_engine('1f1b')
        validate_pipeline_engine('afab')

    def test_validate_pipeline_engine_invalid(self):
        with self.assertRaises(ValueError):
            validate_pipeline_engine('invalid')

    def test_validate_training_parameters_valid(self):
        validate_training_parameters(1, 32, 128)
        validate_training_parameters(4, None, None)

    def test_validate_training_parameters_zero_raises(self):
        with self.assertRaises(ValueError):
            validate_training_parameters(0, 32, 128)
        with self.assertRaises(ValueError):
            validate_training_parameters(1, 0, None)
        with self.assertRaises(ValueError):
            validate_training_parameters(1, None, 0)


class TestDataArguments(unittest.TestCase):

    def test_defaults(self):
        args = DataArguments()
        self.assertEqual(args.data_path, './data')
        self.assertEqual(args.dataset_name, 'wikitext2')
        self.assertEqual(args.split, 'train')
        self.assertEqual(args.num_proc, 1)
        self.assertEqual(args.num_workers, 0)
        self.assertIsNone(args.num_samples)
        self.assertTrue(args.pin_memory)


class TestModelArguments(unittest.TestCase):

    def test_defaults_with_valid_model(self):
        args = ModelArguments(model_name_or_path='./tiny_llama')
        self.assertEqual(args.use_flash_attention, True)
        self.assertEqual(args.dtype, 'bfloat16')

    def test_invalid_model_path_keeps_none(self):
        args = ModelArguments(model_name_or_path='/nonexistent/path')
        self.assertIsNone(args.num_hidden_layers)


class TestParallelArguments(unittest.TestCase):

    def test_defaults(self):
        args = ParallelArguments()
        self.assertEqual(args.tensor_parallel_size, 1)
        self.assertEqual(args.pipeline_parallel_size, 1)
        self.assertEqual(args.data_parallel_size, 1)
        self.assertEqual(args.context_parallel_size, 1)

    def test_invalid_size_raises(self):
        with self.assertRaises(ValueError):
            ParallelArguments(tensor_parallel_size=0)

    def test_invalid_backend_raises(self):
        with self.assertRaises(ValueError):
            ParallelArguments(backend='invalid')

    def test_invalid_engine_raises(self):
        with self.assertRaises(ValueError):
            ParallelArguments(pipeline_parallel_engine='invalid')


class TestLrSchedulerArguments(unittest.TestCase):

    def test_defaults(self):
        args = LrSchedulerArguments()
        self.assertEqual(args.lr_scheduler_type, 'linear')
        self.assertEqual(args.warmup_steps, 0)

    def test_invalid_type_raises(self):
        with self.assertRaises(ValueError):
            LrSchedulerArguments(lr_scheduler_type='invalid')

    def test_cosine_validates_eta_min(self):
        with self.assertRaises(ValueError):
            LrSchedulerArguments(lr_scheduler_type='cosine', eta_min=-1)

    def test_polynomial_validates_power(self):
        with self.assertRaises(ValueError):
            LrSchedulerArguments(lr_scheduler_type='polynomial', power=0)

    def test_step_validates_step_size(self):
        with self.assertRaises(ValueError):
            LrSchedulerArguments(lr_scheduler_type='step', step_size=0)

    def test_step_validates_gamma(self):
        with self.assertRaises(ValueError):
            LrSchedulerArguments(lr_scheduler_type='step', gamma=0)

    def test_onecycle_validates_pct_start(self):
        with self.assertRaises(ValueError):
            LrSchedulerArguments(lr_scheduler_type='onecycle', pct_start=0)


class TestOptimizerArguments(unittest.TestCase):

    def test_defaults(self):
        args = OptimizerArguments()
        self.assertEqual(args.optimizer_type, 'adamw')
        self.assertEqual(args.learning_rate, 1e-3)
        self.assertEqual(args.betas, (0.9, 0.999))

    def test_invalid_type_raises(self):
        with self.assertRaises(ValueError):
            OptimizerArguments(optimizer_type='rmsprop')


class TestTrainingArguments(unittest.TestCase):

    def test_defaults(self):
        args = TrainingArguments()
        self.assertEqual(args.batch_size, 64)
        self.assertEqual(args.gradient_accumulation_steps, 1)
        self.assertEqual(args.seed, 1)
        self.assertEqual(args.epochs, 5)

    def test_invalid_grad_acc_raises(self):
        with self.assertRaises(ValueError):
            TrainingArguments(gradient_accumulation_steps=0)


class TestCheckpointArguments(unittest.TestCase):

    def test_defaults(self):
        args = CheckpointArguments()
        self.assertEqual(args.work_dir, './work_dir')
        self.assertEqual(args.save_frequency, 300)

    def test_invalid_frequency_raises(self):
        with self.assertRaises(ValueError):
            CheckpointArguments(save_frequency=0)


class TestLoggingArguments(unittest.TestCase):

    def test_defaults(self):
        args = LoggingArguments()
        self.assertFalse(args.use_wandb)
        self.assertEqual(args.project_name, 'scaletorch')


class TestScaleTorchArguments(unittest.TestCase):

    def test_default_instantiation(self):
        args = ScaleTorchArguments(model_name_or_path='./tiny_llama')
        self.assertIsNotNone(args)
        self.assertEqual(args.global_batch_size, 64)
        self.assertIsNotNone(args.global_batch_size_token)

    def test_micro_batch_fallback(self):
        args = ScaleTorchArguments(
            model_name_or_path='./tiny_llama',
            batch_size=32,
            micro_batch_size=None
        )
        self.assertEqual(args.micro_batch_size, 32)

    def test_custom_config(self):
        args = ScaleTorchArguments(
            model_name_or_path='./tiny_llama',
            batch_size=16,
            micro_batch_size=4,
            gradient_accumulation_steps=2,
            data_parallel_size=2,
            sequence_length=512,
        )
        # global_batch = dp * micro_batch * grad_acc = 2 * 4 * 2 = 16
        self.assertEqual(args.global_batch_size, 16)
        self.assertEqual(args.global_batch_size_token, 16 * 512)

    def test_no_sequence_length_token_is_none(self):
        args = ScaleTorchArguments(
            model_name_or_path='./tiny_llama',
            sequence_length=None,
        )
        self.assertIsNone(args.global_batch_size_token)

    def test_all_fields_accessible(self):
        args = ScaleTorchArguments(model_name_or_path='./tiny_llama')
        fields = [
            'data_path', 'dataset_name', 'model_name_or_path',
            'tensor_parallel_size', 'pipeline_parallel_size',
            'data_parallel_size', 'context_parallel_size',
            'lr_scheduler_type', 'learning_rate',
            'batch_size', 'gradient_accumulation_steps',
            'work_dir', 'use_wandb',
            'global_batch_size', 'global_batch_size_token',
            'use_cpu', 'max_tokens', 'total_train_steps',
            'gradient_checkpointing', 'max_grad_norm',
            'num_workers', 'num_samples', 'use_fused_adam',
        ]
        for f in fields:
            self.assertTrue(hasattr(args, f), f'Missing field: {f}')


if __name__ == '__main__':
    unittest.main()
