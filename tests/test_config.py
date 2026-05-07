"""Tests for scaletorch.trainer.config module."""

import unittest

from scaletorch.trainer.config import (
    DataArguments,
    ModelArguments,
    ParallelArguments,
    LrSchedulerArguments,
    OptimizerArguments,
    TrainingArguments,
    CheckpointArguments,
    LoggingArguments,
    ScaleTorchArguments,
)


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


class TestLrSchedulerArguments(unittest.TestCase):

    def test_defaults(self):
        args = LrSchedulerArguments()
        self.assertEqual(args.lr_scheduler_type, 'linear')
        self.assertEqual(args.warmup_steps, 0)


class TestOptimizerArguments(unittest.TestCase):

    def test_defaults(self):
        args = OptimizerArguments()
        self.assertEqual(args.optimizer_type, 'adamw')
        self.assertEqual(args.learning_rate, 1e-3)
        self.assertEqual(args.betas, (0.9, 0.999))


class TestTrainingArguments(unittest.TestCase):

    def test_defaults(self):
        args = TrainingArguments()
        self.assertEqual(args.batch_size, 64)
        self.assertEqual(args.gradient_accumulation_steps, 1)
        self.assertEqual(args.seed, 1)
        self.assertEqual(args.epochs, 5)


class TestCheckpointArguments(unittest.TestCase):

    def test_defaults(self):
        args = CheckpointArguments()
        self.assertEqual(args.work_dir, './work_dir')
        self.assertEqual(args.save_frequency, 300)


class TestLoggingArguments(unittest.TestCase):

    def test_defaults(self):
        args = LoggingArguments()
        self.assertFalse(args.use_wandb)
        self.assertEqual(args.project_name, 'scaletorch')


class TestScaleTorchArguments(unittest.TestCase):

    def _make(self, **kwargs):
        kwargs.setdefault('model_name_or_path', './tiny_llama')
        return ScaleTorchArguments(**kwargs)

    # --- instantiation & derived values ---

    def test_default_instantiation(self):
        args = self._make()
        self.assertEqual(args.global_batch_size, 64)
        self.assertIsNotNone(args.global_batch_size_token)

    def test_micro_batch_fallback(self):
        args = self._make(batch_size=32, micro_batch_size=None)
        self.assertEqual(args.micro_batch_size, 32)

    def test_custom_config(self):
        args = self._make(
            batch_size=16,
            micro_batch_size=4,
            gradient_accumulation_steps=2,
            data_parallel_size=2,
            sequence_length=512,
        )
        self.assertEqual(args.global_batch_size, 16)
        self.assertEqual(args.global_batch_size_token, 16 * 512)

    def test_no_sequence_length_token_is_none(self):
        args = self._make(sequence_length=None)
        self.assertIsNone(args.global_batch_size_token)

    def test_all_fields_accessible(self):
        args = self._make()
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

    # --- parallelism validation ---

    def test_invalid_tensor_parallel_size(self):
        with self.assertRaises(ValueError):
            self._make(tensor_parallel_size=0)

    def test_invalid_backend(self):
        with self.assertRaises(ValueError):
            self._make(backend='invalid')

    def test_invalid_pipeline_engine(self):
        with self.assertRaises(ValueError):
            self._make(pipeline_parallel_engine='invalid')

    # --- lr scheduler validation ---

    def test_invalid_lr_scheduler_type(self):
        with self.assertRaises(ValueError):
            self._make(lr_scheduler_type='invalid')

    def test_cosine_validates_eta_min(self):
        with self.assertRaises(ValueError):
            self._make(lr_scheduler_type='cosine', eta_min=-1)

    def test_polynomial_validates_power(self):
        with self.assertRaises(ValueError):
            self._make(lr_scheduler_type='polynomial', power=0)

    def test_step_validates_step_size(self):
        with self.assertRaises(ValueError):
            self._make(lr_scheduler_type='step', step_size=0)

    def test_step_validates_gamma(self):
        with self.assertRaises(ValueError):
            self._make(lr_scheduler_type='step', gamma=0)

    def test_onecycle_validates_pct_start(self):
        with self.assertRaises(ValueError):
            self._make(lr_scheduler_type='onecycle', pct_start=0)

    # --- optimizer validation ---

    def test_invalid_optimizer_type(self):
        with self.assertRaises(ValueError):
            self._make(optimizer_type='rmsprop')

    # --- training parameter validation ---

    def test_invalid_grad_accum_steps(self):
        with self.assertRaises(ValueError):
            self._make(gradient_accumulation_steps=0)

    def test_invalid_micro_batch_size(self):
        with self.assertRaises(ValueError):
            self._make(micro_batch_size=0)

    def test_invalid_sequence_length(self):
        with self.assertRaises(ValueError):
            self._make(sequence_length=0)

    # --- checkpoint validation ---

    def test_invalid_save_frequency(self):
        with self.assertRaises(ValueError):
            self._make(save_frequency=-1)

    # --- cross-cutting validation ---

    def test_sequence_length_not_divisible_by_cp(self):
        with self.assertRaises(ValueError):
            self._make(sequence_length=100, context_parallel_size=3)

    # --- world_size validation ---

    def test_validate_world_size_ok(self):
        args = self._make(
            tensor_parallel_size=2, pipeline_parallel_size=2,
            data_parallel_size=2, context_parallel_size=1,
        )
        args.validate_world_size(8)

    def test_validate_world_size_mismatch(self):
        args = self._make(
            tensor_parallel_size=2, pipeline_parallel_size=2,
            data_parallel_size=2, context_parallel_size=1,
        )
        with self.assertRaises(ValueError):
            args.validate_world_size(4)


if __name__ == '__main__':
    unittest.main()
