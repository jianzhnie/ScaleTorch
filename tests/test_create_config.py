#!/usr/bin/env python3
"""
Test script for create_config.py module.
This script tests the functionality of the configuration creation utility.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from scaletorch.utils.create_config import (_calculate_batch_sizes,
                                            _load_model_configuration,
                                            _validate_parallelism_sizes,
                                            _validate_training_parameters,
                                            create_single_config)


class TestCreateConfig(unittest.TestCase):
    """Test cases for the create_config module."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temporary directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_validate_parallelism_sizes_valid(self):
        """Test _validate_parallelism_sizes with valid inputs."""
        # Should not raise any exceptions
        _validate_parallelism_sizes(1, 1, 1, 1)
        _validate_parallelism_sizes(2, 2, 2, 2)

    def test_validate_parallelism_sizes_invalid(self):
        """Test _validate_parallelism_sizes with invalid inputs."""
        with self.assertRaises(ValueError):
            _validate_parallelism_sizes(0, 1, 1, 1)

        with self.assertRaises(ValueError):
            _validate_parallelism_sizes(1, -1, 1, 1)

    def test_validate_training_parameters_valid(self):
        """Test _validate_training_parameters with valid inputs."""
        # Should not raise any exceptions
        _validate_training_parameters(1, 1, 1024)
        _validate_training_parameters(4, 8, 2048)

    def test_validate_training_parameters_invalid(self):
        """Test _validate_training_parameters with invalid inputs."""
        with self.assertRaises(ValueError):
            _validate_training_parameters(0, 1, 1024)

        with self.assertRaises(ValueError):
            _validate_training_parameters(1, 0, 1024)

        with self.assertRaises(ValueError):
            _validate_training_parameters(1, 1, 0)

    def test_calculate_batch_sizes(self):
        """Test _calculate_batch_sizes function."""
        global_batch, global_batch_tokens = _calculate_batch_sizes(
            2, 4, 3, 512)
        self.assertEqual(global_batch, 24)  # 2 * 4 * 3
        self.assertEqual(global_batch_tokens, 12288)  # 24 * 512

    def test_load_model_configuration_with_valid_model(self):
        """Test _load_model_configuration with a valid model."""
        # Using a small model for faster testing
        config = _load_model_configuration(
            'hf-internal-testing/tiny-random-gpt2', None, None, None)
        self.assertIsNotNone(config['num_hidden_layers'])
        self.assertIsNotNone(config['num_attention_heads'])

    def test_create_single_config_basic(self):
        """Test create_single_config with basic parameters."""
        template_dir = Path(__file__).parent.parent / 'template'
        experiment_dir = create_single_config(
            template_dir=str(template_dir),
            data_parallel_size=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            context_parallel_size=1,
            pipeline_parallel_engine='1f1b',
            model_name_or_path='hf-internal-testing/tiny-random-gpt2',
            experiment_name='test_basic_experiment',
            output_dir=self.temp_dir)

        # Check that the directory was created
        self.assertTrue(experiment_dir.exists())
        self.assertEqual(experiment_dir.name, 'test_basic_experiment')

        # Check that config.json was created
        config_file = experiment_dir / 'config.json'
        self.assertTrue(config_file.exists())

        # Check config content
        with open(config_file, 'r') as f:
            config = json.load(f)

        self.assertIn('distributed', config)
        self.assertIn('model', config)
        self.assertIn('training', config)

        # Check that parameters were set correctly
        self.assertEqual(config['distributed']['data_parallel_size'], 1)
        self.assertEqual(config['distributed']['tensor_parallel_size'], 1)
        self.assertEqual(config['distributed']['pipeline_parallel_size'], 1)
        self.assertEqual(config['distributed']['context_parallel_size'], 1)
        self.assertEqual(config['distributed']['pipeline_parallel_engine'],
                         '1f1b')

    def test_create_single_config_with_training_params(self):
        """Test create_single_config with training parameters."""
        template_dir = Path(__file__).parent.parent / 'template'
        experiment_dir = create_single_config(
            template_dir=str(template_dir),
            data_parallel_size=2,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            context_parallel_size=1,
            pipeline_parallel_engine='1f1b',
            model_name_or_path='hf-internal-testing/tiny-random-gpt2',
            grad_accumulation_steps=3,
            micro_batch_size=2,
            sequence_length=512,
            experiment_name='test_training_params',
            output_dir=self.temp_dir)

        config_file = experiment_dir / 'config.json'
        with open(config_file, 'r') as f:
            config = json.load(f)

        training_config = config['training']
        self.assertEqual(training_config['gradient_accumulation_steps'], 3)
        self.assertEqual(training_config['micro_batch_size'], 2)
        self.assertEqual(training_config['sequence_length'], 512)

    def test_create_single_config_overwrites_existing(self):
        """Test that create_single_config overwrites existing experiment directory."""
        template_dir = Path(__file__).parent.parent / 'template'
        # Create an initial experiment
        experiment_dir = create_single_config(
            template_dir=str(template_dir),
            data_parallel_size=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            context_parallel_size=1,
            pipeline_parallel_engine='1f1b',
            model_name_or_path='hf-internal-testing/tiny-random-gpt2',
            experiment_name='test_overwrite',
            output_dir=self.temp_dir)

        # Create a file in the experiment directory
        dummy_file = experiment_dir / 'dummy.txt'
        with open(dummy_file, 'w') as f:
            f.write('test')

        self.assertTrue(dummy_file.exists())

        # Create the experiment again - should overwrite
        experiment_dir = create_single_config(
            template_dir=str(template_dir),
            data_parallel_size=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            context_parallel_size=1,
            pipeline_parallel_engine='1f1b',
            model_name_or_path='hf-internal-testing/tiny-random-gpt2',
            experiment_name='test_overwrite',
            output_dir=self.temp_dir)

        # Dummy file should no longer exist
        self.assertFalse(dummy_file.exists())

        # Config file should exist
        config_file = experiment_dir / 'config.json'
        self.assertTrue(config_file.exists())


if __name__ == '__main__':
    unittest.main()
