"""Tests for scaletorch.utils.misc module."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import torch.nn as nn

from scaletorch.utils.misc import (
    TRILLION, BILLION, MILLION, THOUSAND,
    assert_no_meta_tensors,
    average_loss_across_dp_cp_ranks,
    get_mfu,
    get_num_params,
    set_all_seed,
    to_readable_format,
)


class TestSetAllSeed(unittest.TestCase):

    def test_valid_seed(self):
        set_all_seed(42)
        py_state = sum(torch.rand(10).tolist())
        set_all_seed(42)
        py_state2 = sum(torch.rand(10).tolist())
        self.assertAlmostEqual(py_state, py_state2, places=5)

    def test_different_seeds_differ(self):
        set_all_seed(1)
        v1 = torch.rand(5).tolist()
        set_all_seed(999)
        v2 = torch.rand(5).tolist()
        self.assertNotAlmostEqual(sum(v1), sum(v2), places=3)

    def test_seed_zero(self):
        set_all_seed(0)
        self.assertIsInstance(torch.rand(1).item(), float)

    def test_negative_seed_raises(self):
        with self.assertRaises(ValueError):
            set_all_seed(-1)

    def test_non_int_seed_raises(self):
        with self.assertRaises(TypeError):
            set_all_seed(42.0)

    def test_sets_numpy_seed(self):
        set_all_seed(42)
        v1 = np.random.rand(5).tolist()
        set_all_seed(42)
        v2 = np.random.rand(5).tolist()
        self.assertEqual(v1, v2)


class TestToReadableFormat(unittest.TestCase):

    def test_zero(self):
        self.assertEqual(to_readable_format(0), '0.00')

    def test_less_than_thousand(self):
        self.assertEqual(to_readable_format(500), '500.00')

    def test_thousands(self):
        self.assertEqual(to_readable_format(1500), '1.50K')

    def test_millions(self):
        self.assertEqual(to_readable_format(2500000), '2.50M')

    def test_billions(self):
        self.assertEqual(to_readable_format(3000000000), '3.00B')

    def test_trillions(self):
        self.assertEqual(to_readable_format(4000000000000), '4.00T')

    def test_negative(self):
        self.assertEqual(to_readable_format(-5000), '-5.00K')

    def test_float_input(self):
        self.assertEqual(to_readable_format(1234.5), '1.23K')

    def test_precision(self):
        self.assertEqual(to_readable_format(1500, precision=0), '2K')
        self.assertEqual(to_readable_format(1500, precision=3), '1.500K')

    def test_negative_precision_raises(self):
        with self.assertRaises(ValueError):
            to_readable_format(100, precision=-1)

    def test_non_numeric_raises(self):
        with self.assertRaises(TypeError):
            to_readable_format('100')


class TestGetMFU(unittest.TestCase):

    def _make_config(self, num_layers=2, hidden_size=128, max_pos=64):
        cfg = MagicMock()
        cfg.num_hidden_layers = num_layers
        cfg.hidden_size = hidden_size
        cfg.max_position_embeddings = max_pos
        return cfg

    def test_basic_mfu(self):
        cfg = self._make_config()
        mfu = get_mfu(1000.0, 1000000, cfg)
        self.assertIsInstance(mfu, float)
        self.assertGreater(mfu, 0)

    def test_zero_tokens_returns_zero(self):
        cfg = self._make_config()
        mfu = get_mfu(0, 1000000, cfg)
        self.assertEqual(mfu, 0.0)

    def test_negative_tokens_raises(self):
        with self.assertRaises(ValueError):
            get_mfu(-1, 1000, self._make_config())

    def test_negative_params_raises(self):
        with self.assertRaises(ValueError):
            get_mfu(100, -1, self._make_config())

    def test_invalid_flops_raises(self):
        with self.assertRaises(ValueError):
            get_mfu(100, 1000, self._make_config(), theoretical_flops=0)

    def test_missing_config_attr_raises(self):
        cfg = MagicMock(spec=[])
        with self.assertRaises(AttributeError):
            get_mfu(100, 1000, cfg)


class TestGetNumParams(unittest.TestCase):

    @patch('scaletorch.utils.misc.pgm', None)
    def test_simple_model_no_dist(self):
        model = nn.Linear(10, 5)
        count = get_num_params(model)
        self.assertEqual(count, 10 * 5 + 5)  # weight + bias

    @patch('scaletorch.utils.misc.pgm', None)
    def test_empty_model(self):
        model = nn.Module()
        count = get_num_params(model)
        self.assertEqual(count, 0)


class TestAssertNoMetaTensors(unittest.TestCase):

    def test_normal_model_passes(self):
        model = nn.Linear(10, 5)
        assert_no_meta_tensors(model)

    def test_meta_tensor_raises(self):
        # Create a parameter directly on meta device
        model = nn.Module()
        meta_param = nn.Parameter(
            torch.empty(10, 5, device='meta'))
        model.meta_weight = meta_param
        with self.assertRaises(RuntimeError):
            assert_no_meta_tensors(model)


class TestAverageLoss(unittest.TestCase):

    @patch('scaletorch.utils.misc.pgm', None)
    def test_single_process_returns_loss(self):
        result = average_loss_across_dp_cp_ranks(3.14, 'cpu')
        self.assertAlmostEqual(result, 3.14)

    @patch('scaletorch.utils.misc.pgm', None)
    def test_none_loss_returns_zero(self):
        result = average_loss_across_dp_cp_ranks(None, 'cpu')
        self.assertEqual(result, 0.0)

    def test_non_numeric_loss_raises(self):
        with self.assertRaises(TypeError):
            average_loss_across_dp_cp_ranks('not_a_loss', 'cpu')


if __name__ == '__main__':
    unittest.main()
