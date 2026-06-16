"""Tests for scaletorch.utils.utils module."""
import unittest
from unittest.mock import MagicMock, patch

import torch

from scaletorch.utils.utils import (
    set_all_seed,
    to_readable_format,
)


class TestToReadableFormat(unittest.TestCase):
    """Test number formatting utility."""

    def test_zero(self):
        self.assertEqual(to_readable_format(0), '0.00')

    def test_thousands(self):
        self.assertEqual(to_readable_format(1500), '1.50K')

    def test_millions(self):
        self.assertEqual(to_readable_format(2_500_000), '2.50M')

    def test_billions(self):
        self.assertEqual(to_readable_format(7_000_000_000), '7.00B')

    def test_trillions(self):
        self.assertEqual(to_readable_format(1_200_000_000_000), '1.20T')

    def test_negative(self):
        self.assertEqual(to_readable_format(-1500), '-1.50K')

    def test_custom_precision(self):
        self.assertEqual(to_readable_format(1234, precision=1), '1.2K')

    def test_small_number(self):
        self.assertEqual(to_readable_format(42), '42.00')

    def test_type_error(self):
        with self.assertRaises(TypeError):
            to_readable_format('not a number')

    def test_negative_precision(self):
        with self.assertRaises(ValueError):
            to_readable_format(100, precision=-1)


class TestSetAllSeed(unittest.TestCase):
    """Test seed setting."""

    @patch('scaletorch.utils.utils.is_accelerator_available', return_value=False)
    def test_cpu_seed(self, mock_accel):
        set_all_seed(42)
        t1 = torch.randn(5)
        set_all_seed(42)
        t2 = torch.randn(5)
        self.assertTrue(torch.equal(t1, t2))

    def test_invalid_seed_type(self):
        with self.assertRaises(TypeError):
            set_all_seed('not_int')

    def test_negative_seed(self):
        with self.assertRaises(ValueError):
            set_all_seed(-1)

    @patch('scaletorch.utils.utils.is_accelerator_available', return_value=False)
    def test_reproducibility(self, mock_accel):
        import random
        import numpy as np

        set_all_seed(123)
        py_val = random.random()
        np_val = np.random.random()
        torch_val = torch.rand(1).item()

        set_all_seed(123)
        self.assertEqual(random.random(), py_val)
        self.assertEqual(np.random.random(), np_val)
        self.assertEqual(torch.rand(1).item(), torch_val)


class TestGetMfu(unittest.TestCase):
    """Test MFU calculation."""

    @patch('scaletorch.utils.utils.get_theoretical_flops', return_value=100e12)
    def test_basic_mfu(self, mock_flops):
        from scaletorch.utils.utils import get_mfu

        config = MagicMock()
        config.num_hidden_layers = 12
        config.hidden_size = 768
        config.max_position_embeddings = 1024

        mfu = get_mfu(
            tokens_per_second=1000,
            num_params=125_000_000,
            model_config=config,
        )
        self.assertGreater(mfu, 0)
        self.assertIsInstance(mfu, float)

    @patch('scaletorch.utils.utils.get_theoretical_flops', return_value=100e12)
    def test_zero_tokens(self, mock_flops):
        from scaletorch.utils.utils import get_mfu

        config = MagicMock()
        config.num_hidden_layers = 12
        config.hidden_size = 768
        config.max_position_embeddings = 1024

        mfu = get_mfu(0, 125_000_000, config)
        self.assertEqual(mfu, 0.0)

    def test_negative_tokens_raises(self):
        from scaletorch.utils.utils import get_mfu

        config = MagicMock()
        config.num_hidden_layers = 12
        config.hidden_size = 768
        config.max_position_embeddings = 1024

        with self.assertRaises(ValueError):
            get_mfu(-1, 100, config)

    def test_missing_config_attrs(self):
        from scaletorch.utils.utils import get_mfu

        config = MagicMock(spec=[])
        with self.assertRaises(AttributeError):
            get_mfu(1000, 100, config)


if __name__ == '__main__':
    unittest.main()
