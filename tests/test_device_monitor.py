"""Tests for scaletorch.utils.device and scaletorch.utils.monitor modules."""

import json
import os
import unittest
from unittest.mock import MagicMock, patch

import torch

from scaletorch.utils.device import get_dist_info, get_visible_devices_keyword


class TestGetDistInfo(unittest.TestCase):

    def test_defaults_without_env(self):
        with patch.dict(os.environ, {}, clear=True):
            rank, world_size, local_rank = get_dist_info()
            self.assertEqual(rank, 0)
            self.assertEqual(world_size, 1)
            self.assertEqual(local_rank, 0)

    def test_with_env_vars(self):
        env = {'RANK': '3', 'WORLD_SIZE': '8', 'LOCAL_RANK': '1'}
        with patch.dict(os.environ, env, clear=True):
            rank, world_size, local_rank = get_dist_info()
            self.assertEqual(rank, 3)
            self.assertEqual(world_size, 8)
            self.assertEqual(local_rank, 1)


class TestGetVisibleDevicesKeyword(unittest.TestCase):

    def test_returns_string(self):
        result = get_visible_devices_keyword()
        self.assertIsInstance(result, str)
        self.assertIn('VISIBLE_DEVICES', result)


class TestPerformanceMonitor(unittest.TestCase):

    def test_monitor_lifecycle(self):
        from scaletorch.utils.monitor import PerformanceMonitor
        config = MagicMock()
        monitor = PerformanceMonitor(config=config, log_dir=None)
        monitor.start()
        monitor.start_iteration(tokens_processed=100)

        stats = monitor.end_iteration()
        self.assertIn('iteration_time', stats)
        self.assertIn('tokens_processed', stats)
        self.assertEqual(stats['tokens_processed'], 100)

    def test_end_without_start_raises(self):
        from scaletorch.utils.monitor import PerformanceMonitor
        config = MagicMock()
        monitor = PerformanceMonitor(config=config)
        with self.assertRaises(RuntimeError):
            monitor.end_iteration()

    def test_average_stats(self):
        from scaletorch.utils.monitor import PerformanceMonitor
        config = MagicMock()
        monitor = PerformanceMonitor(config=config)
        monitor.start()

        monitor.start_iteration(tokens_processed=100)
        monitor.end_iteration()

        monitor.start_iteration(tokens_processed=200)
        monitor.end_iteration()

        avg = monitor.get_average_stats()
        self.assertEqual(avg['total_iterations'], 2)
        self.assertAlmostEqual(avg['total_tokens_processed'], 300)

    def test_save_stats(self):
        import tempfile

        from scaletorch.utils.monitor import PerformanceMonitor

        config = MagicMock()
        config.__dict__ = {'test': True}

        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = PerformanceMonitor(config=config, log_dir=tmpdir)
            monitor.start()
            monitor.start_iteration(tokens_processed=50)
            monitor.end_iteration()

            filepath = monitor.save_stats()
            self.assertTrue(filepath.endswith('.json'))

            with open(filepath) as f:
                data = json.load(f)
            self.assertIn('stats', data)
            self.assertIn('average_stats', data)

    def test_print_summary_no_stats(self):
        from scaletorch.utils.monitor import PerformanceMonitor
        config = MagicMock()
        monitor = PerformanceMonitor(config=config)
        monitor.print_summary()  # Should not raise

    def test_save_no_stats(self):
        from scaletorch.utils.monitor import PerformanceMonitor
        config = MagicMock()
        monitor = PerformanceMonitor(config=config)
        result = monitor.save_stats()
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
