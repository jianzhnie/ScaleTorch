#!/usr/bin/env python3
"""
Test script for process_group.py module.
Tests ProcessGroupManager functionality for distributed training.
"""

import unittest
from unittest.mock import MagicMock, patch

from scaletorch.parallel.process_group import (ProcessGroupManager,
                                            get_process_group_manager,
                                            setup_process_group_manager)


class TestProcessGroupManager(unittest.TestCase):
    """Test cases for ProcessGroupManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self._patchers = []

        patchers = [
            patch('scaletorch.parallel.process_group.is_distributed',
                  return_value=True),
            patch('scaletorch.parallel.process_group.get_rank', return_value=0),
            patch('scaletorch.parallel.process_group.get_world_size',
                  return_value=8),
            patch('scaletorch.parallel.process_group.new_group',
                  return_value=MagicMock()),
        ]
        self.mock_is_distributed, self.mock_get_rank, \
            self.mock_get_world_size, self.mock_new_group = [
                p.start() for p in patchers
            ]
        self._patchers = patchers

        # Clear global process group manager
        import scaletorch.parallel.process_group as pgm
        pgm.process_group_manager = None

    def tearDown(self):
        """Clean up test fixtures."""
        for p in self._patchers:
            p.stop()
        # Clear global process group manager
        import scaletorch.parallel.process_group as pgm
        pgm.process_group_manager = None

    def test_init_valid_configuration(self):
        """Test ProcessGroupManager initialization with valid configuration."""
        # For this test, just verify the validation logic without
        # trying to create actual groups
        with patch.object(ProcessGroupManager, '_initialize_group_properties'):
            process_group = ProcessGroupManager(tp_size=2,
                                             cp_size=2,
                                             pp_size=2,
                                             dp_size=1)

            self.assertEqual(process_group.global_rank, 0)
            self.assertEqual(process_group.world_size, 8)
            self.assertEqual(process_group.dp_rank, 0)
            self.assertEqual(process_group.pp_rank, 0)
            self.assertEqual(process_group.cp_rank, 0)
            self.assertEqual(process_group.tp_rank, 0)

    def test_init_invalid_world_size(self):
        """Test ProcessGroupManager initialization with invalid world size."""
        self.mock_get_world_size.return_value = 16  # Doesn't match 2*2*2*1=8

        with self.assertRaises(ValueError) as context:
            ProcessGroupManager(tp_size=2, cp_size=2, pp_size=2, dp_size=1)

        self.assertIn(
            'World size (16) != TP (2) * CP (2) * PP (2) * DP (1) = 8',
            str(context.exception))

    def test_init_distributed_not_initialized(self):
        """Test ProcessGroupManager when distributed is not initialized."""
        self.mock_is_distributed.return_value = False

        with self.assertRaises(RuntimeError) as context:
            ProcessGroupManager(tp_size=2, cp_size=2, pp_size=2, dp_size=1)

        self.assertIn('Distributed training must be initialized',
                      str(context.exception))

    def test_init_invalid_parallelism_sizes(self):
        """Test ProcessGroupManager with invalid parallelism sizes."""
        test_cases = [
            (0, 2, 2, 2, 'tp_size must be positive'),
            (2, -1, 2, 2, 'cp_size must be positive'),
            (2, 2, 0, 2, 'pp_size must be positive'),
            (2, 2, 2, -5, 'dp_size must be positive'),
        ]

        for tp_size, cp_size, pp_size, dp_size, expected_msg in test_cases:
            with self.assertRaises(ValueError) as context:
                ProcessGroupManager(tp_size=tp_size,
                                    cp_size=cp_size,
                                    pp_size=pp_size,
                                    dp_size=dp_size)

            self.assertIn(expected_msg, str(context.exception))

    def test_process_group_creation(self):
        """Test that process groups are created correctly."""
        with patch.object(ProcessGroupManager, '_initialize_group_properties'):
            ProcessGroupManager(tp_size=2, cp_size=2, pp_size=2, dp_size=1)

            # Verify that new_group was called for each parallelism type
            # Total groups created:
            # TP groups: dp_size * pp_size * cp_size = 1*2*2 = 4
            # CP groups: dp_size * pp_size * tp_size = 1*2*2 = 4
            # PP groups: dp_size * cp_size * tp_size = 1*2*2 = 4
            # DP groups: pp_size * cp_size * tp_size = 2*2*2 = 8
            # CP_DP groups: pp_size * tp_size = 2*2 = 4
            # PP_DP groups: cp_size * tp_size = 2*2 = 4
            # Total: 4+4+4+8+4+4 = 28
            self.assertEqual(self.mock_new_group.call_count,
                             28)  # tp, cp, pp, dp, cp_dp, pp_dp groups

    def test_group_properties_initialization(self):
        """Test that group properties are initialized correctly."""
        # This test verifies that ProcessGroupManager initialization works
        with patch.object(ProcessGroupManager, '_initialize_group_properties'):
            process_group = ProcessGroupManager(tp_size=2,
                                             cp_size=2,
                                             pp_size=2,
                                             dp_size=1)

            # Verify rank assignments are set correctly
            self.assertEqual(process_group.dp_rank, 0)
            self.assertEqual(process_group.pp_rank, 0)
            self.assertEqual(process_group.cp_rank, 0)
            self.assertEqual(process_group.tp_rank, 0)
            # Verify grid is created correctly
            self.assertIsNotNone(process_group.grid)
            self.assertEqual(process_group.grid.numel(), 8)  # 2*2*2*1

    def test_get_info(self):
        """Test get_info method returns correct information."""
        with patch.object(ProcessGroupManager, '_initialize_group_properties'):
            process_group = ProcessGroupManager(tp_size=2,
                                             cp_size=2,
                                             pp_size=2,
                                             dp_size=1)

            # Mock all the properties that get_info needs
            process_group.tp_world_size = 2
            process_group.tp_rank = 0
            process_group.cp_world_size = 2
            process_group.cp_rank = 0
            process_group.pp_world_size = 2
            process_group.pp_is_first_stage = True
            process_group.pp_next_rank = 1
            process_group.pp_prev_rank = None
            process_group.dp_world_size = 1

            # Verify get_info works and returns a string
            info = process_group.get_info()
            self.assertIsInstance(info, str)
            self.assertIn('Rank 0', info)

    def test_string_representation(self):
        """Test string representations of ProcessGroupManager."""
        with patch.object(ProcessGroupManager, '_initialize_group_properties'):
            process_group = ProcessGroupManager(tp_size=2,
                                             cp_size=2,
                                             pp_size=2,
                                             dp_size=1)

            # Mock the properties needed by __str__ and __repr__
            process_group.tp_world_size = 2
            process_group.cp_world_size = 2
            process_group.pp_world_size = 2
            process_group.dp_world_size = 1

            str_repr = str(process_group)
            self.assertIsInstance(str_repr, str)
            self.assertIn('TP(2)', str_repr)

            repr_str = repr(process_group)
            self.assertIsInstance(repr_str, str)

    def test_setup_process_group_manager(self):
        """Test setup_process_group_manager function."""
        with patch.object(ProcessGroupManager, '_initialize_group_properties'):
            process_group = setup_process_group_manager(tp_size=2,
                                                     cp_size=2,
                                                     pp_size=2,
                                                     dp_size=1)

            self.assertIsNotNone(process_group)
            self.assertIsInstance(process_group, ProcessGroupManager)

            # Check that global process_group_manager was set
            import scaletorch.parallel.process_group as pgm
            self.assertIs(pgm.process_group_manager, process_group)

    def test_get_process_group_manager(self):
        """Test get_process_group_manager function."""
        # Test when manager is not set
        import scaletorch.parallel.process_group as pgm
        pgm.process_group_manager = None

        manager = get_process_group_manager()
        self.assertIsNone(manager)

        # Test when manager is set
        with patch.object(ProcessGroupManager, '_initialize_group_properties'):
            process_group = setup_process_group_manager(tp_size=2,
                                                     cp_size=2,
                                                     pp_size=2,
                                                     dp_size=1)
            manager = get_process_group_manager()
            self.assertIs(manager, process_group)


if __name__ == '__main__':
    unittest.main()
