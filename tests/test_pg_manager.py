#!/usr/bin/env python3
"""
Test script for pg_manager.py module.
Tests ProcessGroupManager functionality for distributed training.
"""

import unittest
from unittest.mock import MagicMock, patch

from scaletorch.parallel.pg_manager import (ProcessGroupManager,
                                            get_process_group_manager,
                                            setup_process_group_manager)


class TestProcessGroupManager(unittest.TestCase):
    """Test cases for ProcessGroupManager class."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock torch.distributed functions
        self.dist_patcher = patch('scaletorch.parallel.pg_manager.dist')
        self.mock_dist = self.dist_patcher.start()

        # Configure mock distributed environment
        self.mock_dist.is_initialized.return_value = True
        self.mock_dist.get_rank.return_value = 0

        # Mock process group creation
        self.mock_dist.new_group.return_value = MagicMock()
        # Reset side_effect for each setUp - return world_size=8 for first call,
        # then world_size=2 for group property initialization calls
        self.mock_dist.get_world_size.side_effect = None
        self.mock_dist.get_world_size.return_value = 8

        # Clear global process group manager
        import scaletorch.parallel.pg_manager as pgm
        pgm.process_group_manager = None

    def tearDown(self):
        """Clean up test fixtures."""
        self.dist_patcher.stop()
        # Clear global process group manager
        import scaletorch.parallel.pg_manager as pgm
        pgm.process_group_manager = None

    def test_init_valid_configuration(self):
        """Test ProcessGroupManager initialization with valid configuration."""
        # For this test, just verify the validation logic without
        # trying to create actual groups
        with patch.object(ProcessGroupManager, '_initialize_group_properties'):
            pg_manager = ProcessGroupManager(tp_size=2,
                                             cp_size=2,
                                             pp_size=2,
                                             dp_size=1)

            self.assertEqual(pg_manager.global_rank, 0)
            self.assertEqual(pg_manager.world_size, 8)
            self.assertEqual(pg_manager.dp_rank, 0)
            self.assertEqual(pg_manager.pp_rank, 0)
            self.assertEqual(pg_manager.cp_rank, 0)
            self.assertEqual(pg_manager.tp_rank, 0)

    def test_init_invalid_world_size(self):
        """Test ProcessGroupManager initialization with invalid world size."""
        self.mock_dist.get_world_size.return_value = 16  # Doesn't match 2*2*2*1=8

        with self.assertRaises(ValueError) as context:
            ProcessGroupManager(tp_size=2, cp_size=2, pp_size=2, dp_size=1)

        self.assertIn(
            'World size (16) != TP (2) * CP (2) * PP (2) * DP (1) = 8',
            str(context.exception))

    def test_init_distributed_not_initialized(self):
        """Test ProcessGroupManager when distributed is not initialized."""
        self.mock_dist.is_initialized.return_value = False

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
            self.assertEqual(self.mock_dist.new_group.call_count,
                             28)  # tp, cp, pp, dp, cp_dp, pp_dp groups

    def test_group_properties_initialization(self):
        """Test that group properties are initialized correctly."""
        # This test verifies that ProcessGroupManager initialization works
        with patch.object(ProcessGroupManager, '_initialize_group_properties'):
            pg_manager = ProcessGroupManager(tp_size=2,
                                             cp_size=2,
                                             pp_size=2,
                                             dp_size=1)

            # Verify rank assignments are set correctly
            self.assertEqual(pg_manager.dp_rank, 0)
            self.assertEqual(pg_manager.pp_rank, 0)
            self.assertEqual(pg_manager.cp_rank, 0)
            self.assertEqual(pg_manager.tp_rank, 0)
            # Verify grid is created correctly
            self.assertIsNotNone(pg_manager.grid)
            self.assertEqual(pg_manager.grid.numel(), 8)  # 2*2*2*1

    def test_get_info(self):
        """Test get_info method returns correct information."""
        with patch.object(ProcessGroupManager, '_initialize_group_properties'):
            pg_manager = ProcessGroupManager(tp_size=2,
                                             cp_size=2,
                                             pp_size=2,
                                             dp_size=1)

            # Mock all the properties that get_info needs
            pg_manager.tp_world_size = 2
            pg_manager.tp_rank = 0
            pg_manager.cp_world_size = 2
            pg_manager.cp_rank = 0
            pg_manager.pp_world_size = 2
            pg_manager.pp_is_first_stage = True
            pg_manager.pp_next_rank = 1
            pg_manager.pp_prev_rank = None
            pg_manager.dp_world_size = 1

            # Verify get_info works and returns a string
            info = pg_manager.get_info()
            self.assertIsInstance(info, str)
            self.assertIn('Rank 0', info)

    def test_string_representation(self):
        """Test string representations of ProcessGroupManager."""
        with patch.object(ProcessGroupManager, '_initialize_group_properties'):
            pg_manager = ProcessGroupManager(tp_size=2,
                                             cp_size=2,
                                             pp_size=2,
                                             dp_size=1)

            # Mock the properties needed by __str__ and __repr__
            pg_manager.tp_world_size = 2
            pg_manager.cp_world_size = 2
            pg_manager.pp_world_size = 2
            pg_manager.dp_world_size = 1

            str_repr = str(pg_manager)
            self.assertIsInstance(str_repr, str)
            self.assertIn('TP(2)', str_repr)

            repr_str = repr(pg_manager)
            self.assertIsInstance(repr_str, str)

    def test_setup_process_group_manager(self):
        """Test setup_process_group_manager function."""
        with patch.object(ProcessGroupManager, '_initialize_group_properties'):
            pg_manager = setup_process_group_manager(tp_size=2,
                                                     cp_size=2,
                                                     pp_size=2,
                                                     dp_size=1)

            self.assertIsNotNone(pg_manager)
            self.assertIsInstance(pg_manager, ProcessGroupManager)

            # Check that global process_group_manager was set
            import scaletorch.parallel.pg_manager as pgm
            self.assertIs(pgm.process_group_manager, pg_manager)

    def test_get_process_group_manager(self):
        """Test get_process_group_manager function."""
        # Test when manager is not set
        import scaletorch.parallel.pg_manager as pgm
        pgm.process_group_manager = None

        manager = get_process_group_manager()
        self.assertIsNone(manager)

        # Test when manager is set
        with patch.object(ProcessGroupManager, '_initialize_group_properties'):
            pg_manager = setup_process_group_manager(tp_size=2,
                                                     cp_size=2,
                                                     pp_size=2,
                                                     dp_size=1)
            manager = get_process_group_manager()
            self.assertIs(manager, pg_manager)


if __name__ == '__main__':
    unittest.main()
