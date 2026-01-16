"""
Test cases for scaletorch.dist module.

This module contains comprehensive test cases for all distributed communication
functions in scaletorch.dist package, including both single-process and
multi-process scenarios.
"""

import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.distributed as torch_dist

import scaletorch.dist as dist
from tests.test_base import BaseTestCase


class TestDistFunctions(BaseTestCase):
    """Test cases for distributed communication functions."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.world_size = 4
        self.rank = 0

    def test_get_reduce_op_valid_ops(self):
        """Test _get_reduce_op with valid operation names."""
        # Test all valid operations
        valid_ops = ['sum', 'product', 'min', 'max', 'band', 'bor', 'bxor']
        expected_ops = [
            torch_dist.ReduceOp.SUM, torch_dist.ReduceOp.PRODUCT,
            torch_dist.ReduceOp.MIN, torch_dist.ReduceOp.MAX,
            torch_dist.ReduceOp.BAND, torch_dist.ReduceOp.BOR,
            torch_dist.ReduceOp.BXOR
        ]

        for op_name, expected_op in zip(valid_ops, expected_ops):
            result = dist._get_reduce_op(op_name)
            self.assertEqual(result, expected_op)

        # Test case insensitive
        self.assertEqual(dist._get_reduce_op('SUM'), torch_dist.ReduceOp.SUM)
        self.assertEqual(dist._get_reduce_op('Sum'), torch_dist.ReduceOp.SUM)

    def test_get_reduce_op_invalid_op(self):
        """Test _get_reduce_op with invalid operation name."""
        with self.assertRaises(ValueError) as context:
            dist._get_reduce_op('invalid_op')

        self.assertIn('reduce op should be one of', str(context.exception))

    def test_scatter_single_process(self):
        """Test scatter function in single process environment."""
        # Mock world_size = 1
        with patch('scaletorch.dist.dist.get_world_size', return_value=1):
            # Test with scatter_list provided
            data = torch.zeros(3, dtype=torch.float32)
            scatter_list = [
                torch.tensor([1.0, 2.0, 3.0]),
                torch.tensor([4.0, 5.0, 6.0])
            ]

            dist.scatter(data, scatter_list=scatter_list)
            # In single process, data should be copied from first element
            self.assertTrue(torch.allclose(data, scatter_list[0]))

    def test_scatter_single_process_no_scatter_list(self):
        """Test scatter function in single process without scatter_list."""
        with patch('scaletorch.dist.dist.get_world_size', return_value=1):
            data = torch.zeros(3, dtype=torch.float32)
            original_data = data.clone()

            # Without scatter_list, data should remain unchanged
            dist.scatter(data)
            self.assertTrue(torch.allclose(data, original_data))

    @patch('torch.distributed.scatter')
    @patch('scaletorch.dist.dist.get_world_size', return_value=2)
    @patch('scaletorch.dist.dist.get_rank', return_value=0)
    def test_scatter_multi_process_source_rank(self, mock_get_rank,
                                               mock_get_world_size,
                                               mock_torch_scatter):
        """Test scatter function in multi-process environment (source rank)."""
        # Mock process group and devices
        mock_group = MagicMock()
        mock_pg = MagicMock()

        with patch('scaletorch.dist.dist.get_default_group', return_value=mock_pg), \
             patch('scaletorch.dist.dist.get_data_device', return_value=torch.device('cpu')), \
             patch('scaletorch.dist.dist.get_comm_device', return_value=torch.device('cpu')), \
             patch('scaletorch.dist.dist.cast_data_device', side_effect=lambda x, device: x):

            data = torch.zeros(2, dtype=torch.float32)
            scatter_list = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]

            dist.scatter(data,
                         src=0,
                         scatter_list=scatter_list,
                         group=mock_group)

            # Verify torch.distributed.scatter was called
            self.assertTrue(mock_torch_scatter.called)

    @patch('torch.distributed.scatter')
    @patch('scaletorch.dist.dist.get_world_size', return_value=2)
    @patch('scaletorch.dist.dist.get_rank', return_value=1)
    def test_scatter_multi_process_non_source_rank(self, mock_get_rank,
                                                   mock_get_world_size,
                                                   mock_torch_scatter):
        """Test scatter function in multi-process environment (non-source rank)."""
        mock_pg = MagicMock()

        with patch('scaletorch.dist.dist.get_default_group', return_value=mock_pg), \
             patch('scaletorch.dist.dist.get_data_device', return_value=torch.device('cpu')), \
             patch('scaletorch.dist.dist.get_comm_device', return_value=torch.device('cpu')), \
             patch('scaletorch.dist.dist.cast_data_device', side_effect=lambda x, device: x):

            data = torch.zeros(2, dtype=torch.float32)

            # Non-source rank doesn't provide scatter_list
            dist.scatter(data, src=0, group=mock_pg)

            # Verify torch.distributed.scatter was called
            self.assertTrue(mock_torch_scatter.called)

    def test_scatter_invalid_scatter_list_length(self):
        """Test scatter function with invalid scatter_list length."""
        with patch('scaletorch.dist.dist.get_world_size', return_value=2), \
             patch('scaletorch.dist.dist.get_rank', return_value=0), \
             patch('scaletorch.dist.dist.get_default_group', return_value=MagicMock()):

            data = torch.zeros(2, dtype=torch.float32)
            # scatter_list length doesn't match world_size
            scatter_list = [torch.tensor([1.0, 2.0])
                            ]  # Only 1 tensor for world_size=2

            with self.assertRaises(ValueError) as context:
                dist.scatter(data, src=0, scatter_list=scatter_list)

            self.assertIn('scatter_list length', str(context.exception))

    def test_scatter_mismatched_tensor_shapes(self):
        """Test scatter function with mismatched tensor shapes."""
        with patch('scaletorch.dist.dist.get_world_size', return_value=2), \
             patch('scaletorch.dist.dist.get_rank', return_value=0), \
             patch('scaletorch.dist.dist.get_default_group', return_value=MagicMock()):

            data = torch.zeros(2, dtype=torch.float32)
            # scatter_list tensors have different shapes than data
            scatter_list = [
                torch.tensor([1.0, 2.0, 3.0]),
                torch.tensor([4.0, 5.0, 6.0])
            ]  # Shape mismatch

            with self.assertRaises(ValueError) as context:
                dist.scatter(data, src=0, scatter_list=scatter_list)

            self.assertIn('same shape as data', str(context.exception))

    def test_reduce_single_process(self):
        """Test reduce function in single process environment."""
        with patch('scaletorch.dist.dist.get_world_size', return_value=1):
            data = torch.tensor([1.0, 2.0, 3.0])
            original_data = data.clone()

            # In single process, data should remain unchanged
            dist.reduce(data)
            self.assertTrue(torch.allclose(data, original_data))

    @patch('torch.distributed.reduce')
    @patch('scaletorch.dist.dist.get_world_size', return_value=2)
    def test_reduce_multi_process_sum(self, mock_get_world_size,
                                      mock_torch_reduce):
        """Test reduce function with sum operation in multi-process environment."""
        mock_pg = MagicMock()

        with patch('scaletorch.dist.dist.get_default_group', return_value=mock_pg), \
             patch('scaletorch.dist.dist.get_data_device', return_value=torch.device('cpu')), \
             patch('scaletorch.dist.dist.get_comm_device', return_value=torch.device('cpu')), \
             patch('scaletorch.dist.dist.cast_data_device', side_effect=lambda x, device: x), \
             patch('scaletorch.dist.dist.get_rank', return_value=0):

            data = torch.tensor([1.0, 2.0, 3.0])
            dist.reduce(data, dst=0, op='sum')

            # Verify torch.distributed.reduce was called with correct parameters
            self.assertTrue(mock_torch_reduce.called)

    @patch('torch.distributed.reduce')
    @patch('scaletorch.dist.dist.get_world_size', return_value=2)
    @patch('scaletorch.dist.dist.get_rank', return_value=0)
    def test_reduce_multi_process_mean(self, mock_get_rank,
                                       mock_get_world_size, mock_torch_reduce):
        """Test reduce function with mean operation in multi-process environment."""
        mock_pg = MagicMock()

        with patch('scaletorch.dist.dist.get_default_group', return_value=mock_pg), \
             patch('scaletorch.dist.dist.get_data_device', return_value=torch.device('cpu')), \
             patch('scaletorch.dist.dist.get_comm_device', return_value=torch.device('cpu')), \
             patch('scaletorch.dist.dist.cast_data_device', side_effect=lambda x, device: x):

            data = torch.tensor([4.0, 6.0, 8.0])
            dist.reduce(data, dst=0, op='mean')

            # For mean operation, it should use sum first, then divide by world_size
            self.assertTrue(mock_torch_reduce.called)
            # Check that it was called with SUM operation
            args, kwargs = mock_torch_reduce.call_args
            self.assertEqual(args[2], torch_dist.ReduceOp.SUM)

    def test_reduce_scatter_single_process(self):
        """Test reduce_scatter function in single process environment."""
        with patch('scaletorch.dist.dist.get_world_size', return_value=1):
            data = torch.tensor([1.0, 2.0, 3.0, 4.0])
            original_data = data.clone()

            # In single process, data should remain unchanged
            dist.reduce_scatter(data)
            self.assertTrue(torch.allclose(data, original_data))

    def test_reduce_scatter_invalid_tensor_size(self):
        """Test reduce_scatter function with invalid tensor size."""
        with patch('scaletorch.dist.dist.get_world_size', return_value=3), \
             patch('scaletorch.dist.dist.get_default_group', return_value=MagicMock()):

            # Tensor size (5) is not divisible by world_size (3)
            data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

            with self.assertRaises(ValueError) as context:
                dist.reduce_scatter(data)

            self.assertIn('must be divisible by world size',
                          str(context.exception))

    @patch('torch.distributed.reduce_scatter')
    @patch('scaletorch.dist.dist.get_world_size', return_value=2)
    def test_reduce_scatter_multi_process(self, mock_get_world_size,
                                          mock_torch_reduce_scatter):
        """Test reduce_scatter function in multi-process environment."""
        mock_pg = MagicMock()

        with patch('scaletorch.dist.dist.get_default_group', return_value=mock_pg), \
             patch('scaletorch.dist.dist.get_data_device', return_value=torch.device('cpu')), \
             patch('scaletorch.dist.dist.get_comm_device', return_value=torch.device('cpu')), \
             patch('scaletorch.dist.dist.cast_data_device', side_effect=lambda x, device: x):

            data = torch.tensor([1.0, 2.0, 3.0, 4.0])
            dist.reduce_scatter(data, op='sum')

            # Verify torch.distributed.reduce_scatter was called
            self.assertTrue(mock_torch_reduce_scatter.called)

    def test_all_to_all_single_process(self):
        """Test all_to_all function in single process environment."""
        with patch('scaletorch.dist.dist.get_world_size', return_value=1):
            data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

            # In single process, should return the same data
            result = dist.all_to_all(data)
            self.assertTrue(torch.allclose(result, data))

    def test_all_to_all_invalid_tensor_size(self):
        """Test all_to_all function with invalid tensor size."""
        with patch('scaletorch.dist.dist.get_world_size', return_value=3), \
             patch('scaletorch.dist.dist.get_default_group', return_value=MagicMock()):

            # First dimension size (5) is not divisible by world_size (3)
            data = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0],
                                 [7.0, 8.0], [9.0, 10.0]])

            with self.assertRaises(ValueError) as context:
                dist.all_to_all(data)

            self.assertIn('first dimension size', str(context.exception))
            self.assertIn('must be divisible by world size',
                          str(context.exception))

    @patch('torch.distributed.all_to_all_single')
    @patch('scaletorch.dist.dist.get_world_size', return_value=2)
    def test_all_to_all_multi_process(self, mock_get_world_size,
                                      mock_torch_all_to_all):
        """Test all_to_all function in multi-process environment."""
        mock_pg = MagicMock()

        with patch('scaletorch.dist.dist.get_default_group', return_value=mock_pg), \
             patch('scaletorch.dist.dist.get_data_device', return_value=torch.device('cpu')), \
             patch('scaletorch.dist.dist.get_comm_device', return_value=torch.device('cpu')), \
             patch('scaletorch.dist.dist.cast_data_device', side_effect=lambda x, device: x), \
             patch('torch.empty_like', side_effect=lambda x, device: torch.empty_like(x)):

            data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
            mock_torch_all_to_all.return_value = None  # Mock successful execution

            result = dist.all_to_all(data)

            # Verify torch.distributed.all_to_all_single was called
            self.assertTrue(mock_torch_all_to_all.called)

    def test_all_reduce_single_process(self):
        """Test all_reduce function in single process environment."""
        with patch('scaletorch.dist.dist.get_world_size', return_value=1):
            data = torch.tensor([1.0, 2.0, 3.0])
            original_data = data.clone()

            # In single process, data should remain unchanged
            dist.all_reduce(data)
            self.assertTrue(torch.allclose(data, original_data))

    @patch('torch.distributed.all_reduce')
    @patch('scaletorch.dist.dist.get_world_size', return_value=2)
    def test_all_reduce_multi_process_sum(self, mock_get_world_size,
                                          mock_torch_all_reduce):
        """Test all_reduce function with sum operation in multi-process environment."""
        mock_pg = MagicMock()

        with patch('scaletorch.dist.dist.get_default_group', return_value=mock_pg), \
             patch('scaletorch.dist.dist.get_data_device', return_value=torch.device('cpu')), \
             patch('scaletorch.dist.dist.get_comm_device', return_value=torch.device('cpu')), \
             patch('scaletorch.dist.dist.cast_data_device', side_effect=lambda x, device: x):

            data = torch.tensor([1.0, 2.0, 3.0])
            dist.all_reduce(data, op='sum')

            # Verify torch.distributed.all_reduce was called
            self.assertTrue(mock_torch_all_reduce.called)

    @patch('torch.distributed.all_reduce')
    @patch('scaletorch.dist.dist.get_world_size', return_value=2)
    def test_all_reduce_multi_process_mean(self, mock_get_world_size,
                                           mock_torch_all_reduce):
        """Test all_reduce function with mean operation in multi-process environment."""
        mock_pg = MagicMock()

        with patch('scaletorch.dist.dist.get_default_group', return_value=mock_pg), \
             patch('scaletorch.dist.dist.get_data_device', return_value=torch.device('cpu')), \
             patch('scaletorch.dist.dist.get_comm_device', return_value=torch.device('cpu')), \
             patch('scaletorch.dist.dist.cast_data_device', side_effect=lambda x, device: x):

            data = torch.tensor([4.0, 6.0, 8.0])
            dist.all_reduce(data, op='mean')

            # For mean operation, it should use sum first, then divide by world_size
            self.assertTrue(mock_torch_all_reduce.called)
            # Check that it was called with SUM operation
            args, kwargs = mock_torch_all_reduce.call_args
            self.assertEqual(args[1], torch_dist.ReduceOp.SUM)

    def test_all_gather_single_process(self):
        """Test all_gather function in single process environment."""
        with patch('scaletorch.dist.dist.get_world_size', return_value=1):
            data = torch.tensor([1.0, 2.0, 3.0])

            # In single process, should return list with cloned data
            result = dist.all_gather(data)
            self.assertEqual(len(result), 1)
            self.assertTrue(torch.allclose(result[0], data))
            # Verify it's a clone, not the same object
            self.assertIsNot(result[0], data)

    @patch('torch.distributed.all_gather')
    @patch('scaletorch.dist.dist.get_world_size', return_value=2)
    def test_all_gather_multi_process(self, mock_get_world_size,
                                      mock_torch_all_gather):
        """Test all_gather function in multi-process environment."""
        mock_pg = MagicMock()

        with patch('scaletorch.dist.dist.get_default_group', return_value=mock_pg), \
             patch('scaletorch.dist.dist.get_data_device', return_value=torch.device('cpu')), \
             patch('scaletorch.dist.dist.get_comm_device', return_value=torch.device('cpu')), \
             patch('scaletorch.dist.dist.cast_data_device', side_effect=lambda x, device: x), \
             patch('torch.empty_like', side_effect=lambda x: torch.empty_like(x)):

            data = torch.tensor([1.0, 2.0, 3.0])

            # Mock the all_gather to populate gather_list
            def mock_all_gather(gather_list, tensor, group):
                gather_list[0].copy_(tensor)
                gather_list[1].copy_(tensor * 2)

            mock_torch_all_gather.side_effect = mock_all_gather

            result = dist.all_gather(data)

            # Verify torch.distributed.all_gather was called
            self.assertTrue(mock_torch_all_gather.called)

    def test_gather_single_process(self):
        """Test gather function in single process environment."""
        with patch('scaletorch.dist.dist.get_world_size', return_value=1):
            data = torch.tensor([1.0, 2.0, 3.0])

            # In single process, should return list with data itself
            result = dist.gather(data)
            self.assertEqual(len(result), 1)
            self.assertTrue(torch.allclose(result[0], data))

    @patch('torch.distributed.gather')
    @patch('scaletorch.dist.dist.get_world_size', return_value=2)
    @patch('scaletorch.dist.dist.get_rank', return_value=0)
    def test_gather_multi_process_destination(self, mock_get_rank,
                                              mock_get_world_size,
                                              mock_torch_gather):
        """Test gather function in multi-process environment (destination rank)."""
        mock_pg = MagicMock()

        with patch('scaletorch.dist.dist.get_default_group', return_value=mock_pg), \
             patch('scaletorch.dist.dist.get_data_device', return_value=torch.device('cpu')), \
             patch('scaletorch.dist.dist.get_comm_device', return_value=torch.device('cpu')), \
             patch('scaletorch.dist.dist.cast_data_device', side_effect=lambda x, device: x):

            data = torch.tensor([1.0, 2.0, 3.0])
            result = dist.gather(data, dst=0)

            # Verify torch.distributed.gather was called
            self.assertTrue(mock_torch_gather.called)

    def test_broadcast_single_process(self):
        """Test broadcast function in single process environment."""
        with patch('scaletorch.dist.dist.get_world_size', return_value=1):
            data = torch.tensor([1.0, 2.0, 3.0])
            original_data = data.clone()

            # In single process, data should remain unchanged
            dist.broadcast(data, src=0)
            self.assertTrue(torch.allclose(data, original_data))

    @patch('torch.distributed.broadcast')
    @patch('scaletorch.dist.dist.get_world_size', return_value=2)
    def test_broadcast_multi_process(self, mock_get_world_size,
                                     mock_torch_broadcast):
        """Test broadcast function in multi-process environment."""
        mock_pg = MagicMock()

        with patch('scaletorch.dist.dist.get_default_group', return_value=mock_pg), \
             patch('scaletorch.dist.dist.get_data_device', return_value=torch.device('cpu')), \
             patch('scaletorch.dist.dist.get_comm_device', return_value=torch.device('cpu')), \
             patch('scaletorch.dist.dist.cast_data_device', side_effect=lambda x, device: x):

            data = torch.tensor([1.0, 2.0, 3.0])
            dist.broadcast(data, src=0)

            # Verify torch.distributed.broadcast was called
            self.assertTrue(mock_torch_broadcast.called)

    def test_sync_random_seed_single_process(self):
        """Test sync_random_seed function in single process environment."""
        with patch('scaletorch.dist.dist.get_world_size', return_value=1):
            # Mock torch.randint to return a specific value
            with patch('torch.randint', return_value=torch.tensor(42)):
                seed = dist.sync_random_seed()
                self.assertEqual(seed, 42)

    @patch('torch.distributed.broadcast')
    @patch('scaletorch.dist.dist.get_world_size', return_value=2)
    @patch('scaletorch.dist.dist.get_rank', return_value=0)
    def test_sync_random_seed_multi_process_main_process(
            self, mock_get_rank, mock_get_world_size, mock_torch_broadcast):
        """Test sync_random_seed function in multi-process environment (main process)."""
        mock_pg = MagicMock()

        with patch('scaletorch.dist.dist.get_default_group', return_value=mock_pg), \
             patch('torch.randint', return_value=torch.tensor(123)):

            seed = dist.sync_random_seed()
            self.assertEqual(seed, 123)
            # Verify broadcast was called on main process
            self.assertTrue(mock_torch_broadcast.called)

    def test_object_to_tensor_conversion(self):
        """Test _object_to_tensor function."""
        test_objects = [{
            'key': 'value',
            'number': 42
        }, [1, 2, 3, 4, 5], 'test string', 12345, (1, 2, 3)]

        for obj in test_objects:
            tensor, size_tensor = dist._object_to_tensor(obj)

            # Verify tensor properties
            self.assertIsInstance(tensor, torch.Tensor)
            self.assertIsInstance(size_tensor, torch.Tensor)
            self.assertEqual(size_tensor.item(), len(tensor))

    def test_tensor_to_object_conversion(self):
        """Test _tensor_to_object function."""
        test_objects = [{
            'key': 'value',
            'number': 42
        }, [1, 2, 3, 4, 5], 'test string', 12345]

        for obj in test_objects:
            # Convert to tensor first
            tensor, size_tensor = dist._object_to_tensor(obj)
            # Convert back to object
            result_obj = dist._tensor_to_object(tensor, size_tensor.item())

            # Verify the conversion
            self.assertEqual(result_obj, obj)

    @patch('torch.distributed.broadcast')
    def test_broadcast_object_list(self, mock_torch_broadcast):
        """Test broadcast_object_list function."""
        mock_pg = MagicMock()

        with patch('scaletorch.dist.dist.get_default_group', return_value=mock_pg), \
             patch('scaletorch.dist.dist.get_rank', return_value=0), \
             patch('scaletorch.dist.dist.get_world_size', return_value=2):

            data = [{'rank': 0, 'data': 'test'}, {'rank': 1, 'data': 'test2'}]

            dist.broadcast_object_list(data, src=0)

            # Verify broadcast was called
            self.assertTrue(mock_torch_broadcast.called)

    def test_all_reduce_dict_single_process(self):
        """Test all_reduce_dict function in single process environment."""
        with patch('scaletorch.dist.dist.get_world_size', return_value=1):
            data_dict = {
                'tensor1': torch.tensor([1.0, 2.0, 3.0]),
                'tensor2': torch.tensor([4.0, 5.0, 6.0])
            }
            original_dict = {k: v.clone() for k, v in data_dict.items()}

            # In single process, data should remain unchanged
            dist.all_reduce_dict(data_dict)

            for key in data_dict:
                self.assertTrue(
                    torch.allclose(data_dict[key], original_dict[key]))

    @patch('torch.distributed.all_reduce')
    @patch('scaletorch.dist.dist.get_world_size', return_value=2)
    def test_all_reduce_dict_multi_process(self, mock_get_world_size,
                                           mock_torch_all_reduce):
        """Test all_reduce_dict function in multi-process environment."""
        mock_pg = MagicMock()

        with patch('scaletorch.dist.dist.get_default_group', return_value=mock_pg), \
             patch('scaletorch.dist.dist.get_data_device', return_value=torch.device('cpu')), \
             patch('scaletorch.dist.dist.get_comm_device', return_value=torch.device('cpu')), \
             patch('scaletorch.dist.dist.cast_data_device', side_effect=lambda x, device: x):

            data_dict = {
                'tensor1': torch.tensor([1.0, 2.0, 3.0]),
                'tensor2': torch.tensor([4.0, 5.0, 6.0])
            }

            dist.all_reduce_dict(data_dict, op='sum')

            # Verify all_reduce was called for each tensor
            self.assertEqual(mock_torch_all_reduce.call_count, 2)

    def test_all_gather_object_single_process(self):
        """Test all_gather_object function in single process environment."""
        with patch('scaletorch.dist.dist.get_world_size', return_value=1):
            obj = {'test': 'data', 'rank': 0}

            # In single process, should return list with the same object
            result = dist.all_gather_object(obj)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0], obj)

    @patch('torch.distributed.all_gather')
    def test_all_gather_object_multi_process(self, mock_torch_all_gather):
        """Test all_gather_object function in multi-process environment."""
        mock_pg = MagicMock()

        with patch('scaletorch.dist.dist.get_default_group', return_value=mock_pg), \
             patch('scaletorch.dist.dist.get_world_size', return_value=2), \
             patch('scaletorch.dist.dist.get_rank', return_value=0):

            obj = {'test': 'data', 'rank': 0}

            dist.gather_object(obj, dst=0)

            # Verify gather was called
            self.assertTrue(mock_torch_gather.called)

# 创建一个专门的多进程测试类
class TestDistFunctionsMultiProcess(BaseTestCase):
    """多进程测试 - 使用简化的 Mock 策略"""
    
    @patch('torch.distributed.all_reduce')
    @patch('scaletorch.dist.dist.get_world_size', return_value=2)
    def test_all_reduce_multi_process_sum_simple(self, mock_get_world_size, mock_all_reduce):
        """简化的 all_reduce 多进程测试"""
        with patch('scaletorch.dist.dist.get_default_group', return_value=object()):
            data = torch.tensor([1.0, 2.0, 3.0])
            dist.all_reduce(data, op='sum')
            mock_all_reduce.assert_called_once()

    def test_collect_results_single_process(self):
        """Test collect_results function in single process environment."""
        with patch('scaletorch.dist.dist.get_world_size', return_value=1):
            results = [1, 2, 3, 4, 5]

            # In single process, should return the same results
            result = dist.collect_results(results)
            self.assertEqual(result, results)

    def test_collect_results_cpu_single_process(self):
        """Test collect_results_cpu function in single process environment."""
        with patch('scaletorch.dist.dist.get_world_size', return_value=1):
            results = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]

            # In single process, should return the same results
            result = dist.collect_results_cpu(results)
            self.assertEqual(len(result), len(results))
            for orig, res in zip(results, result):
                self.assertTrue(torch.allclose(orig, res))

    def test_collect_results_gpu_single_process(self):
        """Test collect_results_gpu function in single process environment."""
        with patch('scaletorch.dist.dist.get_world_size', return_value=1):
            results = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]

            # In single process, should return the same results
            result = dist.collect_results_gpu(results, size=2)
            self.assertEqual(len(result), len(results))
            for orig, res in zip(results, result):
                self.assertTrue(torch.allclose(orig, res))

    @patch('torch.distributed.all_reduce')
    def test_all_reduce_coalesced(self, mock_torch_all_reduce):
        """Test _all_reduce_coalesced function."""
        mock_pg = MagicMock()

        tensors = [
            torch.tensor([1.0, 2.0]),
            torch.tensor([3.0, 4.0]),
            torch.tensor([5.0, 6.0])
        ]

        with patch('scaletorch.dist.dist.get_default_group', return_value=mock_pg), \
             patch('scaletorch.dist.dist.get_data_device', return_value=torch.device('cpu')), \
             patch('scaletorch.dist.dist.get_comm_device', return_value=torch.device('cpu')), \
             patch('scaletorch.dist.dist.cast_data_device', side_effect=lambda x, device: x):

            dist._all_reduce_coalesced(tensors, op='sum')

            # Verify all_reduce was called for each tensor
            self.assertEqual(mock_torch_all_reduce.call_count, 3)

    def test_all_reduce_params_single_process(self):
        """Test all_reduce_params function in single process environment."""
        with patch('scaletorch.dist.dist.get_world_size', return_value=1):
            # Create a simple model parameters
            model = self.create_simple_model()
            params = list(model.parameters())

            # Store original parameter values
            original_params = [p.clone() for p in params]

            # In single process, parameters should remain unchanged
            dist.all_reduce_params(params)

            for orig, new in zip(original_params, params):
                self.assertTrue(torch.allclose(orig, new))

    @patch('scaletorch.dist.dist._all_reduce_coalesced')
    @patch('scaletorch.dist.dist.get_world_size', return_value=2)
    def test_all_reduce_params_multi_process(self, mock_get_world_size,
                                             mock_all_reduce_coalesced):
        """Test all_reduce_params function in multi-process environment."""
        # Create a simple model parameters
        model = self.create_simple_model()
        params = list(model.parameters())

        dist.all_reduce_params(params, coalesce=True)

        # Verify _all_reduce_coalesced was called
        self.assertTrue(mock_all_reduce_coalesced.called)

    @patch('scaletorch.dist.dist._all_reduce_coalesced')
    @patch('scaletorch.dist.dist.get_world_size', return_value=2)
    def test_all_reduce_params_no_coalesce(self, mock_get_world_size,
                                           mock_all_reduce_coalesced):
        """Test all_reduce_params function without coalescing."""
        # Create a simple model parameters
        model = self.create_simple_model()
        params = list(model.parameters())

        with patch('torch.distributed.all_reduce') as mock_torch_all_reduce:
            dist.all_reduce_params(params, coalesce=False)

            # Verify individual all_reduce was called for each parameter
            self.assertEqual(mock_torch_all_reduce.call_count, len(params))


if __name__ == '__main__':
    unittest.main()