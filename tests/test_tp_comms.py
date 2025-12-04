#!/usr/bin/env python3
"""
Tests for tensor parallel communication utilities (`tp_comms`).
"""

import unittest
from unittest.mock import MagicMock, patch

import torch

from scaletorch.parallel.tensor_parallel import tp_comms as tc


class TestTensorParallelComms(unittest.TestCase):

    def test_merge_first_two_dims(self):
        go = torch.randn(2, 3, 4)
        inp = torch.randn(2, 3, 4)
        gof, inf = tc.merge_first_two_dims(go, inp)
        self.assertEqual(gof.shape[0], 6)
        self.assertEqual(inf.shape[0], 6)

    def test_split_tensor_along_last_dim(self):
        t = torch.randn(2, 4, 8)
        parts = tc.split_tensor_along_last_dim(t, 4)
        self.assertEqual(len(parts), 4)

        with self.assertRaises(ValueError):
            tc.split_tensor_along_last_dim(t, 3)

        with self.assertRaises(TypeError):
            tc.split_tensor_along_last_dim('not_tensor', 2)

    def test_copy_reduce_functions_when_tp_size_one(self):
        with patch('scaletorch.parallel.tensor_parallel.tp_comms.pgm'
                   ) as mock_pgm:
            mock_pgm.process_group_manager = MagicMock()
            mock_pgm.process_group_manager.tp_world_size = 1
            x = torch.randn(2, 3)
            # Copy forward/backward should just return tensor
            self.assertTrue(
                torch.equal(tc.CopyToModelParallelRegion.forward(None, x), x))
            self.assertTrue(
                torch.equal(tc.ReduceFromModelParallelRegion.forward(None, x),
                            x))
            self.assertTrue(
                torch.equal(tc.GatherFromModelParallelRegion.forward(None, x),
                            x))

    def test_copy_backward_calls_all_reduce_when_tp_size_gt1(self):
        with patch('scaletorch.parallel.tensor_parallel.tp_comms.dist') as mock_dist, \
             patch('scaletorch.parallel.tensor_parallel.tp_comms.pgm') as mock_pgm:
            mock_pgm.process_group_manager = MagicMock()
            mock_pgm.process_group_manager.tp_world_size = 2
            mock_pgm.process_group_manager.tp_group = MagicMock()

            grad = torch.randn(2, 3)
            tc.CopyToModelParallelRegion.backward(None, grad)
            mock_dist.all_reduce.assert_called()

    def test_reduce_forward_calls_all_reduce_when_tp_size_gt1(self):
        with patch('scaletorch.parallel.tensor_parallel.tp_comms.dist') as mock_dist, \
             patch('scaletorch.parallel.tensor_parallel.tp_comms.pgm') as mock_pgm:
            mock_pgm.process_group_manager = MagicMock()
            mock_pgm.process_group_manager.tp_world_size = 2
            mock_pgm.process_group_manager.tp_group = MagicMock()

            x = torch.randn(2, 3)
            tc.ReduceFromModelParallelRegion.forward(None, x)
            mock_dist.all_reduce.assert_called()

    def test_gather_forward_backward_split(self):
        with patch('scaletorch.parallel.tensor_parallel.tp_comms.dist') as mock_dist, \
             patch('scaletorch.parallel.tensor_parallel.tp_comms.pgm') as mock_pgm:
            mgr = MagicMock()
            mgr.tp_world_size = 2
            mgr.tp_rank = 0
            mgr.tp_group = MagicMock()
            mock_pgm.process_group_manager = mgr

            x = torch.randn(2, 3, 4)

            # simulate all_gather side effect to populate tensor_list
            def fake_all_gather(tensor_list, src, group=None):
                # fill other entries with doubled src
                for i in range(len(tensor_list)):
                    if tensor_list[i] is None or i != mgr.tp_rank:
                        tensor_list[i] = src * (i + 1)

            mock_dist.all_gather.side_effect = fake_all_gather

            out = tc.GatherFromModelParallelRegion.forward(None, x)
            # concatenation along last dim
            self.assertEqual(out.size(-1), x.size(-1) * mgr.tp_world_size)

            # backward should split
            grad_out = torch.randn(2, 3, 8)
            with patch(
                    'scaletorch.parallel.tensor_parallel.tp_comms.split_tensor_along_last_dim'
            ) as mock_split:
                mock_split.return_value = [
                    grad_out[..., :4], grad_out[..., 4:]
                ]
                chunk = tc.GatherFromModelParallelRegion.backward(
                    None, grad_out)
                self.assertIsNotNone(chunk)

    def test_linear_with_async_all_reduce_backward_shapes(self):
        with patch('scaletorch.parallel.tensor_parallel.tp_comms.pgm'
                   ) as mock_pgm:
            mock_pgm.process_group_manager = MagicMock()
            mock_pgm.process_group_manager.tp_world_size = 1

            b, s, in_size, out_size = 2, 3, 4, 5
            x = torch.randn(b, s, in_size, requires_grad=True)
            weight = torch.randn(out_size, in_size, requires_grad=True)
            bias = torch.randn(out_size, requires_grad=True)

            # Create mock context with proper save_for_backward
            mock_ctx = MagicMock()
            saved_tensors = []

            def mock_save_for_backward(*args):
                saved_tensors.extend(args)

            mock_ctx.save_for_backward = mock_save_for_backward
            mock_ctx.saved_tensors = tuple(saved_tensors)
            mock_ctx.use_bias = True

            out = tc.LinearWithAsyncAllReduce.forward(mock_ctx, x, weight,
                                                      bias)
            self.assertEqual(out.shape, (b, s, out_size))

            # Update saved_tensors reference after forward
            mock_ctx.saved_tensors = tuple(saved_tensors)

            # run backward by calling backward helper directly
            grad_out = torch.randn(b, s, out_size)
            gi, gw, gb = tc.LinearWithAsyncAllReduce.backward(
                mock_ctx, grad_out)
            self.assertEqual(gi.shape, x.shape)
            self.assertEqual(gw.shape, weight.shape)
            self.assertEqual(gb.shape, bias.shape)


if __name__ == '__main__':
    unittest.main()
