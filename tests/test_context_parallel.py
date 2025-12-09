#!/usr/bin/env python3
"""
Tests for scaletorch.parallel.context_parallel module.
"""

import os
import unittest
from unittest.mock import patch

import torch

from scaletorch.parallel.context_parallel.context_parallel import (
    _validate_attention_inputs, apply_context_parallel,
    ring_attention_backward, ring_attention_forward, update_out_and_lse,
    update_rope_for_context_parallel)


class TestContextParallelHelpers(unittest.TestCase):

    def test_apply_context_parallel_sets_env_var(self):
        with patch('scaletorch.parallel.context_parallel.context_parallel.pgm'
                   ) as mock_pgm:
            mock_pgm.cp_world_size = 4

            model = object()
            apply_context_parallel(model)

            self.assertEqual(os.environ.get('CONTEXT_PARALLEL'), '1')

        with patch('scaletorch.parallel.context_parallel.context_parallel.pgm'
                   ) as mock_pgm:
            mock_pgm.cp_world_size = 1

            model = object()
            apply_context_parallel(model)

            self.assertEqual(os.environ.get('CONTEXT_PARALLEL'), '0')

    def test_validate_attention_inputs_errors(self):
        q = torch.randn(1, 1, 4, 16)
        k = torch.randn(1, 1, 4, 16)
        v = torch.randn(1, 1, 4, 16)

        # dtype mismatch
        with self.assertRaises(ValueError):
            _validate_attention_inputs(q, k, v.double(), 1.0)

        # shape mismatch
        with self.assertRaises(ValueError):
            _validate_attention_inputs(q, k[:, :, :3, :], v, 1.0)

        # not 4D
        with self.assertRaises(ValueError):
            _validate_attention_inputs(q.squeeze(0).squeeze(0), k, v, 1.0)

        # non-positive scale
        with self.assertRaises(ValueError):
            _validate_attention_inputs(q, k, v, 0.0)

        # device mismatch
        if torch.cuda.is_available():
            cpu_t = q.cpu()
            gpu_t = q.cuda()
            with self.assertRaises(ValueError):
                _validate_attention_inputs(cpu_t, gpu_t, gpu_t, 1.0)

    def test_ring_attention_forward_and_backward_shapes(self):
        batch, heads, seq, dim = 2, 2, 4, 8
        q = torch.randn(batch, heads, seq, dim, dtype=torch.float32)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        out, lse = ring_attention_forward(q,
                                          k,
                                          v,
                                          sm_scale=1.0,
                                          is_causal=False)
        self.assertEqual(out.shape, q.shape)
        self.assertEqual(lse.shape, torch.Size([batch, heads, seq]))

        # Backward: dO same shape as out
        dO = torch.randn_like(out)
        dq, dk, dv = ring_attention_backward(dO, q, k, v, out, lse, 1.0, False)
        self.assertEqual(dq.shape, q.shape)
        self.assertEqual(dk.shape, k.shape)
        self.assertEqual(dv.shape, v.shape)

    def test_update_out_and_lse(self):
        batch, heads, seq, dim = 1, 1, 4, 8
        block_out = torch.randn(batch, heads, seq, dim)
        block_lse = torch.randn(batch, heads, seq)

        # first update returns block values
        out, lse = update_out_and_lse(None, None, block_out, block_lse)
        self.assertTrue(torch.allclose(out, block_out.to(torch.float32)))

        # second update with slice
        old_out = out.clone()
        old_lse = lse.clone()
        out2, lse2 = update_out_and_lse(old_out, old_lse, block_out, block_lse)
        self.assertEqual(out2.shape, out.shape)

        # slice on first update should raise - use built-in slice
        with self.assertRaises(RuntimeError):
            update_out_and_lse(None, None, block_out, block_lse, slice(0, 1))

    def test_update_rope_for_context_parallel(self):
        seq_len = 8
        cos = torch.randn(seq_len, 16)
        sin = torch.randn(seq_len, 16)

        with patch('scaletorch.parallel.context_parallel.context_parallel.pgm'
                   ) as mock_pgm:
            mock_pgm.cp_world_size = 2
            mock_pgm.cp_rank = 1

            cos_p, sin_p = update_rope_for_context_parallel(cos, sin)
            self.assertEqual(cos_p.size(0), seq_len // 2)

        # invalid divisibility
        with patch('scaletorch.parallel.context_parallel.context_parallel.pgm'
                   ) as mock_pgm:
            mock_pgm.cp_world_size = 3
            mock_pgm.cp_rank = 0

            with self.assertRaises(ValueError):
                update_rope_for_context_parallel(cos, sin)


if __name__ == '__main__':
    unittest.main()
