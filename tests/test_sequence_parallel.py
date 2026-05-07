"""Tests for scaletorch.parallel.sequence_parallel module."""

import os
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from scaletorch.parallel.sequence_parallel.sp_comms import (
    AllGatherFromSequenceParallelRegion, ReduceScatterToSequenceParallelRegion,
    _gather_along_seq_dim, _split_along_seq_dim)


def _make_mock_pgm(tp_world_size=2, tp_rank=0):
    """Create a mock process group manager for testing."""
    mock = MagicMock()
    mock.tp_world_size = tp_world_size
    mock.tp_rank = tp_rank
    mock.tp_group = MagicMock()
    return mock


class TestSplitGatherHelpers(unittest.TestCase):

    def test_split_and_gather_roundtrip(self):
        x = torch.randn(2, 8, 16)
        chunks = _split_along_seq_dim(x, 4)
        self.assertEqual(len(chunks), 4)
        for c in chunks:
            self.assertEqual(c.shape, (2, 2, 16))
        recovered = _gather_along_seq_dim(chunks)
        self.assertTrue(torch.equal(x, recovered))

    def test_split_non_divisible_raises(self):
        x = torch.randn(2, 7, 16)
        with self.assertRaises(ValueError):
            _split_along_seq_dim(x, 2)

    def test_split_world_size_1(self):
        x = torch.randn(2, 8, 16)
        chunks = _split_along_seq_dim(x, 1)
        self.assertEqual(len(chunks), 1)
        self.assertTrue(torch.equal(chunks[0], x))


class TestAllGatherFromSPRegion(unittest.TestCase):

    @patch('scaletorch.parallel.sequence_parallel.sp_comms.pgm', None)
    def test_no_pgm_returns_input(self):
        x = torch.randn(2, 4, 16)
        out = AllGatherFromSequenceParallelRegion.apply(x)
        self.assertTrue(torch.equal(x, out))

    @patch('scaletorch.parallel.sequence_parallel.sp_comms.pgm')
    def test_tp_world_size_1_returns_input(self, mock_pgm):
        mock_pgm.tp_world_size = 1
        x = torch.randn(2, 4, 16)
        out = AllGatherFromSequenceParallelRegion.apply(x)
        self.assertTrue(torch.equal(x, out))

    @patch('scaletorch.parallel.sequence_parallel.sp_comms.st_dist')
    @patch('scaletorch.parallel.sequence_parallel.sp_comms.pgm')
    def test_forward_all_gathers_along_seq(self, mock_pgm, mock_dist):
        mock_pgm.tp_world_size = 2
        mock_pgm.tp_group = MagicMock()

        rank0_chunk = torch.randn(2, 4, 16)
        rank1_chunk = torch.randn(2, 4, 16)
        mock_dist.all_gather.return_value = [rank0_chunk, rank1_chunk]

        x = torch.randn(2, 4, 16)
        out = AllGatherFromSequenceParallelRegion.apply(x)
        self.assertEqual(out.shape, (2, 8, 16))
        mock_dist.all_gather.assert_called_once_with(x, group=mock_pgm.tp_group)

    @patch('scaletorch.parallel.sequence_parallel.sp_comms.st_dist')
    @patch('scaletorch.parallel.sequence_parallel.sp_comms.pgm')
    def test_backward_reduce_scatters(self, mock_pgm, mock_dist):
        mock_pgm.tp_world_size = 2
        mock_pgm.tp_group = MagicMock()

        # forward: all_gather returns 2 chunks
        chunk0 = torch.randn(2, 4, 16, requires_grad=True)
        chunk1 = torch.randn(2, 4, 16, requires_grad=True)
        mock_dist.all_gather.return_value = [chunk0, chunk1]

        x = torch.randn(2, 4, 16, requires_grad=True)
        out = AllGatherFromSequenceParallelRegion.apply(x)

        # Simulate backward by calling the backward manually
        grad_output = torch.randn(2, 8, 16)
        # Use autograd to trigger backward
        loss = out.sum()
        loss.backward()

        # reduce_scatter should have been called in backward
        mock_dist.reduce_scatter.assert_called_once()


class TestReduceScatterToSPRegion(unittest.TestCase):

    @patch('scaletorch.parallel.sequence_parallel.sp_comms.pgm', None)
    def test_no_pgm_returns_input(self):
        x = torch.randn(2, 8, 16)
        out = ReduceScatterToSequenceParallelRegion.apply(x)
        self.assertTrue(torch.equal(x, out))

    @patch('scaletorch.parallel.sequence_parallel.sp_comms.pgm')
    def test_tp_world_size_1_returns_input(self, mock_pgm):
        mock_pgm.tp_world_size = 1
        x = torch.randn(2, 8, 16)
        out = ReduceScatterToSequenceParallelRegion.apply(x)
        self.assertTrue(torch.equal(x, out))

    @patch('scaletorch.parallel.sequence_parallel.sp_comms.st_dist')
    @patch('scaletorch.parallel.sequence_parallel.sp_comms.pgm')
    def test_forward_reduce_scatters_along_seq(self, mock_pgm, mock_dist):
        mock_pgm.tp_world_size = 2
        mock_pgm.tp_group = MagicMock()

        x = torch.randn(2, 8, 16)
        out = ReduceScatterToSequenceParallelRegion.apply(x)
        # output shape should be (2, 4, 16) — seq dim halved
        self.assertEqual(out.shape, (2, 4, 16))
        mock_dist.reduce_scatter.assert_called_once()


class TestSPModeDisabled(unittest.TestCase):
    """Test that SP mode is off by default (no SEQUENCE_PARALLEL env var)."""

    def test_decoder_layer_no_sp_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            # Remove SEQUENCE_PARALLEL if set
            os.environ.pop('SEQUENCE_PARALLEL', None)
            from scaletorch.models.llama import DecoderLayer
            config = MagicMock()
            config.hidden_size = 64
            config.num_attention_heads = 4
            config.num_key_value_heads = 2
            config.intermediate_size = 128
            config.rms_norm_eps = 1e-6
            config.max_position_embeddings = 32
            config.rope_theta = 10000.0

            with patch('scaletorch.models.llama.pgm', None):
                layer = DecoderLayer(config, layer_idx=0)
                self.assertFalse(layer._use_sp)


class TestLlamaWithSPEnvFlag(unittest.TestCase):
    """Test Llama model with SEQUENCE_PARALLEL=1 but pgm=None (should be disabled)."""

    @patch.dict(os.environ, {'FLASH_ATTEN': '', 'SEQUENCE_PARALLEL': '1'})
    def test_sp_disabled_when_pgm_none(self):
        from scaletorch.models.llama import Llama

        config = MagicMock()
        config.vocab_size = 100
        config.hidden_size = 64
        config.intermediate_size = 128
        config.num_hidden_layers = 2
        config.num_attention_heads = 4
        config.num_key_value_heads = 2
        config.max_position_embeddings = 32
        config.rms_norm_eps = 1e-6
        config.rope_theta = 10000.0

        with patch('scaletorch.models.llama.pgm', None):
            model = Llama(config)
            # SP should be disabled since pgm is None
            self.assertFalse(model._use_sp)
            for layer in model.decoder_layers:
                self.assertFalse(layer._use_sp)

            # Forward should still work normally
            input_ids = torch.randint(0, 100, (2, 8))
            with torch.no_grad():
                out = model(input_ids)
            self.assertEqual(out.shape, (2, 8, 100))


if __name__ == '__main__':
    unittest.main()
