#!/usr/bin/env python3
"""Tests for pipeline parallel communication (`pp_comms`)."""

import unittest
from unittest.mock import MagicMock, patch

import torch

from scaletorch.parallel.pipeline_parallel import pp_comms as pc


class TestPipelineComms(unittest.TestCase):
    def setUp(self):
        self.pgm_patcher = patch("scaletorch.parallel.pipeline_parallel.pp_comms.pgm")
        self.mock_pgm = self.pgm_patcher.start()
        self.mock_pgm.pp_is_first_stage = False
        self.mock_pgm.pp_is_last_stage = False
        self.mock_pgm.pp_rank = 0
        self.mock_pgm.pp_next_rank = None
        self.mock_pgm.pp_prev_rank = None
        self.mock_pgm.pp_group = MagicMock()

        self.torch_dist_patcher = patch(
            "scaletorch.parallel.pipeline_parallel.pp_comms.torch_dist"
        )
        self.mock_torch_dist = self.torch_dist_patcher.start()

        pc.reset_communication_stats()

    def tearDown(self):
        self.pgm_patcher.stop()
        self.torch_dist_patcher.stop()

    def test_validate_operation_raises(self):
        with self.assertRaises(pc.PipelineCommunicationError):
            pc._validate_operation("not_an_op", pc.VALID_OPERATIONS)

    def test_recv_forward_first_stage_returns_none(self):
        self.mock_pgm.pp_is_first_stage = True
        out = pc.pipeline_communicate(
            "recv_forward", "cpu", torch.float32, shapes=(2, 3)
        )
        self.assertIsNone(out)

    def test_send_forward_last_stage_returns_none(self):
        self.mock_pgm.pp_is_last_stage = True
        out = pc.pipeline_communicate(
            "send_forward", "cpu", torch.float32, tensor=torch.randn(2, 3)
        )
        self.assertIsNone(out)

    def test_recv_forward_missing_shapes_raises(self):
        self.mock_pgm.pp_is_first_stage = False
        self.mock_pgm.pp_prev_rank = 1
        with self.assertRaises(pc.PipelineCommunicationError):
            pc.pipeline_communicate("recv_forward", "cpu", torch.float32)

    def test_send_forward_missing_tensor_raises(self):
        self.mock_pgm.pp_is_last_stage = False
        self.mock_pgm.pp_next_rank = 5
        with self.assertRaises(pc.PipelineCommunicationError):
            pc.pipeline_communicate("send_forward", "cpu", torch.float32)

    def test_send_forward_calls_dist_send(self):
        self.mock_pgm.pp_is_last_stage = False
        self.mock_pgm.pp_next_rank = 7
        tensor = torch.randn(2, 3)
        pc.pipeline_communicate("send_forward", "cpu", torch.float32, tensor=tensor)
        self.mock_torch_dist.send.assert_called_once()

    def test_recv_forward_calls_dist_recv(self):
        self.mock_pgm.pp_is_first_stage = False
        self.mock_pgm.pp_prev_rank = 3
        recv = pc.pipeline_communicate(
            "recv_forward", "cpu", torch.float32, shapes=(2, 4)
        )
        self.assertIsNotNone(recv)
        self.mock_torch_dist.recv.assert_called_once()

    def test_bidirectional_send_fwd_recv_bwd(self):
        self.mock_pgm.pp_is_last_stage = False
        self.mock_pgm.pp_next_rank = 9
        send_req = MagicMock()
        self.mock_torch_dist.isend.return_value = send_req

        recv = pc.bidirectional_pipeline_communicate(
            "send_fwd_recv_bwd", torch.randn(2, 3), (2, 3), "cpu", torch.float32
        )
        self.assertIsNotNone(recv)
        self.mock_torch_dist.isend.assert_called_once()
        self.mock_torch_dist.recv.assert_called_once()
        send_req.wait.assert_called_once()

    def test_bidirectional_last_stage_returns_none(self):
        self.mock_pgm.pp_is_last_stage = True
        recv = pc.bidirectional_pipeline_communicate(
            "send_fwd_recv_bwd", torch.randn(2, 3), (2, 3), "cpu", torch.float32
        )
        self.assertIsNone(recv)

    def test_get_and_reset_stats(self):
        pc.reset_communication_stats()
        stats = pc.get_communication_stats()
        self.assertEqual(stats["step"], 0)
        self.assertIn("verbose", stats)


if __name__ == "__main__":
    unittest.main()
