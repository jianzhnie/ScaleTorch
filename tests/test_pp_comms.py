#!/usr/bin/env python3
"""
Tests for pipeline parallel communication (`pp_comms`).
"""

import unittest
from unittest.mock import MagicMock, patch

import torch

from scaletorch.parallel.pipeline_parallel import pp_comms as pc


class TestPipelineComms(unittest.TestCase):

    def setUp(self):
        # default pgm mock
        self.pgm_patcher = patch(
            'scaletorch.parallel.pipeline_parallel.pp_comms.pgm')
        self.mock_pgm = self.pgm_patcher.start()
        self.mock_pgm.process_group_manager = MagicMock()

        # patch dist
        self.dist_patcher = patch(
            'scaletorch.parallel.pipeline_parallel.pp_comms.dist')
        self.mock_dist = self.dist_patcher.start()

    def tearDown(self):
        self.pgm_patcher.stop()
        self.dist_patcher.stop()

    def test_validate_operation_raises(self):
        with self.assertRaises(pc.PipelineCommunicationError):
            pc._validate_operation('not_an_op', pc.VALID_OPERATIONS)

    def test_pipeline_communicate_recv_send_and_errors(self):
        mgr = self.mock_pgm.process_group_manager
        # recv_forward on first stage returns None
        mgr.pp_is_first_stage = True
        out = pc.pipeline_communicate('recv_forward',
                                      'cpu',
                                      torch.float32,
                                      shapes=(2, 3))
        self.assertIsNone(out)

        # send_forward on last stage returns None
        mgr.pp_is_first_stage = False
        mgr.pp_is_last_stage = True
        with self.assertRaises(pc.PipelineCommunicationError):
            pc.pipeline_communicate('recv_forward', 'cpu', torch.float32)

        mgr.pp_is_last_stage = False
        mgr.pp_next_rank = 5
        # missing tensor for send_forward raises
        with self.assertRaises(pc.PipelineCommunicationError):
            pc.pipeline_communicate('send_forward', 'cpu', torch.float32)

        # normal send_forward path executes P2POp and waits
        tensor = torch.randn(2, 3)
        mgr.pp_is_last_stage = False
        mgr.pp_next_rank = 7
        # stub P2POp and batch
        self.mock_dist.P2POp.return_value = MagicMock()
        req = MagicMock()
        self.mock_dist.batch_isend_irecv.return_value = [req]

        res = pc.pipeline_communicate('send_forward',
                                      'cpu',
                                      torch.float32,
                                      tensor=tensor)
        self.assertIsNone(res)
        req.wait.assert_called()

    def test_pipeline_communicate_recv_creates_tensor_and_waits(self):
        mgr = self.mock_pgm.process_group_manager
        mgr.pp_is_first_stage = False
        mgr.pp_prev_rank = 3
        mgr.pp_is_last_stage = False

        req = MagicMock()
        self.mock_dist.P2POp.return_value = MagicMock()
        self.mock_dist.batch_isend_irecv.return_value = [req]

        recv = pc.pipeline_communicate('recv_forward',
                                       'cpu',
                                       torch.float32,
                                       shapes=(2, 4))
        self.assertIsNotNone(recv)
        req.wait.assert_called()

    def test_bidirectional_pipeline_communicate(self):
        mgr = self.mock_pgm.process_group_manager
        mgr.pp_is_first_stage = False
        mgr.pp_is_last_stage = False
        mgr.pp_next_rank = 9
        mgr.pp_prev_rank = 8

        send_tensor = torch.randn(2, 3)
        req1 = MagicMock()
        req2 = MagicMock()
        self.mock_dist.P2POp.return_value = MagicMock()
        self.mock_dist.batch_isend_irecv.return_value = [req1, req2]

        recv = pc.bidirectional_pipeline_communicate('send_fwd_recv_bwd',
                                                     send_tensor, (2, 3),
                                                     'cpu', torch.float32)
        self.assertIsNotNone(recv)
        req1.wait.assert_called()
        req2.wait.assert_called()

    def test_get_and_reset_stats(self):
        pc.reset_communication_stats()
        stats = pc.get_communication_stats()
        self.assertIn('step', stats)


if __name__ == '__main__':
    unittest.main()
