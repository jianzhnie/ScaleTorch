#!/usr/bin/env python3
"""
Tests for pipeline_parallel.PipelineParallel and training step validation.
"""

import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from scaletorch.parallel.pipeline_parallel.pipeline_parallel import (
    PipelineParallel, train_step_pipeline_1f1b, train_step_pipeline_afab)


class DummySubLayer(nn.Module):

    def __init__(self):
        super().__init__()

        # create nested modules with reset_parameters
        class _M:

            def reset_parameters(self):
                return

        self.input_layernorm = _M()
        self.attention = _M()
        self.post_attention_layernorm = _M()
        self.mlp = _M()

    def forward(self, x, position_ids=None):
        return x


class DummyModel:

    def __init__(self, num_layers):
        self.decoder_layers = [DummySubLayer() for _ in range(num_layers)]

        # Add embedding, final_norm, final_proj for all stages
        class _M:

            def reset_parameters(self):
                return

        self.embedding = _M()
        self.final_norm = _M()
        self.final_proj = _M()


class DummyConfig:

    def __init__(self, num_hidden_layers):
        self.num_hidden_layers = num_hidden_layers


class TestPipelineParallel(unittest.TestCase):

    def setUp(self):
        self.pgm_patcher = patch(
            'scaletorch.parallel.pipeline_parallel.pipeline_parallel.pgm')
        self.mock_pgm = self.pgm_patcher.start()
        # default: 2 stages
        self.mock_pgm.pp_world_size = 2

    def tearDown(self):
        self.pgm_patcher.stop()

    def test_distribute_layers_even_uneven(self):
        cfg = DummyConfig(num_hidden_layers=5)
        # stage 0
        self.mock_pgm.pp_rank = 0
        pl0 = PipelineParallel(DummyModel(5), cfg)
        # stage 1
        self.mock_pgm.pp_rank = 1
        pl1 = PipelineParallel(DummyModel(5), cfg)

        # total layers assigned between stages should partition 5 layers
        self.assertEqual(
            len(pl0.layer_distribution) + len(pl1.layer_distribution), 5)

    def test_constructor_missing_num_layers(self):
        with self.assertRaises(AttributeError):
            PipelineParallel(DummyModel(2), object())

    def test_forward_requires_hidden_on_non_first(self):
        cfg = DummyConfig(num_hidden_layers=2)
        self.mock_pgm.pp_rank = 1
        self.mock_pgm.pp_world_size = 2
        self.mock_pgm.pp_is_first_stage = False
        pp = PipelineParallel(DummyModel(2), cfg)

        with self.assertRaises(ValueError):
            pp.forward(input_ids=torch.tensor([1]),
                       position_ids=torch.tensor([1]))

    def test_backward_requires_grad_for_non_last(self):
        cfg = DummyConfig(num_hidden_layers=2)
        self.mock_pgm.pp_rank = 0
        self.mock_pgm.pp_world_size = 2
        self.mock_pgm.pp_is_last_stage = False
        pp = PipelineParallel(DummyModel(2), cfg)

        input_tensor = torch.randn(2, 3, requires_grad=True)
        output_tensor = torch.randn(2, 3)
        with self.assertRaises(ValueError):
            pp.backward(input_tensor, output_tensor, None)

    def test_train_step_validation_errors(self):
        # missing gradient_accumulation_steps
        with self.assertRaises(ValueError):
            train_step_pipeline_afab(None, object(), (2, 3), 'cpu',
                                     torch.float32)

        class DL:
            gradient_accumulation_steps = 0

        with self.assertRaises(ValueError):
            train_step_pipeline_1f1b(None, DL(), (2, 3), 'cpu', torch.float32)


if __name__ == '__main__':
    unittest.main()
