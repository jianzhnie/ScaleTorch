"""Tests for scaletorch.model.model_llama module."""
import os
import unittest
from unittest.mock import MagicMock, patch

import torch

os.environ['FLASH_ATTEN'] = '0'
os.environ['CONTEXT_PARALLEL'] = '0'

from scaletorch.models.model_llama import (
    Llama,
    apply_rotary_pos_emb,
    flash_attention,
    get_cos_sin,
)


class TestGetCosSin(unittest.TestCase):
    """Test RoPE cos/sin generation."""

    def test_basic_shape(self):
        seq_len, head_dim = 128, 64
        cos, sin = get_cos_sin(seq_len, head_dim, device=torch.device('cpu'))
        self.assertEqual(cos.shape, (seq_len, head_dim))
        self.assertEqual(sin.shape, (seq_len, head_dim))

    def test_dtype_bfloat16(self):
        cos, sin = get_cos_sin(32, 64, device=torch.device('cpu'), dtype=torch.bfloat16)
        self.assertEqual(cos.dtype, torch.bfloat16)

    def test_dtype_float32(self):
        cos, sin = get_cos_sin(32, 64, device=torch.device('cpu'), dtype=torch.float32)
        self.assertEqual(cos.dtype, torch.float32)

    def test_odd_head_dim_raises(self):
        with self.assertRaises(AssertionError):
            get_cos_sin(32, 63, device=torch.device('cpu'))

    def test_values_bounded(self):
        cos, sin = get_cos_sin(64, 32, device=torch.device('cpu'), dtype=torch.float32)
        self.assertTrue(cos.abs().max() <= 1.0)
        self.assertTrue(sin.abs().max() <= 1.0)


class TestApplyRotaryPosEmb(unittest.TestCase):
    """Test rotary position embedding application."""

    def test_output_shape(self):
        batch, heads, seq_len, head_dim = 2, 8, 16, 64
        x = torch.randn(batch, heads, seq_len, head_dim)
        cos, sin = get_cos_sin(seq_len, head_dim, device=torch.device('cpu'), dtype=torch.float32)

        out = apply_rotary_pos_emb(x, cos[:seq_len], sin[:seq_len])
        self.assertEqual(out.shape, x.shape)

    def test_zero_input(self):
        batch, heads, seq_len, head_dim = 1, 4, 8, 32
        x = torch.zeros(batch, heads, seq_len, head_dim)
        cos, sin = get_cos_sin(seq_len, head_dim, device=torch.device('cpu'), dtype=torch.float32)

        out = apply_rotary_pos_emb(x, cos[:seq_len], sin[:seq_len])
        self.assertTrue(torch.allclose(out, torch.zeros_like(out)))

    def test_cos_sin_broadcasting(self):
        batch, heads, seq_len, head_dim = 2, 4, 8, 32
        x = torch.randn(batch, heads, seq_len, head_dim)
        cos = torch.ones(seq_len, head_dim)
        sin = torch.zeros(seq_len, head_dim)

        out = apply_rotary_pos_emb(x, cos, sin)
        self.assertTrue(torch.allclose(out, x, atol=1e-5))


class TestFlashAttention(unittest.TestCase):
    """Test flash attention function."""

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "SDPA may require GPU/NPU device in this environment"
    )
    def test_output_shape(self):
        batch, heads, seq_len, head_dim = 2, 8, 16, 64
        q = torch.randn(batch, heads, seq_len, head_dim)
        k = torch.randn(batch, heads, seq_len, head_dim)
        v = torch.randn(batch, heads, seq_len, head_dim)

        out = flash_attention(q, k, v, causal=True)
        self.assertEqual(out.shape, (batch, seq_len, heads, head_dim))

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "SDPA may require GPU/NPU device in this environment"
    )
    def test_non_causal(self):
        batch, heads, seq_len, head_dim = 1, 4, 8, 32
        q = torch.randn(batch, heads, seq_len, head_dim)
        k = torch.randn(batch, heads, seq_len, head_dim)
        v = torch.randn(batch, heads, seq_len, head_dim)

        out = flash_attention(q, k, v, causal=False)
        self.assertEqual(out.shape, (batch, seq_len, heads, head_dim))


class TestLlamaModel(unittest.TestCase):
    """Test Llama model."""

    def _make_config(self, num_layers=2, hidden_size=64, num_heads=4,
                     num_kv_heads=2, intermediate_size=128, vocab_size=100,
                     max_pos=256):
        config = MagicMock()
        config.hidden_size = hidden_size
        config.num_attention_heads = num_heads
        config.num_key_value_heads = num_kv_heads
        config.intermediate_size = intermediate_size
        config.vocab_size = vocab_size
        config.num_hidden_layers = num_layers
        config.max_position_embeddings = max_pos
        config.rms_norm_eps = 1e-5
        config.rope_theta = 10000.0
        return config

    def test_init(self):
        config = self._make_config()
        model = Llama(config)
        self.assertEqual(len(model.decoder_layers), 2)
        self.assertEqual(model.vocab_size, 100)
        self.assertEqual(model.hidden_size, 64)

    def test_forward_shape(self):
        config = self._make_config()
        model = Llama(config)
        model.eval()

        input_ids = torch.randint(0, 100, (2, 16))
        with torch.no_grad():
            out = model(input_ids)

        self.assertEqual(out.shape, (2, 16, 100))

    def test_forward_gradient_checkpointing(self):
        config = self._make_config()
        model = Llama(config)
        model.train()

        input_ids = torch.randint(0, 100, (2, 16))
        out = model(input_ids, gradient_checkpointing=True)
        self.assertEqual(out.shape, (2, 16, 100))

        loss = out.sum()
        loss.backward()

        has_grad = any(p.grad is not None for p in model.parameters())
        self.assertTrue(has_grad)

    def test_config_stored(self):
        config = self._make_config()
        model = Llama(config)
        self.assertIs(model.config, config)
        self.assertIs(model.model_config, config)

    def test_invalid_config(self):
        config = self._make_config(hidden_size=65, num_heads=4)
        with self.assertRaises(AssertionError):
            Llama(config)


if __name__ == '__main__':
    unittest.main()
