"""Tests for scaletorch.utils.checkpoint — pytest style."""

import tempfile
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from scaletorch.utils.checkpoint import (
    CheckpointManager,
    InitializationManager,
    _handle_final_projection,
    init_model_with_dematerialized_weights,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model_and_optimizer():
    model = nn.Linear(4, 2, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    return model, optimizer


def _make_model_config(**overrides):
    defaults = dict(
        model_type="llama",
        hidden_size=64,
        num_attention_heads=8,
        num_key_value_heads=8,
        num_hidden_layers=4,
        intermediate_size=128,
        vocab_size=256,
        tie_word_embeddings=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# init_model_with_dematerialized_weights
# ---------------------------------------------------------------------------

class TestDematerializedWeights:
    def test_params_on_meta_device(self):
        with init_model_with_dematerialized_weights():
            model = nn.Linear(8, 4)

        for param in model.parameters():
            assert param.device == torch.device("meta")

    def test_buffers_unaffected_by_default(self):
        with init_model_with_dematerialized_weights():
            model = nn.BatchNorm1d(4)

        assert model.running_mean.device != torch.device("meta")

    def test_buffers_on_meta_when_included(self):
        with init_model_with_dematerialized_weights(include_buffers=True):
            model = nn.BatchNorm1d(4)

        assert model.running_mean.device == torch.device("meta")

    def test_context_manager_restores_registration(self):
        orig_register = nn.Module.register_parameter
        with init_model_with_dematerialized_weights():
            pass

        assert nn.Module.register_parameter is orig_register

    def test_nested_modules_all_on_meta(self):
        with init_model_with_dematerialized_weights():
            model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))

        for param in model.parameters():
            assert param.device == torch.device("meta")


# ---------------------------------------------------------------------------
# InitializationManager — layer name generation
# ---------------------------------------------------------------------------

class TestInitializationManagerLayerNames:
    @patch("scaletorch.utils.checkpoint.pgm", None)
    def test_llama_layer_names_non_pipeline(self):
        config = _make_model_config(num_hidden_layers=2)
        model = MagicMock(spec=nn.Module)
        mgr = InitializationManager(model, config)
        names = mgr.get_layer_names_in_sft_format()

        assert "model.embed_tokens.weight" in names
        assert "model.norm.weight" in names
        assert "lm_head.weight" in names
        assert any("model.layers.0." in n for n in names)
        assert any("model.layers.1." in n for n in names)
        assert not any("model.layers.2." in n for n in names)

    @patch("scaletorch.utils.checkpoint.pgm", None)
    def test_llama_decoder_components_present(self):
        config = _make_model_config(num_hidden_layers=1)
        model = MagicMock(spec=nn.Module)
        names = InitializationManager(model, config).get_layer_names_in_sft_format()

        expected_components = [
            "input_layernorm",
            "mlp.down_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "post_attention_layernorm",
            "self_attn.k_proj",
            "self_attn.o_proj",
            "self_attn.q_proj",
            "self_attn.v_proj",
        ]
        for comp in expected_components:
            assert any(comp in n for n in names), f"Missing component: {comp}"

    @patch("scaletorch.utils.checkpoint.pgm", None)
    def test_qwen3_adds_qk_norms(self):
        config = _make_model_config(model_type="qwen3", num_hidden_layers=1)
        model = MagicMock(spec=nn.Module)
        names = InitializationManager(model, config).get_layer_names_in_sft_format()

        assert any("self_attn.q_norm" in n for n in names)
        assert any("self_attn.k_norm" in n for n in names)

    @patch("scaletorch.utils.checkpoint.pgm", None)
    def test_tie_word_embeddings_excludes_lm_head(self):
        config = _make_model_config(tie_word_embeddings=True, num_hidden_layers=1)
        model = MagicMock(spec=nn.Module)
        names = InitializationManager(model, config).get_layer_names_in_sft_format()

        assert "lm_head.weight" not in names
        assert "model.embed_tokens.weight" in names

    @patch("scaletorch.utils.checkpoint.pgm", None)
    def test_qwen3_moe_expert_names(self):
        config = _make_model_config(
            model_type="qwen3_moe",
            num_hidden_layers=1,
            num_experts=4,
        )
        model = MagicMock(spec=nn.Module)
        names = InitializationManager(model, config).get_layer_names_in_sft_format()

        for eid in range(4):
            assert any(
                f"mlp.experts.{eid}.gate_proj" in n for n in names
            ), f"Missing expert {eid}"
        assert any("mlp.gate" in n for n in names)
        assert not any(
            "mlp.down_proj.weight" == n.split(".")[-2] + ".weight"
            and "experts" not in n
            for n in names
            if "mlp.down_proj" in n and "experts" not in n
        )


# ---------------------------------------------------------------------------
# InitializationManager — adjust_tensor_size
# ---------------------------------------------------------------------------

class TestAdjustTensorSize:
    @patch("scaletorch.utils.checkpoint.pgm", None)
    def test_no_tp_returns_same_tensor(self):
        config = _make_model_config()
        model = MagicMock(spec=nn.Module)
        mgr = InitializationManager(model, config)

        tensor = torch.randn(64, 64)
        result = mgr.adjust_tensor_size(tensor, "decoder_layers.0.mlp.gate_proj.weight")
        assert result.shape == tensor.shape

    def test_embedding_sharded_by_tp(self, mock_pgm):
        pgm_mock = mock_pgm(tp_size=2, rank=1)
        with patch("scaletorch.utils.checkpoint.pgm", pgm_mock):
            config = _make_model_config(vocab_size=256)
            model = MagicMock(spec=nn.Module)
            mgr = InitializationManager(model, config)

            tensor = torch.randn(256, 64)
            result = mgr.adjust_tensor_size(tensor, "embedding.weight")
            assert result.shape == (128, 64)

    def test_q_proj_sharded_by_tp(self, mock_pgm):
        pgm_mock = mock_pgm(tp_size=2, rank=0)
        with patch("scaletorch.utils.checkpoint.pgm", pgm_mock):
            config = _make_model_config(
                hidden_size=64, num_attention_heads=8, num_key_value_heads=8
            )
            model = MagicMock(spec=nn.Module)
            mgr = InitializationManager(model, config)

            tensor = torch.randn(64, 64)
            result = mgr.adjust_tensor_size(
                tensor, "decoder_layers.0.attention.q_proj.weight"
            )
            assert result.shape[0] == 32

    def test_mlp_gate_proj_sharded_by_tp(self, mock_pgm):
        pgm_mock = mock_pgm(tp_size=2, rank=0)
        with patch("scaletorch.utils.checkpoint.pgm", pgm_mock):
            config = _make_model_config(intermediate_size=128)
            model = MagicMock(spec=nn.Module)
            mgr = InitializationManager(model, config)

            tensor = torch.randn(128, 64)
            result = mgr.adjust_tensor_size(
                tensor, "decoder_layers.0.mlp.gate_proj.weight"
            )
            assert result.shape == (64, 64)

    def test_moe_expert_not_sharded_by_tp(self, mock_pgm):
        pgm_mock = mock_pgm(tp_size=2, rank=0)
        with patch("scaletorch.utils.checkpoint.pgm", pgm_mock):
            config = _make_model_config()
            model = MagicMock(spec=nn.Module)
            mgr = InitializationManager(model, config)

            tensor = torch.randn(128, 64)
            result = mgr.adjust_tensor_size(
                tensor, "decoder_layers.0.moe.experts.0.gate_proj.weight"
            )
            assert result.shape == tensor.shape


# ---------------------------------------------------------------------------
# InitializationManager — convert_safetensors_to_hf_name
# ---------------------------------------------------------------------------

class TestConvertSafetensorsName:
    @patch("scaletorch.utils.checkpoint.pgm", None)
    def test_basic_layer_renaming(self):
        config = _make_model_config()
        mgr = InitializationManager(MagicMock(spec=nn.Module), config)

        assert mgr.convert_safetensors_to_hf_name(
            "model.layers.0.self_attn.q_proj.weight"
        ) == "decoder_layers.0.attention.q_proj.weight"

    @patch("scaletorch.utils.checkpoint.pgm", None)
    def test_embedding_renaming(self):
        config = _make_model_config()
        mgr = InitializationManager(MagicMock(spec=nn.Module), config)

        assert mgr.convert_safetensors_to_hf_name(
            "model.embed_tokens.weight"
        ) == "embedding.weight"

    @patch("scaletorch.utils.checkpoint.pgm", None)
    def test_final_layers_renaming(self):
        config = _make_model_config()
        mgr = InitializationManager(MagicMock(spec=nn.Module), config)

        assert mgr.convert_safetensors_to_hf_name("lm_head.weight") == "final_proj.weight"
        assert mgr.convert_safetensors_to_hf_name("model.norm.weight") == "final_norm.weight"

    @patch("scaletorch.utils.checkpoint.pgm", None)
    def test_out_proj_renaming(self):
        config = _make_model_config()
        mgr = InitializationManager(MagicMock(spec=nn.Module), config)

        result = mgr.convert_safetensors_to_hf_name(
            "model.layers.0.self_attn.o_proj.weight"
        )
        assert "out_proj" in result


# ---------------------------------------------------------------------------
# _handle_final_projection
# ---------------------------------------------------------------------------

class TestHandleFinalProjection:
    @patch("scaletorch.utils.checkpoint.pgm", None)
    def test_tied_embeddings_copies_weight(self):
        model = MagicMock(spec=nn.Module)
        config = _make_model_config(tie_word_embeddings=True)
        state_dict = {"embedding.weight": torch.randn(256, 64)}

        _handle_final_projection(model, config, state_dict)

        assert "final_proj.weight" in state_dict
        assert torch.equal(
            state_dict["final_proj.weight"], state_dict["embedding.weight"]
        )

    @patch("scaletorch.utils.checkpoint.pgm", None)
    def test_no_tie_creates_zero_weight(self):
        model = MagicMock(spec=nn.Module)
        config = _make_model_config(tie_word_embeddings=False, vocab_size=256)
        state_dict = {"embedding.weight": torch.randn(256, 64)}

        _handle_final_projection(model, config, state_dict)

        assert "final_proj.weight" in state_dict
        assert state_dict["final_proj.weight"].shape == (256, 64)
        assert (state_dict["final_proj.weight"] == 0).all()

    @patch("scaletorch.utils.checkpoint.pgm", None)
    def test_already_present_skips(self):
        model = MagicMock(spec=nn.Module)
        config = _make_model_config()
        original = torch.randn(256, 64)
        state_dict = {"final_proj.weight": original}

        _handle_final_projection(model, config, state_dict)

        assert torch.equal(state_dict["final_proj.weight"], original)


# ---------------------------------------------------------------------------
# CheckpointManager
# ---------------------------------------------------------------------------

class TestCheckpointManagerPytest:
    @patch("scaletorch.utils.checkpoint.pgm", None)
    def test_checkpoint_path_format(self):
        mgr = CheckpointManager()
        path = mgr._get_checkpoint_path("/tmp/ckpt")
        assert "tp_rank_world_size=0_1" in str(path)
        assert "pp_rank_world_size=0_1" in str(path)
        assert str(path).endswith(".pth")

    @patch("scaletorch.utils.checkpoint.pgm", None)
    @patch("scaletorch.utils.checkpoint.st_dist")
    def test_save_and_load_round_trip(self, mock_dist):
        mock_dist.is_distributed.return_value = False
        tmp_dir = tempfile.mkdtemp()
        try:
            mgr = CheckpointManager()
            model, optimizer = _make_model_and_optimizer()

            out = model(torch.randn(2, 4))
            out.sum().backward()
            optimizer.step()
            original_weight = model.weight.data.clone()

            mgr.save_checkpoint(
                model, optimizer, trained_steps=10, trained_tokens=5000, out_dir=tmp_dir
            )
            ckpt_path = mgr._get_checkpoint_path(tmp_dir)
            assert ckpt_path.exists()

            model.weight.data.zero_()
            steps, tokens = mgr.load_checkpoint(model, optimizer, tmp_dir)
            assert steps == 10
            assert tokens == 5000
            assert torch.allclose(model.weight.data, original_weight)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    @patch("scaletorch.utils.checkpoint.pgm", None)
    @patch("scaletorch.utils.checkpoint.st_dist")
    def test_load_missing_checkpoint_raises(self, mock_dist):
        mock_dist.is_distributed.return_value = False
        mgr = CheckpointManager()
        model, optimizer = _make_model_and_optimizer()
        with pytest.raises(FileNotFoundError):
            mgr.load_checkpoint(model, optimizer, "/nonexistent/path")

    @patch("scaletorch.utils.checkpoint.pgm", None)
    @patch("scaletorch.utils.checkpoint.st_dist")
    def test_load_invalid_format_raises(self, mock_dist):
        mock_dist.is_distributed.return_value = False
        tmp_dir = tempfile.mkdtemp()
        try:
            mgr = CheckpointManager()
            ckpt_path = mgr._get_checkpoint_path(tmp_dir)
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": {}}, ckpt_path)

            model, optimizer = _make_model_and_optimizer()
            with pytest.raises(RuntimeError):
                mgr.load_checkpoint(model, optimizer, tmp_dir)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    @patch("scaletorch.utils.checkpoint.st_dist")
    def test_save_skipped_for_non_zero_dp_rank(self, mock_dist, mock_pgm):
        mock_dist.is_distributed.return_value = False
        pgm_mock = mock_pgm(dp_size=2, rank=1)
        with patch("scaletorch.utils.checkpoint.pgm", pgm_mock):
            tmp_dir = tempfile.mkdtemp()
            try:
                mgr = CheckpointManager()
                model, optimizer = _make_model_and_optimizer()
                mgr.save_checkpoint(
                    model, optimizer, trained_steps=1, trained_tokens=100, out_dir=tmp_dir
                )
                ckpt_path = mgr._get_checkpoint_path(tmp_dir)
                assert not ckpt_path.exists()
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    @patch("scaletorch.utils.checkpoint.st_dist")
    def test_save_skipped_for_non_zero_cp_rank(self, mock_dist, mock_pgm):
        mock_dist.is_distributed.return_value = False
        pgm_mock = mock_pgm(cp_size=2, rank=1)
        with patch("scaletorch.utils.checkpoint.pgm", pgm_mock):
            tmp_dir = tempfile.mkdtemp()
            try:
                mgr = CheckpointManager()
                model, optimizer = _make_model_and_optimizer()
                mgr.save_checkpoint(
                    model, optimizer, trained_steps=1, trained_tokens=100, out_dir=tmp_dir
                )
                ckpt_path = mgr._get_checkpoint_path(tmp_dir)
                assert not ckpt_path.exists()
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    @patch("scaletorch.utils.checkpoint.st_dist")
    def test_checkpoint_path_reflects_tp_pp(self, mock_dist, mock_pgm):
        mock_dist.is_distributed.return_value = False
        pgm_mock = mock_pgm(tp_size=4, pp_size=2, rank=5)
        with patch("scaletorch.utils.checkpoint.pgm", pgm_mock):
            mgr = CheckpointManager()
            path = mgr._get_checkpoint_path("/out")
            assert f"tp_rank_world_size={5 % 4}_{4}" in str(path)
            assert f"pp_rank_world_size={5 % 2}_{2}" in str(path)
