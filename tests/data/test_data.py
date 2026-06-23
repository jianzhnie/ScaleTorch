"""Tests for scaletorch.data — dataset processing, tokenization, and dataloader."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from scaletorch.data.dataset import (
    DatasetProcessor,
    _tokenize_and_chunk,
    get_tokenize_strategy,
    register_tokenize_strategy,
)


# ---------------------------------------------------------------------------
# Tokenize-strategy registry
# ---------------------------------------------------------------------------

class TestTokenizeStrategyRegistry:
    def test_builtin_concat_chunk_registered(self):
        strategy = get_tokenize_strategy("concat_chunk")
        assert callable(strategy)
        assert strategy is _tokenize_and_chunk

    def test_unknown_strategy_raises(self):
        with pytest.raises(KeyError, match="Unknown tokenize strategy"):
            get_tokenize_strategy("nonexistent_strategy")

    def test_register_custom_strategy(self):
        @register_tokenize_strategy("_test_custom")
        def _dummy(examples, tokenizer, sequence_length):
            return {"input_ids": []}

        assert get_tokenize_strategy("_test_custom") is _dummy

    def test_register_overwrites(self):
        @register_tokenize_strategy("_test_overwrite")
        def _v1(examples, tokenizer, sequence_length):
            return {"input_ids": [[1]]}

        @register_tokenize_strategy("_test_overwrite")
        def _v2(examples, tokenizer, sequence_length):
            return {"input_ids": [[2]]}

        assert get_tokenize_strategy("_test_overwrite") is _v2


# ---------------------------------------------------------------------------
# _tokenize_and_chunk
# ---------------------------------------------------------------------------

class TestTokenizeAndChunk:
    def _make_tokenizer(self, token_ids_per_text):
        mock_tok = MagicMock()
        mock_tok.side_effect = lambda texts, **kw: {
            "input_ids": [token_ids_per_text] * len(texts)
        }
        return mock_tok

    def test_basic_chunking(self):
        tokenizer = MagicMock()
        tokenizer.return_value = {"input_ids": [list(range(11))]}
        result = _tokenize_and_chunk(["hello world"], tokenizer, sequence_length=5)

        assert "input_ids" in result
        for chunk in result["input_ids"]:
            assert len(chunk) == 6

    def test_short_text_returns_empty(self):
        tokenizer = MagicMock()
        tokenizer.return_value = {"input_ids": [list(range(3))]}
        result = _tokenize_and_chunk(["hi"], tokenizer, sequence_length=10)

        assert result == {"input_ids": []}

    def test_exact_length_produces_one_chunk(self):
        tokenizer = MagicMock()
        tokenizer.return_value = {"input_ids": [list(range(6))]}
        result = _tokenize_and_chunk(["text"], tokenizer, sequence_length=5)

        assert len(result["input_ids"]) == 1
        assert len(result["input_ids"][0]) == 6

    def test_multiple_texts_concatenated(self):
        tokenizer = MagicMock()
        tokenizer.return_value = {
            "input_ids": [list(range(6)), list(range(6, 12))]
        }
        result = _tokenize_and_chunk(["a", "b"], tokenizer, sequence_length=5)

        assert len(result["input_ids"]) == 2
        for chunk in result["input_ids"]:
            assert len(chunk) == 6


# ---------------------------------------------------------------------------
# DatasetProcessor — init validation
# ---------------------------------------------------------------------------

class TestDatasetProcessorValidation:
    def test_empty_tokenizer_name_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            DatasetProcessor("", device="cpu")

    def test_none_tokenizer_name_raises(self):
        with pytest.raises(ValueError):
            DatasetProcessor(None, device="cpu")

    def test_non_string_tokenizer_name_raises(self):
        with pytest.raises(ValueError):
            DatasetProcessor(123, device="cpu")


# ---------------------------------------------------------------------------
# DatasetProcessor — tokenizer_group_text
# ---------------------------------------------------------------------------

class TestTokenizerGroupText:
    def _make_processor(self):
        with patch.object(DatasetProcessor, "__init__", lambda self, *a, **kw: None):
            proc = DatasetProcessor.__new__(DatasetProcessor)
            proc.pgm = None

            mock_tok = MagicMock()
            mock_tok.return_value = {"input_ids": [list(range(20))]}
            proc.tokenizer = mock_tok
            return proc

    def test_basic_group_text(self):
        proc = self._make_processor()
        result = proc.tokenizer_group_text(["hello world"], sequence_length=5)

        assert "input_ids" in result
        for chunk in result["input_ids"]:
            assert len(chunk) == 6

    def test_zero_sequence_length_raises(self):
        proc = self._make_processor()
        with pytest.raises(ValueError, match="sequence_length must be positive"):
            proc.tokenizer_group_text(["hello"], sequence_length=0)

    def test_negative_sequence_length_raises(self):
        proc = self._make_processor()
        with pytest.raises(ValueError, match="sequence_length must be positive"):
            proc.tokenizer_group_text(["hello"], sequence_length=-1)

    def test_empty_examples_raises(self):
        proc = self._make_processor()
        with pytest.raises(ValueError, match="cannot be empty"):
            proc.tokenizer_group_text([], sequence_length=10)

    def test_short_text_returns_empty(self):
        proc = self._make_processor()
        proc.tokenizer.return_value = {"input_ids": [list(range(3))]}
        result = proc.tokenizer_group_text(["hi"], sequence_length=10)
        assert result == {"input_ids": []}


# ---------------------------------------------------------------------------
# DatasetProcessor — tokenize_dataset
# ---------------------------------------------------------------------------

class TestTokenizeDataset:
    def _make_processor_with_tokenizer(self):
        with patch.object(DatasetProcessor, "__init__", lambda self, *a, **kw: None):
            proc = DatasetProcessor.__new__(DatasetProcessor)
            proc.pgm = None
            proc.tokenizer = MagicMock()
            return proc

    def test_none_tokenizer_raises(self):
        with patch.object(DatasetProcessor, "__init__", lambda self, *a, **kw: None):
            proc = DatasetProcessor.__new__(DatasetProcessor)
            proc.pgm = None
            proc.tokenizer = None

            mock_dataset = MagicMock()
            mock_dataset.column_names = ["text"]

            with pytest.raises(AttributeError, match="Tokenizer not initialized"):
                proc.tokenize_dataset(
                    mock_dataset, "text", sequence_length=32, num_proc=1
                )

    def test_invalid_sequence_length_raises(self):
        proc = self._make_processor_with_tokenizer()
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["text"]

        with pytest.raises(ValueError, match="sequence_length must be positive"):
            proc.tokenize_dataset(mock_dataset, "text", sequence_length=0, num_proc=1)

    def test_invalid_num_proc_raises(self):
        proc = self._make_processor_with_tokenizer()
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["text"]

        with pytest.raises(ValueError, match="num_proc must be at least 1"):
            proc.tokenize_dataset(mock_dataset, "text", sequence_length=32, num_proc=0)

    def test_missing_column_raises(self):
        proc = self._make_processor_with_tokenizer()
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["content"]

        with pytest.raises(ValueError, match="Column 'text' not found"):
            proc.tokenize_dataset(
                mock_dataset, "text", sequence_length=32, num_proc=1
            )


# ---------------------------------------------------------------------------
# DatasetProcessor — load_dataset
# ---------------------------------------------------------------------------

class TestDatasetProcessorLoadDataset:
    def _make_processor(self):
        with patch.object(DatasetProcessor, "__init__", lambda self, *a, **kw: None):
            proc = DatasetProcessor.__new__(DatasetProcessor)
            proc.pgm = None
            proc.tokenizer = MagicMock()
            return proc

    def test_non_positive_num_samples_raises(self):
        proc = self._make_processor()
        with pytest.raises(ValueError, match="num_samples must be positive"):
            proc.load_dataset("test", split="train", subset_name=None, num_samples=0)

    def test_negative_num_samples_raises(self):
        proc = self._make_processor()
        with pytest.raises(ValueError, match="num_samples must be positive"):
            proc.load_dataset("test", split="train", subset_name=None, num_samples=-5)


# ---------------------------------------------------------------------------
# MicroBatchDataLoader (migrated from unittest to pytest style)
# ---------------------------------------------------------------------------

from scaletorch.data.dataloader import MicroBatchDataLoader


class TestMicroBatchDataLoaderValidation:
    def test_invalid_micro_batch_size(self):
        with pytest.raises(ValueError):
            MicroBatchDataLoader(
                micro_batch_size=0, sequence_length=32, dataset_name="t",
                tokenizer_name="t", num_workers=0, num_proc=1,
                gradient_accumulation_steps=1, device="cpu",
            )

    def test_invalid_sequence_length(self):
        with pytest.raises(ValueError):
            MicroBatchDataLoader(
                micro_batch_size=2, sequence_length=0, dataset_name="t",
                tokenizer_name="t", num_workers=0, num_proc=1,
                gradient_accumulation_steps=1, device="cpu",
            )

    def test_invalid_gradient_accumulation(self):
        with pytest.raises(ValueError):
            MicroBatchDataLoader(
                micro_batch_size=2, sequence_length=32, dataset_name="t",
                tokenizer_name="t", num_workers=0, num_proc=1,
                gradient_accumulation_steps=0, device="cpu",
            )

    def test_invalid_prefetch_factor(self):
        with pytest.raises(ValueError):
            MicroBatchDataLoader(
                micro_batch_size=2, sequence_length=32, dataset_name="t",
                tokenizer_name="t", num_workers=0, num_proc=1,
                gradient_accumulation_steps=1, device="cpu", prefetch_factor=0,
            )


def _make_loader_with_mock_dataset(seq_len=32, micro_batch=2):
    with (
        patch("scaletorch.data.dataloader.DatasetProcessor") as MockDP,
        patch("scaletorch.data.dataloader.pgm", None),
    ):
        mock_processor = MagicMock()
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=100)
        mock_processor.load_dataset.return_value = mock_dataset
        mock_processor.tokenize_dataset.return_value = mock_dataset
        MockDP.return_value = mock_processor

        loader = MicroBatchDataLoader(
            micro_batch_size=micro_batch, sequence_length=seq_len,
            dataset_name="test", tokenizer_name="test", num_workers=0,
            num_proc=1, gradient_accumulation_steps=1, device="cpu",
        )
        return loader


class TestMicroBatchDataLoaderCollate:
    def test_collate_batch_shapes(self):
        loader = _make_loader_with_mock_dataset(seq_len=32, micro_batch=2)
        batch = [{"input_ids": torch.randint(0, 100, (33,))} for _ in range(2)]
        result = loader.collate_batch(batch)

        assert result["input_ids"].shape == (2, 32)
        assert result["target_ids"].shape == (2, 32)
        assert result["position_ids"].shape == (2, 32)

    def test_target_ids_shifted_by_one(self):
        loader = _make_loader_with_mock_dataset(seq_len=32, micro_batch=1)
        tokens = torch.arange(33)
        result = loader.collate_batch([{"input_ids": tokens}])

        expected_target = tokens[1:33]
        assert torch.equal(result["target_ids"][0], expected_target)

    def test_position_ids_start_at_zero(self):
        loader = _make_loader_with_mock_dataset(seq_len=16, micro_batch=1)
        tokens = torch.randint(0, 100, (17,))
        result = loader.collate_batch([{"input_ids": tokens}])

        assert torch.equal(result["position_ids"][0], torch.arange(16))

    def test_hidden_states_is_none(self):
        loader = _make_loader_with_mock_dataset(seq_len=8, micro_batch=1)
        tokens = torch.randint(0, 100, (9,))
        result = loader.collate_batch([{"input_ids": tokens}])
        assert result["hidden_states"] is None

    def test_empty_batch_raises(self):
        loader = _make_loader_with_mock_dataset(seq_len=8, micro_batch=1)
        with pytest.raises(RuntimeError):
            loader.collate_batch([])


class TestMicroBatchDataLoaderSingleProcess:
    def test_single_process_attributes(self):
        with (
            patch("scaletorch.data.dataloader.DatasetProcessor") as MockDP,
            patch("scaletorch.data.dataloader.pgm", None),
        ):
            mock_processor = MagicMock()
            mock_dataset = MagicMock()
            mock_processor.load_dataset.return_value = mock_dataset
            mock_processor.tokenize_dataset.return_value = mock_dataset
            MockDP.return_value = mock_processor

            loader = MicroBatchDataLoader(
                micro_batch_size=4, sequence_length=64, dataset_name="t",
                tokenizer_name="t", num_workers=0, num_proc=1,
                gradient_accumulation_steps=2, device="cpu",
            )

            assert loader.global_batch_size == 4 * 2
            assert loader.cp_world_size == 1
            assert loader.dp_world_size == 1
            assert loader.sequence_length_per_gpu == 64
