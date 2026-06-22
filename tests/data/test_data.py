"""Tests for scaletorch.data.dataloader module."""

import unittest
from unittest.mock import MagicMock, patch

import torch

from scaletorch.data.dataloader import MicroBatchDataLoader


class TestMicroBatchDataLoaderValidation(unittest.TestCase):
    def test_invalid_micro_batch_size(self):
        with self.assertRaises(ValueError):
            MicroBatchDataLoader(
                micro_batch_size=0,
                sequence_length=32,
                dataset_name="test",
                tokenizer_name="test",
                num_workers=0,
                num_proc=1,
                gradient_accumulation_steps=1,
                device="cpu",
            )

    def test_invalid_sequence_length(self):
        with self.assertRaises(ValueError):
            MicroBatchDataLoader(
                micro_batch_size=2,
                sequence_length=0,
                dataset_name="test",
                tokenizer_name="test",
                num_workers=0,
                num_proc=1,
                gradient_accumulation_steps=1,
                device="cpu",
            )

    def test_invalid_gradient_accumulation(self):
        with self.assertRaises(ValueError):
            MicroBatchDataLoader(
                micro_batch_size=2,
                sequence_length=32,
                dataset_name="test",
                tokenizer_name="test",
                num_workers=0,
                num_proc=1,
                gradient_accumulation_steps=0,
                device="cpu",
            )

    def test_invalid_prefetch_factor(self):
        with self.assertRaises(ValueError):
            MicroBatchDataLoader(
                micro_batch_size=2,
                sequence_length=32,
                dataset_name="test",
                tokenizer_name="test",
                num_workers=0,
                num_proc=1,
                gradient_accumulation_steps=1,
                device="cpu",
                prefetch_factor=0,
            )


class TestMicroBatchDataLoaderCollate(unittest.TestCase):
    def _make_loader_with_mock_dataset(self, seq_len=32, micro_batch=2):
        """Create a MicroBatchDataLoader with a mock dataset processor."""
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
                micro_batch_size=micro_batch,
                sequence_length=seq_len,
                dataset_name="test",
                tokenizer_name="test",
                num_workers=0,
                num_proc=1,
                gradient_accumulation_steps=1,
                device="cpu",
            )
            return loader

    def test_collate_batch_shapes(self):
        loader = self._make_loader_with_mock_dataset(seq_len=32, micro_batch=2)
        # Simulate tokenized sequences of length seq_len + 1 = 33
        batch = [{"input_ids": torch.randint(0, 100, (33,))} for _ in range(2)]
        result = loader.collate_batch(batch)

        self.assertIn("input_ids", result)
        self.assertIn("target_ids", result)
        self.assertIn("position_ids", result)

        # In single-process mode, cp_world_size=1, so seq_per_gpu = 32
        self.assertEqual(result["input_ids"].shape, (2, 32))
        self.assertEqual(result["target_ids"].shape, (2, 32))
        self.assertEqual(result["position_ids"].shape, (2, 32))

    def test_target_ids_shifted_by_one(self):
        loader = self._make_loader_with_mock_dataset(seq_len=32, micro_batch=1)
        tokens = torch.arange(33)
        batch = [{"input_ids": tokens}]
        result = loader.collate_batch(batch)

        # target_ids should be shifted by 1 relative to input_ids
        result["input_ids"][0]
        target_ids = result["target_ids"][0]
        expected_target = tokens[1:33]
        self.assertTrue(torch.equal(target_ids, expected_target))

    def test_position_ids_start_at_zero(self):
        loader = self._make_loader_with_mock_dataset(seq_len=16, micro_batch=1)
        tokens = torch.randint(0, 100, (17,))
        batch = [{"input_ids": tokens}]
        result = loader.collate_batch(batch)

        pos_ids = result["position_ids"][0]
        expected = torch.arange(16)
        self.assertTrue(torch.equal(pos_ids, expected))

    def test_hidden_states_is_none(self):
        loader = self._make_loader_with_mock_dataset(seq_len=8, micro_batch=1)
        tokens = torch.randint(0, 100, (9,))
        batch = [{"input_ids": tokens}]
        result = loader.collate_batch(batch)
        self.assertIsNone(result["hidden_states"])

    def test_empty_batch_raises(self):
        loader = self._make_loader_with_mock_dataset(seq_len=8, micro_batch=1)
        with self.assertRaises(RuntimeError):
            loader.collate_batch([])


class TestMicroBatchDataLoaderSingleProcess(unittest.TestCase):
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
                micro_batch_size=4,
                sequence_length=64,
                dataset_name="test",
                tokenizer_name="test",
                num_workers=0,
                num_proc=1,
                gradient_accumulation_steps=2,
                device="cpu",
            )

            self.assertEqual(loader.global_batch_size, 4 * 2)
            self.assertEqual(loader.cp_world_size, 1)
            self.assertEqual(loader.dp_world_size, 1)
            self.assertEqual(loader.sequence_length_per_gpu, 64)


if __name__ == "__main__":
    unittest.main()
