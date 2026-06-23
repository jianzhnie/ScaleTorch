"""Tests for scaletorch.utils.checkpoint — CheckpointManager save/load round-trip."""

import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn

from scaletorch.utils.checkpoint import CheckpointManager


class TestCheckpointManager(unittest.TestCase):
    """Round-trip save / load tests for CheckpointManager."""

    def setUp(self):
        self.pgm_patcher = patch("scaletorch.utils.checkpoint.pgm", None)
        self.pgm_patcher.start()
        self.dist_patcher = patch(
            "scaletorch.utils.checkpoint.st_dist"
        )
        self.mock_dist = self.dist_patcher.start()
        self.mock_dist.is_distributed.return_value = False

    def tearDown(self):
        self.pgm_patcher.stop()
        self.dist_patcher.stop()

    def _make_model_and_optimizer(self):
        model = nn.Linear(4, 2, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        return model, optimizer

    def test_checkpoint_path_format(self):
        mgr = CheckpointManager()
        path = mgr._get_checkpoint_path("/tmp/ckpt")
        self.assertIn("tp_rank_world_size=0_1", str(path))
        self.assertIn("pp_rank_world_size=0_1", str(path))
        self.assertTrue(str(path).endswith(".pth"))

    def test_save_and_load_round_trip(self, tmp_path=None):
        import tempfile

        tmp_dir = tempfile.mkdtemp()
        try:
            mgr = CheckpointManager()
            model, optimizer = self._make_model_and_optimizer()

            # Run a fake training step to populate optimizer state
            out = model(torch.randn(2, 4))
            out.sum().backward()
            optimizer.step()

            original_weight = model.weight.data.clone()

            mgr.save_checkpoint(model, optimizer, trained_steps=10, trained_tokens=5000, out_dir=tmp_dir)

            # Verify checkpoint file exists
            ckpt_path = mgr._get_checkpoint_path(tmp_dir)
            self.assertTrue(ckpt_path.exists(), f"Checkpoint not written to {ckpt_path}")

            # Corrupt model weights then load
            model.weight.data.zero_()
            self.assertFalse(torch.allclose(model.weight.data, original_weight))

            steps, tokens = mgr.load_checkpoint(model, optimizer, tmp_dir)
            self.assertEqual(steps, 10)
            self.assertEqual(tokens, 5000)
            self.assertTrue(torch.allclose(model.weight.data, original_weight))
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_load_missing_checkpoint_raises(self):
        mgr = CheckpointManager()
        model, optimizer = self._make_model_and_optimizer()
        with self.assertRaises(FileNotFoundError):
            mgr.load_checkpoint(model, optimizer, "/nonexistent/path")

    def test_load_invalid_format_raises(self):
        import tempfile

        tmp_dir = tempfile.mkdtemp()
        try:
            mgr = CheckpointManager()
            # Write a checkpoint with missing keys
            ckpt_path = mgr._get_checkpoint_path(tmp_dir)
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": {}}, ckpt_path)

            model, optimizer = self._make_model_and_optimizer()
            with self.assertRaises(RuntimeError):
                mgr.load_checkpoint(model, optimizer, tmp_dir)
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
