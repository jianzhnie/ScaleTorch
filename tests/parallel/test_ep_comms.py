"""Tests for scaletorch.parallel.expert_parallel.ep_comms — unit-level logic tests."""

import unittest
from unittest.mock import MagicMock, patch

import torch

# Pre-import the module so patch.object() can resolve attributes
import scaletorch.parallel.expert_parallel.ep_comms as ep_mod


class TestDispatchTokensLogic(unittest.TestCase):
    """Test the token routing / sorting logic inside dispatch_tokens.

    Because dispatch_tokens internally calls dist.all_to_all_single, we mock
    that out and verify the *non-communication* logic: destination-rank
    assignment, sorting, split counts, and reorder index.
    """

    def _run_dispatch(self, num_tokens, top_k, num_experts, ep_size, ep_rank):
        """Call dispatch_tokens with mocked distributed comms."""
        mock_pgm = MagicMock()
        mock_pgm.ep_group = MagicMock()

        def fake_all_to_all_single(out, inp, **kwargs):
            out.copy_(inp)

        with (
            patch.object(ep_mod, "pgm", mock_pgm),
            patch.object(ep_mod, "dist") as mock_dist,
        ):
            mock_dist.all_to_all_single.side_effect = fake_all_to_all_single

            hidden = 8
            hidden_states = torch.randn(num_tokens, hidden)
            topk_indices = torch.randint(0, num_experts, (num_tokens, top_k))
            topk_weights = torch.rand(num_tokens, top_k)

            result = ep_mod.dispatch_tokens(
                hidden_states, topk_indices, topk_weights,
                num_experts, ep_size, ep_rank,
            )
            (recv_tokens, recv_expert_ids, recv_weights,
             send_splits, recv_splits, reorder_idx) = result

            return {
                "recv_tokens": recv_tokens,
                "recv_expert_ids": recv_expert_ids,
                "recv_weights": recv_weights,
                "send_splits": send_splits,
                "recv_splits": recv_splits,
                "reorder_idx": reorder_idx,
                "num_tokens": num_tokens,
                "top_k": top_k,
                "ep_size": ep_size,
                "num_experts": num_experts,
            }

    def test_send_splits_sum(self):
        """send_splits should sum to num_tokens * top_k."""
        r = self._run_dispatch(
            num_tokens=16, top_k=2, num_experts=8, ep_size=2, ep_rank=0)
        self.assertEqual(sum(r["send_splits"]), r["num_tokens"] * r["top_k"])
        self.assertEqual(len(r["send_splits"]), r["ep_size"])

    def test_reorder_idx_is_permutation(self):
        """reorder_idx should be a permutation of 0..num_tokens*top_k-1."""
        r = self._run_dispatch(
            num_tokens=8, top_k=2, num_experts=4, ep_size=2, ep_rank=0)
        n = r["num_tokens"] * r["top_k"]
        idx = r["reorder_idx"]
        self.assertEqual(idx.numel(), n)
        self.assertEqual(set(idx.tolist()), set(range(n)))

    def test_recv_expert_ids_are_local(self):
        """Received expert IDs should be in [0, experts_per_rank)."""
        r = self._run_dispatch(
            num_tokens=10, top_k=2, num_experts=8, ep_size=2, ep_rank=0)
        experts_per_rank = r["num_experts"] // r["ep_size"]
        self.assertTrue((r["recv_expert_ids"] >= 0).all())
        self.assertTrue((r["recv_expert_ids"] < experts_per_rank).all())

    def test_single_ep_rank(self):
        """With ep_size=1, all tokens stay local."""
        r = self._run_dispatch(
            num_tokens=4, top_k=1, num_experts=4, ep_size=1, ep_rank=0)
        self.assertEqual(r["send_splits"], [4])
        self.assertEqual(r["recv_tokens"].shape[0], 4)


class TestGatherTokensLogic(unittest.TestCase):
    """Test gather_tokens reorder logic."""

    def test_reorder_reversal(self):
        """gather_tokens should undo the sort from dispatch."""
        mock_pgm = MagicMock()
        mock_pgm.ep_group = MagicMock()

        def fake_all_to_all_single(out, inp, **kwargs):
            out.copy_(inp)

        with (
            patch.object(ep_mod, "pgm", mock_pgm),
            patch.object(ep_mod, "dist") as mock_dist,
        ):
            mock_dist.all_to_all_single.side_effect = fake_all_to_all_single

            hidden = 4
            num_tokens = 6
            top_k = 2
            total = num_tokens * top_k

            expert_output = torch.arange(
                total * hidden, dtype=torch.float).reshape(total, hidden)

            reorder_idx = torch.arange(total)
            result = ep_mod.gather_tokens(
                expert_output,
                send_splits=[total],
                recv_splits=[total],
                reorder_idx=reorder_idx,
                num_tokens=num_tokens,
                top_k=top_k,
                hidden=hidden,
            )
            self.assertTrue(torch.equal(result, expert_output))


if __name__ == "__main__":
    unittest.main()
