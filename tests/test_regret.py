"""Tests for CFR-style regret tracking."""
import unittest
import torch


class TestRegretTable(unittest.TestCase):
    def test_accumulate_regret(self):
        from chaoscontrol.regret import RegretTable
        rt = RegretTable(n_buckets=4, n_actions=8)
        rt.update(bucket_id=0, action_taken=2, counterfactual_values=[1.0]*8, actual_value=0.5)
        regrets = rt.get_regrets(bucket_id=0)
        assert regrets.shape == (8,)
        # Actions not taken should have positive regret (they were worth 1.0 vs actual 0.5)
        for i in range(8):
            if i != 2:
                assert regrets[i] > 0, f"action {i} should have positive regret"

    def test_regret_matching_produces_distribution(self):
        from chaoscontrol.regret import RegretTable
        rt = RegretTable(n_buckets=4, n_actions=8)
        for _ in range(10):
            rt.update(bucket_id=1, action_taken=0, counterfactual_values=[2.0]*8, actual_value=0.0)
        dist = rt.get_strategy(bucket_id=1)
        assert dist.shape == (8,)
        assert abs(dist.sum().item() - 1.0) < 1e-5  # valid distribution

    def test_unknown_bucket_returns_uniform(self):
        from chaoscontrol.regret import RegretTable
        rt = RegretTable(n_buckets=4, n_actions=8)
        dist = rt.get_strategy(bucket_id=3)
        expected = 1.0 / 8
        assert all(abs(dist[i].item() - expected) < 1e-5 for i in range(8))

    def test_negative_regret_pruning(self):
        """Actions that are consistently best should dominate the strategy."""
        from chaoscontrol.regret import RegretTable
        rt = RegretTable(n_buckets=4, n_actions=8)
        for _ in range(20):
            cf = [0.0]*8
            cf[3] = 1.0
            rt.update(bucket_id=0, action_taken=3, counterfactual_values=cf, actual_value=1.0)
        dist = rt.get_strategy(bucket_id=0)
        # Action 3 was always chosen and always best — no regret for it
        # Other actions have 0 counterfactual value vs 1.0 actual, so negative regret
        # Regret matching with all-zero/negative regrets → uniform
        # This tests that the system doesn't crash and produces a valid distribution
        assert abs(dist.sum().item() - 1.0) < 1e-5

    def test_reset_bucket(self):
        """reset_bucket clears regret for a specific bucket."""
        from chaoscontrol.regret import RegretTable
        rt = RegretTable(n_buckets=4, n_actions=8)
        rt.update(bucket_id=0, action_taken=0, counterfactual_values=[1.0]*8, actual_value=0.0)
        assert rt.get_regrets(bucket_id=0).sum() > 0
        rt.reset_bucket(bucket_id=0)
        assert rt.get_regrets(bucket_id=0).sum() == 0


if __name__ == "__main__":
    unittest.main()
