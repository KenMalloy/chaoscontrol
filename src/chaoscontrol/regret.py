"""Experience-weighted regret tracking per semantic type (Wernicke bucket).

Tracks counterfactual regret for actions not taken, grouped by information
set (bucket_id). Uses regret matching to bias future exploration toward
under-explored alternatives. Inspired by CFR but adapted for single-agent
language modeling — no adversary, regret drives exploration not equilibrium.
"""
from __future__ import annotations
import torch


class RegretTable:
    """Experience-weighted regret tracking per semantic type (Wernicke bucket).

    Tracks counterfactual regret for actions not taken, grouped by information
    set (bucket_id). Uses regret matching to bias future exploration toward
    under-explored alternatives. Inspired by CFR but adapted for single-agent
    language modeling — no adversary, regret drives exploration not equilibrium.
    """

    def __init__(self, n_buckets: int, n_actions: int) -> None:
        self.n_buckets = n_buckets
        self.n_actions = n_actions
        self.cumulative_regret = torch.zeros(n_buckets, n_actions)

    def update(
        self,
        bucket_id: int,
        action_taken: int,
        counterfactual_values: list[float],
        actual_value: float,
    ) -> None:
        """Accumulate regret for actions not taken."""
        for a in range(self.n_actions):
            if a != action_taken:
                regret = counterfactual_values[a] - actual_value
                self.cumulative_regret[bucket_id, a] += regret

    def get_regrets(self, bucket_id: int) -> torch.Tensor:
        """Return cumulative regret vector for a bucket."""
        return self.cumulative_regret[bucket_id].clone()

    def get_strategy(self, bucket_id: int) -> torch.Tensor:
        """Regret matching: probability proportional to positive cumulative regret."""
        positive = self.cumulative_regret[bucket_id].clamp(min=0)
        total = positive.sum()
        if total > 0:
            return positive / total
        return torch.ones(self.n_actions) / self.n_actions

    def reset_bucket(self, bucket_id: int) -> None:
        """Clear regret for a specific bucket."""
        self.cumulative_regret[bucket_id] = 0
