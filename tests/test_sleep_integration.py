"""End-to-end integration test for the full sleep cycle.

Builds a ChaosStudentLM with memory + Wernicke, populates slots,
fills a WakeCache with moments, runs a full_cycle SleepCycle, and
verifies that diagnostics are correct.
"""
from __future__ import annotations

import torch

from chaoscontrol.model import ChaosStudentLM
from chaoscontrol.sleep import SleepConfig, SleepCycle
from chaoscontrol.wake_cache import WakeCache
from chaoscontrol.regret import RegretTable


def _build_model() -> ChaosStudentLM:
    """Build a ChaosStudentLM with outer_model_dim=16, multislot, wernicke."""
    torch.manual_seed(42)
    return ChaosStudentLM(
        vocab_size=256,
        dim=16,
        num_layers=2,
        outer_model_dim=16,
        outer_model_type="multislot",
        outer_max_slots=64,
        semantic_tier_bases=4,
        wernicke_enabled=True,
        wernicke_k_max=4,
        wernicke_window=4,
        typed_storage=True,
    )


def _populate_slots(model: ChaosStudentLM, n: int = 20) -> None:
    """Write n slots with varying survival scores."""
    om = model.outer_model
    for i in range(n):
        h = torch.randn(1, model.dim)
        bucket = i % 4  # distribute across 4 buckets
        om.write(h, bucket_id=bucket)
        # Varying survival: some low (pruneable), some high
        om._survival[-1] = float(i) * 0.05


def _fill_cache(model: ChaosStudentLM, n_moments: int = 8) -> WakeCache:
    """Fill a WakeCache with n_moments of random data."""
    cache = WakeCache(max_moments=32)
    seq_len = 32
    for i in range(n_moments):
        inputs = torch.randint(0, 256, (1, seq_len))
        targets = torch.randint(0, 256, (1, seq_len))
        hidden = torch.randn(1, seq_len, model.dim)
        cache.record_moment(
            surprise=float(i + 1),
            inputs=inputs,
            targets=targets,
            hidden=hidden,
        )
    return cache


class TestSleepIntegration:
    """Full end-to-end sleep cycle integration test."""

    def test_full_cycle_integration(self):
        """Build model, populate, fill cache, run full_cycle, verify diagnostics."""
        model = _build_model()
        _populate_slots(model, n=20)
        cache = _fill_cache(model, n_moments=8)
        regret_table = RegretTable(n_buckets=4, n_actions=4)

        cfg = SleepConfig(
            stages="full_cycle",
            budget=128,
            n2_budget=64,
            rem_budget=64,
            n2_batches=4,
            rem_dreams=2,
            rem_length=8,
            survival_floor=0.1,
        )
        cycle = SleepCycle(cfg)
        diag = cycle.run(model, cache, device="cpu", regret_table=regret_table)

        # Not skipped
        assert "skipped" not in diag, f"Sleep cycle was skipped: {diag.get('skipped')}"

        # All 4 stages present in diagnostics
        assert "n1" in diag, "N1 diagnostics missing"
        assert "n2" in diag, "N2 diagnostics missing"
        assert "n3" in diag, "N3 diagnostics missing"
        assert "rem" in diag, "REM diagnostics missing"

        # N2 scored some slots
        assert diag["n2"]["slots_scored"] > 0, (
            f"N2 should have scored slots, got {diag['n2']['slots_scored']}"
        )

        # N3 did something (pruned or proposed merges or both)
        n3 = diag["n3"]
        n3_did_something = (
            n3["pruned"] > 0
            or n3["merges_proposed"] > 0
            or n3["latent_traces_created"] > 0
        )
        assert n3_did_something, f"N3 should have done work: {n3}"

        # Total ops > 0
        assert diag["total_ops"] > 0, f"Expected total_ops > 0, got {diag['total_ops']}"
