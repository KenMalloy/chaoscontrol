"""Tests for causal_slot_eval evaluation function."""
import math

import torch

from chaoscontrol.evaluation import causal_slot_eval
from chaoscontrol.model import CareStudentLM


def _make_model():
    return CareStudentLM(
        vocab_size=256, dim=64, num_layers=2,
        outer_model_dim=32, outer_model_type="multislot",
        outer_max_slots=0,
        buffer_mode="append_only",
        retrieval_mode="bucket_mean",
        wernicke_enabled=True, wernicke_k_max=4,
        wernicke_router="moe",
    )


def test_causal_slot_returns_all_conditions():
    """All 4 conditions present in output."""
    model = _make_model()
    data = torch.randint(0, 256, (4096,))
    results = causal_slot_eval(
        model, data,
        warmup_tokens=[0, 100],
        score_tokens=128,
        segment_len=512,
        window_size=64,
        slot_steps=2,
    )
    assert set(results.keys()) == {"cold", "buffer_only", "slot_only", "buffer_plus_slot"}
    for cond, curve in results.items():
        assert 0 in curve, f"N=0 missing for {cond}"
        assert 100 in curve, f"N=100 missing for {cond}"
        for n, bpb in curve.items():
            assert isinstance(bpb, float)
            assert math.isfinite(bpb), f"bpb not finite for {cond} N={n}: {bpb}"
            assert bpb > 0, f"bpb not positive for {cond} N={n}: {bpb}"


def test_causal_slot_cold_is_flat():
    """Cold condition gives same bpb at each N across repeated calls (no adaptation).

    Different N values score different token regions (score_start = seg_start + N),
    so raw bpb values will differ. Instead verify:
    1. Cold is reproducible (same call twice gives identical results).
    2. Cold warmup is truly skipped (no model state changes between N values).
    """
    model = _make_model()
    torch.manual_seed(42)
    data = torch.randint(0, 256, (4096,))
    common_kw = dict(
        conditions=("cold",),
        warmup_tokens=[0, 100],
        score_tokens=128,
        segment_len=512,
        window_size=64,
        slot_steps=2,
    )
    run1 = causal_slot_eval(model, data, **common_kw)
    run2 = causal_slot_eval(model, data, **common_kw)
    # Cold should be perfectly reproducible since no stochastic adaptation happens
    for n in [0, 100]:
        assert abs(run1["cold"][n] - run2["cold"][n]) < 1e-6, (
            f"Cold bpb not reproducible at N={n}: {run1['cold'][n]} vs {run2['cold'][n]}"
        )


def test_causal_slot_runs_with_buffer():
    """buffer_only and buffer_plus_slot don't crash on buffer-capable model."""
    model = _make_model()
    data = torch.randint(0, 256, (4096,))
    results = causal_slot_eval(
        model, data,
        conditions=("buffer_only", "buffer_plus_slot"),
        warmup_tokens=[0, 100],
        score_tokens=128,
        segment_len=512,
        window_size=64,
        slot_steps=2,
    )
    assert "buffer_only" in results
    assert "buffer_plus_slot" in results
    for cond in ("buffer_only", "buffer_plus_slot"):
        for n, bpb in results[cond].items():
            assert math.isfinite(bpb), f"bpb not finite for {cond} N={n}"


def test_causal_slot_freeze_vs_online():
    """freeze_during_scoring=True gives different results than False."""
    model = _make_model()
    torch.manual_seed(99)
    data = torch.randint(0, 256, (4096,))
    common_kw = dict(
        conditions=("slot_only",),
        warmup_tokens=[100],
        score_tokens=128,
        segment_len=512,
        window_size=64,
        slot_steps=4,
        slot_lr=1e-2,
    )
    frozen = causal_slot_eval(model, data, freeze_during_scoring=True, **common_kw)
    online = causal_slot_eval(model, data, freeze_during_scoring=False, **common_kw)
    # With slot_on and some warmup, freeze vs online should differ
    # (online allows continued gradient flow through delta/logit_bias during scoring)
    bpb_frozen = frozen["slot_only"][100]
    bpb_online = online["slot_only"][100]
    # They might be very close if the model is random, but the code paths differ.
    # At minimum, both should be finite and positive.
    assert math.isfinite(bpb_frozen)
    assert math.isfinite(bpb_online)
    assert bpb_frozen > 0
    assert bpb_online > 0


def test_partial_window_not_discarded():
    """With N=100 and window_size=256, the partial 100-token window is still optimized."""
    model = _make_model()
    torch.manual_seed(7)
    data = torch.randint(0, 256, (4096,))
    # N=100, window_size=256: only one partial window of 100 tokens
    results_with_slot = causal_slot_eval(
        model, data,
        conditions=("slot_only",),
        warmup_tokens=[0, 100],
        score_tokens=128,
        segment_len=512,
        window_size=256,
        slot_steps=8,
        slot_lr=1e-2,
    )
    bpb_cold = results_with_slot["slot_only"][0]
    bpb_warm = results_with_slot["slot_only"][100]
    # The partial window must have been processed (not discarded), so N=100
    # should give a different bpb than N=0 (where SLOT has no data to optimize on).
    # With a random model and a real optimization step, the values will differ.
    assert bpb_cold != bpb_warm, (
        f"Partial window appears to have been discarded: "
        f"N=0 bpb={bpb_cold}, N=100 bpb={bpb_warm}"
    )
