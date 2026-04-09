"""Tests for TTT warming-curve evaluation."""
import torch
from chaoscontrol.evaluation import evaluate_warming_curve
from chaoscontrol.model import ChaosStudentLM


def _make_model():
    return ChaosStudentLM(
        vocab_size=256, dim=64, num_layers=2,
        outer_model_dim=32, outer_model_type="multislot",
        outer_max_slots=0,
        buffer_mode="append_only",
        retrieval_mode="bucket_mean",
        wernicke_enabled=True, wernicke_k_max=4,
        wernicke_router="moe",
    )


def test_warming_curve_returns_dict():
    model = _make_model()
    data = torch.randint(0, 256, (4096,))
    curve = evaluate_warming_curve(
        model, data, warmup_tokens=[0, 100, 500],
        score_tokens=256, segment_len=1024,
    )
    assert isinstance(curve, dict)
    assert 0 in curve
    assert 100 in curve
    assert 500 in curve
    for n, bpb in curve.items():
        assert isinstance(bpb, float)
        assert bpb > 0


def test_warming_curve_all_values_finite():
    """All bpb values should be finite."""
    model = _make_model()
    data = torch.randint(0, 256, (4096,))
    curve = evaluate_warming_curve(
        model, data, warmup_tokens=[0, 100],
        score_tokens=128, segment_len=512,
    )
    import math
    for n, bpb in curve.items():
        assert math.isfinite(bpb), f"bpb for N={n} is not finite: {bpb}"


def test_warming_curve_buffer_reset_between_segments():
    """Buffer should be reset between segments -- each evaluation is fresh."""
    model = _make_model()
    data = torch.randint(0, 256, (4096,))
    # First call
    curve1 = evaluate_warming_curve(
        model, data, warmup_tokens=[0],
        score_tokens=128, segment_len=512,
    )
    # Second call should give same result since buffer is reset
    curve2 = evaluate_warming_curve(
        model, data, warmup_tokens=[0],
        score_tokens=128, segment_len=512,
    )
    # Should be identical (same model, same data, same protocol)
    assert abs(curve1[0] - curve2[0]) < 1e-4


def test_warming_curve_with_topk():
    """Works with bucket_topk retrieval mode."""
    model = ChaosStudentLM(
        vocab_size=256, dim=64, num_layers=2,
        outer_model_dim=32, outer_model_type="multislot",
        outer_max_slots=0,
        buffer_mode="append_only",
        retrieval_mode="bucket_topk", retrieval_k=4,
        wernicke_enabled=True, wernicke_k_max=4,
        wernicke_router="moe",
    )
    data = torch.randint(0, 256, (4096,))
    curve = evaluate_warming_curve(
        model, data, warmup_tokens=[0, 100],
        score_tokens=128, segment_len=512,
    )
    assert len(curve) == 2
