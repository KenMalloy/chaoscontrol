import json
from pathlib import Path
from chaoscontrol.eval_stream.metrics import MetricsCollector


def test_writes_per_doc_record(tmp_path):
    out = tmp_path / "metrics.jsonl"
    collector = MetricsCollector(output_path=out)
    collector.record_doc(
        doc_id=0, bpb=1.5, tokens=128, loss_before=0.5, loss_after=0.48,
        step_count=2, wall_ms=123.4, grad_norm=0.8, state_norm=1.1,
    )
    collector.close()
    lines = out.read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["doc_id"] == 0
    assert rec["bpb"] == 1.5
    assert rec["state_norm"] == 1.1


def test_stability_gate_flags_collapse(tmp_path):
    out = tmp_path / "metrics.jsonl"
    collector = MetricsCollector(
        output_path=out, stability_window=5, stability_sd_threshold=3.0,
    )
    # Steady losses then a spike persisting 5 docs
    for i in range(10):
        loss = 2.0 + (0.01 * i) + (20.0 if i >= 5 else 0.0)
        collector.record_doc(doc_id=i, bpb=loss, tokens=100, loss_before=loss,
                             loss_after=None, step_count=0, wall_ms=1.0,
                             grad_norm=0.0, state_norm=1.0)
    assert collector.collapsed is True


def test_baseline_frozen_across_long_run(tmp_path):
    """Baseline is frozen to the first `stability_window` docs; it must not drift.

    Regression test for the bug where `deque(maxlen=N)[:window]` silently
    shifted the baseline past doc N — the gate's detection window and Exp 20's
    collapse-detection window overlap exactly at doc 10K-30K.
    """
    out = tmp_path / "metrics.jsonl"
    # Small window, small threshold, many docs.
    collector = MetricsCollector(
        output_path=out, stability_window=10, stability_sd_threshold=3.0,
    )
    # First 10 docs: low steady loss → baseline mean ~1.0, sd ~0.01
    for i in range(10):
        collector.record_doc(doc_id=i, bpb=1.0, tokens=1, loss_before=1.0 + 0.01 * (i % 2),
                             loss_after=None, step_count=0, wall_ms=1.0,
                             grad_norm=0.0, state_norm=1.0)
    # Docs 10..9999: slowly drift upward to a new steady state around 1.5 —
    # a transformer-gradient-style slow drift that a rolling baseline would
    # absorb and stop flagging. Frozen baseline keeps flagging.
    for i in range(10, 10_000):
        collector.record_doc(doc_id=i, bpb=1.5, tokens=1, loss_before=1.5,
                             loss_after=None, step_count=0, wall_ms=1.0,
                             grad_norm=0.0, state_norm=1.0)
    # With a frozen baseline and enough consecutive drift, collapsed must be True.
    assert collector.collapsed is True


def test_gate_does_not_fire_on_steady_model(tmp_path):
    """No collapse when losses stay inside the baseline band."""
    out = tmp_path / "metrics.jsonl"
    collector = MetricsCollector(
        output_path=out, stability_window=10, stability_sd_threshold=3.0,
    )
    import random
    rng = random.Random(0)
    for i in range(500):
        # All losses in [0.98, 1.02] — way inside 3σ of any reasonable baseline.
        collector.record_doc(doc_id=i, bpb=1.0, tokens=1,
                             loss_before=1.0 + 0.02 * (rng.random() - 0.5),
                             loss_after=None, step_count=0, wall_ms=1.0,
                             grad_norm=0.0, state_norm=1.0)
    assert collector.collapsed is False


def test_context_manager_closes_file(tmp_path):
    out = tmp_path / "metrics.jsonl"
    with MetricsCollector(output_path=out) as collector:
        collector.record_doc(doc_id=0, bpb=1.0, tokens=1, loss_before=1.0,
                             loss_after=None, step_count=0, wall_ms=1.0,
                             grad_norm=0.0, state_norm=1.0)
    # File should be closed after the with-block exits.
    assert collector._fh.closed


def test_all_fields_roundtrip_through_json(tmp_path):
    """Every field written by record_doc must round-trip through JSON. Pins
    the on-disk schema so a refactor that renames `state_norm` → `state_norm_avg`
    or drops a field gets caught here, not silently in Phase F analysis.

    Also confirms loss_after=None serializes as JSON null (Python None → null
    → None on read).
    """
    out = tmp_path / "metrics.jsonl"
    with MetricsCollector(output_path=out) as collector:
        # Record with loss_after populated
        collector.record_doc(
            doc_id=11, bpb=1.234, tokens=500,
            loss_before=0.987, loss_after=0.654,
            step_count=3, wall_ms=456.7, grad_norm=1.23, state_norm=4.56,
        )
        # Record with loss_after=None (no-weight-TTT case)
        collector.record_doc(
            doc_id=12, bpb=2.345, tokens=750,
            loss_before=1.111, loss_after=None,
            step_count=0, wall_ms=78.9, grad_norm=0.0, state_norm=5.67,
        )
    lines = out.read_text().splitlines()
    assert len(lines) == 2
    r0 = json.loads(lines[0])
    r1 = json.loads(lines[1])

    # Every field from the record_doc signature, first record.
    assert r0["doc_id"] == 11
    assert r0["bpb"] == 1.234
    assert r0["tokens"] == 500
    assert r0["loss_before"] == 0.987
    assert r0["loss_after"] == 0.654
    assert r0["step_count"] == 3
    assert r0["wall_ms"] == 456.7
    assert r0["grad_norm"] == 1.23
    assert r0["state_norm"] == 4.56

    # Second record: loss_after=None must round-trip as JSON null → Python None.
    assert r1["doc_id"] == 12
    assert r1["loss_after"] is None
    assert r1["step_count"] == 0
