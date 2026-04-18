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
