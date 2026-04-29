"""CPU evidence-engine boundary tests."""
from __future__ import annotations

import pytest
import torch

from chaoscontrol.kernels import _cpu_ssm_controller as _ext


def test_amx_nll_is_not_python_public() -> None:
    assert not hasattr(_ext, "amx_bf16_nll")
    if getattr(_ext, "_C", None) is not None:
        assert not hasattr(_ext._C, "amx_bf16_nll")


def test_cpu_evidence_engine_surface_and_starvation_diagnostics() -> None:
    if getattr(_ext, "_C", None) is None:
        pytest.skip("cpu_ssm_controller extension not built")

    from chaoscontrol.evidence import CpuEvidenceEngine

    assert CpuEvidenceEngine is _ext.CpuEvidenceEngine
    engine = _ext.CpuEvidenceEngine(lanes=2, d_model=384)
    try:
        diag = engine.diagnostics()
        assert diag["lanes"] == 2
        assert diag["t_probe_max"] == 32
        assert diag["k_tile"] == 64
        assert diag["gpu3_starvation_reason"] in {
            "ok",
            "no_slots",
            "confidence_gate",
            "frame_stale",
            "scheduler_behind",
            "job_ring_empty",
            "result_ring_full",
            "oracle_saturated",
        }
        assert "gpu3_idle_seconds_by_reason" in diag
        assert "lane_tile_advances_total" in diag
        assert "lane_tile_drops_total" in diag
        assert "lane_work_items_emitted_total" in diag
        assert "lane_cue_nll_seconds_total" in diag
    finally:
        engine.shutdown()


def test_cpu_evidence_engine_ingest_rejects_wrong_shape_before_work() -> None:
    if getattr(_ext, "_C", None) is None:
        pytest.skip("cpu_ssm_controller extension not built")

    engine = _ext.CpuEvidenceEngine(lanes=1, d_model=384)
    try:
        cue_hidden = torch.zeros((16, 384), dtype=torch.bfloat16)
        cue_targets = torch.zeros((32,), dtype=torch.int32)
        with pytest.raises(RuntimeError, match="cue_hidden shape mismatch"):
            engine.ingest_frame(cue_hidden, cue_targets, frame_id=1, step=2)
    finally:
        engine.shutdown()
