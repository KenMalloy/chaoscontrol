"""DuckDB-ready action-space trace sink for learned controller heads."""
from __future__ import annotations

import json

from chaoscontrol.episodic.diagnostics import (
    ACTION_SPACE_TRACE_SCHEMA,
    ActionSpaceTraceLogger,
)


def test_action_space_trace_logger_writes_pinned_json_columns(tmp_path):
    path = tmp_path / "action_space.ndjson"
    logger = ActionSpaceTraceLogger(path)
    logger.append({
        "gpu_step": 17,
        "event_type": "action_space_delta",
        "head_name": "write_admission",
        "raw_action": [0.1, -0.2],
        "bounded_action": [0.01, -0.02],
        "invariant_name": "bounded_residual",
        "clamp_amount": 0.0,
        "readiness": 0.5,
        "reward_context": {"bucket": 2, "score": float("nan")},
        "accepted": True,
    })
    logger.close()

    row = json.loads(path.read_text().strip())
    assert tuple(row.keys()) == ACTION_SPACE_TRACE_SCHEMA
    assert row["gpu_step"] == 17
    assert row["head_name"] == "write_admission"
    assert json.loads(row["raw_action_json"]) == [0.1, -0.2]
    assert json.loads(row["bounded_action_json"]) == [0.01, -0.02]
    assert json.loads(row["reward_context_json"]) == {
        "bucket": 2,
        "score": None,
    }
