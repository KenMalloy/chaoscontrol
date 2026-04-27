"""Tests for the per-replay diagnostic NDJSON log writer (Phase 3.2).

Per Decision 0.9 of ``docs/plans/2026-04-25-memory-aware-optimizer-plan.md``:
the per-replay log lives at ``run_dir/episodic_replay_log_rank{R}.ndjson``;
each row carries the columns pinned in the schema. The writer is
append-only NDJSON so DuckDB's ``read_json_auto`` consumes it without
transformation in Phase 3.5.

Tests:

* ``test_diagnostics_log_writes_one_row_per_replay`` — calling the
  writer with synthetic event data appends one row per call; the
  serialized JSON round-trips back to the input keys/values.
* ``test_diagnostics_log_schema_matches_decision_0_9`` — pin all
  documented column names; rows missing a column or carrying extras
  surface immediately.
* ``test_diagnostics_log_handles_nan_for_phase1_simplification`` —
  Phase 1 logs NaN for replay-grad cosines (no rare-grad EMA in scope
  per Decision 0.10 with episodic-incompatible ScOpt). NaN must
  serialize as ``null`` so DuckDB treats it as a missing value rather
  than coercing to a string.
"""
from __future__ import annotations

import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from chaoscontrol.episodic.diagnostics import (
    REPLAY_LOG_SCHEMA,
    DiagnosticsLogger,
)


def _synthetic_row(**overrides: object) -> dict[str, object]:
    """Build a row that satisfies every column in REPLAY_LOG_SCHEMA."""
    row = {
        "step": 7,
        "slot": 3,
        "key_fp": 1234567890123,
        "write_step": 4,
        "write_pressure": 0.42,
        "write_bucket": 1,
        "query_cosine": 0.61,
        "utility_pre": 0.55,
        "replay_id": 700,
        "query_event_id": 701,
        "source_write_id": 702,
        "selection_step": 6,
        "policy_version": 2,
        "selected_rank": 0,
        "teacher_score": 0.61,
        "controller_logit": 0.58,
        "replay_loss": 1.273,
        "ce_before_replay": float("nan"),
        "ce_after_replay": 1.273,
        "ce_delta_raw": float("nan"),
        "bucket_baseline": 0.0,
        "reward_shaped": float("nan"),
        "replay_grad_norm": 2.4e-2,
        "replay_grad_cos_common": 0.11,
        "replay_grad_cos_rare": 0.22,
        "replay_grad_cos_total": 0.13,
        "utility_signal_raw": 0.22,
        "utility_signal_transformed": 0.22,
        "utility_post": 0.553,
        "outcome_status": "ok",
        "flags": 0,
        "arm": "unit",
        "chosen_idx": 0,
        "p_chosen": 0.25,
        "p_behavior": [0.25, 0.25, 0.25, 0.25],
        "entropy": 1.38629436,
        "gerber_weight": 1.0,
        "advantage_raw": 0.5,
        "advantage_corrected": 0.4,
        "lambda_hxh": 0.1,
        "feature_manifest_hash": "unit_hash",
        "candidate_slot_ids": [0, 1, 2, 3],
        "candidate_scores": [0.9, 0.7, 0.2, 0.1],
        "logits": [2.0, 1.0, 0.0, -1.0],
    }
    row.update(overrides)
    return row


class TestDiagnosticsSchema(unittest.TestCase):
    """The replay-outcome columns are the contract."""

    def test_diagnostics_log_schema_matches_decision_0_9(self):
        """Pin all documented column names against the canonical
        list. Adding/removing a column without updating Decision 0.9 is
        a breaking change; this test forces the conversation."""
        expected = (
            "step",
            "slot",
            "key_fp",
            "write_step",
            "write_pressure",
            "write_bucket",
            "query_cosine",
            "utility_pre",
            "replay_id",
            "query_event_id",
            "source_write_id",
            "selection_step",
            "policy_version",
            "selected_rank",
            "teacher_score",
            "controller_logit",
            "replay_loss",
            "ce_before_replay",
            "ce_after_replay",
            "ce_delta_raw",
            "bucket_baseline",
            "reward_shaped",
            "replay_grad_norm",
            "replay_grad_cos_common",
            "replay_grad_cos_rare",
            "replay_grad_cos_total",
            "utility_signal_raw",
            "utility_signal_transformed",
            "utility_post",
            "outcome_status",
            "flags",
            "arm",
            "chosen_idx",
            "p_chosen",
            "p_behavior",
            "entropy",
            "gerber_weight",
            "advantage_raw",
            "advantage_corrected",
            "lambda_hxh",
            "feature_manifest_hash",
            "candidate_slot_ids",
            "candidate_scores",
            "logits",
        )
        self.assertEqual(REPLAY_LOG_SCHEMA, expected)
        self.assertEqual(len(REPLAY_LOG_SCHEMA), 44)


class TestDiagnosticsWriter(unittest.TestCase):

    def test_diagnostics_log_writes_one_row_per_replay(self):
        """N writes append N JSON lines to the per-rank log file."""
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "episodic_replay_log_rank0.ndjson"
            logger = DiagnosticsLogger(path)
            try:
                logger.write_row(_synthetic_row(step=0, slot=0))
                logger.write_row(_synthetic_row(step=1, slot=2))
                logger.write_row(_synthetic_row(step=2, slot=5))
            finally:
                logger.close()

            lines = path.read_text().splitlines()
            self.assertEqual(len(lines), 3)
            parsed = [json.loads(line) for line in lines]
            for i, row in enumerate(parsed):
                # Every row carries every documented column.
                for col in REPLAY_LOG_SCHEMA:
                    self.assertIn(col, row)
                self.assertEqual(row["step"], i)
            # Distinct slots came through correctly.
            self.assertEqual([r["slot"] for r in parsed], [0, 2, 5])

    def test_writer_rejects_unknown_columns(self):
        """A row with a column outside the schema is a programming bug
        — surface it so Phase 3.5's DuckDB consumer never sees a row
        whose schema drifted out from under it."""
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "episodic_replay_log_rank0.ndjson"
            logger = DiagnosticsLogger(path)
            try:
                row = _synthetic_row()
                row["mystery_column"] = 42
                with self.assertRaises(KeyError):
                    logger.write_row(row)
            finally:
                logger.close()

    def test_writer_rejects_missing_columns(self):
        """A row missing any documented column is also a bug — Phase 5
        analytics may join on these columns, and a missing field would
        widen NULLs silently."""
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "episodic_replay_log_rank0.ndjson"
            logger = DiagnosticsLogger(path)
            try:
                row = _synthetic_row()
                del row["query_cosine"]
                with self.assertRaises(KeyError):
                    logger.write_row(row)
            finally:
                logger.close()

    def test_diagnostics_log_handles_nan_for_phase1_simplification(self):
        """Phase 1 logs NaN for the three replay-grad cosines and the
        utility_signal_raw/transformed fields when no live rare-grad
        EMA is in scope (per Decision 0.10's Phase 1 simplification +
        ScOpt-episodic incompatibility). The writer must serialize NaN
        as ``null`` so DuckDB ingests it as a missing value rather than
        a string ``"NaN"`` (which fails ``read_json_auto`` typing).
        """
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "episodic_replay_log_rank0.ndjson"
            logger = DiagnosticsLogger(path)
            try:
                logger.write_row(
                    _synthetic_row(
                        replay_grad_cos_common=float("nan"),
                        replay_grad_cos_rare=float("nan"),
                        replay_grad_cos_total=float("nan"),
                        utility_signal_raw=float("nan"),
                    )
                )
            finally:
                logger.close()

            line = path.read_text().splitlines()[0]
            # NaN serialized as null.
            self.assertNotIn("NaN", line)
            self.assertIn("null", line)
            parsed = json.loads(line)
            for col in (
                "replay_grad_cos_common",
                "replay_grad_cos_rare",
                "replay_grad_cos_total",
                "utility_signal_raw",
            ):
                self.assertIsNone(parsed[col])
            # Numeric columns still parse as numbers.
            self.assertIsInstance(parsed["replay_loss"], float)
            self.assertEqual(parsed["step"], 7)

    def test_writer_appends_across_invocations(self):
        """Open + write + close + reopen + write produces a single
        contiguous NDJSON file. Crash recovery and the per-cell
        re-launch path both rely on append semantics."""
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "episodic_replay_log_rank0.ndjson"
            logger = DiagnosticsLogger(path)
            logger.write_row(_synthetic_row(step=0))
            logger.close()
            logger2 = DiagnosticsLogger(path)
            logger2.write_row(_synthetic_row(step=1))
            logger2.close()
            lines = path.read_text().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[0])["step"], 0)
            self.assertEqual(json.loads(lines[1])["step"], 1)


if __name__ == "__main__":
    unittest.main()
