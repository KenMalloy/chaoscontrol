"""Replay-outcome NDJSON schema regression pin (Phase D3 of
``docs/plans/2026-04-26-cpu-ssm-controller.md``).

The on-disk per-rank NDJSON log
(``run_dir/episodic_replay_log_rank{R}.ndjson``) is the primary corpus
the offline PyTorch BC + value-prediction pretrain pipeline (Phase D4)
will consume to bootstrap the CPU SSM controller. Two contracts must
hold:

1. **Wire-schema parity (subset).** Every field in the
   ``REPLAY_OUTCOME`` wire struct
   (``src/chaoscontrol/kernels/_cpu_ssm_controller/src/wire_events.h``)
   must be present in the NDJSON schema. The NDJSON is a *superset* —
   it carries additional GPU-side context columns (``replay_loss``,
   ``write_pressure``, ``query_cosine``, ``utility_signal_*``,
   ``replay_grad_norm``, ``replay_grad_cos_common``, ``utility_post``,
   etc.) that BC + value-prediction wants. The wire schema is the
   inter-process IPC contract; the NDJSON is the richer training
   corpus. We pin the rename map below as a single source of truth so
   D4's loader can import it without re-deriving.

2. **NaN-reserved Phase 4 columns serialize correctly.** Per the
   substrate landing (commit ``1295301``) and Decision 0.10's Phase 1
   policy, ``replay_grad_cos_rare`` / ``replay_grad_cos_total`` are
   reserved as NaN until Phase 4 wires the rare-grad direction. NaN
   must serialize to JSON ``null`` (not the string ``"NaN"``) so
   DuckDB ``read_json_auto`` ingests the column as a numeric with
   missing values, not as a string column. The existing diagnostics
   suite covers this for the three ``replay_grad_cos_*`` columns; this
   file pins the regression for the two Phase-4-reserved columns
   specifically.

D4 reader notes (surfaced here so the next agent doesn't have to
re-derive them):

* ``outcome_status`` is a string in the NDJSON (``"ok"`` / ``"pending"``
  / ``"slot_missing"`` / ``"stale"`` / ``"nan"`` / ``"skipped"``) but a
  ``uint8`` in the wire struct (``0``–``4``). The reader either keeps
  strings (D4 trains on strings via embedding lookup) or maps via
  the canonical enum order in ``wire_events.h``.
* ``event_type`` is implicit per file — every row in
  ``episodic_replay_log_rank{R}.ndjson`` is a REPLAY_OUTCOME — so the
  NDJSON omits it. The reader injects the constant ``3`` when joining
  against other event-type tables.
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


# Wire → NDJSON column rename map. Every key is a field of the
# ``ReplayOutcome`` C++ struct in
# ``src/chaoscontrol/kernels/_cpu_ssm_controller/src/wire_events.h``.
# Every value is the corresponding column name in
# ``REPLAY_LOG_SCHEMA``. ``event_type`` is intentionally absent — see
# the module docstring for the per-file-implicit rationale.
#
# This map is the single source of truth for D4's loader. Importing it
# from this test (or copying the dict literal) keeps the rename
# discipline in one place.
WIRE_TO_NDJSON_COLUMN_MAP: dict[str, str] = {
    # Identity / linkage
    "replay_id": "replay_id",
    "gpu_step": "step",
    "query_event_id": "query_event_id",
    "source_write_id": "source_write_id",
    "slot_id": "slot",
    "selection_step": "selection_step",
    "policy_version": "policy_version",
    "selected_rank": "selected_rank",
    # Policy outputs
    "teacher_score": "teacher_score",
    "controller_logit": "controller_logit",
    # Reward signal (V1 reward = ce_delta_raw - bucket_baseline)
    "ce_before_replay": "ce_before_replay",
    "ce_after_replay": "ce_after_replay",
    "ce_delta_raw": "ce_delta_raw",
    "bucket_baseline": "bucket_baseline",
    "reward_shaped": "reward_shaped",
    # Phase 4 reserved (NaN until rare-grad direction wires up)
    "grad_cos_rare": "replay_grad_cos_rare",
    "grad_cos_total": "replay_grad_cos_total",
    # Bookkeeping
    "outcome_status": "outcome_status",
    "flags": "flags",
}


def _synthetic_row(**overrides: object) -> dict[str, object]:
    """Build a row that satisfies every column in REPLAY_LOG_SCHEMA.

    Mirrors ``tests/test_episodic_diagnostics.py::_synthetic_row`` so
    the writer's schema-pin checks accept it without modification.
    """
    row: dict[str, object] = {
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
        "ce_before_replay": 1.5,
        "ce_after_replay": 1.273,
        "ce_delta_raw": 0.227,
        "bucket_baseline": 0.05,
        "reward_shaped": 0.177,
        "replay_grad_norm": 2.4e-2,
        "replay_grad_cos_common": 0.11,
        "replay_grad_cos_rare": float("nan"),
        "replay_grad_cos_total": float("nan"),
        "utility_signal_raw": 0.22,
        "utility_signal_transformed": 0.22,
        "utility_post": 0.553,
        "outcome_status": "ok",
        "flags": 0,
    }
    row.update(overrides)
    return row


class TestReplayOutcomeWireParity(unittest.TestCase):
    """The NDJSON must carry every wire-event field, possibly renamed."""

    def test_every_wire_field_has_an_ndjson_column(self):
        """For each ``ReplayOutcome`` field in ``wire_events.h`` (minus
        the per-file-implicit ``event_type``), the renamed column must
        be present in ``REPLAY_LOG_SCHEMA``. If the C++ struct gains a
        field, this test fails until D3's rename map and (if needed) the
        diagnostics schema catch up — exactly the regression catch the
        Phase D3 plan asks for."""
        ndjson_columns = set(REPLAY_LOG_SCHEMA)
        missing: list[tuple[str, str]] = []
        for wire_field, ndjson_column in WIRE_TO_NDJSON_COLUMN_MAP.items():
            if ndjson_column not in ndjson_columns:
                missing.append((wire_field, ndjson_column))
        self.assertEqual(
            missing,
            [],
            f"NDJSON schema is missing columns required by the "
            f"REPLAY_OUTCOME wire struct (wire → ndjson): {missing}. "
            f"Either extend REPLAY_LOG_SCHEMA or update "
            f"WIRE_TO_NDJSON_COLUMN_MAP in this test, whichever side "
            f"the wire-schema change wants to land on.",
        )

    def test_rename_map_is_well_formed(self):
        """No two wire fields may rename to the same NDJSON column —
        that would silently merge distinct upstream signals when D4
        joins the trace tables."""
        ndjson_columns = list(WIRE_TO_NDJSON_COLUMN_MAP.values())
        duplicates = sorted(
            {col for col in ndjson_columns if ndjson_columns.count(col) > 1}
        )
        self.assertEqual(
            duplicates,
            [],
            f"WIRE_TO_NDJSON_COLUMN_MAP collisions: {duplicates} — each "
            f"NDJSON column may name at most one wire field.",
        )

    def test_event_type_is_per_file_implicit(self):
        """Document the per-file-implicit policy as a test so it can't
        be undone by accident: ``event_type`` is implicit per file (one
        REPLAY_OUTCOME event type per ``episodic_replay_log_rank{R}``
        file) and therefore intentionally absent from the rename map."""
        self.assertNotIn("event_type", WIRE_TO_NDJSON_COLUMN_MAP)
        self.assertNotIn("event_type", REPLAY_LOG_SCHEMA)


class TestReplayOutcomeWriterRoundTrip(unittest.TestCase):
    """Write one synthetic replay outcome and verify the columns the
    wire schema requires arrive readable from the NDJSON."""

    def test_writer_round_trips_every_wire_column(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "episodic_replay_log_rank0.ndjson"
            logger = DiagnosticsLogger(path)
            try:
                logger.write_row(_synthetic_row())
            finally:
                logger.close()
            (line,) = path.read_text().splitlines()
            row = json.loads(line)
            for wire_field, ndjson_column in WIRE_TO_NDJSON_COLUMN_MAP.items():
                self.assertIn(
                    ndjson_column,
                    row,
                    f"NDJSON row missing column {ndjson_column!r} (wire "
                    f"field {wire_field!r}) after writer round-trip.",
                )

    def test_phase4_reserved_columns_serialize_nan_as_null(self):
        """``replay_grad_cos_rare`` and ``replay_grad_cos_total`` are
        reserved as NaN until Phase 4 wires the rare-grad direction.
        The writer must serialize NaN as JSON ``null`` so DuckDB
        ``read_json_auto`` types these columns as numeric-with-missing
        rather than coercing the whole column to string. This pins the
        invariant for the two specific columns the wire schema reserves
        (the existing diagnostics suite covers the broader NaN policy
        for all three ``replay_grad_cos_*`` columns)."""
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "episodic_replay_log_rank0.ndjson"
            logger = DiagnosticsLogger(path)
            try:
                logger.write_row(
                    _synthetic_row(
                        replay_grad_cos_rare=float("nan"),
                        replay_grad_cos_total=float("nan"),
                    )
                )
            finally:
                logger.close()
            line = path.read_text().splitlines()[0]
            self.assertNotIn("NaN", line)
            parsed = json.loads(line)
            for col in ("replay_grad_cos_rare", "replay_grad_cos_total"):
                self.assertIsNone(
                    parsed[col],
                    f"Phase-4-reserved column {col} must serialize NaN "
                    f"to JSON null; got {parsed[col]!r}",
                )

    def test_outcome_status_serializes_as_string(self):
        """The wire struct types ``outcome_status`` as ``uint8`` (the
        enum value); the NDJSON keeps it as the enum name string for
        human-readable analytics. D4's loader must therefore either
        train on the string label directly (embedding lookup) or remap
        via the canonical enum order in ``wire_events.h``. This test
        pins the string-on-disk form so the contract is explicit."""
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "episodic_replay_log_rank0.ndjson"
            logger = DiagnosticsLogger(path)
            try:
                logger.write_row(_synthetic_row(outcome_status="ok"))
                logger.write_row(_synthetic_row(outcome_status="slot_missing"))
            finally:
                logger.close()
            rows = [json.loads(ln) for ln in path.read_text().splitlines()]
            self.assertEqual(rows[0]["outcome_status"], "ok")
            self.assertEqual(rows[1]["outcome_status"], "slot_missing")
            self.assertIsInstance(rows[0]["outcome_status"], str)


if __name__ == "__main__":
    unittest.main()
