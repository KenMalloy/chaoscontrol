"""Per-replay diagnostic NDJSON log writer (Phase 3.2 of the memory-
aware-optimizer plan, schema per Decision 0.9).

The episodic-rank step body emits one row per replay event. Rows land
at ``run_dir/episodic_replay_log_rank{R}.ndjson`` so DuckDB's
``read_json_auto`` consumes them in Phase 3.5 without transformation.

Schema (rows pinned to exactly these columns; see the CPU SSM controller
event-log design):

    step                       int64    -- training step at which replay fired
    slot                       int64    -- cache slot that was replayed
    key_fp                     int64    -- entry's fingerprint
    write_step                 int64    -- step at which entry was originally written
    write_pressure             float64  -- pressure at write time
    write_bucket               int8     -- token-bucket index (0..3) of value_anchor_id
    query_cosine               float64  -- retrieval ranking signal that selected this
                                        --   entry. Under cosine_utility_weighted mode
                                        --   this is `cosine × utility_u` (the score),
                                        --   NOT raw cosine. Under pressure_only mode
                                        --   this is the pressure proxy. Phase 3.5 can
                                        --   backsolve raw cosine via score / utility_pre
                                        --   for the cosine_utility_weighted arm only.
                                        --   Renaming to `query_score` is queued as a
                                        --   Decision 0.9 amendment; for now the
                                        --   semantic is documented inline.
    utility_pre                float64  -- utility_ema before this replay
    replay_id                  int64    -- controller-issued action id
    query_event_id             int64    -- triggering query id
    source_write_id            int64    -- original admitted write id
    selection_step             int64    -- controller step at selection time
    policy_version             int64    -- controller policy version
    selected_rank              int64    -- 0-based selected rank
    teacher_score              float64  -- heuristic score used as BC teacher
    controller_logit           float64  -- learned controller logit
    replay_loss                float64  -- CE on value tokens after replay forward
    ce_before_replay           float64  -- reserved, null until before/after CE is wired
    ce_after_replay            float64  -- replay CE after forward
    ce_delta_raw               float64  -- before - after, null until wired
    bucket_baseline            float64  -- EMA baseline for shaped reward
    reward_shaped              float64  -- ce_delta_raw - bucket_baseline
    replay_grad_norm           float64  -- CUMULATIVE replay-only grad L2 across the
                                        --   step's replays so far. Per-replay
                                        --   contribution = consecutive-row delta.
    replay_grad_cos_common     float64  -- cosine vs live common-grad direction
    replay_grad_cos_rare       float64  -- cosine vs live rare-grad direction
    replay_grad_cos_total      float64  -- cosine vs total grad
    utility_signal_raw         float64  -- raw signal fed to update_utility (signed)
    utility_signal_transformed float64  -- transformed/clamped value actually applied
    utility_post               float64  -- updated utility_ema after this replay (or
                                        --   == utility_pre if utility update was
                                        --   skipped this replay — see Phase 1 NaN
                                        --   policy in compute_utility_signal)
    outcome_status             str      -- ok / slot_missing / stale / nan / skipped
    flags                      int64    -- reserved bit flags

NaN values serialize as ``null`` so DuckDB ingests them as missing
values; Phase 1 logs NaN for the three replay-grad cosines and the
utility signals when no live rare-grad EMA is in scope (per Decision
0.10's Phase 1 simplification — ScOpt is gated incompatible with
episodic, so a real rare-grad direction isn't available; the column
order and presence are preserved so Phase 4+ can backfill without
schema migration).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


# Pin the documented column names. Re-ordering is a breaking change
# for the DuckDB analytics layer (Phase 3.5) and the Phase 4 ablation
# matrix's downstream readers. Keep in lockstep with Decision 0.9.
REPLAY_LOG_SCHEMA: tuple[str, ...] = (
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
)
_REPLAY_LOG_SCHEMA_SET: frozenset[str] = frozenset(REPLAY_LOG_SCHEMA)


def _coerce_serializable(value: Any) -> Any:
    """Convert a logged value to a JSON-friendly form.

    Float NaN serializes as ``null`` (Python's ``json.dumps`` writes
    ``NaN`` by default, which DuckDB would type as a string and refuse
    to parse as numeric). Tensors / numpy scalars route through
    ``.item()`` so the row is plain-Python-only by the time we hit
    ``json.dumps``.
    """
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, (int, str, bool)):
        return value
    # torch.Tensor / numpy scalar — convert via .item() if available.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            scalar = item()
        except Exception:
            scalar = value
        return _coerce_serializable(scalar)
    return value


class DiagnosticsLogger:
    """Append-only per-rank NDJSON writer for the per-replay log.

    Open with a target path (the runner threads
    ``run_dir/episodic_replay_log_rank{R}.ndjson`` in production).
    Each ``write_row`` appends one JSON object terminated by a
    newline. ``close`` flushes and releases the file handle; reopening
    the same path appends to the existing file (so re-launches and
    retries don't truncate).

    The writer enforces the Decision 0.9 schema: rows must contain
    exactly the columns in ``REPLAY_LOG_SCHEMA`` — no more, no less.
    Schema drift is a programming bug, not a runtime condition, so
    invalid rows raise ``KeyError`` immediately.
    """

    __slots__ = ("path", "_fh")

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Append mode preserves any rows from prior process invocations
        # that survived a crash or matrix re-launch. DuckDB consumes
        # the file via ``read_json_auto`` regardless of how many
        # writers wrote to it, as long as each row is a complete JSON
        # object on its own line.
        self._fh = self.path.open("a", encoding="utf-8")

    def write_row(self, row: dict[str, Any]) -> None:
        """Validate against the schema then append the row as NDJSON.

        Raises ``KeyError`` on missing or extraneous columns.
        """
        keys = set(row.keys())
        missing = _REPLAY_LOG_SCHEMA_SET - keys
        if missing:
            raise KeyError(
                f"DiagnosticsLogger row missing columns: {sorted(missing)}; "
                f"REPLAY_LOG_SCHEMA = {REPLAY_LOG_SCHEMA}"
            )
        extras = keys - _REPLAY_LOG_SCHEMA_SET
        if extras:
            raise KeyError(
                f"DiagnosticsLogger row has unknown columns: "
                f"{sorted(extras)}; the schema is pinned to "
                f"{REPLAY_LOG_SCHEMA}. Update Decision 0.9 first."
            )
        # Preserve the documented column order — DuckDB doesn't care,
        # but a human reading the file will.
        ordered = {col: _coerce_serializable(row[col]) for col in REPLAY_LOG_SCHEMA}
        # ``allow_nan=False`` would raise on NaN; we already coerced
        # NaN to None above so the default writer suffices.
        self._fh.write(json.dumps(ordered) + "\n")

    def flush(self) -> None:
        """Force the OS buffer to disk; useful for tests inspecting
        the file mid-run."""
        self._fh.flush()

    def close(self) -> None:
        """Flush and close the underlying file handle."""
        if not self._fh.closed:
            self._fh.flush()
            self._fh.close()

    def __enter__(self) -> "DiagnosticsLogger":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - GC ordering varies
        try:
            self.close()
        except Exception:
            pass
