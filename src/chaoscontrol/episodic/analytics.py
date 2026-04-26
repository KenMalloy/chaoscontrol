"""DuckDB-backed analytics over the episodic per-replay diagnostic log.

Phase 3.5 substrate. Query layer for the NDJSON event stream produced
during Phase 3's falsifier matrix run (per Decision 0.9 schema in
``docs/plans/2026-04-25-memory-aware-optimizer-plan.md``).

The per-replay log lives at ``run_dir/episodic_replay_log_rank{R}.ndjson``;
each row carries: ``step, slot, key_fp, write_step, write_pressure,
write_bucket, query_cosine, utility_pre, replay_loss, replay_grad_norm,
replay_grad_cos_common, replay_grad_cos_rare, replay_grad_cos_total,
utility_signal_raw, utility_signal_transformed, utility_post``.

This module wraps ``duckdb.connect()`` + ``read_json_auto(...)`` and
exposes the canonical queries the controller's priority function uses
(Phase 5+) and the post-hoc analyses the Phase 3 result-readout uses
(immediately).

Skeleton in Phase 1; Phase 3.5 fills in the implementations as the
log schema solidifies. All current functions raise NotImplementedError
with a clear pointer; tests pin the API surface so consumers can be
written against it now.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # duckdb is an optional dep at module-import time — avoid forcing
    # an install on machines that don't run analytics.
    import duckdb


_LOG_GLOB = "episodic_replay_log_rank*.ndjson"


@dataclass(frozen=True)
class ReplayLogQuery:
    """Read-only handle wrapping a DuckDB connection over the replay log.

    Construction reads all matching NDJSON files into a single in-memory
    table (DuckDB does this natively via ``read_json_auto``). Queries
    are pure read; no mutation. Connection is closed on context exit.
    """

    run_dir: Path
    _con: "duckdb.DuckDBPyConnection"

    @classmethod
    def open(cls, run_dir: Path | str) -> "ReplayLogQuery":
        """Open a query handle over ``run_dir``'s per-rank replay logs.

        Raises ``FileNotFoundError`` if no logs match the canonical
        pattern. Caller is responsible for ``.close()`` (or use as a
        context manager once Phase 3.5 wires ``__enter__``/``__exit__``).
        """
        try:
            import duckdb  # noqa: F401  (deferred-import guard)
        except ImportError as exc:  # pragma: no cover - opt-dep guard
            raise ImportError(
                "ReplayLogQuery requires duckdb; install with "
                "`pip install duckdb` or use the analytics-extras"
            ) from exc
        raise NotImplementedError(
            "Phase 3.5 substrate stub. Implement when the per-replay "
            "log schema lands in Phase 3.2."
        )

    def close(self) -> None:
        raise NotImplementedError("see ReplayLogQuery.open")

    # ------------------------------------------------------------------
    # Canonical queries — Phase 5 controller priority function reads
    # these; Phase 3 post-hoc analysis also reads these.
    # ------------------------------------------------------------------

    def cohort_replay_utility(self) -> Any:
        """Average replay-grad-cos-rare per write-bucket cohort.

        Returns a DuckDB result with columns
        ``(write_bucket, n_replays, mean_cos_rare, mean_replay_loss,
        mean_utility_post)``. Avenue 2 of the perf hit list (cohort
        analysis): identifies which bucket-cohorts produce the most
        useful replays.
        """
        raise NotImplementedError(
            "Phase 3.5 — implement once the log schema is fixed"
        )

    def drift_trajectory_per_entry(self, *, window_steps: int = 512) -> Any:
        """Per-entry drift trajectory, computed over a rolling window.

        Returns rows ``(slot, write_step, key_fp, drift_norm_avg)`` —
        used by Phase 3.6's drift-correction design (the data feeds the
        decision among full-state / prefix-token / no-refresh).
        """
        raise NotImplementedError("Phase 3.5 — see Phase 3.6 prerequisite")

    def surprise_frontier(self, *, bucket_steps: int = 100) -> Any:
        """Aggregate write-pressure × replay-loss over (step, bucket).

        Avenue 6 of the perf hit list. Used by the controller (Phase 5)
        to bias writes toward the live frontier of rare-event
        difficulty.
        """
        raise NotImplementedError("Phase 3.5 / Phase 5")

    def retrieval_utility_correlation(self) -> Any:
        """Correlation between query_cosine and replay_grad_cos_rare.

        Tests the central thesis empirically: does cosine retrieval
        actually predict replay utility? If correlation is near zero,
        Phase 5.4 predictor research becomes urgent.
        """
        raise NotImplementedError("Phase 3.5 — central thesis check")
