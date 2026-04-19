from __future__ import annotations


def compute_usable_ttt_budget(
    *,
    total_budget_seconds: float,
    score_floor_seconds: float,
    safety_margin_seconds: float,
) -> float:
    """Return wall-clock seconds available for adaptation after scoring rent."""
    return max(0.0, total_budget_seconds - score_floor_seconds - safety_margin_seconds)


class BudgetTracker:
    """Tracks Exp20 wall-clock accounting under the score-floor framing.

    ``score_floor_seconds`` is measured by a separate no-TTT run and represents
    the whole score-only eval elapsed time, including tokenization and Python
    overhead. If absent on a score-only run, ``summary()`` uses that run's
    elapsed time as the discovered floor.
    """

    def __init__(
        self,
        *,
        total_budget_seconds: float,
        score_floor_seconds: float = 0.0,
        safety_margin_seconds: float = 0.0,
    ) -> None:
        self.total_budget_seconds = float(total_budget_seconds)
        self.configured_score_floor_seconds = max(0.0, float(score_floor_seconds))
        self.safety_margin_seconds = max(0.0, float(safety_margin_seconds))
        self.score_wall_seconds = 0.0
        self.adapt_wall_seconds = 0.0

    @property
    def usable_ttt_budget_seconds(self) -> float:
        return compute_usable_ttt_budget(
            total_budget_seconds=self.total_budget_seconds,
            score_floor_seconds=self.configured_score_floor_seconds,
            safety_margin_seconds=self.safety_margin_seconds,
        )

    @property
    def slack_remaining_seconds(self) -> float:
        return max(0.0, self.usable_ttt_budget_seconds - self.adapt_wall_seconds)

    def can_adapt(self) -> bool:
        return self.slack_remaining_seconds > 0.0

    def add_score_time(self, seconds: float) -> None:
        self.score_wall_seconds += max(0.0, float(seconds))

    def add_adapt_time(self, seconds: float) -> None:
        self.adapt_wall_seconds += max(0.0, float(seconds))

    def _score_floor_for_summary(self, *, score_only_mode: bool, elapsed_seconds: float) -> float:
        if self.configured_score_floor_seconds > 0.0:
            return self.configured_score_floor_seconds
        if score_only_mode:
            return max(0.0, float(elapsed_seconds))
        return 0.0

    def summary(
        self,
        *,
        docs_scored: int,
        chunks_scored: int,
        tokens_scored: int,
        adapt_steps: int,
        timed_out: bool,
        collapsed: bool,
        score_only_mode: bool,
        elapsed_seconds: float,
        ckpt_sha256: str | None = None,
        ckpt_cfg_hash: str | None = None,
        stream_seed: int | None = None,
        gpu_name: str | None = None,
        torch_version: str | None = None,
        cuda_version: str | None = None,
        chunk_size: int | None = None,
        max_docs: int | None = None,
    ) -> dict:
        """Return a summary dict.

        The budget/accounting fields (``total_budget_seconds``, ``elapsed_seconds``,
        ``score_wall_seconds``, ...) are required for the slack-budget arithmetic.
        The ``provenance`` sub-dict is optional but strongly recommended for any
        run whose summary will be cited — it pins the measurement to the exact
        ckpt bytes, ckpt config hash, stream seed, GPU model, and library
        versions that produced it. Without provenance, a summary is not
        independently reproducible.
        """
        score_floor_seconds = self._score_floor_for_summary(
            score_only_mode=score_only_mode,
            elapsed_seconds=elapsed_seconds,
        )
        usable_ttt_budget_seconds = compute_usable_ttt_budget(
            total_budget_seconds=self.total_budget_seconds,
            score_floor_seconds=score_floor_seconds,
            safety_margin_seconds=self.safety_margin_seconds,
        )
        return {
            "total_budget_seconds": self.total_budget_seconds,
            "elapsed_seconds": float(elapsed_seconds),
            "score_wall_seconds": self.score_wall_seconds,
            "adapt_wall_seconds": self.adapt_wall_seconds,
            "other_wall_seconds": max(
                0.0,
                float(elapsed_seconds) - self.score_wall_seconds - self.adapt_wall_seconds,
            ),
            "score_floor_seconds": score_floor_seconds,
            "safety_margin_seconds": self.safety_margin_seconds,
            "usable_ttt_budget_seconds": usable_ttt_budget_seconds,
            "ttt_budget_used_seconds": self.adapt_wall_seconds,
            "slack_remaining_seconds": max(
                0.0, usable_ttt_budget_seconds - self.adapt_wall_seconds
            ),
            "docs_scored": int(docs_scored),
            "chunks_scored": int(chunks_scored),
            "tokens_scored": int(tokens_scored),
            "adapt_steps": int(adapt_steps),
            "timed_out": bool(timed_out),
            "collapsed": bool(collapsed),
            "score_only_mode": bool(score_only_mode),
            "provenance": {
                "ckpt_sha256": ckpt_sha256,
                "ckpt_cfg_hash": ckpt_cfg_hash,
                "stream_seed": stream_seed,
                "gpu_name": gpu_name,
                "torch_version": torch_version,
                "cuda_version": cuda_version,
                "chunk_size": chunk_size,
                "max_docs": max_docs,
            },
        }
