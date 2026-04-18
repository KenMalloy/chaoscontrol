from __future__ import annotations
import json
from pathlib import Path


class MetricsCollector:
    """Per-doc JSONL logger with in-run stability gate.

    Stability gate: tracks the first `stability_window` per-doc losses as
    a frozen baseline (mean, SD), then flags `collapsed` when loss stays
    > `stability_sd_threshold` SDs above the baseline mean for
    `stability_window // 2` consecutive subsequent docs.
    """

    def __init__(
        self,
        *,
        output_path: Path,
        stability_window: int = 100,
        stability_sd_threshold: float = 3.0,
    ) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.output_path.open("w")
        self.stability_window = stability_window
        self.stability_sd_threshold = stability_sd_threshold
        self._pre_window_losses: list[float] = []
        self._baseline_stats: tuple[float, float] | None = None  # (mean, sd) once frozen
        self._consecutive_drift = 0
        self.collapsed = False

    def record_doc(
        self, *, doc_id: int, bpb: float, tokens: int,
        loss_before: float, loss_after: float | None,
        step_count: int, wall_ms: float, grad_norm: float, state_norm: float,
    ) -> None:
        rec = dict(
            doc_id=doc_id, bpb=bpb, tokens=tokens,
            loss_before=loss_before, loss_after=loss_after,
            step_count=step_count, wall_ms=wall_ms,
            grad_norm=grad_norm, state_norm=state_norm,
        )
        self._fh.write(json.dumps(rec) + "\n")
        self._fh.flush()
        # Gate uses loss_before (pre-adapt score) — loss_after can be None,
        # and pre-adapt drift is what signals the model's held-out quality dropping.
        self._update_stability(loss_before)

    def _update_stability(self, loss: float) -> None:
        # Collect pre-window losses until we have enough to freeze a baseline.
        if self._baseline_stats is None:
            self._pre_window_losses.append(loss)
            if len(self._pre_window_losses) >= self.stability_window:
                baseline = self._pre_window_losses[:self.stability_window]
                mean = sum(baseline) / len(baseline)
                var = sum((x - mean) ** 2 for x in baseline) / len(baseline)
                sd = var ** 0.5 if var > 0 else 1e-6
                self._baseline_stats = (mean, sd)
            return
        mean, sd = self._baseline_stats
        if loss - mean > self.stability_sd_threshold * sd:
            self._consecutive_drift += 1
        else:
            self._consecutive_drift = 0
        if self._consecutive_drift >= self.stability_window // 2:
            self.collapsed = True

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
