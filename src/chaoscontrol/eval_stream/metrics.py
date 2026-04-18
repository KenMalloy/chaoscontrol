from __future__ import annotations
import json
from pathlib import Path
from collections import deque


class MetricsCollector:
    """Per-doc JSONL logger with in-run stability gate.

    Stability gate: tracks a rolling window of per-doc loss; flags `collapsed`
    if loss remains > N SDs above the pre-window mean for K consecutive docs.
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
        self._loss_history: deque[float] = deque(maxlen=10_000)
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
        self._update_stability(loss_before)

    def _update_stability(self, loss: float) -> None:
        self._loss_history.append(loss)
        if len(self._loss_history) < self.stability_window + self.stability_window // 2:
            return
        baseline = list(self._loss_history)[:self.stability_window]
        mean = sum(baseline) / len(baseline)
        var = sum((x - mean) ** 2 for x in baseline) / len(baseline)
        sd = var ** 0.5 if var > 0 else 1e-6
        if loss - mean > self.stability_sd_threshold * sd:
            self._consecutive_drift += 1
        else:
            self._consecutive_drift = 0
        if self._consecutive_drift >= self.stability_window // 2:
            self.collapsed = True

    def close(self) -> None:
        self._fh.close()
