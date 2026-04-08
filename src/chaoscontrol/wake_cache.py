"""WakeCache — stores high-signal moments from training for sleep consolidation.

During the wake (training) phase the model encounters surprising inputs that
deserve extra consolidation.  WakeCache keeps a bounded buffer of the most
surprising moments so the sleep cycle can replay them.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional

import torch


class WakeCache:
    """Bounded cache of high-signal training moments for offline replay.

    Parameters
    ----------
    max_moments : int
        Maximum number of moment dicts retained.  When the limit is reached
        the moment with the lowest ``abs(surprise)`` is evicted.
    max_hidden_buffer : int
        Rolling buffer size for raw hidden states pushed via
        :meth:`push_hidden`.
    """

    def __init__(
        self,
        max_moments: int = 32,
        max_hidden_buffer: int = 64,
    ) -> None:
        self.max_moments = max_moments
        self.max_hidden_buffer = max_hidden_buffer

        self.moments: List[Dict[str, Any]] = []
        self.hidden_buffer: deque[torch.Tensor] = deque(maxlen=max_hidden_buffer)
        self._bucket_counts: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Moment recording
    # ------------------------------------------------------------------

    def record_moment(
        self,
        *,
        surprise: float,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        hidden: torch.Tensor,
        bucket_ids: Optional[torch.Tensor] = None,
        slot_cues: Optional[torch.Tensor] = None,
    ) -> None:
        """Store a training moment, evicting lowest-signal entry if at capacity.

        All tensor fields are detached and moved to CPU so the cache never
        holds onto GPU memory or the computation graph.
        """
        moment: Dict[str, Any] = {
            "surprise": float(surprise),
            "inputs": inputs.detach().cpu(),
            "targets": targets.detach().cpu(),
            "hidden": hidden.detach().cpu(),
        }
        if bucket_ids is not None:
            moment["bucket_ids"] = bucket_ids.detach().cpu()
        if slot_cues is not None:
            moment["slot_cues"] = slot_cues.detach().cpu()

        if len(self.moments) < self.max_moments:
            self.moments.append(moment)
        else:
            # Find the moment with the lowest abs(surprise).
            min_idx = 0
            min_val = abs(self.moments[0]["surprise"])
            for i in range(1, len(self.moments)):
                val = abs(self.moments[i]["surprise"])
                if val < min_val:
                    min_val = val
                    min_idx = i

            # Only insert if the new moment outranks the weakest.
            if abs(surprise) > min_val:
                self.moments[min_idx] = moment

    # ------------------------------------------------------------------
    # Hidden buffer
    # ------------------------------------------------------------------

    def push_hidden(self, hidden: torch.Tensor) -> None:
        """Append a hidden state to the rolling buffer (detached, CPU)."""
        self.hidden_buffer.append(hidden.detach().cpu())

    # ------------------------------------------------------------------
    # Bucket statistics
    # ------------------------------------------------------------------

    def update_bucket_counts(self, bucket_ids: torch.Tensor) -> None:
        """Accumulate a bincount of Wernicke bucket assignments."""
        ids = bucket_ids.detach().cpu().long().reshape(-1)
        counts = torch.bincount(ids)
        if self._bucket_counts is None:
            self._bucket_counts = counts
        else:
            # Pad the shorter tensor so shapes match before adding.
            max_len = max(self._bucket_counts.shape[0], counts.shape[0])
            if self._bucket_counts.shape[0] < max_len:
                self._bucket_counts = torch.nn.functional.pad(
                    self._bucket_counts, (0, max_len - self._bucket_counts.shape[0])
                )
            if counts.shape[0] < max_len:
                counts = torch.nn.functional.pad(
                    counts, (0, max_len - counts.shape[0])
                )
            self._bucket_counts = self._bucket_counts + counts

    def bucket_distribution(self, n_buckets: int) -> torch.Tensor:
        """Return a normalised frequency distribution over *n_buckets*.

        If no bucket data has been recorded yet, returns a uniform
        distribution of shape ``(n_buckets,)``.
        """
        if self._bucket_counts is None:
            return torch.ones(n_buckets) / n_buckets

        counts = self._bucket_counts
        if counts.shape[0] < n_buckets:
            counts = torch.nn.functional.pad(
                counts, (0, n_buckets - counts.shape[0])
            )
        elif counts.shape[0] > n_buckets:
            counts = counts[:n_buckets]

        total = counts.sum()
        if total == 0:
            return torch.ones(n_buckets) / n_buckets
        return counts.float() / total.float()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Reset all state — moments, hidden buffer, and bucket counts."""
        self.moments.clear()
        self.hidden_buffer.clear()
        self._bucket_counts = None
