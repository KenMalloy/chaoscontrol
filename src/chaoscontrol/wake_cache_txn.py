"""Transactional WakeCache wrapper for CRCT causal memory scoring.

CRCT's rank-3 oracle compares ``memory_mode='off'`` and
``memory_mode='force_on'`` for the same batch.  Writes from that batch must
not become visible to either side of the comparison.  This module provides a
small MVCC-style transaction layer around :class:`WakeCache`: ``begin_batch``
captures a read cutoff, writes stage inside the transaction, and ``commit``
publishes them with strictly newer event ids.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from chaoscontrol.wake_cache import WakeCache


@dataclass
class CausalEventClock:
    """Monotone event id source used by transactional cache commits."""

    current: int = 0

    def next(self) -> int:
        self.current += 1
        return self.current


@dataclass
class CacheTxn:
    """Batch transaction with a stable read cutoff and staged writes."""

    read_cutoff: int
    staged_moments: list[dict[str, Any]] = field(default_factory=list)
    staged_hidden: list[torch.Tensor] = field(default_factory=list)
    committed: bool = False
    rolled_back: bool = False


class TransactionalWakeCache:
    """MVCC wrapper that preserves WakeCache's bounded-surprise policy.

    Existing moments without an ``_event_id`` are treated as event ``0`` so
    wrapping an already-populated ``WakeCache`` makes old entries visible to
    every future transaction.
    """

    def __init__(
        self,
        base: WakeCache | None = None,
        *,
        max_moments: int = 32,
        max_hidden_buffer: int = 64,
        clock: CausalEventClock | None = None,
    ) -> None:
        self.base = base or WakeCache(
            max_moments=max_moments,
            max_hidden_buffer=max_hidden_buffer,
        )
        self.clock = clock or CausalEventClock()

    @property
    def moments(self) -> list[dict[str, Any]]:
        return self.base.moments

    @property
    def hidden_buffer(self):
        return self.base.hidden_buffer

    def begin_batch(self) -> CacheTxn:
        return CacheTxn(read_cutoff=int(self.clock.current))

    def commit(self, txn: CacheTxn) -> None:
        if txn.committed:
            raise RuntimeError("CacheTxn has already been committed")
        if txn.rolled_back:
            raise RuntimeError("CacheTxn has already been rolled back")
        for moment in txn.staged_moments:
            self._insert_moment(moment, event_id=self.clock.next())
        for hidden in txn.staged_hidden:
            hidden = hidden.detach().cpu()
            setattr(hidden, "_event_id", self.clock.next())
            self.base.hidden_buffer.append(hidden)
        txn.committed = True

    def rollback(self, txn: CacheTxn) -> None:
        if txn.committed:
            raise RuntimeError("CacheTxn has already been committed")
        if txn.rolled_back:
            raise RuntimeError("CacheTxn has already been rolled back")
        txn.staged_moments.clear()
        txn.staged_hidden.clear()
        txn.rolled_back = True

    def reserve_event_ids(
        self,
        n: int,
        *,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Reserve committed memory event ids for an external slot writer.

        ``rank3_score_batch_causal`` writes model memory through
        ``CareStudentLM.append_memory_from_hidden`` rather than through
        ``WakeCache`` itself.  Reserving ids from the same clock keeps the
        model-side slot MVCC and the cache-side transaction cutoff on one
        monotone timeline.
        """
        count = int(n)
        if count < 0:
            raise ValueError(f"cannot reserve a negative number of event ids: {n}")
        ids = [self.clock.next() for _ in range(count)]
        return torch.tensor(ids, dtype=torch.long, device=device)

    def record_moment(
        self,
        *,
        surprise: float,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        hidden: torch.Tensor,
        bucket_ids: torch.Tensor | None = None,
        slot_cues: torch.Tensor | None = None,
        txn: CacheTxn | None = None,
    ) -> None:
        moment: dict[str, Any] = {
            "surprise": float(surprise),
            "inputs": inputs.detach().cpu(),
            "targets": targets.detach().cpu(),
            "hidden": hidden.detach().cpu(),
        }
        if bucket_ids is not None:
            moment["bucket_ids"] = bucket_ids.detach().cpu()
        if slot_cues is not None:
            moment["slot_cues"] = slot_cues.detach().cpu()

        if txn is None:
            self._insert_moment(moment, event_id=self.clock.next())
        else:
            if txn.committed or txn.rolled_back:
                raise RuntimeError("cannot stage writes on a closed CacheTxn")
            txn.staged_moments.append(moment)

    def push_hidden(self, hidden: torch.Tensor, *, txn: CacheTxn | None = None) -> None:
        hidden_cpu = hidden.detach().cpu()
        if txn is None:
            setattr(hidden_cpu, "_event_id", self.clock.next())
            self.base.hidden_buffer.append(hidden_cpu)
        else:
            if txn.committed or txn.rolled_back:
                raise RuntimeError("cannot stage writes on a closed CacheTxn")
            txn.staged_hidden.append(hidden_cpu)

    def visible_moments(self, read_cutoff: int | None) -> list[dict[str, Any]]:
        if read_cutoff is None:
            return list(self.base.moments)
        cutoff = int(read_cutoff)
        return [
            m for m in self.base.moments
            if int(m.get("_event_id", 0)) <= cutoff
        ]

    def visible_hidden(self, read_cutoff: int | None) -> list[torch.Tensor]:
        if read_cutoff is None:
            return list(self.base.hidden_buffer)
        cutoff = int(read_cutoff)
        return [
            h for h in self.base.hidden_buffer
            if int(getattr(h, "_event_id", 0)) <= cutoff
        ]

    def clear(self) -> None:
        self.base.clear()
        self.clock.current = 0

    def _insert_moment(self, moment: dict[str, Any], *, event_id: int) -> None:
        moment = dict(moment)
        moment["_event_id"] = int(event_id)

        if len(self.base.moments) < self.base.max_moments:
            self.base.moments.append(moment)
            return

        min_idx = 0
        min_val = abs(float(self.base.moments[0]["surprise"]))
        for i in range(1, len(self.base.moments)):
            val = abs(float(self.base.moments[i]["surprise"]))
            if val < min_val:
                min_val = val
                min_idx = i
        if abs(float(moment["surprise"])) > min_val:
            self.base.moments[min_idx] = moment


__all__ = [
    "CacheTxn",
    "CausalEventClock",
    "TransactionalWakeCache",
]
