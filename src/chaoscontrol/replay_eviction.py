"""Replay-eviction loop for CRCT rank-3 idle maintenance.

The SSM trunk and Muon optimizer aggressively smooth representations.
Episodic memory captures sharp details that smoothing would lose. As
the SSM trains and internalizes patterns, some cached entries become
redundant — the trunk can now handle those cases without the cache.

This module continuously replays existing memory slots through the
evolving SSM on rank 3's idle cycles. Slots whose utility has decayed
to near-zero are evicted. Slots that remain useful are retained. The
result is a living cache that stays synchronized with the trunk's
evolving capabilities rather than a static store that fills up and
gets mechanically compressed.

The core scoring uses two forward passes (same cost as one teacher
scoring): encode with memory off, encode with memory on, then
attribute per-token utility to individual slots via cached retrieval
weights. This is amortized — all slots are scored simultaneously.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from .cache_utility import chunked_nll_from_hidden

__all__ = ["ReplayEvictionLoop"]


@torch.inference_mode()
def replay_score_slots(
    *,
    model: Any,
    probe_input_ids: torch.Tensor,
    probe_valid_mask: torch.Tensor,
    cache_read_cutoff: int | None = None,
) -> dict[str, Any]:
    """Score all visible memory slots' utility in two forward passes.

    Returns per-slot utility attributed via retrieval weights, plus
    diagnostics. The retrieval weights are cached by the model's
    outer_model.read() during the memory-on encode pass.
    """
    x = probe_input_ids[:, :-1]
    y = probe_input_ids[:, 1:]
    mask = probe_valid_mask[:, 1:].bool()

    device_type = x.device.type
    ac = torch.autocast(device_type, dtype=torch.bfloat16) if device_type == "cuda" else torch.autocast("cpu", dtype=torch.bfloat16)

    with ac:
        h_off = model.encode(x, memory_mode="off", cache_read_cutoff=cache_read_cutoff)
        h_mem = model.encode(x, memory_mode="force_on", cache_read_cutoff=cache_read_cutoff)

    nll_off = chunked_nll_from_hidden(model, h_off, y)
    nll_mem = chunked_nll_from_hidden(model, h_mem, y)

    per_token_utility = (nll_off - nll_mem) * mask.float()

    outer = getattr(model, "outer_model", None)
    if outer is None:
        return {"slot_utilities": torch.tensor([]), "slot_indices": [], "mean_utility": 0.0}

    retrieval_weights = getattr(outer, "_retrieval_weights", None)
    retrieval_indices = getattr(outer, "_retrieval_indices", None)

    if retrieval_weights is None or retrieval_indices is None:
        return {"slot_utilities": torch.tensor([]), "slot_indices": [], "mean_utility": 0.0}

    # retrieval_weights: (batch, num_visible_slots) from softmax attention
    # per_token_utility: (batch, seq-1)
    # Credit attribution: weight each token's utility by how much each
    # slot contributed to that token's retrieval.
    #
    # For sequence-level retrieval (weights are per-batch, not per-token),
    # the credit for slot j = mean_over_batch(weight_j * mean_token_utility)
    batch_mean_utility = (per_token_utility.sum(dim=1) / mask.float().sum(dim=1).clamp(min=1)).float()
    slot_credit = (retrieval_weights.float().T @ batch_mean_utility.unsqueeze(-1)).squeeze(-1)
    slot_credit = slot_credit / max(1.0, float(retrieval_weights.shape[0]))

    return {
        "slot_utilities": slot_credit.cpu(),
        "slot_indices": list(retrieval_indices),
        "mean_utility": float(per_token_utility[mask].mean().item()) if mask.any() else 0.0,
        "n_tokens_scored": int(mask.sum().item()),
    }


class ReplayEvictionLoop:
    """Runs on rank 3 in idle time between teacher scoring intervals.

    Each tick replays a cached probe batch through the model, measures
    per-slot replay utility, updates per-slot EMA estimates, and evicts
    slots whose smoothed utility has decayed below threshold.
    """

    def __init__(
        self,
        *,
        eviction_threshold: float = 0.01,
        eviction_ema_beta: float = 0.9,
        min_slot_age_steps: int = 128,
        max_seconds_per_tick: float = 0.5,
        trace_path: str | None = None,
        trace_max_rows: int = 0,
    ) -> None:
        self._threshold = float(eviction_threshold)
        self._ema_beta = float(eviction_ema_beta)
        self._min_age = int(min_slot_age_steps)
        self._max_seconds = float(max_seconds_per_tick)

        self._slot_utility_ema: dict[int, float] = {}
        self._slot_first_seen_step: dict[int, int] = {}
        self._slot_score_count: dict[int, int] = {}

        self._probe_input_ids: torch.Tensor | None = None
        self._probe_valid_mask: torch.Tensor | None = None
        self._probe_cache_cutoff: int | None = None
        self._probe_step: int = 0

        self._tick_count: int = 0
        self._evictions_total: int = 0
        self._replays_total: int = 0
        self._slots_scored_total: int = 0

        self._trace_path = None if trace_path in (None, "") else Path(str(trace_path))
        self._trace_max_rows = max(0, int(trace_max_rows))
        self._trace_rows_written = 0
        self._trace_buffer: list[str] = []

    def cache_probe(
        self,
        *,
        input_ids: torch.Tensor,
        valid_mask: torch.Tensor,
        cache_read_cutoff: int | None,
        step: int,
    ) -> None:
        """Cache a probe batch from the most recent teacher scoring."""
        self._probe_input_ids = input_ids.detach()
        self._probe_valid_mask = valid_mask.detach()
        self._probe_cache_cutoff = cache_read_cutoff
        self._probe_step = step

    def has_probe(self) -> bool:
        return self._probe_input_ids is not None

    def tick(
        self,
        *,
        model: Any,
        step: int,
    ) -> list[int]:
        """One maintenance tick. Returns list of slot indices evicted."""
        if self._probe_input_ids is None:
            return []

        t0 = time.monotonic()
        self._tick_count += 1

        result = replay_score_slots(
            model=model,
            probe_input_ids=self._probe_input_ids,
            probe_valid_mask=self._probe_valid_mask,
            cache_read_cutoff=self._probe_cache_cutoff,
        )

        slot_utilities = result["slot_utilities"]
        slot_indices = result["slot_indices"]

        if len(slot_indices) == 0:
            return []

        self._replays_total += 1
        self._slots_scored_total += len(slot_indices)

        for i, slot_idx in enumerate(slot_indices):
            utility = float(slot_utilities[i].item()) if i < len(slot_utilities) else 0.0

            if slot_idx not in self._slot_first_seen_step:
                self._slot_first_seen_step[slot_idx] = step
                self._slot_utility_ema[slot_idx] = utility
                self._slot_score_count[slot_idx] = 1
            else:
                old = self._slot_utility_ema.get(slot_idx, utility)
                self._slot_utility_ema[slot_idx] = self._ema_beta * old + (1.0 - self._ema_beta) * utility
                self._slot_score_count[slot_idx] = self._slot_score_count.get(slot_idx, 0) + 1

        elapsed = time.monotonic() - t0
        if elapsed > self._max_seconds:
            return []

        evicted = self._apply_evictions(model=model, step=step)
        return evicted

    def _apply_evictions(
        self,
        *,
        model: Any,
        step: int,
    ) -> list[int]:
        """Evict slots whose smoothed utility is below threshold and old enough."""
        outer = getattr(model, "outer_model", None)
        if outer is None:
            return []

        evict_indices: list[int] = []
        for slot_idx, ema_utility in self._slot_utility_ema.items():
            first_seen = self._slot_first_seen_step.get(slot_idx, step)
            age = step - first_seen
            score_count = self._slot_score_count.get(slot_idx, 0)

            if age < self._min_age:
                continue
            if score_count < 2:
                continue
            if ema_utility < self._threshold:
                evict_indices.append(slot_idx)

        if not evict_indices:
            return []

        actual_evicted = _evict_slots(outer, evict_indices)

        for idx in actual_evicted:
            self._slot_utility_ema.pop(idx, None)
            self._slot_first_seen_step.pop(idx, None)
            self._slot_score_count.pop(idx, None)

        self._evictions_total += len(actual_evicted)

        self._maybe_trace(step=step, evicted=actual_evicted)

        # Reindex tracking dicts after eviction (slot indices shift down)
        if actual_evicted:
            self._reindex_after_eviction(actual_evicted)

        return actual_evicted

    def _reindex_after_eviction(self, evicted: list[int]) -> None:
        """After slots are removed, higher indices shift down. Rebuild tracking."""
        evicted_set = set(evicted)
        sorted_evicted = sorted(evicted_set, reverse=True)

        new_ema: dict[int, float] = {}
        new_first_seen: dict[int, int] = {}
        new_count: dict[int, int] = {}

        for old_idx in sorted(self._slot_utility_ema.keys()):
            if old_idx in evicted_set:
                continue
            shift = sum(1 for e in sorted_evicted if e < old_idx)
            new_idx = old_idx - shift
            new_ema[new_idx] = self._slot_utility_ema[old_idx]
            if old_idx in self._slot_first_seen_step:
                new_first_seen[new_idx] = self._slot_first_seen_step[old_idx]
            if old_idx in self._slot_score_count:
                new_count[new_idx] = self._slot_score_count[old_idx]

        self._slot_utility_ema = new_ema
        self._slot_first_seen_step = new_first_seen
        self._slot_score_count = new_count

    def _maybe_trace(self, *, step: int, evicted: list[int]) -> None:
        if self._trace_path is None:
            return
        for idx in evicted:
            if self._trace_max_rows > 0 and self._trace_rows_written >= self._trace_max_rows:
                break
            row = {
                "row_type": "replay_eviction",
                "step": step,
                "slot_index": idx,
                "ema_utility": self._slot_utility_ema.get(idx, 0.0),
                "score_count": self._slot_score_count.get(idx, 0),
                "tick": self._tick_count,
            }
            self._trace_buffer.append(json.dumps(row, separators=(",", ":")) + "\n")
            self._trace_rows_written += 1

    def flush_trace(self) -> None:
        if self._trace_path is None or not self._trace_buffer:
            return
        try:
            self._trace_path.parent.mkdir(parents=True, exist_ok=True)
            with self._trace_path.open("a", encoding="utf-8") as fh:
                fh.write("".join(self._trace_buffer))
            self._trace_buffer.clear()
        except Exception:
            pass

    def diagnostics(self) -> dict[str, Any]:
        self.flush_trace()
        n_tracked = len(self._slot_utility_ema)
        utilities = list(self._slot_utility_ema.values())
        return {
            "tick_count": self._tick_count,
            "replays_total": self._replays_total,
            "evictions_total": self._evictions_total,
            "slots_scored_total": self._slots_scored_total,
            "slots_tracked": n_tracked,
            "mean_ema_utility": sum(utilities) / max(1, n_tracked),
            "min_ema_utility": min(utilities) if utilities else 0.0,
            "max_ema_utility": max(utilities) if utilities else 0.0,
            "eviction_threshold": self._threshold,
            "has_probe": self.has_probe(),
            "probe_step": self._probe_step,
        }


def _evict_slots(outer: Any, indices: list[int]) -> list[int]:
    """Remove slots from a MultiSlotOuterModel by index.

    Processes in descending order to avoid index invalidation.
    Returns the list of indices actually evicted.
    """
    if not hasattr(outer, "_slots"):
        return []

    n_slots = len(outer._slots)
    valid = sorted([i for i in indices if 0 <= i < n_slots], reverse=True)

    for i in valid:
        del outer._slots[i]
        if i < len(outer._survival):
            del outer._survival[i]
        if i < len(outer._slot_buckets):
            del outer._slot_buckets[i]
        if i < len(outer._slot_event_ids):
            del outer._slot_event_ids[i]

    return sorted(valid)
