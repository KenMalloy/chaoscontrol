"""Primitive Dreamworld helpers for Experiment 24.

This module intentionally provides just the small, testable building blocks
needed for the first phase of DREAM replay work:

* a bounded replay buffer with hard age-out and diagnostics,
* replay-window extraction utilities,
* entry capture using cached pre-replay state, and
* a single teacher-forced replay backward pass, and
* a curated-cache replay path (Phase 3.1) that runs replay backward
  from an episodic cache entry's value-token span.
"""

from dataclasses import dataclass
from typing import Any

import math

import torch
import torch.nn.functional as F

try:
    from chaoscontrol.train_ssm import (
        full_lm_head_backward,
        fused_lm_head_backend_for_mode,
        fused_lm_head_backward,
    )
except Exception:  # pragma: no cover - local import fallback for primitive tests.
    full_lm_head_backward = None
    fused_lm_head_backend_for_mode = None
    fused_lm_head_backward = None


@dataclass
class DreamReplayEntry:
    step: int
    states: list[torch.Tensor]
    replay_tokens: torch.Tensor


class DreamReplayBuffer:
    """Bounded deque-like replay storage with hard age-based eviction."""

    def __init__(self, max_entries: int, max_age_steps: int) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be greater than 0")
        if max_age_steps <= 0:
            raise ValueError("max_age_steps must be greater than 0")

        self.max_entries = int(max_entries)
        self.max_age_steps = int(max_age_steps)
        self._entries: list[DreamReplayEntry] = []
        self.add_count = 0
        self.sample_count = 0
        self.drop_count = 0

    def __len__(self) -> int:
        return len(self._entries)

    def _prune_stale(self, current_step: int) -> None:
        keep: list[DreamReplayEntry] = []
        for entry in self._entries:
            age = current_step - entry.step
            if age <= self.max_age_steps:
                keep.append(entry)
            else:
                self.drop_count += 1
        self._entries = keep

    def _drop_over_capacity(self) -> None:
        while len(self._entries) > self.max_entries:
            self._entries.pop(0)
            self.drop_count += 1

    def add(
        self,
        *,
        step: int,
        states: list[torch.Tensor],
        replay_tokens: torch.Tensor,
    ) -> None:
        """Store detached cloned tensors and enforce age/capacity bounds."""
        sanitized_states = [state.detach().clone() for state in states]
        sanitized_replay = replay_tokens.detach().clone()
        self._entries.append(
            DreamReplayEntry(
                step=int(step),
                states=sanitized_states,
                replay_tokens=sanitized_replay,
            )
        )
        self.add_count += 1
        self._prune_stale(int(step))
        self._drop_over_capacity()

    def sample(self, generator: torch.Generator, current_step: int) -> DreamReplayEntry | None:
        """Return one random sample for the current step, or ``None`` if empty."""
        self._prune_stale(current_step)
        self._drop_over_capacity()
        self.sample_count += 1
        if not self._entries:
            return None
        idx = int(torch.randint(0, len(self._entries), (1,), generator=generator).item())
        return self._entries[idx]

    def diagnostics(self, current_step: int) -> dict[str, Any]:
        """Return buffer counters and age statistics."""
        self._prune_stale(current_step)
        self._drop_over_capacity()
        ages = [current_step - entry.step for entry in self._entries]
        if ages:
            age_min = int(min(ages))
            age_max = int(max(ages))
            age_mean = float(sum(ages) / len(ages))
        else:
            age_min = 0
            age_max = 0
            age_mean = 0.0

        return {
            "size": len(self._entries),
            "max_entries": self.max_entries,
            "max_age_steps": self.max_age_steps,
            "add_count": self.add_count,
            "sample_count": self.sample_count,
            "drop_count": self.drop_count,
            "age_min": age_min,
            "age_max": age_max,
            "age_mean": age_mean,
        }


def build_dream_replay_tokens(
    inputs: torch.Tensor,
    prefix_tokens: int,
    replay_tokens: int,
) -> torch.Tensor:
    """Build a replay token span that starts at seed token and includes targets."""
    if prefix_tokens <= 0:
        raise ValueError(f"prefix_tokens must be > 0, got {prefix_tokens}")
    if replay_tokens <= 0:
        raise ValueError(f"replay_tokens must be > 0, got {replay_tokens}")
    if inputs.dim() != 2:
        raise ValueError(f"inputs must be 2D (batch, seq), got shape {tuple(inputs.shape)}")

    start = prefix_tokens - 1
    end = prefix_tokens + replay_tokens
    if end > inputs.shape[1]:
        raise ValueError(
            f"replay span [{start}, {end}) does not fit into seq length {inputs.shape[1]}"
        )
    return inputs[:, start:end].detach()


def capture_dream_entry(
    model: Any,
    inputs: torch.Tensor,
    *,
    step: int,
    prefix_tokens: int,
    replay_tokens: int,
) -> DreamReplayEntry:
    """Capture states and replay window for one dream seed event."""
    if prefix_tokens < 2:
        raise ValueError(f"prefix_tokens must be at least 2, got {prefix_tokens}")

    with torch.no_grad():
        _, states = model.encode(
            inputs[:, : prefix_tokens - 1],
            return_final_states=True,
        )
        replay = build_dream_replay_tokens(
            inputs=inputs,
            prefix_tokens=prefix_tokens,
            replay_tokens=replay_tokens,
        )
    return DreamReplayEntry(step=step, states=states, replay_tokens=replay)


def _subsample_replay_entry(
    entry: DreamReplayEntry,
    *,
    replay_batch_size: int,
    generator: torch.Generator | None,
) -> DreamReplayEntry:
    batch = int(entry.replay_tokens.shape[0])
    size = int(replay_batch_size)
    if size <= 0 or size >= batch:
        return entry

    row_idx_cpu = torch.randperm(batch, generator=generator)[:size]
    replay_idx = row_idx_cpu.to(device=entry.replay_tokens.device)
    replay = entry.replay_tokens.index_select(0, replay_idx)
    states = [
        state.index_select(0, row_idx_cpu.to(device=state.device))
        for state in entry.states
    ]
    return DreamReplayEntry(step=entry.step, states=states, replay_tokens=replay)


def dreamworld_replay_backward(
    model: Any,
    entry: DreamReplayEntry,
    weight: float,
    *,
    lm_head_backward_mode: str = "single",
    lm_head_tile_size: int = 1024,
    replay_batch_size: int = 0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Run one teacher-forced replay backward pass and return unweighted loss."""
    entry = _subsample_replay_entry(
        entry,
        replay_batch_size=replay_batch_size,
        generator=generator,
    )
    device = next(model.parameters()).device
    states = [state.to(device=device) for state in entry.states]
    replay = entry.replay_tokens.to(device=device)
    replay_inputs = replay[:, :-1].to(torch.int32)
    targets = replay[:, 1:].to(torch.long)

    hidden = model.encode(replay_inputs, initial_states=states, return_final_states=False)
    mode = str(lm_head_backward_mode).strip().lower()
    if mode == "single" or fused_lm_head_backward is None:
        if full_lm_head_backward is not None:
            return full_lm_head_backward(
                hidden,
                model.final_norm,
                model.lm_head,
                targets,
                loss_weight=float(weight),
            )
        logits = model.lm_head(model.final_norm(hidden))
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="mean",
        )
        (loss * float(weight)).backward()
        return loss.detach()

    if fused_lm_head_backend_for_mode is None:
        raise RuntimeError("fused LM-head backend resolver is unavailable")
    backend_name = fused_lm_head_backend_for_mode(mode)
    return fused_lm_head_backward(
        hidden=hidden,
        final_norm=model.final_norm,
        lm_head=model.lm_head,
        targets=targets,
        backend=backend_name,
        tile_size=int(lm_head_tile_size),
        loss_weight=float(weight),
    )


def _replay_grad_norm_flat(model: Any) -> float:
    """Sum-of-squares L2 norm over every grad-bearing model param.

    Used for the per-replay diagnostic log's ``replay_grad_norm`` field.
    Computed BEFORE the SUM all-reduce so the norm reflects the
    replay-only contribution, not the post-reduce ``main_avg + replay``
    aggregate.
    """
    total = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        total += float(p.grad.detach().pow(2).sum().item())
    return math.sqrt(total)


def compute_utility_signal(
    replay_grad_cos_rare: float,
) -> tuple[float, float]:
    """Decision 0.10: utility-signal pipeline (raw → clamped).

    ``replay_grad_cos_rare`` is the cosine between the replay grad
    direction and the live rare-grad direction (signed in [-1, 1]).
    The cache scoring function is ``score = cosine × utility_u``, so
    a negative ``utility_u`` would invert the cosine ordering at
    retrieval time (entries with anti-aligned replay grads would
    LOOK BETTER for queries with anti-aligned residuals). Clamping
    to [0, 1] avoids that pathology while preserving the
    "this-replay-was-useful" signal as ``[0, 1]``-valued evidence.

    NaN policy: a NaN raw signal (Phase 1's no-rare-EMA fallback)
    transforms to 0.0 deterministically — feeding NaN into the
    utility EMA would corrupt every subsequent update_utility call
    against the slot. Negative-non-NaN inputs clamp to 0.0 by the
    same rationale.

    Returns ``(raw, transformed)``. ``raw`` round-trips the input as-is
    (preserving NaN) so the diagnostic log can show the original
    cosine for Phase 4 analytics; ``transformed`` is what the cache
    actually consumes.
    """
    raw = float(replay_grad_cos_rare)
    if math.isnan(raw):
        transformed = 0.0
    else:
        transformed = max(0.0, raw)
    return raw, transformed


def dreamworld_replay_from_cache_entry(
    *,
    model: Any,
    cache: Any,
    slot: int,
    current_step: int,
    weight: float = 1.0,
    lm_head_backward_mode: str = "single",
    lm_head_tile_size: int = 1024,
) -> dict[str, Any] | None:
    """Run one replay backward pass from a curated cache slot.

    Reads ``cache.value_tok_ids[slot]`` + ``cache.value_anchor_id[slot]``,
    builds a synthetic single-row input batch from the value-token span,
    runs forward → fused (or fallback) CE → backward. Gradients
    accumulate into ``param.grad`` so the runner's existing all_group
    SUM all-reduce sweeps them up alongside the train-rank main grads
    (per the topology section of the memory-aware-optimizer plan).

    Returns ``None`` for an unoccupied slot (so the runner's per-step
    drain can race against eviction without crashing) and for a
    too-short value span (``span_length < 2`` leaves no targets after
    the input/target split).

    Returns a dict carrying the diagnostic fields the runner needs to
    write a per-replay log row (Decision 0.9 schema):

    * ``replay_loss`` — scalar CE on the value-token targets.
    * ``replay_grad_norm`` — L2 norm of the replay-only param grads.
    * ``replay_grad_cos_common`` — cosine vs live common-grad direction.
    * ``replay_grad_cos_rare`` — cosine vs live rare-grad direction.
    * ``replay_grad_cos_total`` — cosine vs total grad direction.
    * ``utility_signal_raw`` — signed cosine, before clamp.
    * ``utility_signal_transformed`` — clamped value fed to
      ``cache.update_utility``.

    **Phase 1 simplification (per Decision 0.10):** the three replay-grad
    cosines and ``utility_signal_raw`` return ``NaN`` because no live
    rare-grad EMA is in scope on the episodic rank — ScOpt is gated
    incompatible with episodic, and the train-rank average gradient is
    not available at the point we run replay backward (skip-main on the
    episodic rank means ``param.grad`` was None pre-replay). Phase 4
    will replace the NaN fallback with a proper EMA snapshot once the
    topology stabilizes. ``utility_signal_transformed`` returns 0.0
    deterministically so the EMA in ``cache.update_utility`` stays
    well-defined.

    Replay grad accumulation is additive — pre-existing ``param.grad``
    values are preserved, so multiple drained replay items in one step
    sum into a single SUM all-reduce contribution.
    """
    if not (0 <= int(slot) < int(cache.capacity)):
        # Out-of-range slot is a programming bug, not a race; bail
        # loudly so the controller's invariant gets surfaced.
        raise IndexError(
            f"slot {slot} out of range for cache capacity {cache.capacity}"
        )
    occupied_t = cache.occupied[int(slot)]
    if not bool(occupied_t.item()):
        # Slot evicted between the controller's tag and our drain;
        # nothing to replay.
        return None

    span_length = int(cache.span_length)
    if span_length < 2:
        # No room for an input/target split — the LM-head backward path
        # needs at least one input and one target token. Phase 1's
        # default span_length is 4, so this branch is only hit by
        # pathological configs / unit tests.
        return None

    value_tok_ids = cache.value_tok_ids[int(slot)].detach()
    if value_tok_ids.dim() != 1 or int(value_tok_ids.shape[0]) != span_length:
        raise ValueError(
            f"cache slot {slot} value_tok_ids shape "
            f"{tuple(value_tok_ids.shape)} doesn't match span_length="
            f"{span_length}"
        )

    device = next(model.parameters()).device
    # Single-row replay batch: ``[1, span_length]`` ints. Cast to
    # int32 for the encode call (matches the existing
    # ``dreamworld_replay_backward`` convention) and int64 for targets.
    value_row = value_tok_ids.to(device=device).reshape(1, span_length)
    replay_inputs = value_row[:, :-1].to(torch.int32)
    targets = value_row[:, 1:].to(torch.long)

    hidden = model.encode(replay_inputs)

    mode = str(lm_head_backward_mode).strip().lower()
    if mode == "single" or fused_lm_head_backward is None:
        if full_lm_head_backward is not None:
            loss = full_lm_head_backward(
                hidden,
                model.final_norm,
                model.lm_head,
                targets,
                loss_weight=float(weight),
            )
        else:  # pragma: no cover - import-fallback path
            logits = model.lm_head(model.final_norm(hidden))
            ce = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction="mean",
            )
            (ce * float(weight)).backward()
            loss = ce.detach()
    else:
        if fused_lm_head_backend_for_mode is None:
            raise RuntimeError("fused LM-head backend resolver is unavailable")
        backend_name = fused_lm_head_backend_for_mode(mode)
        loss = fused_lm_head_backward(
            hidden=hidden,
            final_norm=model.final_norm,
            lm_head=model.lm_head,
            targets=targets,
            backend=backend_name,
            tile_size=int(lm_head_tile_size),
            loss_weight=float(weight),
        )

    replay_loss = float(loss.detach().item())
    # Capture the post-replay grad norm BEFORE any caller-side
    # all-reduce so the diagnostic reflects the replay-only contribution.
    grad_norm = _replay_grad_norm_flat(model)

    # Phase 1 simplification (Decision 0.10): no live rare-grad EMA in
    # scope on the episodic rank, so the cosine columns log NaN. Phase
    # 4 will replace the NaN fallback with a proper post-allreduce EMA
    # snapshot once the topology stabilizes (skip-main on the episodic
    # rank means ``param.grad`` was None pre-replay; ScOpt is gated
    # incompatible with episodic so its rare-grad EMA isn't available).
    cos_rare = float("nan")
    raw, transformed = compute_utility_signal(cos_rare)

    return {
        "replay_loss": replay_loss,
        "replay_grad_norm": grad_norm,
        "replay_grad_cos_common": float("nan"),
        "replay_grad_cos_rare": cos_rare,
        "replay_grad_cos_total": float("nan"),
        "utility_signal_raw": raw,
        # ``utility_signal_transformed = max(0.0, cos_rare)`` with NaN
        # → 0.0 keeps the cache utility EMA well-defined. See
        # ``compute_utility_signal``.
        "utility_signal_transformed": transformed,
    }
