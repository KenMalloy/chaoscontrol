"""Primitive Dreamworld helpers for Experiment 24.

This module intentionally provides just the small, testable building blocks
needed for the first phase of DREAM replay work:

* a bounded replay buffer with hard age-out and diagnostics,
* replay-window extraction utilities,
* entry capture using cached pre-replay state, and
* a single teacher-forced replay backward pass.
"""

from dataclasses import dataclass
from typing import Any

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
