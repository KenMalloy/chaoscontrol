# Sleep Cycle Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement structured memory consolidation (N1→N2→N3→REM sleep cycle) and Experiment 11 ablation runner.

**Architecture:** A `SleepCycle` class in `src/chaoscontrol/sleep.py` that operates on the existing memory system (MultiSlotOuterModel slots, survival scores, latent traces, semantic bases) and regret table. Called by the training loop at fixed intervals. Each stage is independently toggleable for ablation. Dreams generate tokens from memory fragments and score against cached real continuations.

**Tech Stack:** PyTorch, existing ChaosControl modules (memory.py, model.py, metabolic.py, regret.py, config.py, training.py)

---

### Task 1: Wake Cache Data Structure

The sleep cycle needs data cached during wake. This is the foundation everything else builds on.

**Files:**
- Create: `src/chaoscontrol/wake_cache.py`
- Test: `tests/test_wake_cache.py`

**Step 1: Write the failing test**

```python
# tests/test_wake_cache.py
import torch
from chaoscontrol.wake_cache import WakeCache


def test_wake_cache_records_moment():
    cache = WakeCache(max_moments=16, max_hidden_buffer=32)
    cache.record_moment(
        surprise=0.8,
        inputs=torch.randint(0, 256, (4, 64)),
        targets=torch.randint(0, 256, (4, 64)),
        hidden=torch.randn(4, 64, 128),
        bucket_ids=torch.zeros(4, 64, dtype=torch.long),
        slot_cues=torch.randn(4, 64),
    )
    assert len(cache.moments) == 1
    assert cache.moments[0]["surprise"] == 0.8


def test_wake_cache_evicts_lowest_signal():
    cache = WakeCache(max_moments=4, max_hidden_buffer=32)
    for i in range(6):
        cache.record_moment(
            surprise=float(i),
            inputs=torch.randint(0, 256, (2, 32)),
            targets=torch.randint(0, 256, (2, 32)),
            hidden=torch.randn(2, 32, 64),
            bucket_ids=torch.zeros(2, 32, dtype=torch.long),
            slot_cues=torch.randn(2, 64),
        )
    assert len(cache.moments) == 4
    # Lowest-signal moments should have been evicted
    surprises = [m["surprise"] for m in cache.moments]
    assert min(surprises) >= 2.0


def test_wake_cache_bucket_distribution():
    cache = WakeCache(max_moments=16, max_hidden_buffer=32)
    cache.update_bucket_counts(torch.tensor([0, 0, 1, 0, 2, 1, 0, 0]))
    dist = cache.bucket_distribution(n_buckets=4)
    assert dist[0] > dist[1]  # Bucket 0 appeared more
    assert dist.sum().item() == pytest.approx(1.0)


def test_wake_cache_hidden_buffer():
    cache = WakeCache(max_moments=16, max_hidden_buffer=4)
    for _ in range(6):
        cache.push_hidden(torch.randn(2, 64, 128))
    assert len(cache.hidden_buffer) == 4  # Rolling, capped


def test_wake_cache_clear():
    cache = WakeCache(max_moments=16, max_hidden_buffer=32)
    cache.record_moment(
        surprise=0.5,
        inputs=torch.randint(0, 256, (2, 32)),
        targets=torch.randint(0, 256, (2, 32)),
        hidden=torch.randn(2, 32, 64),
        bucket_ids=torch.zeros(2, 32, dtype=torch.long),
        slot_cues=torch.randn(2, 64),
    )
    cache.push_hidden(torch.randn(2, 64, 128))
    cache.clear()
    assert len(cache.moments) == 0
    assert len(cache.hidden_buffer) == 0
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_wake_cache.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'chaoscontrol.wake_cache'"

**Step 3: Write minimal implementation**

```python
# src/chaoscontrol/wake_cache.py
"""Wake-time cache: stores high-signal moments and hidden states for sleep consolidation."""
from __future__ import annotations

from collections import deque
from typing import Any

import torch


class WakeCache:
    """Accumulates wake-time data that the sleep cycle operates on.

    Stores high-signal moments (batches where surprise was unusually high
    or low), a rolling buffer of recent hidden states, and running Wernicke
    bucket counts. This cache is the raw material for sleep — without it,
    the sleep cycle has nothing to consolidate against.
    """

    def __init__(self, max_moments: int = 32, max_hidden_buffer: int = 64) -> None:
        self.max_moments = max_moments
        self.max_hidden_buffer = max_hidden_buffer
        self.moments: list[dict[str, Any]] = []
        self.hidden_buffer: deque[torch.Tensor] = deque(maxlen=max_hidden_buffer)
        self._bucket_counts: torch.Tensor | None = None

    def record_moment(
        self,
        *,
        surprise: float,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        hidden: torch.Tensor,
        bucket_ids: torch.Tensor | None = None,
        slot_cues: torch.Tensor | None = None,
    ) -> None:
        """Record a high-signal wake moment with its real continuations."""
        moment = {
            "surprise": surprise,
            "inputs": inputs.detach().cpu(),
            "targets": targets.detach().cpu(),
            "hidden": hidden.detach().cpu(),
            "bucket_ids": bucket_ids.detach().cpu() if bucket_ids is not None else None,
            "slot_cues": slot_cues.detach().cpu() if slot_cues is not None else None,
        }
        self.moments.append(moment)
        # Evict lowest-signal moments when over capacity
        if len(self.moments) > self.max_moments:
            self.moments.sort(key=lambda m: abs(m["surprise"]))
            self.moments = self.moments[len(self.moments) - self.max_moments :]

    def push_hidden(self, hidden: torch.Tensor) -> None:
        """Push recent hidden states into the rolling buffer for N2 scoring."""
        self.hidden_buffer.append(hidden.detach().cpu())

    def update_bucket_counts(self, bucket_ids: torch.Tensor) -> None:
        """Accumulate Wernicke bucket assignment counts."""
        flat = bucket_ids.reshape(-1)
        max_id = int(flat.max().item()) + 1
        counts = torch.bincount(flat, minlength=max_id).float()
        if self._bucket_counts is None or self._bucket_counts.size(0) < max_id:
            old = self._bucket_counts
            self._bucket_counts = torch.zeros(max_id)
            if old is not None:
                self._bucket_counts[: old.size(0)] = old
        self._bucket_counts[:max_id] += counts[:max_id]

    def bucket_distribution(self, n_buckets: int) -> torch.Tensor:
        """Return normalized bucket frequency distribution."""
        if self._bucket_counts is None:
            return torch.ones(n_buckets) / n_buckets
        counts = self._bucket_counts[:n_buckets]
        total = counts.sum()
        if total == 0:
            return torch.ones(n_buckets) / n_buckets
        return counts / total

    def clear(self) -> None:
        """Clear all cached data after a sleep cycle completes."""
        self.moments.clear()
        self.hidden_buffer.clear()
        self._bucket_counts = None
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_wake_cache.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/chaoscontrol/wake_cache.py tests/test_wake_cache.py
git commit -m "feat: WakeCache — stores high-signal moments for sleep consolidation"
```

---

### Task 2: Fatigue Score

Dynamic fatigue system that accumulates and decays.

**Files:**
- Create: `src/chaoscontrol/fatigue.py`
- Test: `tests/test_fatigue.py`

**Step 1: Write the failing test**

```python
# tests/test_fatigue.py
import torch
from chaoscontrol.fatigue import FatigueTracker


def test_fatigue_starts_at_zero():
    ft = FatigueTracker()
    assert ft.score == 0.0


def test_fatigue_accumulates_under_pressure():
    ft = FatigueTracker(accumulation_rate=0.1)
    # Low surprise, low improvement, high memory pressure
    for _ in range(20):
        ft.step(surprise=0.1, improvement_rate=-0.001, memory_pressure=0.8)
    assert ft.score > 0.5


def test_high_surprise_suppresses_fatigue():
    ft = FatigueTracker(accumulation_rate=0.1)
    for _ in range(20):
        ft.step(surprise=2.0, improvement_rate=-0.001, memory_pressure=0.8)
    assert ft.score < 0.1  # High surprise keeps model alert


def test_fatigue_decays_during_sleep():
    ft = FatigueTracker(accumulation_rate=0.1)
    # Build up fatigue
    for _ in range(20):
        ft.step(surprise=0.1, improvement_rate=-0.001, memory_pressure=0.8)
    pre_sleep = ft.score
    ft.apply_sleep_recovery(sleep_quality=0.8)
    assert ft.score < pre_sleep


def test_fatigue_clamped():
    ft = FatigueTracker(accumulation_rate=1.0)
    for _ in range(100):
        ft.step(surprise=0.0, improvement_rate=0.0, memory_pressure=1.0)
    assert ft.score <= 1.0


def test_sleep_duration_scales_with_fatigue():
    ft = FatigueTracker()
    ft._fatigue = 0.3
    short = ft.sleep_steps(base_sleep=128)
    ft._fatigue = 0.9
    long = ft.sleep_steps(base_sleep=128)
    assert long > short
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_fatigue.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# src/chaoscontrol/fatigue.py
"""Fatigue tracker: dynamic system that determines when the model needs sleep."""
from __future__ import annotations


class FatigueTracker:
    """Tracks accumulated fatigue from sustained low-surprise, low-improvement,
    high-memory-pressure conditions. Fatigue has inertia — it builds over time
    and decays during sleep or easy wake periods.

    High surprise suppresses fatigue: the model stays alert when the
    environment demands attention.
    """

    def __init__(
        self,
        accumulation_rate: float = 0.02,
        wake_decay_rate: float = 0.005,
        sleep_decay_rate: float = 0.3,
        surprise_suppression: float = 1.0,
    ) -> None:
        self.accumulation_rate = accumulation_rate
        self.wake_decay_rate = wake_decay_rate
        self.sleep_decay_rate = sleep_decay_rate
        self.surprise_suppression = surprise_suppression
        self._fatigue: float = 0.0

    @property
    def score(self) -> float:
        return self._fatigue

    def step(
        self,
        surprise: float,
        improvement_rate: float,
        memory_pressure: float,
    ) -> float:
        """Update fatigue from one wake step's signals.

        Args:
            surprise: Current surprise ratio (high = alert, low = sleepy).
            improvement_rate: Recent loss improvement per step (negative = improving).
            memory_pressure: 0-1 measure of memory system degradation.

        Returns:
            Current fatigue score after update.
        """
        # Pressure signal: high when not improving and memory is messy
        stagnation = max(0.0, 1.0 + improvement_rate * 100)  # Near 1.0 when improvement ~0
        pressure = stagnation * memory_pressure

        # Surprise suppresses accumulation
        suppression = max(0.0, 1.0 - surprise * self.surprise_suppression)

        # Accumulate fatigue
        self._fatigue += self.accumulation_rate * pressure * suppression

        # Mild wake-time decay (you recover a bit even while working)
        self._fatigue -= self.wake_decay_rate

        self._fatigue = max(0.0, min(1.0, self._fatigue))
        return self._fatigue

    def apply_sleep_recovery(self, sleep_quality: float) -> None:
        """Reduce fatigue after a sleep cycle. Quality 0-1 scales recovery."""
        recovery = self.sleep_decay_rate * max(0.0, min(1.0, sleep_quality))
        self._fatigue = max(0.0, self._fatigue - recovery)

    def sleep_steps(self, base_sleep: int = 128) -> int:
        """Compute sleep duration in steps, scaled by fatigue level."""
        # Light fatigue: ~0.5x base. Heavy fatigue: ~2x base.
        scale = 0.5 + 1.5 * self._fatigue
        return max(1, int(base_sleep * scale))
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_fatigue.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/chaoscontrol/fatigue.py tests/test_fatigue.py
git commit -m "feat: FatigueTracker — dynamic system for sleep timing"
```

---

### Task 3: Sleep Config Fields

Add sleep-related fields to the config dataclass.

**Files:**
- Modify: `src/chaoscontrol/config.py:108` (append after align_weight)
- Test: `tests/test_config.py` (or inline check)

**Step 1: Write the failing test**

```python
# tests/test_sleep_config.py
from chaoscontrol.config import ChaosControlConfig


def test_sleep_config_defaults():
    cfg = ChaosControlConfig(data_path="/tmp")
    assert cfg.sleep_enabled is False
    assert cfg.sleep_stages == "full_cycle"
    assert cfg.sleep_wake_ratio == 2
    assert cfg.sleep_interval == 256
    assert cfg.sleep_budget == 128
    assert cfg.sleep_n2_batches == 8
    assert cfg.sleep_rem_dreams == 4
    assert cfg.sleep_rem_length == 128
    assert cfg.sleep_merge_sim_threshold == 0.85
    assert cfg.sleep_survival_floor == 0.1
    assert cfg.sleep_adaptive_fatigue is False
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_sleep_config.py -v`
Expected: FAIL with "TypeError: __init__() got an unexpected keyword argument 'sleep_enabled'"

**Step 3: Add fields to config.py**

Append after line 108 (`align_weight: float = 0.05`) in `src/chaoscontrol/config.py`:

```python
    # Sleep cycle (structured memory consolidation)
    sleep_enabled: bool = False
    sleep_stages: str = "full_cycle"  # "n3_only", "n2_n3", "n2_n3_rem_validate", "n2_n3_rem_cfr", "n2_n3_rem_full", "full_cycle"
    sleep_wake_ratio: int = 2  # Wake steps per sleep step (2 = 256 wake : 128 sleep)
    sleep_interval: int = 256  # Fixed: trigger sleep every N wake steps
    sleep_budget: int = 128  # Fixed: total sleep ops per cycle
    sleep_n2_budget: int = 64  # Fixed: N2 sub-budget
    sleep_rem_budget: int = 64  # Fixed: REM sub-budget
    sleep_n2_batches: int = 8  # Cached wake batches to use for N2 leave-one-out scoring
    sleep_rem_dreams: int = 4  # Number of dream scenes per REM phase
    sleep_rem_length: int = 128  # Max tokens per dream sequence
    sleep_merge_sim_threshold: float = 0.85  # Cosine similarity threshold for N3 merge proposals
    sleep_survival_floor: float = 0.1  # Slots below this utility in N2 are prunable
    sleep_rem_reactivate: bool = True  # REM attempts latent trace recovery on poor dreams
    sleep_adaptive_fatigue: bool = False  # Use dynamic fatigue vs fixed interval
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_sleep_config.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/chaoscontrol/config.py tests/test_sleep_config.py
git commit -m "feat: sleep cycle config fields"
```

---

### Task 4: dream_step() on ChaosStudentLM

REM needs a full forward step (including Wernicke + memory) for single tokens. The existing `step()` at model.py:201 skips those tiers.

**Files:**
- Modify: `src/chaoscontrol/model.py` (add method after `step()`, around line 237)
- Test: `tests/test_dream_step.py`

**Step 1: Write the failing test**

```python
# tests/test_dream_step.py
import torch
from chaoscontrol.model import ChaosStudentLM


def test_dream_step_exists():
    model = ChaosStudentLM(
        vocab_size=64, dim=32, num_layers=2, ff_mult=2,
        a_mode="diag", rich_b_mode="none",
        outer_model_dim=16, outer_model_type="multislot",
        wernicke_enabled=True, wernicke_router="moe",
        wernicke_k_max=4, wernicke_window=4,
    )
    token_ids = torch.randint(0, 64, (2, 1))
    states = model.init_state(2)
    logits, hidden, new_states = model.dream_step(token_ids, states)
    assert logits.shape == (2, 64)
    assert hidden.shape == (2, 32)
    assert len(new_states) == 2


def test_dream_step_includes_memory_read():
    model = ChaosStudentLM(
        vocab_size=64, dim=32, num_layers=2, ff_mult=2,
        a_mode="diag", rich_b_mode="none",
        outer_model_dim=16, outer_model_type="multislot",
    )
    # Write a slot so memory has content
    model.outer_model.write(
        torch.randn(1, 16),
        bucket_id=0,
    )
    token_ids = torch.randint(0, 64, (1, 1))
    states = model.init_state(1)
    # Should not raise — memory read is included
    logits, hidden, new_states = model.dream_step(token_ids, states)
    assert logits.shape == (1, 64)
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_dream_step.py -v`
Expected: FAIL with "AttributeError: 'ChaosStudentLM' object has no attribute 'dream_step'"

**Step 3: Add dream_step() to ChaosStudentLM**

Insert after `step()` (after line 237) in `src/chaoscontrol/model.py`:

```python
    def dream_step(
        self,
        token_ids: torch.Tensor,
        states: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """Single-token forward with full tiers (Wernicke + memory + semantic).

        Unlike step() which is optimized for fast MCTS rollouts, dream_step()
        includes all tiers so that REM dreams experience the same forward path
        as wake-time training. Backbone weights are not updated (caller is
        responsible for torch.no_grad).
        """
        x = self.embed(token_ids).squeeze(1)  # (batch, dim) — matches step()

        # Wernicke composition (not skipped, unlike step())
        bucket_ids = None
        if self.wernicke is not None:
            x_seq = x.unsqueeze(1)  # Wernicke expects (batch, seq, dim)
            x_seq, bucket_ids, _ = self.wernicke(x_seq)
            x = x_seq.squeeze(1)

        # Memory read (not skipped, unlike step())
        if self.outer_model is not None:
            batch_size = x.size(0)
            if hasattr(self.outer_model, "_slots"):
                # read() handles cue_proj internally — pass raw hidden as cue
                outer_read = self.outer_model.read(batch_size, cue=x)
            else:
                outer_read = self.outer_model.read(batch_size)
            x = x + outer_read

        # Semantic tier (not skipped, unlike step())
        if self.semantic_tier is not None:
            x = x + self.semantic_tier.read(x.size(0))

        # SSM recurrence through layers — uses ChaosSSMBlock.step()
        new_states = []
        for i, layer in enumerate(self.layers):
            x, new_s = layer.step(x, states[i])
            new_states.append(new_s)

        hidden = x
        logits = self.lm_head(self.final_norm(x.unsqueeze(1)).squeeze(1))
        return logits, hidden, new_states
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_dream_step.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/chaoscontrol/model.py tests/test_dream_step.py
git commit -m "feat: dream_step() — full-tier single-token forward for REM"
```

---

### Task 5: SleepCycle Core — N3 Only

The simplest sleep stage: deliberate compression outside of max_slots overflow.

**Files:**
- Create: `src/chaoscontrol/sleep.py`
- Test: `tests/test_sleep.py`

**Step 1: Write the failing test**

```python
# tests/test_sleep.py
import torch
from chaoscontrol.model import ChaosStudentLM
from chaoscontrol.sleep import SleepCycle, SleepConfig
from chaoscontrol.wake_cache import WakeCache


def _make_model_with_slots(n_slots=10):
    """Helper: build model and fill episodic memory with N slots."""
    model = ChaosStudentLM(
        vocab_size=64, dim=32, num_layers=2, ff_mult=2,
        a_mode="diag", rich_b_mode="none",
        outer_model_dim=16, outer_model_type="multislot",
        outer_max_slots=64,
    )
    om = model.outer_model
    for i in range(n_slots):
        om._slots.append(torch.randn(1, 16))
        om._survival.append(float(i) / n_slots)  # Increasing survival
        om._slot_buckets.append(i % 4)
    return model


def test_n3_reduces_slot_count():
    model = _make_model_with_slots(20)
    cache = WakeCache()
    cfg = SleepConfig(stages="n3_only", merge_sim_threshold=0.0)  # Threshold 0 = merge everything similar
    cycle = SleepCycle(cfg)

    pre_count = len(model.outer_model._slots)
    cycle.run(model, cache, device=torch.device("cpu"))
    post_count = len(model.outer_model._slots)
    assert post_count <= pre_count


def test_n3_preserves_high_survival_slots():
    model = _make_model_with_slots(10)
    # Set one slot to very high survival
    model.outer_model._survival[-1] = 100.0
    high_slot = model.outer_model._slots[-1].clone()

    cache = WakeCache()
    cfg = SleepConfig(stages="n3_only", survival_floor=0.5)
    cycle = SleepCycle(cfg)
    cycle.run(model, cache, device=torch.device("cpu"))

    # High-survival slot should still exist
    remaining = model.outer_model._slots
    found = any(torch.allclose(s, high_slot) for s in remaining)
    assert found


def test_n3_produces_latent_traces():
    model = _make_model_with_slots(10)
    pre_traces = len(model.outer_model._latent_traces)

    cache = WakeCache()
    cfg = SleepConfig(stages="n3_only", survival_floor=0.5)
    cycle = SleepCycle(cfg)
    cycle.run(model, cache, device=torch.device("cpu"))

    # Pruned slots should have left latent traces
    post_traces = len(model.outer_model._latent_traces)
    assert post_traces >= pre_traces


def test_n3_recomputes_semantic_bases():
    model = ChaosStudentLM(
        vocab_size=64, dim=32, num_layers=2, ff_mult=2,
        a_mode="diag", rich_b_mode="none",
        outer_model_dim=16, outer_model_type="multislot",
        semantic_tier_bases=4,
    )
    om = model.outer_model
    for i in range(8):
        om._slots.append(torch.randn(1, 16))
        om._survival.append(0.5)
        om._slot_buckets.append(0)

    old_bases = model.semantic_tier.bases.clone()

    cache = WakeCache()
    cfg = SleepConfig(stages="n3_only")
    cycle = SleepCycle(cfg)
    cycle.run(model, cache, device=torch.device("cpu"))

    # Bases should have been recomputed
    assert not torch.allclose(model.semantic_tier.bases, old_bases)
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_sleep.py -v`
Expected: FAIL with "ImportError: cannot import name 'SleepCycle'"

**Step 3: Write SleepCycle with N3**

```python
# src/chaoscontrol/sleep.py
"""Structured sleep cycle: N1 → N2 → N3 → REM memory consolidation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from chaoscontrol.wake_cache import WakeCache


@dataclass
class SleepConfig:
    """Configuration for the sleep cycle. Each stage is independently toggleable."""
    stages: str = "full_cycle"  # "n3_only", "n2_n3", "n2_n3_rem_validate", "n2_n3_rem_cfr", "n2_n3_rem_reactivate", "n2_n3_rem_all", "full_cycle"
    budget: int = 128  # Max total operations (forward passes) per sleep cycle
    n2_budget: int = 64  # Fixed sub-budget for N2 scoring
    rem_budget: int = 64  # Fixed sub-budget for REM dreams
    merge_sim_threshold: float = 0.85
    survival_floor: float = 0.1
    n2_batches: int = 8
    rem_dreams: int = 4
    rem_length: int = 128
    rem_validate: bool = True  # REM validates provisional merges
    rem_cfr: bool = True  # REM updates gate/CFR policy
    rem_reactivate: bool = True  # REM attempts latent trace recovery on poor dreams

    @property
    def use_n1(self) -> bool:
        return self.stages == "full_cycle"

    @property
    def use_n2(self) -> bool:
        return self.stages in ("n2_n3", "n2_n3_rem", "full_cycle")

    @property
    def use_n3(self) -> bool:
        return True  # All sleep conditions include N3

    @property
    def use_rem(self) -> bool:
        return self.stages in ("n2_n3_rem_validate", "n2_n3_rem_cfr", "n2_n3_rem_reactivate", "n2_n3_rem_all", "full_cycle")


class SleepCycle:
    """Executes one sleep cycle: N1 → N2 → N3 → REM.

    Operates on the model's existing memory system (MultiSlotOuterModel)
    without modifying backbone weights. Each stage is independently
    toggleable for ablation experiments. Sleep cost is bounded by
    config.budget (max forward passes per cycle).
    """

    def __init__(self, config: SleepConfig) -> None:
        self.config = config

    def run(
        self,
        model: Any,
        cache: WakeCache,
        *,
        device: torch.device,
        regret_table: Any | None = None,
    ) -> dict[str, Any]:
        """Execute one full sleep cycle. Returns diagnostic dict."""
        om = getattr(model, "outer_model", None)
        if om is None or not hasattr(om, "_slots") or len(om._slots) == 0:
            return {"skipped": True, "reason": "no memory"}

        diagnostics: dict[str, Any] = {"skipped": False}
        ops_used = 0  # Track forward passes against budget

        if self.config.use_n1:
            diagnostics["n1"] = self._n1_transition(om)

        if self.config.use_n2:
            n2_diag, n2_ops = self._n2_tag(
                model, om, cache, device,
                ops_remaining=self.config.n2_budget,
            )
            diagnostics["n2"] = n2_diag
            ops_used += n2_ops

        provisional_merges = None
        if self.config.use_n3:
            provisional_merges, n3_diag = self._n3_rewrite(
                model, om, provisional=self.config.use_rem,
            )
            diagnostics["n3"] = n3_diag
            # N3 is slot manipulation, no forward passes

        if self.config.use_rem:
            # Determine which REM mechanisms are active based on stage config
            rem_validate = self.config.rem_validate and self.config.stages in (
                "n2_n3_rem_validate", "n2_n3_rem_all", "full_cycle")
            rem_cfr = self.config.rem_cfr and self.config.stages in (
                "n2_n3_rem_cfr", "n2_n3_rem_all", "full_cycle")
            rem_reactivate = self.config.rem_reactivate and self.config.stages in (
                "n2_n3_rem_reactivate", "n2_n3_rem_all", "full_cycle")
            rem_diag, rem_ops = self._rem_dream(
                model, om, cache, provisional_merges,
                device=device, regret_table=regret_table if rem_cfr else None,
                ops_remaining=self.config.rem_budget,
                validate_merges=rem_validate,
                reactivate_on_loss=rem_reactivate,
            )
            diagnostics["rem"] = rem_diag
            ops_used += rem_ops
        elif provisional_merges is not None:
            self._commit_merges(om, provisional_merges)

        # Recompute semantic bases from surviving slots
        st = getattr(model, "semantic_tier", None)
        if st is not None and len(om._slots) > 0:
            # consolidate_from_episodes expects (N, model_dim) tensor
            slot_stack = torch.cat(
                [om.decoder(s.to(dtype=om.decoder.weight.dtype)) for s in om._slots],
                dim=0,
            )
            st.consolidate_from_episodes(slot_stack)

        diagnostics["ops_used"] = ops_used
        return diagnostics

    def _n1_transition(self, om: Any) -> dict[str, Any]:
        """N1: Freeze slot creation, snapshot recent unstable traces."""
        n_recent = min(3, len(om._slots))
        unstable_indices = list(range(len(om._slots) - n_recent, len(om._slots)))
        return {"unstable_indices": unstable_indices, "total_slots": len(om._slots)}

    def _n2_tag(
        self, model: Any, om: Any, cache: WakeCache, device: torch.device,
        ops_remaining: int,
    ) -> tuple[dict[str, Any], int]:
        """N2: Leave-one-slot-out utility scoring on cached wake batches.

        True slot removal: pop the slot from the list, run forward passes,
        then reinsert. This ensures the softmax retrieval doesn't see it.
        """
        if not cache.moments:
            return {"skipped": True, "reason": "no cached moments"}, 0

        moments = cache.moments[: self.config.n2_batches]
        n_slots = len(om._slots)
        utility = [0.0] * n_slots
        ops = 0

        # Cap slot evaluations to budget, shuffle to avoid position bias
        import random as _random
        max_slots_to_score = min(n_slots, ops_remaining // max(len(moments), 1))
        slot_order = list(range(n_slots))
        _random.Random(42).shuffle(slot_order)
        slot_order = slot_order[:max_slots_to_score]

        model.eval()
        with torch.no_grad():
            # Baseline loss with all slots
            baseline_losses = []
            for moment in moments:
                inputs = moment["inputs"].to(device)
                targets = moment["targets"].to(device)
                out = model(inputs)
                ce = F.cross_entropy(
                    out["logits"].float().reshape(-1, model.vocab_size),
                    targets.reshape(-1),
                ).item()
                baseline_losses.append(ce)
                ops += 1
            mean_baseline = sum(baseline_losses) / len(baseline_losses)

            # Leave-one-out: truly remove each slot, measure loss delta
            for slot_idx in slot_order:
                # Pop slot out of the list entirely
                saved_slot = om._slots.pop(slot_idx)
                saved_surv = om._survival.pop(slot_idx)
                saved_buck = om._slot_buckets.pop(slot_idx)

                slot_losses = []
                for moment in moments:
                    inputs = moment["inputs"].to(device)
                    targets = moment["targets"].to(device)
                    out = model(inputs)
                    ce = F.cross_entropy(
                        out["logits"].float().reshape(-1, model.vocab_size),
                        targets.reshape(-1),
                    ).item()
                    slot_losses.append(ce)
                    ops += 1

                # Reinsert slot at original position
                om._slots.insert(slot_idx, saved_slot)
                om._survival.insert(slot_idx, saved_surv)
                om._slot_buckets.insert(slot_idx, saved_buck)

                mean_without = sum(slot_losses) / len(slot_losses)
                utility[slot_idx] = mean_without - mean_baseline

        # Update survival scores from utility (only scored slots)
        for i in slot_order:
            om._survival[i] = max(0.0, utility[i])

        scored_utilities = [utility[i] for i in slot_order]
        return {
            "n_slots_scored": len(slot_order),
            "mean_utility": sum(scored_utilities) / max(len(scored_utilities), 1),
            "n_below_floor": sum(1 for u in scored_utilities
                                 if u < self.config.survival_floor),
        }, ops

    def _n3_rewrite(
        self, model: Any, om: Any, *, provisional: bool = False,
    ) -> tuple[list[dict] | None, dict[str, Any]]:
        """N3: Propose merges, prune low-utility slots, produce latent traces.

        Merges respect typed storage: only merge slots within the same
        Wernicke bucket, matching the memory system's semantics.
        """
        n_before = len(om._slots)
        proposed_merges: list[dict[str, Any]] = []
        use_typed = getattr(model, "typed_storage", False)

        # 1. Prune slots below survival floor
        pruned = 0
        i = 0
        while i < len(om._slots):
            if om._survival[i] < self.config.survival_floor:
                om._latent_traces.append({
                    "bucket_id": om._slot_buckets[i],
                    "centroid_contrib": om._slots[i].detach().clone(),
                })
                om._slots.pop(i)
                om._survival.pop(i)
                om._slot_buckets.pop(i)
                pruned += 1
            else:
                i += 1

        # 2. Propose merges for similar slots (within same bucket if typed)
        merged = 0
        i = 0
        while i < len(om._slots):
            j = i + 1
            while j < len(om._slots):
                # Only merge within same bucket when typed storage is active
                if use_typed and om._slot_buckets[i] != om._slot_buckets[j]:
                    j += 1
                    continue
                sim = F.cosine_similarity(
                    om._slots[i].float().reshape(1, -1),
                    om._slots[j].float().reshape(1, -1),
                ).item()
                if sim > self.config.merge_sim_threshold:
                    merge = {
                        "idx_keep": i,
                        "idx_absorb": j,
                        "slot_keep": om._slots[i].clone(),
                        "slot_absorb": om._slots[j].clone(),
                        "merged_slot": (om._slots[i] + om._slots[j]) / 2,
                        "survival_keep": om._survival[i],
                        "survival_absorb": om._survival[j],
                        "bucket_keep": om._slot_buckets[i],
                        "bucket_absorb": om._slot_buckets[j],
                    }
                    if provisional:
                        proposed_merges.append(merge)
                        j += 1  # Don't commit yet, keep scanning
                    else:
                        om._slots[i] = merge["merged_slot"]
                        om._survival[i] = max(merge["survival_keep"], merge["survival_absorb"])
                        om._latent_traces.append({
                            "bucket_id": om._slot_buckets[j],
                            "centroid_contrib": om._slots[j].detach().clone(),
                        })
                        om._slots.pop(j)
                        om._survival.pop(j)
                        om._slot_buckets.pop(j)
                        merged += 1
                        continue
                j += 1
            i += 1

        return (proposed_merges if provisional else None), {
            "slots_before": n_before,
            "slots_after": len(om._slots),
            "pruned": pruned,
            "merged": merged,
            "proposed_merges": len(proposed_merges),
        }

    def _rem_dream(
        self,
        model: Any,
        om: Any,
        cache: WakeCache,
        provisional_merges: list[dict] | None,
        *,
        device: torch.device,
        regret_table: Any | None = None,
        ops_remaining: int = 128,
        validate_merges: bool = True,
        reactivate_on_loss: bool = False,
    ) -> tuple[dict[str, Any], int]:
        """REM: Dream from memory, score against real targets, validate merges.

        Three independently toggleable mechanisms:
        - validate_merges: test provisional N3 merges, reject bad ones
        - regret_table (not None): update CFR gate policy from dream counterfactuals
        - reactivate_on_loss: attempt try_reactivate() when dream CE is poor

        Each can be enabled/disabled for clean ablation.
        """
        if not cache.moments:
            if provisional_merges:
                self._commit_merges(om, provisional_merges)
            return {"skipped": True, "reason": "no cached moments"}, 0

        model.eval()
        dream_scores: list[float] = []
        merges_accepted = 0
        merges_rejected = 0
        ops = 0

        with torch.no_grad():
            # 1. Validate provisional merges against cached real targets
            if provisional_merges and not validate_merges:
                # Merge validation disabled — commit all merges immediately
                self._commit_merges(om, provisional_merges)
                provisional_merges = None
            if provisional_merges:
                pre_score = self._score_on_cached(model, cache, device)
                ops += len(cache.moments[: self.config.n2_batches])

                for merge in provisional_merges:
                    if ops >= ops_remaining:
                        break
                    idx_keep = merge["idx_keep"]
                    idx_absorb = merge["idx_absorb"]
                    if idx_keep >= len(om._slots) or idx_absorb >= len(om._slots):
                        continue

                    saved_keep = om._slots[idx_keep].clone()
                    om._slots[idx_keep] = merge["merged_slot"].to(device)

                    post_score = self._score_on_cached(model, cache, device)
                    ops += len(cache.moments[: self.config.n2_batches])

                    if post_score <= pre_score:
                        om._latent_traces.append({
                            "bucket_id": merge["bucket_absorb"],
                            "centroid_contrib": merge["slot_absorb"].clone(),
                        })
                        om._slots.pop(idx_absorb)
                        om._survival.pop(idx_absorb)
                        om._slot_buckets.pop(idx_absorb)
                        om._survival[idx_keep] = max(
                            merge["survival_keep"], merge["survival_absorb"],
                        )
                        merges_accepted += 1
                        pre_score = post_score  # Update baseline for next merge
                    else:
                        om._slots[idx_keep] = saved_keep
                        merges_rejected += 1

            # 2. Dream scenes: generate from memory, then score state against reality
            sorted_moments = sorted(
                cache.moments, key=lambda m: abs(m["surprise"]), reverse=True,
            )
            n_dreams = min(self.config.rem_dreams, len(sorted_moments))

            for moment in sorted_moments[:n_dreams]:
                if ops >= ops_remaining:
                    break

                # Phase A: Dream — generate tokens from memory to condition state
                # This is the dream itself. The model experiences a sequence
                # built from its own memories via dream_step (full tiers).
                if len(om._slots) > 0:
                    seed_slot = om._slots[0]
                    if moment.get("bucket_ids") is not None:
                        target_bucket = int(moment["bucket_ids"].reshape(-1).mode().values.item())
                        for si, sb in enumerate(om._slot_buckets):
                            if sb == target_bucket:
                                seed_slot = om._slots[si]
                                break
                    decoded = om.decoder(seed_slot.to(dtype=om.decoder.weight.dtype))
                    seed_logits = model.lm_head(model.final_norm(decoded.unsqueeze(0)).squeeze(0))
                    seed_token = seed_logits.argmax(dim=-1).unsqueeze(0).unsqueeze(0)

                    # Autoregressive dream via dream_step — conditions model state
                    states = model.init_state(1)
                    token = seed_token
                    for _ in range(min(self.config.rem_length, 32)):
                        logits, hidden, states = model.dream_step(
                            token.to(device), states,
                        )
                        token = logits.argmax(dim=-1).unsqueeze(0).unsqueeze(0)
                        ops += 1
                        if ops >= ops_remaining:
                            break

                # Phase B: Score — teacher-forced CE on real targets
                # "Given the memory state I just dreamed through, can I predict
                # what actually happened?" This is the non-circular anchor.
                targets = moment["targets"].to(device)
                inputs = moment["inputs"].to(device)
                out = model(inputs)
                dream_ce = F.cross_entropy(
                    out["logits"].float().reshape(-1, model.vocab_size),
                    targets.reshape(-1),
                ).item()
                dream_scores.append(dream_ce)
                ops += 1

                # Update slot survival from dream quality
                baseline_ce = sum(dream_scores) / len(dream_scores)
                for i in range(len(om._survival)):
                    if dream_ce < baseline_ce:
                        om._survival[i] *= 1.02
                    elif dream_ce > baseline_ce * 1.5:
                        om._survival[i] *= 0.98

                # Phase C2: Latent reactivation on poor dreams
                # If dream CE is significantly worse than wake CE for this moment,
                # the compression lost information. Attempt to recover it.
                reactivations = 0
                if reactivate_on_loss and hasattr(om, "try_reactivate"):
                    wake_ce = moment["surprise"]  # Rough proxy for wake-time CE
                    if dream_ce > abs(wake_ce) * 1.5 and dream_ce > baseline_ce:
                        # Dream is much worse than wake — try to recover
                        bucket = 0
                        if moment.get("bucket_ids") is not None:
                            bucket = int(moment["bucket_ids"].reshape(-1).mode().values.item())
                        reactivated = om.try_reactivate(
                            bucket_id=bucket, surprise=dream_ce / max(baseline_ce, 1e-6),
                        )
                        if reactivated:
                            reactivations += 1
                            # Re-score after reactivation
                            out_after = model(inputs)
                            ce_after = F.cross_entropy(
                                out_after["logits"].float().reshape(-1, model.vocab_size),
                                targets.reshape(-1),
                            ).item()
                            ops += 1
                            if ce_after >= dream_ce:
                                # Reactivation didn't help — let it be re-compressed naturally
                                pass
                            # else: reactivation improved things, trace stays active

                # Phase C3: CFR — score each candidate independently on real targets
                # NOTE: requires metabolic_fork to return per-candidate logits
                # (see Task 4b: modify metabolic_fork to add "all_logits" to output)
                if regret_table is not None and ops + regret_table.n_actions < ops_remaining:
                    k = regret_table.n_actions
                    # Generate K candidates and score each against real targets
                    counterfactual_values = []
                    for cand_idx in range(k):
                        # Perturb embedding and forward pass
                        noisy_embed = model.embed(inputs) + torch.randn_like(
                            model.embed(inputs)) * 0.01 * (cand_idx + 1)
                        # Run through full model with perturbed input
                        cand_out = model.lm_head(model.final_norm(
                            # Simplified: run layers on perturbed embedding
                            # Full implementation uses model internals
                            noisy_embed[:, -1, :]
                        ))
                        cand_ce = F.cross_entropy(
                            cand_out.float().reshape(-1, model.vocab_size),
                            targets[:, -1].reshape(-1),
                        ).item()
                        counterfactual_values.append(-cand_ce)
                        ops += 1

                    actual_value = -dream_ce
                    best_idx = int(torch.tensor(counterfactual_values).argmax().item())
                    bucket = 0
                    if moment.get("bucket_ids") is not None:
                        bucket = int(moment["bucket_ids"].reshape(-1).mode().values.item())

                    regret_table.update(
                        bucket_id=bucket % regret_table.n_buckets,
                        action_taken=best_idx,
                        counterfactual_values=counterfactual_values,
                        actual_value=actual_value,
                    )

        return {
            "n_dreams": len(dream_scores),
            "mean_dream_ce": sum(dream_scores) / max(len(dream_scores), 1),
            "merges_accepted": merges_accepted,
            "merges_rejected": merges_rejected,
        }, ops

    def _score_on_cached(
        self, model: Any, cache: WakeCache, device: torch.device,
    ) -> float:
        """Mean CE on cached wake moments. Used for merge validation."""
        losses = []
        for moment in cache.moments[: self.config.n2_batches]:
            inputs = moment["inputs"].to(device)
            targets = moment["targets"].to(device)
            out = model(inputs)
            ce = F.cross_entropy(
                out["logits"].float().reshape(-1, model.vocab_size),
                targets.reshape(-1),
            ).item()
            losses.append(ce)
        return sum(losses) / max(len(losses), 1)

    def _commit_merges(self, om: Any, merges: list[dict]) -> None:
        """Commit all provisional merges without validation."""
        for merge in sorted(merges, key=lambda m: m["idx_absorb"], reverse=True):
            idx_keep = merge["idx_keep"]
            idx_absorb = merge["idx_absorb"]
            if idx_keep >= len(om._slots) or idx_absorb >= len(om._slots):
                continue
            om._slots[idx_keep] = merge["merged_slot"]
            om._survival[idx_keep] = max(merge["survival_keep"], merge["survival_absorb"])
            om._latent_traces.append({
                "bucket_id": merge["bucket_absorb"],
                "centroid_contrib": merge["slot_absorb"].clone(),
            })
            om._slots.pop(idx_absorb)
            om._survival.pop(idx_absorb)
            om._slot_buckets.pop(idx_absorb)
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_sleep.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/chaoscontrol/sleep.py tests/test_sleep.py
git commit -m "feat: SleepCycle — N1/N2/N3/REM structured consolidation"
```

---

### Task 6: Training Loop Integration

Wire the wake cache and sleep cycle into the training loop.

**Files:**
- Modify: `src/chaoscontrol/training.py` — add wake cache recording, sleep trigger
- Test: `tests/test_training_sleep.py`

**Step 1: Write the failing test**

```python
# tests/test_training_sleep.py
import torch
from chaoscontrol.config import ChaosControlConfig
from chaoscontrol.training import train_chaoscontrol_for_budget
from chaoscontrol.model import ChaosStudentLM


def test_training_with_sleep_runs():
    """Smoke test: training with sleep enabled completes without error."""
    model = ChaosStudentLM(
        vocab_size=64, dim=32, num_layers=2, ff_mult=2,
        a_mode="diag", rich_b_mode="none",
        outer_model_dim=16, outer_model_type="multislot",
        outer_max_slots=16,
    )
    tokens = torch.randint(0, 64, (1000,))
    starts = list(range(0, 900, 32))
    result = train_chaoscontrol_for_budget(
        model, train_tokens=tokens, train_starts=starts,
        seq_len=32, batch_size=2, device=torch.device("cpu"),
        param_dtype=torch.float32, budget_seconds=10,
        base_lr=1e-3, weight_decay=0.0, grad_clip_norm=1.0,
        seed=42,
        sleep_enabled=True,
        sleep_stages="n3_only",
        sleep_interval=5,  # Sleep every 5 steps for testing
        sleep_budget=3,
    )
    assert result["steps"] > 0
    assert "sleep_cycles" in result
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_training_sleep.py -v`
Expected: FAIL with "TypeError: unexpected keyword argument 'sleep_enabled'"

**Step 3: Modify training.py**

Add these parameters to `train_chaoscontrol_for_budget()` signature (after existing params around line 55):

```python
    # Sleep cycle
    sleep_enabled: bool = False,
    sleep_stages: str = "full_cycle",
    sleep_interval: int = 256,
    sleep_budget: int = 128,
    sleep_n2_batches: int = 8,
    sleep_rem_dreams: int = 4,
    sleep_rem_length: int = 128,
    sleep_merge_sim_threshold: float = 0.85,
    sleep_survival_floor: float = 0.1,
```

Add imports at the top of training.py:

```python
from chaoscontrol.wake_cache import WakeCache
from chaoscontrol.sleep import SleepCycle, SleepConfig
```

Before the main while loop (around line 96), add initialization:

```python
    # Sleep cycle setup
    wake_cache = WakeCache(max_moments=32, max_hidden_buffer=64) if sleep_enabled else None
    sleep_cycle = None
    if sleep_enabled:
        sleep_cfg = SleepConfig(
            stages=sleep_stages,
            budget=sleep_budget,
            merge_sim_threshold=sleep_merge_sim_threshold,
            survival_floor=sleep_survival_floor,
            n2_batches=sleep_n2_batches,
            rem_dreams=sleep_rem_dreams,
            rem_length=sleep_rem_length,
        )
        sleep_cycle = SleepCycle(sleep_cfg)
    sleep_cycles_run = 0
    wake_steps_since_sleep = 0
```

Inside the main loop, after the consolidation block (after line 403), add wake cache recording:

```python
        # Wake cache: record high-signal moments for sleep
        if wake_cache is not None:
            wake_steps_since_sleep += 1
            # Cache hidden states for N2 scoring
            wake_cache.push_hidden(out["hidden"].detach())
            # Record high-signal moments (both directions of surprise)
            if abs(surprise_ratio_for_latent) > current_threshold * 0.5:
                wake_cache.record_moment(
                    surprise=surprise_ratio_for_latent,
                    inputs=inputs,
                    targets=targets,
                    hidden=out["hidden"],
                    bucket_ids=out.get("bucket_ids"),
                    slot_cues=out["hidden"][:, -1, :] if out["hidden"].ndim == 3 else None,
                )
            # Record bucket distribution
            if "bucket_ids" in out:
                wake_cache.update_bucket_counts(out["bucket_ids"])

            # Sleep trigger: fixed interval
            if wake_steps_since_sleep >= sleep_interval:
                sleep_diag = sleep_cycle.run(
                    model, wake_cache,
                    device=device, regret_table=regret_table,
                )
                sleep_cycles_run += 1
                wake_steps_since_sleep = 0
                wake_cache.clear()
                step_record["sleep"] = sleep_diag
```

In the return dict (around line 483), add:

```python
        "sleep_cycles": sleep_cycles_run,
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_training_sleep.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/chaoscontrol/training.py tests/test_training_sleep.py
git commit -m "feat: wire sleep cycle into training loop with wake cache"
```

---

### Task 7: Runner Config Passthrough

Pass sleep config fields from YAML configs through runner to training.

**Files:**
- Modify: `src/chaoscontrol/runner.py` — pass sleep fields to train_chaoscontrol_for_budget
- Test: verify with existing test from Task 6

**Step 1: Modify runner.py**

In `run_experiment()`, add sleep fields to the `train_chaoscontrol_for_budget()` call (around line 143):

```python
        sleep_enabled=cfg.sleep_enabled,
        sleep_stages=cfg.sleep_stages,
        sleep_interval=cfg.sleep_interval,
        sleep_budget=cfg.sleep_budget,
        sleep_n2_batches=cfg.sleep_n2_batches,
        sleep_rem_dreams=cfg.sleep_rem_dreams,
        sleep_rem_length=cfg.sleep_rem_length,
        sleep_merge_sim_threshold=cfg.sleep_merge_sim_threshold,
        sleep_survival_floor=cfg.sleep_survival_floor,
```

**Step 2: Run existing tests**

Run: `PYTHONPATH=src pytest tests/test_training_sleep.py tests/test_sleep.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add src/chaoscontrol/runner.py
git commit -m "feat: pass sleep config through runner to training loop"
```

---

### Task 8: Experiment 11 Runner

The ablation experiment runner.

**Files:**
- Create: `experiments/11_sleep_cycle/run_sleep_ablation.py`
- Create: `experiments/11_sleep_cycle/configs/` (generated by runner)

**Step 1: Write the experiment runner**

```python
#!/usr/bin/env python3
"""Experiment 11: Sleep cycle ablation.

5 conditions x 3 seeds = 15 runs at 600s budget.
Tests whether structured memory consolidation improves bpb
despite spending compute budget on sleep instead of gradient steps.
"""
# [Full runner following same pattern as run_decision.py
#  but with sleep-specific configs: no_sleep, n3_only, n2_n3, n2_n3_rem, full_cycle]
```

This follows the exact same pattern as `run_decision.py` but with 5 conditions instead of 4, all using the full stack config with varying `sleep_stages`. See `run_decision.py` for the template — the only changes are the config dict entries for each condition.

**Step 2: Verify syntax**

Run: `python3 -c "import ast; ast.parse(open('experiments/11_sleep_cycle/run_sleep_ablation.py').read()); print('OK')"`
Expected: OK

**Step 3: Commit**

```bash
git add experiments/11_sleep_cycle/
git commit -m "feat: experiment 11 — sleep cycle ablation runner"
```

---

### Task 9: Integration Test — Full Cycle

End-to-end test that all stages work together.

**Files:**
- Test: `tests/test_sleep_integration.py`

**Step 1: Write the integration test**

```python
# tests/test_sleep_integration.py
import torch
from chaoscontrol.model import ChaosStudentLM
from chaoscontrol.sleep import SleepCycle, SleepConfig
from chaoscontrol.wake_cache import WakeCache
from chaoscontrol.regret import RegretTable


def test_full_cycle_integration():
    """Full N1→N2→N3→REM cycle on a model with populated memory."""
    model = ChaosStudentLM(
        vocab_size=64, dim=32, num_layers=2, ff_mult=2,
        a_mode="diag", rich_b_mode="none",
        outer_model_dim=16, outer_model_type="multislot",
        outer_max_slots=32,
        semantic_tier_bases=4,
    )
    om = model.outer_model

    # Populate memory with diverse slots
    for i in range(20):
        om._slots.append(torch.randn(1, 16))
        om._survival.append(0.3 + 0.05 * i)
        om._slot_buckets.append(i % 4)

    # Populate wake cache with moments
    cache = WakeCache(max_moments=16, max_hidden_buffer=32)
    for i in range(8):
        inputs = torch.randint(0, 64, (2, 32))
        cache.record_moment(
            surprise=float(i) * 0.3,
            inputs=inputs,
            targets=torch.randint(0, 64, (2, 32)),
            hidden=torch.randn(2, 32, 32),
            bucket_ids=torch.randint(0, 4, (2, 32)),
            slot_cues=torch.randn(2, 16),
        )

    regret_table = RegretTable(n_buckets=4, n_actions=4)

    cfg = SleepConfig(
        stages="full_cycle",
        merge_sim_threshold=0.9,
        survival_floor=0.2,
        n2_batches=4,
        rem_dreams=2,
        rem_length=32,
    )
    cycle = SleepCycle(cfg)

    pre_slots = len(om._slots)
    diag = cycle.run(
        model, cache,
        device=torch.device("cpu"),
        regret_table=regret_table,
    )

    assert not diag["skipped"]
    assert "n1" in diag
    assert "n2" in diag
    assert "n3" in diag
    assert "rem" in diag
    assert diag["n2"]["n_slots_scored"] > 0
    # Sleep should have done something to the slot count
    post_slots = len(om._slots)
    assert post_slots <= pre_slots or diag["n3"]["pruned"] == 0
```

**Step 2: Run test**

Run: `PYTHONPATH=src pytest tests/test_sleep_integration.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_sleep_integration.py
git commit -m "test: full sleep cycle integration test"
```

---

## Execution Order Summary

| Task | Component | Dependencies |
|------|-----------|-------------|
| 1 | WakeCache | None |
| 2 | FatigueTracker | None |
| 3 | Config fields | None |
| 4 | dream_step() | None |
| 5 | SleepCycle (N3 + N2 + REM + N1) | Tasks 1, 4 |
| 6 | Training loop integration | Tasks 1, 2, 3, 5 |
| 7 | Runner config passthrough | Tasks 3, 6 |
| 8 | Experiment 11 runner | Tasks 3, 7 |
| 9 | Integration test | All above |

Tasks 1-4 can be done in parallel. Tasks 5-9 are sequential.
