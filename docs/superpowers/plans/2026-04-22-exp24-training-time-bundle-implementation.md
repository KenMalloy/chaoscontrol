# Exp24 Training-Time Bundle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Status update, 2026-04-22:** the implemented first-wave matrix now includes
> scheduled Dreamworld replay, a fast/slow+Dreamworld stack, and an
> `event_sleep` loss-triggered Dreamworld replay arm. `event_sleep` is the
> project nickname for a DDP-summed loss-pressure gate; it is not rank-local
> partitioned sleep and it does not let one GPU take private optimizer steps.
> Treat the checked-in source/tests as authority where this historical plan
> shows earlier SGNS/SemanticOptimizer ordering.

**Goal:** Build a tested Exp24 fast-path training harness that can run apples-to-apples 600s training-time mechanism ablations on the current fastest SSM base.

**Architecture:** Keep Exp23 as the hot-loop substrate and add Exp24 as a thin experiment layer around it. New matrix/build/report code lives under `experiments/24_training_time_bundle/`; reusable low-level hooks go beside the Exp23 runner because they are training-loop capabilities, not one-off analysis scripts.

**Tech Stack:** Python 3.12, PyTorch, existing Exp23 fast-path runner, existing `_ssm_scan` and fused linear-CE kernels, pytest.

---

## File Structure

- Create `experiments/24_training_time_bundle/exp24.py`: Exp24 matrix builders, artifact-impact tags, seed-noise summary helpers, and result-row normalization.
- Create `experiments/24_training_time_bundle/run_exp24.py`: CLI for Ring 0, Phase A, and named mechanism matrices using the existing `launch.run_matrix_entries`.
- Create `experiments/24_training_time_bundle/README.md`: runbook for local dry runs, 1xH100 smokes, 8xH100 gates, and result interpretation.
- Create `experiments/23_fast_path/dreamworld.py`: Dreamworld ring-buffer state replay primitives for the fast-path runner.
- Create `tests/test_exp24_training_bundle.py`: tests for Exp24 matrix construction and summaries.
- Create `tests/test_exp24_training_hooks.py`: tests for fast/slow, spectral, embedding-freeze, and predictive-aux hook behavior.
- Create `tests/test_exp24_dreamworld.py`: tests for Dreamworld buffer, state capture, and replay loss.
- Modify `src/chaoscontrol/model.py`: let `ChaosStudentLM.encode()` return final SSM states and accept cached initial states without materializing logits.
- Modify `experiments/23_fast_path/fast_path.py`: add O(1) shuffled-epoch sampling helpers and matrix-friendly defaults.
- Modify `experiments/23_fast_path/runner_fast_path.py`: thread training-time mechanisms through `_run_train_step`, `train_fast_for_budget`, `_build_optimizer`, and result JSON.
- Modify `tests/test_exp23_fast_path.py`: pin new sampling mode and runner integration behavior.
- Modify `tests/test_encode_forward_equivalence.py`: pin state-returning encode equivalence against `forward()`.
- Modify `docs/plans/2026-04-22-exp24-training-time-bundle.md`: link to this implementation plan after code lands.

## Shared Config Keys

Use these exact config keys across matrix builders and runner code:

```yaml
exp24_phase: ring0
exp24_mechanism: control
artifact_impact: artifact_changes_weights_only
submit_valid: true
train_sampling_mode: random
fast_slow_enabled: false
fast_slow_interval: 0
fast_slow_alpha: 0.0
fast_slow_eval_copy: fast
spectral_reg_lambda_dead: 0.0
spectral_reg_lambda_sticky: 0.0
spectral_reg_min_a: 0.05
spectral_reg_max_a: 0.98
predictive_aux_weight: 0.0
predictive_aux_horizon: 0
predictive_aux_dim: 0
embed_freeze_steps: 0
dreamworld_enabled: false
dreamworld_cache_interval: 0
dreamworld_interval: 0
dreamworld_weight: 0.0
dreamworld_prefix_tokens: 128
dreamworld_replay_tokens: 64
dreamworld_buffer_size: 16
dreamworld_min_size: 2
dreamworld_max_age_steps: 256
event_sleep_enabled: false
event_sleep_loss_ratio: 1.10
event_sleep_pressure_threshold: 0.05
event_sleep_ema_decay: 0.99
event_sleep_warmup_steps: 32
event_sleep_min_interval: 8
event_sleep_weight: 0.0
semantic_layer_index: 0
semantic_momentum_min: 0.5
semantic_overhead_gate: 0.08
```

`budget_seconds` remains the actual wall-clock training budget. Exp24 matrix builders should set it to `600.0` for Ring 0 and mechanism runs unless a caller explicitly overrides it for a smoke.

## Task 1: Exp24 Matrix And Noise-Floor Scaffold

**Files:**
- Create: `experiments/24_training_time_bundle/exp24.py`
- Create: `tests/test_exp24_training_bundle.py`

- [ ] **Step 1: Write the failing tests**

Add this test file:

```python
"""Tests for Exp24 training-time bundle matrix helpers."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
EXP24_PATH = REPO / "experiments" / "24_training_time_bundle" / "exp24.py"


def _load_exp24():
    spec = importlib.util.spec_from_file_location("exp24_training_bundle", EXP24_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_ring0_control_matrix_uses_seed_ladder_and_600s_budget():
    mod = _load_exp24()

    entries = mod.build_ring0_control_matrix(
        speed_config={"batch_size": 1024, "chunk_size": 64},
        world_size=8,
    )

    assert [entry["seed"] for entry in entries] == [1337, 2674, 4011]
    assert [entry["name"] for entry in entries] == [
        "exp24_ring0_control_s1337",
        "exp24_ring0_control_s2674",
        "exp24_ring0_control_s4011",
    ]
    for entry in entries:
        assert entry["exp24_phase"] == "ring0"
        assert entry["exp24_mechanism"] == "control"
        assert entry["budget_seconds"] == 600.0
        assert entry["world_size"] == 8
        assert entry["train_sampling_mode"] == "random"
        assert entry["artifact_impact"] == "artifact_changes_weights_only"
        assert entry["submit_valid"] is True


def test_phase_a_sampling_matrix_is_mechanism_free():
    mod = _load_exp24()

    entries = mod.build_phase_a_sampling_matrix(
        speed_config={"batch_size": 1024, "chunk_size": 64},
        seeds=[1337],
        world_size=8,
    )

    assert [entry["train_sampling_mode"] for entry in entries] == [
        "random",
        "sequential_epoch",
        "shuffled_epoch",
    ]
    assert [entry["name"] for entry in entries] == [
        "exp24_phaseA_random_s1337",
        "exp24_phaseA_sequential_epoch_s1337",
        "exp24_phaseA_shuffled_epoch_s1337",
    ]
    assert all(entry["exp24_mechanism"] == "sampling_policy" for entry in entries)
    assert all(entry["fast_slow_enabled"] is False for entry in entries)
    assert all(entry["spectral_reg_lambda_dead"] == 0.0 for entry in entries)
    assert all(entry["predictive_aux_weight"] == 0.0 for entry in entries)


def test_control_noise_summary_uses_sample_std_and_min_max():
    mod = _load_exp24()
    results = [
        {"config": {"seed": 1337}, "eval": {"bpb": 1.05}, "train": {"elapsed_s": 599.0, "aggregate_tokens_per_sec": 41.0}},
        {"config": {"seed": 2674}, "eval": {"bpb": 1.07}, "train": {"elapsed_s": 598.0, "aggregate_tokens_per_sec": 43.0}},
        {"config": {"seed": 4011}, "eval": {"bpb": 1.06}, "train": {"elapsed_s": 597.0, "aggregate_tokens_per_sec": 42.0}},
    ]

    summary = mod.summarize_control_noise(results)

    assert summary["seeds"] == [1337, 2674, 4011]
    assert summary["count"] == 3
    assert math.isclose(summary["bpb_mean"], 1.06)
    assert math.isclose(summary["bpb_sample_std"], 0.01)
    assert summary["bpb_min"] == 1.05
    assert summary["bpb_max"] == 1.07
    assert summary["tokens_per_sec_mean"] == 42.0
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_exp24_training_bundle.py -q
```

Expected: collection fails because `experiments/24_training_time_bundle/exp24.py` does not exist.

- [ ] **Step 3: Implement Exp24 scaffold**

Create `experiments/24_training_time_bundle/exp24.py`:

```python
#!/usr/bin/env python3
"""Exp24 training-time bundle matrices and summaries."""
from __future__ import annotations

import copy
import statistics
from collections.abc import Sequence
from typing import Any

DEFAULT_CONTROL_SEEDS = (1337, 2674, 4011)
ARTIFACT_CHANGES_WEIGHTS_ONLY = "artifact_changes_weights_only"
ARTIFACT_TRAINING_ONLY = "artifact_training_only"


def _base_entry(
    *,
    speed_config: dict[str, Any],
    world_size: int,
    budget_seconds: float,
) -> dict[str, Any]:
    entry = copy.deepcopy(speed_config)
    entry.update({
        "mode": "exp24_training_time_bundle",
        "world_size": int(world_size),
        "budget_seconds": float(budget_seconds),
        "eval_batches": int(entry.get("eval_batches", 0)),
        "train_sampling_mode": str(entry.get("train_sampling_mode", "random")),
        "artifact_impact": str(
            entry.get("artifact_impact", ARTIFACT_CHANGES_WEIGHTS_ONLY)
        ),
        "submit_valid": bool(entry.get("submit_valid", True)),
        "fast_slow_enabled": bool(entry.get("fast_slow_enabled", False)),
        "fast_slow_interval": int(entry.get("fast_slow_interval", 0)),
        "fast_slow_alpha": float(entry.get("fast_slow_alpha", 0.0)),
        "fast_slow_eval_copy": str(entry.get("fast_slow_eval_copy", "fast")),
        "spectral_reg_lambda_dead": float(entry.get("spectral_reg_lambda_dead", 0.0)),
        "spectral_reg_lambda_sticky": float(entry.get("spectral_reg_lambda_sticky", 0.0)),
        "spectral_reg_min_a": float(entry.get("spectral_reg_min_a", 0.05)),
        "spectral_reg_max_a": float(entry.get("spectral_reg_max_a", 0.98)),
        "predictive_aux_weight": float(entry.get("predictive_aux_weight", 0.0)),
        "predictive_aux_horizon": int(entry.get("predictive_aux_horizon", 0)),
        "predictive_aux_dim": int(entry.get("predictive_aux_dim", 0)),
        "embed_freeze_steps": int(entry.get("embed_freeze_steps", 0)),
        "semantic_layer_index": int(entry.get("semantic_layer_index", 0)),
        "semantic_momentum_min": float(entry.get("semantic_momentum_min", 0.5)),
        "semantic_overhead_gate": float(entry.get("semantic_overhead_gate", 0.08)),
    })
    return entry


def _named_entry(
    *,
    base: dict[str, Any],
    phase: str,
    mechanism: str,
    arm: str,
    seed: int,
) -> dict[str, Any]:
    entry = copy.deepcopy(base)
    entry.update({
        "name": f"exp24_{phase}_{arm}_s{int(seed)}",
        "seed": int(seed),
        "exp24_phase": phase,
        "exp24_mechanism": mechanism,
    })
    return entry


def build_ring0_control_matrix(
    *,
    speed_config: dict[str, Any],
    world_size: int = 8,
    budget_seconds: float = 600.0,
    seeds: Sequence[int] = DEFAULT_CONTROL_SEEDS,
) -> list[dict[str, Any]]:
    base = _base_entry(
        speed_config=speed_config,
        world_size=world_size,
        budget_seconds=budget_seconds,
    )
    base["train_sampling_mode"] = "random"
    return [
        _named_entry(
            base=base,
            phase="ring0",
            mechanism="control",
            arm="control",
            seed=int(seed),
        )
        for seed in seeds
    ]


def build_phase_a_sampling_matrix(
    *,
    speed_config: dict[str, Any],
    world_size: int = 8,
    budget_seconds: float = 600.0,
    seeds: Sequence[int] = (1337,),
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for policy in ("random", "sequential_epoch", "shuffled_epoch"):
        base = _base_entry(
            speed_config=speed_config,
            world_size=world_size,
            budget_seconds=budget_seconds,
        )
        base["train_sampling_mode"] = policy
        for seed in seeds:
            entries.append(_named_entry(
                base=base,
                phase="phaseA",
                mechanism="sampling_policy",
                arm=policy,
                seed=int(seed),
            ))
    return entries


def summarize_control_noise(results: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [
        {
            "seed": int(row.get("config", {}).get("seed")),
            "bpb": float(row.get("eval", {}).get("bpb")),
            "elapsed_s": float(row.get("train", {}).get("elapsed_s")),
            "tokens_per_sec": float(
                row.get("train", {}).get("aggregate_tokens_per_sec")
            ),
        }
        for row in results
    ]
    bpbs = [row["bpb"] for row in rows]
    tps = [row["tokens_per_sec"] for row in rows]
    return {
        "count": len(rows),
        "seeds": [row["seed"] for row in rows],
        "bpb_mean": float(statistics.fmean(bpbs)),
        "bpb_sample_std": float(statistics.stdev(bpbs)) if len(bpbs) > 1 else 0.0,
        "bpb_min": float(min(bpbs)),
        "bpb_max": float(max(bpbs)),
        "elapsed_s_mean": float(statistics.fmean(row["elapsed_s"] for row in rows)),
        "tokens_per_sec_mean": float(statistics.fmean(tps)),
    }
```

- [ ] **Step 4: Run the tests and verify they pass**

Run:

```bash
.venv/bin/python -m pytest tests/test_exp24_training_bundle.py -q
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add experiments/24_training_time_bundle/exp24.py tests/test_exp24_training_bundle.py
git commit -m "exp24: add training bundle matrix scaffold"
```

## Task 2: O(1) Shuffled-Epoch Sampling

**Files:**
- Modify: `experiments/23_fast_path/fast_path.py`
- Modify: `experiments/23_fast_path/runner_fast_path.py`
- Modify: `tests/test_exp23_fast_path.py`

- [ ] **Step 1: Write failing sampler tests**

Append to `tests/test_exp23_fast_path.py`:

```python
def test_shuffled_epoch_sampling_covers_each_rank_without_materializing():
    mod = _load_module()
    num_tokens = 34
    seq_len = 3
    stride = 3
    batch_size = 2
    world_size = 3

    total_starts = mod.count_lm_starts(num_tokens, seq_len, stride)
    epoch_steps = mod.sequential_epoch_steps(
        num_tokens=num_tokens,
        seq_len=seq_len,
        stride=stride,
        batch_size=batch_size,
        world_size=world_size,
    )

    covered = []
    shuffled_order = []
    sequential_order = []
    for rank in range(world_size):
        for step in range(epoch_steps):
            starts = mod.shuffled_epoch_sharded_lm_starts(
                num_tokens=num_tokens,
                seq_len=seq_len,
                stride=stride,
                batch_size=batch_size,
                rank=rank,
                world_size=world_size,
                step=step,
                seed=1337,
            )
            seq_starts = mod.sequential_sharded_lm_starts(
                num_tokens=num_tokens,
                seq_len=seq_len,
                stride=stride,
                batch_size=batch_size,
                rank=rank,
                world_size=world_size,
                step=step,
            )
            covered.extend(int(start // stride) for start in starts.tolist())
            shuffled_order.extend(int(start // stride) for start in starts.tolist())
            sequential_order.extend(int(start // stride) for start in seq_starts.tolist())

    assert sorted(set(covered)) == list(range(total_starts))
    assert shuffled_order != sequential_order


def test_train_fast_for_budget_accepts_shuffled_epoch_sampling():
    mod = _load_runner_module()
    model = _TinyTokenTrainModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    result = mod.train_fast_for_budget(
        model,
        train_tokens=torch.arange(25, dtype=torch.int16) % 6,
        train_num_tokens=25,
        stride=3,
        seq_len=3,
        batch_size=2,
        device=torch.device("cpu"),
        optimizer=optimizer,
        budget_seconds=300.0,
        chunk_size=2,
        grad_clip_norm=0.0,
        fused_grad_clip=False,
        rank=0,
        world_size=1,
        seed=123,
        precision="bf16",
        stop_check_interval=1,
        stop_margin_seconds=0.0,
        vocab_size=6,
        max_steps=None,
        prefetch_batches=True,
        train_sampling_mode="shuffled_epoch",
    )

    assert result["steps"] == 4
    assert result["sampling_mode"] == "shuffled_epoch"
    assert result["epoch_steps"] == 4
    assert result["unique_start_count"] == 7
    assert result["epoch_complete"] is True
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_exp23_fast_path.py::test_shuffled_epoch_sampling_covers_each_rank_without_materializing tests/test_exp23_fast_path.py::test_train_fast_for_budget_accepts_shuffled_epoch_sampling -q
```

Expected: first test fails with `AttributeError: module 'exp23_fast_path' has no attribute 'shuffled_epoch_sharded_lm_starts'`.

- [ ] **Step 3: Implement shuffled-epoch helpers**

In `experiments/23_fast_path/fast_path.py`, add these helpers after `sequential_sharded_lm_starts`:

```python
def _coprime_stride(size: int, seed: int) -> int:
    size = int(size)
    if size <= 1:
        return 1
    stride = (abs(int(seed)) % size) | 1
    while math.gcd(stride, size) != 1:
        stride += 2
        if stride >= size:
            stride = 1
    return stride


def shuffled_epoch_sharded_lm_starts(
    *,
    num_tokens: int,
    seq_len: int,
    stride: int,
    batch_size: int,
    rank: int,
    world_size: int,
    step: int,
    seed: int,
) -> torch.Tensor:
    """Return fixed-size starts from an O(1)-memory per-rank permutation."""
    total = count_lm_starts(num_tokens, seq_len, stride)
    sharded = count_sharded_lm_starts(
        total_starts=total,
        rank=rank,
        world_size=world_size,
    )
    if sharded <= 0:
        raise RuntimeError(
            f"rank {rank} has no train starts after world_size={world_size} sharding"
        )
    first = int(step) * int(batch_size)
    raw = torch.arange(first, first + int(batch_size), dtype=torch.long)
    perm_stride = _coprime_stride(sharded, int(seed) + int(rank) * 104_729)
    offset = (abs(int(seed)) + int(rank) * 65_537) % sharded
    local_idx = (raw * perm_stride + offset) % sharded
    global_idx = int(rank) + local_idx * int(world_size)
    return global_idx * int(stride)


class ShuffledEpochShardedStartSampler:
    """Stateful shuffled-epoch sampler compatible with Exp23BatchPrefetcher."""

    def __init__(self, *, seed: int) -> None:
        self._seed = int(seed)
        self._step = 0
        self._lock = threading.Lock()

    def __call__(
        self,
        *,
        num_tokens: int,
        seq_len: int,
        stride: int,
        batch_size: int,
        rank: int,
        world_size: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        del generator
        with self._lock:
            step = self._step
            self._step += 1
        return shuffled_epoch_sharded_lm_starts(
            num_tokens=num_tokens,
            seq_len=seq_len,
            stride=stride,
            batch_size=batch_size,
            rank=rank,
            world_size=world_size,
            step=step,
            seed=self._seed,
        )
```

Also add `import math` near the top of `fast_path.py`.

- [ ] **Step 4: Thread the mode through the runner**

In `experiments/23_fast_path/runner_fast_path.py`, import `ShuffledEpochShardedStartSampler` and `shuffled_epoch_sharded_lm_starts` from `fast_path`.

Change the sampling-mode validation to:

```python
if sampling_mode not in {"random", "sequential_epoch", "shuffled_epoch"}:
    raise ValueError(
        "train_sampling_mode must be 'random', 'sequential_epoch', "
        f"or 'shuffled_epoch', got {train_sampling_mode!r}"
    )
```

Change epoch-step handling to:

```python
if sampling_mode in {"sequential_epoch", "shuffled_epoch"}:
    epoch_steps = sequential_epoch_steps(
        num_tokens=train_num_tokens,
        seq_len=seq_len,
        stride=stride,
        batch_size=batch_size,
        world_size=world_size_,
    )
    if max_steps is None:
        max_steps = epoch_steps
```
Change sampler construction to:

```python
if sampling_mode == "sequential_epoch":
    batch_sampler = SequentialShardedStartSampler()
elif sampling_mode == "shuffled_epoch":
    batch_sampler = ShuffledEpochShardedStartSampler(seed=seed)
else:
    batch_sampler = None
```

Pass `batch_sampler=batch_sampler` into `Exp23BatchPrefetcher`.

Change the non-prefetch branch to:

```python
if sampling_mode == "sequential_epoch":
    starts = sequential_sharded_lm_starts(
        num_tokens=train_num_tokens,
        seq_len=seq_len,
        stride=stride,
        batch_size=batch_size,
        rank=rank_,
        world_size=world_size_,
        step=steps,
    )
elif sampling_mode == "shuffled_epoch":
    starts = shuffled_epoch_sharded_lm_starts(
        num_tokens=train_num_tokens,
        seq_len=seq_len,
        stride=stride,
        batch_size=batch_size,
        rank=rank_,
        world_size=world_size_,
        step=steps,
        seed=seed,
    )
else:
    starts = sample_sharded_lm_starts(
        num_tokens=train_num_tokens,
        seq_len=seq_len,
        stride=stride,
        batch_size=batch_size,
        rank=rank_,
        world_size=world_size_,
        generator=rng,
    )
```

Change result accounting to treat both epoch modes as unique-coverage modes:

```python
"unique_start_count": (
    min(total_starts, steps * int(batch_size) * world_size_)
    if sampling_mode in {"sequential_epoch", "shuffled_epoch"}
    else None
),
```

- [ ] **Step 5: Run targeted tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_exp23_fast_path.py::test_shuffled_epoch_sampling_covers_each_rank_without_materializing tests/test_exp23_fast_path.py::test_train_fast_for_budget_accepts_shuffled_epoch_sampling tests/test_exp23_fast_path.py::test_train_fast_for_budget_sequential_epoch_stops_after_epoch -q
```

Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add experiments/23_fast_path/fast_path.py experiments/23_fast_path/runner_fast_path.py tests/test_exp23_fast_path.py
git commit -m "exp23: add shuffled epoch sampling mode"
```

## Task 3: Training-Time Hook Helpers

**Files:**
- Create: `experiments/23_fast_path/training_hooks.py`
- Create: `tests/test_exp24_training_hooks.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_exp24_training_hooks.py`:

```python
"""Tests for Exp24 training-time fast-path hooks."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import torch.nn as nn


REPO = Path(__file__).resolve().parents[1]
HOOKS_PATH = REPO / "experiments" / "23_fast_path" / "training_hooks.py"


def _load_hooks():
    spec = importlib.util.spec_from_file_location("exp23_training_hooks", HOOKS_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Core(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_a = nn.Parameter(torch.tensor([-4.0, 0.0, 5.0]))


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = _Core()


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(8, 3)
        self.layers = nn.ModuleList([_Block()])
        self.linear = nn.Linear(3, 3, bias=False)


def test_fast_slow_consolidator_updates_shadow_and_can_restore():
    hooks = _load_hooks()
    model = _Model()
    consolidator = hooks.FastSlowConsolidator.from_config(
        model,
        {"fast_slow_enabled": True, "fast_slow_interval": 1, "fast_slow_alpha": 0.5},
    )

    with torch.no_grad():
        model.linear.weight.add_(2.0)
    consolidator.after_optimizer_step(model, step=1)
    diag = consolidator.diagnostics(model)

    assert diag["enabled"] is True
    assert diag["sync_count"] == 1
    assert diag["fast_slow_l2"] > 0.0

    consolidator.copy_slow_to_model(model)
    assert torch.allclose(model.linear.weight, consolidator.slow_state["linear.weight"])


def test_spectral_regularization_penalizes_out_of_band_a():
    hooks = _load_hooks()
    model = _Model()

    penalty = hooks.spectral_regularization_loss(
        model,
        lambda_dead=2.0,
        lambda_sticky=3.0,
        min_a=0.1,
        max_a=0.9,
    )
    summary = hooks.spectral_summary(model)

    assert penalty is not None
    assert float(penalty) > 0.0
    assert summary["log_a_param_count"] == 1
    assert summary["a_min"] < 0.1
    assert summary["a_max"] > 0.9


def test_zero_embedding_grad_until_step_zeroes_only_before_boundary():
    hooks = _load_hooks()
    model = _Model()
    model.embed.weight.grad = torch.ones_like(model.embed.weight)

    hooks.zero_embedding_grad_until(model, step=2, freeze_steps=3)
    assert torch.count_nonzero(model.embed.weight.grad) == 0

    model.embed.weight.grad = torch.ones_like(model.embed.weight)
    hooks.zero_embedding_grad_until(model, step=3, freeze_steps=3)
    assert torch.count_nonzero(model.embed.weight.grad) == model.embed.weight.numel()
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_exp24_training_hooks.py -q
```

Expected: collection fails because `training_hooks.py` does not exist.

- [ ] **Step 3: Implement hook helpers**

Create `experiments/23_fast_path/training_hooks.py`:

```python
"""Training-time mechanism hooks shared by Exp23/Exp24 fast-path runners."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class FastSlowConsolidator:
    enabled: bool
    interval: int
    alpha: float
    slow_state: dict[str, torch.Tensor]
    sync_count: int = 0

    @classmethod
    def from_config(
        cls,
        model: torch.nn.Module,
        config: dict[str, Any],
    ) -> "FastSlowConsolidator":
        enabled = bool(config.get("fast_slow_enabled", False))
        interval = int(config.get("fast_slow_interval", 0))
        alpha = float(config.get("fast_slow_alpha", 0.0))
        if not enabled:
            interval = 0
            alpha = 0.0
        slow_state = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
        }
        return cls(enabled=enabled, interval=interval, alpha=alpha, slow_state=slow_state)

    def after_optimizer_step(self, model: torch.nn.Module, *, step: int) -> None:
        if not self.enabled or self.interval <= 0:
            return
        if int(step) % self.interval != 0:
            return
        self.sync_count += 1
        with torch.no_grad():
            for name, param in model.named_parameters():
                self.slow_state[name].lerp_(param.detach(), self.alpha)

    def copy_slow_to_model(self, model: torch.nn.Module) -> None:
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.copy_(self.slow_state[name].to(device=param.device, dtype=param.dtype))

    def diagnostics(self, model: torch.nn.Module) -> dict[str, Any]:
        sq = 0.0
        with torch.no_grad():
            for name, param in model.named_parameters():
                diff = param.detach().float() - self.slow_state[name].to(param.device).float()
                sq += float(diff.square().sum().cpu())
        return {
            "enabled": self.enabled,
            "interval": self.interval,
            "alpha": self.alpha,
            "sync_count": self.sync_count,
            "fast_slow_l2": math.sqrt(sq),
        }


def iter_log_a_params(model: torch.nn.Module):
    for name, param in model.named_parameters():
        if name.endswith(".log_a") and param.ndim == 1:
            yield name, param


def spectral_regularization_loss(
    model: torch.nn.Module,
    *,
    lambda_dead: float,
    lambda_sticky: float,
    min_a: float,
    max_a: float,
) -> torch.Tensor | None:
    losses = []
    for _name, log_a in iter_log_a_params(model):
        a = torch.sigmoid(log_a.float())
        dead = torch.relu(float(min_a) - a).square().mean()
        sticky = torch.relu(a - float(max_a)).square().mean()
        losses.append(float(lambda_dead) * dead + float(lambda_sticky) * sticky)
    if not losses:
        return None
    return torch.stack(losses).mean()


def spectral_summary(model: torch.nn.Module) -> dict[str, Any]:
    vals = [torch.sigmoid(param.detach().float()).reshape(-1).cpu() for _name, param in iter_log_a_params(model)]
    if not vals:
        return {"log_a_param_count": 0}
    merged = torch.cat(vals)
    return {
        "log_a_param_count": len(vals),
        "a_min": float(merged.min()),
        "a_max": float(merged.max()),
        "a_mean": float(merged.mean()),
        "a_p05": float(torch.quantile(merged, 0.05)),
        "a_p50": float(torch.quantile(merged, 0.50)),
        "a_p95": float(torch.quantile(merged, 0.95)),
    }


def zero_embedding_grad_until(
    model: torch.nn.Module,
    *,
    step: int,
    freeze_steps: int,
) -> None:
    if int(freeze_steps) <= 0 or int(step) >= int(freeze_steps):
        return
    embed = getattr(model, "embed", None)
    if embed is not None and getattr(embed, "weight", None) is not None:
        grad = embed.weight.grad
        if grad is not None:
            grad.zero_()
```

- [ ] **Step 4: Run tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_exp24_training_hooks.py -q
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add experiments/23_fast_path/training_hooks.py tests/test_exp24_training_hooks.py
git commit -m "exp23: add training mechanism hooks"
```

## Task 4: Thread Fast/Slow, Spectral, And SGNS Freeze Through Runner

**Files:**
- Modify: `experiments/23_fast_path/runner_fast_path.py`
- Modify: `tests/test_exp23_fast_path.py`

- [ ] **Step 1: Write failing runner integration tests**

Append to `tests/test_exp23_fast_path.py`:

```python
def test_train_fast_for_budget_applies_spectral_extra_loss(monkeypatch):
    mod = _load_runner_module()
    calls = []

    def fake_spectral_loss(model, **kwargs):
        calls.append(kwargs)
        return torch.tensor(0.25, requires_grad=True)

    monkeypatch.setattr(mod, "spectral_regularization_loss", fake_spectral_loss)
    model = _TinyTokenTrainModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    result = mod.train_fast_for_budget(
        model,
        train_tokens=torch.arange(128, dtype=torch.int16) % 6,
        train_num_tokens=128,
        stride=4,
        seq_len=3,
        batch_size=2,
        device=torch.device("cpu"),
        optimizer=optimizer,
        budget_seconds=300.0,
        chunk_size=2,
        grad_clip_norm=0.0,
        fused_grad_clip=False,
        rank=0,
        world_size=1,
        seed=123,
        precision="bf16",
        stop_check_interval=1,
        stop_margin_seconds=0.0,
        vocab_size=6,
        max_steps=1,
        prefetch_batches=False,
        spectral_reg_lambda_dead=0.1,
        spectral_reg_lambda_sticky=0.2,
    )

    assert calls == [{"lambda_dead": 0.1, "lambda_sticky": 0.2, "min_a": 0.05, "max_a": 0.98}]
    assert result["mechanisms"]["spectral"]["enabled"] is True


def test_train_fast_for_budget_zeroes_embed_grad_during_freeze(monkeypatch):
    mod = _load_runner_module()
    zero_calls = []

    def fake_zero(model, *, step, freeze_steps):
        zero_calls.append((step, freeze_steps))

    monkeypatch.setattr(mod, "zero_embedding_grad_until", fake_zero)
    model = _TinyTokenTrainModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    mod.train_fast_for_budget(
        model,
        train_tokens=torch.arange(128, dtype=torch.int16) % 6,
        train_num_tokens=128,
        stride=4,
        seq_len=3,
        batch_size=2,
        device=torch.device("cpu"),
        optimizer=optimizer,
        budget_seconds=300.0,
        chunk_size=2,
        grad_clip_norm=0.0,
        fused_grad_clip=False,
        rank=0,
        world_size=1,
        seed=123,
        precision="bf16",
        stop_check_interval=1,
        stop_margin_seconds=0.0,
        vocab_size=6,
        max_steps=2,
        prefetch_batches=False,
        embed_freeze_steps=2,
    )

    assert zero_calls == [(0, 2), (1, 2)]
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_exp23_fast_path.py::test_train_fast_for_budget_applies_spectral_extra_loss tests/test_exp23_fast_path.py::test_train_fast_for_budget_zeroes_embed_grad_during_freeze -q
```

Expected: `TypeError` because `train_fast_for_budget()` does not accept the new keyword arguments.

- [ ] **Step 3: Extend `_run_train_step` for extra loss**

In `runner_fast_path.py`, add parameters to `_run_train_step`:

```python
    spectral_reg_lambda_dead: float = 0.0,
    spectral_reg_lambda_sticky: float = 0.0,
    spectral_reg_min_a: float = 0.05,
    spectral_reg_max_a: float = 0.98,
```

After the CE loss branch and before all-reduce, add:

```python
    spectral_extra = None
    if spectral_reg_lambda_dead > 0.0 or spectral_reg_lambda_sticky > 0.0:
        spectral_extra = spectral_regularization_loss(
            model,
            lambda_dead=spectral_reg_lambda_dead,
            lambda_sticky=spectral_reg_lambda_sticky,
            min_a=spectral_reg_min_a,
            max_a=spectral_reg_max_a,
        )
        if spectral_extra is not None:
            spectral_extra.backward()
```

Keep the returned `loss` as the CE scalar so historical train-loss logs remain comparable.

- [ ] **Step 4: Extend `train_fast_for_budget` for mechanisms**

Import from `training_hooks`:

```python
from training_hooks import (
    FastSlowConsolidator,
    spectral_regularization_loss,
    spectral_summary,
    zero_embedding_grad_until,
)
```

Add these keyword arguments to `train_fast_for_budget`:

```python
    fast_slow_enabled: bool = False,
    fast_slow_interval: int = 0,
    fast_slow_alpha: float = 0.0,
    fast_slow_eval_copy: str = "fast",
    spectral_reg_lambda_dead: float = 0.0,
    spectral_reg_lambda_sticky: float = 0.0,
    spectral_reg_min_a: float = 0.05,
    spectral_reg_max_a: float = 0.98,
    embed_freeze_steps: int = 0,
```

Before the loop, create:

```python
    fast_slow = FastSlowConsolidator.from_config(model, {
        "fast_slow_enabled": fast_slow_enabled,
        "fast_slow_interval": fast_slow_interval,
        "fast_slow_alpha": fast_slow_alpha,
    })
    spectral_before = spectral_summary(model)
```

Pass spectral arguments into `_run_train_step`.

After backward and before gradient clipping, call:

```python
            zero_embedding_grad_until(
                model,
                step=steps,
                freeze_steps=embed_freeze_steps,
            )
```

After `optimizer.step()`, call:

```python
            fast_slow.after_optimizer_step(model, step=steps + 1)
```

Before result construction, if the final copy selection is slow:

```python
    if fast_slow.enabled and str(fast_slow_eval_copy).strip().lower() == "slow":
        fast_slow.copy_slow_to_model(model)
```

Add to `result`:

```python
        "mechanisms": {
            "fast_slow": fast_slow.diagnostics(model),
            "spectral": {
                "enabled": spectral_reg_lambda_dead > 0.0 or spectral_reg_lambda_sticky > 0.0,
                "before": spectral_before,
                "after": spectral_summary(model),
            },
            "embed_freeze": {
                "freeze_steps": int(embed_freeze_steps),
                "enabled": int(embed_freeze_steps) > 0,
            },
        },
```

- [ ] **Step 5: Thread config from `run_condition`**

In the `train_fast_for_budget` call inside `run_condition`, pass:

```python
        fast_slow_enabled=bool(config.get("fast_slow_enabled", False)),
        fast_slow_interval=int(config.get("fast_slow_interval", 0)),
        fast_slow_alpha=float(config.get("fast_slow_alpha", 0.0)),
        fast_slow_eval_copy=str(config.get("fast_slow_eval_copy", "fast")),
        spectral_reg_lambda_dead=float(config.get("spectral_reg_lambda_dead", 0.0)),
        spectral_reg_lambda_sticky=float(config.get("spectral_reg_lambda_sticky", 0.0)),
        spectral_reg_min_a=float(config.get("spectral_reg_min_a", 0.05)),
        spectral_reg_max_a=float(config.get("spectral_reg_max_a", 0.98)),
        embed_freeze_steps=int(config.get("embed_freeze_steps", 0)),
```

- [ ] **Step 6: Run targeted tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_exp23_fast_path.py::test_train_fast_for_budget_applies_spectral_extra_loss tests/test_exp23_fast_path.py::test_train_fast_for_budget_zeroes_embed_grad_during_freeze tests/test_exp24_training_hooks.py -q
```

Expected: all targeted tests pass.

- [ ] **Step 7: Commit**

```bash
git add experiments/23_fast_path/runner_fast_path.py tests/test_exp23_fast_path.py tests/test_exp24_training_hooks.py
git commit -m "exp23: thread training-time mechanism hooks"
```

## Task 5: SemanticOptimizer Fast-Path Wiring And Overhead Gate

**Files:**
- Modify: `experiments/23_fast_path/runner_fast_path.py`
- Modify: `experiments/24_training_time_bundle/exp24.py`
- Modify: `tests/test_exp23_fast_path.py`
- Modify: `tests/test_exp24_training_bundle.py`

- [ ] **Step 1: Write failing optimizer tests**

Append to `tests/test_exp23_fast_path.py`:

```python
def test_build_optimizer_can_create_semantic_optimizer(monkeypatch):
    mod = _load_runner_module()

    class FakeSemanticOptimizer(torch.optim.SGD):
        def __init__(self, params, **kwargs):
            self.kwargs = kwargs
            self.bound = None
            super().__init__(params, lr=kwargs["lr"])

        def bind_param_names(self, named_params):
            self.bound = list(named_params)

        def beta_trace(self):
            return {"beta_min": 0.5, "beta_max": 0.95, "beta_mean": 0.7}

    monkeypatch.setattr(mod, "SemanticOptimizer", FakeSemanticOptimizer)

    class TinySemanticModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Module()])
            self.layers[0].core = nn.Module()
            self.layers[0].core.log_a = nn.Parameter(torch.zeros(4))
            self.layers[0].core.in_proj = nn.Linear(4, 4, bias=False)
            self.layers[0].core.select_proj = nn.Linear(4, 4, bias=False)
            self.layers[0].core.gate_proj = nn.Linear(4, 4, bias=False)
            self.layers[0].core.out_proj = nn.Linear(4, 4, bias=False)

    model = TinySemanticModel()
    opt = mod._build_optimizer({
        "optimizer": "semantic",
        "base_lr": 0.01,
        "weight_decay": 0.02,
        "semantic_layer_index": 0,
        "semantic_momentum_min": 0.25,
    }, model)

    assert isinstance(opt, FakeSemanticOptimizer)
    assert opt.kwargs["a_param_name"] == "layers.0.core.log_a"
    assert opt.kwargs["channel_map"]["layers.0.core.in_proj.weight"] == 0
    assert opt.kwargs["channel_map"]["layers.0.core.out_proj.weight"] == 1
    assert opt.kwargs["momentum_min"] == 0.25
    assert opt.bound is not None
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
.venv/bin/python -m pytest tests/test_exp23_fast_path.py::test_build_optimizer_can_create_semantic_optimizer -q
```

Expected: `AttributeError` or `NameError` because `SemanticOptimizer` is not imported/wired.

- [ ] **Step 3: Implement semantic optimizer resolver**

In `runner_fast_path.py`, import:

```python
from chaoscontrol.optim.semantic import SemanticOptimizer
```

Add this helper before `_build_optimizer`:

```python
def _semantic_optimizer_config(
    model: torch.nn.Module,
    *,
    layer_index: int,
) -> dict[str, Any]:
    prefix = f"layers.{int(layer_index)}.core"
    named = dict(model.named_parameters())
    a_name = f"{prefix}.log_a"
    if a_name not in named:
        raise ValueError(f"SemanticOptimizer requires {a_name!r} in model parameters")
    channel_map: dict[str, int] = {}
    for suffix, axis in {
        "in_proj.weight": 0,
        "select_proj.weight": 0,
        "gate_proj.weight": 0,
        "delta_proj.weight": 0,
        "out_proj.weight": 1,
    }.items():
        name = f"{prefix}.{suffix}"
        if name in named:
            channel_map[name] = axis
    if not channel_map:
        raise ValueError(f"SemanticOptimizer found no channel-coupled matrices under {prefix}")
    return {"a_param_name": a_name, "channel_map": channel_map}
```

Add branch in `_build_optimizer`:

```python
    if name == "semantic":
        semantic_cfg = _semantic_optimizer_config(
            model,
            layer_index=int(config.get("semantic_layer_index", 0)),
        )
        opt = SemanticOptimizer(
            params,
            lr=base_lr,
            weight_decay=weight_decay,
            adamw_lr=base_lr,
            adamw_weight_decay=weight_decay,
            momentum_min=float(config.get("semantic_momentum_min", 0.5)),
            **semantic_cfg,
        )
        opt.bind_param_names(list(model.named_parameters()))
        return opt
```

- [ ] **Step 4: Log optimizer diagnostics**

Add helper in `runner_fast_path.py`:

```python
def _optimizer_diagnostics(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    trace_fn = getattr(optimizer, "beta_trace", None)
    if trace_fn is None:
        return {"type": optimizer.__class__.__name__}
    trace = trace_fn()
    if trace is None:
        return {"type": optimizer.__class__.__name__, "beta_trace": None}
    return {
        "type": optimizer.__class__.__name__,
        "beta_trace": {
            key: value
            for key, value in trace.items()
            if key in {"beta_min", "beta_max", "beta_mean"}
        },
    }
```

Add to `train_fast_for_budget` result:

```python
        "optimizer": _optimizer_diagnostics(optimizer),
```

- [ ] **Step 5: Add semantic overhead-gate matrix helper**

In `experiments/24_training_time_bundle/exp24.py`, add:

```python
def build_semantic_overhead_gate_matrix(
    *,
    speed_config: dict[str, Any],
    seed: int = 1337,
    world_size: int = 1,
    budget_seconds: float = 90.0,
) -> list[dict[str, Any]]:
    base = _base_entry(
        speed_config=speed_config,
        world_size=world_size,
        budget_seconds=budget_seconds,
    )
    entries = []
    for opt_name in ("muon", "semantic"):
        entry = copy.deepcopy(base)
        entry.update({
            "name": f"exp24_semantic_gate_{opt_name}_s{int(seed)}",
            "seed": int(seed),
            "exp24_phase": "smoke",
            "exp24_mechanism": "semantic_optimizer_gate",
            "optimizer": opt_name,
            "artifact_impact": ARTIFACT_CHANGES_WEIGHTS_ONLY,
        })
        entries.append(entry)
    return entries
```

Add a test in `tests/test_exp24_training_bundle.py` that asserts this returns Muon and Semantic rows with `world_size=1`, `budget_seconds=90.0`, and matching `semantic_overhead_gate == 0.08`.

- [ ] **Step 6: Run tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_exp23_fast_path.py::test_build_optimizer_can_create_semantic_optimizer tests/test_exp24_training_bundle.py -q
```

Expected: all targeted tests pass.

- [ ] **Step 7: Commit**

```bash
git add experiments/23_fast_path/runner_fast_path.py experiments/24_training_time_bundle/exp24.py tests/test_exp23_fast_path.py tests/test_exp24_training_bundle.py
git commit -m "exp24: wire semantic optimizer overhead gate"
```

## Task 6: Predictive Coding Auxiliary Hook

**Files:**
- Modify: `experiments/23_fast_path/training_hooks.py`
- Modify: `experiments/23_fast_path/runner_fast_path.py`
- Modify: `tests/test_exp24_training_hooks.py`
- Modify: `tests/test_exp23_fast_path.py`

- [ ] **Step 1: Write failing predictive-aux tests**

Append to `tests/test_exp24_training_hooks.py`:

```python
def test_predictive_auxiliary_loss_uses_detached_future_hidden():
    hooks = _load_hooks()
    hidden = torch.randn(2, 5, 4, requires_grad=True)
    proj = nn.Linear(4, 4, bias=False)

    loss = hooks.predictive_auxiliary_loss(
        hidden,
        projection=proj,
        horizon=2,
    )

    assert loss is not None
    loss.backward()
    assert hidden.grad is not None
    assert proj.weight.grad is not None
```

Append to `tests/test_exp23_fast_path.py`:

```python
def test_run_train_step_applies_predictive_auxiliary(monkeypatch):
    mod = _load_runner_module()
    calls = []

    def fake_aux(hidden, *, projection, horizon):
        calls.append((tuple(hidden.shape), horizon, projection is not None))
        return hidden.float().mean() * 0.0 + 0.5

    monkeypatch.setattr(mod, "predictive_auxiliary_loss", fake_aux)
    model = _TinyTrainStepModel()
    inputs = torch.randn(2, 3, 4)
    targets = torch.zeros(2, 3, dtype=torch.long)
    aux = nn.Linear(4, 4, bias=False)

    loss = mod._run_train_step(
        model=model,
        inputs=inputs,
        targets=targets,
        chunk_size=2,
        precision="bf16",
        ddp_active=False,
        world_size=1,
        predictive_aux_weight=0.1,
        predictive_aux_horizon=1,
        predictive_aux_projection=aux,
    )

    assert loss.ndim == 0
    assert calls == [((2, 3, 4), 1, True)]
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_exp24_training_hooks.py::test_predictive_auxiliary_loss_uses_detached_future_hidden tests/test_exp23_fast_path.py::test_run_train_step_applies_predictive_auxiliary -q
```

Expected: first test fails because `predictive_auxiliary_loss` is missing.

- [ ] **Step 3: Implement predictive helper**

In `training_hooks.py`, add:

```python
def predictive_auxiliary_loss(
    hidden: torch.Tensor,
    *,
    projection: torch.nn.Module,
    horizon: int,
) -> torch.Tensor | None:
    horizon = int(horizon)
    if horizon <= 0 or hidden.size(1) <= horizon:
        return None
    pred = projection(hidden[:, :-horizon])
    target = hidden[:, horizon:].detach()
    return torch.nn.functional.mse_loss(pred.float(), target.float())
```

- [ ] **Step 4: Wire predictive aux through runner**

Import `predictive_auxiliary_loss`.

Add parameters to `_run_train_step`:

```python
    predictive_aux_weight: float = 0.0,
    predictive_aux_horizon: int = 0,
    predictive_aux_projection: torch.nn.Module | None = None,
```

Compute and backprop the auxiliary loss immediately after `hidden = model.encode(inputs)`
and before the LM-head helper runs. The fused CE helpers already backprop through
the encoder graph, so the aux backward needs `retain_graph=True` and must happen
while that graph is still alive:

```python
        aux = None
        if (
            predictive_aux_weight > 0.0
            and predictive_aux_horizon > 0
            and predictive_aux_projection is not None
        ):
            aux = predictive_auxiliary_loss(
                hidden,
                projection=predictive_aux_projection,
                horizon=predictive_aux_horizon,
            )
            if aux is not None:
                (float(predictive_aux_weight) * aux).backward(retain_graph=True)
```

Add these keyword arguments to `train_fast_for_budget`:

```python
    predictive_aux_weight: float = 0.0,
    predictive_aux_horizon: int = 0,
    predictive_aux_dim: int = 0,
```

In `train_fast_for_budget`, create the projection and its separate optimizer
before the loop. Keep it outside `model` so `_save_output_ckpt` cannot include it:

```python
    predictive_aux_projection = None
    predictive_aux_optimizer = None
    if predictive_aux_weight > 0.0 and predictive_aux_horizon > 0:
        dim = int(getattr(model, "dim", 0) or getattr(model.lm_head, "in_features"))
        aux_dim = int(predictive_aux_dim) if int(predictive_aux_dim) > 0 else dim
        if aux_dim != dim:
            raise ValueError("predictive_aux_dim must be 0 or equal to model dim in v1")
        predictive_aux_projection = torch.nn.Linear(dim, dim, bias=False).to(
            device=device,
            dtype=next(model.parameters()).dtype,
        )
        predictive_aux_optimizer = torch.optim.AdamW(
            predictive_aux_projection.parameters(),
            lr=float(getattr(optimizer, "param_groups", [{"lr": 0.0}])[0]["lr"]),
            weight_decay=0.0,
        )
```

Immediately after `optimizer.zero_grad(set_to_none=True)`, add:

```python
            if predictive_aux_optimizer is not None:
                predictive_aux_optimizer.zero_grad(set_to_none=True)
```

Pass the predictive arguments into `_run_train_step`.

Immediately after `optimizer.step()`, add:

```python
            if predictive_aux_optimizer is not None:
                predictive_aux_optimizer.step()
```

In the `train_fast_for_budget` call inside `run_condition`, pass:

```python
        predictive_aux_weight=float(config.get("predictive_aux_weight", 0.0)),
        predictive_aux_horizon=int(config.get("predictive_aux_horizon", 0)),
        predictive_aux_dim=int(config.get("predictive_aux_dim", 0)),
```

Add result diagnostics:

```python
            "predictive_aux": {
                "enabled": predictive_aux_projection is not None,
                "weight": float(predictive_aux_weight),
                "horizon": int(predictive_aux_horizon),
                "artifact_impact": "artifact_training_only",
            },
```

- [ ] **Step 5: Run tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_exp24_training_hooks.py::test_predictive_auxiliary_loss_uses_detached_future_hidden tests/test_exp23_fast_path.py::test_run_train_step_applies_predictive_auxiliary -q
```

Expected: targeted tests pass.

- [ ] **Step 6: Commit**

```bash
git add experiments/23_fast_path/training_hooks.py experiments/23_fast_path/runner_fast_path.py tests/test_exp24_training_hooks.py tests/test_exp23_fast_path.py
git commit -m "exp24: add predictive coding auxiliary hook"
```

## Task 7: Dreamworld Ring-Buffer Hidden-State Replay

**Files:**
- Modify: `src/chaoscontrol/model.py`
- Create: `experiments/23_fast_path/dreamworld.py`
- Modify: `experiments/23_fast_path/runner_fast_path.py`
- Modify: `tests/test_encode_forward_equivalence.py`
- Create: `tests/test_exp24_dreamworld.py`
- Modify: `tests/test_exp23_fast_path.py`

**Scope invariant:** Dreamworld v0 is teacher-forced hidden-state replay. It
must run one batched forward over cached real continuation tokens from cached
SSM state. Do not add token-by-token autoregressive sampling, generation
helpers, or a dream rollout loop in this task. Autoregressive dreams are a v1+
diagnostic because they break fast-path parity.

**State/token alignment invariant:** Cache state before the replay seed token,
not after it. With `prefix_tokens = p`, `capture_dream_entry()` must encode
`inputs[:, : p - 1]` and cache replay tokens
`inputs[:, p - 1 : p + replay_tokens]`. During replay, shifted CE consumes the
seed token once and supervises the next `replay_tokens` real tokens. Do not
cache state after `inputs[:, :p]` and then replay from `inputs[:, p - 1]`; that
double-consumes the boundary token.

- [ ] **Step 1: Write failing encode-state tests**

Append to `tests/test_encode_forward_equivalence.py`:

```python
    def test_encode_can_return_final_states(self, bare_ssm_model: ChaosStudentLM) -> None:
        model = bare_ssm_model
        inputs = _make_input(batch=2, seq=12, vocab=64, seed=4)

        with torch.no_grad():
            forward_out = model(inputs)
            hidden, final_states = model.encode(inputs, return_final_states=True)

        assert torch.equal(forward_out["hidden"], hidden)
        assert len(final_states) == len(model.layers)
        for got, expected in zip(final_states, forward_out["final_states"]):
            assert torch.equal(got, expected)

    def test_encode_accepts_initial_states(self, bare_ssm_model: ChaosStudentLM) -> None:
        model = bare_ssm_model
        inputs = _make_input(batch=2, seq=12, vocab=64, seed=5)
        initial_states = model.init_states(
            batch_size=2,
            device=inputs.device,
            dtype=next(model.parameters()).dtype,
        )
        for idx, state in enumerate(initial_states):
            state.fill_(0.1 * (idx + 1))

        with torch.no_grad():
            forward_out = model(inputs, initial_states=initial_states)
            hidden, final_states = model.encode(
                inputs,
                initial_states=initial_states,
                return_final_states=True,
            )

        assert torch.equal(forward_out["hidden"], hidden)
        for got, expected in zip(final_states, forward_out["final_states"]):
            assert torch.equal(got, expected)
```

- [ ] **Step 2: Run encode tests and verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_encode_forward_equivalence.py::TestEncodeBareSSMEquivalence::test_encode_can_return_final_states tests/test_encode_forward_equivalence.py::TestEncodeBareSSMEquivalence::test_encode_accepts_initial_states -q
```

Expected: both tests fail because `ChaosStudentLM.encode()` does not accept `return_final_states` or `initial_states`.

- [ ] **Step 3: Extend `ChaosStudentLM.encode()`**

In `src/chaoscontrol/model.py`, change the signature to:

```python
    def encode(
        self,
        input_ids: torch.Tensor,
        *,
        initial_states: list[torch.Tensor] | None = None,
        return_final_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
```

At the beginning of `encode()`, add the same length check used by `forward()`:

```python
        if initial_states is not None:
            if len(initial_states) != len(self.layers):
                raise ValueError(
                    f"initial_states length {len(initial_states)} does not match "
                    f"num_layers {len(self.layers)}"
                )
```

Replace the layer loop in `encode()` with this state-aware version:

```python
        use_ckpt = self.activation_checkpoint and torch.is_grad_enabled() and x.requires_grad
        final_states: list[torch.Tensor | None] = [None] * len(self.layers)
        for layer_idx in self._virtual_layer_indices:
            layer = self.layers[layer_idx]
            init_state = initial_states[layer_idx] if initial_states is not None else None
            if use_ckpt:
                result = _checkpoint(
                    layer,
                    x,
                    return_jacobian_stats=False,
                    initial_state=init_state,
                    return_final_state=return_final_states,
                    use_reentrant=False,
                )
            else:
                result = layer(
                    x,
                    return_jacobian_stats=False,
                    initial_state=init_state,
                    return_final_state=return_final_states,
                )
            if return_final_states:
                x, fstate = result
                final_states[layer_idx] = fstate
            else:
                x = result
```

At the end of `encode()`, return states only when requested:

```python
        if return_final_states:
            assert all(s is not None for s in final_states), (
                "final_states has unvisited slots — virtual layer indices did not "
                "cover every physical layer"
            )
            return x, final_states
        return x
```

Keep the default `model.encode(inputs)` behavior byte-identical for existing Exp23 calls.

- [ ] **Step 4: Write failing Dreamworld primitive tests**

Create `tests/test_exp24_dreamworld.py`:

```python
"""Tests for Exp24 Dreamworld hidden-state replay primitives."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import torch.nn as nn


REPO = Path(__file__).resolve().parents[1]
DREAM_PATH = REPO / "experiments" / "23_fast_path" / "dreamworld.py"


def _load_dreamworld():
    spec = importlib.util.spec_from_file_location("exp23_dreamworld", DREAM_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _TinyDreamModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(16, 4)
        self.final_norm = nn.Identity()
        self.lm_head = nn.Linear(4, 16, bias=False)
        self.layers = nn.ModuleList([nn.Identity()])
        self.encode_calls: list[dict[str, object]] = []

    def encode(
        self,
        inputs: torch.Tensor,
        *,
        initial_states: list[torch.Tensor] | None = None,
        return_final_states: bool = False,
    ):
        self.encode_calls.append({
            "shape": tuple(inputs.shape),
            "has_initial_states": initial_states is not None,
        })
        hidden = self.embed(inputs)
        if initial_states is None:
            carry = torch.zeros(inputs.size(0), 4, device=inputs.device)
        else:
            carry = initial_states[0].to(inputs.device)
        hidden = hidden + carry[:, None, :]
        final_state = hidden[:, -1, :].detach()
        if return_final_states:
            return hidden, [final_state]
        return hidden


def test_dream_replay_buffer_detaches_evicts_and_ages_entries():
    dream = _load_dreamworld()
    buffer = dream.DreamReplayBuffer(max_entries=2, max_age_steps=5)
    states = [torch.ones(2, 4, requires_grad=True)]
    tokens = torch.arange(10).view(2, 5)

    buffer.add(step=1, states=states, replay_tokens=tokens)
    buffer.add(step=2, states=[torch.full((2, 4), 2.0)], replay_tokens=tokens + 1)
    buffer.add(step=9, states=[torch.full((2, 4), 3.0)], replay_tokens=tokens + 2)

    assert len(buffer) == 1
    entry = buffer.sample(generator=torch.Generator().manual_seed(0), current_step=9)
    assert entry is not None
    assert entry.step == 9
    assert entry.states[0].requires_grad is False
    assert entry.replay_tokens.requires_grad is False


def test_build_dream_replay_tokens_slices_seed_plus_targets():
    dream = _load_dreamworld()
    inputs = torch.arange(2 * 10).view(2, 10)

    replay = dream.build_dream_replay_tokens(
        inputs,
        prefix_tokens=4,
        replay_tokens=3,
    )

    assert replay.tolist() == [
        [3, 4, 5, 6],
        [13, 14, 15, 16],
    ]


def test_capture_dream_entry_uses_state_before_replay_seed():
    dream = _load_dreamworld()
    model = _TinyDreamModel()
    inputs = torch.arange(2 * 10).view(2, 10) % 16

    entry = dream.capture_dream_entry(
        model,
        inputs,
        step=7,
        prefix_tokens=4,
        replay_tokens=3,
    )

    assert entry.step == 7
    assert entry.replay_tokens.tolist() == [
        [3, 4, 5, 6],
        [13, 14, 15, 0],
    ]
    assert model.encode_calls == [
        {"shape": (2, 3), "has_initial_states": False},
    ]


def test_dreamworld_replay_backward_uses_cached_initial_state():
    dream = _load_dreamworld()
    model = _TinyDreamModel()
    replay_tokens = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    entry = dream.DreamReplayEntry(
        step=0,
        states=[torch.ones(2, 4)],
        replay_tokens=replay_tokens,
    )

    loss = dream.dreamworld_replay_backward(
        model,
        entry=entry,
        weight=0.25,
    )

    assert loss.ndim == 0
    assert model.encode_calls == [
        {"shape": (2, 2), "has_initial_states": True},
    ]
    assert model.embed.weight.grad is not None
    assert model.lm_head.weight.grad is not None
```

- [ ] **Step 5: Run Dreamworld tests and verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_exp24_dreamworld.py -q
```

Expected: collection fails because `experiments/23_fast_path/dreamworld.py` does not exist.

- [ ] **Step 6: Implement Dreamworld primitives**

Create `experiments/23_fast_path/dreamworld.py`:

```python
"""Dreamworld hidden-state replay primitives for Exp24.

V0 is deliberately teacher-forced: cache compact SSM state immediately before
a replay seed token plus the real seed-and-target continuation, then replay
that continuation from the cached state with CE loss.
The replay pass is the same batched forward shape as waking training, only over
short cached continuations. Autoregressive generation is a diagnostic/later
variant; teacher forcing keeps the first measured arm on the fast path.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class DreamReplayEntry:
    step: int
    states: list[torch.Tensor]
    replay_tokens: torch.Tensor


class DreamReplayBuffer:
    def __init__(self, *, max_entries: int, max_age_steps: int) -> None:
        self.max_entries = int(max_entries)
        self.max_age_steps = int(max_age_steps)
        self._entries: deque[DreamReplayEntry] = deque()
        self.add_count = 0
        self.sample_count = 0
        self.drop_count = 0

    def __len__(self) -> int:
        return len(self._entries)

    def _drop_stale(self, *, current_step: int) -> None:
        if self.max_age_steps <= 0:
            return
        while self._entries and int(current_step) - self._entries[0].step > self.max_age_steps:
            self._entries.popleft()
            self.drop_count += 1

    def add(
        self,
        *,
        step: int,
        states: list[torch.Tensor],
        replay_tokens: torch.Tensor,
    ) -> None:
        self._drop_stale(current_step=step)
        entry = DreamReplayEntry(
            step=int(step),
            states=[state.detach().clone() for state in states],
            replay_tokens=replay_tokens.detach().clone(),
        )
        self._entries.append(entry)
        self.add_count += 1
        while len(self._entries) > self.max_entries:
            self._entries.popleft()
            self.drop_count += 1

    def sample(
        self,
        *,
        generator: torch.Generator,
        current_step: int,
    ) -> DreamReplayEntry | None:
        self._drop_stale(current_step=current_step)
        if not self._entries:
            return None
        idx = int(torch.randint(
            low=0,
            high=len(self._entries),
            size=(),
            generator=generator,
        ).item())
        self.sample_count += 1
        return self._entries[idx]

    def diagnostics(self, *, current_step: int) -> dict[str, int | float]:
        self._drop_stale(current_step=current_step)
        ages = [int(current_step) - entry.step for entry in self._entries]
        return {
            "size": len(self._entries),
            "max_entries": self.max_entries,
            "max_age_steps": self.max_age_steps,
            "add_count": self.add_count,
            "sample_count": self.sample_count,
            "drop_count": self.drop_count,
            "age_min": min(ages) if ages else 0,
            "age_max": max(ages) if ages else 0,
            "age_mean": float(sum(ages) / len(ages)) if ages else 0.0,
        }


def build_dream_replay_tokens(
    inputs: torch.Tensor,
    *,
    prefix_tokens: int,
    replay_tokens: int,
) -> torch.Tensor:
    """Return seed-plus-target tokens for teacher-forced replay.

    With `prefix_tokens = p`, the returned slice is
    `inputs[:, p - 1 : p + replay_tokens]`. The cached state must be produced
    by encoding `inputs[:, : p - 1]`, so replay consumes the seed token exactly
    once and predicts the next `replay_tokens` real tokens.
    """
    prefix = int(prefix_tokens)
    replay = int(replay_tokens)
    if prefix <= 0:
        raise ValueError(f"prefix_tokens must be positive, got {prefix}")
    if replay <= 0:
        raise ValueError(f"replay_tokens must be positive, got {replay}")
    end = prefix + replay
    if end > inputs.size(1):
        raise ValueError(
            f"prefix_tokens + replay_tokens must fit inside inputs: "
            f"{prefix} + {replay} > {inputs.size(1)}"
        )
    return inputs[:, prefix - 1:end].detach()


@torch.no_grad()
def capture_dream_entry(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    *,
    step: int,
    prefix_tokens: int,
    replay_tokens: int,
) -> DreamReplayEntry:
    prefix = int(prefix_tokens)
    if prefix <= 1:
        raise ValueError(
            "prefix_tokens must be at least 2 so Dreamworld can encode state "
            "before the replay seed token"
        )
    prefix_input = inputs[:, : prefix - 1]
    replay = build_dream_replay_tokens(
        inputs,
        prefix_tokens=prefix_tokens,
        replay_tokens=replay_tokens,
    )
    _hidden, states = model.encode(prefix_input, return_final_states=True)
    return DreamReplayEntry(
        step=int(step),
        states=[state.detach().clone() for state in states],
        replay_tokens=replay.detach().clone(),
    )


def dreamworld_replay_backward(
    model: torch.nn.Module,
    *,
    entry: DreamReplayEntry,
    weight: float,
) -> torch.Tensor:
    """Backprop one teacher-forced replay pass.

    This intentionally calls `model.encode()` once on the shifted cached real
    continuation. There is no autoregressive sampling loop in Dreamworld v0.
    """
    replay = entry.replay_tokens
    device = next(model.parameters()).device
    replay = replay.to(device=device, non_blocking=True)
    states = [state.to(device=device, non_blocking=True) for state in entry.states]
    replay_inputs = replay[:, :-1].to(dtype=torch.int32)
    targets = replay[:, 1:].to(dtype=torch.long)
    hidden = model.encode(replay_inputs, initial_states=states)
    logits = model.lm_head(model.final_norm(hidden))
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(),
        targets.reshape(-1),
        reduction="mean",
    )
    (float(weight) * loss).backward()
    return loss.detach()
```

- [ ] **Step 7: Thread Dreamworld through the runner**

In `runner_fast_path.py`, import:

```python
from dreamworld import (
    DreamReplayBuffer,
    capture_dream_entry,
    dreamworld_replay_backward,
)
```

Add these keyword arguments to `train_fast_for_budget`:

```python
    dreamworld_enabled: bool = False,
    dreamworld_cache_interval: int = 0,
    dreamworld_interval: int = 0,
    dreamworld_weight: float = 0.0,
    dreamworld_prefix_tokens: int = 128,
    dreamworld_replay_tokens: int = 64,
    dreamworld_buffer_size: int = 16,
    dreamworld_min_size: int = 2,
    dreamworld_max_age_steps: int = 256,
```

Add these keyword arguments to `_run_train_step`. The current signature already
uses a bare `*`, so keep these keyword-only and do not create or rely on
positional call sites:

```python
    dreamworld_entry: Any | None = None,
    dreamworld_weight: float = 0.0,
```

Inside `_run_train_step`, call Dreamworld replay before the existing DDP
all-reduce block so replay gradients are reduced together with waking gradients:

```python
    if dreamworld_entry is not None and dreamworld_weight > 0.0:
        dreamworld_replay_backward(
            model,
            entry=dreamworld_entry,
            weight=dreamworld_weight,
        )
```

Before the loop, create:

```python
    dream_buffer = (
        DreamReplayBuffer(
            max_entries=dreamworld_buffer_size,
            max_age_steps=dreamworld_max_age_steps,
        )
        if dreamworld_enabled
        else None
    )
```

After batch assembly and before `optimizer.zero_grad(set_to_none=True)`, cache entries:

```python
            if (
                dream_buffer is not None
                and dreamworld_cache_interval > 0
                and steps % dreamworld_cache_interval == 0
            ):
                entry = capture_dream_entry(
                    model,
                    inputs,
                    step=steps,
                    prefix_tokens=dreamworld_prefix_tokens,
                    replay_tokens=dreamworld_replay_tokens,
                )
                dream_buffer.add(
                    step=entry.step,
                    states=entry.states,
                    replay_tokens=entry.replay_tokens,
                )
```

Before calling `_run_train_step`, sample a replay entry:

```python
            dream_entry = None
            if (
                dream_buffer is not None
                and dreamworld_interval > 0
                and dreamworld_weight > 0.0
                and len(dream_buffer) >= int(dreamworld_min_size)
                and steps % dreamworld_interval == 0
            ):
                dream_entry = dream_buffer.sample(generator=rng, current_step=steps)
```

Pass `dreamworld_entry=dream_entry` and `dreamworld_weight=dreamworld_weight`
into `_run_train_step`.

Pass Dreamworld config from `run_condition` into `train_fast_for_budget`:

```python
        dreamworld_enabled=bool(config.get("dreamworld_enabled", False)),
        dreamworld_cache_interval=int(config.get("dreamworld_cache_interval", 0)),
        dreamworld_interval=int(config.get("dreamworld_interval", 0)),
        dreamworld_weight=float(config.get("dreamworld_weight", 0.0)),
        dreamworld_prefix_tokens=int(config.get("dreamworld_prefix_tokens", 128)),
        dreamworld_replay_tokens=int(config.get("dreamworld_replay_tokens", 64)),
        dreamworld_buffer_size=int(config.get("dreamworld_buffer_size", 16)),
        dreamworld_min_size=int(config.get("dreamworld_min_size", 2)),
        dreamworld_max_age_steps=int(config.get("dreamworld_max_age_steps", 256)),
```

Add result diagnostics:

```python
            "dreamworld": {
                "enabled": dream_buffer is not None,
                "weight": float(dreamworld_weight),
                "cache_interval": int(dreamworld_cache_interval),
                "dream_interval": int(dreamworld_interval),
                "prefix_tokens": int(dreamworld_prefix_tokens),
                "replay_tokens": int(dreamworld_replay_tokens),
                "artifact_impact": "artifact_training_only",
                "buffer": (
                    dream_buffer.diagnostics(current_step=steps)
                    if dream_buffer is not None
                    else None
                ),
            },
```

V0 intentionally implements hard age-out only through
`dreamworld_max_age_steps`. Do not implement soft-decay replay weights in this
task; if hard age-out looks brittle, add soft decay as a v1 mechanism with its
own config key and tests.

- [ ] **Step 8: Write runner integration test**

Append to `tests/test_exp23_fast_path.py`:

```python
def test_train_fast_for_budget_runs_dreamworld_replay(monkeypatch):
    mod = _load_runner_module()
    events = []

    class FakeDreamBuffer:
        def __init__(self, **kwargs):
            self.entries = []
            events.append(("buffer", kwargs))

        def __len__(self):
            return len(self.entries)

        def add(self, *, step, states, replay_tokens):
            self.entries.append((step, states, replay_tokens))
            events.append(("add", step))

        def sample(self, *, generator, current_step):
            events.append(("sample", current_step))
            return object()

        def diagnostics(self, *, current_step):
            return {"size": len(self.entries), "current_step": current_step}

    monkeypatch.setattr(mod, "DreamReplayBuffer", FakeDreamBuffer)
    monkeypatch.setattr(mod, "capture_dream_entry", lambda model, inputs, **kwargs: type("E", (), {
        "step": kwargs["step"],
        "states": [torch.zeros(inputs.size(0), 4)],
        "replay_tokens": inputs[:, :3],
    })())
    monkeypatch.setattr(mod, "dreamworld_replay_backward", lambda model, *, entry, weight: torch.tensor(0.1))

    model = _TinyTokenTrainModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    result = mod.train_fast_for_budget(
        model,
        train_tokens=torch.arange(128, dtype=torch.int16) % 6,
        train_num_tokens=128,
        stride=4,
        seq_len=6,
        batch_size=2,
        device=torch.device("cpu"),
        optimizer=optimizer,
        budget_seconds=300.0,
        chunk_size=2,
        grad_clip_norm=0.0,
        fused_grad_clip=False,
        rank=0,
        world_size=1,
        seed=123,
        precision="bf16",
        stop_check_interval=1,
        stop_margin_seconds=0.0,
        vocab_size=6,
        max_steps=2,
        prefetch_batches=False,
        dreamworld_enabled=True,
        dreamworld_cache_interval=1,
        dreamworld_interval=1,
        dreamworld_weight=0.25,
        dreamworld_prefix_tokens=3,
        dreamworld_replay_tokens=2,
        dreamworld_min_size=1,
    )

    assert ("sample", 0) in events
    assert result["mechanisms"]["dreamworld"]["enabled"] is True
    assert result["mechanisms"]["dreamworld"]["artifact_impact"] == "artifact_training_only"
```

- [ ] **Step 9: Run targeted Dreamworld tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_encode_forward_equivalence.py tests/test_exp24_dreamworld.py tests/test_exp23_fast_path.py::test_train_fast_for_budget_runs_dreamworld_replay -q
```

Expected: targeted tests pass.

- [ ] **Step 10: Commit**

```bash
git add src/chaoscontrol/model.py experiments/23_fast_path/dreamworld.py experiments/23_fast_path/runner_fast_path.py tests/test_encode_forward_equivalence.py tests/test_exp24_dreamworld.py tests/test_exp23_fast_path.py
git commit -m "exp24: add dreamworld hidden-state replay hook"
```

## Task 8: Exp24 Mechanism Matrices And Launcher

**Files:**
- Modify: `experiments/24_training_time_bundle/exp24.py`
- Create: `experiments/24_training_time_bundle/run_exp24.py`
- Modify: `tests/test_exp24_training_bundle.py`

- [ ] **Step 1: Write failing launcher/matrix tests**

Append to `tests/test_exp24_training_bundle.py`:

```python
def test_first_wave_mechanism_matrix_names_and_tags():
    mod = _load_exp24()
    entries = mod.build_first_wave_matrix(
        speed_config={"batch_size": 1024, "chunk_size": 64},
        seeds=[1337],
        world_size=8,
    )

    names = [entry["name"] for entry in entries]
    assert "exp24_fastslow_i16_a050_s1337" in names
    assert "exp24_spectral_d001_s001_s1337" in names
    assert "exp24_sgns_freeze100_s1337" in names
    assert "exp24_dreamworld_c4_i4_w025_s1337" in names

    fastslow = next(entry for entry in entries if entry["name"] == "exp24_fastslow_i16_a050_s1337")
    assert fastslow["fast_slow_enabled"] is True
    assert fastslow["artifact_impact"] == "artifact_training_only"

    spectral = next(entry for entry in entries if entry["name"] == "exp24_spectral_d001_s001_s1337")
    assert spectral["spectral_reg_lambda_dead"] == 0.01
    assert spectral["spectral_reg_lambda_sticky"] == 0.01
    assert spectral["artifact_impact"] == "artifact_changes_weights_only"

    freeze = next(entry for entry in entries if entry["name"] == "exp24_sgns_freeze100_s1337")
    assert freeze["embed_freeze_steps"] == 100
    assert freeze["artifact_impact"] == "artifact_changes_weights_only"

    dream = next(entry for entry in entries if entry["name"] == "exp24_dreamworld_c4_i4_w025_s1337")
    assert dream["dreamworld_enabled"] is True
    assert dream["dreamworld_cache_interval"] == 4
    assert dream["dreamworld_interval"] == 4
    assert dream["dreamworld_weight"] == 0.25
    assert dream["artifact_impact"] == "artifact_training_only"
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
.venv/bin/python -m pytest tests/test_exp24_training_bundle.py::test_first_wave_mechanism_matrix_names_and_tags -q
```

Expected: `AttributeError` because `build_first_wave_matrix` is missing.

- [ ] **Step 3: Add first-wave matrix builder**

In `exp24.py`, add:

```python
def build_first_wave_matrix(
    *,
    speed_config: dict[str, Any],
    world_size: int = 8,
    budget_seconds: float = 600.0,
    seeds: Sequence[int] = (1337,),
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    arms = [
        {
            "name_arm": "fastslow_i16_a050",
            "exp24_mechanism": "fast_slow",
            "artifact_impact": ARTIFACT_TRAINING_ONLY,
            "fast_slow_enabled": True,
            "fast_slow_interval": 16,
            "fast_slow_alpha": 0.50,
            "fast_slow_eval_copy": "slow",
        },
        {
            "name_arm": "spectral_d001_s001",
            "exp24_mechanism": "spectral",
            "artifact_impact": ARTIFACT_CHANGES_WEIGHTS_ONLY,
            "spectral_reg_lambda_dead": 0.01,
            "spectral_reg_lambda_sticky": 0.01,
        },
        {
            "name_arm": "sgns_freeze100",
            "exp24_mechanism": "sgns_freeze",
            "artifact_impact": ARTIFACT_CHANGES_WEIGHTS_ONLY,
            "embed_freeze_steps": 100,
        },
        {
            "name_arm": "dreamworld_c4_i4_w025",
            "exp24_mechanism": "dreamworld",
            "artifact_impact": ARTIFACT_TRAINING_ONLY,
            "dreamworld_enabled": True,
            "dreamworld_cache_interval": 4,
            "dreamworld_interval": 4,
            "dreamworld_weight": 0.25,
            "dreamworld_prefix_tokens": 128,
            "dreamworld_replay_tokens": 64,
            "dreamworld_buffer_size": 16,
            "dreamworld_min_size": 2,
            "dreamworld_max_age_steps": 256,
        },
    ]
    for arm in arms:
        for seed in seeds:
            entry = _base_entry(
                speed_config=speed_config,
                world_size=world_size,
                budget_seconds=budget_seconds,
            )
            entry.update(arm)
            name_arm = str(entry.pop("name_arm"))
            entry.update({
                "name": f"exp24_{name_arm}_s{int(seed)}",
                "seed": int(seed),
                "exp24_phase": "first_wave",
            })
            entries.append(entry)
    return entries
```

- [ ] **Step 4: Create launcher**

Create `experiments/24_training_time_bundle/run_exp24.py`:

```python
#!/usr/bin/env python3
"""Run Exp24 training-time bundle matrices."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXP23 = REPO / "experiments" / "23_fast_path"
EXP24 = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP23))
sys.path.insert(0, str(EXP24))
sys.path.insert(0, str(REPO / "src"))

from chaoscontrol.data import load_fineweb_tokens  # noqa: E402
from exp24 import (  # noqa: E402
    build_first_wave_matrix,
    build_phase_a_sampling_matrix,
    build_ring0_control_matrix,
    build_semantic_overhead_gate_matrix,
)
from fast_path import read_speed_config, write_matrix  # noqa: E402
from launch import run_matrix_entries  # noqa: E402


def _prebuild_cache(data_path: str) -> None:
    train_tokens, val_tokens = load_fineweb_tokens(data_path)
    print(
        "[exp24] data cache ready "
        f"train={int(train_tokens.numel()):,} val={int(val_tokens.numel()):,}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["ring0", "phase-a", "first-wave", "semantic-gate"], required=True)
    parser.add_argument("--speed-config", type=Path, required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--sp-model-path-16384", required=True)
    parser.add_argument("--results-dir", type=Path, default=EXP24 / "results")
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--budget", type=float, default=600.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1337])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-cache-prep", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    args = parser.parse_args(argv)

    speed_config = read_speed_config(args.speed_config)
    if args.phase == "ring0":
        entries = build_ring0_control_matrix(
            speed_config=speed_config,
            world_size=args.world_size,
            budget_seconds=args.budget,
            seeds=args.seeds,
        )
    elif args.phase == "phase-a":
        entries = build_phase_a_sampling_matrix(
            speed_config=speed_config,
            world_size=args.world_size,
            budget_seconds=args.budget,
            seeds=args.seeds,
        )
    elif args.phase == "semantic-gate":
        entries = build_semantic_overhead_gate_matrix(
            speed_config=speed_config,
            seed=args.seeds[0],
            world_size=args.world_size,
            budget_seconds=args.budget,
        )
    else:
        entries = build_first_wave_matrix(
            speed_config=speed_config,
            world_size=args.world_size,
            budget_seconds=args.budget,
            seeds=args.seeds,
        )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    write_matrix(args.results_dir / "matrix.json", entries)
    if not args.dry_run and not args.skip_cache_prep:
        _prebuild_cache(args.data_path)
    summary = run_matrix_entries(
        entries=entries,
        runner_path=EXP23 / "runner_fast_path.py",
        data_path=args.data_path,
        sp_model_paths={16384: args.sp_model_path_16384},
        results_dir=args.results_dir,
        world_size=args.world_size,
        limit=args.limit,
        dry_run=args.dry_run,
        skip_existing=args.skip_existing,
    )
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run tests and a dry-run command**

Run:

```bash
.venv/bin/python -m pytest tests/test_exp24_training_bundle.py -q
.venv/bin/python experiments/24_training_time_bundle/run_exp24.py \
  --phase first-wave \
  --speed-config experiments/23_fast_path/configs/base_seq_epoch_lr0064_full_corpus.yaml \
  --data-path baselines/parameter_golf/datasets/fineweb10B_sp16384 \
  --sp-model-path-16384 baselines/parameter_golf/tokenizers/fineweb_16384_bpe.model \
  --results-dir /tmp/exp24_first_wave_dry \
  --world-size 1 \
  --budget 5 \
  --dry-run
```

Expected: tests pass and dry-run prints torchrun commands without launching training.

- [ ] **Step 6: Commit**

```bash
git add experiments/24_training_time_bundle/exp24.py experiments/24_training_time_bundle/run_exp24.py tests/test_exp24_training_bundle.py
git commit -m "exp24: add first-wave launcher"
```

## Task 9: Result Metadata And Artifact Bookkeeping

**Files:**
- Modify: `experiments/23_fast_path/runner_fast_path.py`
- Modify: `experiments/23_fast_path/launch.py`
- Modify: `tests/test_exp23_fast_path.py`

- [ ] **Step 1: Write failing metadata test**

Append to `tests/test_exp23_fast_path.py`:

```python
def test_run_condition_result_preserves_exp24_artifact_metadata(monkeypatch, tmp_path):
    mod = _load_runner_module()

    monkeypatch.setattr(mod, "_init_distributed", lambda _world_size: (0, 1, 0))
    monkeypatch.setattr(mod, "_pick_device", lambda _rank, _device: torch.device("cpu"))
    monkeypatch.setattr(mod, "resolve_param_dtype", lambda _dtype, _device: torch.float32)
    monkeypatch.setattr(mod, "verify_diag_recurrence", lambda _device: None)
    monkeypatch.setattr(mod, "load_fineweb_tokens", lambda _path: (torch.arange(64, dtype=torch.int16), torch.arange(64, dtype=torch.int16)))
    monkeypatch.setattr(mod, "build_sentencepiece_luts", lambda *_args: (None, None, None))
    monkeypatch.setattr(mod, "choose_lm_starts_lazy", lambda **_kwargs: [])
    monkeypatch.setattr(mod, "build_model", lambda *_args: _TinyTokenTrainModel())
    monkeypatch.setattr(mod, "_apply_embed_init", lambda *_args: None)
    monkeypatch.setattr(mod, "_reject_unsupported", lambda _model: None)
    monkeypatch.setattr(mod, "_warmup", lambda **_kwargs: None)
    monkeypatch.setattr(mod, "_build_optimizer", lambda _config, model: torch.optim.SGD(model.parameters(), lr=0.01))
    monkeypatch.setattr(mod, "train_fast_for_budget", lambda *args, **kwargs: {
        "steps": 1,
        "elapsed_s": 1.0,
        "initial_loss": 1.0,
        "final_loss": 0.5,
        "aggregate_tokens_per_sec": 1.0,
        "peak_vram_mb": 0.0,
    })

    class FakeSP:
        def Load(self, _path):
            return True

    monkeypatch.setitem(sys.modules, "sentencepiece", type("M", (), {"SentencePieceProcessor": FakeSP}))

    result = mod.run_condition(
        {
            "name": "metadata_smoke",
            "vocab_size": 6,
            "seq_len": 3,
            "stride": 1,
            "batch_size": 2,
            "artifact_impact": "artifact_training_only",
            "submit_valid": False,
            "exp24_phase": "first_wave",
            "exp24_mechanism": "predictive_aux",
        },
        data_path="unused",
        sp_model_path="unused.model",
        budget_seconds=1.0,
        output_json=None,
        output_ckpt=None,
        world_size_override=1,
    )

    assert result["artifact"]["artifact_impact"] == "artifact_training_only"
    assert result["artifact"]["submit_valid"] is False
    assert result["artifact"]["artifact_bytes_estimate"] > 0
    assert result["exp24"]["phase"] == "first_wave"
    assert result["exp24"]["mechanism"] == "predictive_aux"
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
.venv/bin/python -m pytest tests/test_exp23_fast_path.py::test_run_condition_result_preserves_exp24_artifact_metadata -q
```

Expected: fails because result has no `artifact` or `exp24` keys.

- [ ] **Step 3: Add result metadata**

In `run_condition`, before result construction:

```python
    artifact = {
        "artifact_impact": str(config.get("artifact_impact", "artifact_changes_weights_only")),
        "submit_valid": bool(config.get("submit_valid", True)),
        "artifact_bytes_estimate": int(model.artifact_bytes()) if hasattr(model, "artifact_bytes") else int(model_params * 2),
        "compressed_artifact_bytes": config.get("compressed_artifact_bytes"),
    }
    exp24 = {
        "phase": config.get("exp24_phase"),
        "mechanism": config.get("exp24_mechanism"),
    }
```

Add both to `result`:

```python
        "artifact": artifact,
        "exp24": exp24,
```

In `launch.summarize_result_dir`, include the metadata in each ranked row:

```python
artifact = data.get("artifact") or {}
exp24 = data.get("exp24") or {}
"artifact_impact": artifact.get("artifact_impact"),
"submit_valid": artifact.get("submit_valid"),
"exp24_phase": exp24.get("phase"),
"exp24_mechanism": exp24.get("mechanism"),
```

- [ ] **Step 4: Run tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_exp23_fast_path.py::test_run_condition_result_preserves_exp24_artifact_metadata -q
```

Expected: targeted test passes.

- [ ] **Step 5: Commit**

```bash
git add experiments/23_fast_path/runner_fast_path.py experiments/23_fast_path/launch.py tests/test_exp23_fast_path.py
git commit -m "exp23: record exp24 artifact metadata"
```

## Task 10: Documentation And Final Verification

**Files:**
- Create: `experiments/24_training_time_bundle/README.md`
- Modify: `docs/plans/2026-04-22-exp24-training-time-bundle.md`

- [ ] **Step 1: Write README**

Create `experiments/24_training_time_bundle/README.md`:

```markdown
# Exp24 Training-Time Bundle

Exp24 tests training-time mechanisms on top of the current Exp23 fastest SSM
base. Evaluation remains the fixed post-training scorer; no eval-time TTT,
temporal heads, or scoring-time state changes belong in this bundle.

## Run Order

1. Ring 0 control: 2-3 seeds, 600s wall-clock, full validation after training.
2. Phase A sampling policy gate: no extra mechanism, same budget, compare to
   Ring 0 noise floor.
3. SemanticOptimizer overhead gate on 1xH100 before any 8xH100 semantic run.
4. First-wave mechanisms: fast/slow, spectral, predictive auxiliary,
   scheduled Dreamworld ring-buffer replay, fast/slow+Dreamworld, and
   fast/slow+Dreamworld+event_sleep.

## Dry Run

```bash
.venv/bin/python experiments/24_training_time_bundle/run_exp24.py \
  --phase first-wave \
  --speed-config experiments/23_fast_path/configs/base_seq_epoch_lr0064_full_corpus.yaml \
  --data-path baselines/parameter_golf/datasets/fineweb10B_sp16384 \
  --sp-model-path-16384 baselines/parameter_golf/tokenizers/fineweb_16384_bpe.model \
  --results-dir /tmp/exp24_first_wave_dry \
  --world-size 1 \
  --budget 5 \
  --dry-run
```

## Interpretation Rule

A mechanism whose BPB delta is smaller than the Ring 0 control sample standard
deviation is exploratory. It can motivate a follow-up; it is not a paper-quality
claim.

SGNS arms in this bundle answer the practical fast-path recipe question. They
do not replace the historical Exp21 semantic-vs-distributional controls.
```

- [ ] **Step 2: Link the implementation plan**

At the end of `docs/plans/2026-04-22-exp24-training-time-bundle.md`, add:

```markdown
## Implementation Plan

Implementation checklist:
`docs/superpowers/plans/2026-04-22-exp24-training-time-bundle-implementation.md`
```

- [ ] **Step 3: Run full targeted verification**

Run:

```bash
.venv/bin/python -m pytest tests/test_exp24_training_bundle.py tests/test_exp24_training_hooks.py tests/test_exp24_dreamworld.py tests/test_encode_forward_equivalence.py tests/test_exp23_fast_path.py -q
.venv/bin/python experiments/24_training_time_bundle/run_exp24.py \
  --phase ring0 \
  --speed-config experiments/23_fast_path/configs/base_seq_epoch_lr0064_full_corpus.yaml \
  --data-path baselines/parameter_golf/datasets/fineweb10B_sp16384 \
  --sp-model-path-16384 baselines/parameter_golf/tokenizers/fineweb_16384_bpe.model \
  --results-dir /tmp/exp24_ring0_dry \
  --world-size 1 \
  --budget 5 \
  --dry-run
```

Expected: tests pass and dry-run emits three Ring 0 commands for seeds `1337`, `2674`, and `4011`.

- [ ] **Step 4: Commit**

```bash
git add experiments/24_training_time_bundle/README.md docs/plans/2026-04-22-exp24-training-time-bundle.md
git commit -m "docs: document exp24 execution path"
```

## Final Operator Notes

- Do not provision 8xH100 until the dry-run matrix is inspected and the 1xH100 smoke for any new mechanism logs its diagnostics.
- Ring 0 is not optional if we want paper-quality interpretation; use at least two seeds if credits force triage.
- SemanticOptimizer is gated by measured wall-clock overhead, not by taste.
- Predictive auxiliary is `artifact_training_only`; if any checkpoint path saves its aux projection, mark that arm `artifact_invalid_until_stripped` until fixed.
- Dreamworld v0 is teacher-forced hidden-state replay from cached SSM states. Token-by-token autoregressive dreaming is a later diagnostic unless the ring-buffer CE arm earns more work.
- The Dreamworld unit tests verify single-GPU replay ordering. The first 8xH100 Dreamworld smoke should inspect per-rank final losses and gradient/all-reduce diagnostics for divergence, because there is no cheap local NCCL regression test in this plan.
- Shuffled epoch is a sampling-policy arm, not a record claim by itself. The record claim still depends on fixed full-validation scoring after training.
