# Exp22 Temporal Heads Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the minimum runnable Exp22 temporal-head eval path: parameter-free horizon heads, legality-safe gating helpers, same-horizon control helpers, and a script that emits metrics/summary JSON.

**Architecture:** Add a focused `temporal_heads.py` module under `src/chaoscontrol/eval_stream/`. It owns temporal-head scoring, uniform log-prob mixing, previous-chunk gating, and same-horizon virtual-depth config helpers. Add `scripts/run_exp22_temporal_heads.py` as a thin runner that reuses the Exp20 checkpoint/stream/budget contracts.

**Tech Stack:** PyTorch, existing `DeltaModulator`, `DocStreamer`, `BudgetTracker`, `EvalDeadline`, `compute_bpb`, pytest.

---

## File Structure

- Create `src/chaoscontrol/eval_stream/temporal_heads.py`: dataclasses, log-prob mixture, temporal chunk scoring, previous-chunk gate, same-horizon control config helper.
- Create `tests/test_eval_stream_temporal_heads.py`: unit tests for mixture math, single-head equivalence, independent states, gating semantics, same-horizon control helper.
- Create `scripts/run_exp22_temporal_heads.py`: CLI runner for score-only, single-horizon pilot, temporal heads, and same-horizon virtual-depth conditions. `gated_temporal_heads` must fail fast until the pre-registered gate is wired into the runner.
- Create `tests/test_run_exp22_temporal_heads.py`: subprocess smoke test on a tiny SentencePiece stream and tiny checkpoint.
- Modify `src/chaoscontrol/eval_stream/__init__.py` only if imports are already exported there; otherwise leave it untouched.

## Task 1: Temporal-Head Core

**Files:**
- Create: `src/chaoscontrol/eval_stream/temporal_heads.py`
- Test: `tests/test_eval_stream_temporal_heads.py`

- [x] **Step 1: Write failing tests for uniform log-prob mixing and single-head identity**

Add tests:

```python
def test_uniform_logprob_mixture_one_head_is_identity():
    logp = torch.log_softmax(torch.randn(2, 3, 5), dim=-1)
    mixed = uniform_logprob_mixture([logp])
    assert torch.equal(mixed, logp)


def test_uniform_logprob_mixture_matches_probability_average():
    logits_a = torch.tensor([[[2.0, 0.0]]])
    logits_b = torch.tensor([[[0.0, 2.0]]])
    logp_a = torch.log_softmax(logits_a, dim=-1)
    logp_b = torch.log_softmax(logits_b, dim=-1)

    mixed = uniform_logprob_mixture([logp_a, logp_b])
    expected = torch.log((logp_a.exp() + logp_b.exp()) / 2.0)

    assert torch.allclose(mixed, expected)
```

Run:

```bash
pytest tests/test_eval_stream_temporal_heads.py::test_uniform_logprob_mixture_one_head_is_identity tests/test_eval_stream_temporal_heads.py::test_uniform_logprob_mixture_matches_probability_average -q
```

Expected: import failure because `temporal_heads.py` does not exist.

- [x] **Step 2: Implement `uniform_logprob_mixture`**

Create:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


def uniform_logprob_mixture(log_probs: list[torch.Tensor]) -> torch.Tensor:
    if not log_probs:
        raise ValueError("uniform_logprob_mixture requires at least one tensor")
    if len(log_probs) == 1:
        return log_probs[0]
    weight = torch.log(torch.tensor(1.0 / len(log_probs), device=log_probs[0].device, dtype=log_probs[0].dtype))
    stacked = torch.stack([lp + weight for lp in log_probs], dim=0)
    return torch.logsumexp(stacked, dim=0)
```

Run the same tests. Expected: both pass.

- [x] **Step 3: Write failing tests for temporal-head scoring identity and independent states**

Add tests that instantiate a tiny `ChaosStudentLM`, score a chunk through `score_temporal_heads_chunk` with `horizon_shifts=(0.0,)`, and compare summed CE to direct model scoring. Add a second test that uses `horizon_shifts=(-0.5, 0.0, 0.5)` and asserts every head has its own returned state list object and tensors do not share storage.

Run:

```bash
pytest tests/test_eval_stream_temporal_heads.py::test_single_zero_shift_matches_direct_model tests/test_eval_stream_temporal_heads.py::test_temporal_heads_keep_independent_states -q
```

Expected: import failure for `TemporalHeadConfig` / `score_temporal_heads_chunk`.

- [x] **Step 4: Implement temporal-head scoring**

Add:

```python
@dataclass(frozen=True)
class TemporalHeadConfig:
    horizon_shifts: tuple[float, ...] = (-0.5, 0.0, 0.5)
    horizon_knob: Literal["log_a_shift"] = "log_a_shift"
    mixer: Literal["uniform_logprob"] = "uniform_logprob"
    policy: Literal["always", "previous_chunk_priority"] = "always"
    threshold: float | None = None


@dataclass
class TemporalHeadChunkResult:
    loss_nats: float
    tokens_scored: int
    mixed_log_probs: torch.Tensor
    final_states_by_shift: dict[float, list[torch.Tensor]]
    per_head_loss_nats: dict[float, float]
```

Implement `score_temporal_heads_chunk(model, chunk, states_by_shift, cfg)` using `DeltaModulator(model, log_a_shift=shift)`, `torch.no_grad()`, full-chunk model forward, `F.cross_entropy(..., reduction="sum")`, `torch.log_softmax`, and `uniform_logprob_mixture`.

Run:

```bash
pytest tests/test_eval_stream_temporal_heads.py -q
```

Expected: current tests pass.

## Task 2: Gating And Same-Horizon Control Helpers

**Files:**
- Modify: `src/chaoscontrol/eval_stream/temporal_heads.py`
- Modify: `tests/test_eval_stream_temporal_heads.py`

- [x] **Step 1: Write failing tests for primary gate not using head disagreement**

Add tests for `PreviousChunkPriorityGate`:

```python
def test_primary_gate_ignores_stale_head_disagreement_after_base_only_chunk():
    gate = PreviousChunkPriorityGate(threshold=1.0, entropy_weight=1.0, loss_spike_weight=0.0, state_delta_weight=0.0)
    gate.update_after_chunk(entropy=0.5, loss_spike=0.0, state_delta_norm=0.0, head_disagreement=999.0, temporal_heads_ran=False)
    assert not gate.should_run(extra_cost_seconds=0.1, slack_remaining_seconds=1.0)
```

Run:

```bash
pytest tests/test_eval_stream_temporal_heads.py::test_primary_gate_ignores_stale_head_disagreement_after_base_only_chunk -q
```

Expected: import failure.

- [x] **Step 2: Implement `PreviousChunkPriorityGate`**

Implement a dataclass with weights for entropy, loss spike, and state delta. Store only previous scalar features. Ignore `head_disagreement` unless `use_disagreement_ema=True`, which defaults to `False`.

Run gate tests. Expected: pass.

- [x] **Step 3: Write failing tests for same-horizon config helper**

Add test:

```python
def test_same_horizon_virtual_depth_config_uses_all_layers_when_no_group():
    cfg = {"vocab_size": 64, "dim": 16, "num_layers": 3, "block_type": "ssm", "a_mode": "diag"}
    out = make_same_horizon_virtual_depth_config(cfg, depth_recurrence_count=3)
    assert out["depth_recurrence_shared_layers"] == [0, 1, 2]
    assert out["depth_recurrence_count"] == 3
```

Run expected import failure.

- [x] **Step 4: Implement `make_same_horizon_virtual_depth_config`**

Copy the config dict, preserve an existing non-empty `depth_recurrence_shared_layers`, otherwise set all physical layer indices. Set `depth_recurrence_count`.

Run:

```bash
pytest tests/test_eval_stream_temporal_heads.py -q
```

Expected: pass.

## Task 3: Exp22 Runner

**Files:**
- Create: `scripts/run_exp22_temporal_heads.py`
- Test: `tests/test_run_exp22_temporal_heads.py`

- [x] **Step 1: Write failing subprocess smoke test**

Create a tiny SentencePiece model, JSONL stream, tiny checkpoint, and config:

```json
{
  "condition": "temporal_heads",
  "horizon_shifts": [-0.5, 0.0, 0.5],
  "chunk_size": 32,
  "max_docs": 2
}
```

Run:

```bash
pytest tests/test_run_exp22_temporal_heads.py::test_exp22_runner_writes_metrics_and_summary -q
```

Expected: script missing.

- [x] **Step 2: Implement script bootstrap, config dataclass, model load, and stream loop**

Mirror `scripts/run_exp20_eval.py` bootstrap. Support conditions:

- `score_only`
- `single_horizon`
- `temporal_heads`
- `gated_temporal_heads` as a fail-fast guard until routing is implemented
- `same_horizon_virtual_depth`

For `temporal_heads`, initialize one state bundle per horizon shift at each doc boundary, score chunks via `score_temporal_heads_chunk`, and charge each chunk's elapsed time to `BudgetTracker.add_score_time`.

For `same_horizon_virtual_depth`, modify the checkpoint config through `make_same_horizon_virtual_depth_config`, load the same state dict strictly, and score through the normal full-chunk path.

Write per-doc JSONL records with `doc_id`, `bpb`, `tokens`, `wall_ms`, and condition metadata. Write summary JSON with `condition`, `horizon_shifts`, `docs_scored`, `tokens_scored`, budget summary fields, and `evidence_label="exploratory"` unless config explicitly supplies a stronger label.

Run smoke test. Expected: pass.

- [x] **Step 3: Run focused verification**

Run:

```bash
pytest tests/test_eval_stream_temporal_heads.py tests/test_run_exp22_temporal_heads.py tests/test_eval_stream_delta_mod.py tests/test_run_exp20_eval.py -q
```

Expected: all pass.

## Task 4: Experiment Directory

**Files:**
- Create: `experiments/22_temporal_heads/README.md`
- Create: `experiments/22_temporal_heads/configs/phase0_single_horizon_log_a_m050.json`
- Create: `experiments/22_temporal_heads/configs/phaseA_score_only.json`
- Create: `experiments/22_temporal_heads/configs/phaseA_temporal_heads_3_uniform.json`
- Create: `experiments/22_temporal_heads/configs/phaseA_same_horizon_virtual_depth.json`

- [x] **Step 1: Add README and config templates**

README must state that templates require real checkpoint/data paths. Configs should include intentionally invalid sentinel strings, such as `"/path/to/final_ckpt.pt"`, and the README must say they are templates.

- [x] **Step 2: Verify docs/config files are valid JSON**

Run:

```bash
python3 -m json.tool experiments/22_temporal_heads/configs/phase0_single_horizon_log_a_m050.json >/tmp/phase0.json
python3 -m json.tool experiments/22_temporal_heads/configs/phaseA_temporal_heads_3_uniform.json >/tmp/phaseA.json
```

Expected: both commands exit 0.

## Final Verification

Run:

```bash
pytest tests/test_eval_stream_temporal_heads.py tests/test_run_exp22_temporal_heads.py tests/test_eval_stream_delta_mod.py tests/test_run_exp20_eval.py -q
python3 -m json.tool experiments/22_temporal_heads/configs/phase0_single_horizon_log_a_m050.json >/tmp/phase0.json
python3 -m json.tool experiments/22_temporal_heads/configs/phaseA_temporal_heads_3_uniform.json >/tmp/phaseA.json
git diff --check
```

Expected: tests pass, JSON validates, diff check passes.
