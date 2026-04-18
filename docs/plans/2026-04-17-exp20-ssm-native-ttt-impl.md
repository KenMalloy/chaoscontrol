# Exp 20 — SSM-Native TTT Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Parameter-Golf-compliant 50k-doc replay harness and execute a three-axis SSM-native TTT ablation (what-adapts × what-persists × Δ-modulation) producing a bpb-vs-compute Pareto curve on an Exp 19 base checkpoint.

**Architecture:** New `src/chaoscontrol/eval_stream/` package (sibling of `runner.py`, not a subclass). Five components — `DocStreamer`, `LegalityController`, `TTTRunner`, `DeltaModulator`, `MetricsCollector` — composed by a driver script `scripts/run_exp20_eval.py`. Legality enforced structurally via chunk-level Score-then-Adapt; no flags that can be set wrong.

**Tech Stack:** PyTorch, bf16 autocast, Muon optimizer (existing `src/chaoscontrol/optim/muon.py`), ChaosStudentLM + ChaosSSMCore (existing), manual DDP all-reduce (existing `src/chaoscontrol/distributed.py`). JSONL logging + parquet post-hoc. pytest for tests.

**Design doc:** `docs/plans/2026-04-17-exp20-ssm-native-ttt-design.md`.

**Key existing code to reference:**
- `src/chaoscontrol/model.py:487` — ChaosStudentLM. `forward(input_ids) -> dict`, `embed`, `lm_head`, `.blocks`.
- `src/chaoscontrol/core.py:232` — ChaosSSMCore with `log_a`, `delta_proj`, `in_proj`, `select_proj`, `gate_proj`, `out_proj`, `step()`, `forward()`.
- `src/chaoscontrol/evaluation.py:30` — `compute_bpb(total_ce_nats, total_raw_bytes)`; do NOT re-implement.
- `src/chaoscontrol/data.py` — tokenized-data loaders (reuse at doc granularity).
- `src/chaoscontrol/optim/muon.py` — Muon; bf16 Newton-Schulz on CUDA.
- `src/chaoscontrol/distributed.py` — manual all-reduce (seed-parallel mode does NOT need it; DDP mode in Phase G does).

---

## Pre-flight: worktree

Use `superpowers:using-git-worktrees` to create an isolated worktree for this work before Task 1. Branch name: `exp20-ssm-native-ttt`.

---

## Phase I — Harness (Tasks 1-10, plus new Task 3.5)

Days 1-2 of the post-Exp-19 calendar. Build must complete before any ablation run.

**Task map (updated after 2026-04-17 dry-run review):**

| # | Task | Notes |
|---|---|---|
| 1 | Scaffold `eval_stream/` types | baseline |
| 2 | `DocStreamer` (JSONL + on-the-fly SP tokenization) | **retrofitted** — see Task 2 Retrofit note |
| 3 | `MetricsCollector` | baseline |
| 3.5 | **Thread `initial_states` through core + model forward** | **NEW — shared-code change; prerequisite for real Axis 2 persistence** |
| 4 | `DeltaModulator` | + Axis 1/3 compatibility assertion |
| 5 | `LegalityController` | stable chunk hash, empty-CE guard, cross-doc re-adapt test |
| 6 | Param-group selector (`TTTRunner`) | exact-match embed, `trainable_h0` pattern, `all` coverage |
| 7 | Persistence modes (`StateManager`, `trainable_h0`) | depends on Task 3.5; device+dtype placement; gradient-preserving init |
| 8 | Driver `scripts/run_exp20_eval.py` | single-Muon optimizer; strict load; CUDA seed + LOCAL_RANK |
| 8.5 | **Exp 20b slack-budget accounting** | summary JSON; score-only floor; gradient TTT slack guard |
| 9 | Phase A smoke (CPU, bit-exact) | adds state-plumbing canary test |
| 10 | Pod smoke vs Exp 18 Test 4b | requires `configs/exp20/smoke_test4b.json`; re-run baseline eval path for apples-to-apples gate; pre-push cleanliness check |

Task 3.5 is inserted between Task 3 and Task 4; Tasks 4 through 10 retain their original numbers.

### Task 8.5: Exp 20b slack-budget accounting

**Files:**
- Create: `src/chaoscontrol/eval_stream/budget.py`
- Modify: `src/chaoscontrol/eval_stream/types.py`
- Modify: `src/chaoscontrol/eval_stream/__init__.py`
- Modify: `scripts/run_exp20_eval.py`
- Test: `tests/test_eval_stream_budget.py`
- Test: `tests/test_run_exp20_eval.py`

**Behavior:**
- Score-only runs (`adapt_set="none"` or `steps_per_chunk=0`) write a summary where `score_floor_seconds` is the whole elapsed eval time.
- TTT runs accept `score_floor_seconds` and `safety_margin_seconds` in config.
- Usable adaptation time is `budget_seconds - score_floor_seconds - safety_margin_seconds`, clamped at zero.
- Gradient adaptation is skipped after the slack budget is exhausted; scoring continues until the normal total budget/doc/collapse gates stop the run.

**Smoke commands:**

```bash
.venv/bin/python -m pytest tests/test_eval_stream_budget.py tests/test_run_exp20_eval.py -q
```

**Primary output fields:**

```json
{
  "score_floor_seconds": 410.0,
  "usable_ttt_budget_seconds": 160.0,
  "ttt_budget_used_seconds": 155.0,
  "slack_remaining_seconds": 5.0,
  "score_wall_seconds": 390.0,
  "adapt_wall_seconds": 155.0,
  "other_wall_seconds": 20.0
}
```


### Task 1: Scaffold `eval_stream/` package

**Files:**
- Create: `src/chaoscontrol/eval_stream/__init__.py`
- Create: `src/chaoscontrol/eval_stream/types.py`
- Create: `tests/test_eval_stream_types.py`

**Step 1: Write failing test**

```python
# tests/test_eval_stream_types.py
from chaoscontrol.eval_stream.types import DocRecord, ChunkRecord, RunConfig


def test_docrecord_fields():
    rec = DocRecord(doc_id=0, tokens=[1, 2, 3], raw_bytes=10)
    assert rec.doc_id == 0
    assert len(rec.tokens) == 3


def test_runconfig_defaults():
    cfg = RunConfig()
    assert cfg.chunk_size == 256
    assert cfg.steps_per_chunk == 1
    assert cfg.adapt_set == "none"
    assert cfg.persistence_mode == "reset"
    assert cfg.delta_scale == 1.0
    assert cfg.log_a_shift == 0.0
    assert cfg.persistent_muon_moments is False
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_eval_stream_types.py -v`
Expected: FAIL with `ModuleNotFoundError`.

**Step 3: Implement types**

```python
# src/chaoscontrol/eval_stream/types.py
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class DocRecord:
    doc_id: int
    tokens: list[int]
    raw_bytes: int  # needed for bpb denominator; property of text, not tokenizer


@dataclass
class ChunkRecord:
    doc_id: int
    chunk_idx: int
    tokens: list[int]
    loss_before_adapt: float  # score-before-update loss
    loss_after_adapt: float | None  # None if no weight TTT applied


@dataclass
class RunConfig:
    # Axis 1 — what adapts
    adapt_set: str = "none"  # none, log_a, delta_proj, log_a+delta_proj, B_side, C_side, embed_rows_seen, lm_head, lora_r8, all
    # Axis 2 — what persists across doc boundaries
    persistence_mode: str = "reset"  # reset, carry_state, carry_weights, carry_both, trainable_h0, trainable_h0+carry
    # Axis 3 — Δ modulation (no-grad)
    delta_scale: float = 1.0
    log_a_shift: float = 0.0
    # Schedule
    chunk_size: int = 256  # tokens; whole_doc = -1
    steps_per_chunk: int = 1
    eval_lr: float = 0.064
    persistent_muon_moments: bool = False
    warmup_steps: int = 20  # Param Golf contract: 20 warmup + state restore pre-timer
    # Run
    seed: int = 0
    max_docs: int = 50_000
    budget_seconds: float = 600.0
    checkpoint_path: str = ""
    output_path: str = ""
```

```python
# src/chaoscontrol/eval_stream/__init__.py
from chaoscontrol.eval_stream.types import DocRecord, ChunkRecord, RunConfig

__all__ = ["DocRecord", "ChunkRecord", "RunConfig"]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_eval_stream_types.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/chaoscontrol/eval_stream/ tests/test_eval_stream_types.py
git commit -m "exp20(harness): scaffold eval_stream package with RunConfig types"
```

---

### Task 2: `DocStreamer` — iterate FineWeb eval docs

**Retrofit note (2026-04-17):** the original spec had `DocStreamer` source from `.bin` shards and split on an EOS sentinel. The canonical SP shards built by `scripts/build_sp_shards.py` set `append_eos=False` (see `scripts/build_sp_shards.py:348` and its module docstring, line 17) — there is **no** EOS between docs inside our shards. EOS-splitting would fail silently (one giant "doc" per shard).

Source instead from `docs_selected.jsonl` (the raw-JSONL path used at tokenization time) + an SP tokenizer handle, tokenizing on the fly. This matches how the shards themselves are built, keeps the eval split disjoint by choosing a held-out slice of the JSONL, and removes the need for an in-band sentinel. Throughput is a non-issue at 50k docs.

**Task 2 already executed on this worktree; the DocStreamer and its tests must be rewritten per this retrofit before any downstream task uses it.** The signature changes: `DocStreamer(jsonl_paths: list[Path], sp_model_path: Path, max_docs: int = 50_000)`.

The `__iter__` filter pipeline mirrors `scripts/build_sp_shards.py:_iter_docs` (strip, skip blank lines, skip `json.JSONDecodeError`, skip missing/empty `"text"`) so eval `doc_id` is consistent with training ordering — diverging filters would silently mis-align the stream.

**Files:**
- Create: `src/chaoscontrol/eval_stream/doc_stream.py`
- Create: `tests/test_eval_stream_doc_stream.py`

**Context:** FineWeb docs live in JSONL (`docs_selected.jsonl`) one doc per line with a `"text"` field. SP tokenization happens on the fly with a persistent `SentencePieceProcessor` handle. `doc_id` is zero-based across the provided JSONL paths in given order. Eval-split disjointness vs Exp 19 train is enforced by the caller choosing non-overlapping JSONL paths. For raw-bytes (the bpb denominator) we use `len(text.encode("utf-8"))` — exact, not an estimate.

**Step 1: Write failing test**

Tests use a tiny SP model trained in `tmp_path` (keeps the test hermetic). If the repo already ships a fixture SP model under `tests/fixtures/` or an existing SP model path visible from tests — check `grep -r "SentencePiece" tests/` and the build_sp_shards tests — prefer the shipped model. Otherwise train a 64-piece SP model on a handful of synthetic lines in `tmp_path` via the `sentencepiece` training API.

```python
# tests/test_eval_stream_doc_stream.py
import json
from pathlib import Path

import pytest
import sentencepiece as spm

from chaoscontrol.eval_stream.doc_stream import DocStreamer


@pytest.fixture
def sp_model(tmp_path: Path) -> Path:
    """Train a tiny SP model for hermetic testing."""
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join([
        "the quick brown fox jumps over the lazy dog",
        "sphinx of black quartz judge my vow",
        "pack my box with five dozen liquor jugs",
    ] * 50))
    model_prefix = tmp_path / "sp"
    spm.SentencePieceTrainer.Train(
        input=str(corpus),
        model_prefix=str(model_prefix),
        vocab_size=64,
        character_coverage=1.0,
        model_type="bpe",
    )
    return Path(f"{model_prefix}.model")


def _write_jsonl(path: Path, texts: list[str]) -> None:
    with path.open("w") as fh:
        for t in texts:
            fh.write(json.dumps({"text": t}) + "\n")


def test_iterates_docs_in_order(tmp_path, sp_model):
    jsonl = tmp_path / "docs.jsonl"
    _write_jsonl(jsonl, ["alpha beta gamma", "delta epsilon", "zeta"])

    docs = list(DocStreamer(jsonl_paths=[jsonl], sp_model_path=sp_model, max_docs=10))

    assert len(docs) == 3
    assert [d.doc_id for d in docs] == [0, 1, 2]
    assert all(len(d.tokens) > 0 for d in docs)


def test_respects_max_docs(tmp_path, sp_model):
    jsonl = tmp_path / "docs.jsonl"
    _write_jsonl(jsonl, [f"doc number {i}" for i in range(8)])
    docs = list(DocStreamer(jsonl_paths=[jsonl], sp_model_path=sp_model, max_docs=2))
    assert len(docs) == 2


def test_raw_bytes_equal_utf8_length(tmp_path, sp_model):
    jsonl = tmp_path / "docs.jsonl"
    text = "hello"
    _write_jsonl(jsonl, [text])
    docs = list(DocStreamer(jsonl_paths=[jsonl], sp_model_path=sp_model, max_docs=1))
    assert docs[0].raw_bytes == len(text.encode("utf-8"))


def test_doc_id_continues_across_jsonl_files(tmp_path, sp_model):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    _write_jsonl(a, ["one", "two"])
    _write_jsonl(b, ["three"])
    docs = list(DocStreamer(jsonl_paths=[a, b], sp_model_path=sp_model, max_docs=10))
    assert [d.doc_id for d in docs] == [0, 1, 2]


def test_empty_text_is_skipped(tmp_path, sp_model):
    jsonl = tmp_path / "docs.jsonl"
    _write_jsonl(jsonl, ["", "real doc"])
    docs = list(DocStreamer(jsonl_paths=[jsonl], sp_model_path=sp_model, max_docs=10))
    # empty text tokenizes to [] which DocStreamer skips; doc_id for the real doc is 0
    assert len(docs) == 1
    assert docs[0].doc_id == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_eval_stream_doc_stream.py -v`
Expected: FAIL `ModuleNotFoundError`.

**Step 3: Implement**

```python
# src/chaoscontrol/eval_stream/doc_stream.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterator

import sentencepiece as spm

from chaoscontrol.eval_stream.types import DocRecord


class DocStreamer:
    """Iterates docs from FineWeb JSONL files. Tokenizes each line on the fly
    with a persistent SP handle.

    Canonical SP shards are built with `append_eos=False` (see
    `scripts/build_sp_shards.py:348`) — there is no EOS sentinel inside them,
    so we must source from the raw JSONL that feeds the shard builder rather
    than the .bin shards themselves. doc_id is zero-based, counted across the
    provided JSONL files in given order.

    Doc filtering mirrors the shard builder's `_iter_docs` pipeline (blank
    lines, JSONDecodeError, missing/empty `"text"` field) so our doc_id
    sequence is consistent with training's. Diverging filters would silently
    mis-align eval ordering.

    Eval-split disjointness vs Exp 19 train is enforced by the caller
    choosing non-overlapping JSONL paths.

    Not safely re-iterable — each call to `__iter__` restarts doc_id at 0.
    If you iterate twice, downstream doc_id-keyed metrics will collide.
    """

    def __init__(
        self,
        *,
        jsonl_paths: list[Path],
        sp_model_path: Path,
        max_docs: int = 50_000,
    ) -> None:
        self.jsonl_paths = [Path(p) for p in jsonl_paths]
        self.sp = spm.SentencePieceProcessor(model_file=str(sp_model_path))
        self.max_docs = max_docs

    def __iter__(self) -> Iterator[DocRecord]:
        doc_id = 0
        for p in self.jsonl_paths:
            with open(p, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = obj.get("text", "")
                    if not text:
                        continue
                    tokens = self.sp.encode(text, out_type=int)
                    if not tokens:
                        # SP could still produce [] on all-whitespace / unknowable input;
                        # keep the guard so we don't yield a DocRecord with zero tokens.
                        continue
                    yield DocRecord(
                        doc_id=doc_id,
                        tokens=tokens,
                        raw_bytes=len(text.encode("utf-8")),
                    )
                    doc_id += 1
                    if doc_id >= self.max_docs:
                        return
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_eval_stream_doc_stream.py -v`
Expected: 8 PASS.

**Step 5: Commit**

```bash
git add src/chaoscontrol/eval_stream/doc_stream.py tests/test_eval_stream_doc_stream.py
git commit -m "exp20(harness): DocStreamer iterates FineWeb JSONL with on-the-fly SP tokenization"
```

---

### Task 3: `MetricsCollector` — JSONL logging

**Files:**
- Create: `src/chaoscontrol/eval_stream/metrics.py`
- Create: `tests/test_eval_stream_metrics.py`

**Step 1: Write failing test**

```python
# tests/test_eval_stream_metrics.py
import json
from pathlib import Path
from chaoscontrol.eval_stream.metrics import MetricsCollector


def test_writes_per_doc_record(tmp_path):
    out = tmp_path / "metrics.jsonl"
    collector = MetricsCollector(output_path=out)
    collector.record_doc(
        doc_id=0, bpb=1.5, tokens=128, loss_before=0.5, loss_after=0.48,
        step_count=2, wall_ms=123.4, grad_norm=0.8, state_norm=1.1,
    )
    collector.close()
    lines = out.read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["doc_id"] == 0
    assert rec["bpb"] == 1.5
    assert rec["state_norm"] == 1.1


def test_stability_gate_flags_collapse(tmp_path):
    out = tmp_path / "metrics.jsonl"
    collector = MetricsCollector(
        output_path=out, stability_window=5, stability_sd_threshold=3.0,
    )
    # Steady losses then a spike persisting 5 docs
    for i in range(10):
        loss = 2.0 + (0.01 * i) + (20.0 if i >= 5 else 0.0)
        collector.record_doc(doc_id=i, bpb=loss, tokens=100, loss_before=loss,
                             loss_after=None, step_count=0, wall_ms=1.0,
                             grad_norm=0.0, state_norm=1.0)
    assert collector.collapsed is True


def test_baseline_frozen_across_long_run(tmp_path):
    """Baseline is frozen to the first `stability_window` docs; it must not drift.

    Regression test for the bug where `deque(maxlen=N)[:window]` silently
    shifted the baseline past doc N — the gate's detection window and Exp 20's
    collapse-detection window overlap exactly at doc 10K-30K.
    """
    out = tmp_path / "metrics.jsonl"
    # Small window, small threshold, many docs.
    collector = MetricsCollector(
        output_path=out, stability_window=10, stability_sd_threshold=3.0,
    )
    # First 10 docs: low steady loss → baseline mean ~1.0, sd ~0.01
    for i in range(10):
        collector.record_doc(doc_id=i, bpb=1.0, tokens=1, loss_before=1.0 + 0.01 * (i % 2),
                             loss_after=None, step_count=0, wall_ms=1.0,
                             grad_norm=0.0, state_norm=1.0)
    # Docs 10..9999: slowly drift upward to a new steady state around 1.5 —
    # a transformer-gradient-style slow drift that a rolling baseline would
    # absorb and stop flagging. Frozen baseline keeps flagging.
    for i in range(10, 10_000):
        collector.record_doc(doc_id=i, bpb=1.5, tokens=1, loss_before=1.5,
                             loss_after=None, step_count=0, wall_ms=1.0,
                             grad_norm=0.0, state_norm=1.0)
    # With a frozen baseline and enough consecutive drift, collapsed must be True.
    assert collector.collapsed is True


def test_gate_does_not_fire_on_steady_model(tmp_path):
    """No collapse when losses stay inside the baseline band."""
    out = tmp_path / "metrics.jsonl"
    collector = MetricsCollector(
        output_path=out, stability_window=10, stability_sd_threshold=3.0,
    )
    import random
    rng = random.Random(0)
    for i in range(500):
        # All losses in [0.98, 1.02] — way inside 3σ of any reasonable baseline.
        collector.record_doc(doc_id=i, bpb=1.0, tokens=1,
                             loss_before=1.0 + 0.02 * (rng.random() - 0.5),
                             loss_after=None, step_count=0, wall_ms=1.0,
                             grad_norm=0.0, state_norm=1.0)
    assert collector.collapsed is False


def test_context_manager_closes_file(tmp_path):
    out = tmp_path / "metrics.jsonl"
    with MetricsCollector(output_path=out) as collector:
        collector.record_doc(doc_id=0, bpb=1.0, tokens=1, loss_before=1.0,
                             loss_after=None, step_count=0, wall_ms=1.0,
                             grad_norm=0.0, state_norm=1.0)
    # File should be closed after the with-block exits.
    assert collector._fh.closed
```

**Step 2: Run**

Run: `pytest tests/test_eval_stream_metrics.py -v`
Expected: FAIL.

**Step 3: Implement**

```python
# src/chaoscontrol/eval_stream/metrics.py
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
```

**Step 4: Run**

Expected: 5 PASS (`test_writes_per_doc_record`, `test_stability_gate_flags_collapse`, `test_baseline_frozen_across_long_run`, `test_gate_does_not_fire_on_steady_model`, `test_context_manager_closes_file`).

**Step 5: Commit**

```bash
git add src/chaoscontrol/eval_stream/metrics.py tests/test_eval_stream_metrics.py
git commit -m "exp20(harness): MetricsCollector with in-run stability gate"
```

---

### Task 3.5: Thread `initial_states` through `ChaosSSMCore.forward` and `ChaosStudentLM.forward`

**Files:**
- Modify: `src/chaoscontrol/core.py` (`ChaosSSMCore.forward`, `ChaosSSMBlock.forward` if it wraps)
- Modify: `src/chaoscontrol/model.py` (`ChaosStudentLM.forward`)
- Create: `tests/test_initial_states_regression.py`

**Context (why this exists):** `src/chaoscontrol/core.py:416` hardcodes `state = x.new_zeros((batch, dim))` and `ChaosStudentLM.forward` has no state-threading argument at all. Without this plumbing, every Axis 2 persistence mode that claims to carry state across chunks / docs is silently identical to `reset`. The persistence logic in Task 7 is a no-op until this lands.

This is the **only** shared-code change required by Exp 20. Exp 20 owns the merge-last cost — if mainline changes `forward()` in the meantime, we rebase here.

**Contract:**
- `ChaosSSMCore.forward(x, initial_state=None, ...)`: if `initial_state` is provided, use it to seed the recurrence; otherwise behave exactly as today (`x.new_zeros(...)`). Return the final state alongside the output so callers can thread it to the next chunk. If `return_jacobian_stats=True` was the prior contract, the stats tuple now includes `final_state`, or the core returns a dedicated `(y, final_state[, stats])` tuple — whichever keeps the existing call-sites compiling with a minimal change. Decide this by inspecting `core.py` at the top of the task.
- `ChaosSSMBlock.forward(x, initial_state=None)` threads through to the core's kwarg.
- `ChaosStudentLM.forward(input_ids, initial_states: list[Tensor] | None = None)`: one tensor per block; `None` preserves the current zero-init behavior. The return dict gains `"final_states": list[Tensor]`.

**Step 1: Write regression tests (MUST fail before implementation)**

```python
# tests/test_initial_states_regression.py
import torch
from chaoscontrol.model import ChaosStudentLM


def _tiny_lm():
    torch.manual_seed(0)
    return ChaosStudentLM(vocab_size=32, dim=16, num_layers=2, block_type="ssm", a_mode="diag")


def test_backward_compat_no_state_kwarg():
    """model(input_ids) with no state kwarg must match prior behavior."""
    m = _tiny_lm()
    m.eval()
    ids = torch.randint(0, 32, (1, 32))
    with torch.no_grad():
        out = m(ids)
    logits = out["logits"] if isinstance(out, dict) else out
    assert logits.shape == (1, 32, 32)
    # final_states is newly added; must be a list with one tensor per layer
    assert "final_states" in out
    assert len(out["final_states"]) == 2
    for s in out["final_states"]:
        assert s.shape == (1, 16)


def test_zeros_initial_states_match_default():
    """Passing zeros for initial_states must be bit-identical to passing nothing."""
    m = _tiny_lm()
    m.eval()
    ids = torch.randint(0, 32, (1, 32))
    with torch.no_grad():
        out_default = m(ids)
        zeros = [torch.zeros(1, 16) for _ in range(2)]
        out_zeros = m(ids, initial_states=zeros)
    torch.testing.assert_close(
        out_default["logits"], out_zeros["logits"], rtol=0, atol=0,
    )


def test_nonzero_initial_state_changes_output():
    m = _tiny_lm()
    m.eval()
    ids = torch.randint(0, 32, (1, 32))
    with torch.no_grad():
        out_zero = m(ids)
        nz = [torch.randn(1, 16) for _ in range(2)]
        out_nz = m(ids, initial_states=nz)
    assert not torch.allclose(out_zero["logits"], out_nz["logits"], atol=1e-5)


def test_final_state_equals_chunked_sequential():
    """Whole-doc forward must equal concat of two chunked forwards threaded through final_state.

    This is the canary for the state-plumbing invariant. It must FAIL before
    Task 3.5 lands and PASS after.
    """
    m = _tiny_lm()
    m.eval()
    ids = torch.randint(0, 32, (1, 64))
    with torch.no_grad():
        whole = m(ids)
        first = m(ids[:, :32])
        second = m(ids[:, 32:], initial_states=first["final_states"])
    # Compare the second-half logits
    torch.testing.assert_close(
        whole["logits"][:, 32:], second["logits"], rtol=1e-5, atol=1e-5,
    )
```

**Step 2: Run** — expected FAIL (`initial_states` kwarg unknown, no `final_states` key).

**Step 3: Implement**

1. In `ChaosSSMCore.forward`: accept `initial_state: torch.Tensor | None = None`. Replace the hardcoded `state = x.new_zeros((batch, dim))` with:
   ```python
   if initial_state is None:
       state = x.new_zeros((batch, dim))
   else:
       # caller owns device/dtype match; do not implicitly cast
       assert initial_state.shape == (batch, dim), (
           f"initial_state shape {tuple(initial_state.shape)} != ({batch}, {dim})"
       )
       state = initial_state
   ```
   At end of forward, return the final state. Keep the existing return shape for callers that don't want it: if the function currently returns `y` or `(y, stats)`, the minimal extension is `(y, stats, final_state)` or swap to a namedtuple. Choose whichever keeps `ChaosSSMBlock.forward`'s call-site a one-line diff.

2. In `ChaosSSMBlock.forward`: accept `initial_state=None`, pass through, capture the returned `final_state`, and make it available to `ChaosStudentLM.forward` (return it alongside the block output).

3. In `ChaosStudentLM.forward`: accept `initial_states: list[Tensor] | None = None`. Iterate blocks with `initial_states[i] if initial_states is not None else None`, collect each block's `final_state`, and include `"final_states": [...]` in the returned dict.

**Step 4: Run** — 4 PASS.

**Step 5: Commit**

```bash
git add src/chaoscontrol/core.py src/chaoscontrol/model.py tests/test_initial_states_regression.py
git commit -m "exp20(harness): thread initial_states through core/model forward

Shared-code change: ChaosSSMCore.forward and ChaosStudentLM.forward now
accept optional initial_state / initial_states kwargs and return
final_state(s). Default behavior (no kwarg) is bit-identical to prior.
Prerequisite for Exp 20 Axis 2 persistence modes — without this plumbing,
carry_state and trainable_h0 were silently no-ops."
```

---

### Task 4: `DeltaModulator` — no-grad Δ rescaling via forward hooks

**Files:**
- Create: `src/chaoscontrol/eval_stream/delta_mod.py`
- Create: `tests/test_eval_stream_delta_mod.py`

**Context:** Axis 3 is pure inference — rescale `delta_proj(x)` output multiplicatively (`delta_scale`) and shift `log_a` pre-sigmoid (`log_a_shift`). No weight update. Implemented as forward hooks we attach before the run and remove after.

**Step 1: Write failing test**

```python
# tests/test_eval_stream_delta_mod.py
import torch
from chaoscontrol.eval_stream.delta_mod import DeltaModulator
from chaoscontrol.core import ChaosSSMCore


def test_delta_scale_identity_when_one():
    core = ChaosSSMCore(dim=16, a_mode="diag")
    x = torch.randn(2, 4, 16)
    y_base = core.forward(x)
    with DeltaModulator(core, delta_scale=1.0, log_a_shift=0.0):
        y_mod = core.forward(x)
    torch.testing.assert_close(y_base, y_mod)


def test_delta_scale_changes_output():
    core = ChaosSSMCore(dim=16, a_mode="diag")
    x = torch.randn(2, 4, 16)
    y_base = core.forward(x)
    with DeltaModulator(core, delta_scale=2.0, log_a_shift=0.0):
        y_mod = core.forward(x)
    assert not torch.allclose(y_base, y_mod, atol=1e-4)


def test_hooks_removed_after_context():
    core = ChaosSSMCore(dim=16, a_mode="diag")
    assert len(core.delta_proj._forward_hooks) == 0
    with DeltaModulator(core, delta_scale=2.0):
        assert len(core.delta_proj._forward_hooks) == 1
    assert len(core.delta_proj._forward_hooks) == 0
```

**Step 2: Run** — FAIL.

**Step 3: Implement**

```python
# src/chaoscontrol/eval_stream/delta_mod.py
from __future__ import annotations
import torch
import torch.nn as nn


class DeltaModulator:
    """Context manager that attaches forward hooks to every ChaosSSMCore in a model
    to rescale delta_proj output and shift log_a at eval. No gradients involved.
    """

    def __init__(self, module: nn.Module, *, delta_scale: float = 1.0,
                 log_a_shift: float = 0.0, adapt_set_hint: str | None = None):
        self.module = module
        self.delta_scale = float(delta_scale)
        self.log_a_shift = float(log_a_shift)
        # Optional hint so we can fail loud on Axis 1 × Axis 3 log_a overlap.
        self._adapt_set_hint = adapt_set_hint
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._log_a_originals: list[tuple[nn.Parameter, torch.Tensor]] = []

    def _find_cores(self) -> list[nn.Module]:
        from chaoscontrol.core import ChaosSSMCore
        return [m for m in self.module.modules() if isinstance(m, ChaosSSMCore)]

    def __enter__(self):
        # Axis 3 (log_a_shift) is designed ⊥ Axis 1 (log_a adaptation). If the
        # caller is adapting log_a AND shifting it, DeltaModulator will
        # restore the pre-shift value on exit, wiping the adaptation. Caller
        # must avoid this combination — enforced by the driver entry check in
        # Task 8 as well. We assert here as a backstop.
        if self.log_a_shift != 0.0 and getattr(self, "_adapt_set_hint", None) in (
            "log_a", "log_a+delta_proj", "all",
        ):
            raise ValueError(
                "DeltaModulator.log_a_shift is incompatible with adapting log_a "
                "(log_a_shift reverts log_a on exit; Axis 3 is ⊥ Axis 1)."
            )
        scale = self.delta_scale
        for core in self._find_cores():
            if scale != 1.0:
                h = core.delta_proj.register_forward_hook(
                    lambda mod, inp, out, s=scale: out * s
                )
                self._handles.append(h)
            if self.log_a_shift != 0.0:
                # log_a is a Parameter read in forward; we must mutate then restore
                self._log_a_originals.append((core.log_a, core.log_a.detach().clone()))
                with torch.no_grad():
                    core.log_a.add_(self.log_a_shift)
        return self

    def __exit__(self, *args):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        for param, orig in self._log_a_originals:
            with torch.no_grad():
                param.copy_(orig)
        self._log_a_originals.clear()
```

**Step 4: Run** — 3 PASS.

**Step 5: Commit**

```bash
git add src/chaoscontrol/eval_stream/delta_mod.py tests/test_eval_stream_delta_mod.py
git commit -m "exp20(harness): DeltaModulator forward-hook Δ rescaling (axis 3)"
```

---

### Task 5: `LegalityController` — chunk-level Score-then-Adapt (with LEAK-DETECTION CONTRACT TEST)

**Files:**
- Create: `src/chaoscontrol/eval_stream/legality.py`
- Create: `tests/test_eval_stream_legality.py`

**Context:** Issue #1017 rule: token N must be scored under weights updated only on 1..N-1. This controller enforces it structurally. **The leak-detection contract test is non-negotiable — if the test harness itself doesn't catch a forced leak, we cannot trust any Exp 20 result.**

**Step 1: Write failing test** (leak-detection contract — the most important test in the codebase for Exp 20)

```python
# tests/test_eval_stream_legality.py
import torch
import torch.nn as nn
import pytest
from chaoscontrol.eval_stream.legality import LegalityController, LeakDetectedError


class _TinyLM(nn.Module):
    def __init__(self, vocab=32, dim=16):
        super().__init__()
        self.vocab_size = vocab
        self.embed = nn.Embedding(vocab, dim)
        self.lm_head = nn.Linear(dim, vocab, bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        return {"logits": self.lm_head(x)}


def _loss(logits, targets):
    import torch.nn.functional as F
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def test_score_is_under_pre_update_weights():
    torch.manual_seed(0)
    model = _TinyLM()
    tokens = torch.randint(0, 32, (1, 64))
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    controller = LegalityController(model, loss_fn=_loss)
    # Score chunk of 32 tokens under frozen weights, then adapt on them
    chunk = tokens[:, :32]
    loss_before = controller.score_chunk(chunk)
    # Capture weights snapshot
    w_snap = model.lm_head.weight.detach().clone()
    controller.adapt_on_chunk(chunk, optimizer=opt, steps=1)
    # Weights must have CHANGED (we adapted)
    assert not torch.allclose(w_snap, model.lm_head.weight)
    # The score is the score under the OLD weights — this is validated by
    # re-forwarding under a rolled-back snapshot and checking equality.
    with torch.no_grad():
        model.lm_head.weight.copy_(w_snap)
        logits = model(chunk)["logits"]
        loss_rollback = _loss(logits[:, :-1], chunk[:, 1:])
    torch.testing.assert_close(loss_before, loss_rollback.item(), rtol=0, atol=1e-6)


def test_leak_detected_when_scoring_under_updated_weights():
    """CONTRACT TEST: a forced leak MUST be detected. If this fails, the harness is invalid."""
    torch.manual_seed(0)
    model = _TinyLM()
    tokens = torch.randint(0, 32, (1, 32))
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    controller = LegalityController(model, loss_fn=_loss, leak_detection=True)

    # Intentional leak: update weights FIRST, then score the same chunk
    controller.adapt_on_chunk(tokens, optimizer=opt, steps=1)
    with pytest.raises(LeakDetectedError):
        # Attempting to score a chunk that was already adapted-on is a leak
        controller.score_chunk(tokens)
```

**Step 2: Run** — FAIL.

**Step 3: Implement**

```python
# src/chaoscontrol/eval_stream/legality.py
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Callable


class LeakDetectedError(RuntimeError):
    pass


class LegalityController:
    """Enforces Issue #1017 score-before-update rule structurally.

    Contract:
      score_chunk(chunk)    : forward-only under current weights; returns scalar loss.
                              Records chunk's token hash in _scored_chunks.
      adapt_on_chunk(chunk) : runs optimizer step(s) on `chunk`. Records chunk's
                              token hash in _adapted_chunks.

    Leak detection (optional, for contract testing):
      If a chunk is score_chunk'd AFTER being adapt_on_chunk'd, LeakDetectedError.
    """

    def __init__(self, model: nn.Module, *, loss_fn: Callable,
                 leak_detection: bool = False):
        self.model = model
        self.loss_fn = loss_fn
        self.leak_detection = leak_detection
        self._adapted_chunks: set[int] = set()

    @staticmethod
    def _chunk_hash(chunk: torch.Tensor) -> bytes:
        # Stable hash across processes and PYTHONHASHSEED values. Python's
        # builtin `hash(bytes)` is randomized per-process — fine for the
        # in-run set, but we want reproducible diagnostics across launches.
        import hashlib
        return hashlib.blake2b(
            chunk.detach().cpu().numpy().tobytes(), digest_size=8
        ).digest()

    def score_chunk(self, chunk: torch.Tensor) -> float:
        # Empty-CE guard: CE needs at least one target token, i.e. chunk length
        # >= 2 after shifting. Callers must have already filtered these, but a
        # silent NaN here would poison the stability gate.
        if chunk.size(1) < 2:
            raise ValueError(
                f"score_chunk needs chunk length >= 2 for teacher-forcing CE; "
                f"got shape {tuple(chunk.shape)}."
            )
        h = self._chunk_hash(chunk)
        if self.leak_detection and h in self._adapted_chunks:
            raise LeakDetectedError(
                f"Chunk hash {h.hex()} was adapt_on_chunk'd before score_chunk: "
                "Issue #1017 violation."
            )
        self.model.eval()
        with torch.no_grad():
            out = self.model(chunk)
            logits = out["logits"] if isinstance(out, dict) else out
            loss = self.loss_fn(logits[:, :-1], chunk[:, 1:])
        return float(loss.item())

    def adapt_on_chunk(
        self, chunk: torch.Tensor, *, optimizer, steps: int = 1,
    ) -> float | None:
        """Runs `steps` gradient updates on the chunk. Returns final loss, or None if steps==0."""
        if steps <= 0:
            return None
        h = self._chunk_hash(chunk)
        self._adapted_chunks.add(h)
        self.model.train()
        final_loss = None
        for _ in range(steps):
            optimizer.zero_grad()
            out = self.model(chunk)
            logits = out["logits"] if isinstance(out, dict) else out
            loss = self.loss_fn(logits[:, :-1], chunk[:, 1:])
            loss.backward()
            optimizer.step()
            final_loss = float(loss.item())
        return final_loss

    def mark_new_epoch(self) -> None:
        """Reset chunk-reuse tracking at doc boundary — chunks are per-doc."""
        self._adapted_chunks.clear()
```

**Additional tests to add to `tests/test_eval_stream_legality.py`:**

```python
def test_empty_chunk_raises_valueerror():
    m = _TinyLM()
    controller = LegalityController(m, loss_fn=_loss)
    with pytest.raises(ValueError):
        controller.score_chunk(torch.randint(0, 32, (1, 1)))  # length < 2


def test_cross_doc_re_adapt_same_bytes_is_ok_after_mark_new_epoch():
    """Same chunk content appearing in two different docs is not a leak.
    mark_new_epoch() clears tracking; re-adapting then re-scoring the same
    bytes in the next doc must succeed without raising.
    """
    torch.manual_seed(0)
    m = _TinyLM()
    opt = torch.optim.SGD(m.parameters(), lr=0.01)
    controller = LegalityController(m, loss_fn=_loss, leak_detection=True)

    chunk = torch.randint(0, 32, (1, 16))
    controller.adapt_on_chunk(chunk, optimizer=opt, steps=1)
    controller.mark_new_epoch()
    # Next doc: same bytes reappear, legitimate score-then-adapt order.
    _ = controller.score_chunk(chunk)
    controller.adapt_on_chunk(chunk, optimizer=opt, steps=1)  # must not raise
```

**Step 4: Run** — 4 PASS.

**Step 5: Commit**

```bash
git add src/chaoscontrol/eval_stream/legality.py tests/test_eval_stream_legality.py
git commit -m "exp20(harness): LegalityController with leak-detection contract test

Score-before-update enforced structurally. Leak detection fires when a chunk
is score_chunk'd after adapt_on_chunk'd — non-negotiable contract for any
Exp 20 result to be valid."
```

---

### Task 6: `TTTRunner` — param-group filtering + inner loop

**Files:**
- Create: `src/chaoscontrol/eval_stream/ttt_runner.py`
- Create: `tests/test_eval_stream_ttt_runner.py`

**Context:** Axis 1 is a string filter over param-group names; we map each tag to the list of matching `nn.Parameter` instances.

**Step 1: Write failing test**

```python
# tests/test_eval_stream_ttt_runner.py
import torch
from chaoscontrol.eval_stream.ttt_runner import select_adapt_params
from chaoscontrol.model import ChaosStudentLM


def _tiny_ssm_lm():
    return ChaosStudentLM(
        vocab_size=32, dim=16, num_layers=2, block_type="ssm", a_mode="diag",
    )


def test_log_a_selection_is_small_and_correct():
    m = _tiny_ssm_lm()
    params = select_adapt_params(m, adapt_set="log_a")
    # 2 layers * dim=16 each = 32 scalars; count parameters not tensors
    total = sum(p.numel() for p in params)
    assert total == 32
    # Every selected param's name should contain "log_a"
    names = {n for n, p in m.named_parameters() if any(p is q for q in params)}
    assert all("log_a" in n for n in names)


def test_delta_proj_selection():
    m = _tiny_ssm_lm()
    params = select_adapt_params(m, adapt_set="delta_proj")
    # dim=16, delta_proj is Linear(dim, dim) -> 256 params per layer × 2 layers
    assert sum(p.numel() for p in params) == 2 * 16 * 16


def test_lm_head_selection():
    m = _tiny_ssm_lm()
    params = select_adapt_params(m, adapt_set="lm_head")
    # vocab=32, dim=16 -> 32*16=512
    assert sum(p.numel() for p in params) == 32 * 16


def test_none_selection_is_empty():
    m = _tiny_ssm_lm()
    params = select_adapt_params(m, adapt_set="none")
    assert params == []


def test_all_selection_covers_every_param():
    m = _tiny_ssm_lm()
    params = select_adapt_params(m, adapt_set="all")
    expected = sum(p.numel() for p in m.parameters())
    assert sum(p.numel() for p in params) == expected


def test_embed_rows_seen_is_exact_match():
    """Must match 'embed.weight' exactly — no collision with any hypothetical
    embed_norm / embedding_* future sibling params.
    """
    m = _tiny_ssm_lm()
    params = select_adapt_params(m, adapt_set="embed_rows_seen")
    names = {n for n, p in m.named_parameters() if any(p is q for q in params)}
    assert names == {"embed.weight"}


def test_trainable_h0_pattern(tmp_path):
    from chaoscontrol.eval_stream.persistence import attach_trainable_h0
    m = _tiny_ssm_lm()
    attach_trainable_h0(m)
    params = select_adapt_params(m, adapt_set="trainable_h0")
    names = {n for n, p in m.named_parameters() if any(p is q for q in params)}
    assert len(names) == 2  # 2 layers
    assert all("_trainable_h0" in n for n in names)
```

**Step 2: Run** — FAIL.

**Step 3: Implement**

```python
# src/chaoscontrol/eval_stream/ttt_runner.py
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Iterable


# Pattern forms:
#   "substr:X"    — match if X appears anywhere in the param's FQN
#   "exact:X"     — match only if the FQN equals X exactly
# (Plain strings in the list are treated as substr: for backward compatibility.)
ADAPT_SET_PATTERNS: dict[str, list[str]] = {
    "none": [],
    "log_a": ["substr:log_a"],
    "delta_proj": ["substr:delta_proj"],
    "log_a+delta_proj": ["substr:log_a", "substr:delta_proj"],
    "B_side": ["substr:in_proj", "substr:select_proj"],
    "C_side": ["substr:out_proj", "substr:gate_proj"],
    # embed_rows_seen: exact match to avoid collisions with any future
    # "embed_dim" / "embedding_norm" / etc. parameter names.
    "embed_rows_seen": ["exact:embed.weight"],
    "lm_head": ["substr:lm_head"],
    "lora_r8": ["substr:lora_"],  # lora adapters named lora_A_<name> / lora_B_<name>
    "trainable_h0": ["substr:_trainable_h0"],  # Axis 2 trainable h0, see Task 7
    "all": ["*"],
}


def _matches(name: str, patterns: list[str]) -> bool:
    for pat in patterns:
        if pat.startswith("exact:"):
            if name == pat[len("exact:"):]:
                return True
        elif pat.startswith("substr:"):
            if pat[len("substr:"):] in name:
                return True
        else:  # legacy plain string == substr
            if pat in name:
                return True
    return False


def select_adapt_params(module: nn.Module, *, adapt_set: str) -> list[nn.Parameter]:
    """Return the list of parameters that match the adapt_set filter."""
    if adapt_set not in ADAPT_SET_PATTERNS:
        raise ValueError(f"unknown adapt_set: {adapt_set}")
    patterns = ADAPT_SET_PATTERNS[adapt_set]
    if not patterns:
        return []
    if patterns == ["*"]:
        return list(module.parameters())
    out: list[nn.Parameter] = []
    seen: set[int] = set()
    for name, p in module.named_parameters():
        if _matches(name, patterns):
            if id(p) not in seen:
                out.append(p)
                seen.add(id(p))
    return out
```

**Step 4: Run** — 7 PASS.

**Step 5: Commit**

```bash
git add src/chaoscontrol/eval_stream/ttt_runner.py tests/test_eval_stream_ttt_runner.py
git commit -m "exp20(harness): param-group selector for Axis 1 adapt sets"
```

---

### Task 7: Persistence modes — state carry + trainable_h0

**Depends on Task 3.5.** `StateManager.get_state()` values are only actually used by the model once `ChaosStudentLM.forward` accepts `initial_states`. Run Task 3.5's regression tests and confirm they pass before starting this task.

**Files:**
- Create: `src/chaoscontrol/eval_stream/persistence.py`
- Create: `tests/test_eval_stream_persistence.py`
- Modify: `src/chaoscontrol/eval_stream/ttt_runner.py` (add weight-snapshot helpers)

**Context:** Axis 2 — at doc boundary, choose: reset state, carry state, carry weight deltas, trainable `h₀`, or compositions. State is a list of per-block tensors threaded into `ChaosStudentLM.forward(..., initial_states=...)` (Task 3.5); `trainable_h0` is a learnable parameter we add to the model at harness setup.

**Step 1: Write failing test**

```python
# tests/test_eval_stream_persistence.py
import torch
from chaoscontrol.eval_stream.persistence import (
    StateManager, attach_trainable_h0, detach_trainable_h0,
)
from chaoscontrol.model import ChaosStudentLM


def _tiny_ssm_lm():
    return ChaosStudentLM(vocab_size=32, dim=16, num_layers=2, block_type="ssm", a_mode="diag")


def test_reset_mode_zeros_state():
    m = _tiny_ssm_lm()
    sm = StateManager(m, persistence_mode="reset")
    sm.start_doc(doc_id=1, batch_size=1)
    state = sm.get_state()
    assert all(torch.all(s == 0) for s in state)


def test_carry_mode_preserves_state_across_docs():
    m = _tiny_ssm_lm()
    sm = StateManager(m, persistence_mode="carry_state")
    sm.start_doc(doc_id=0, batch_size=1)
    sm.set_state([torch.randn(1, 16), torch.randn(1, 16)])
    prev = [s.clone() for s in sm.get_state()]
    sm.start_doc(doc_id=1, batch_size=1)  # should NOT reset
    for a, b in zip(prev, sm.get_state()):
        torch.testing.assert_close(a, b)


def test_trainable_h0_is_learnable():
    m = _tiny_ssm_lm()
    attach_trainable_h0(m)
    # h0 should appear in model.named_parameters()
    names = [n for n, _ in m.named_parameters()]
    h0_names = [n for n in names if "trainable_h0" in n]
    assert len(h0_names) == 2  # 2 layers
    detach_trainable_h0(m)
    names = [n for n, _ in m.named_parameters()]
    assert not any("trainable_h0" in n for n in names)


def test_detach_trainable_h0_clears_state_dict():
    """After detach, state_dict must not contain any _trainable_h0 keys —
    otherwise saving/loading checkpoints leaks eval-only state.
    """
    m = _tiny_ssm_lm()
    attach_trainable_h0(m)
    assert any("_trainable_h0" in k for k in m.state_dict().keys())
    detach_trainable_h0(m)
    assert not any("_trainable_h0" in k for k in m.state_dict().keys())


def test_trainable_h0_receives_gradient():
    """Pre-req for Task 3.5 integration: with h0 threaded through
    initial_states, loss.backward() must accumulate grad on _trainable_h0.
    """
    m = _tiny_ssm_lm()
    attach_trainable_h0(m)
    ids = torch.randint(0, 32, (1, 16))
    sm = StateManager(m, persistence_mode="trainable_h0")
    sm.start_doc(doc_id=0, batch_size=1)
    out = m(ids, initial_states=sm.get_state())
    loss = out["logits"].sum()
    loss.backward()
    for core in m.modules():
        from chaoscontrol.core import ChaosSSMCore
        if isinstance(core, ChaosSSMCore):
            assert core._trainable_h0.grad is not None
            assert core._trainable_h0.grad.abs().sum() > 0
```

**Step 2: Run** — FAIL.

**Step 3: Implement**

```python
# src/chaoscontrol/eval_stream/persistence.py
from __future__ import annotations
import torch
import torch.nn as nn
from chaoscontrol.core import ChaosSSMCore


class StateManager:
    """Manages per-block recurrence state across chunks and doc boundaries.

    Modes:
      reset:             zero state at each doc boundary.
      carry_state:       preserve state across doc boundaries.
      carry_weights:     reset state, but keep weight deltas (no-op here; handled by not reverting weights).
      carry_both:        preserve state AND weight deltas.
      trainable_h0:      init state from trainable h0 param at doc boundary; reset at chunk boundary.
      trainable_h0+carry: init first doc from h0, carry thereafter.
    """

    def __init__(self, model: nn.Module, *, persistence_mode: str):
        self.model = model
        self.mode = persistence_mode
        self._cores = [m for m in model.modules() if isinstance(m, ChaosSSMCore)]
        self._state: list[torch.Tensor] = []
        self._doc_idx = -1

    def start_doc(self, *, doc_id: int, batch_size: int) -> None:
        self._doc_idx += 1
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        if self.mode in ("reset", "carry_weights"):
            self._state = [torch.zeros(batch_size, c.dim, device=device, dtype=dtype) for c in self._cores]
        elif self.mode in ("carry_state", "carry_both"):
            if not self._state:
                self._state = [torch.zeros(batch_size, c.dim, device=device, dtype=dtype) for c in self._cores]
            # else keep prior state
        elif self.mode == "trainable_h0":
            # `.clone()` on a view of a Parameter breaks autograd flow back to
            # the Parameter — we want gradients on _trainable_h0 to accumulate,
            # so keep the graph edge intact. `.expand()` is a view and can be
            # fed to the model directly (task 3.5 initial_states kwarg takes
            # ownership of batching).
            self._state = [c._trainable_h0.expand(batch_size, -1) for c in self._cores]
        elif self.mode == "trainable_h0+carry":
            if self._doc_idx == 0:
                self._state = [c._trainable_h0.expand(batch_size, -1) for c in self._cores]
            # else keep prior state
        else:
            raise ValueError(f"unknown persistence_mode: {self.mode}")

    def get_state(self) -> list[torch.Tensor]:
        return self._state

    def set_state(self, state: list[torch.Tensor]) -> None:
        self._state = state


def attach_trainable_h0(model: nn.Module) -> None:
    """Add a trainable h0 vector to each ChaosSSMCore. Eval-time only.

    Placed on the core's own device+dtype so subsequent `initial_states`
    threading doesn't trigger implicit copies. `nn.Parameter(...)` is registered
    via attribute assignment because ChaosSSMCore inherits from nn.Module.
    """
    for core in model.modules():
        if isinstance(core, ChaosSSMCore):
            if not hasattr(core, "_trainable_h0"):
                # Use an existing core parameter to pin device+dtype.
                ref = next(core.parameters())
                core._trainable_h0 = nn.Parameter(
                    torch.zeros(1, core.dim, device=ref.device, dtype=ref.dtype)
                )


def detach_trainable_h0(model: nn.Module) -> None:
    """Remove the _trainable_h0 parameter from every core. After this call,
    `model.state_dict()` must contain no `_trainable_h0` keys.
    """
    for core in model.modules():
        if isinstance(core, ChaosSSMCore):
            if hasattr(core, "_trainable_h0"):
                # Attribute delete also removes from _parameters.
                del core._trainable_h0
```

**Step 4: Run** — 5 PASS.

**Step 5: Commit**

```bash
git add src/chaoscontrol/eval_stream/persistence.py tests/test_eval_stream_persistence.py
git commit -m "exp20(harness): StateManager for Axis 2 persistence modes + trainable_h0"
```

---

### Task 8: Driver script `scripts/run_exp20_eval.py`

**Files:**
- Create: `scripts/run_exp20_eval.py`
- Create: `tests/test_run_exp20_eval.py` (integration-lite, tiny model + tiny stream)

**Context:** Composes the 5 components. Takes a YAML/JSON config + checkpoint path + output path. Handles doc loop, chunk loop, metrics, stability gate. Muon optimizer (existing `src/chaoscontrol/optim/muon.py`).

**Step 1: Write failing integration test**

```python
# tests/test_run_exp20_eval.py
import subprocess
import sys
import json
import numpy as np
from pathlib import Path
import torch


def test_driver_runs_tiny_stream(tmp_path):
    # Tiny SP model + JSONL doc file — matches the DocStreamer retrofit
    # (JSONL + on-the-fly SP tokenization).
    import sentencepiece as spm
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join(["alpha beta gamma", "delta epsilon"] * 50))
    sp_prefix = tmp_path / "sp"
    spm.SentencePieceTrainer.Train(
        input=str(corpus), model_prefix=str(sp_prefix),
        vocab_size=64, character_coverage=1.0, model_type="bpe",
    )
    sp_model_path = f"{sp_prefix}.model"

    jsonl = tmp_path / "docs.jsonl"
    with jsonl.open("w") as fh:
        for t in ["hello world this is a doc", "another small doc", "and a third"]:
            fh.write(json.dumps({"text": t}) + "\n")

    # Tiny checkpoint — vocab_size must match the SP model's piece count.
    from chaoscontrol.model import ChaosStudentLM
    m = ChaosStudentLM(vocab_size=64, dim=16, num_layers=2, block_type="ssm", a_mode="diag")
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save({"model": m.state_dict(),
                "config": {"vocab_size": 64, "dim": 16, "num_layers": 2,
                           "block_type": "ssm", "a_mode": "diag"}}, ckpt_path)

    out_path = tmp_path / "metrics.jsonl"
    cfg_path = tmp_path / "run.json"
    cfg_path.write_text(json.dumps({
        "adapt_set": "none", "persistence_mode": "reset",
        "chunk_size": 32, "steps_per_chunk": 0,
        "max_docs": 3, "seed": 0,
        "jsonl_paths": [str(jsonl)],
        "sp_model_path": sp_model_path,
        "checkpoint_path": str(ckpt_path),
        "output_path": str(out_path),
    }))
    result = subprocess.run(
        [sys.executable, "scripts/run_exp20_eval.py", "--config", str(cfg_path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    lines = out_path.read_text().strip().splitlines()
    assert len(lines) == 3  # 3 docs
```

**Step 2: Run** — FAIL.

**Step 3: Implement**

```python
# scripts/run_exp20_eval.py
"""Exp 20 driver. Composes DocStreamer + LegalityController + TTTRunner
+ DeltaModulator + MetricsCollector. Reads a JSON config.
"""
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from chaoscontrol.eval_stream.types import RunConfig
from chaoscontrol.eval_stream.doc_stream import DocStreamer
from chaoscontrol.eval_stream.legality import LegalityController
from chaoscontrol.eval_stream.persistence import StateManager, attach_trainable_h0
from chaoscontrol.eval_stream.ttt_runner import select_adapt_params, ADAPT_SET_PATTERNS
from chaoscontrol.eval_stream.delta_mod import DeltaModulator
from chaoscontrol.eval_stream.metrics import MetricsCollector
from chaoscontrol.evaluation import compute_bpb


def _ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                           targets.reshape(-1), reduction="sum")


def _build_model(ckpt_path: Path) -> tuple[torch.nn.Module, dict]:
    from chaoscontrol.model import ChaosStudentLM
    # weights_only=False because the checkpoint payload is a dict with
    # {model, config, ...}, not a pure tensor. We trust our own checkpoints.
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = blob["config"]
    model = ChaosStudentLM(**cfg)
    # strict=True: any unexpected / missing key is a checkpoint mismatch we
    # want to surface, not paper over. attach_trainable_h0 runs AFTER this
    # (see run() below) so eval-only params never need a loose load.
    model.load_state_dict(blob["model"], strict=True)
    return model, cfg


def _build_optimizer(params, lr: float):
    """Single Muon optimizer. Muon handles 1D/0D params internally via its
    decoupled AdamW path (see src/chaoscontrol/optim/muon.py:145), so a
    prior "Muon for 2D + AdamW for scalars" split produced: (1) double-
    backward per chunk, (2) stale-grad bleed between optimizer steps, (3)
    catastrophic AdamW LR when Muon's LR (e.g. 0.064) was passed through
    unchanged. Collapsed to a single optimizer.
    """
    from chaoscontrol.optim.muon import Muon
    if not params:
        return []
    return [Muon(params, lr=lr)]


def _iter_chunks(tokens: list[int], chunk_size: int):
    if chunk_size < 0:  # whole_doc
        yield tokens
        return
    for i in range(0, len(tokens), chunk_size):
        yield tokens[i:i + chunk_size]


def run(cfg: RunConfig, jsonl_paths: list[str], sp_model_path: str) -> None:
    # Entry assertion: Axis 1 adapting log_a is incompatible with Axis 3
    # log_a_shift (DeltaModulator reverts log_a on exit, wiping the
    # adaptation). Enforced in DeltaModulator.__enter__ as well; caught here
    # for an earlier, clearer error.
    patterns = ADAPT_SET_PATTERNS.get(cfg.adapt_set, [])
    adapts_log_a = any("log_a" in p for p in patterns) or patterns == ["*"]
    if adapts_log_a and cfg.log_a_shift != 0.0:
        raise ValueError(
            f"adapt_set={cfg.adapt_set!r} adapts log_a but log_a_shift="
            f"{cfg.log_a_shift} is nonzero; Axis 1 × Axis 3 overlap on log_a."
        )

    # Seed-parallel launches may share a node; pin this process to its local
    # GPU before any CUDA work so rank 0 doesn't collide with rank N.
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = _build_model(Path(cfg.checkpoint_path))
    model.to(device)

    # attach_trainable_h0 AFTER load_state_dict so strict=True works on the
    # checkpoint (which has no h0 keys) and the newly-created h0 params live
    # on the model's device+dtype from birth (see persistence.attach_trainable_h0).
    if "trainable_h0" in cfg.persistence_mode:
        attach_trainable_h0(model)

    adapt_params = select_adapt_params(model, adapt_set=cfg.adapt_set)
    optimizers = _build_optimizer(adapt_params, cfg.eval_lr) if adapt_params else []

    streamer = DocStreamer(
        jsonl_paths=[Path(p) for p in jsonl_paths],
        sp_model_path=Path(sp_model_path),
        max_docs=cfg.max_docs,
    )
    state_mgr = StateManager(model, persistence_mode=cfg.persistence_mode)
    controller = LegalityController(model, loss_fn=_ce)
    collector = MetricsCollector(output_path=Path(cfg.output_path))

    with DeltaModulator(model, delta_scale=cfg.delta_scale,
                        log_a_shift=cfg.log_a_shift,
                        adapt_set_hint=cfg.adapt_set):
        run_start = time.monotonic()
        for doc in streamer:
            if time.monotonic() - run_start > cfg.budget_seconds:
                break
            if collector.collapsed:
                break
            state_mgr.start_doc(doc_id=doc.doc_id, batch_size=1)
            controller.mark_new_epoch()

            doc_ce_nats = 0.0
            doc_tokens = 0
            step_count = 0
            loss_before_sum = 0.0
            loss_after_sum = 0.0
            chunk_count = 0
            t0 = time.monotonic()

            for chunk_list in _iter_chunks(doc.tokens, cfg.chunk_size):
                if len(chunk_list) < 2:
                    continue
                chunk = torch.tensor(chunk_list, dtype=torch.long, device=device).unsqueeze(0)
                # Score (legality-guarded)
                loss_before = controller.score_chunk(chunk)
                loss_before_sum += loss_before
                # nats per chunk (sum): loss_before is mean-CE from controller; convert
                # But our _ce uses reduction="sum" — so loss_before IS summed nats.
                doc_ce_nats += loss_before
                doc_tokens += chunk.size(1) - 1
                chunk_count += 1

                # Adapt — single Muon optimizer; `optimizers` is at most len==1.
                if optimizers and cfg.steps_per_chunk > 0:
                    loss_after = controller.adapt_on_chunk(
                        chunk, optimizer=optimizers[0], steps=cfg.steps_per_chunk,
                    )
                    # Using `or 0.0` would drop legitimate 0.0 losses;
                    # be explicit about the None case.
                    loss_after_sum += 0.0 if loss_after is None else loss_after
                    step_count += cfg.steps_per_chunk

            wall_ms = (time.monotonic() - t0) * 1000.0
            bpb = compute_bpb(doc_ce_nats, doc.raw_bytes) if doc.raw_bytes > 0 else 0.0
            grad_norm = 0.0  # populated by controller in later task; placeholder
            state_norm = sum(float(s.norm()) for s in state_mgr.get_state()) / max(len(state_mgr.get_state()), 1)
            collector.record_doc(
                doc_id=doc.doc_id, bpb=bpb, tokens=doc_tokens,
                loss_before=loss_before_sum / max(chunk_count, 1),
                loss_after=loss_after_sum / max(chunk_count, 1) if loss_after_sum else None,
                step_count=step_count, wall_ms=wall_ms,
                grad_norm=grad_norm, state_norm=state_norm,
            )
    collector.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    raw = json.loads(Path(args.config).read_text())
    jsonl_paths = raw.pop("jsonl_paths")
    sp_model_path = raw.pop("sp_model_path")
    cfg = RunConfig(**{k: v for k, v in raw.items() if k in RunConfig.__dataclass_fields__})
    run(cfg, jsonl_paths=jsonl_paths, sp_model_path=sp_model_path)


if __name__ == "__main__":
    main()
```

**Step 4: Run** — 1 PASS.

**Step 5: Commit**

```bash
git add scripts/run_exp20_eval.py tests/test_run_exp20_eval.py
git commit -m "exp20(harness): driver composes 5 components; JSON-config CLI"
```

---

### Task 9: End-to-end Phase A smoke — bit-exact forward-only match

**Files:**
- Create: `tests/test_exp20_phase_a_smoke.py`

**Context:** With `adapt_set=none, persistence_mode=reset, delta_scale=1.0, log_a_shift=0.0, steps_per_chunk=0`, the harness must produce bit-identical per-token losses to the existing forward-only eval path for the first doc. This is the non-negotiable baseline-harness invariant.

**Step 1: Write failing test**

```python
# tests/test_exp20_phase_a_smoke.py
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from chaoscontrol.model import ChaosStudentLM
from chaoscontrol.eval_stream.doc_stream import DocStreamer
from chaoscontrol.eval_stream.legality import LegalityController


def test_legality_controller_matches_naive_forward(tmp_path):
    torch.manual_seed(0)
    # CPU-pin the test: bf16/tf32 on GPU exceeds the 1e-4 tolerance; the
    # bit-exact guarantee is about the harness composition, not the dtype path.
    device = torch.device("cpu")
    m = ChaosStudentLM(
        vocab_size=32, dim=16, num_layers=2, block_type="ssm", a_mode="diag",
    ).to(device)
    m.eval()  # must be in eval for deterministic comparison under block_type="attention" too
    tokens = torch.randint(1, 32, (1, 64), device=device)

    # Naive forward-only — call .eval() first so the comparison is invariant
    # under blocks that include dropout (attention variant).
    with torch.no_grad():
        out = m(tokens)
        logits = out["logits"] if isinstance(out, dict) else out
        naive = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)).float(),
            tokens[:, 1:].reshape(-1), reduction="sum",
        ).item()

    controller = LegalityController(
        m, loss_fn=lambda lg, tg: F.cross_entropy(
            lg.reshape(-1, lg.size(-1)).float(), tg.reshape(-1), reduction="sum"
        ),
    )
    harness_score = controller.score_chunk(tokens)
    assert abs(harness_score - naive) < 1e-4


def test_chunked_carry_state_equals_whole_doc(tmp_path):
    """Canary for the Task 3.5 state-plumbing invariant.

    Running the driver with chunk_size=32 on a 64-token doc at
    persistence_mode=carry_state must match a single whole-doc forward
    to within float noise. Before Task 3.5 lands (state kwarg threaded
    into ChaosStudentLM.forward), carry_state is a silent no-op and
    this test FAILS. After Task 3.5, it PASSES.

    This is the regression gate that would have caught S2 in-flight.
    """
    import json
    import subprocess
    import sys

    import sentencepiece as spm

    # Tiny SP model
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join(["alpha beta gamma", "delta epsilon zeta"] * 50))
    sp_prefix = tmp_path / "sp"
    spm.SentencePieceTrainer.Train(
        input=str(corpus), model_prefix=str(sp_prefix),
        vocab_size=64, character_coverage=1.0, model_type="bpe",
    )

    # JSONL with one doc long enough to produce >= 64 SP tokens
    jsonl = tmp_path / "docs.jsonl"
    with jsonl.open("w") as fh:
        text = " ".join(["alpha beta gamma delta epsilon zeta"] * 12)
        fh.write(json.dumps({"text": text}) + "\n")

    # Tiny checkpoint
    m = ChaosStudentLM(vocab_size=64, dim=16, num_layers=2, block_type="ssm", a_mode="diag")
    ckpt = tmp_path / "ckpt.pt"
    torch.save({"model": m.state_dict(),
                "config": {"vocab_size": 64, "dim": 16, "num_layers": 2,
                           "block_type": "ssm", "a_mode": "diag"}}, ckpt)

    def _run(chunk_size: int, persistence: str, out_name: str) -> float:
        cfg_path = tmp_path / f"cfg_{out_name}.json"
        out_path = tmp_path / f"{out_name}.jsonl"
        cfg_path.write_text(json.dumps({
            "adapt_set": "none", "persistence_mode": persistence,
            "chunk_size": chunk_size, "steps_per_chunk": 0,
            "max_docs": 1, "seed": 0,
            "jsonl_paths": [str(jsonl)],
            "sp_model_path": f"{sp_prefix}.model",
            "checkpoint_path": str(ckpt),
            "output_path": str(out_path),
        }))
        result = subprocess.run(
            [sys.executable, "scripts/run_exp20_eval.py", "--config", str(cfg_path)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        rec = json.loads(out_path.read_text().strip().splitlines()[0])
        # loss_before is mean-per-chunk sum-CE; doc-level summed CE = loss_before * chunk_count
        return rec["bpb"]

    bpb_whole = _run(chunk_size=-1, persistence="reset", out_name="whole")
    bpb_carry = _run(chunk_size=32, persistence="carry_state", out_name="carry")
    # They should be close; the looser tolerance here reflects float noise
    # across chunked summation order vs whole-doc summation.
    assert abs(bpb_whole - bpb_carry) < 1e-3, (bpb_whole, bpb_carry)
```

**Step 2: Run** — expected FAIL if anything is off.

**Step 3: Fix any drift** discovered. Likely causes: autocast mismatches, model.eval() vs train() inconsistency, dtype coercions. The test is the acceptance.

**Step 4: Run** — PASS.

**Step 5: Commit**

```bash
git add tests/test_exp20_phase_a_smoke.py
git commit -m "exp20(harness): Phase A smoke — harness bpb matches forward-only eval"
```

---

### Task 10: Harness wrap-up — pod smoke against Exp 18 Test 4b checkpoint

**Files:** `configs/exp20/smoke_test4b.json` + `docs/runs/2026-04-XX-exp20-harness-smoke.md`.

**Step 0: Pre-push cleanliness check** — `git status` must show no untracked or modified files before push. Memory rule: "only committed code on pods."

```bash
git status
# Expected: "working tree clean"
git push origin exp20-ssm-native-ttt
```

**Step 1: Write the smoke config.** Create `configs/exp20/smoke_test4b.json` with the explicit contents below. `adapt_set=none, persistence_mode=reset` means this is a pure forward-only run — it exercises the full harness pipeline (DocStreamer → LegalityController → MetricsCollector) against the Exp 18 Test 4b checkpoint on a held-out eval slice.

```json
{
  "adapt_set": "none",
  "persistence_mode": "reset",
  "chunk_size": -1,
  "steps_per_chunk": 0,
  "delta_scale": 1.0,
  "log_a_shift": 0.0,
  "max_docs": 1000,
  "seed": 0,
  "jsonl_paths": ["/volume/fineweb/eval_docs_selected.jsonl"],
  "sp_model_path": "/volume/tokenizers/sp16384.model",
  "checkpoint_path": "/volume/ckpts/exp18_test4b_final.pt",
  "output_path": "/volume/exp20/smoke_test4b_metrics.jsonl"
}
```

Paths above are placeholders — update to the actual pod paths before running. Commit the config with real paths.

**Step 2: Rerun the Test 4b baseline eval path on the same 1000 docs.** The published Test 4b bpb number used a different chunking / state-reset contract from our harness, so an absolute-bpb comparison is not apples-to-apples. The apples-to-apples comparison is:

1. Rerun Test 4b's own eval code on the same 1000 docs (same JSONL slice).
2. Run the Exp 20 harness smoke config above on the same 1000 docs.
3. Compare bpb from (1) and (2).

```bash
# On pod:
python scripts/eval_test4b_baseline.py --docs /volume/fineweb/eval_docs_selected.jsonl --max-docs 1000 \
    --ckpt /volume/ckpts/exp18_test4b_final.pt --out /volume/exp20/test4b_rerun.jsonl
python scripts/run_exp20_eval.py --config configs/exp20/smoke_test4b.json
```

**Step 3: Acceptance gate.** The two bpb numbers must agree to within 0.01. If they don't, the harness has a bug — the reference rerun is ground truth.

**Step 4: Record findings** in `docs/runs/2026-04-XX-exp20-harness-smoke.md` (actual date). Include: both bpb numbers, delta, wall-time, any stability gate activity.

**Step 5: Commit the run log.**

```bash
git add docs/runs/*.md configs/exp20/smoke_test4b.json
git commit -m "exp20(harness): Phase A pod smoke vs Test 4b re-run baseline; bpb matches within 0.01"
```

**GATE:** if bpb drift between Test 4b rerun and Exp 20 smoke exceeds 0.05, stop and debug before any Phase B-G run. A broken harness invalidates all downstream.

---

## Phase II — Ablation runs (Tasks 11-16)

Tasks 11-16 are driven by config grids, not new modeling code (except Task 16 which optionally builds LoRA-r8). Each task: generate configs → dispatch seed-parallel launches on the 8-GPU pod → aggregate metrics → record findings.

Use `superpowers:dispatching-parallel-agents` for launching independent seeds concurrently on the 8 ranks.

### Task 11: Phase B — Axis 2 alone (no weight TTT)

**Files:**
- Create: `configs/exp20/phase_b/`
- Create: `docs/runs/2026-04-XX-exp20-phase-b.md` (the day you run it)

**Configs:** 6 × 3 seeds = 18 runs. Each config is `{adapt_set: none, persistence_mode: M, delta_scale: 1.0, log_a_shift: 0.0, chunk_size: whole_doc, steps_per_chunk: 0, max_docs: 50_000, checkpoint_path: <Exp19_final>, seed: s}` for M ∈ {reset, carry_state} and s ∈ {0, 1, 2}. (Carry-weights-only doesn't apply without weight TTT; skip. `carry_both` skipped for the same reason. Test trainable_h0 only where `steps_per_chunk>0`.)

**Step 1:** Generate configs programmatically via a small helper:

```python
# configs/exp20/phase_b/_gen.py
import json
from pathlib import Path

base = {
    "adapt_set": "none",
    "chunk_size": -1,  # whole_doc
    "steps_per_chunk": 0,
    "delta_scale": 1.0,
    "log_a_shift": 0.0,
    "max_docs": 50000,
    "eos_token": 0,
    "shard_paths": ["/volume/fineweb/eval.bin"],
    "checkpoint_path": "/volume/ckpts/exp19_final.pt",
}
out_dir = Path(__file__).parent
for mode in ["reset", "carry_state"]:
    for seed in [0, 1, 2]:
        cfg = dict(base, persistence_mode=mode, seed=seed,
                   output_path=f"/volume/exp20/phase_b/{mode}_s{seed}.jsonl")
        (out_dir / f"{mode}_s{seed}.json").write_text(json.dumps(cfg, indent=2))
```

**Step 2:** Launch on pod with 8-way seed-parallel. Use a launcher that assigns one config per GPU rank until the grid is exhausted.

**Step 3:** Aggregate: per-config mean bpb at doc {1, 10, 100, 1K, 10K, 50K}. Paired t-test: `carry_state - reset` across seeds.

**Step 4:** Record findings: mean bpb, paired-t statistic, p-value. Decision criterion: H1 rejects if `carry_state` beats `reset` by > 0.025 bpb and paired t-test p < 0.05.

**Step 5:** Commit run log.

```bash
git add docs/runs/2026-04-XX-exp20-phase-b.md configs/exp20/phase_b/
git commit -m "exp20(phase B): Axis 2 alone — <pass|fail> H1 with delta <X> bpb at n=3"
```

---

### Task 12: Phase C — Axis 1 screen at Axis 2 winner

**Files:**
- Create: `configs/exp20/phase_c/`
- Create: `docs/runs/2026-04-XX-exp20-phase-c.md`

**Configs:** Top-5 adapt sets × winner_from_B × 3 seeds = 15 runs. Chunk_size=256 (standard Legal TTT default). steps_per_chunk=1, eval_lr=0.064 initially.

Adapt sets to screen (Axis 1, pruned):
- `log_a` (SSM-native, smallest)
- `log_a+delta_proj` (joint memory+selectivity)
- `B_side` (in_proj + select_proj)
- `embed_rows_seen` (sparse; requires sparse-gradient handling in the optimizer)
- `lm_head` (architecture-agnostic anchor)

**Step 1:** Generate configs analogous to Task 11.
**Step 2:** Launch seed-parallel.
**Step 3:** Rank by accumulated bpb; verify stability gate didn't fire.
**Step 4:** Record: per-adapt-set bpb, per-doc curves, stability, compute-per-doc.
**Step 5:** Commit.

---

### Task 13: Phase D — Axis 3 alone (Δ modulation sweep)

**Configs:** no-grad; `adapt_set=none`, winner_B persistence, chunk_size=-1 (whole_doc since no TTT). Sweep delta_scale ∈ {0.25, 0.5, 1.0, 2.0, 4.0} × log_a_shift ∈ {-1, -0.5, 0, +0.5, +1} = 25 configs × 3 seeds = 75 runs. Forward-only, fast.

**Deliverable:** heatmap of (delta_scale, log_a_shift) → bpb. Pick winner(s) with largest bpb drop vs (1.0, 0.0).

---

### Task 14: Phase E — Compose winners

**Configs:** Top-2 from Phase C × winner_B × winner_D × 5 seeds = 10 runs. This is the H5 test: does the stack beat its best component?

Hypothesis-test: paired t-test on stacked config vs best-single-component config across seeds.

---

### Task 15: Phase F — Schedule Pareto at Phase E winner

**Configs:** Phase E winner × {chunk_size ∈ {64, 256, 1024, -1}} × {steps_per_chunk ∈ {1, 2, 5}} × {eval_lr ∈ {0.016, 0.032, 0.064, 0.128}} × 3 seeds. Full grid is 48 cells × 3 = 144; prune by first running the 3 chunk_size × 3 steps = 9 matrix at eval_lr=0.064 to find a plateau, then sweep LR only around the plateau.

**Deliverable:** bpb-vs-compute Pareto curve with each cell as a point. Pick the Pareto-extremal config whose bpb gain scales monotonically with compute (Section 1 rejection criterion).

---

### Task 16: Phase G — Envelope (`lm_head`, `lora_r8`, `all`)

**Files (only if lora_r8 needed):**
- Create: `src/chaoscontrol/eval_stream/lora.py`
- Create: `tests/test_eval_stream_lora.py`
- Modify: `src/chaoscontrol/eval_stream/ttt_runner.py` (LoRA param selection)

**Context:** Build LoRA-r8 only if Phase C's SSM-native winners don't already explain most of the effect. Otherwise skip to saves 1 day.

**Configs:** Phase E winner schedule × {lm_head, lora_r8, all} × 5 seeds. Test whether SSM-native stack beats architecture-agnostic baselines.

**LoRA implementation sketch (gated):**

```python
# src/chaoscontrol/eval_stream/lora.py
import torch.nn as nn

class LoRAAdapter(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8):
        super().__init__()
        self.base = base
        self.lora_A = nn.Parameter(torch.randn(rank, base.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A.T) @ self.lora_B.T


def wrap_projections_with_lora(model, rank=8):
    from chaoscontrol.core import ChaosSSMCore
    for core in model.modules():
        if isinstance(core, ChaosSSMCore):
            for name in ["in_proj", "out_proj", "select_proj", "gate_proj", "delta_proj"]:
                base = getattr(core, name)
                setattr(core, name, LoRAAdapter(base, rank=rank))
```

**Deliverable:** Phase G table: SSM-native stack vs `lm_head` vs `lora_r8` vs `all` at matched compute. Paper claim stands if SSM-native wins or ties within noise at lower param cost.

---

## Phase III — Analysis (Tasks 17-18)

### Task 17: Pareto extraction + experiment writeup

**Files:**
- Create: `notebooks/exp20_analysis.ipynb` OR `scripts/exp20_analysis.py` (no notebook dependency preferred)
- Create: `docs/reports/2026-04-XX-exp20-results.md`

**Step 1:** Consolidate all Phase B-G JSONL logs into a single parquet.
**Step 2:** Generate figures: Pareto curve (bpb vs compute), accumulation curve for top-3 configs, stability traces.
**Step 3:** Write results doc following Exp 21 design/report style: hypothesis outcomes, numbers, figures, decisions. Name winner config for submission handoff.
**Step 4:** Commit.

### Task 18: Submission-repo config handoff

**Files:** none in this repo (submission repo is separate).

Produce a self-contained config + checkpoint pair for the submission repo:
- Winning `RunConfig`
- Exp 19 base checkpoint
- Any LoRA adapter weights (if Phase G winner includes them)
- A README snippet describing the eval-time TTT protocol precisely enough that the submission repo can reproduce it.

Deliver to the submission-repo maintainer via the agreed channel. Mark Exp 20 complete.

---

## Phase I → II Gate

**Do not start Phase II (Task 11) until:**
1. All Task 9 smoke tests pass — including the chunked-carry-state canary (`test_chunked_carry_state_equals_whole_doc`).
2. Task 10 pod-smoke bpb matches the Test 4b rerun within 0.01 bpb (apples-to-apples — not the published Test 4b headline).
3. Leak-detection contract test (Task 5) passes on the pod with the real model, not just the tiny test model.
4. Task 3.5 `tests/test_initial_states_regression.py` passes (state plumbing works end-to-end).

**If any gate fails, the harness is the problem, not the ablation.** Do not proceed.

---

## Execution reminders

- After each task's commit, use `superpowers:verification-before-completion` to verify the commit cleanly before moving on — skill explicitly mandated for work that ends with "is this done?"
- Use `superpowers:systematic-debugging` when any test fails unexpectedly.
- After Task 10 and Task 16 (the largest deliverables), use `superpowers:requesting-code-review` before merging the worktree.
- Memory rules active: "Only committed code on pods," "Subagent review when away," "Run tests after every edit."
