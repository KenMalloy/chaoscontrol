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

## Phase I — Harness (Tasks 1-10)

Days 1-2 of the post-Exp-19 calendar. Build must complete before any ablation run.

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

**Files:**
- Create: `src/chaoscontrol/eval_stream/doc_stream.py`
- Create: `tests/test_eval_stream_doc_stream.py`

**Context:** FineWeb shards are tokenized in `src/chaoscontrol/data.py`. For eval we need doc-level iteration, not packed batches. Eval split must be disjoint from Exp 19 train. We use the existing eval shard path and split on doc boundary markers (0 or EOS token depending on tokenizer).

**Step 1: Write failing test**

```python
# tests/test_eval_stream_doc_stream.py
import numpy as np
import pytest
from chaoscontrol.eval_stream.doc_stream import DocStreamer


def test_iterates_docs_in_order(tmp_path):
    # Build a synthetic shard with 3 docs separated by EOS token (0)
    toks = np.array([10, 11, 12, 0,  20, 21, 0,  30, 31, 32, 33, 0], dtype=np.int32)
    shard = tmp_path / "eval.bin"
    toks.tofile(shard)

    streamer = DocStreamer(shard_paths=[shard], eos_token=0, max_docs=10)
    docs = list(streamer)

    assert len(docs) == 3
    assert docs[0].tokens == [10, 11, 12]
    assert docs[1].tokens == [20, 21]
    assert docs[2].tokens == [30, 31, 32, 33]
    assert all(d.doc_id == i for i, d in enumerate(docs))


def test_respects_max_docs(tmp_path):
    toks = np.array([1, 0, 2, 0, 3, 0, 4, 0], dtype=np.int32)
    shard = tmp_path / "eval.bin"
    toks.tofile(shard)
    docs = list(DocStreamer(shard_paths=[shard], eos_token=0, max_docs=2))
    assert len(docs) == 2


def test_raw_bytes_recorded(tmp_path):
    toks = np.array([100, 101, 0], dtype=np.int32)
    shard = tmp_path / "eval.bin"
    toks.tofile(shard)
    # Raw bytes computed from detokenizer; in this test we stub it via constant
    docs = list(DocStreamer(shard_paths=[shard], eos_token=0, max_docs=1,
                            bytes_per_token_estimate=4.0))
    # 2 tokens × 4 bytes/token = 8
    assert docs[0].raw_bytes == 8
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_eval_stream_doc_stream.py -v`
Expected: FAIL `ModuleNotFoundError`.

**Step 3: Implement**

```python
# src/chaoscontrol/eval_stream/doc_stream.py
from __future__ import annotations
from pathlib import Path
from typing import Iterator
import numpy as np

from chaoscontrol.eval_stream.types import DocRecord


class DocStreamer:
    """Iterates docs from tokenized eval shards. Splits on EOS token.

    Doc order is deterministic (shard order, then position). Eval-split disjointness
    vs Exp 19 train stream is enforced by caller via shard_paths choice.
    """

    def __init__(
        self,
        *,
        shard_paths: list[Path],
        eos_token: int,
        max_docs: int = 50_000,
        bytes_per_token_estimate: float = 4.0,  # FineWeb avg, ~4 bytes/subword for SP8192
    ) -> None:
        self.shard_paths = [Path(p) for p in shard_paths]
        self.eos_token = eos_token
        self.max_docs = max_docs
        self.bytes_per_token_estimate = bytes_per_token_estimate

    def __iter__(self) -> Iterator[DocRecord]:
        doc_id = 0
        for shard in self.shard_paths:
            arr = np.fromfile(str(shard), dtype=np.int32)
            buf: list[int] = []
            for t in arr:
                t_int = int(t)
                if t_int == self.eos_token:
                    if buf:
                        yield DocRecord(
                            doc_id=doc_id,
                            tokens=buf,
                            raw_bytes=int(len(buf) * self.bytes_per_token_estimate),
                        )
                        doc_id += 1
                        buf = []
                        if doc_id >= self.max_docs:
                            return
                else:
                    buf.append(t_int)
            if buf:
                yield DocRecord(
                    doc_id=doc_id,
                    tokens=buf,
                    raw_bytes=int(len(buf) * self.bytes_per_token_estimate),
                )
                doc_id += 1
                if doc_id >= self.max_docs:
                    return
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_eval_stream_doc_stream.py -v`
Expected: 3 PASS.

**Step 5: Commit**

```bash
git add src/chaoscontrol/eval_stream/doc_stream.py tests/test_eval_stream_doc_stream.py
git commit -m "exp20(harness): DocStreamer iterates eval docs on EOS boundary"
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
```

**Step 4: Run**

Expected: 2 PASS.

**Step 5: Commit**

```bash
git add src/chaoscontrol/eval_stream/metrics.py tests/test_eval_stream_metrics.py
git commit -m "exp20(harness): MetricsCollector with in-run stability gate"
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

    def __init__(self, module: nn.Module, *, delta_scale: float = 1.0, log_a_shift: float = 0.0):
        self.module = module
        self.delta_scale = float(delta_scale)
        self.log_a_shift = float(log_a_shift)
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._log_a_originals: list[tuple[nn.Parameter, torch.Tensor]] = []

    def _find_cores(self) -> list[nn.Module]:
        from chaoscontrol.core import ChaosSSMCore
        return [m for m in self.module.modules() if isinstance(m, ChaosSSMCore)]

    def __enter__(self):
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
    def _chunk_hash(chunk: torch.Tensor) -> int:
        # Hash based on token ids + chunk content to detect re-use
        return int(chunk.detach().cpu().numpy().tobytes().__hash__())

    def score_chunk(self, chunk: torch.Tensor) -> float:
        h = self._chunk_hash(chunk)
        if self.leak_detection and h in self._adapted_chunks:
            raise LeakDetectedError(
                f"Chunk hash {h} was adapt_on_chunk'd before score_chunk: Issue #1017 violation."
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

**Step 4: Run** — 2 PASS.

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
```

**Step 2: Run** — FAIL.

**Step 3: Implement**

```python
# src/chaoscontrol/eval_stream/ttt_runner.py
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Iterable


ADAPT_SET_PATTERNS: dict[str, list[str]] = {
    "none": [],
    "log_a": ["log_a"],
    "delta_proj": ["delta_proj"],
    "log_a+delta_proj": ["log_a", "delta_proj"],
    "B_side": ["in_proj", "select_proj"],
    "C_side": ["out_proj", "gate_proj"],
    "embed_rows_seen": ["embed"],  # sparse grad handled at opt step; param set is the whole table
    "lm_head": ["lm_head"],
    "lora_r8": ["lora_"],  # lora adapters named lora_A_<name> / lora_B_<name>; see Task 16
    "all": ["*"],  # sentinel handled below
}


def select_adapt_params(module: nn.Module, *, adapt_set: str) -> list[nn.Parameter]:
    """Return the list of parameters that match the adapt_set filter.

    Filter is by substring in the parameter's fully-qualified name. The `all`
    sentinel returns every parameter.
    """
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
        if any(pat in name for pat in patterns):
            if id(p) not in seen:
                out.append(p)
                seen.add(id(p))
    return out
```

**Step 4: Run** — 4 PASS.

**Step 5: Commit**

```bash
git add src/chaoscontrol/eval_stream/ttt_runner.py tests/test_eval_stream_ttt_runner.py
git commit -m "exp20(harness): param-group selector for Axis 1 adapt sets"
```

---

### Task 7: Persistence modes — state carry + trainable_h0

**Files:**
- Create: `src/chaoscontrol/eval_stream/persistence.py`
- Create: `tests/test_eval_stream_persistence.py`
- Modify: `src/chaoscontrol/eval_stream/ttt_runner.py` (add weight-snapshot helpers)

**Context:** Axis 2 — at doc boundary, choose: reset state, carry state, carry weight deltas, trainable `h₀`, or compositions. State is a list of per-block tensors returned by the SSM's `.step()` method; `trainable_h0` is a learnable parameter we add to the model at harness setup.

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
            self._state = [c._trainable_h0.expand(batch_size, -1).clone() for c in self._cores]
        elif self.mode == "trainable_h0+carry":
            if self._doc_idx == 0:
                self._state = [c._trainable_h0.expand(batch_size, -1).clone() for c in self._cores]
            # else keep prior state
        else:
            raise ValueError(f"unknown persistence_mode: {self.mode}")

    def get_state(self) -> list[torch.Tensor]:
        return self._state

    def set_state(self, state: list[torch.Tensor]) -> None:
        self._state = state


def attach_trainable_h0(model: nn.Module) -> None:
    """Add a trainable h0 vector to each ChaosSSMCore. Eval-time only."""
    for core in model.modules():
        if isinstance(core, ChaosSSMCore):
            if not hasattr(core, "_trainable_h0"):
                core._trainable_h0 = nn.Parameter(torch.zeros(1, core.dim))


def detach_trainable_h0(model: nn.Module) -> None:
    for core in model.modules():
        if isinstance(core, ChaosSSMCore):
            if hasattr(core, "_trainable_h0"):
                delattr(core, "_trainable_h0")
```

**Step 4: Run** — 3 PASS.

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
    # Tiny synthetic eval shard
    toks = np.concatenate([np.random.randint(1, 32, size=50).astype(np.int32),
                           np.array([0], dtype=np.int32)] * 5)
    shard = tmp_path / "eval.bin"
    toks.tofile(shard)

    # Tiny checkpoint
    from chaoscontrol.model import ChaosStudentLM
    m = ChaosStudentLM(vocab_size=32, dim=16, num_layers=2, block_type="ssm", a_mode="diag")
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save({"model": m.state_dict(),
                "config": {"vocab_size": 32, "dim": 16, "num_layers": 2,
                           "block_type": "ssm", "a_mode": "diag"}}, ckpt_path)

    out_path = tmp_path / "metrics.jsonl"
    cfg_path = tmp_path / "run.json"
    cfg_path.write_text(json.dumps({
        "adapt_set": "none", "persistence_mode": "reset",
        "chunk_size": 32, "steps_per_chunk": 0,
        "max_docs": 3, "eos_token": 0, "seed": 0,
        "shard_paths": [str(shard)],
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
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from chaoscontrol.eval_stream.types import RunConfig
from chaoscontrol.eval_stream.doc_stream import DocStreamer
from chaoscontrol.eval_stream.legality import LegalityController
from chaoscontrol.eval_stream.persistence import StateManager, attach_trainable_h0
from chaoscontrol.eval_stream.ttt_runner import select_adapt_params
from chaoscontrol.eval_stream.delta_mod import DeltaModulator
from chaoscontrol.eval_stream.metrics import MetricsCollector
from chaoscontrol.evaluation import compute_bpb


def _ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                           targets.reshape(-1), reduction="sum")


def _build_model(ckpt_path: Path) -> tuple[torch.nn.Module, dict]:
    from chaoscontrol.model import ChaosStudentLM
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = blob["config"]
    model = ChaosStudentLM(**cfg)
    model.load_state_dict(blob["model"], strict=False)
    return model, cfg


def _build_optimizer(params, lr: float, persistent_moments: bool):
    # Use Muon for non-scalar params; fall back to AdamW for scalars (log_a)
    from chaoscontrol.optim.muon import Muon
    non_scalar = [p for p in params if p.ndim >= 2]
    scalar = [p for p in params if p.ndim < 2]
    opt_groups = []
    if non_scalar:
        opt_groups.append(Muon(non_scalar, lr=lr))
    if scalar:
        opt_groups.append(torch.optim.AdamW(scalar, lr=lr))
    return opt_groups


def _iter_chunks(tokens: list[int], chunk_size: int):
    if chunk_size < 0:  # whole_doc
        yield tokens
        return
    for i in range(0, len(tokens), chunk_size):
        yield tokens[i:i + chunk_size]


def run(cfg: RunConfig, shard_paths: list[str], eos_token: int) -> None:
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = _build_model(Path(cfg.checkpoint_path))
    model.to(device)

    if "trainable_h0" in cfg.persistence_mode:
        attach_trainable_h0(model)

    adapt_params = select_adapt_params(model, adapt_set=cfg.adapt_set)
    optimizers = _build_optimizer(adapt_params, cfg.eval_lr, cfg.persistent_muon_moments) \
        if adapt_params else []

    streamer = DocStreamer(shard_paths=[Path(p) for p in shard_paths],
                           eos_token=eos_token, max_docs=cfg.max_docs)
    state_mgr = StateManager(model, persistence_mode=cfg.persistence_mode)
    controller = LegalityController(model, loss_fn=_ce)
    collector = MetricsCollector(output_path=Path(cfg.output_path))

    with DeltaModulator(model, delta_scale=cfg.delta_scale, log_a_shift=cfg.log_a_shift):
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

                # Adapt
                if adapt_params and cfg.steps_per_chunk > 0:
                    for opt in optimizers:
                        loss_after = controller.adapt_on_chunk(
                            chunk, optimizer=opt, steps=cfg.steps_per_chunk,
                        )
                    loss_after_sum += loss_after or 0.0
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
    shard_paths = raw.pop("shard_paths")
    eos_token = raw.pop("eos_token", 0)
    cfg = RunConfig(**{k: v for k, v in raw.items() if k in RunConfig.__dataclass_fields__})
    run(cfg, shard_paths=shard_paths, eos_token=eos_token)


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
    m = ChaosStudentLM(vocab_size=32, dim=16, num_layers=2, block_type="ssm", a_mode="diag")
    m.eval()
    tokens = torch.randint(1, 32, (1, 64))

    # Naive forward-only
    with torch.no_grad():
        out = m(tokens)
        logits = out["logits"] if isinstance(out, dict) else out
        naive = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                                tokens[:, 1:].reshape(-1), reduction="sum").item()

    controller = LegalityController(
        m, loss_fn=lambda lg, tg: F.cross_entropy(
            lg.reshape(-1, lg.size(-1)), tg.reshape(-1), reduction="sum"
        ),
    )
    harness_score = controller.score_chunk(tokens)
    # Float-noise tolerance: bf16 path has ~1e-2 relative error, but on fp32 model this should be tight
    assert abs(harness_score - naive) < 1e-4
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

**Files:** none (operational verification)

**Step 1:** Upload the harness-only branch to the pod. Memory rule: only committed code runs on pods.

```bash
git push origin exp20-ssm-native-ttt
```

**Step 2:** On the pod, check out the branch and run the Exp 18 Test 4b checkpoint through the harness with `adapt_set=none, persistence=reset` on 1000 eval docs.

```bash
python scripts/run_exp20_eval.py --config configs/exp20/smoke_test4b.json
```

**Step 3:** Verify the bpb matches the Test 4b reported bpb to ~0.01.

**Step 4:** Record findings in `docs/runs/2026-04-18-exp20-harness-smoke.md` (or whatever the day is — actual date when running).

**Step 5:** Commit the run log.

```bash
git add docs/runs/*.md configs/exp20/smoke_test4b.json
git commit -m "exp20(harness): Phase A pod smoke vs Test 4b baseline; bpb matches within 0.01"
```

**GATE:** if bpb drift exceeds 0.05, stop and debug before any Phase B-G run. A broken harness invalidates all downstream.

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
1. All Task 9 smoke tests pass.
2. Task 10 pod-smoke bpb matches Exp 18 Test 4b within 0.01 bpb.
3. Leak-detection contract test (Task 5) passes on the pod with the real model, not just the tiny test model.

**If any gate fails, the harness is the problem, not the ablation.** Do not proceed.

---

## Execution reminders

- After each task's commit, use `superpowers:verification-before-completion` to verify the commit cleanly before moving on — skill explicitly mandated for work that ends with "is this done?"
- Use `superpowers:systematic-debugging` when any test fails unexpectedly.
- After Task 10 and Task 16 (the largest deliverables), use `superpowers:requesting-code-review` before merging the worktree.
- Memory rules active: "Only committed code on pods," "Subagent review when away," "Run tests after every edit."
