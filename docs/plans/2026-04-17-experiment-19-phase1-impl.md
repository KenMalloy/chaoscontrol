# Exp 19 Phase 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the Exp 18 → Exp 19 training-time underfit gap by removing every piece of per-step overhead that currently subtracts from steps-per-second at matched wall-clock, and unlock fp8 as a genuine throughput lever at our `dim=256` scale by replacing TransformerEngine's high-overhead kernel path with a bespoke cuBLASLt path.

**Architecture:** Two parallel tracks that converge into one ablation matrix.
- **Phase 1A (bf16 throughput, sequential):** fused grad clip, fused Muon step, full-path `torch.compile`. Each lands behind a config flag so the ablation can toggle it. TDD gates ensure each kernel produces byte-identical output to the reference it replaces before a throughput claim is made.
- **Phase 1B (bespoke fp8, parallel track):** cuBLASLt fused cast+GEMM+cast with deferred amax tracking, plumbed behind the existing `precision` abstraction. Numerical validation against stock TE confirms the kernel is arithmetically correct; bf16-matched-seed training trajectory validation confirms it trains.
- **Phase 1C (ablation matrix):** One-at-a-time lever-leave-out matrix × {bf16, fp8} run via the persistent-DDP launcher. Summarizer reports tokens/sec, final_loss, bpb, and the marginal contribution of each lever.

Tracks 1A and 1B can run in parallel via worktree isolation. Track 1C gates on both.

**Tech Stack:**
- `torch` 2.11 + `torch.compile` (inductor)
- `torch.distributed` on gloo (tests) / NCCL (pods)
- `transformer_engine.pytorch` 2.13 (fp8 baseline for numerical validation)
- `cuBLASLt` via `torch.ops.aten._scaled_mm` (fp8 fused path)
- `pytest` + the persistent-DDP launcher from `experiments/19_prereqs/`

**Prerequisites (all landed this session):**
- Persistent-DDP launcher (`experiments/19_prereqs/run_persistent_launcher.py`)
- Config-sensitive idempotent skip (`experiments/19_prereqs/runner_persistent_ddp.py:108`)
- Warmup-restore contract (`experiments/19_prereqs/runner_persistent_ddp.py:_warmup_and_restore`)
- Stale-marker + world-size + zero-runnable pre-flights (launcher)
- GPU-side loss accumulator + coalesced all-reduce (`src/chaoscontrol/train_ssm.py`, `src/chaoscontrol/distributed.py`)
- Precision abstraction (`src/chaoscontrol/precision.py`)

**Not in scope (Phase 1B/C only if Phase 1A doesn't hit throughput target):**
- Selective SSM / Mamba S6 kernels (Phase 1 item #4 in the overview memory — defer until Phase 1A/B land)
- Megakernels (same)
- dim / layers / d_state sweeps (Phase 3)

---

## Track 1A — bf16 per-step overhead (sequential)

### Task 1A-1: Fused gradient clipping

**Files:**
- Modify: `src/chaoscontrol/distributed.py` (add `clip_grad_norm_fused`)
- Modify: `src/chaoscontrol/train_ssm.py:304-305` (call site toggled by config flag)
- Test: `tests/test_distributed.py` (new `TestClipGradNormFused` class)

**Why:** `torch.nn.utils.clip_grad_norm_` iterates parameters in Python, computes per-tensor norms, stacks them, computes a global norm, and then applies a scalar to every grad — five kernel launches minimum per step, more with a per-param coalesce. The fused version computes the global norm via a single coalesced op over the already-flattened grad buffer produced by `allreduce_grads` and applies one in-place `.mul_` across the flat buffer.

**Step 1: Write the failing test**

Add to `tests/test_distributed.py`:

```python
class TestClipGradNormFused:
    """Byte-identical to torch.nn.utils.clip_grad_norm_ for the L2 norm case."""

    @staticmethod
    def _build_model_with_mixed_grads(seed: int) -> nn.Module:
        torch.manual_seed(seed)
        model = nn.Sequential(nn.Linear(8, 4), nn.Linear(4, 2))
        for i, p in enumerate(model.parameters()):
            p.grad = torch.randn_like(p) * (i + 1) + float(i)
        return model

    def test_matches_reference_on_clipping_regime(self) -> None:
        """Grads well above max_norm — clipping must fire on both paths."""
        from chaoscontrol.distributed import clip_grad_norm_fused
        model_ref = self._build_model_with_mixed_grads(seed=7)
        model_new = self._build_model_with_mixed_grads(seed=7)
        max_norm = 0.1  # chosen to fire clipping on these grads
        ref_total = torch.nn.utils.clip_grad_norm_(
            model_ref.parameters(), max_norm,
        )
        new_total = clip_grad_norm_fused(model_new.parameters(), max_norm)
        assert torch.allclose(ref_total, new_total, rtol=1e-6, atol=0.0), (
            f"total_norm mismatch: ref={ref_total} new={new_total}"
        )
        for (n1, p1), (n2, p2) in zip(
            model_ref.named_parameters(), model_new.named_parameters(),
        ):
            assert n1 == n2
            assert torch.equal(p1.grad, p2.grad), (
                f"clipped grad mismatch on {n1!r}"
            )

    def test_matches_reference_below_clip_threshold(self) -> None:
        """Grads below max_norm — both paths must be no-ops for values."""
        from chaoscontrol.distributed import clip_grad_norm_fused
        model_ref = self._build_model_with_mixed_grads(seed=11)
        model_new = self._build_model_with_mixed_grads(seed=11)
        max_norm = 1e6  # above any realistic grad norm
        torch.nn.utils.clip_grad_norm_(model_ref.parameters(), max_norm)
        clip_grad_norm_fused(model_new.parameters(), max_norm)
        for p1, p2 in zip(model_ref.parameters(), model_new.parameters()):
            assert torch.equal(p1.grad, p2.grad)

    def test_empty_params_is_no_op(self) -> None:
        from chaoscontrol.distributed import clip_grad_norm_fused
        total = clip_grad_norm_fused([], max_norm=1.0)
        assert float(total) == 0.0

    def test_grad_identity_preserved(self) -> None:
        """p.grad tensor identity must survive — optimizer holds refs."""
        from chaoscontrol.distributed import clip_grad_norm_fused
        model = self._build_model_with_mixed_grads(seed=23)
        ptrs_before = [p.grad.data_ptr() for p in model.parameters()]
        clip_grad_norm_fused(model.parameters(), max_norm=0.1)
        ptrs_after = [p.grad.data_ptr() for p in model.parameters()]
        assert ptrs_before == ptrs_after
```

**Step 2: Run test to verify it fails**

```bash
source .venv/bin/activate
python -m pytest tests/test_distributed.py::TestClipGradNormFused -v
```
Expected: FAIL with `ImportError: cannot import name 'clip_grad_norm_fused'`.

**Step 3: Write minimal implementation**

Add to `src/chaoscontrol/distributed.py`:

```python
def clip_grad_norm_fused(
    parameters: Iterable[torch.nn.Parameter],
    max_norm: float,
    norm_type: float = 2.0,
) -> torch.Tensor:
    """Coalesced L2 grad clip — byte-equivalent to torch.nn.utils.clip_grad_norm_.

    Flattens every grad into one buffer, computes the global L2 norm via
    a single kernel, and applies one in-place multiplicative clip factor
    across the buffer. Avoids the per-param norm + stack + global-norm
    dance the stdlib helper walks through in Python.

    Grad tensor identity (.data_ptr()) is preserved — the clip writes
    back into the original tensors via unflatten + copy_, matching the
    contract allreduce_grads established on 2026-04-17.
    """
    if float(norm_type) != 2.0:
        raise NotImplementedError(
            "clip_grad_norm_fused only implements L2 (norm_type=2.0); "
            "fall back to torch.nn.utils.clip_grad_norm_ for other norms."
        )
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return torch.zeros(())
    flat = torch._utils._flatten_dense_tensors(grads)
    total_norm = flat.norm(p=2)
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    if clip_coef_clamped.item() < 1.0:
        flat.mul_(clip_coef_clamped)
        synced = torch._utils._unflatten_dense_tensors(flat, grads)
        for g, s in zip(grads, synced):
            g.copy_(s)
    return total_norm
```

Also add `from typing import Iterable` to the imports at the top of the file if not present.

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_distributed.py -v
```
Expected: all `TestClipGradNormFused` + existing tests PASS.

**Step 5: Wire the call site behind a config flag**

Modify `src/chaoscontrol/train_ssm.py:304-305`:

```python
if grad_clip_norm > 0.0:
    if fused_grad_clip:
        clip_grad_norm_fused(model.parameters(), grad_clip_norm)
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
```

Add `fused_grad_clip: bool = False` to `train_ssm_for_budget`'s signature (after `grad_clip_norm`). Thread the flag through `run_one_seed` in `experiments/19_prereqs/runner_persistent_ddp.py` reading `config.get("fused_grad_clip", False)`.

**Step 6: Add parity regression to the step-level test**

Extend `tests/test_train_ssm.py::TestTrainSSMStepEquivalence` with a test that runs `train_ssm_for_budget` with `fused_grad_clip=False` and `fused_grad_clip=True` on the same seed and asserts identical final parameters (bit-equal on CPU, because the fused path is a strict rewrite of an identical math operation).

```bash
python -m pytest tests/test_train_ssm.py tests/test_distributed.py -v
```

**Step 7: Commit**

```bash
git add src/chaoscontrol/distributed.py src/chaoscontrol/train_ssm.py tests/test_distributed.py tests/test_train_ssm.py
git commit -m "exp19(phase1a): fused L2 grad clip + config flag + parity tests"
```

---

### Task 1A-2: Fused Muon step

**Files:**
- Modify: `src/chaoscontrol/optim/muon.py` (add a `fused: bool` path to `step()`)
- Test: `tests/test_optim_muon.py` (new `TestFusedMuonParity` class)

**Why:** `Muon.step` at `src/chaoscontrol/optim/muon.py:107-164` iterates params in Python, running Newton-Schulz on each matrix param one-by-one and an inline AdamW on each scalar param one-by-one. At 10.7M params across ~40 nn.Linear submodules, that's ~40 Python-level loop iterations per optimizer step, each dispatching ~6 kernels. The fused version groups matrix params by shape (so NS can run on a stacked batch), coalesces scalar params into one AdamW update, and drops the per-param Python overhead.

**Key constraint (correction from 2026-04-17 review):** Newton-Schulz runs in bf16 on CUDA per `src/chaoscontrol/optim/muon.py:32`. The fused path MUST preserve this compute dtype — fp32 NS is materially slower on Hopper tensor cores and fp8 NS is numerically unstable.

**Step 1: Write the failing test**

Add to `tests/test_optim_muon.py`:

```python
class TestFusedMuonParity:
    """Fused Muon must produce byte-identical updates to the reference loop
    when the two paths see the same grads and optimizer state.

    The fused path groups matrix params by shape and runs a batched NS;
    the reference iterates one-by-one. Mathematically equivalent; the
    test locks this against a future batched-NS edit that silently
    changes numerical reduction order.
    """

    def _build_model(self, seed: int) -> nn.Sequential:
        torch.manual_seed(seed)
        # Mix of matrix shapes and a scalar param to exercise both branches.
        m = nn.Sequential(
            nn.Linear(8, 4),
            nn.Linear(4, 4),  # same shape as another — groupable
            nn.Linear(4, 4),
            nn.LayerNorm(4),  # scalar params
        )
        return m

    def test_updates_match_reference_loop(self) -> None:
        from chaoscontrol.optim.muon import Muon
        m_ref = self._build_model(seed=13)
        m_new = self._build_model(seed=13)
        for p_r, p_n in zip(m_ref.parameters(), m_new.parameters()):
            assert torch.equal(p_r.data, p_n.data)  # identical init
            p_r.grad = torch.randn_like(p_r)
            p_n.grad = p_r.grad.clone()

        opt_ref = Muon(m_ref.parameters(), lr=0.02, fused=False)
        opt_new = Muon(m_new.parameters(), lr=0.02, fused=True)
        opt_ref.step()
        opt_new.step()

        for (n_r, p_r), (n_n, p_n) in zip(
            m_ref.named_parameters(), m_new.named_parameters(),
        ):
            assert n_r == n_n
            assert torch.allclose(p_r.data, p_n.data, rtol=1e-5, atol=1e-6), (
                f"fused Muon drift on {n_r!r}: "
                f"max|diff|={(p_r.data - p_n.data).abs().max().item()}"
            )

    def test_fused_step_dispatches_fewer_python_iterations(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Structural: fused path must not call newton_schulz_orthogonalize
        once per matrix param. Count invocations via a wrapped helper;
        unfused = N_matrix_params calls, fused = number of unique shapes.
        """
        from chaoscontrol.optim import muon as muon_mod
        calls_unfused: list[tuple] = []
        calls_fused: list[tuple] = []

        orig = muon_mod.newton_schulz_orthogonalize
        def counting_ns(tensor, **kw):
            return orig(tensor, **kw)
        monkeypatch.setattr(muon_mod, "newton_schulz_orthogonalize", counting_ns)
        # ... [test records invocations — see full impl]

    def test_ns_compute_dtype_unchanged_on_cuda(self) -> None:
        """Fused path must keep NS compute_dtype=bf16 on CUDA, fp32 on CPU
        — matches the unfused default at optim/muon.py:32.
        """
        # See impl notes: verifies that batching doesn't silently upcast.
        ...
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_optim_muon.py::TestFusedMuonParity -v
```
Expected: FAIL — `Muon.__init__() got an unexpected keyword argument 'fused'`.

**Step 3-4: Implement the fused path**

Add a `fused: bool = False` kwarg to `Muon.__init__` (`src/chaoscontrol/optim/muon.py:74`). Branch in `step()`:

- **Unfused path (current):** keep the existing per-param loop verbatim — it's the reference.
- **Fused path:**
  1. Partition params by (`is_matrix`, shape). Each shape-group of matrix params gets stacked into a single tensor; batched `newton_schulz_orthogonalize` runs once per shape-group.
  2. Scalar params are coalesced into one flat `exp_avg` / `exp_avg_sq` buffer (bucketed by their existing state entries) so the AdamW update becomes one in-place `addcdiv_` per *bucket*, not per-param.
  3. Write results back into individual `p.data` via `_unflatten_dense_tensors` (pattern already established in `allreduce_grads`).

Keep the existing `compute_dtype` logic from line 31-32 — apply it uniformly inside the batched NS so per-shape-group NS runs in bf16 on CUDA.

**Step 5: Run tests**

```bash
python -m pytest tests/test_optim_muon.py tests/test_train_ssm.py -v
```
Expected: PASS.

**Step 6: Add end-to-end parity regression**

Extend `tests/test_train_ssm.py::TestTrainSSMStepEquivalence` with a test that runs `train_ssm_for_budget` with `fused_muon=True` and `fused_muon=False` on the same seed, assert identical loss trajectory for the first 10 steps.

**Step 7: Commit**

```bash
git add src/chaoscontrol/optim/muon.py tests/test_optim_muon.py tests/test_train_ssm.py
git commit -m "exp19(phase1a): fused Muon step (batched NS by shape + coalesced AdamW) + parity"
```

---

### Task 1A-3: Full-path `torch.compile`

**Files:**
- Modify: `src/chaoscontrol/train_ssm.py` (wrap `train_ssm_step` in `torch.compile` behind a flag)
- Test: `tests/test_train_ssm.py` (new `TestFullPathCompile` class)

**Why:** Currently only `core._diag_recurrence` is compiled (`src/chaoscontrol/core.py`). The rest of the forward pass (projection layers, norms, LM head) runs eager — every one of those ops pays Python dispatch overhead. `torch.compile`ing the whole step closes that gap.

**Risk:** Compile failures on the SSM path are documented in memory (`feedback_bf16_dtype_gotchas`, `project_criticality_status_2026-04-12`). A graph break inside `torch.compile(dynamic=False)` is a silent fallback to eager — the compile succeeds at torchlevel but individual ops drop back to Python. This task MUST include a graph-break audit.

**Step 1: Write the failing test**

```python
class TestFullPathCompile:
    """Full-path compile of train_ssm_step: no graph breaks, parity with eager."""

    def test_compiled_step_matches_eager_on_small_batch(
        self, bare_ssm_model: ChaosStudentLM,
    ) -> None:
        """Compiled and eager paths produce identical grads at fp32/CPU.

        Eager and inductor are byte-identical for deterministic ops on
        CPU; any drift > fp32 epsilon is a graph-break or a silent
        fallback to a different reduction order.
        """
        inputs, targets = _make_batch(batch=2, seq=16, vocab=64, seed=77)

        model_eager = bare_ssm_model
        model_comp = copy.deepcopy(model_eager)

        # Eager baseline grads.
        model_eager.zero_grad(set_to_none=True)
        train_ssm_step(
            model=model_eager, inputs=inputs, targets=targets,
            chunk_size=32, compile_full_path=False,
        )
        eager_grads = {
            n: p.grad.clone()
            for n, p in model_eager.named_parameters() if p.grad is not None
        }

        model_comp.zero_grad(set_to_none=True)
        train_ssm_step(
            model=model_comp, inputs=inputs, targets=targets,
            chunk_size=32, compile_full_path=True,
        )
        comp_grads = {
            n: p.grad.clone()
            for n, p in model_comp.named_parameters() if p.grad is not None
        }

        for name in eager_grads:
            diff = (eager_grads[name] - comp_grads[name]).abs().max().item()
            assert diff < 1e-5, f"compile drift on {name!r}: {diff}"

    def test_graph_break_free(
        self, bare_ssm_model: ChaosStudentLM, capsys: pytest.CaptureFixture,
    ) -> None:
        """Assert zero graph breaks via TORCH_LOGS=graph_breaks.

        A graph break in the compiled step means some ops fall back to
        eager inside the compile region — the "throughput win" from
        compile is then a fraction of what it should be. Fail the build
        on any break so we see regressions at CI time, not pod time.
        """
        import os
        os.environ["TORCH_LOGS"] = "graph_breaks"
        torch._dynamo.reset()
        inputs, targets = _make_batch(batch=2, seq=16, vocab=64, seed=78)
        bare_ssm_model.zero_grad(set_to_none=True)
        train_ssm_step(
            model=bare_ssm_model, inputs=inputs, targets=targets,
            chunk_size=32, compile_full_path=True,
        )
        captured = capsys.readouterr()
        assert "Graph break" not in captured.err, (
            f"compiled step has graph breaks (expected zero). "
            f"stderr:\n{captured.err}"
        )
```

**Step 2: Run test to verify it fails**

Expected: `TypeError: train_ssm_step() got an unexpected keyword argument 'compile_full_path'`.

**Step 3: Implement**

Add `compile_full_path: bool = False` to `train_ssm_step`. When True, wrap the body in a `torch.compile(dynamic=False, fullgraph=True)` closure. `fullgraph=True` raises on graph break instead of silently falling back — which is what we want for Phase 1A correctness gating.

```python
@functools.lru_cache(maxsize=4)
def _compiled_step_fn(fullgraph: bool):
    return torch.compile(_train_ssm_step_impl, fullgraph=fullgraph, dynamic=False)

def train_ssm_step(..., compile_full_path: bool = False):
    if compile_full_path:
        return _compiled_step_fn(fullgraph=True)(...)
    return _train_ssm_step_impl(...)
```

Expect the `test_graph_break_free` assertion to fail the first time; address each graph break at the SSM model level (likely causes: Python-level dict accesses on model config, `.item()` calls in the loss chunker, Python-level branching on tensor shapes). Every fix is a small code change + test rerun.

**Step 4: Run tests**

```bash
python -m pytest tests/test_train_ssm.py::TestFullPathCompile -v
```

Expected: PASS after graph-break cleanup.

**Step 5: Wire config flag**

Thread `compile_full_path` through `run_one_seed` in `experiments/19_prereqs/runner_persistent_ddp.py` via `config.get("compile_full_path", False)`.

**Step 6: Commit (after each graph-break fix)**

```bash
git commit -m "exp19(phase1a): full-path torch.compile behind flag + graph-break audit"
```

If graph-break cleanup is material, split into commits per underlying fix.

---

### Task 1A-4: bf16 microbenchmark — verify Phase 1A claims

**Files:**
- Create: `experiments/19_phase1/bench_phase1a.py`
- Test: N/A (this is a benchmark, not a test; results go to a json under `results/`)

**Why:** Each of 1A-1..3 was landed with a correctness gate but no throughput measurement. Before claiming "~X% faster per step," measure it. Memory rule: "No rigid shutdown automation ... gate destructive actions on success checks; DONE marker is not proof of success" applies here — a successful compile is not proof of a throughput win.

**Step 1: Write the microbenchmark script**

```python
# experiments/19_phase1/bench_phase1a.py
"""Phase 1A throughput microbenchmark.

Runs train_ssm_for_budget for a fixed step count on the submission
regime model (dim=256, layers=4, V=16384, seq=512, bs=1024/rank, ws=2 CPU
test / ws=4 pod). Cycles through every combination of Phase 1A flags:

    {fused_grad_clip: 0/1} × {fused_muon: 0/1} × {compile_full_path: 0/1}

Reports tokens/sec and peak_vram_mb per combination. Warmup-restore
applies so first-step compile doesn't contaminate timing.
"""
```

Build on the existing `runner_persistent_ddp.py` patterns (same tokens, same warmup-restore, same seed handling). Output a JSON per combination.

**Step 2: Run on a pod with the 4×H100 CUDA 13 image**

```bash
# On pod:
python experiments/19_phase1/bench_phase1a.py \
    --data-path baselines/parameter_golf/datasets/fineweb10B_sp16384 \
    --sp-model-path baselines/parameter_golf/tokenizers/fineweb_16384_bpe.model \
    --output-dir experiments/19_phase1/results_bench_phase1a/ \
    --world-size 4 --seeds 1337 2674 --n-steps 200
```

**Step 3: Success gate**

Decision criterion for each flag:
- Tokens/sec with flag on vs off, averaged over seeds: ≥ **+5%** speedup.
- Peak VRAM with flag on vs off: within ±5% (so we don't trade speed for headroom).
- Final loss after 200 steps (same seed, same data): within 0.02 (these changes are throughput-only, should not move loss).

If a flag fails any gate, open a followup to investigate; do NOT include it in the Phase 1C matrix default-on condition until the gate is met.

**Step 4: Commit bench script + results summary**

Results JSON stays untracked (per repo convention); the bench script and a short `RESULTS_BENCH_PHASE1A.md` in `experiments/19_phase1/` summarizing the decision get committed.

```bash
git add experiments/19_phase1/bench_phase1a.py experiments/19_phase1/RESULTS_BENCH_PHASE1A.md
git commit -m "exp19(phase1a): bf16 microbenchmark + Phase 1A lever decisions"
```

---

## Track 1B — bespoke fp8 matmul path (parallel track)

**Time estimate:** 3-5 days focused work. Track can run in a separate worktree in parallel with Track 1A; does not depend on 1A landing.

**High-level architecture:** Replace the body of `te.Linear.forward` with a direct `torch.ops.aten._scaled_mm` call path that fuses cast + GEMM + cast into one cuBLASLt invocation and tracks amax in a deferred on-device buffer. The rest of the model sees the same `.forward(input) -> output` contract, so the precision abstraction and `maybe_promote_linears_to_te` plumbing stay intact. The new kernel lives behind a new precision flag `"fp8_fused"` (distinct from `"fp8"` which remains stock TE as the numerical reference).

### Task 1B-1: cuBLASLt scaffold + numerical reference

**Files:**
- Create: `src/chaoscontrol/kernels/fp8_linear.py` (the bespoke Linear class)
- Create: `tests/test_fp8_linear.py`

**Why:** Before writing the fused kernel, establish (a) the Module API it implements, (b) a harness that runs stock TE and the bespoke path on identical inputs and compares outputs, (c) fixed-seed numerical tolerance bounds. The reference against which future kernel edits are validated is the existing stock-TE path — we already know bf16 ≠ stock fp8, so "matches stock TE fp8" is the correctness bar, not "matches bf16."

**Step 1: Write the test — numerical reference framework**

```python
# tests/test_fp8_linear.py
"""Bespoke fp8 Linear numerical validation against stock TE.

The bespoke path must produce arithmetically-equivalent output to stock
te.Linear for every input shape we use in training. "Equivalent" here
means: within fp8 representation granularity — the two paths run the
same math but may use different amax scaling strategies. Tolerance is
tuned per test and documented with its rationale.
"""
import pytest
pytest.importorskip("transformer_engine", reason="bespoke fp8 tests require TE as reference")
pytest.importorskip("torch")

import torch
import transformer_engine.pytorch as te

from chaoscontrol.kernels.fp8_linear import FusedFP8Linear


@pytest.fixture
def cuda_required():
    if not torch.cuda.is_available():
        pytest.skip("bespoke fp8 requires CUDA")


class TestFusedFP8LinearMatchesStockTE:
    """Output-level parity at fixed inputs. Tight tolerance documented per test."""

    def test_forward_matches_te_on_submission_regime_shape(
        self, cuda_required,
    ) -> None:
        """dim=256, one sample — the Phase 1 target shape."""
        torch.manual_seed(0)
        ref = te.Linear(256, 256, device="cuda", params_dtype=torch.bfloat16)
        bespoke = FusedFP8Linear.from_nn_linear(
            torch.nn.Linear(256, 256, device="cuda", dtype=torch.bfloat16),
        )
        # Copy weights so both layers have identical storage.
        bespoke.weight.data.copy_(ref.weight.data)
        x = torch.randn(32, 256, device="cuda", dtype=torch.bfloat16)
        with te.fp8_autocast(enabled=True):
            y_ref = ref(x)
        y_new = bespoke(x)
        # fp8 reps differ at ~1e-2 relative; absolute tolerance set to
        # 3x the worst per-element error observed on a calibration run.
        assert torch.allclose(y_ref, y_new, rtol=3e-2, atol=3e-2), (
            f"bespoke fp8 output drift: max abs diff = "
            f"{(y_ref - y_new).abs().max().item()}"
        )
```

**Step 2: Run to verify it fails**

Expected: `ImportError: cannot import ... FusedFP8Linear`.

**Step 3-4: Stub the module with a stock-TE delegate**

Land `FusedFP8Linear` as a thin subclass/wrapper that internally calls `te.Linear`. Tests pass by construction. The scaffold exists; the actual fused kernel lands in 1B-2 and 1B-3.

**Step 5: Commit the scaffold**

```bash
git commit -m "exp19(phase1b): FusedFP8Linear scaffold + stock-TE numerical reference"
```

### Task 1B-2: Fused cast+GEMM+cast kernel

**Files:**
- Modify: `src/chaoscontrol/kernels/fp8_linear.py`
- Test: `tests/test_fp8_linear.py` (add performance + correctness tests)

**Why:** The whole point of the track — collapse the 3 kernel launches of `cast→GEMM→cast` into one `torch.ops.aten._scaled_mm` call. `_scaled_mm` on CUDA 13 + H100 dispatches a single fused cuBLASLt op.

**Step 1-4:** Implement the forward path using `_scaled_mm`:

```python
@staticmethod
def _fused_forward(
    x_bf16: torch.Tensor,          # [B, K] bf16
    w_bf16: torch.Tensor,           # [N, K] bf16 master
    x_amax_history: torch.Tensor,   # on-device rolling max
    w_amax_history: torch.Tensor,   # on-device rolling max
) -> torch.Tensor:
    # 1. Compute per-tensor scales from amax history.
    # 2. _scaled_mm fuses cast(x)+cast(w)+mm+cast(y) into one kernel.
    # 3. Update amax history (deferred — no sync per call).
    x_scale = _scale_from_amax(x_amax_history)
    w_scale = _scale_from_amax(w_amax_history)
    y = torch.ops.aten._scaled_mm(
        x_bf16.to(torch.float8_e4m3fn) * x_scale,
        w_bf16.to(torch.float8_e4m3fn).t() * w_scale,
        out_dtype=torch.bfloat16,
        scale_a=x_scale,
        scale_b=w_scale,
    )
    _update_amax(x_amax_history, x_bf16.abs().amax())
    _update_amax(w_amax_history, w_bf16.abs().amax())
    return y
```

**Step 5: Run numerical tests**

```bash
python -m pytest tests/test_fp8_linear.py -v
```

Expected: PASS. If a test fails with tolerance overflow, investigate *why* — the fused kernel should produce output matching stock TE at fp8 granularity. Material divergence means the scale factors are computed differently than TE expects.

**Step 6: Add throughput microbench**

Inside `tests/test_fp8_linear.py` add a `pytest.mark.skip_in_ci` or `pytest.mark.slow` test that benchmarks the fused path vs stock TE on the submission shape and asserts `bespoke_time < 0.7 * te_time` — i.e., at least a 30% speedup, anything less isn't worth the engineering.

**Step 7: Commit**

```bash
git commit -m "exp19(phase1b): _scaled_mm fused cast+GEMM+cast path + numerical parity"
```

### Task 1B-3: Deferred amax tracking

**Files:**
- Modify: `src/chaoscontrol/kernels/fp8_linear.py`

**Why:** TE's per-call amax update is a GPU→GPU reduction that forces a sync. Accumulate amax over N calls and sync once at the end of training iteration (or step). The tradeoff is slightly less accurate scale factors for up to N matmuls — acceptable for training, not for inference.

**Step 1-4:** Implement a module-level amax history buffer. Every `.forward()` writes its per-tensor max into a ring buffer without syncing; a `flush_amax_history()` call (invoked once per optimizer step) reduces the buffer into the active scale factor. Document the semantic tradeoff in the module docstring.

**Step 5: Test:** Training-dynamics parity — two training runs at the same seed, one with stock TE, one with the bespoke fused + deferred amax path, first 20 losses within 2% of each other. If they drift materially, the scale factor trajectory is materially different; investigate.

**Step 6: Commit**

### Task 1B-4: Integrate via precision abstraction

**Files:**
- Modify: `src/chaoscontrol/precision.py` (add `"fp8_fused"` dtype + `maybe_promote_linears_to_fused_fp8`)
- Modify: `experiments/19_prereqs/runner_persistent_ddp.py:297-308` (route `precision == "fp8_fused"` to the new promoter)

**Why:** Drop the bespoke path in as a sibling of stock fp8 so the ablation matrix can toggle between `bf16` / `fp8` / `fp8_fused` without code changes.

**Step 1-5:** Add the dtype string to `autocast_context` (same TE `fp8_autocast` — the autocast context still wraps the compute), add a `maybe_promote_linears_to_fused_fp8` helper paralleling the existing TE promoter, wire the precision string through the launcher default matrix to optionally include `fp8_fused` as a condition.

**Step 6: Commit**

### Task 1B-5: Training-trajectory validation

**Files:**
- Create: `experiments/19_phase1/validate_fp8_fused.py`

**Why:** Numerical parity on a single forward pass is necessary but not sufficient. A working fp8 kernel must also train — accumulated drift over a realistic step count must not materially hurt final loss.

**Step 1:** Run the persistent launcher at the submission regime with `precision=fp8_fused` vs `precision=fp8` (stock TE), 4 seeds, 600s budget each. Use the existing Test 10 matrix structure with the new condition added.

**Step 2: Success gate:**
- Final loss: within 0.05 of stock TE fp8 at the same seed (matches the bf16↔fp8 "quality gate" from Exp 18 Test 10).
- tokens/sec: ≥ 50% faster than stock TE fp8 at matched precision.
- Meets bf16 tokens/sec within 20% (the original claim that motivated the whole track — fp8 should be at least competitive with bf16 once the kernel overhead is gone).

If any gate fails, the bespoke path is either wrong or not worth it. Park and report.

**Step 3: Commit results summary**

```bash
git commit -m "exp19(phase1b): fp8_fused training-trajectory validation + verdict"
```

---

## Track 1C — ablation matrix

Prerequisite: Track 1A bf16 levers meeting their +5% throughput gates (Task 1A-4). Track 1B optional; if `fp8_fused` meets its training gate, it's a third precision condition in the matrix, otherwise the matrix stays bf16-only.

### Task 1C-1: Matrix assembly

**Files:**
- Create: `experiments/19_phase1/build_matrix_phase1.py`

**Why:** The Phase 1 matrix is a one-at-a-time lever-leave-out: N conditions where each toggles exactly ONE lever off relative to the "all on" baseline, plus the "all off" stock baseline. With 3 bf16 levers (grad clip, Muon, compile) + optional fp8_fused, the matrix is:

```
(bf16_stock,
 bf16_all,
 bf16_no_fused_clip,
 bf16_no_fused_muon,
 bf16_no_compile,
 [fp8_stock, fp8_all, fp8_no_clip, ...] if fp8_fused ships)
```

× 4 seeds = 20 entries (bf16-only) or 36 entries (with fp8_fused).

**Step 1-4:** Write `build_matrix_phase1()` returning a list of config dicts compatible with `run_persistent_launcher.py --matrix-json`. Each entry sets the appropriate flags. Write unit tests covering: matrix size matches the expected count; each lever is toggled off in exactly one non-baseline condition; seed × condition is unique.

**Step 5: Commit**

### Task 1C-2: Run the matrix on the pod

**Step 1:** Launch via persistent-DDP with `--matrix-json experiments/19_phase1/matrix_phase1.json --budget 600` × 2 seeds first (sanity), then × 4 seeds full.

**Step 2:** Verify all gates on post-run summary (`_check_output_integrity` must report `success=N, errors=0`). Any `real_error` is a blocker for the final verdict — no silent-success bias.

### Task 1C-3: Summarizer

**Files:**
- Create: `experiments/19_phase1/summarize_phase1.py`

**Why:** Report the *marginal contribution* of each lever as `(all_on_tok_per_s - lever_off_tok_per_s) / all_on_tok_per_s`, with paired-t-test significance vs the same baseline. Ken's Exp 18 summarizer pattern (coherent-pair resolver + full-coverage gate + quality gate) is the template — reuse its code, don't rewrite.

**Step 1-4:** Write the summarizer. Produce a markdown table per precision with one row per lever and columns: marginal-tok-per-sec, ΔΔ-final-loss, paired-t p-value, verdict (ship/park).

**Step 5: Commit**

---

## Validation protocols

### Numerical equivalence bars

| Change | Reference | Tolerance | Rationale |
|---|---|---|---|
| Fused grad clip (1A-1) | `torch.nn.utils.clip_grad_norm_` | bit-equal on CPU fp32 | Strict rewrite of same math |
| Fused Muon (1A-2) | existing per-param Muon.step | rtol=1e-5 on CPU fp32 | Batched NS may reorder reductions minimally |
| Full-path compile (1A-3) | eager train_ssm_step | 1e-5 on CPU fp32 | Inductor and eager should agree at fp32 |
| Fused fp8 Linear (1B-2) | stock te.Linear under fp8_autocast | rtol=3e-2 on CUDA | fp8 representation granularity |
| Fused fp8 training (1B-5) | stock TE fp8 training, same seed | ΔΔ final_loss ≤ 0.05 | Matches Exp 18 Test 10 quality gate |

### Throughput gates

- Per lever (1A): ≥ +5% tok/s on the submission regime, ±5% peak VRAM, Δ final_loss ≤ 0.02.
- fp8_fused (1B): ≥ +50% tok/s vs stock TE fp8, within 20% of bf16 tok/s.

### Ablation matrix gates

- `_check_output_integrity` must report zero real errors before the summarizer runs.
- Paired-t p-value for each lever's marginal contribution < 0.05 before declaring "ship."
- Conditions that fail the significance test but show mean improvement get a "inconclusive — rerun with more seeds" verdict, not a silent ship.

---

## Success criteria

Phase 1 lands when:
1. Every lever in the Phase 1A matrix has a signed, significant marginal contribution (positive or documented-negative).
2. If fp8_fused ships, it either wins vs bf16 on matched-wall-clock final_loss or it's explicitly parked with the measurements that justify parking.
3. The "all levers on" condition's tokens/sec at ws=4 on the submission regime is measurably higher than the current Exp 18 Test 4b number (2.107× baseline; any improvement over that is the phase-win bar).
4. `torch.compile(fullgraph=True)` reports zero graph breaks on the training step.
5. All new tests pass; `git status` clean; commit messages stand alone for a public-repo reader.

---

## Deferred / explicitly out of scope

- Selective SSM / Mamba S6 kernels (Phase 1 item #4 in the plan memory) — defer until the per-step overhead baseline is measured; they're replacements for the diag recurrence, not additions, so the ablation shape changes.
- Megakernels — same reason.
- dim / layers / d_state sweeps — Phase 3.
- 8-GPU scaling (ws=8 LR re-screen, ws=8 NCCL stability) — ride into the first Phase 1 8-GPU run per `project_exp19_prereqs_note.md`, not a Phase 1 work item.
- Optimizer re-screen at lower-loss regime — Exp 19b, after Phase 1 lands.
