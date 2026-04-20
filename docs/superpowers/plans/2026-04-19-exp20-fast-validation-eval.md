# Exp20 Fast Validation Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fast, distributed, full-validation eval engine that can prove Parameter Golf 50k-document throughput before Exp20 TTT claims are run.

**Architecture:** Keep the existing Exp20 JSONL runner as the correctness harness, but add a cache-backed runtime path for performance. The fast path uses a generated validation cache with token IDs and doc metadata, shards docs across ranks, aggregates exact BPB components, and records explicit parallel semantics for TTT behavior.

**Tech Stack:** Python, PyTorch, torch.distributed/torchrun, SentencePiece, pytest, existing `chaoscontrol.eval_stream` modules.

---

### Task 1: Validation Cache Schema

**Files:**
- Create: `src/chaoscontrol/eval_stream/val_cache.py`
- Test: `tests/test_eval_stream_val_cache.py`

- [ ] **Step 1: Write tests for cache round-trip**

  Add tests that build a tiny JSONL fixture, encode it with SentencePiece, write a cache, load the cache, and assert doc order, token IDs, token spans, raw byte counts, and total docs.

- [ ] **Step 2: Implement the cache schema**

  Add a small API:

  ```python
  @dataclass(frozen=True)
  class CachedDoc:
      doc_id: int
      token_start: int
      token_len: int
      raw_bytes: int

  def write_val_cache(...): ...
  def load_val_cache(cache_dir: Path) -> ValCache: ...
  ```

Store generated arrays under a cache directory and keep a JSON manifest with tokenizer path/hash, source path/size/mtime, `max_docs`, and schema version.

- [ ] **Step 3: Verify cache tests**

  Run:

  ```bash
  .venv/bin/python -m pytest tests/test_eval_stream_val_cache.py -q
  ```

### Task 2: Cache Builder CLI

**Files:**
- Create: `scripts/build_exp20_val_cache.py`
- Test: `tests/test_build_exp20_val_cache.py`
- Docs: `experiments/20_ssm_native_ttt/README.md`

- [ ] **Step 1: Write CLI tests**

  Use a tiny JSONL + SentencePiece model and assert the CLI writes a cache directory and manifest. The test must not commit generated cache files.

- [ ] **Step 2: Implement CLI**

  Add:

  ```bash
  python scripts/build_exp20_val_cache.py \
    --jsonl-path baselines/parameter_golf/datasets/docs_selected.jsonl \
    --sp-model-path baselines/parameter_golf/tokenizers/fineweb_8192_bpe.model \
    --cache-dir /workspace/cache/exp20_val_8192 \
    --max-docs 50000
  ```

  The command should be idempotent: if the manifest matches inputs, skip rebuild unless `--force` is passed.

- [ ] **Step 3: Document generated-artifact policy**

  Update Exp20 docs: commit builder/schema/tests, do not commit cache contents by default.

### Task 3: Fast Score-Only Runner

**Files:**
- Create: `scripts/run_exp20_fast_score.py`
- Modify: `src/chaoscontrol/eval_stream/val_cache.py`
- Test: `tests/test_run_exp20_fast_score.py`

- [ ] **Step 1: Write tiny-run tests**

  Compare fast scorer output against `scripts/run_exp20_eval.py` for `reset + no-TTT` on a tiny fixture. Assert same aggregate BPB within numerical tolerance.

- [ ] **Step 2: Implement single-process fast score**

Load cached docs, form batches of chunks, score under `torch.inference_mode()`, accumulate CE and raw bytes, and write a summary with the existing result-status fields.

Record-facing mode must include chunk-boundary targets: for chunked scoring,
the final logit from chunk N scores the first token of chunk N+1. Keep a
legacy compatibility flag for comparisons against `scripts/run_exp20_eval.py`,
which historically skipped those boundary targets.

Hot-loop shape policy:

- Stage cache tokens onto the target device once before the scoring loop.
- Length-sort each rank's work for dense full-width chunk groups, but preserve
  original doc order in the JSONL output.
- Treat `doc_batch_size` as an upper bound and cap effective microbatches with
  `max_batch_tokens / chunk_size` so sorted longest-doc batches do not OOM.
- Keep recurrent states as per-layer batch tensors instead of rebuilding them
  with per-doc slice/cat operations each chunk.
- Expose `--torch-compile-mode reduce-overhead` as an explicit GPU benchmark
  knob; do not silently enable it until pod measurements show a net win. The
  initial full-shape compile probe exceeded several minutes before first batch
  completion, so CUDA graph capture or a smaller fixed-shape wrapper should be
  investigated separately.

- [ ] **Step 3: Remove hot-loop CPU syncs**

  Accumulate CE tensors on device and synchronize once per batch or at end. Avoid per-chunk `.item()`.

### Task 4: Distributed Sharding

**Files:**
- Modify: `scripts/run_exp20_fast_score.py`
- Test: `tests/test_run_exp20_fast_score.py`

- [ ] **Step 1: Add rank range partitioning tests**

  For N docs and W ranks, assert every doc is assigned exactly once and ranges are deterministic.

- [ ] **Step 2: Add torch.distributed support**

  Read `RANK`, `WORLD_SIZE`, and `LOCAL_RANK`; set CUDA device; shard docs by rank; all-reduce CE, raw bytes, tokens, and docs; write final summary on rank 0.

- [ ] **Step 3: Record throughput projection**

  For `world_size=4`, write projected 8x wall time as `elapsed_seconds * 4 / 8` and classify the performance gate separately from `record_eligible`.

### Task 5: Parallel Semantics Strategy

**Files:**
- Create: `src/chaoscontrol/eval_stream/parallel_semantics.py`
- Modify: `scripts/run_exp20_fast_score.py`
- Test: `tests/test_eval_stream_parallel_semantics.py`

- [ ] **Step 1: Define selectable behaviors**

  Implement strategy names:

  ```text
  reset
  rank_local_carry_state
  global_sequential_carry_state
  rank_local_weight_ttt
  global_weight_ttt
  ```

- [ ] **Step 2: Enforce implemented/unimplemented states**

  `reset` works first. Other strategies should parse and write into summaries, but raise a clear `NotImplementedError` until implemented.

- [ ] **Step 3: Document implications**

  Explain that rank-local carry/adaptation is legal if each rank adapts only after scoring its shard tokens, but it is not identical to a single global sequential online trajectory.

### Task 6: Existing Harness Cleanup

**Files:**
- Modify: `src/chaoscontrol/eval_stream/legality.py`
- Test: existing `tests/test_eval_stream_legality.py`, `tests/test_run_exp20_eval.py`

- [ ] **Step 1: Add regression test for disabled leak detection**

  Assert `score_chunk(..., leak_detection=False)` does not call `_chunk_hash`.

- [ ] **Step 2: Gate chunk hashing**

  Only hash scored chunks when leak detection is enabled. Only hash adapted chunks when needed for future leak checks.

- [ ] **Step 3: Verify correctness harness**

  Run:

  ```bash
  .venv/bin/python -m pytest tests/test_eval_stream_legality.py tests/test_run_exp20_eval.py -q
  ```

### Task 7: Pod Performance Gate

**Files:**
- Docs only unless fixes are needed.

- [ ] **Step 1: Build cache on pod**

  Generate cache under `/workspace/cache/...`; do not harvest generated cache into git.

- [ ] **Step 2: Run 1xH100 smoke**

  Score a small doc prefix and compare against the correctness harness.

- [ ] **Step 3: Run 4xH100 full validation**

  Use `torchrun --standalone --nproc_per_node=4 scripts/run_exp20_fast_score.py ...`.

- [ ] **Step 4: Decide gate**

  Green if projected 8x wall time is comfortably under 600s. Amber if barely under. Red if above 600s.

### Task 8: Fast TTT Reintroduction

**Files:**
- Add only after Tasks 1-7 pass.

- [ ] **Step 1: Add no-gradient mechanisms first**

  Implement fast `reset`, `rank_local_carry_state`, `delta_scale`, and `log_a_shift`.

- [ ] **Step 2: Add tiny adapt sets**

  Add `log_a` and `delta_proj` score-before-adapt loops with rank-local semantics first.

- [ ] **Step 3: Run bounded and full-validation TTT gates**

  Label all runs with `parallel_semantics`, `full_validation_complete`, and `record_eligible`.

---

## Generated Artifact Policy

Commit source code, tests, schemas, and docs. Do not commit generated validation caches, pod outputs, or matrix config directories unless there is a specific reproducibility reason and the artifact size/content is reviewed.

## Success Criteria

- Fast score-only `reset` matches correctness-harness BPB on tiny fixtures.
- Full 50k validation completes on 4xH100 with a credible projection to 8x under 600 seconds.
- A true 8x run can produce `record_eligible: true` for score-only.
- TTT runs record their exact parallel semantics and cannot be confused with final Parameter Golf scores unless `full_validation_complete` and `record_eligible` are true.
