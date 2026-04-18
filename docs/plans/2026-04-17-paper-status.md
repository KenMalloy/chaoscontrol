# 2026-04-17 Paper Status and Mainline Plan

## Purpose

This is the authoritative "where we are" document after merging the active
experiment worktrees back onto `main` on 2026-04-17.

It replaces the old paper framing that centered Wernicke / sleep /
structured-memory architecture. That older material is still useful as repo
history, but it is not the current paper target.

## Mainline status

`main` now contains the merged code paths from:

- `exp19-phase1a` — bespoke fp8 / TransformerEngine-fork work
- `exp20-ssm-native-ttt` — legal score-before-update eval-stream harness
- `exp21-sgns-tokenizer` — offline SGNS tokenizer-initialization pipeline

### What is now true on `main`

| Area | Status on `main` | Verification |
|---|---|---|
| Exp 19 fp8 path | Merged, including descriptor-cache flyweight and bespoke BGRADB path | Python-side tests skip cleanly without CUDA; GPU compile/runtime validation still required |
| Exp 20 legality harness | Merged and fixed so adaptation can thread the same recurrent state context used for scoring | `tests/test_eval_stream_legality.py` |
| Exp 20 state plumbing | `ChaosStudentLM.forward(..., initial_states=...)` and final-state threading are present | `tests/test_initial_states_regression.py` |
| Exp 20 driver | `scripts/run_exp20_eval.py` runs tiny-stream smoke flow | `tests/test_run_exp20_eval.py` |
| Exp 21 SGNS training | Negative sampling now follows the model device; pair generation no longer materializes the full pair set | `tests/unit/test_sgns_sampler.py`, `tests/unit/test_sgns_train.py` |
| Exp 21 paper verdict logic | `thesis_validating` now requires both primary/secondary effects and both control families | `tests/unit/test_exp21_analyze.py` |
| Shard rebuild safety | Build manifest now keys idempotent-skip on `sp_train_docs`, `shard_size`, and `num_workers` too | `tests/test_build_sp_shards.py` |

### Verification run on 2026-04-17

The following passed on the merged `main` state:

```bash
pytest tests/test_eval_stream_legality.py tests/test_initial_states_regression.py tests/test_run_exp20_eval.py -q
pytest tests/unit/test_sgns_sampler.py tests/unit/test_sgns_train.py tests/unit/test_exp21_analyze.py tests/test_build_sp_shards.py -q
python3 -m py_compile scripts/train_sgns.py scripts/exp21_analyze.py scripts/build_sp_shards.py scripts/run_exp20_eval.py src/chaoscontrol/sgns/train.py src/chaoscontrol/sgns/sampler.py src/chaoscontrol/eval_stream/legality.py
pytest tests/test_cublaslt_fp8.py -q
```

Observed result:

- 52 targeted tests passed
- `tests/test_cublaslt_fp8.py` skipped cleanly on the non-CUDA environment

## Important implementation changes worth mentioning in the paper

### 1. Exp 20 legality fix

The score-before-update harness now threads the same recurrent state context
through both the scoring pass and the adaptation pass. Without this, a
`carry_state` run can look "legal" while adapting from a reset state.

```python
score_loss, final_states = controller.score_chunk(
    chunk,
    initial_states=prev_state if prev_state else None,
)
loss_after = controller.adapt_on_chunk(
    chunk,
    optimizer=opt,
    steps=cfg.steps_per_chunk,
    initial_states=prev_state if prev_state else None,
)
```

That change matters for correctness, not just software hygiene.

### 2. Exp 21 SGNS memory/performance fix

The original SGNS epoch loop built every `(center, context)` pair up front and
then shuffled the full materialized pair set. That is fine on toy streams and a
bad failure mode on tens of millions of tokens.

The merged version streams per-offset slices instead:

```python
for spec_idx in spec_order:
    offset, forward = specs[spec_idx]
    centers = stream[offset:] if forward else stream[:-offset]
    contexts = stream[:-offset] if forward else stream[offset:]
    for batch_idx in batch_order:
        yield centers[start:end], contexts[start:end]
```

This keeps the algorithm the same while removing the hidden full-pair memory
cliff.

### 3. Exp 19 fp8 descriptor safety fix

The descriptor cache no longer assumes every valid tensor at a fixed shape has
the same effective operand alignment. The cache key now includes the actual
runtime alignment bytes for cuBLAS A/B/C/D slots, so sliced-but-layout-valid
CUDA tensors do not reuse a heuristic chosen for a fresher 256-byte-aligned
allocation.

This is merged, but it still needs H100 validation.

## Safe claims right now

These are the claims that are currently supportable from the merged codebase
and the review pass.

1. The repo now has a legal score-before-update SSM-native TTT harness with
   chunk-level state threading and explicit regression tests for that contract.
2. The repo now has an offline SGNS tokenizer-initialization pipeline whose
   implementation no longer contains the obvious CUDA-device bug or the
   full-pair materialization cliff.
3. The repo now has stricter paper-gate logic for Exp 21: primary/secondary
   wins alone are not enough; control arms are required for a
   `thesis_validating` verdict.
4. The current merged codebase is in a much better position for a paper about
   fair comparison and negative results than for a paper claiming a final SOTA
   number.

## Claims that are not safe yet

These should **not** be stated as paper conclusions yet.

1. Any claim that Exp 20 has a paper-valid positive result. The harness is now
   correct, but the legacy analysis paths still contain exploratory-statistics
   shortcuts.
2. Any claim that Exp 21 is thesis-validating in the actual experimental
   record. The gating logic is correct now, but the required control data still
   has to exist and pass.
3. Any claim that the Exp 19 fp8 path is production-safe across operand
   alignment patterns. The cache fix is merged, but it has not yet been
   validated on H100 with the compiled extension.
4. Any claim that CFR or gate ablations in the legacy decision/eval scripts are
   paper-grade. Two specific reasons:
   - `experiments/09_revised_architecture/run_eval_ablation.py` still uses a
     bucket-0 CFR prior in eval and unpaired Welch-style summaries.
   - `experiments/09_revised_architecture/run_decision.py` still reflects an
     older exploratory decision workflow rather than the stricter paired,
     confirmatory framing needed for the paper.

## Current paper framing

The strongest paper story is no longer "biologically inspired full architecture
beats the field." The strongest paper story is:

1. Parameter Golf inspired the testbed and the budget discipline.
2. We built an equal-infrastructure comparison where SSM-native mechanisms are
   tested under the same legality and compute rules as the transformer-heavy
   frontier.
3. Several plausible ideas failed under stricter controls.
4. Those failures are scientifically useful because they show where local wins,
   proxy improvements, or unfair eval assumptions do not transfer.

In other words: the paper should center fair methodology, negative results, and
the boundary conditions of SSM-native improvements.

## Paper-safe experiment buckets

| Bucket | Status | Paper role |
|---|---|---|
| Exp 19 infrastructure / fp8 | merged, partly verified | systems contribution / enabling infra |
| Exp 20 eval-stream harness | merged, verified as harness | methodology contribution |
| Exp 20 positive effect sizes | not yet paper-safe | future result section only after paired confirmatory runs |
| Exp 21 SGNS pipeline | merged, verified as implementation | experiment pipeline / ablation substrate |
| Exp 21 thesis verdict | not yet paper-safe | future confirmatory result only if controls pass |
| Exp 14/16/17 failures | already evidenced in repo docs/reports | negative-results section / motivation for current design |

## Immediate next steps

### Before calling the codebase "paper-ready"

1. Run the fp8 extension on H100 and explicitly validate the alignment-keyed
   cache path, ideally including a sliced-view regression.
2. Upgrade the legacy decision/eval summary scripts from exploratory unpaired
   tests to paired repeated-measures summaries wherever the seed structure
   exists, or clearly mark them as exploratory-only outputs.
3. Run the missing Exp 21 controls and only then allow any
   `thesis-validating` language outside internal docs.
4. Create a single machine-readable paper-results registry so confirmatory
   tables cannot silently mix exploratory and final runs.

### Suggested writing order

1. Write the methodology and evaluation-protocol sections first.
2. Write the negative-results / failed-hypothesis section second.
3. Only then freeze the headline claim for Exp 20 or Exp 21, based on the
   confirmatory evidence actually available.

## Recommended paper outline from here

1. **Introduction** — Parameter Golf as the motivating benchmark and why fair
   legality/compute constraints matter.
2. **Methodology** — equal-infrastructure harness, score-before-update
   contract, matched-budget comparison rules.
3. **Implementation** — Exp 19 fp8 infra, Exp 20 eval harness, Exp 21 SGNS
   pipeline.
4. **Negative results** — typed buffer, state-oracle sparse attention, local
   wins that did not transfer.
5. **Confirmatory results** — only the experiments that pass the stricter
   paper gates.
6. **Discussion** — what was learned about SSM-native mechanisms under real
   competition-style constraints.

## Bottom line

`main` is now unified around the current research direction, and the immediate
merge blockers that would have invalidated the story have been fixed.

What we have today is a credible paper substrate and a much sharper internal
truth standard. What we do **not** yet have is permission to overclaim the
results. That distinction should stay visible in every draft until the missing
confirmatory runs are complete.
