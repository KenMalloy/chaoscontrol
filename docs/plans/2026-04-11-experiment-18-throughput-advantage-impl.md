# [DEPRECATED] Exp 18: Throughput Advantage — Implementation Plan

> **⚠ DEPRECATED 2026-04-12.**
>
> This implementation plan has been superseded. See
> `docs/plans/2026-04-12-experiment-18-throughput-levers-design.md`
> for the new direction.
>
> **Why deprecated:** This plan targeted a "sweep → rescore → targeted
> retrain" workflow that Phase 0 benchmarks (run overnight 2026-04-11→12)
> rendered moot. At peak throughput (bs=1024 on 8×H100 DDP, ~786K tok/s),
> we see only ~4.7% of the 10B-token corpus in 600s — not enough coverage
> for "hardest N%" subset selection to be a meaningful axis.
>
> Phase 0 infrastructure (`bench_throughput.py`, runner, tests) remains
> valid. Phase A conditions (`baseline_b32`, `sweep_only`, `sweep_target_*`,
> `sweep_random_retrain`) are no longer planned.
>
> Keep this doc as historical context for the "sweep-and-target" framing
> and the preflight/expensive boundary discussion, both of which apply
> equally to the new design.

---

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement Experiment 18 as a clean training-strategy experiment that tests whether the fast compiled-scan SSM can convert high-batch throughput into better learning by sweeping a large fraction of the corpus, rescoring with a frozen model, and spending the remaining budget on harder windows.

**Framing:** This experiment is only scientifically meaningful once it reaches the 8xH100 pod. Before that, we can validate software correctness, but not the throughput hypothesis itself. The implementation should therefore be split into a cheap preflight stage and an expensive hardware-bound stage with an explicit handoff.

**Tech Stack:** Python 3.11+, PyTorch, sentencepiece, yaml, existing ChaosControl training/eval infrastructure, Exp 15/16 experiment-runner patterns.

---

## Cheap vs Expensive Boundary

### Preflight (cheap)

Runs on local or inexpensive hardware. This stage exists to verify
mechanics, not to support scientific claims.

**Allowed conclusions:**

- coverage/window math is correct
- wall-clock accounting is correct
- frozen rescoring is correct
- subset selection and control wiring are correct
- CLIs and result schemas work

**Not allowed as evidence:**

- max feasible batch size
- true tokens/sec
- true VRAM ceiling
- H100 large-batch LR stability
- whether sweep + target is actually better than baseline

### Phase 0 and beyond (expensive)

The expensive portion of Exp 18 begins at **Phase 0** on the **8xH100
pod**. That is where the real experiment starts:

- tokenizer screen (`SP8192` vs `SP16384`)
- max feasible batch size
- tokens/sec and projected coverage
- VRAM ceiling
- LR stability at large batch

**Rule:** No throughput or learning-strategy claim should be made from
preflight runs. Scientific evidence starts at Phase 0.

---

## Decisions Locked In

### 1. Tokenizer choice

Do **not** hardcode `SP8192` up front.

Exp 15.5 found:

- `sp_d256_L4` (`SP8192`): mean bpb `1.9674`
- `sp16384_d256_L4` (`SP16384`): mean bpb `1.9595`

That is only a modest quality improvement, but it is real enough that
Exp 18 should treat tokenizer choice as part of the throughput-aware
design space.

**Selection rule after Phase 0:**

1. Both tokenizers must pass large-batch stability.
2. Primary selector: highest projected **useful coverage** inside the
   sweep budget.
3. If one tokenizer materially changes feasibility, choose it even if
   its old standalone bpb was slightly worse.
4. If the throughput/coverage difference is small and does not change
   the feasibility regime, prefer `SP16384` because it already showed a
   small quality edge.

**Definition of "small" for this experiment:**

- projected coverage delta under ~10%, and
- no change in whether the intended sweep/rescore/retarget schedule is
  executable inside the wall-clock budget.

### 2. First expensive Phase A pass

Start with the **4-condition core matrix** only:

| Condition | Purpose |
|---|---|
| `baseline_b32` | Current honest control |
| `sweep_only` | Tests whether coverage helps by itself |
| `sweep_target_top10` | Tests whether hardness targeting adds value |
| `sweep_random_retrain` | Controls for two-phase structure and equal overhead |

Defer `top25` and `top5` until the first expensive pass shows that
`top10` is clearly earning its keep.

**Promotion rule:**

Expand to `top25` / `top5` only if `sweep_target_top10` beats both
`sweep_only` and `sweep_random_retrain` by a meaningful margin and with
consistent per-seed direction.

### 3. Partial coverage policy

If a full intended sweep is infeasible at the validated max batch,
Phase A should use the **largest feasible partial sweep** over the real
dataset.

Do **not** redefine the dataset to manufacture a fake "full sweep."

### 4. No minimum-coverage kill gate, but add a reconsideration gate

Do **not** impose a hard minimum-coverage kill gate such as "must reach
50% or abort." Even a 20% sweep could still be scientifically useful if
the two-phase structure plus hardness targeting beats baseline.

However, Phase 0 should emit a **reconsideration warning** when
projected sweep coverage is very low.

**Policy:**

- If projected sweep coverage is `< 25%` of the intended corpus during
  the sweep phase, do not automatically kill the experiment.
- Instead, flag the run as **low-coverage regime** and require the Phase
  0 summary to report:
  - projected coverage
  - projected rescore overhead
  - projected retarget time remaining
  - whether the sweep/target split should be reconsidered before Phase A

This preserves scientific flexibility while making it obvious when Exp
18 has drifted from "coverage-first" into "modest coverage plus smart
revisit" territory.

---

## File Scope

Keep Exp 18 **experiment-local first**:

```text
experiments/18_throughput_advantage/
  DESIGN.md
  bench_throughput.py
  runner_exp18.py
  run_exp18.py
  test_exp18.py
```

### Shared-code policy

Default to keeping scheduling and selection logic inside
`runner_exp18.py`.

Only extract a helper into `src/chaoscontrol/` if:

- the helper is small,
- it is clearly reusable, and
- duplication would otherwise be more confusing than the shared surface
  area.

Examples of acceptable extraction:

- a small deterministic "batch from explicit start indices" helper
- a tiny "per-window unreduced CE scorer" helper

Examples of out-of-scope refactors:

- reworking the general training loop around callbacks
- restructuring the full experiment framework
- adding distributed training infrastructure

---

## Task 1: Scaffold the Experiment and Preflight Harness

**Files:**

- Create: `experiments/18_throughput_advantage/bench_throughput.py`
- Create: `experiments/18_throughput_advantage/runner_exp18.py`
- Create: `experiments/18_throughput_advantage/run_exp18.py`
- Create: `experiments/18_throughput_advantage/test_exp18.py`
- Create: `experiments/18_throughput_advantage/DESIGN.md` (copy/symlink)

**Goal:** Stand up the experiment directory and cheap preflight
validation path before any pod usage.

**Implementation notes:**

- Copy the launcher/result structure from
  `experiments/16_entropy_sparse_attention/run_exp16.py`.
- Keep the top-level flow explicit:

```python
result = {
    "phase": "preflight" | "phase0" | "phaseA",
    "tokenizer": "sp8192" | "sp16384",
    "condition": "...",
    "timings": {...},
    "coverage": {...},
    "selection": {...},
    "eval": {...},
}
```

- Add a cheap smoke mode with tiny synthetic token tensors so the CLIs
  can be exercised without FineWeb or H100s.

**Verification:**

- `PYTHONPATH=src .venv/bin/python -m pytest experiments/18_throughput_advantage/test_exp18.py -q`
- tiny synthetic smoke run for both CLIs
- one smoke run per Phase A condition on toy data
- one smoke run that exercises the full `sweep -> rescore -> select -> retarget`
  chain with asserted phase timings and subset sizes

---

## Task 2: Implement Coverage Planning and Budget Accounting

**Files:**

- Modify: `experiments/18_throughput_advantage/runner_exp18.py`
- Test: `experiments/18_throughput_advantage/test_exp18.py`

**Goal:** Make coverage and wall-clock accounting explicit and testable.

**Requirements:**

- Generate non-overlapping sweep windows with `stride=seq_len`
- Measure coverage as **unique prediction targets scored**
- Support:
  - full intended sweep
  - largest feasible partial sweep
  - random subset control
- Track every phase against the same wall clock:

```text
sweep + frozen_rescore + subset_build + retarget <= budget_seconds
```

**Implementation sketch:**

```python
@dataclass
class PhaseBudget:
    total_s: float
    sweep_s: float = 0.0
    rescore_s: float = 0.0
    subset_s: float = 0.0
    retarget_s: float = 0.0

    @property
    def remaining_s(self) -> float:
        return max(0.0, self.total_s - self.sweep_s - self.rescore_s - self.subset_s - self.retarget_s)
```

**Tests:**

- start generation is deterministic
- non-overlapping coverage math is correct
- partial-sweep target picks the largest feasible prefix/set
- no condition can overspend the declared budget

---

## Task 3: Build the Phase 0 Benchmark Harness

**Files:**

- Modify: `experiments/18_throughput_advantage/bench_throughput.py`
- Test: `experiments/18_throughput_advantage/test_exp18.py`

**Goal:** Make Phase 0 the first expensive, scientifically meaningful
gate.

**Phase 0 responsibilities:**

### Part 1: Throughput screen

For both `SP8192` and `SP16384`, benchmark candidate batch sizes on a
single GPU.

Report:

- wall time per step
- tokens/sec
- peak VRAM
- projected unique-coverage in the sweep budget
- projected sweep time for the intended target

### Part 2: LR stability screen

At the largest feasible batch for each tokenizer, run a short training
screen with:

- linear-scaled LR
- sqrt-scaled LR
- fixed LR

**Failure criteria:**

- NaN
- obvious divergence
- loss flatlines badly enough that the regime is not credible

### 8xH100 execution policy

Use the pod to parallelize **single-GPU** benchmark jobs, not to do
distributed training.

Example scheduling model:

```text
GPU0-3: SP8192 batch sweep
GPU4-7: SP16384 batch sweep
then:
run LR screens for each tokenizer at its candidate large batch
```

**Deliverable:** a Phase 0 summary JSON that names:

- winning tokenizer
- winning large batch
- validated LR regime
- projected sweep coverage
- projected rescore tax
- whether the run is in the low-coverage reconsideration regime

---

## Task 4: Implement Tokenizer Selection Logic

**Files:**

- Modify: `experiments/18_throughput_advantage/run_exp18.py`
- Modify: `experiments/18_throughput_advantage/runner_exp18.py`
- Test: `experiments/18_throughput_advantage/test_exp18.py`

**Goal:** Convert the Phase 0 results into a single tokenizer choice for
the first expensive Phase A pass.

**Selection function:**

```python
def choose_tokenizer(phase0_results: list[dict]) -> str:
    # 1. filter to stable tokenizer/LR candidates
    # 2. maximize projected useful coverage within sweep budget
    # 3. if nearly tied, prefer SP16384 on prior quality grounds
```

**Output should explain why the tokenizer was chosen:**

- coverage delta
- batch delta
- LR stability result
- whether the tie-break rule was invoked

**Test cases:**

- one tokenizer unstable -> choose the stable one
- one tokenizer materially better on coverage -> choose it
- nearly tied throughput -> prefer `SP16384`

---

## Task 5: Implement Phase A Core Matrix

**Files:**

- Modify: `experiments/18_throughput_advantage/runner_exp18.py`
- Modify: `experiments/18_throughput_advantage/run_exp18.py`
- Test: `experiments/18_throughput_advantage/test_exp18.py`

**Goal:** Run the smallest expensive matrix that answers the highest-value
scientific questions.

### First expensive matrix

| Condition | Description |
|---|---|
| `baseline_b32` | Existing baseline, random sampling, 600s |
| `sweep_only` | Large-batch sweep using the max validated setting; if time remains, continue on the same sweep data rather than idling |
| `sweep_target_top10` | Sweep + frozen rescore + retarget on top 10% hardest windows |
| `sweep_random_retrain` | Same sweep and rescore overhead, but retarget a random 10% subset |

### Condition policy

- Use the tokenizer selected in Phase 0
- Use the largest feasible validated batch for the sweep
- Use the largest feasible partial sweep if full sweep is not possible
- Keep total wall clock fixed at 600s for all conditions

### Fairness note

`sweep_only` correctly gets the full 600s for training. The targeting
conditions must therefore beat a control that does **not** pay the
rescore/selection overhead. That is intentional: if targeting helps, it
must help enough to overcome its own tax.

**Implementation detail:**

Represent conditions as explicit config dicts rather than scattered
if/else logic:

```python
CONDITIONS = {
    "baseline_b32": {...},
    "sweep_only": {...},
    "sweep_target_top10": {...},
    "sweep_random_retrain": {...},
}
```

---

## Task 6: Implement Frozen Rescoring and Hardness Selection

**Files:**

- Modify: `experiments/18_throughput_advantage/runner_exp18.py`
- Test: `experiments/18_throughput_advantage/test_exp18.py`

**Goal:** Measure "still finds hard" in the cleanest possible way.

### Rule

Score windows **after** the sweep using the **frozen end-of-sweep
model**.

### Primary hardness metric

- per-window mean token cross-entropy

### Why

This avoids time-bias:

- online sweep loss is contaminated by model age
- a frozen post-sweep pass scores all windows under the same model

**Implementation sketch:**

```python
per_tok = F.cross_entropy(
    logits.reshape(-1, vocab_size),
    targets.reshape(-1),
    reduction="none",
).reshape(batch, seq_len)

per_window = per_tok.mean(dim=1)
```

### Selection outputs

- ranked `(start, loss)` pairs
- top 10% subset
- matched-size random subset

### Tests

- ranking is deterministic
- top10 selects the highest-loss windows
- random control is matched-size but not loss-ranked

---

## Task 7: Add Summaries and Expansion Gate

**Files:**

- Modify: `experiments/18_throughput_advantage/run_exp18.py`

**Goal:** Produce outputs that make the cheap/expensive boundary and the
core scientific result obvious.

### Summary requirements

For every result:

- tokenizer used
- large sweep batch used
- validated LR regime
- sweep coverage achieved
- phase timings
- explicit rescore tax:
  - `rescore_s`
  - `rescore_frac_of_budget`
  - `selection_overhead_s = rescore_s + subset_build_s`
- retarget subset size
- final validation bpb

For the aggregate summary:

- rank conditions by mean bpb
- report paired deltas:
  - `sweep_only - baseline_b32`
  - `sweep_target_top10 - sweep_only`
  - `sweep_target_top10 - sweep_random_retrain`
- print the mean rescore tax for each targeting condition so it is
  obvious how much wall-clock budget was spent on selection rather than
  gradient updates

### Expansion gate

Only add `top25` and `top5` if:

- `sweep_target_top10` beats `sweep_only`, and
- `sweep_target_top10` beats `sweep_random_retrain`, and
- the per-seed direction is consistent enough to justify more pod spend

This should be encoded as an explicit recommendation in the summary.

---

## Verification Checklist

### Cheap preflight verification

```bash
PYTHONPATH=src .venv/bin/python -m pytest experiments/18_throughput_advantage/test_exp18.py -q
PYTHONPATH=src .venv/bin/python experiments/18_throughput_advantage/bench_throughput.py --smoke
PYTHONPATH=src .venv/bin/python experiments/18_throughput_advantage/runner_exp18.py --smoke
```

### First expensive verification

```bash
PYTHONPATH=src .venv/bin/python experiments/18_throughput_advantage/bench_throughput.py \
    --data-path /path/to/fineweb_sp_data \
    --sp8192-model /path/to/sp8192.model \
    --sp16384-model /path/to/sp16384.model \
    --num-gpus 8
```

### First expensive Phase A pass

```bash
PYTHONPATH=src .venv/bin/python experiments/18_throughput_advantage/run_exp18.py \
    --data-path /path/to/fineweb_sp_data \
    --phase0-summary experiments/18_throughput_advantage/results/phase0_summary.json \
    --budget 600 \
    --num-gpus 8
```

---

## Success Criteria

The implementation is complete when:

1. Preflight tests prove the experiment mechanics are correct.
2. Phase 0 cleanly identifies:
   - tokenizer
   - max feasible sweep batch
   - stable LR regime
3. Phase A can run the 4-condition expensive core matrix with strict
   wall-clock accounting.
4. The output clearly separates:
   - cheap software-validation evidence
   - expensive H100-backed scientific evidence
