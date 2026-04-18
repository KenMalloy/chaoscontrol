# Exp 20 — SSM-Native Test-Time Training Ablation

**Date:** 2026-04-17
**Status:** Design — approved
**Target:** Parameter Golf (FineWeb eval, 50k-doc accumulated stream, 600 s eval budget, Issue #1017 score-before-update)
**Sequencing:** Runs after Exp 19 ships a base checkpoint; precedes Exp 21.

## Motivation

Exp 18/19 closes the training-time underfit gap. Exp 20 answers the complementary question: *how far can an SSM close the remaining bpb gap at eval time, using only SSM-native mechanisms, within Parameter Golf's 600 s eval budget and Issue #1017's score-before-update rule.*

Every top Parameter Golf PR uses Legal TTT (per-window SGD TTT #1655, Multi-Phase Global SGD TTT #1670, etc.). No SSM has broken 1.22 bpb; the frontier is transformer-heavy. The bet: TTT designed around SSM architecture — not ported from transformer recipes — has unexplored slots that compose. Differentiated by construction; no prior art to compete against.

## Hypothesis

A per-doc SSM-native TTT stack can close materially more eval-time bpb than a transformer-ported TTT stack at equal compute, by exploiting slots that do not exist in transformers:

1. Hidden-state carry across doc boundaries (zero weights updated, Legal TTT-compliant by construction).
2. Adaptation of the A-matrix dynamics (`log_a`, 256 params/block; `delta_proj`, 65 K/block) — both unique to selective SSMs.
3. Eval-time Δ-modulation (scalar rescaling of `delta_proj(x)` or `log_a`) — a memory-horizon knob with zero parameters changed.

**Primary:** a stacked SSM-native configuration beats an architecture-agnostic baseline (`lm_head`-only TTT) at equal compute by more than seed noise (target ≥ 0.025 bpb at n = 5).

**Secondary:** each SSM-native axis contributes non-zero marginal bpb alone, establishing that the stack is not carried by a single component.

## Scope

**In scope:**
- 50k-doc replay harness with Issue #1017 enforcement.
- Three-axis ablation: what-adapts × what-persists × Δ-modulation.
- Screen + compose + schedule-tune on an Exp 19 base checkpoint.
- bpb-vs-compute Pareto curves, not a single bpb number.

**Out of scope:**
- Submission-repo assembly, final 8×H100 submission runs (separate repo).
- Weight-update TTT on the SGNS embedding table from Exp 21 (Exp 21 runs after Exp 20; the `embed_rows` axis in Exp 20 is measured against the random-init base and can be re-run post-Exp-21).
- Multi-timescale SSM stacks (architecture change, not eval-time mechanism — parked as future experiment).
- Nonparametric kNN-LM caches over residual-stream features (orthogonal to weight-TTT; backlogged).

## Metric and success criteria

**Primary metric:** accumulated bpb on the 50k-doc stream, measured against a fixed Exp 19 base checkpoint.

**Secondary metrics (logged per run):**
- **Accumulation curve:** bpb vs. docs-seen at {1, 10, 100, 1K, 10K, 50K}. Diagnoses early vs long-horizon gains.
- **Compute-per-doc:** gradient steps × FLOPs/step. Required for Pareto extrapolation from our 8-GPU ablation pod to Parameter Golf's 8×H100 submission.
- **Stability:** per-doc loss drift, ‖grad‖, state-norm. Catches adaptation-collapse at doc 10K-30K before a single final bpb hides it.

**Budget portability.** Rather than matching a single 600 s wall-clock at our compute, the harness is parameterized on `(budget_seconds, grad_steps_per_chunk, tokens_per_step)`. Output is a **bpb-vs-compute Pareto curve**; winning configs are the ones whose gain extrapolates linearly under budget scaling. Non-linear extrapolators are rejected — a gain that saturates at low compute won't cash in under the real submission budget.

**Success tiers:**
- **Floor:** any (adapt, persist, Δ-mod) triple beats the full-reset + no-TTT baseline by > 0.025 bpb at n = 5 seeds.
- **Competitive:** best configuration beats the no-TTT baseline by 0.1-0.2 bpb — enough to plausibly close the remaining gap to leaderboard SOTA when stacked with Exp 19 training gains.
- **Paper-worthy:** a purely SSM-native mechanism (`log_a`-only, Δ-modulation, or state-carry alone) contributes ≥ 0.025 bpb without weight updates. Validates the SSM-native thesis independent of absolute placement.

## Architecture — the eval-stream harness

Five components, each with a clean contract.

```
DocStreamer        Score-then-Adapt     TTT Inner Loop     Δ-Modulator      MetricsCollector
  ┌─────┐            ┌──────────┐         ┌──────────┐       ┌────────┐         ┌──────────┐
  │iter │──docs───▶│chunk-level │──loss─▶│grad step on  │      │scale Δ │       │per-doc bpb│
  │50 K │            │pre-update  │         │param group   │       │online  │       │curve      │
  │FW   │            │snapshot    │         │(log_a, etc.) │       │no grad │       │compute    │
  │eval │            │+ score it  │         └──────────┘       └────────┘         │stability  │
  └─────┘            └──────────┘                                                  └──────────┘
```

### Component contracts

1. **DocStreamer.** Iterates FineWeb eval docs in fixed order; doc-level disjoint from Exp 19's train stream. Emits `(doc_tokens, doc_id)`. Reuses `src/chaoscontrol/data.py` at doc granularity.

2. **Score-then-Adapt controller.** Enforces Issue #1017 structurally, not by flag. For each **chunk** within each doc:
   1. Snapshot weight state (reference, not copy).
   2. Score the chunk forward-only under the snapshot; collect per-token loss.
   3. Optionally roll the SSM recurrence state forward from the prior chunk / prior doc (Axis 2).
   4. With gradients enabled, invoke the TTT inner loop on the chunk already scored.

   Chunk size is a schedule parameter. Spectrum:
   - `whole_doc` — one adapt step per doc (coarsest legal).
   - `{64, 256, 1024}` tokens — standard Legal TTT, matches top-PR practice.
   - `1` token — extreme per-token TTT (included as envelope, expected prohibitive).
   - `per-position state only` — SSM recurrence, implicitly per-token, no weight update, free.

   Doc boundary and adapt boundary are independent: doc boundary triggers Axis 2 persistence decisions; adapt boundary (every chunk) triggers Axis 1 weight updates.

3. **TTT Inner Loop.** Parameterized by:
   - **Adapt set** (Axis 1, below) — param-group filter by name.
   - **Persistence mode** (Axis 2, below).
   - **Optimizer** — Muon, matching base model. `persistent_muon_moments` across docs is a boolean sub-knob.
   - **Schedule** — `(chunk_size, steps_per_chunk, eval_lr, warmup)`. Fixed per run; cross-run sweep in Phase F.

4. **Δ-Modulator (Axis 3).** Forward-hook-based scalar knobs, zero weight update:
   - `delta_scale`: multiplicative on `delta_proj(x)` output (lengthens/shortens memory uniformly).
   - `log_a_shift`: additive pre-sigmoid on `log_a` (rescales base decay).
   - Reverted between runs to keep the base checkpoint immutable.

5. **MetricsCollector.** Flat JSONL/parquet log: `(doc_id, bpb, tokens, adapt_loss_before, adapt_loss_after, step_count, wall_ms, ||grad||, state_norm)`. Stability gate fires in-run: if per-doc loss diverges by > N seed-SDs for K consecutive docs, the run is tagged `collapsed` but completes for post-mortem.

### Repo placement

First-class submission-day infrastructure, not bench scaffolding.

- `src/chaoscontrol/eval_stream/` — module (`DocStreamer`, `LegalityController`, `TTTRunner`, `DeltaModulator`, `MetricsCollector`).
- `scripts/run_exp20_eval.py` — driver; loads a checkpoint, wires a config, writes results.
- `tests/test_eval_stream.py` — contract tests. Non-negotiable:
  - Score-before-update invariant (test that a forced leak is *detected*, not just absent).
  - Doc-1 bpb with `reset + no-TTT` matches existing forward-only eval bit-identically.
  - `carry_state` produces different per-doc loss than `reset`.
  - `delta_scale=1.0` produces identical output to `delta_mod_none`.

Sibling to `src/chaoscontrol/runner.py`, not a subclass. `runner.py` trains on packed batches — wrong shape for doc-stream semantics with per-doc state threading. Separation prevents accidental training-path contamination during eval.

## The three axes

### Axis 1 — What adapts at eval time

Reference: `src/chaoscontrol/core.py:232-361` (ChaosSSMCore, `a_mode="diag"`, Exp 18 Test 4b config). Our SSM is S6/Mamba-style selective; the per-token recurrence is

```
candidate = tanh(in_proj(x))                              # B side
update    = sigmoid(select_proj(x)) * candidate
Δ         = softplus(delta_proj(x))                       # A side (input-modulated step)
decay     = exp(-Δ · sigmoid(log_a))                      # A side (base decay rate)
state     = decay * state + update
y         = out_proj(sigmoid(gate_proj(x)) * state)       # C side
```

| Tag | Adapt set | Params (d=256, V=8192, 4 blocks) | SSM-native | Notes |
|---|---|---|---|---|
| `none` | — | 0 | baseline | forward-only; anchors Axis 2/3 measurement |
| `log_a` | per-block `log_a` | ≈ 1 K total | **yes** | retunes memory decay uniformly — smallest SSM-native slot |
| `delta_proj` | per-block `delta_proj` | ≈ 260 K | **yes** | retunes input-dependent selectivity |
| `log_a+delta_proj` | both | ≈ 261 K | **yes** | joint memory + selectivity tune |
| `B_side` | `in_proj + select_proj` | ≈ 524 K | **yes** | input → state path |
| `C_side` | `out_proj + gate_proj` | ≈ 524 K | **yes** | state → residual path |
| `embed_rows_seen` | rows of `model.embed.weight` hit in current doc | sparse, ≤ 2.1 M | architecture-agnostic | sparse grad on embedding |
| `lm_head` | `lm_head.weight` | 2.1 M | architecture-agnostic | baseline anchor — matches top-PR output-head-only TTT |
| `lora_r8` | LoRA r=8 on all projections | ≈ 82 K | neutral | build-on-demand; only if mid-axis winners don't explain the effect |
| `all` | everything | ≈ 10.7 M | — | envelope only, expected unstable |

`log_a` is the architectural heart: 256 scalars per block control how long state persists. `delta_proj` is the selectivity mechanism — without it the SSM degenerates to an LTI filter with token-agnostic decay.

### Axis 2 — What persists across doc boundaries

| Tag | State | Weight deltas | State-init | Notes |
|---|---|---|---|---|
| `reset` | — | — | random/fixed | baseline; forces clean measurement of accumulation gains |
| `carry_state` | yes | — | n/a | **zero-cost SSM-native** — do not zero the state at doc boundary |
| `carry_weights` | — (zero each doc) | yes | random/fixed | standard Legal TTT; matches top-PR behavior |
| `carry_both` | yes | yes | n/a | composition; expected stack winner |
| `trainable_h0` | — | yes (incl. `h₀`) | **adapted online** — 256 floats treated as TTT parameter | eval-time version of "trainable state-init"; no Exp 19 changes required |
| `trainable_h0+carry` | resumes from last-doc final state | yes | adapted online | full stack |

`trainable_h0` is the SSM-native version of cross-doc memory without dragging pre-training into it — we TTT `h₀` like any other small-param slot.

### Axis 3 — Eval-time Δ modulation (no-grad)

| Tag | Knob | Range |
|---|---|---|
| `delta_mod_none` | baseline | — |
| `delta_scale` | scalar × `delta_proj(x)` output | log-sweep {0.25, 0.5, 1.0, 2.0, 4.0} |
| `log_a_shift` | pre-sigmoid additive on `log_a` | {-1, -0.5, 0, +0.5, +1} |
| `both` | combined | 2-D grid at winners only |
| `online_delta_mod` | scalar adapted per-doc from doc length / embedding stats | stretch goal; needs a controller |

Pure inference mechanism. If it buys even 0.02 bpb alone, it's strictly dominant — free, legal, no weight update.

### Schedule sub-axes (tuned after winners, not a full axis)

`chunk_size ∈ {64, 256, 1024, whole_doc}` · `steps_per_chunk ∈ {1, 2, 5}` · `eval_lr` log-swept around 0.064 (Exp 18 Test 5b training optimum, Muon carries over) · `persistent_muon_moments` boolean.

## Execution plan

Main-effects-first, interactions-second. The full cross-product (8 × 6 × 3 × schedule sub-axes ≈ 10 K cells) is not the plan; sequenced screens catch the main effects cheaply and compose.

| Phase | What | Configs | Hypothesis |
|---|---|---|---|
| **A** | Harness smoke | `none × reset × none` vs existing forward-only eval | Scoring path bit-exact. Non-negotiable. |
| **B** | Axis 2 alone (no weight TTT) | `none × {reset, carry_state} × none`, 3 seeds | H1: SSM-native thesis — does state carry buy bpb for free? |
| **C** | Axis 1 screen at Axis 2 winner | top 5 adapt-sets × winner_B × none, 3 seeds | H2-H4: which param group responds best per $? |
| **D** | Axis 3 alone | `none × winner_B × {delta_scale sweep, log_a_shift sweep}`, 3 seeds | H3: Δ-mod as zero-cost knob |
| **E** | Compose | top-2 from C × winner_B × winner_D, 5 seeds | H5: stack beats vanilla Legal TTT at equal compute |
| **F** | Schedule tune at Phase E winner | chunk_size × steps × LR Pareto, 3 seeds | Pareto on bpb-vs-compute |
| **G** | Envelope sanity | `lm_head`, `lora_r8` at winner schedule, 5 seeds | Does the SSM-native stack match or beat architecture-agnostic TTT? |

**Compute at 8 GPUs (seed-parallel ablation):**

| Phase | Runs | 8-GPU wall-clock |
|---|---|---|
| A | ~5 | < 20 min |
| B | 30 | ~40 min |
| C | 25 | ~40 min |
| D | 75 | ~100 min |
| E | 50 | ~65 min |
| F | 60 | ~80 min |
| G | 10 | ~15 min |

**Total ≈ 6-7 hours of 8-GPU wall-clock.** Build (harness + tests) is the bottleneck, not runs.

**Execution mode.** Ablation phases (A-E) use seed-parallel launches across the 8 ranks with no inter-rank communication — each rank is an independent complete doc stream at a different seed. Phase G includes 4- or 8-way DDP runs (reusing Exp 18's manual all-reduce) for `lora_r8` and `all`, validating that per-chunk TTT scales cleanly under data-parallel gradient steps.

**Calendar.** Sequenced after Exp 19 per locked experiment ordering:

| Day (post-Exp-19-base) | Track |
|---|---|
| 1-2 | Build harness. Contract tests first (legality invariant, bit-exact smoke). |
| 3 | Phase A + B + D on Exp 19 base (one 8-GPU day). |
| 4 | Phase C + E. |
| 5 | Phase F schedule Pareto. |
| 6 | Phase G envelope + `lora_r8` build. |
| 7-8 | Writeup, submission-repo handoff. Buffer for numerical debug + re-seeds. |

## Risks and mitigations

1. **Adaptation collapse at long horizons.** Per-doc loss diverges at doc 10K-30K as accumulated TTT drifts out of training distribution. *Mitigation:* ≥ 100-step stability gate; collapse is a scientific finding, not a bug.
2. **Chunk-level legality bugs.** Easy to fake a great bpb with a score-before-update violation. *Mitigation:* contract test deliberately attempts a leak and asserts the harness flags it. Non-negotiable.
3. **Scan numerics under weight drift.** `exp(-Δ · sigmoid(log_a))` can blow up for long chunks after weight adaptation. *Mitigation:* per-step state-norm monitor, clip if needed; fp32 fallback on the scan core if bf16 drifts. Aligns with the bf16 gotchas memory.
4. **8-GPU ablation → 8×H100 submission gap.** Small (same DDP world size, same per-rank math), but non-zero. *Mitigation:* Phase F/G include DDP-mode validation runs; Pareto framing already rejects gain-saturating configs.
5. **Compute blow-up from per-chunk TTT.** `all` or `lm_head` at chunk_size=64, steps=5 × 50 K docs = millions of grad steps per run. *Mitigation:* budget gate at setup — abort if projected wall-clock exceeds N hours.
6. **Persistence × scan-kernel interaction.** `carry_state + carry_weights` may interact with the scan kernel's zero-initial-condition assumptions. *Mitigation:* Phase B isolates state-carry alone; if numerically clean, weight-carry on top is a smaller perturbation.

## Kill criteria

- **Phase A:** harness bpb must match forward-only eval within float noise. Fix before continuing.
- **Phase B:** if `carry_state` fails to beat `reset` by seed noise, the SSM-native thesis is in trouble at eval time — write up and continue to C (weight TTT is a separate lever).
- **Phase C:** if no Axis 1 slot beats `lm_head`, downgrade the paper claim from "SSM-native wins" to "SSM-native is competitive at lower param cost." Continue.
- **Phase E:** if the stack fails to beat its best individual component, main effects aren't composing — run one interaction probe (winner + second-winner at fixed schedule) before declaring main-effects-only.
- **Phase F:** Pareto winner must extrapolate monotonically. If gain shrinks as compute scales, pick the more robust runner-up.
- **Phase G:** if `lora_r8` or `all` beats the SSM-native stack at matched compute, the SSM-native thesis is falsified for TTT specifically (not for training). Honest writeup.

## Dependencies

- **Exp 19 base checkpoint.** Exp 20 runs against the final Exp 19 checkpoint. Harness build (Days 1-2) can start before Exp 19 completes since it is pure code.
- **Exp 18 Test 4b checkpoint** (already available) — used only for harness smoke validation before Exp 19's base ships, not for primary results.
- **No Exp 21 dependency.** `embed_rows` TTT is measured against random-init embeddings. Post-Exp-21 re-run on SGNS init is a backlog item.
- **No new third-party infrastructure.** Reuses existing Muon, manual DDP all-reduce, chunked LM backward, ChaosStudentLM/ChaosSSMCore. `lora_r8` is the only net-new modeling surface and is gated (build only if warranted by C).

## Deferred

- SGNS-init `embed_rows` TTT — after Exp 21 completes.
- Multi-timescale SSM stacks — architecture experiment, not eval-time.
- Nonparametric kNN-LM cache over residual-stream features — orthogonal; if Phases A-G under-perform expectations, revisit as a Phase H.
- `online_delta_mod` controller (per-doc Δ from doc statistics) — stretch; included only if simpler Δ-modulation succeeds.

## Open questions for redline

- Doc-ordering sub-sweep (random / shortest-first / difficulty-ordered) — technically free, but paired-seed statistics require a locked ordering per run. Proposal: pin one canonical ordering for the main ablation, run a 3-ordering sanity sweep at Phase E winner to confirm rank-stability.
- Stability-gate thresholds (N seed-SDs, K consecutive docs) — pilot first 5 Phase B runs, tune from observed variance.
