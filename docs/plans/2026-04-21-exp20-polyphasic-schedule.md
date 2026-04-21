# Exp 20 Take 2 — Polyphasic Schedule for Eval-Time TTT

**Goal:** Resurrect the unused Exp 12 polyphasic scheduler in the *eval-time* regime only. Turn "which parameters to adapt" and "when to adapt" into one unified rotation schedule over the full-val chunk stream, bound by the 600s eval budget.

**Status:** Design (2026-04-21). Complements `docs/plans/2026-04-21-exp20-take2-full-val.md` as a refinement of its Axis B (schedule). Training-time polyphasic partitioning is explicitly out of scope for this plan.

## Why this mapping works

Exp 12's original design partitioned model slices across GPUs and rotated which were "awake" vs "asleep" each step. That never ran because the bare-SSM pivot (Exp 14) removed the typed memory slots the scheme was organized around.

But Exp 20 Take 2 has the same structural question in a different regime. The Param Golf eval budget forces us to adapt on ≤ 0.3% of chunks — every chunk is a wake-or-sleep decision. The take-2 schedule axis already handles *when any adaptation happens*. Polyphasic extends that to **per-group wake/sleep with independent rotations**: at each chunk, each parameter group (log_a, delta_proj, B_side, C_side, lm_head) is individually awake or asleep.

The generalization is clean:

|Exp 12 (paper)|Exp 20 take 2 polyphasic (this plan)|
|---|---|
|Partition = GPU slice|Partition = parameter group|
|Awake = processes live batch|Awake = receives gradient update this chunk|
|Asleep = consolidates|Asleep = frozen this chunk (state still flows)|
|Wake fraction K/N|Wake fraction k(g) per group, independent|
|Rotation over steps|Rotation over chunks|
|Multi-GPU hardware|Single eval stream, no hardware implication|

The consolidation work in Exp 12's design (N2 utility scoring, REM dreaming) is explicitly *not* ported. Exp 11 already killed those at 600s budget. A sleeping group in take-2-polyphasic does nothing — it just doesn't update. Its computation (forward pass, state propagation) still happens because the stream requires it for scoring.

## What the scheduler decides

At each chunk `t`, for each parameter group `g`, the scheduler returns a boolean `awake(g, t)`. If awake, the current chunk's gradient updates `g`. If asleep, the chunk is scored under frozen `g` and nothing else happens for that group.

**Hard constraint:** `Σ_t Σ_g awake(g, t) · step_cost(g) ≤ usable_ttt_budget`.

## Schedule primitives

Three families, composable:

1. **Uniform rotation** — each group has a wake period `T(g)` and phase offset `φ(g)`. Awake when `(t − φ(g)) mod T(g) == 0`. Independent per group. Budget is deterministic.

2. **High-loss triggered** — a group wakes only when the current chunk's bpb exceeds a per-group threshold `θ(g)`. Budget depends on the data; enforce a hard per-group cap so no single group can exhaust the budget alone.

3. **State-stability gated** — when state_norm exceeds a threshold, wake the memory-horizon groups (log_a, delta_proj) to let the recurrence reconfigure. When state is stable, wake the read/write groups (B_side, C_side, lm_head) since the memory is settled and only surface-adaptation helps.

These are primitives. A schedule is a rule that combines them, e.g. "fast rotation on delta_proj (T=256) + high-loss trigger on lm_head (θ=1.8 bpb) + state-gated wake on log_a." One schedule = one cell in the sweep.

## How this composes with take-2 Phase 2

Take-2's Axis B is `{every_Kth, doc_boundary, high_loss_gate}` — one global adaptation signal per chunk. Polyphasic is a strict superset: a global schedule is the special case where all groups share one wake rule.

Concrete replacement in Phase 2:

- take-2 cell "`delta_proj, every_512th, steps=1, lr=0.004`" → polyphasic "`T(delta_proj)=512, everything else asleep`". Identical behavior.
- take-2 cell "`log_a+delta_proj, every_2048th, steps=4, lr=0.008`" → polyphasic "`T(log_a)=2048 φ=0, T(delta_proj)=2048 φ=1024, steps=4 each wake`". Same budget, but the two groups never adapt on the *same* chunk — removes gradient interference between them.

The second example is the testable claim: **anti-phase rotation > same-chunk simultaneous adaptation** for a pair of groups that influence the same recurrence. Exp 11's n2_n3/full_cycle losses pointed at interference between mechanisms running under one budget; polyphasic is the eval-time analog of the "don't stack mechanisms on the same step" lesson.

## Phase 2 polyphasic cells to add

After take-2 Phase 2's uniform-schedule cells complete, add these as a refinement (~6 cells, not a full sweep):

1. `log_a T=2048 φ=0, delta_proj T=2048 φ=1024, steps=1, lr=0.008` — anti-phase rotation, matched budget against take-2's combined cell
2. `log_a T=2048 φ=0, delta_proj T=2048 φ=0, steps=1, lr=0.008` — same-phase baseline, same budget, tests the interference claim directly
3. `delta_proj T=256 φ=0, lm_head high_loss_gate θ=1.8, log_a state_gated ‖h‖>τ` — triple-primitive cell
4. `delta_proj high_loss_gate θ=1.6, rest asleep` — pure high-loss triggering on the best single group from take-2's uniform cells
5. Winner-of-above + Δ-modulation from Phase 3 — combined stack
6. Winner-of-above with `carry_state` (only if Phase 0 resolved the carry regression)

## What polyphasic does NOT add

- **No training-time scheduling.** This plan is eval-only. Training-time partition rotation across GPUs is a separate question and stays parked.
- **No N2 utility scoring during sleep.** Exp 11 killed that at 600s; it stays killed.
- **No REM-style replay or dreaming.** Same reason. A sleeping group is simply frozen.
- **No memory-slot consolidation.** The typed memory subsystem is gone; there's nothing to consolidate.
- **No hemispheric metaphor beyond the schedule.** The biological framing was useful for design intuition; the implementation is plain per-group boolean gating.

## Files

- `src/chaoscontrol/eval_stream/polyphasic.py` — `PolyphasicSchedule` class composing the three primitives per group
- Extension to `src/chaoscontrol/eval_stream/schedule.py` (from take-2 plan) — take the polyphasic schedule as a drop-in
- `tests/test_eval_stream_polyphasic.py` — budget-bound enforcement, anti-phase correctness, high-loss trigger math, state-gated trigger math

## Kill criteria

- Anti-phase vs same-phase (cells 1 vs 2) shows no bpb difference at matched budget → per-group rotation is not doing useful work; drop polyphasic complexity, keep global schedules only.
- Triple-primitive cell (3) loses to best single-primitive cell from take-2 → composition isn't helping; keep things simple.
- Any polyphasic cell beats the best take-2 uniform-schedule cell by > 2σ → polyphasic earns its place and cells 4–6 get funded.

## One-line thesis test

Does separating "when *any* group adapts" from "*which* group adapts *this* chunk" produce better bpb-per-TTT-second than a single global schedule, at the full 50k eval budget?
