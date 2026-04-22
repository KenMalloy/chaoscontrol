# Exp 24 Phase 0: Base-Lock Tuning Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Tune the `fast_slow + scheduled dreamworld` stack on a 3-rung ladder (DW sweep → FS sweep around DW winner → top-2 × 3-seed confirm) and commit the winning config as the Exp 24 base for all downstream arms. Same pod run, immediately after `exp24_base.yaml` is written, execute Phase 0b: entropy-gated `log_a` reread evaluated on the locked checkpoint. Phase 0b may promote an eval-time arm, but it is sealed against base selection — its results cannot rewrite `exp24_base.yaml`.

**Architecture:** Phase 0 runs as a data-collection plan, not a mechanism invention plan. Each tuning rung is a new matrix builder in `experiments/24_training_time_bundle/exp24.py`, dispatched from `run_exp24.py`. Screening rungs (2 & 3) use seed=1337 single-seed; the confirm rung runs top-2 × 3 seeds with full-val. Winner is locked by mean BPB with run-to-run stability as tiebreaker. Phase 0b is deliberately downstream: it uses the locked checkpoint/config, existing `DeltaModulator(log_a_shift=...)` semantics, and a causal gate based on model confidence. Its result can promote a future eval-time arm, but it cannot rewrite `exp24_base.yaml`.

**Out of scope for base selection (but in scope for same-pod execution):** `event_sleep`, predictive aux, spectral, ScOpt — none of these run in this plan. The rigor+speed plan for `fast_slow_dreamworld_event_sleep` (`2026-04-22-exp24-rigor-and-speed-implementation.md`) stays frozen until Phase 0 lands, because its test anchors and profile harness bake in FS+DW defaults that Phase 0 is about to choose. Entropy-gated reread (Phase 0b) runs same-pod after base lock, on the locked checkpoint — in scope for execution, sealed from base selection.

**Tech Stack:** Python, existing `run_exp24.py` / `exp24.py` matrix dispatch, **4×H100 pod** (cost choice, not technical requirement — see Budget section for ranking-transfer tradeoff vs ws=8 submission regime), 600s train budget, full-val eval (~164s on 4x per memory), FineWeb data already staged on the pod volume.

---

## Anchor (reference config)

The anchor below is the starting guess the ladder tunes around. It is included as one point in Task 1's DW sweep (the `c8_i8 × w=0.25` cell).

- `fast_slow_enabled=True`, `fast_slow_interval=32`, `fast_slow_alpha=0.50`, `fast_slow_eval_copy="slow"`
- `dreamworld_enabled=True`, `dreamworld_cache_interval=8`, `dreamworld_interval=8`, `dreamworld_weight=0.25`
- `dreamworld_replay_batch_size=128`, `dreamworld_prefix_tokens=128`, `dreamworld_replay_tokens=64`
- `dreamworld_buffer_size=16`, `dreamworld_min_size=2`, `dreamworld_max_age_steps=256`

All other knobs track the existing base config (`experiments/23_fast_path/configs/base_seq_epoch_lr0064_full_corpus.yaml`): 4-layer SSM, dim=256, bf16, LR=0.064, bs=1024 × 8 ranks, seq=512, Muon, sequential-epoch sampling.

---

## Phase 0b thesis addendum: entropy-gated `log_a` reread

Phase 0's base claim is training-time: `fast_slow + scheduled dreamworld` produces the best weights/checkpoint. Phase 0b's separate claim is eval-time: an SSM can spend opt-in compute by rereading a recent suffix with a different hidden-state decay rate. This is the native SSM axis that Exp 22 and Exp 24 both point toward: `log_a` is a traversal knob, not a fixed constant.

**Engagement with Exp 20 TTT null (2026-04-20).** The 128-doc TTT pilot showed no cell beat the reset floor at steps=1, and the cause was not compute-wall. That result shelved generic TTT for the current base. Phase 0b is not a rerun of Exp 20 with a new knob; its sharper hypothesis is that the gate must be SSM-native (hidden-state confusion drives hidden-state decay-rate change) rather than generic param-update TTT. If entropy-gated reread lands within noise of the score-only floor, that is consistent with Exp 20, not independent evidence, and Phase 0b shelves to join generic TTT on the "not for this base" pile until we have a new reason to revisit.

The first Phase 0b version is intentionally small, but these four pieces are not optional:

- **The model picks when to shift.** Trigger on the model's own predictive entropy at step `t`, measured from the softmax over the base-pass logits before token `t` is scored. A percentile threshold (e.g. fire on steps whose entropy is in the top 10% of a rolling window) defines "high entropy." No run-schedule component.
- **The second reading blends into the pre-score state.** Replacement erases the base pass; concatenation changes the compute/artifact shape. **Primary path:** compute a soft mix `state_t = (1 - beta) * state_base + beta * state_reread` *before* scoring token `t`, so the reread influences the scored prediction. `beta` is fixed (pilot: 0.5) or severity-driven from the pre-score entropy signal. This is the promoted mechanism.
- **Direction is target-free at the primary path.** Because the primary path blends before scoring, the direction signal cannot use target-`t` loss — the target is still unrevealed. Pick direction from a target-free candidate signal: reread-vs-base state divergence, predictive-entropy delta across candidate rereads, or agreement across `log_a_shift` sign candidates. A deferred-blend variant (score with base, use target-`t` loss to pick direction for `t+1` state) is allowed as a secondary arm, not the promoted mechanism — its mechanism and control set are called out below.
- **The rewind is short.** The reread is "I am confused now; reread what I just saw differently." Start with a tight suffix (`K` in `{32, 64}`, with `128` as an explicit boundary diagnostic), not long windows that drag in unrelated context.

**Matched-budget controls (procedural pin).** Run entropy-gated first, record its realized fire count `N_fire` and total reread token count over the full eval stream. Then configure each matched control so it sees the same per-stream reread-token budget:

- **Score-only floor** — no reread, `600s` eval budget. This is the apples-to-apples base.
- **Scheduled-reread control** — fire every `stream_length / N_fire` steps with the same `K`, same blend. Matched fire count.
- **Entropy-gated same-horizon control** — same gate, `log_a_shift=0.0`. Isolates the effect of the decay-rate change from the effect of the reread pass itself.
- **Entropy-gated bidirectional control** — candidates are `+log_a_shift` and `-log_a_shift`, direction chosen per-fire from the target-free signal.
- **Fixed-direction shifted reread** — ablation only, not promoted.

All controls use the same `600s` eval budget (per `feedback_train_eval_budget_separation`); reread compute is accounted against eval, never training.

**Preregistered success criteria.** Written before the run, not after:

- **Promote to eval-time arm if:** entropy-gated primary beats score-only floor by `>= 0.015 BPB` mean over 3 seeds AND beats scheduled-reread control by `>= 0.005 BPB` mean over 3 seeds AND seed-to-seed stddev `< 0.01 BPB`. Both thresholds must hold.
- **Shelve and record consistent-with-Exp-20 if:** entropy-gated primary is within `0.005 BPB` of score-only floor (either direction), OR entropy-gated primary beats scheduled-reread by `< 0.002 BPB`. In either case, the mechanism is not doing native-SSM work beyond generic reread compute.
- **Ambiguous / warrants follow-up if:** primary beats score-only floor by `0.005 to 0.015 BPB` but doesn't clear the scheduled-reread gap. Record, do not promote, design a sharper follow-up.
- **Deferred-blend secondary** is evaluated on the same thresholds but against its own score-only floor at matched reread-token budget; it promotes only if primary also promotes (otherwise deferred-blend alone is not enough to claim an eval-time arm).

This addendum is not a fourth base-lock rung — it cannot influence which config lands in `exp24_base.yaml`. But it does run same-pod, immediately after base lock, against the locked checkpoint. Task 9 implements the preregistration doc and the eval runs. Task 10 handles the event_sleep plan rebase.

---

## Task 1: Add `build_phase0_dreamworld_sweep` to `exp24.py`

**Files:**
- Modify: `experiments/24_training_time_bundle/exp24.py` (add builder alongside `build_fastslow_dreamworld_matrix` at line 324)
- Modify: `experiments/24_training_time_bundle/run_exp24.py:138-153` (register matrix name `phase0_dreamworld_sweep`)

**Step 1: Add builder function**

Add after `build_fastslow_dreamworld_matrix`:

```python
def build_phase0_dreamworld_sweep(
    *,
    speed_config: dict[str, Any],
    world_size: int = 4,
    budget_seconds: float = 600.0,
    seed_values: Sequence[int] = (1337,),
) -> list[dict[str, Any]]:
    """Phase 0 rung 1: sweep DW interval × weight with FS pinned at anchor."""
    intervals = [4, 8, 16]
    weights = [0.10, 0.25, 0.50]
    entries: list[dict[str, Any]] = []
    for interval in intervals:
        for weight in weights:
            arm = {
                "name_arm": f"phase0_fs_i32a050_dw_c{interval}i{interval}_w{int(weight*100):03d}",
                "exp24_mechanism": "fast_slow_dreamworld",
                "artifact_impact": ARTIFACT_TRAINING_ONLY,
                "fast_slow_enabled": True,
                "fast_slow_interval": 32,
                "fast_slow_alpha": 0.50,
                "fast_slow_eval_copy": "slow",
                "dreamworld_enabled": True,
                "dreamworld_cache_interval": interval,
                "dreamworld_interval": interval,
                "dreamworld_weight": weight,
                "dreamworld_prefix_tokens": 128,
                "dreamworld_replay_tokens": 64,
                "dreamworld_replay_batch_size": 128,
                "dreamworld_buffer_size": 16,
                "dreamworld_min_size": 2,
                "dreamworld_max_age_steps": 256,
            }
            for seed in seed_values:
                entry = _base_entry(
                    speed_config=speed_config,
                    world_size=world_size,
                    budget_seconds=budget_seconds,
                )
                entry.update(arm)
                name_arm = str(entry.pop("name_arm"))
                entries.append(
                    _named_entry(
                        base=entry,
                        phase="phase0",
                        mechanism=str(entry["exp24_mechanism"]),
                        arm=name_arm,
                        seed=int(seed),
                    )
                )
    return entries
```

**Step 2: Register matrix in `run_exp24.py`**

In `_build_entries` (line ~56), add a branch before the `"all"` fallback:

```python
    if matrix == "phase0_dreamworld_sweep":
        return build_phase0_dreamworld_sweep(
            speed_config=speed_config,
            world_size=world_size,
            budget_seconds=budget_seconds,
            seed_values=seeds,
        )
```

In the argparse `choices=[...]` list at line 145, add `"phase0_dreamworld_sweep"`.

Also add `build_phase0_dreamworld_sweep` to the import list from `exp24` at the top of `run_exp24.py`.

**Step 3: Dry-run to verify 9 entries**

```bash
cd experiments/24_training_time_bundle
python run_exp24.py --matrix phase0_dreamworld_sweep --seeds 1337 --world-size 4 --show
```

Expected: 9 entries named `exp24_phase0_fast_slow_dreamworld_phase0_fs_i32a050_dw_c{4,8,16}i{4,8,16}_w{010,025,050}_s1337`.

**Step 4: Commit**

```bash
git add experiments/24_training_time_bundle/exp24.py experiments/24_training_time_bundle/run_exp24.py
git commit -m "exp24: add phase0 dreamworld interval×weight sweep matrix"
```

---

## Task 2: Launch Phase 0 DW sweep on 4×H100

**Files:**
- Output dir: `experiments/24_training_time_bundle/phase0_dw_sweep_4x_<timestamp>/`

**Step 1: Verify pod state**

```bash
runpodctl get pod --all
```

Ensure a 4×H100 pod is RUNNING with `/workspace/venv` and FineWeb shards on the volume. If stopped, start it and wait for SSH.

**Step 2: Commit current branch and push**

Per project memory (only committed code on pods): no rsync of uncommitted state.

```bash
git status
git push origin main
```

**Step 3: On the pod, launch the sweep**

```bash
source /workspace/venv/bin/activate
cd /workspace/chaoscontrol
git pull
cd experiments/24_training_time_bundle
OUT=phase0_dw_sweep_4x_$(date -u +%Y%m%dT%H%M%SZ)
python run_exp24.py --matrix phase0_dreamworld_sweep --seeds 1337 --world-size 4 --output-dir $OUT
```

Expected: 9 runs × ~785s each (600s train + ~164s full-val + ~20s startup) ≈ 118 min total wall time on 4x (runs execute sequentially, not in parallel — each uses all 4 ranks). Note: `run_exp24.py` has `default_world_size=8` hardcoded for non-`semantic_overhead_gate` matrices; pass `--world-size 4` on every launch to override.

**Step 4: Monitor**

Stream `progress.jsonl` or watch `summary.json` for per-arm BPB. Abort if any arm errors (OOM, NCCL timeout, NaN loss) — investigate before relaunching.

**Step 5: Rsync results back to laptop**

```bash
rsync -av <pod>:/workspace/chaoscontrol/experiments/24_training_time_bundle/$OUT/ \
    ~/Local\ Documents/Developer/chaoscontrol/experiments/24_training_time_bundle/$OUT/
```

**Step 6: Commit results**

```bash
git add experiments/24_training_time_bundle/phase0_dw_sweep_4x_*/
git commit -m "exp24: record phase0 dreamworld sweep results"
```

---

## Task 3: Pick DW winner

**Files:**
- Create: `experiments/24_training_time_bundle/PHASE0_DW_WINNER.md`

**Step 1: Rank the 9 arms by val BPB**

From `summary.json`, extract `val_bpb` per arm. Sort ascending.

**Step 2: Sanity checks**

- Sanity floor at ws=4. The sweep has no control arm (all 9 are FS+DW). Cross-check by comparing against today's 8x `dreamworld_c4_i4_w025` run's BPB at seed=2674 or seed=4011 (from `exp24_muon_fullval_8x_20260422T143312Z/`). At ws=4 the same arm should score *worse* (smaller effective batch, less data) — if Phase 0's `c4i4_w025` cell at seed=1337 beats the 8x run, something is wrong (dataset mismatch, bug). If it scores within ~0.05–0.10 BPB worse, that's the expected ws=4 penalty.
- Is the winner at a corner of the grid? If so, note this is a "sweep boundary" — the true optimum may be outside (e.g., interval=32 or weight=0.75). Mark as a follow-up question but don't expand Phase 0 unconditionally.
- Is the top-to-second gap > plausible seed noise? With only seed=1337, ordering within ~0.01 BPB is unreliable. If top 3 are within noise, carry all three into Task 5 (not just 1).

**Step 3: Write the winner doc**

Record:
- Full BPB table (all 9 arms)
- Picked winner config name + its exact DW settings
- Noise-band note: if multiple candidates tied, which ones we carry into FS sweep
- Commit.

```bash
git add experiments/24_training_time_bundle/PHASE0_DW_WINNER.md
git commit -m "exp24: record phase0 DW sweep winner"
```

---

## Task 4: Add `build_phase0_fastslow_sweep` around DW winner

**Files:**
- Modify: `experiments/24_training_time_bundle/exp24.py`
- Modify: `experiments/24_training_time_bundle/run_exp24.py`

**Step 1: Add builder with DW-winner values hardcoded**

Hardcode the Task 3 winner's DW settings in the builder (simpler than threading through CLI args for a one-shot sweep). If Task 3 carried forward more than one DW config due to noise, produce one matrix per carried config (expand to e.g. 12 arms: 2 DW candidates × 3 FS intervals × 2 alphas).

```python
def build_phase0_fastslow_sweep(
    *,
    speed_config: dict[str, Any],
    world_size: int = 4,
    budget_seconds: float = 600.0,
    seed_values: Sequence[int] = (1337,),
) -> list[dict[str, Any]]:
    """Phase 0 rung 2: sweep FS interval × alpha around Task 3 DW winner.

    DW settings below must match PHASE0_DW_WINNER.md. If changed, bump the
    commit hash in that doc and re-run Task 5 from scratch.
    """
    dw_cache_interval = <WINNER>  # fill from PHASE0_DW_WINNER.md
    dw_interval = <WINNER>
    dw_weight = <WINNER>

    fs_intervals = [16, 32, 64]
    fs_alphas = [0.25, 0.50]
    entries: list[dict[str, Any]] = []
    for fs_interval in fs_intervals:
        for fs_alpha in fs_alphas:
            arm = {
                "name_arm": (
                    f"phase0_fs_i{fs_interval}_a{int(fs_alpha*100):03d}_"
                    f"dw_c{dw_cache_interval}i{dw_interval}_w{int(dw_weight*100):03d}"
                ),
                "exp24_mechanism": "fast_slow_dreamworld",
                "artifact_impact": ARTIFACT_TRAINING_ONLY,
                "fast_slow_enabled": True,
                "fast_slow_interval": fs_interval,
                "fast_slow_alpha": fs_alpha,
                "fast_slow_eval_copy": "slow",
                "dreamworld_enabled": True,
                "dreamworld_cache_interval": dw_cache_interval,
                "dreamworld_interval": dw_interval,
                "dreamworld_weight": dw_weight,
                "dreamworld_prefix_tokens": 128,
                "dreamworld_replay_tokens": 64,
                "dreamworld_replay_batch_size": 128,
                "dreamworld_buffer_size": 16,
                "dreamworld_min_size": 2,
                "dreamworld_max_age_steps": 256,
            }
            for seed in seed_values:
                entry = _base_entry(
                    speed_config=speed_config,
                    world_size=world_size,
                    budget_seconds=budget_seconds,
                )
                entry.update(arm)
                name_arm = str(entry.pop("name_arm"))
                entries.append(
                    _named_entry(
                        base=entry,
                        phase="phase0",
                        mechanism=str(entry["exp24_mechanism"]),
                        arm=name_arm,
                        seed=int(seed),
                    )
                )
    return entries
```

**Step 2: Register in `run_exp24.py`** (same pattern as Task 1 Step 2; matrix name `phase0_fastslow_sweep`).

**Step 3: Dry-run — verify 6 entries**

```bash
python run_exp24.py --matrix phase0_fastslow_sweep --seeds 1337 --world-size 4 --show
```

**Step 4: Commit**

```bash
git add experiments/24_training_time_bundle/exp24.py experiments/24_training_time_bundle/run_exp24.py
git commit -m "exp24: add phase0 fast-slow sweep around DW winner"
```

---

## Task 5: Launch Phase 0 FS sweep on 4×H100

Same pattern as Task 2. 6 arms × ~700s ≈ 70 min.

**Step 1: Push commits, pull on pod, launch**

```bash
OUT=phase0_fs_sweep_4x_$(date -u +%Y%m%dT%H%M%SZ)
python run_exp24.py --matrix phase0_fastslow_sweep --seeds 1337 --world-size 4 --output-dir $OUT
```

**Step 2: Rsync back, commit**

```bash
git add experiments/24_training_time_bundle/phase0_fs_sweep_4x_*/
git commit -m "exp24: record phase0 fast-slow sweep results"
```

---

## Task 6: Pick top-2 configs for confirm

**Files:**
- Create: `experiments/24_training_time_bundle/PHASE0_TOP2.md`

**Step 1: Combine rankings**

Merge: Task 3 sweep (9 arms) ∪ Task 5 sweep (6 arms). Note the DW-sweep anchor cell (FS=32, α=0.50, winner DW) appears in both; dedupe by config, keep the lower BPB as the data point (or average if seed=1337 identical run).

**Step 2: Rank merged set by BPB; take top 2**

Criteria: mean BPB (only one seed so far, so just BPB). Break near-ties (< 0.005 BPB) by preferring the less aggressive schedule (larger FS interval, smaller DW weight) on the theory that it's more likely to generalize across seeds.

**Step 3: Document top-2 in `PHASE0_TOP2.md`**

Include full config for each, BPB from screening, and the tie-break rationale if applied. Commit.

```bash
git add experiments/24_training_time_bundle/PHASE0_TOP2.md
git commit -m "exp24: pick phase0 top-2 configs for confirm"
```

---

## Task 7: Add `build_phase0_confirm` and launch 3-seed confirm

**Files:**
- Modify: `experiments/24_training_time_bundle/exp24.py`
- Modify: `experiments/24_training_time_bundle/run_exp24.py`

**Step 1: Add builder with top-2 configs hardcoded**

```python
def build_phase0_confirm(
    *,
    speed_config: dict[str, Any],
    world_size: int = 4,
    budget_seconds: float = 600.0,
    seed_values: Sequence[int] = (1337, 2674, 4011),
) -> list[dict[str, Any]]:
    """Phase 0 rung 3: confirm top-2 configs × 3 seeds (full-val)."""
    top2 = [
        # Config A: fill from PHASE0_TOP2.md
        {"name_arm": "phase0_confirm_A_<short_label>", ...},
        # Config B
        {"name_arm": "phase0_confirm_B_<short_label>", ...},
    ]
    entries: list[dict[str, Any]] = []
    for arm in top2:
        for seed in seed_values:
            entry = _base_entry(
                speed_config=speed_config,
                world_size=world_size,
                budget_seconds=budget_seconds,
            )
            entry.update(arm)
            name_arm = str(entry.pop("name_arm"))
            entries.append(
                _named_entry(
                    base=entry,
                    phase="phase0",
                    mechanism=str(entry["exp24_mechanism"]),
                    arm=name_arm,
                    seed=int(seed),
                )
            )
    return entries
```

**Step 2: Register matrix** (same pattern; name `phase0_confirm`).

**Step 3: Dry-run — verify 6 entries (2 configs × 3 seeds)**

```bash
python run_exp24.py --matrix phase0_confirm --world-size 4 --show
```

**Step 4: Launch on pod**

```bash
OUT=phase0_confirm_4x_$(date -u +%Y%m%dT%H%M%SZ)
python run_exp24.py --matrix phase0_confirm --world-size 4 --output-dir $OUT
```

6 runs × ~700s ≈ 70 min.

**Step 5: Rsync, commit results**

```bash
git add experiments/24_training_time_bundle/phase0_confirm_4x_*/
git add experiments/24_training_time_bundle/exp24.py experiments/24_training_time_bundle/run_exp24.py
git commit -m "exp24: run phase0 top-2 × 3-seed confirm"
```

---

## Task 8: Lock the Exp 24 base config

**Files:**
- Create: `experiments/24_training_time_bundle/PHASE0_BASE_LOCK.md`
- Create: `experiments/24_training_time_bundle/configs/exp24_base.yaml` (the locked config)

**Step 1: Compute mean BPB per config across 3 seeds**

For each of the 2 confirm configs: mean and stddev of val BPB.

**Step 2: Decide the winner**

- Primary: lower mean BPB.
- Tiebreaker (within 1 stddev): lower seed-to-seed stddev. If still tied, prefer the simpler / less aggressive config.

**Step 3: Write `exp24_base.yaml`**

Full YAML with every knob needed for a reproducible run. Use `base_seq_epoch_lr0064_full_corpus.yaml` as the template for non-FS/DW knobs.

**Step 4: Document the decision**

`PHASE0_BASE_LOCK.md` includes:
- Full 2×3 BPB table
- Picked winner + rationale
- Any noise-band caveats
- Any sweep-boundary caveats from Task 3 that might warrant a follow-up Phase 0b
- Link to `exp24_base.yaml`

**Step 5: Commit**

```bash
git add experiments/24_training_time_bundle/PHASE0_BASE_LOCK.md \
        experiments/24_training_time_bundle/configs/exp24_base.yaml
git commit -m "exp24: lock phase0 base config"
```

---

## Task 9: Preregister Phase 0b entropy-gated `log_a` reread

**Files:**
- Create: `experiments/24_training_time_bundle/PHASE0B_ENTROPY_REREAD.md`
- Later implementation target, after prereg approval: `src/chaoscontrol/eval_stream/entropy_reread.py` or a thin runner around `src/chaoscontrol/eval_stream/temporal_heads.py`
- Later tests: `tests/test_eval_stream_entropy_reread.py`

**Step 1: Write the preregistration doc**

The doc must begin with the locked base reference:

- `PHASE0_BASE_LOCK.md`
- `configs/exp24_base.yaml`
- exact checkpoint path(s) used for Phase 0 confirm winner

Then state the mechanism:

```text
At a chunk boundary, use only information available before the next chunk is
scored. If the previous/base predictive entropy exceeds a frozen threshold,
replay a short suffix with both a longer-memory and shorter-memory
DeltaModulator(log_a_shift=s), choose or weight the direction causally, blend
the selected reread boundary state into the base boundary state, and score the
next chunk from that blended state.
```

Chunk-level gating is preferred for v0 because it makes leakage review simple. Token-level gating is allowed only if the entropy used for token `t` is computed before seeing the target token scored at `t`.

**Step 2: Freeze the calibration protocol**

Use a calibration stream that is not the primary full-val scoring stream. Record:

- entropy statistic: raw mean entropy or z-scored entropy over the previous chunk
- fire-rate thresholds: choose from target rates `{0.05, 0.10, 0.20}`; this sets the sensitivity, not a schedule
- suffix length `K`: choose from `{32, 64}` tokens, with `128` only as a boundary diagnostic
- shift pair: choose a symmetric or near-symmetric pair, e.g. `(-0.1, +0.1)` or `(-0.2, +0.2)`; do not choose a single primary direction
- blend policy: freeze a soft mix rule, e.g. constant `beta` or `beta = clamp((entropy_z - tau) / width, 0, beta_max)`
- direction policy: target-free candidate entropy/margin picker before scoring, or causal next-step-loss update after scoring that affects only subsequent state

Do not tune thresholds or shifts on the primary full-val stream. If that happens, mark the run exploratory and rerun with frozen settings.

**Step 3: Pre-register controls**

Run the locked checkpoint under:

1. `score_only`: no reread.
2. `scheduled_reread_same_budget`: deterministic/scheduled rereads at the same observed fire rate.
3. `entropy_reread_shift0`: entropy gate with `log_a_shift=0.0`, isolating extra scan/state-refresh compute from memory-horizon traversal.
4. `entropy_reread_bidirectional_blend`: entropy gate, short rewind, both shift directions available, soft state blend.
5. Diagnostic only: fixed longer-memory shift and fixed shorter-memory shift. These are not the promoted mechanism unless they expose that the direction picker is broken.
6. Optional diagnostic only: always-on best single shift. This is not the promoted mechanism unless it beats gating at lower or equal wall time.

**Step 4: Legality contract**

The implementation must have a regression test that proves current-chunk targets cannot affect the decision for current-chunk scoring. In review terms:

```python
# Legal: target-free confidence controls the next chunk.
gate_next = entropy(logits_for_previous_chunk) > threshold

# Legal: both reread directions run on a triggered boundary, then a target-free
# candidate signal chooses the state to blend before scoring the next target.
direction = choose_by_entropy_or_margin(long_state, short_state)
state_next = lerp(base_state, selected_reread_state, beta)

# Legal if lagged: current target loss may choose the carried state for future
# scoring only after the current score has already been recorded.
winner_for_future = argmin(nll_long_current, nll_short_current)

# Invalid for primary reporting: realized target loss controls the chunk
# whose score includes that target.
gate_current = nll(logits_current, targets_current) > threshold
```

Surprise/loss may be logged for analysis and may feed a lagged direction update, but it must not retroactively choose the score for the target that produced that loss.

**Step 5: Success and kill criteria**

Promote the mechanism only if `entropy_reread_bidirectional_blend`:

- improves full-val BPB over `score_only` by at least `0.003` and preferably `0.005`,
- beats `scheduled_reread_same_budget` on BPB per extra eval second,
- beats `entropy_reread_shift0`, showing that the gain is horizon traversal rather than mere repeated compute,
- beats both fixed-direction diagnostics or explains the fixed-direction winner as a stable regime worth splitting into a follow-up,
- fires on no more than 20% of chunks unless the eval budget still has obvious slack,
- stays within the 600s eval-time accounting rules used for the Param Golf path.

Kill or park the mechanism if same-horizon reread matches it, if the best always-on single shift is faster and better, if soft blending collapses to replacement/zero-blend across most triggers, or if any leakage review finds target-dependent current-chunk gating.

**Step 6: Commit**

```bash
git add experiments/24_training_time_bundle/PHASE0B_ENTROPY_REREAD.md
git commit -m "exp24: preregister phase0b entropy-gated log-a reread"
```

---

## Task 10: Unblock the event_sleep rigor+speed plan

**Files:**
- Modify: `docs/superpowers/plans/2026-04-22-exp24-rigor-and-speed-implementation.md`

**Step 1: Update that plan's header with the Phase 0 result**

Add a note near the top pointing at `PHASE0_BASE_LOCK.md` and the new `exp24_base.yaml`. Replace any baked-in FS/DW defaults in Task 3's synthetic config and Task 4's anchor discussion so they match the locked base.

**Step 2: Re-anchor Task 4's expectations**

Task 4 was flagged as needing re-anchor after any FS/DW change (revision log). Note explicitly that the first execution of Task 4 will produce a new anchor against the locked base — expect the previously discussed anchor trajectory to shift.

**Step 3: Re-anchor Task 7's profile harness config**

Task 7 loads a minimal `fast_slow_dreamworld_event_sleep` config. Update the harness reference so it reads from `exp24_base.yaml` instead of hard-coded defaults.

**Step 4: Commit**

```bash
git add docs/superpowers/plans/2026-04-22-exp24-rigor-and-speed-implementation.md
git commit -m "plan: rebase exp24 event_sleep plan on phase0 base lock"
```

---

## Final verification

- [ ] `exp24_base.yaml` exists and is reproducible via `run_exp24.py` with the matching matrix entry.
- [ ] `PHASE0_BASE_LOCK.md` records the full decision trail.
- [ ] `PHASE0B_ENTROPY_REREAD.md` exists after base lock and explicitly states that Phase 0b cannot alter `exp24_base.yaml`.
- [ ] Event_sleep plan references the locked base, not the placeholder anchor.
- [ ] All 3 matrices (`phase0_dreamworld_sweep`, `phase0_fastslow_sweep`, `phase0_confirm`) remain registered in `run_exp24.py` for reproducibility — do not delete them after locking the base.

---

## Budget & risk notes

**Total compute:** 9 + 6 + 6 = 21 runs × ~785s = **~4.6 hours of 4×H100 wall time** for base lock, plus Phase 0b eval-only on locked checkpoint (see below). Plus ~30 min of rsync/analysis between rungs. Plan on one full day end-to-end, same pod.

**Cost estimate.** 4×H100 at ~$12/hr × ~4.6h ≈ **~$55** for base lock. Phase 0b ≤ ~1h × $12/hr ≈ ~$12. **Total ≤ ~$70** for Phase 0 + 0b combined. (Reference: the original 8×H100 draft was ~$100 for base lock alone.)

**Phase 0b compute (same pod, after base lock).** Eval-only on the locked checkpoint — no training. Ballpark: 4 primary arms (`score_only`, `scheduled_reread_same_budget`, `entropy_reread_shift0`, `entropy_reread_bidirectional_blend`) × 3 seeds for confirm + ~4 diagnostic calibration runs on a held-out stream = ~16 runs × ~185s (full-val ~164s + startup ~20s) ≈ 50 min. If Task 9 pre-registration sweeps multiple thresholds or `K` values before picking one, add 20–40 min. Total Phase 0b wall time ≤ ~1.5 h on 4x.

**Ranking-transfer risk (ws=4 → ws=8 submission).** All Phase 0 rungs run at ws=4 (effective batch 4096) to cut cost. Submission regime is ws=8 (effective batch 8192). FS+DW knob rankings are expected to be roughly invariant to effective batch at this scale — interval/weight tune replay frequency, not batch dynamics — but "expected" is not "certain." The locked `exp24_base.yaml` may be suboptimal at ws=8. Catch happens at submit-time: when we first run the locked config on 8x, if BPB is off from the 4x-tuned expectation, re-run the FS sweep at ws=8 around the winner (6 arms × 1 seed × 8x ≈ 60 min, one-shot correction). LR stays at 0.064 per Exp 18 Test 5b validation at bs=1024/rank across ws∈{2,4,8}, so no LR re-tune needed for the ws swap.

**Single-seed screening risk.** Rungs 1 and 2 run seed=1337 only. Seed noise at ws=4 is plausibly larger than the ~0.01 BPB seen in prior 8x runs because the smaller effective batch (4096 vs 8192) yields noisier step trajectories — budget the noise band as ~0.015 BPB until the 3-seed confirm rung measures it directly. Task 3 and Task 6 include explicit noise-band handling; if >3 configs tie, escalate for a seed=2674 mini-replication rather than guessing.

**Sweep-boundary risk.** Both sweeps are 3-point grids at fixed endpoints. If a winner sits at a corner (e.g., interval=16 or weight=0.50 for DW), the true optimum may be outside the grid. Document as a Phase 0b candidate, don't expand Phase 0 mid-flight.

**Phase 0b leakage risk.** Predictive entropy is legal only when computed before the targets it affects are revealed. Current-token or current-chunk realized loss is not legal as a trigger or direction choice for that same scoring window. If both directions are scored and the lower next-step loss is kept, that choice must be committed only after the current score is recorded, affecting future state rather than the score that generated the loss. Prefer previous-chunk gating for the first implementation because it is easy to audit.

**Pod state.** Per project memory, `/workspace/venv` survives stop/start, pip-into-system-python doesn't. Always `source /workspace/venv/bin/activate` before launching. `runpodctl get pod --all` to see stopped pods. RunPod disk migrations copy tree only — if the pod migrates, `/workspace` shards may be gone and need re-staging.
