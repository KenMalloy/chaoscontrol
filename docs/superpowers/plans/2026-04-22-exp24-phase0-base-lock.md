# Exp 24 Phase 0: Base-Lock Tuning Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Tune the `fast_slow + scheduled dreamworld` stack on a 3-rung ladder (DW sweep → FS sweep around DW winner → top-2 × 3-seed confirm) and commit the winning config as the Exp 24 base for all downstream arms.

**Architecture:** Phase 0 runs as a data-collection plan, not a code plan. Each tuning rung is a new matrix builder in `experiments/24_training_time_bundle/exp24.py`, dispatched from `run_exp24.py`. Screening rungs (2 & 3) use seed=1337 single-seed; the confirm rung runs top-2 × 3 seeds with full-val. Winner is locked by mean BPB with run-to-run stability as tiebreaker.

**Out of scope:** `event_sleep`, predictive aux, spectral, ScOpt. The rigor+speed plan for `fast_slow_dreamworld_event_sleep` (`2026-04-22-exp24-rigor-and-speed-implementation.md`) stays frozen until Phase 0 lands, because its test anchors and profile harness bake in FS+DW defaults that Phase 0 is about to choose.

**Tech Stack:** Python, existing `run_exp24.py` / `exp24.py` matrix dispatch, 8×H100 pod, 600s train budget, full-val eval (~82s on 8x per memory), FineWeb data already staged on the pod volume.

---

## Anchor (reference config)

The anchor below is the starting guess the ladder tunes around. It is included as one point in Task 1's DW sweep (the `c8_i8 × w=0.25` cell).

- `fast_slow_enabled=True`, `fast_slow_interval=32`, `fast_slow_alpha=0.50`, `fast_slow_eval_copy="slow"`
- `dreamworld_enabled=True`, `dreamworld_cache_interval=8`, `dreamworld_interval=8`, `dreamworld_weight=0.25`
- `dreamworld_replay_batch_size=128`, `dreamworld_prefix_tokens=128`, `dreamworld_replay_tokens=64`
- `dreamworld_buffer_size=16`, `dreamworld_min_size=2`, `dreamworld_max_age_steps=256`

All other knobs track the existing base config (`experiments/23_fast_path/configs/base_seq_epoch_lr0064_full_corpus.yaml`): 4-layer SSM, dim=256, bf16, LR=0.064, bs=1024 × 8 ranks, seq=512, Muon, sequential-epoch sampling.

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
    world_size: int = 8,
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
python run_exp24.py --matrix phase0_dreamworld_sweep --seeds 1337 --show
```

Expected: 9 entries named `exp24_phase0_fast_slow_dreamworld_phase0_fs_i32a050_dw_c{4,8,16}i{4,8,16}_w{010,025,050}_s1337`.

**Step 4: Commit**

```bash
git add experiments/24_training_time_bundle/exp24.py experiments/24_training_time_bundle/run_exp24.py
git commit -m "exp24: add phase0 dreamworld interval×weight sweep matrix"
```

---

## Task 2: Launch Phase 0 DW sweep on 8×H100

**Files:**
- Output dir: `experiments/24_training_time_bundle/phase0_dw_sweep_8x_<timestamp>/`

**Step 1: Verify pod state**

```bash
runpodctl get pod --all
```

Ensure an 8×H100 pod is RUNNING with `/workspace/venv` and FineWeb shards on the volume. If stopped, start it and wait for SSH.

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
OUT=phase0_dw_sweep_8x_$(date -u +%Y%m%dT%H%M%SZ)
python run_exp24.py --matrix phase0_dreamworld_sweep --seeds 1337 --output-dir $OUT
```

Expected: 9 runs × ~700s each (600s train + ~82s full-val + ~20s startup) ≈ 105 min total wall time on 8x (runs execute sequentially, not in parallel — each uses all 8 ranks).

**Step 4: Monitor**

Stream `progress.jsonl` or watch `summary.json` for per-arm BPB. Abort if any arm errors (OOM, NCCL timeout, NaN loss) — investigate before relaunching.

**Step 5: Rsync results back to laptop**

```bash
rsync -av <pod>:/workspace/chaoscontrol/experiments/24_training_time_bundle/$OUT/ \
    ~/Local\ Documents/Developer/chaoscontrol/experiments/24_training_time_bundle/$OUT/
```

**Step 6: Commit results**

```bash
git add experiments/24_training_time_bundle/phase0_dw_sweep_8x_*/
git commit -m "exp24: record phase0 dreamworld sweep results"
```

---

## Task 3: Pick DW winner

**Files:**
- Create: `experiments/24_training_time_bundle/PHASE0_DW_WINNER.md`

**Step 1: Rank the 9 arms by val BPB**

From `summary.json`, extract `val_bpb` per arm. Sort ascending.

**Step 2: Sanity checks**

- Does the winner pass the control floor (prior 8x control ran 1.53ish per project memory, seed 1337 alone will vary a bit)? If no arm beats its control-seed baseline, flag and decide whether Phase 0 continues or we retune fast-slow first.
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
    world_size: int = 8,
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
python run_exp24.py --matrix phase0_fastslow_sweep --seeds 1337 --show
```

**Step 4: Commit**

```bash
git add experiments/24_training_time_bundle/exp24.py experiments/24_training_time_bundle/run_exp24.py
git commit -m "exp24: add phase0 fast-slow sweep around DW winner"
```

---

## Task 5: Launch Phase 0 FS sweep on 8×H100

Same pattern as Task 2. 6 arms × ~700s ≈ 70 min.

**Step 1: Push commits, pull on pod, launch**

```bash
OUT=phase0_fs_sweep_8x_$(date -u +%Y%m%dT%H%M%SZ)
python run_exp24.py --matrix phase0_fastslow_sweep --seeds 1337 --output-dir $OUT
```

**Step 2: Rsync back, commit**

```bash
git add experiments/24_training_time_bundle/phase0_fs_sweep_8x_*/
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
    world_size: int = 8,
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
python run_exp24.py --matrix phase0_confirm --show
```

**Step 4: Launch on pod**

```bash
OUT=phase0_confirm_8x_$(date -u +%Y%m%dT%H%M%SZ)
python run_exp24.py --matrix phase0_confirm --output-dir $OUT
```

6 runs × ~700s ≈ 70 min.

**Step 5: Rsync, commit results**

```bash
git add experiments/24_training_time_bundle/phase0_confirm_8x_*/
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

## Task 9: Unblock the event_sleep rigor+speed plan

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
- [ ] Event_sleep plan references the locked base, not the placeholder anchor.
- [ ] All 3 matrices (`phase0_dreamworld_sweep`, `phase0_fastslow_sweep`, `phase0_confirm`) remain registered in `run_exp24.py` for reproducibility — do not delete them after locking the base.

---

## Budget & risk notes

**Total compute:** 9 + 6 + 6 = 21 runs × ~700s = ~4.1 hours of 8×H100 wall time, plus ~30 min of rsync/analysis between rungs. Plan on one full day end-to-end.

**Single-seed screening risk.** Rungs 1 and 2 run seed=1337 only. If the true seed noise is ~0.01 BPB (plausible from prior 8x runs), cells within that band are indistinguishable. Task 3 and Task 6 include explicit noise-band handling; if >3 configs tie, escalate for a seed=2674 mini-replication rather than guessing.

**Sweep-boundary risk.** Both sweeps are 3-point grids at fixed endpoints. If a winner sits at a corner (e.g., interval=16 or weight=0.50 for DW), the true optimum may be outside the grid. Document as a Phase 0b candidate, don't expand Phase 0 mid-flight.

**Pod state.** Per project memory, `/workspace/venv` survives stop/start, pip-into-system-python doesn't. Always `source /workspace/venv/bin/activate` before launching. `runpodctl get pod --all` to see stopped pods. RunPod disk migrations copy tree only — if the pod migrates, `/workspace` shards may be gone and need re-staging.
