# Exp 24 Phase 0: Base-Lock Tuning Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Tune the `fast_slow + scheduled dreamworld` stack on a 3-rung ladder (DW sweep → FS sweep around DW winner → top-2 × 3-seed confirm) and commit the winning config as the Exp 24 base for all downstream arms. Write a short Phase 0b preregistration doc as the last step (thesis + success criteria only) and point it at a follow-up plan for the actual implementation.

**Architecture:** Phase 0 runs as a data-collection plan, not a mechanism invention plan. Each tuning rung is a new matrix builder in `experiments/24_training_time_bundle/exp24.py`, dispatched from `run_exp24.py`. Screening rungs use seed=1337 single-seed; the confirm rung runs top-2 × 3 seeds with full-val. Winner is locked by mean BPB with run-to-run stability as tiebreaker. Phase 0b is deferred: entropy-gated `log_a` reread requires a new eval-stream module (pre-score state blend, direction picker, compute-matched control harness, CLI) that does not exist yet — `src/chaoscontrol/eval_stream/temporal_heads.py` only implements post-score log-prob mixing. That engineering belongs in its own plan; this plan only preregisters the thesis and success criteria so the follow-up plan inherits them unchanged.

**Out of scope:** `event_sleep`, predictive aux, spectral, ScOpt, and the actual Phase 0b implementation. The rigor+speed plan for `fast_slow_dreamworld_event_sleep` (`2026-04-22-exp24-rigor-and-speed-implementation.md`) stays frozen until Phase 0 lands. Phase 0b implementation becomes a follow-up plan after base lock, citing this plan's addendum as its preregistered thesis.

**Tech Stack:** Python, existing `run_exp24.py` / `exp24.py` matrix dispatch, **4×H100 pod** (cost choice, not technical requirement — see Budget section for ranking-transfer tradeoff vs ws=8 submission regime), 600s train budget, full-val eval (~164s on 4x per memory), FineWeb data already staged on the pod volume.

---

## Anchor (reference config)

The anchor below is the starting guess the ladder tunes around. It is included as one point in Task 3's DW sweep (the `c8_i8 × w=0.25` cell).

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

**Matched-budget controls (procedural pin).** The primary is bidirectional, so each trigger runs **two** reread passes (`+log_a_shift` and `-log_a_shift`). Define the per-stream reread-token budget as

```
B_reread = N_fire × K × num_candidate_passes
```

Run entropy-gated bidirectional primary first, record `N_fire` and `B_reread` over the full eval stream. Each compute-matched control must reproduce `B_reread` — *not* `N_fire` alone, which is only fire-count-matched and confounds gate-mechanism with compute:

- **Score-only floor** — `B_reread = 0`. Baseline; measures pure reread-compute value vs no reread.
- **Scheduled-reread compute-matched** — scheduled single-pass rereads at `2 × N_fire` total fires × `K` tokens = `B_reread`. Isolates *entropy-gated vs scheduled* at identical total reread compute.
- **Entropy-gated shift=0 compute-matched** — same entropy gate, same `N_fire`, but 2 redundant passes at `log_a_shift = 0.0` per fire = `B_reread`. Isolates *decay-shift effect vs mere refresh compute* while holding gate identity and budget fixed.
- **Fixed-direction shifted reread** — ablation only (single pass per fire, *not* budget-matched); not promoted. Run only if the bidirectional direction-picker underperforms, to diagnose whether the picker is broken or the mechanism itself is flat.

All controls use the same `600s` eval budget (per `feedback_train_eval_budget_separation`); reread compute is accounted against eval, never training. If any compute-matched control exhausts the `600s` eval budget before finishing the stream, record that as a failed feasibility check, not a win for the unmatched arm.

**Preregistered success criteria.** Written before the run, not after:

- **Promote to eval-time arm if all four hold:** (1) primary beats score-only floor by `≥ 0.015 BPB` mean over 3 seeds (clears ws=4 seed noise band ~0.015); (2) beats `scheduled_reread_compute_matched` by `≥ 0.005 BPB` mean (gate-type value at matched compute); (3) beats `entropy_reread_shift0_compute_matched` by `≥ 0.005 BPB` mean (decay-shift value beyond refresh compute); (4) seed-to-seed stddev `< 0.01 BPB`.
- **Shelve and record consistent-with-Exp-20 if:** primary is within `0.005 BPB` of score-only floor (either direction), OR beats `scheduled_reread_compute_matched` by `< 0.002 BPB`. The mechanism is not doing native-SSM work beyond generic reread compute.
- **Ambiguous / warrants follow-up if:** primary beats score-only floor by `0.005 to 0.015 BPB` but fails any of the other three promotion gates. Record, do not promote, design a sharper follow-up.
- **Deferred-blend secondary** is evaluated on the same four thresholds against its own matched-budget control set; it promotes only if primary also promotes (deferred-blend alone is not enough to claim an eval-time arm).

This addendum is not a fourth base-lock rung and does not run in this plan — it cannot influence which config lands in `exp24_base.yaml`. Task 11 writes `PHASE0B_ENTROPY_REREAD.md` as a preregistration doc only (thesis, controls, success criteria) so the follow-up plan inherits it unchanged. No Phase 0b eval runs in this plan. Task 12 handles the event_sleep plan rebase.

---

## Task 1: Extend `run_exp24.py` with checkpoint saving and full-val scoring

**Why this is Task 1.** As of `539307c`, `run_exp24.py` calls `run_matrix_entries` without `checkpoint_dir` (see `experiments/23_fast_path/launch.py:114` — the param is supported but unused) and `launch.py:67 summarize_result_dir` ranks by `tokens_per_sec`, not BPB. The base config has `eval_batches: 0`. Today's `exp24_muon_fullval_8x_20260422T143312Z` run produced BPB via a separate `scripts/run_exp20_fast_score.py` invocation per checkpoint — that orchestration was ad-hoc on the pod, not in the repo. Phase 0 cannot rank on BPB unless this path is wired.

**Files:**
- Modify: `experiments/24_training_time_bundle/run_exp24.py`
- Modify: `experiments/23_fast_path/launch.py` (extend `summarize_result_dir` to merge full-val BPB from the `full_val/` dir it writes)
- Reference: `scripts/run_exp20_fast_score.py:1129-1172` (scorer CLI surface)
- Reference: `experiments/24_training_time_bundle/exp24_muon_fullval_8x_20260422T143312Z/logs/*.full_val.log` (today's command shape)

**Step 1: Add `--checkpoint-dir` and `--full-val-score` CLI flags to `run_exp24.py`**

```python
parser.add_argument(
    "--checkpoint-dir",
    type=Path,
    default=None,
    help="If set, save per-entry training checkpoints here via runner --output-ckpt.",
)
parser.add_argument(
    "--full-val-score",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="After training each entry, run scripts/run_exp20_fast_score.py on the saved checkpoint.",
)
parser.add_argument(
    "--val-cache-dir",
    type=Path,
    default=Path("/workspace/cache/exp23_val_16384"),
    help="Tokenized val cache dir for the fast scorer.",
)
parser.add_argument(
    "--val-budget-seconds",
    type=float,
    default=600.0,
    help="Eval budget passed to run_exp20_fast_score.py.",
)
```

Gate: `--full-val-score` requires `--checkpoint-dir` — raise at parse time if one is set without the other.

**Step 2: Pass `checkpoint_dir` into `run_matrix_entries`**

```python
summary = run_matrix_entries(
    ...
    checkpoint_dir=args.checkpoint_dir,
)
```

Default `args.checkpoint_dir` to `args.output_dir / "checkpoints"` when `--full-val-score` is set but `--checkpoint-dir` is not (convenience).

**Step 3: After `run_matrix_entries`, if `--full-val-score`, score each entry**

Add a helper in `run_exp24.py`:

```python
def _score_full_val(
    *,
    entries: list[dict[str, Any]],
    checkpoint_dir: Path,
    results_dir: Path,
    world_size: int,
    cache_dir: Path,
    budget_seconds: float,
) -> None:
    full_val_dir = results_dir / "full_val"
    full_val_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = results_dir / "logs"
    scorer = REPO / "scripts" / "run_exp20_fast_score.py"
    for entry in entries:
        name = str(entry["name"])
        ckpt = checkpoint_dir / f"{name}.pt"
        if not ckpt.exists():
            continue  # training errored; launch.py will have recorded it
        summary_path = full_val_dir / f"{name}.summary.json"
        if summary_path.exists():
            continue  # resume-friendly
        jsonl_path = full_val_dir / f"{name}.jsonl"
        log_path = logs_dir / f"{name}.full_val.log"
        cmd = [
            "python", "-m", "torch.distributed.run",
            f"--nproc_per_node={world_size}",
            "--rdzv-endpoint=localhost:0",
            "--rdzv-backend=c10d",
            f"--rdzv-id=score_{name}",
            str(scorer),
            "--cache-dir", str(cache_dir),
            "--checkpoint-path", str(ckpt),
            "--output-path", str(jsonl_path),
            "--summary-path", str(summary_path),
            "--chunk-size", "256",
            "--budget-seconds", str(budget_seconds),
            "--doc-batch-size", "4096",
            "--max-forward-tokens", "auto",
            "--score-boundary-targets",
            "--doc-packing", "chunk_count_tail",
        ]
        with log_path.open("w") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
```

Pattern must match the command shape in today's 8x run logs (cited above). Do not invent flags; copy from the observed working invocation.

**Step 4: Extend `summarize_result_dir` in `launch.py` to include `val_bpb`**

Change `summarize_result_dir` so it reads `full_val/<name>.summary.json` when present and adds `val_bpb` (from `aggregate_bpb`) and `val_docs_scored` to the ranked entry dict. Re-sort by `val_bpb` ascending when all entries have it; fall back to `tokens_per_sec` descending if any entry lacks it. This keeps backward compatibility with the existing non-full-val matrices.

**Step 5: Unit test the new scorer hookup**

Add to `tests/test_exp24_training_bundle.py`:

```python
def test_run_exp24_full_val_requires_checkpoint_dir():
    # --full-val-score without --checkpoint-dir must raise at parse time
    ...

def test_score_full_val_builds_expected_cmd(tmp_path):
    # Inspect the cmd list (dry-run), assert it matches the observed shape
    ...

def test_summarize_result_dir_merges_val_bpb(tmp_path):
    # Write a fake entry json + full_val/*.summary.json, assert val_bpb appears and ranking sorts ascending by BPB
    ...
```

**Step 6: Verify locally with `--dry-run --show`**

```bash
python run_exp24.py --matrix fastslow_dreamworld --full-val-score --checkpoint-dir /tmp/ck --val-cache-dir /tmp/v --world-size 4 --dry-run --show
```

Expected: no crash, entries printed, `[exp24] full-val-score enabled` note, and the scorer command shape visible in the printed plan. Do not actually execute.

**Step 7: Commit**

```bash
git add experiments/24_training_time_bundle/run_exp24.py \
        experiments/23_fast_path/launch.py \
        tests/test_exp24_training_bundle.py
git commit -m "exp24: wire checkpoint saving and full-val BPB scoring into run_exp24"
```

---

## Task 2: Matrix-shape unit tests for the three phase0 builders

**Files:**
- Modify: `tests/test_exp24_training_bundle.py`

Pin entry count, names, phase tag, and load-bearing knobs for each Phase 0 builder so downstream edits don't silently break the sweep. One test per builder:

```python
def test_build_phase0_dreamworld_sweep_shape_and_knobs():
    speed_config = {...minimal valid...}
    entries = build_phase0_dreamworld_sweep(
        speed_config=speed_config, world_size=4, budget_seconds=600.0, seed_values=(1337,),
    )
    assert len(entries) == 9
    names = [e["name"] for e in entries]
    # Exactly the 3x3 grid, names pinned
    expected = {
        f"exp24_phase0_fs_i32a050_dw_c{i}i{i}_w{int(w*100):03d}_s1337"
        for i in (4, 8, 16) for w in (0.10, 0.25, 0.50)
    }
    assert set(names) == expected
    # Fast-slow pinned at anchor across every arm
    for e in entries:
        assert e["fast_slow_enabled"] is True
        assert e["fast_slow_interval"] == 32
        assert e["fast_slow_alpha"] == 0.50
        assert e["fast_slow_eval_copy"] == "slow"
        assert e["dreamworld_enabled"] is True
        # DW interval and cache move together
        assert e["dreamworld_cache_interval"] == e["dreamworld_interval"]
        assert e["exp24_phase"] == "phase0"


def test_build_phase0_fastslow_sweep_shape_and_knobs():
    # 6 entries: 3 FS intervals x 2 alphas, DW pinned at Task 5 winner
    ...


def test_build_phase0_confirm_shape_and_knobs():
    # 6 entries: 2 configs x 3 seeds, fullval-ready
    assert any(e["seed"] == 1337 for e in entries)
    assert any(e["seed"] == 2674 for e in entries)
    assert any(e["seed"] == 4011 for e in entries)
```

**Step 1-3: write tests, run them to fail (builders don't exist yet), commit as "tests: pin phase0 matrix shapes and names"**

Builders land in Tasks 3, 6, 9 respectively. Each builder task will re-run its test to green before commit.

---

## Task 3: Add `build_phase0_dreamworld_sweep` to `exp24.py`

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
                "name_arm": f"fs_i32a050_dw_c{interval}i{interval}_w{int(weight*100):03d}",
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

Expected: 9 entries named `exp24_phase0_fs_i32a050_dw_c{4,8,16}i{4,8,16}_w{010,025,050}_s1337`. Verify the exact name format by reading `exp24.py:_named_entry` (line ~114) — it formats as `f"exp24_{phase}_{arm}_s{seed}"` when phase is set, so the arm string must not duplicate the `phase0_` prefix.

**Step 4: Commit**

```bash
git add experiments/24_training_time_bundle/exp24.py experiments/24_training_time_bundle/run_exp24.py
git commit -m "exp24: add phase0 dreamworld interval×weight sweep matrix"
```

---

## Task 4: Launch Phase 0 DW sweep on 4×H100

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
python run_exp24.py --matrix phase0_dreamworld_sweep --seeds 1337 --world-size 4 \
    --checkpoint-dir $OUT/checkpoints --full-val-score --output-dir $OUT
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

## Task 5: Pick DW winner

**Files:**
- Create: `experiments/24_training_time_bundle/PHASE0_DW_WINNER.md`

**Step 1: Rank the 9 arms by val BPB**

Per Task 1, the `--full-val-score` path writes `full_val/<name>.summary.json` per arm and the extended `summarize_result_dir` merges each arm's `aggregate_bpb` into `summary.json`'s `ranked` list as `val_bpb`. Sort ascending.

```bash
jq '.ranked | sort_by(.val_bpb) | .[] | {name, val_bpb}' $OUT/summary.json
```

If `val_bpb` is missing for any arm, inspect that arm's `logs/<name>.full_val.log` — training may have succeeded but scoring errored (e.g., checkpoint corrupt, val cache missing).

**Step 2: Sanity checks**

- Smoke-level sanity (not a pass/fail gate). The 9-arm sweep has no bare control and today's 8x reference (`dreamworld_c4_i4_w025`, FS disabled) is a different mechanism than any Phase 0 arm (all have FS on), so direct cross-run BPB comparison is apples-to-oranges. Loose smoke checks:
  - No arm's BPB exceeds 2.0 (known-pathological; would signal a bug or dataset mismatch).
  - Within-sweep ordering is coherent: adjacent grid cells (e.g. `c8i8_w025` vs `c8i8_w050`) differ by less than ~0.3 BPB. Large discontinuities signal a bug in that arm's config.
  - If any arm errors or hangs, investigate before continuing — do not treat its absence as a ranking signal.
- Is the winner at a corner of the grid? If so, note this is a "sweep boundary" — the true optimum may be outside (e.g., interval=32 or weight=0.75). Mark as a follow-up question but don't expand Phase 0 unconditionally.
- Is the top-to-second gap > plausible seed noise? With only seed=1337, ordering within ~0.015 BPB is unreliable (see ws=4 noise budget in Risk section). If top 3 are within noise, carry all three into Task 7 (not just 1).

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

## Task 6: Add `build_phase0_fastslow_sweep` around DW winner

**Files:**
- Modify: `experiments/24_training_time_bundle/exp24.py`
- Modify: `experiments/24_training_time_bundle/run_exp24.py`

**Step 1: Add builder with DW-winner values hardcoded**

Hardcode the Task 5 winner's DW settings in the builder (simpler than threading through CLI args for a one-shot sweep). If Task 5 carried forward more than one DW config due to noise, produce one matrix per carried config (expand to e.g. 12 arms: 2 DW candidates × 3 FS intervals × 2 alphas).

```python
def build_phase0_fastslow_sweep(
    *,
    speed_config: dict[str, Any],
    world_size: int = 4,
    budget_seconds: float = 600.0,
    seed_values: Sequence[int] = (1337,),
) -> list[dict[str, Any]]:
    """Phase 0 rung 2: sweep FS interval × alpha around Task 5 DW winner.

    DW settings below must match PHASE0_DW_WINNER.md. If changed, bump the
    commit hash in that doc and re-run Task 7 from scratch.
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
                    f"fs_i{fs_interval}_a{int(fs_alpha*100):03d}_"
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

**Step 2: Register in `run_exp24.py`** (same pattern as Task 3 Step 2; matrix name `phase0_fastslow_sweep`).

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

## Task 7: Launch Phase 0 FS sweep on 4×H100

Same pattern as Task 4. 6 arms × ~785s ≈ 78 min on 4x.

**Step 1: Push commits, pull on pod, launch**

```bash
OUT=phase0_fs_sweep_4x_$(date -u +%Y%m%dT%H%M%SZ)
python run_exp24.py --matrix phase0_fastslow_sweep --seeds 1337 --world-size 4 \
    --checkpoint-dir $OUT/checkpoints --full-val-score --output-dir $OUT
```

**Step 2: Rsync back, commit**

```bash
git add experiments/24_training_time_bundle/phase0_fs_sweep_4x_*/
git commit -m "exp24: record phase0 fast-slow sweep results"
```

---

## Task 8: Pick top-2 configs for confirm

**Files:**
- Create: `experiments/24_training_time_bundle/PHASE0_TOP2.md`

**Step 1: Combine rankings**

Merge: Task 4 DW sweep results (9 arms) ∪ Task 7 FS sweep results (6 arms). Note the DW-sweep anchor cell (FS=32, α=0.50, winner DW) appears in both; dedupe by config, keep the lower BPB as the data point (or average if seed=1337 identical run).

**Step 2: Rank merged set by BPB; take top 2**

Criteria: mean BPB (only one seed so far, so just BPB). Break near-ties (< 0.005 BPB) by preferring the less aggressive schedule (larger FS interval, smaller DW weight) on the theory that it's more likely to generalize across seeds.

**Step 3: Document top-2 in `PHASE0_TOP2.md`**

Include full config for each, BPB from screening, and the tie-break rationale if applied. Commit.

```bash
git add experiments/24_training_time_bundle/PHASE0_TOP2.md
git commit -m "exp24: pick phase0 top-2 configs for confirm"
```

---

## Task 9: Add `build_phase0_confirm` and launch 3-seed confirm

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
python run_exp24.py --matrix phase0_confirm --world-size 4 \
    --checkpoint-dir $OUT/checkpoints --full-val-score --output-dir $OUT
```

6 runs × ~700s ≈ 70 min.

**Step 5: Rsync, commit results**

```bash
git add experiments/24_training_time_bundle/phase0_confirm_4x_*/
git add experiments/24_training_time_bundle/exp24.py experiments/24_training_time_bundle/run_exp24.py
git commit -m "exp24: run phase0 top-2 × 3-seed confirm"
```

---

## Task 10: Lock the Exp 24 base config

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
- Any sweep-boundary caveats from Task 5 that might warrant a follow-up tuning pass
- Link to `exp24_base.yaml`

**Step 5: Commit**

```bash
git add experiments/24_training_time_bundle/PHASE0_BASE_LOCK.md \
        experiments/24_training_time_bundle/configs/exp24_base.yaml
git commit -m "exp24: lock phase0 base config"
```

---

## Task 11: Preregister Phase 0b entropy-gated `log_a` reread (doc only)

**No eval runs in this task.** Phase 0b requires a new eval-stream module (pre-score state blend, direction picker, compute-matched control harness, CLI) that does not yet exist. `src/chaoscontrol/eval_stream/temporal_heads.py` only implements post-score log-prob mixing. The actual implementation and evaluation belong in a follow-up plan. This task writes the preregistration doc and stubs the follow-up plan so the thesis, controls, and thresholds are frozen before any implementation starts.

**Files:**
- Create: `experiments/24_training_time_bundle/PHASE0B_ENTROPY_REREAD.md` (the preregistration — thesis, controls, success criteria, legality contract)
- Create: `docs/superpowers/plans/2026-04-23-exp24-phase0b-entropy-reread-implementation.md` (follow-up plan stub that cites this preregistration)
- Future implementation target (not this plan): `src/chaoscontrol/eval_stream/entropy_reread.py`, `scripts/run_exp20_entropy_reread.py`, `tests/test_eval_stream_entropy_reread.py`

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

**Step 3: Pre-register controls (compute-matched per addendum)**

Run the locked checkpoint under the addendum's compute-matched control set. Budget `B_reread = N_fire × K × 2` is measured from the primary (`entropy_reread_bidirectional_blend`) run first; each control then reproduces that same total reread-token count.

1. `score_only`: no reread. `B_reread = 0`.
2. `scheduled_reread_compute_matched`: scheduled single-pass rereads at `2 × N_fire` fires × `K` tokens. Matches `B_reread`. Isolates gate-type (entropy vs schedule) at matched compute.
3. `entropy_reread_shift0_compute_matched`: entropy gate at `N_fire` fires × 2 redundant passes at `log_a_shift = 0.0`. Matches `B_reread`. Isolates decay-shift effect from refresh compute.
4. `entropy_reread_bidirectional_blend`: the primary. `N_fire` fires × 2 passes (`+log_a_shift`, `-log_a_shift`) × `K` tokens = `B_reread`.
5. Ablation only (not budget-matched): fixed longer-memory shift, fixed shorter-memory shift. Single pass per fire. Run these only if the direction-picker in control 4 underperforms, to diagnose whether the picker is broken or the mechanism itself is flat.
6. Ablation only (not budget-matched): always-on best single shift. Run only if controls 2–4 all fail to promote, to check whether gating itself is the dead mechanism.

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

Thresholds are inherited from the addendum (lines 53–56) and are load-bearing — do not weaken them without a revision-log entry:

- **Promote** `entropy_reread_bidirectional_blend` **only if all four hold**:
  1. Beats `score_only` floor by `≥ 0.015 BPB` mean over 3 seeds (clears ws=4 seed noise band ~0.015).
  2. Beats `scheduled_reread_compute_matched` by `≥ 0.005 BPB` mean over 3 seeds (gate adds value at identical compute).
  3. Beats `entropy_reread_shift0_compute_matched` by `≥ 0.005 BPB` mean over 3 seeds (decay-shift adds value beyond refresh compute).
  4. Seed-to-seed stddev `< 0.01 BPB` across the 3 seeds (result is stable, not seed-picked).
- **Shelve and record consistent-with-Exp-20 if:** primary is within `0.005 BPB` of `score_only` (either direction), OR beats `scheduled_reread_compute_matched` by `< 0.002 BPB`. The mechanism is not doing native-SSM work beyond generic reread compute.
- **Ambiguous / follow-up if:** primary beats `score_only` by `0.005 to 0.015 BPB` but fails any of the other three promotion gates. Record, do not promote, design a sharper follow-up.

Additional operational constraints (always applied, orthogonal to promotion):

- Primary must fire on `≤ 20%` of chunks unless the eval budget still has obvious slack after a full pass.
- All controls and the primary must stay within the `600s` eval-time accounting rule; any arm that exhausts the eval budget before finishing the stream is recorded as a failed feasibility check, not a result.
- Fixed-direction diagnostic arms (5, 6) do *not* gate promotion — they only inform post-hoc diagnosis of which component (gate / picker / shift) is the bottleneck if the primary fails.
- Deferred-blend secondary is evaluated on the same four thresholds against its own matched-budget control set; it promotes only if the primary also promotes.

Kill or park the mechanism if same-horizon reread matches it, if the best always-on single shift is faster and better, if soft blending collapses to replacement/zero-blend across most triggers, or if any leakage review finds target-dependent current-chunk gating.

**Step 6: Write the follow-up plan stub**

Create `docs/superpowers/plans/2026-04-23-exp24-phase0b-entropy-reread-implementation.md` with:

- Header pointing at `PHASE0B_ENTROPY_REREAD.md` as the preregistered thesis (frozen; must not be weakened without a revision-log entry in this Phase 0 plan).
- Pointer to `PHASE0_BASE_LOCK.md` and `exp24_base.yaml` for the locked checkpoint.
- Placeholder task list (actual tasks designed in the follow-up plan): build `eval_stream/entropy_reread.py` with pre-score state blend; build direction picker and legality tests; build compute-matched control harness; build CLI `scripts/run_exp20_entropy_reread.py`; calibrate on a held-out stream; run the 4 compute-matched arms × 3 seeds; evaluate against success criteria from the preregistration.
- Budget placeholder: to be estimated by the follow-up plan once the implementation is designed.

The stub is a pointer, not an implementation plan — the follow-up plan's author uses `superpowers:writing-plans` to expand it.

**Step 7: Commit**

```bash
git add experiments/24_training_time_bundle/PHASE0B_ENTROPY_REREAD.md \
        docs/superpowers/plans/2026-04-23-exp24-phase0b-entropy-reread-implementation.md
git commit -m "exp24: preregister phase0b entropy-gated log-a reread; stub follow-up plan"
```

---

## Task 12: Unblock the event_sleep rigor+speed plan

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

- [ ] `run_exp24.py` supports `--checkpoint-dir` and `--full-val-score`, and a dry-run shows the scorer command shape from today's 8x run logs.
- [ ] Matrix-shape unit tests (Task 2) pass and pin names/knobs for all three phase0 builders.
- [ ] `exp24_base.yaml` exists and is reproducible via `run_exp24.py` with the matching matrix entry.
- [ ] `PHASE0_BASE_LOCK.md` records the full decision trail.
- [ ] `PHASE0B_ENTROPY_REREAD.md` exists as a preregistration doc (thesis, controls, success criteria) and explicitly states Phase 0b cannot alter `exp24_base.yaml`.
- [ ] `docs/superpowers/plans/2026-04-23-exp24-phase0b-entropy-reread-implementation.md` stub exists and cites the preregistration.
- [ ] Event_sleep plan references the locked base, not the placeholder anchor.
- [ ] All 3 matrices (`phase0_dreamworld_sweep`, `phase0_fastslow_sweep`, `phase0_confirm`) remain registered in `run_exp24.py` for reproducibility — do not delete them after locking the base.

---

## Budget & risk notes

**Total compute:** 9 + 6 + 6 = 21 runs × ~785s training + eval-scoring = **~4.6 hours of 4×H100 wall time** for base lock. Plus ~30 min of rsync/analysis between rungs. Plan on one full day end-to-end.

**Cost estimate.** 4×H100 at ~$12/hr × ~4.6h ≈ **~$55** for base lock. (Reference: the original 8×H100 draft was ~$100.) Phase 0b compute is not in this plan's budget — it runs from the follow-up plan after the actual eval-stream implementation lands.

**Ranking-transfer risk (ws=4 → ws=8 submission).** All Phase 0 rungs run at ws=4 (effective batch 4096) to cut cost. Submission regime is ws=8 (effective batch 8192). FS+DW knob rankings are expected to be roughly invariant to effective batch at this scale — interval/weight tune replay frequency, not batch dynamics — but "expected" is not "certain." The locked `exp24_base.yaml` may be suboptimal at ws=8. Catch happens at submit-time: when we first run the locked config on 8x, if BPB is off from the 4x-tuned expectation, re-run the FS sweep at ws=8 around the winner (6 arms × 1 seed × 8x ≈ 60 min, one-shot correction). LR stays at 0.064 per Exp 18 Test 5b validation at bs=1024/rank across ws∈{2,4,8}, so no LR re-tune needed for the ws swap.

**Single-seed screening risk.** Rungs for Tasks 4 and 7 run seed=1337 only. Seed noise at ws=4 is plausibly larger than the ~0.01 BPB seen in prior 8x runs because the smaller effective batch (4096 vs 8192) yields noisier step trajectories — budget the noise band as ~0.015 BPB until the 3-seed confirm rung (Task 9) measures it directly. Tasks 5 and 8 include explicit noise-band handling; if >3 configs tie, escalate for a seed=2674 mini-replication rather than guessing.

**Sweep-boundary risk.** Both sweeps are 3-point grids at fixed endpoints. If a winner sits at a corner (e.g., interval=16 or weight=0.50 for DW), the true optimum may be outside the grid. Document as a follow-up candidate, don't expand Phase 0 mid-flight.

**Pod state.** Per project memory, `/workspace/venv` survives stop/start, pip-into-system-python doesn't. Always `source /workspace/venv/bin/activate` before launching. `runpodctl get pod --all` to see stopped pods. RunPod disk migrations copy tree only — if the pod migrates, `/workspace` shards may be gone and need re-staging.
