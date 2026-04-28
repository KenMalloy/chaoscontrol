# 600s/600s BPB Review and RunPod Action Plan (2026-04-28)

## Objective

Maximize held-out **BPB** under the competition constraint:
- **600s training budget**
- **600s eval budget**
- full 50k-doc FineWeb validation for final score

## What the existing results currently say

### Best observed full-val BPB in repo artifacts (phase0 family)

From existing `summary.json` artifacts under `experiments/24_training_time_bundle/`:

1. `phase0_fastslow_only_control_4x_20260423T193800Z`
   - Best seed run: `exp24_phase0_control_fastslow_only_i64a025_s4011`
   - `val_bpb = 1.4788831133674571`
   - `steps = 3560`, `per_gpu_tokens_per_sec ≈ 3.12M`

2. `phase0_fs_sweep_4x_20260423T005417Z`
   - Best run: `exp24_phase0_fs_i16a025_dw_c16i16_w010_s1337`
   - `val_bpb = 1.4799531117747762`

3. `phase0_confirm_4x_20260423T044539Z`
   - Best run: `exp24_phase0_confirm_A_fs_i32a025_dw_c16i16_w010_s1337`
   - `val_bpb = 1.4800138840188832`

4. `phase0_dw_sweep_4x_20260422T224040Z`
   - Best run: `exp24_phase0_fs_i32a050_dw_c16i16_w010_s1337`
   - `val_bpb = 1.4841480759076888`

### Interpretation

- Current local evidence says **fastslow-only control (`i64a025`) is the current leader** over dreamworld-coupled variants for the same budget regime.
- The margin is small but non-trivial for this stage:
  - ~`0.00107` BPB better than best `phase0_fs_sweep` cell
  - ~`0.00113` BPB better than best `phase0_confirm` cell
  - ~`0.00527` BPB better than best `phase0_dw_sweep` cell
- Throughput appears healthy and stable around ~3.1M per-GPU tok/s in the best family, which is exactly what we need under a hard 600s timer.

## Candidate ideas to improve BPB next (ranked)

1. **Exploit seed-variance with low-risk replication on the current winner.**
   - Run more seeds for `fastslow_only i64 a0.25` first.
   - Why: cheapest expected gain, keeps throughput and legality stable.

2. **Do a narrow local sweep around the winner, not a broad sweep.**
   - Suggested grid around winner:
     - `interval`: `[48, 64, 80]`
     - `alpha`: `[0.20, 0.25, 0.30]`
   - Keep mechanism otherwise unchanged; single-factor drift control.

3. **Use two-stage screening to protect 600s eval budget.**
   - Stage A: shorter surrogate scoring on a fixed subset (e.g., 2k docs) for ranking.
   - Stage B: full 50k-doc score only on top-k configs.

4. **Treat dreamworld as a conditional branch, not the default path.**
   - Re-introduce only if a specific setting closes the `~0.001+` BPB gap without harming step count.

5. **Harden run reproducibility and anti-regression checks.**
   - Freeze manifest + config hash + runtime env snapshot per cell.
   - Require full-val recompute for any candidate selected for final submission path.

## Proposed 48-hour action plan

### Phase 1: Revalidate current best (today)
- Run winner config (`fastslow_only i64 a0.25`) across seeds:
  - recommended seeds: `1337, 2674, 4011, 7331, 8893, 9901`
- Budget per run:
  - train `600s`
  - eval `600s` (full val)
- Success gate:
  - mean BPB improves vs current 3-seed baseline OR
  - p10 seed BPB better than current best single-seed baseline

### Phase 2: Local hyper-neighborhood sweep (same day/night)
- 3×3 grid:
  - interval in `{48,64,80}`
  - alpha in `{0.20,0.25,0.30}`
- Use one fixed seed for screening, then expand top 2 to 3 seeds.
- Keep all else fixed to avoid confounders.

### Phase 3: Final selection protocol (tomorrow)
- Choose top 2 candidates from Phase 1+2 by:
  1. mean BPB
  2. tail robustness across seeds
  3. steps completed / throughput stability
- Run each top candidate at 5 seeds + full 50k eval.
- Lock final submission candidate from aggregate result.

## RunPod execution checklist

### Preflight
- Confirm pod GPU shape and world size match intended matrix.
- Confirm `/workspace/venv` activated.
- Confirm data/cache paths available and writable.
- Confirm commit SHA + config bundle recorded in output metadata.

### During run
- Capture per-rank tok/s, step counts, and completion status.
- Persist raw rank outputs + merged summary json.
- Stop on first systemic failure signal (OOM/timeout/incomplete eval).

### Post-run
- Rank by full-val BPB.
- Emit compact leaderboard table (name, seed, BPB, steps, tok/s).
- Keep all raw artifacts for auditability.

## Risks and mitigations

- **Risk:** Overfitting to seed luck.  
  **Mitigation:** Require mean + p10 improvements over multi-seed baseline.

- **Risk:** Throughput regressions can hide BPB gains.  
  **Mitigation:** hard floor on step count / tok/s per GPU when accepting a candidate.

- **Risk:** eval budget overshoot.  
  **Mitigation:** enforce strict wall-clock timers and top-k promotion policy.

## Immediate next command set to run on RunPod (template)

> Use this as a template; adapt launcher/flags to the current matrix harness in your pod session.

```bash
# 0) environment
source /workspace/venv/bin/activate
cd /workspace/chaoscontrol

# 1) sanity tests before long run
CHAOSCONTROL_DIAG_SCAN_BACKEND=chunked \
PYTHONPATH=/workspace/chaoscontrol/src \
python -m pytest \
  tests/test_crct_runner_integration.py \
  tests/test_exp24_training_bundle.py \
  -q --tb=short

# 2) run winner replication matrix (example placeholder)
PYTHONPATH=/workspace/chaoscontrol/src \
python experiments/24_training_time_bundle/run_exp24.py \
  --matrix crct_v1 \
  --arm arm_a_fastslow_control \
  --seeds 1337,2674,4011,7331,8893,9901 \
  --train-budget-seconds 600 \
  --eval-budget-seconds 600 \
  --full-val-docs 50000 \
  --output-dir experiments/24_training_time_bundle/results/phase0_best_repl_$(date -u +%Y%m%dT%H%M%SZ)
```

If `run_exp24.py` CLI flags differ in your current branch, keep the same semantic budget constraints and output discipline.

## Automation helper added in-repo

Use `tools/exp24_best_bpb.py` to quickly rank the best `val_bpb` entries across all Exp24 `summary.json` artifacts:

```bash
python tools/exp24_best_bpb.py --top 10
python tools/exp24_best_bpb.py --top 10 --json
```

This makes winner selection reproducible and avoids manual scanning mistakes.

## SSH key note

Public keys alone are useful for pod account setup, but they are **not sufficient** for this environment to initiate SSH by themselves. A runnable pod endpoint plus matching private key material (or configured `runpodctl`) is still required here for direct remote execution from this container.
