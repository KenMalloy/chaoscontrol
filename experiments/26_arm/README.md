# Exp26 Adaptive Residual Memory

Headline measurement of the streaming Adaptive Residual Memory architecture
on top of CRCT. Replaces exp24's training-time-bundle umbrella with a
focused staged experiment: first run a short runtime smoke, observe signal
distributions in shadow mode, calibrate threshold counterfactuals from
observed percentiles, then run the 4-arm headline matrix.

Exp26 locks the trunk at `model_dim=384`. That is the largest comfortable
increase over the exp24 `256` lock under the 16 MB artifact budget with the
16k vocabulary: local artifact-pipeline sizing of the CRCT+bucket-prototype
shape gives `384 -> 13.71 MB`, `416 -> 15.19 MB`, `448 -> 16.73 MB`, and
`512 -> 20.16 MB` using the current int6/LZMA path. `384` leaves enough
headroom for trained-weight entropy and metadata drift; `512` does not fit.

## Two-stage discipline

**Phase 0 — smoke.** Two short cells, isolated under `smoke/`: locked
fast/slow control plus active CRCT+ARM. This is not a matrix result and does
not write calibration/headline artifacts; it exists to catch runner, DDP,
data/tokenizer, mailbox, replay-maintenance, prototype, and trace-path bugs
before spending on calibration or 600s headline cells.

**Stage 1 — calibration.** One cell, single seed, ~180s, full ARM streaming
pipeline in shadow mode (`replay_eviction_mode="shadow"`). The policy proposes
maintenance actions but never mutates the cache. The per-decision trace
records `utility_ema`, `peak_utility`, `peak_sharpness`, `marginal_gain_ema`,
`contradiction_ema`, and three drift signals at every shadow-policy decision.

**Stage 2 — analysis.** `calibrate.analyze` reads the trace, computes
percentile summaries per signal, and writes `calibration/manifest.json`
with one threshold-counterfactual set:

- `thresholds_balanced` — anchored to p50 utility, p75 peak, p75 drift.

It rides on the observed distributions for post-hoc rule replay. It is not
an active commit regime.

**Stage 3 — headline.** `build_arm_v1_matrix` accepts the manifest path for
CLI continuity, but the active arm does not consume threshold manifests. The
active commit decision is owned by the learned Full-A action simplex plus
GPU3 physics confirmation.

## Arms

| Arm | What it tests | Mode | Thresholds |
|---|---|---|---|
| `arm_a_fastslow_control` | locked Phase-0 trunk, no CRCT, no maintenance | — | — |
| `arm_b_crct_controller` | CRCT alone (no maintenance) | — | — |
| `arm_c_crct_replay_shadow` | maintenance signal only, no mutation | shadow | permissive |
| `arm_d_crct_replay_active_learned` | active maintenance with learned Full-A commit authority | active | none |

The older hand-threshold split was removed because it put fixed thresholds in
charge of the commit decision. `agreement_count` remains only for the legacy
rule mode; the learned active arm acts from controller confidence plus GPU3
physics confirmation.

Replay maintenance uses `replay_eviction_scoring_mode=oracle`: GPU3 is the
memory/massage worker and scores scheduled slots with the real
`force_on` / no-sidecar / hide-slot path. CPU is the evidence/control plane:
it schedules bounded slot work, owns starvation telemetry, and keeps the
shared-memory job/result rings moving. It does not own exact oracle truth and
there is no `cpu_scorer` path in the oracle. The CPU conductor is the native
`ArmMaintenanceScheduler`; each run cell gets its own shared-memory job/result
ring namespace so scheduled slot work and GPU3 completion evidence remain
separate under parallel launches.

## Reading the matrix

- arm_a vs arm_b: does CRCT alone help over the locked trunk?
- arm_b vs arm_c: does the maintenance *signal* correlate with anything useful?
- arm_c vs arm_d: does learned controller authority over maintenance commits help?

## Layout

```
experiments/26_arm/
  README.md
  exp26.py               # matrix builders (calibration + headline)
  calibrate.py           # trace analyzer + manifest writer
  run_exp26.py           # three-stage orchestrator
  smoke/
    matrix.json          # populated by phase 0
    exp26_smoke_*        # short sanity-run results
  calibration/
    trace.ndjson         # populated by stage 1
    manifest.json        # populated by stage 2
  results/
    matrix.json          # headline matrix
    exp26_phase3_arm_*_s*.json   # per-cell results
    traces/
      arm_v1_arm_*_s*.ndjson     # per-arm-per-seed action traces
      crct_conflict.ndjson        # CRCT gradient-conflict trace (shared)
```

## Usage

```bash
# Full run on 4xH100 (smoke -> calibrate -> analyze -> headline).
python experiments/26_arm/run_exp26.py --stage all

# Phase-0 runtime smoke only.
python experiments/26_arm/run_exp26.py --stage smoke --smoke-budget 30

# Calibrate only (writes manifest, then stops).
python experiments/26_arm/run_exp26.py --stage calibrate
python experiments/26_arm/run_exp26.py --stage analyze

# Headline only (uses prior calibration if present).
python experiments/26_arm/run_exp26.py --stage headline

# Restrict headline to a subset of arms.
python experiments/26_arm/run_exp26.py --stage headline \
    --arms arm_a_fastslow_control arm_d_crct_replay_active_learned

# Dry-run any stage to inspect entries without launching.
python experiments/26_arm/run_exp26.py --stage all --dry-run
```

## What makes this not exp24

Exp24's framing was "training-time bundle" — many candidate mechanisms
under one umbrella. Exp26's framing is "Adaptive Residual Memory headline" —
one focused architecture. The fast/slow trunk lock and CRCT contract carry
over unchanged from exp24; what's new here is the calibration discipline
and learned controller authority over memory maintenance commits.

If you want the legacy CRCT v1 matrix (4 arms, hard-coded thresholds), use
`experiments/24_training_time_bundle/run_exp24.py --matrix crct_v1`.
