# Exp26 Adaptive Residual Memory Validation

Exp26 is no longer a headline ablation matrix. It is a fixed systems canary
for the architecture we actually mean:

```text
locked fast/slow trunk
+ CRCT evidence/oracle substrate
+ streaming Adaptive Residual Memory maintenance
+ learned Full-A commit authority
+ GPU3 physics confirmation
```

CRCT-only, shadow-mode, calibration, and headline-arm switches were removed.
They made it too easy to run a diagnostic scaffold and mistake it for the
architecture. CRCT is the evidence substrate inside ARM here, not a standalone
mechanism.

## Validation Cells

| Cell | Purpose |
|---|---|
| `validation_fastslow_control` | locked fast/slow trunk, no CRCT, no maintenance |
| `validation_adaptive_residual_memory` | full ARM path: CRCT evidence, GPU3 oracle, learned maintenance, traces |

The run is not intended to produce a BPB claim. It answers: does the system
light up without breaking the trunk-throughput contract?

## Artifact Lock

Exp26 locks the trunk at `model_dim=384`. That is the largest comfortable
increase over the exp24 `256` lock under the 16 MB artifact budget with the
16k vocabulary: local artifact-pipeline sizing of the CRCT+bucket-prototype
shape gives `384 -> 13.71 MB`, `416 -> 15.19 MB`, `448 -> 16.73 MB`, and
`512 -> 20.16 MB` using the current int6/LZMA path.

## What To Check

- Both cells complete and write JSON results under `validation/`.
- The ARM cell writes replay-maintenance traces under `validation/traces/`.
- `replay_eviction_arm_runtime_enabled` is true and the runtime namespace is
  per-cell.
- GPU3 oracle/maintenance telemetry is non-empty.
- Learned commit feedback updates are non-zero once oracle confirmations land.
- Fail-open/stale/drop counters are visible and bounded.
- Train-rank throughput is not catastrophically coupled to memory work.

## Usage

```bash
# Inspect the exact two validation entries without launching.
PYTHONPATH=src .venv/bin/python experiments/26_arm/run_exp26.py --dry-run

# Run the fixed validation canary.
PYTHONPATH=src .venv/bin/python experiments/26_arm/run_exp26.py --budget 45
```

Operational flags remain for paths, world size, seed, budget, and dry-run.
There is no `--stage`, no `--arms`, no CRCT-only mode, and no shadow/headline
matrix path.
