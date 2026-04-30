# Exp26 Adaptive Residual Memory Validation

Exp26 is no longer a headline ablation matrix. It is a fixed systems canary
for the architecture we actually mean:

```text
locked fast/slow trunk
+ CRCT evidence/oracle substrate
+ streaming Adaptive Residual Memory maintenance
+ learned Full-A commit authority
+ GPU3 physics confirmation
+ latest-complete GPU0->GPU3 teacher-weight mirror
```

CRCT-only, shadow-mode, calibration, and headline-arm switches were removed.
They made it too easy to run a diagnostic scaffold and mistake it for the
architecture. CRCT is the evidence substrate inside ARM here, not a standalone
mechanism.

## Runtime Contract

The thesis is not "sometimes run with memory." The trunk should always expose
an episodic residual input lane, and that lane must never make the trunk wait.

Target steady-state shape:

```text
GPU0-2 trunk ranks
  train the SSM at full speed
  consume fixed-shape latest-complete episodic residual buffers
  never synchronously read the cache
  never wait for GPU3 or the CPU controller

GPU3 memory/oracle rank
  owns the populated episodic sidecar
  runs memory_off / force_on / hide-slot / refresh-candidate physics
  may warm or capture its own fixed-shape oracle and maintenance CUDA graphs
  cannot warm or capture GPU0-2's graphs for them

CPU controller plane
  schedules work, maintains evidence, proposes bounded actions, and logs traces
  does not own exact oracle truth
  does not sit in the train-rank hot path
```

GPU0-2 graph compatibility comes from stable tensor addresses and shapes:
the recurrent stream always has a residual input buffer and gate buffer. The
contents may be zero, stale-but-safe, or latest-complete, but the trunk graph
does not change shape and does not block waiting for a fresher packet.
The train-rank encoder compile path has two static targets: the bare encoder
for no-packet steps and the packet encoder for latest-complete residual steps.
This keeps ARM on the compiled trunk path without making missing packets pay
zero-residual packet math.

`memory_mode="off"` remains valid only as an oracle/counterfactual mode: GPU3
uses it to measure marginal memory value. It is not the final product path.
Likewise, `force_on` and hide-slot modes are physics probes owned by GPU3, not
ablation knobs for normal trunk training.

The CPU/maintenance controller is the off-path evidence and scheduling plane;
learned Full-A authority lives there for maintenance/commit decisions. There is
no trunk-local `memory_controller` head in this path: semantic framing and cue
construction happen in the trunk, targeted episodic retrieval happens on the
memory plane, and the result returns only as a residual packet.

The current memory injection point is a gated residual added to the recurrent
stream before the SSM layers. It is not direct A-matrix modulation. If we add
A-modulation later, it should be named as a separate mechanism rather than
quietly overloading the residual lane.

The optimizer receives a sibling packet on the same latest-complete protocol:
GPU3 computes per-channel `plasticity_budget` from the correlation between
`abs(h_mem - h_off)` and positive memory utility, the mailbox carries the EMA
alongside the residual packet, and Muon uses it as a bounded LR multiplier on
SSM-channel parameters. This is the training-time "gist" signal: where episodic
memory is demonstrably supporting the recurrent state, the trunk can keep that
channel more plastic without waiting for the memory plane.

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
- Plasticity telemetry is present:
  `transport_summary.health.plasticity_packets_received`,
  `plasticity_budget_mean_received`, and optimizer
  `plasticity_budget.lr_multiplier_max`.
- The ARM cell offers teacher work every step; stream backpressure and the
  latest-complete mirror decide what GPU3 actually adopts.
- `transport_summary.health.weight_snapshot_published` and
  `weight_snapshot_applied` are non-zero in the ARM cell; mirror copy/save/apply
  timing and version lag are visible.
- Learned commit feedback updates are non-zero once oracle confirmations land.
- Fail-open/stale/drop counters are visible and bounded.
- Train-rank throughput is not catastrophically coupled to memory work.
- Treat any train-rank synchronous cache read, rank-3 wait, or memory-owned
  collective in the trunk step as a thesis violation.

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
