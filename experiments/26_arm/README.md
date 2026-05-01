# Exp26 Adaptive Residual Memory Validation

Exp26 is no longer a headline ablation matrix. It is a fixed systems canary
for the architecture we actually mean:

```text
locked fast/slow trunk
+ CRCT evidence/oracle substrate
+ streaming Adaptive Residual Memory maintenance
+ learned Full-A commit authority
+ dedicated packet-serving and maintenance memory ranks
+ latest-complete train->memory teacher-weight mirror
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
GPU0-5 trunk ranks on the final 8x run
  train the SSM at full speed
  consume fixed-shape latest-complete episodic residual buffers
  never synchronously read the cache
  never wait for a memory GPU or the CPU controller

GPU6 packet-serving rank
  owns the low-latency populated sidecar used to produce residual packets
  builds packets from the pre-recurrence episodic read path without full
    off/force recurrence or LM-head scoring
  publishes residual packets without trunk rendezvous
  does not emit authoritative utility, controller-target, or plasticity labels

GPU7 maintenance rank
  owns learned slot coverage, refresh, distill, and replay-style maintenance
  owns exact memory_off / force_on / hide-slot / refresh-candidate physics
    against its own request stream
  emits ordered slot commits to GPU6 after CPU-side legality/commit authority
  may warm or capture its own fixed-shape oracle and maintenance CUDA graphs
  cannot warm or capture trunk-rank graphs for them

CPU controller plane
  schedules work, maintains evidence, proposes bounded actions, and logs traces
  stamps slot-maintenance commits with ordering and legality evidence
  does not own exact oracle truth
  does not sit in the train-rank hot path
```

Four-GPU smoke runs use the compact 3+1 version of the same contract: GPU0-2
train and GPU3 shares packet serving plus maintenance. At eight GPUs, the
topology derives 6+2 automatically when maintenance is enabled; it is not an
ablation arm.

When GPU6 and GPU7 split, the two memory ranks hold replicated slot identity.
GPU6 is the serving authority for low-latency packets: it bootstrap-appends
packet-serving slots from the same pre-recurrence stream the trunk will consume
and publishes generation-stamped append commits to GPU7. Those appends are
serving proposals, not evidence labels. GPU7 owns the exact counterfactual
physics and learns against the replicated serving generations. After real
physics confirms a maintenance action, GPU7 sends a compact slot commit back to
GPU6 over the memory-rank peer lane. Commits contain the slot id, event id,
base generation, new generation, action, and an optional one-slot tensor
payload. GPU6 applies maintenance commits only if the generation matches; stale
or divergent commits are dropped and counted. The train ranks do not participate
in this lane. Capacity is owned by learned maintenance in the ARM config:
packet serving appends only into free slots and does not run local lossy
compression that GPU7 could not replay exactly.

Train-rank graph compatibility comes from stable tensor addresses and shapes:
the recurrent stream always has a residual input buffer and gate buffer. The
contents may be zero, stale-but-safe, or latest-complete, but the trunk graph
does not change shape and does not block waiting for a fresher packet.
The train-rank encoder compile path has two static targets: the bare encoder
for no-packet steps and the packet encoder for latest-complete residual steps.
This keeps ARM on the compiled trunk path without making missing packets pay
zero-residual packet math.
`cuda_graph_mode="probe"` is enabled for the ARM validation config so graph
eligibility is reported instead of silently disabled. Multi-rank CRCT still
keeps NCCL/DDP collectives outside full-step graph capture; the important
contract is that the packet encoder itself is compile-clean.

`memory_mode="off"` remains valid only as a counterfactual scoring mode: the
memory ranks use it to measure marginal memory value. It is not the final
product path. Likewise, `force_on` and hide-slot modes are physics probes
owned by the memory ranks, not ablation knobs for normal trunk training.

The CPU/maintenance controller is the off-path evidence and scheduling plane;
learned Full-A authority lives there for maintenance/commit decisions. There is
no trunk-local `memory_controller` head in this path: semantic framing and cue
construction happen in the trunk, targeted episodic retrieval happens on the
memory plane, and the result returns only as a residual packet.

The current memory injection point is a gated residual added to the recurrent
stream before the SSM layers. It is not direct A-matrix modulation. If we add
A-modulation later, it should be named as a separate mechanism rather than
quietly overloading the residual lane.

The optimizer-side packet uses the same latest-complete discipline but only
exact counterfactual physics may author it: per-channel `plasticity_budget`
comes from the correlation between `abs(h_mem - h_off)` and positive memory
utility. The low-latency GPU6 serving path deliberately does not synthesize a
plasticity packet from the approximate residual. When no fresh exact packet is
available, Muon falls back to last-good or neutral plasticity, matching the
residual lane's fail-open semantics.
Muon also owns the SSM role policy for the ARM cell: `delta_proj` stays on the
AdamW fallback because it encodes per-token timescale specialization, while
`log_a` gets per-channel beta from a slow EMA of its own value. The EMA is the
damping layer that prevents the old SemanticOptimizer feedback loop.
Memory-side modules are excluded from the train optimizer in CRCT runs; memory
ranks use them as the scoring/memory substrate, but train ranks do not spend
optimizer state on params their packet-mode trunk never reads.

## Validation Cells

| Cell | Purpose |
|---|---|
| `validation_fastslow_control` | locked fast/slow trunk, no CRCT, no maintenance |
| `validation_adaptive_residual_memory` | full ARM path: CRCT evidence, memory-rank physics, learned maintenance, traces |

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
- Packet-serving and maintenance telemetry is non-empty:
  `payloads_served`, `packet_service_seconds_*`, maintenance replay ticks, and
  exact maintenance/physics counters.
- Plasticity telemetry is explicit even when neutral:
  `transport_summary.health.plasticity_packets_received`,
  `plasticity_packets_missing`, `plasticity_budget_mean_received`, and optimizer
  `plasticity_budget.lr_multiplier_max` should tell us whether exact evidence
  packets actually reached the train optimizer.
- Optimizer role telemetry is present under `optimizer.ssm_role` and
  `optimizer.excluded_params`, including `log_a_beta_*` summaries and counts of
  memory-side params kept out of the train optimizer.
- The ARM cell offers teacher work every step; stream backpressure and the
  latest-complete mirror decide what the memory ranks actually adopt.
- ARM maintenance uses `replay_eviction_max_seconds=0.0`, meaning no
  software wall-clock governor. Ring occupancy, frame arrival, and memory-rank/CPU
  throughput are the backpressure sources; duty-cycle telemetry tells us if
  the memory plane is actually hot.
- `transport_summary.health.weight_snapshot_published` and
  `weight_snapshot_applied` are non-zero in the ARM cell; mirror copy/save/apply
  timing and version lag are visible.
- Learned commit feedback updates are non-zero once oracle confirmations land.
- On 8x runs, slot-commit telemetry is present:
  `transport_summary.slot_commit_packet`,
  `transport_summary.slot_commit_maintenance`, and health counters
  `append_commits_sent`, `append_commits_applied`,
  `maintenance_commits_sent`, `maintenance_commits_applied`,
  `slot_commit_drops`, `slot_commit_stale_generation_drops`, and
  `slot_commit_replica_capacity_full_drops`.
- Fail-open/stale/drop counters are visible and bounded.
- Train-rank throughput is not catastrophically coupled to memory work.
- Treat any train-rank synchronous cache read, memory-rank wait, or memory-owned
  collective in the trunk step as a thesis violation.

## Usage

```bash
# Inspect the exact two validation entries without launching.
PYTHONPATH=src .venv/bin/python experiments/26_arm/run_exp26.py --dry-run

# Run the fixed validation canary.
PYTHONPATH=src .venv/bin/python experiments/26_arm/run_exp26.py --budget 45

# Short active-path pulse for wall-clock/telemetry debugging.
PYTHONPATH=src .venv/bin/python experiments/26_arm/profile_exp26.py --budget 15
```

Operational flags remain for paths, world size, seed, budget, and dry-run.
There is no `--stage`, no `--arms`, no CRCT-only mode, and no shadow/headline
matrix path.
