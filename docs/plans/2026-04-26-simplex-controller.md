# Simplex CPU SSM Controller — Implementation Plan

> **Companion to:** `docs/plans/2026-04-26-simplex-controller-design.md`.
> Read that first; this plan assumes the design is locked.

## Phases

Five parallel-safe phases, each scoped to a non-overlapping file set so
subagents can run in worktrees without merge conflicts.

| phase | scope | files (primary) |
|---|---|---|
| **S1** | C++ data model + forward kernel | `src/.../src/simplex_policy.{cpp,h}`, `tests/test_simplex_policy.py` |
| **S2** | C++ controller online learning (REINFORCE backward + SGD + EMA) | `src/.../src/online_learning.{cpp,h}` (rewrite), `tests/test_online_learning_loop.py` (rewrite) |
| **S3** | Wire-event schema bump + producer-side candidate emission | `src/.../src/wire_events.h`, `src/.../src/cpu_ssm_controller.cpp` (binding glue), `experiments/23_fast_path/runner_fast_path.py` (producer) |
| **S4** | Python pretrain pipeline (BC) + CSWG export format bump | `experiments/25_controller_pretrain/pretrain_controller.py` (rewrite), `dump_to_cpp.py`, `tests/test_controller_pretrain.py` |
| **S5** | F1 matrix update + bench rewrite | `experiments/24_training_time_bundle/exp24.py`, `experiments/25_controller_pretrain/bench_amx.py`, `bench_amx.md` |

S1 + S2 + S3 + S4 are independent in their primary file sets. S5 depends on
all of them landing for the bench numbers to mean anything; the matrix arm
update can dispatch in parallel since it only touches `exp24.py`.

## Phase S1 — Simplex policy forward kernel (C++)

### Files

- Create `src/chaoscontrol/kernels/_cpu_ssm_controller/src/simplex_policy.h`
- Create `src/chaoscontrol/kernels/_cpu_ssm_controller/src/simplex_policy.cpp`
- Create `tests/test_simplex_policy.py`
- Modify `setup_ext.py` to add `simplex_policy.cpp` to sources
- Modify `cpu_ssm_controller.cpp` — add pybind binding for the forward call

### Public surface (header)

```cpp
namespace chaoscontrol::simplex {

struct SimplexWeights {
  uint32_t K_v;     // vertex feature dim
  uint32_t K_e;     // edge feature dim per pair (V1 = 1)
  uint32_t K_s;     // simplex feature dim
  uint32_t H;       // hidden dim
  uint32_t N;       // simplex size (V1 = 16)

  std::vector<float> W_vp;         // (K_v, H) row-major
  std::vector<float> b_vp;         // (H,)
  std::vector<float> W_lh;         // (H,) — single-output head
  float b_lh;
  std::vector<float> W_sb;         // (K_s,) — simplex_bias scalar
  float alpha;                     // edge-mixing scalar
  float temperature;               // softmax temperature
  std::vector<float> bucket_embed; // (n_buckets, embed_dim)
};

struct SimplexForwardOutput {
  std::vector<float> logits;       // (N,)
  std::vector<float> p;            // (N,) softmax over logits
  std::vector<float> vertex_h;     // (N, H) — saved for backward
  std::vector<float> mixed_h;      // (N, H) — saved for backward
  std::vector<float> attn;         // (N, N) — saved for backward
};

// Single forward pass for one query. Computes logits + softmax probabilities
// over the N=16 simplex vertices given vertex matrix V[N, K_v] and edge
// matrix E[N, N] (cosine pairs; diagonal = 1).
SimplexForwardOutput simplex_forward(
    const SimplexWeights& weights,
    const std::vector<float>& V,                  // [N * K_v]
    const std::vector<float>& E,                  // [N * N]
    const std::vector<float>& simplex_features    // [K_s]
);

}  // namespace chaoscontrol::simplex
```

### Implementation notes

- Use `at::matmul` (or AMX kernel via `chaoscontrol::amx::amx_bf16_matmul`
  when shapes fit) for the three GEMMs in the design doc:
  1. `V @ W_vp`: `(N, K_v) @ (K_v, H)` — single AMX tile fit
  2. `vertex_h @ vertex_h.T`: `(N, H) @ (H, N)` — single AMX tile fit
  3. `mixed_h @ W_lh`: `(N, H) @ (H, 1)` — N=1 wastes the AMX N-axis but
     correct
- For arm64 / non-AMX builds, fall back to `at::matmul` (fp32 throughout
  on arm64; bf16-cast around the AMX path on SPR).
- GeLU after Layer 1: `0.5 * x * (1 + erf(x / sqrt(2)))`.
- Softmax: stable form (subtract max). Same for the inner `softmax_j(attn_logits[i])`.
- `simplex_bias` scalar adds to all logits uniformly.

### Tests (CPU-only, arm64-validatable)

1. `test_simplex_forward_output_shapes` — V[16, 16], E[16, 16], simplex_features[4].
   Output shapes are logits[16], p[16], p sums to 1.
2. `test_simplex_forward_softmax_stability` — large logits don't overflow;
   p stays in [0, 1].
3. `test_simplex_forward_matches_python_reference` — pure-NumPy reference
   in the test file mirrors the C++ forward; assert agreement at
   atol=1e-4 over a few random seeds.
4. `test_simplex_forward_alpha_zero_recovers_per_vertex` — when alpha=0 and
   the attention is initialized to identity-like weights, the forward
   reduces to per-vertex MLP + softmax. Sanity check.

### Out of scope

- Backward / SGD / EMA — that's S2.
- AMX dispatch from inside the kernel — S1 uses `at::matmul`; the
  AMX path is a follow-up optimization once correctness is pinned.

### Commit

```
ssm_controller: simplex policy forward (V1, three-layer head)
```

## Phase S2 — Online learning (REINFORCE)

### Files

- Modify `src/chaoscontrol/kernels/_cpu_ssm_controller/src/online_learning.h`
- Modify `src/chaoscontrol/kernels/_cpu_ssm_controller/src/online_learning.cpp`
- Rewrite `tests/test_online_learning_loop.py`
- Modify `cpu_ssm_controller.cpp` — pybind for new ActionHistoryEntry shape
- Modify `src/.../src/action_history.{cpp,h}` — schema bump

### ActionHistoryEntry shape (new)

Fields needed for REINFORCE backward:

```cpp
struct ActionHistoryEntry {
  uint32_t action_type;      // 1 = simplex selection (V1)
  uint64_t gpu_step;
  uint32_t policy_version;
  uint32_t chosen_idx;       // 0..N-1
  uint64_t chosen_slot_id;   // for replay-outcome match
  float p_chosen_decision;   // for off-policy correction (V2 uses)
  std::vector<float> V;      // [N * K_v] candidate matrix snapshot
  std::vector<float> E;      // [N * N] edge matrix snapshot
  std::vector<float> simplex_features;  // [K_s]
};
```

The old fields (features, global_state, slot_state) are removed in V1 —
they were per-slot artifacts. The simplex snapshot replaces them.

### OnlineLearningController API delta

```cpp
// Replaces initialize_weights from per-slot version.
void initialize_simplex_weights(const SimplexWeights& weights);

// Replaces record_replay_selection from per-slot version. Stores the
// whole simplex snapshot, not just the chosen vertex.
void record_simplex_decision(
    uint32_t slot_id,           // chosen_slot_id
    uint64_t gpu_step,
    uint32_t policy_version,
    uint32_t chosen_idx,
    float p_chosen_decision,
    std::vector<float> V,
    std::vector<float> E,
    std::vector<float> simplex_features);

// Same wire signal (replay outcome) as before; the consumer is now the
// REINFORCE loss, not per-slot regression.
void on_replay_outcome(const ReplayOutcome& ev);
```

### Backward path

```
on_replay_outcome(ev):
    entry = history.find_match(ev.slot_id, ev.gpu_step)
    if entry is None: skip with telemetry counter

    # Re-run forward to get current p_chosen (V1: skip importance
    # ratio, just use stored p_chosen_decision)
    fwd = simplex_forward(self.fast_weights, entry.V, entry.E,
                          entry.simplex_features)
    chosen = entry.chosen_idx
    p_chosen_now = fwd.p[chosen]

    advantage = ev.ce_delta_raw - ev.bucket_baseline
    advantage *= recency_decay(ev.gpu_step - entry.gpu_step)
    advantage *= gerber_concordance(...)  # existing shaping

    # Loss = -advantage * log p_chosen_now
    # d/dlogits log p_chosen = (1[i=chosen] - p[i])
    grad_logits = p_now.copy()
    grad_logits[chosen] -= 1.0
    grad_logits *= advantage  # scaled gradient

    # Backprop through Layer 3 (logit head):
    grad_mixed_h = grad_logits @ W_lh.T  (shape [N, H])
    grad_W_lh += mixed_h.T @ grad_logits
    grad_b_lh += sum(grad_logits)

    # Backprop through Layer 2 (edge-aware mixing) — chain rule through
    # the residual + softmax_j(attn_logits) + dot product.
    # ... (standard attention backward)

    # Backprop through Layer 1 (vertex projection):
    grad_V_proj = grad_vertex_h @ W_vp.T
    grad_W_vp += V.T @ grad_vertex_h_pre_gelu
    grad_b_vp += sum(grad_vertex_h_pre_gelu, axis=0)

    # Accumulate into grad_weights_; SGD/EMA ticks reuse existing logic.
```

The standard attention-block backward is well-documented; the V1 spec is
intentionally a single attention head + residual so the gradient is small
enough to derive carefully and pin in tests.

### Tests

1. `test_simplex_online_records_history_with_full_snapshot` — verify the
   full V, E, simplex_features are stored on `record_simplex_decision`.
2. `test_simplex_reinforce_pushes_chosen_logit_up_for_positive_advantage` —
   single-event smoke: p[chosen] increases after one SGD step with
   advantage > 0.
3. `test_simplex_reinforce_pushes_chosen_logit_down_for_negative_advantage` —
   converse.
4. `test_simplex_gradient_matches_torch_autograd_reference` — pure-Python
   torch-autograd reference computes the policy gradient for a single
   event; C++ kernel matches at atol=1e-4 across all weights. Hardest test;
   pins correctness.
5. `test_simplex_sgd_apply_and_slow_ema_blend` — multi-event run; verify
   sgd_steps and ema_blends counters increment, fast/slow weights diverge
   then converge after enough steps.
6. `test_simplex_skip_when_history_match_missing` — replay outcome
   without a matching record_simplex_decision → telemetry counter increments,
   no crash.

### Commit

```
ssm_controller: simplex REINFORCE backward + SGD + EMA blend
```

## Phase S3 — Wire-event schema + producer-side emission

### Files

- Modify `src/chaoscontrol/kernels/_cpu_ssm_controller/src/wire_events.h`
- Modify `src/.../src/cpu_ssm_controller.cpp` — pybind dict ↔ struct glue
- Modify `experiments/23_fast_path/runner_fast_path.py` — emit candidate set
- Modify `tests/test_shm_ring_wire_events.py` — verify new fields round-trip
- Modify `tests/test_runner_episodic_controller_wiring.py` — pin producer

### QueryEvent schema bump

Add (under existing `#pragma pack(push, 1)`):

```cpp
struct QueryEvent {
  // ... existing fields ...
  uint64_t candidate_slot_ids[16];  // +128B, sentinel-padded for heuristic-only
  float candidate_cosines[16];      // +64B, edge column vs query (Layer 1 input)
};
// New size: 544 + 128 + 64 = 736B.
```

The receiver builds `E[i, j] = candidate_cosines[i] * candidate_cosines[j]`
or recomputes from the cache (cosine of slot key reps) — design choice
left to S1's forward implementation. V1 stores the precomputed cosines.

### Producer-side emission

In `runner_fast_path.py`'s controller-query path, when emitting a
QueryEvent:

```python
candidates = heuristic_topk(query, K=16)
event = build_query_event_dict(
    # ... existing fields ...
    candidate_slot_ids=[s.slot_id for s in candidates] + [SENTINEL] * (16 - len(candidates)),
    candidate_cosines=[cosine(query, s.key_rep) for s in candidates] + [0.0] * (16 - len(candidates)),
)
```

If fewer than 16 candidates are available, sentinel-pad. The C++ controller
ignores sentinels in the simplex forward (effectively N_actual ≤ 16).

### Backward compatibility

QueryEvent producers that don't populate the new fields (legacy heuristic-only
arms) emit zero-padded values. The C++ side detects all-sentinel and falls
back to single-slot scoring (the V0 path). This lets the simplex pivot land
without breaking the heuristic baseline arm.

### Tests

1. `test_query_event_sizes` — update wire_event_sizes() to reflect 736B.
2. `test_shm_ring_query_event_roundtrip_with_candidates` — push a dict with
   populated `candidate_slot_ids[16]` + `candidate_cosines[16]`, pop and
   verify byte equality.
3. `test_runner_emit_query_event_with_simplex_candidates` — patched
   `heuristic_topk` returns 16 known candidates; runner emits a QueryEvent
   whose payload has those slot_ids in order.

### Commit

```
ssm_controller: QueryEvent schema bump for simplex candidate set
```

## Phase S4 — Pretrain pipeline (BC) + CSWG export

### Files

- Rewrite `experiments/25_controller_pretrain/pretrain_controller.py`
- Modify `experiments/25_controller_pretrain/dump_to_cpp.py` — new CSWG format
- Rewrite `tests/test_controller_pretrain.py` and `tests/test_controller_weight_dump.py`

### Pretrain target

Given heuristic argmax decisions logged offline (admission traces +
synthetic candidate sets reconstructed from cache state at decision time):

```
target = one_hot(heuristic_argmax_idx, num_classes=16)
loss = cross_entropy(simplex_forward(...).logits, target)
```

Train for some N steps until policy_acc on heuristic argmax ≈ 0.7-0.9
(perfect mimicry is overkill — the controller should already start close
to the heuristic but with room to deviate during online training).

### CSWG schema bump

CSWG v3 replaces the duplicated fixed-order v2 layout with a self-describing
binary:

```
fixed header: "CSWG" | version=3 | dtype | manifest_nbytes | reserved
manifest: JSON { dims, tensors: [{name, shape, dtype, offset, nbytes}, ...] }
payload: concatenated fp16 tensor bytes
```

The manifest is load-bearing: HxH tensors (`W_q`, `W_k`, `W_v`, `W_o`,
`W_e`, `lambda_hxh`) are described by name and shape, so the runner can
load base-only and residual-HxH artifacts without a second hard-coded tensor
order.

### Tests

1. `test_simplex_pretrain_synthetic_convergence` — policy_acc on
   heuristic argmax target reaches ≥ 0.7 on 1000 synthetic queries.
2. `test_cswg_v3_header_and_manifest_layout` — header + manifest dims.
3. `test_cswg_v3_manifest_is_self_describing_for_hxh` — residual HxH tensors
   are named and shaped in the manifest.

### Commit

```
controller_pretrain: simplex BC pipeline + CSWG v3 export
```

## Phase S5 — F1 matrix + bench update

### Files

- Modify `experiments/24_training_time_bundle/exp24.py` — replace V1 arms
- Modify `experiments/25_controller_pretrain/bench_amx.py` — drive simplex forward
- Modify `experiments/25_controller_pretrain/bench_amx.md` — interpretation
- Modify `tests/test_exp24_training_bundle.py` — pin new arm structure

### Matrix arms (final)

```
arm_a_control          — episodic off
arm_b_heuristic        — episodic on, heuristic argmax
arm_c_simplex_frozen   — trained simplex, no online SGD
arm_d_simplex_online   — trained simplex, online SGD enabled
arm_e_simplex_warm     — trained simplex, online, ckpt-warm cache
```

5 arms × 3 seeds = 15 cells (down from 18; arm_b_warm collapses since
heuristic doesn't benefit from warm cache differently than cold).

### Bench update

Per-query `simplex_forward` end-to-end timing replaces the per-slot bench
hook. Modes:

- `controller_per_query.scalar` — `set_use_avx512_matops(False)` + at::matmul
- `controller_per_query.avx512` — AVX-512 head + at::matmul layers
- `controller_per_query.amx` — AMX path enabled (SPR only)

### Commit

```
exp24: simplex controller F1 matrix (5 arms x 3 seeds)
bench: per-query simplex forward across three modes
```

## Sequencing

S1 + S3 + S4 dispatch in parallel as worktree subagents. S2 waits for S1
(needs the forward kernel) and S3 (needs the new ActionHistoryEntry shape).
S5 waits on all four.

I do S2 myself sequentially after the parallel three return — it's the
trickiest piece (gradient correctness), and review-after-the-fact for a
new gradient implementation is harder than just doing it carefully.

## Blocker defensive tests

These tests guard against silent invalidation bugs in the simplex
producer -> controller -> replay-credit path. They are blockers before
launching the 5-arm matrix:

1. `test_simplex_replay_credits_when_selection_step_matches_producer_gpu_step`
   in `tests/test_simplex_learner.py`: record a decision at `gpu_step=N`,
   replay with `selection_step=N`, assert exactly one credit and non-zero
   advantage. Paired negative case replays `selection_step=N+1` and asserts
   zero credit.
2. `test_build_simplex_learner_from_cswg_real_artifact` in
   `tests/test_runner_episodic_controller_wiring.py`: synthetic mini-pretrain
   -> CSWG v3 dump -> `_build_simplex_learner_from_cswg(path)` -> assert
   loaded shapes match the CSWG manifest and the learner produces a
   non-uniform forward.
3. `test_run_controller_cycle_simplex_path_credits_a_full_round_trip` in
   `tests/test_episodic_controller.py`: real tiny cache -> real query event
   -> `run_controller_cycle(controller_runtime=learner,
   action_recorder=learner)` -> tag -> matching ReplayOutcome -> assert
   credited.
4. `test_simplex_learner_does_not_credit_sentinel_padded_slots` in
   `tests/test_simplex_learner.py`: record a decision with four real
   candidates plus twelve sentinels, then replay a sentinel/collision slot
   and assert no spurious credit.

## Important defensive tests

These tests are not launch blockers by themselves, but they pin the
architecture's scientific invariants:

1. `test_simplex_policy_reduces_to_heuristic_argmax_under_degenerate_weights`
   in `tests/test_simplex_policy.py`: degenerate weights recover heuristic
   utility argmax, proving the simplex policy is a strict superset of the
   baseline selector.
2. `test_gerber_correction_collapses_to_one_when_behavior_equals_current` in
   `tests/test_simplex_learner.py`: when behavior and current categorical
   margins match, Gerber weight is 1.0 and credit proceeds.
3. `test_sample_mode_is_deterministic_under_fixed_seed` in
   `tests/test_episodic_controller.py`: two `run_controller_cycle` sample
   calls with the same seed produce the same chosen vertex and slot.

## Verification at end

```
.venv/bin/python src/chaoscontrol/kernels/_cpu_ssm_controller/setup_ext.py build_ext --inplace
.venv/bin/python -m pytest tests/test_simplex_policy.py tests/test_online_learning_loop.py tests/test_shm_ring_wire_events.py tests/test_runner_episodic_controller_wiring.py tests/test_controller_pretrain.py tests/test_exp24_training_bundle.py -v --no-header
```

Expected: all simplex tests pass on arm64; hardware-gated AMX tests skip;
matrix tests reflect 5-arm structure.
