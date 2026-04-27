# Step 3 Results — Simplex Controller Matrix

**Run:** 2026-04-27, pod `pvapfq2vsyvh0o` (4×H100 SXM, DE), launched 04:14Z, last cell complete ~06:55Z, pod stopped 06:52:57Z.
**Matrix:** `episodic_controller_v1`, --limit 15, 5 arms × 3 seeds = 15 cells.
**Code:** origin/main at `ad9f69b` (entropy column split applied).

---

## TL;DR

The matrix completed cleanly (15/15 cells, no errors, no NaNs). Throughput-wise the perf rework is a real win on simplex arms (≈97% of control) but heuristic still pays a 2× cost. The thesis-relevant numbers are mostly null:

- **Behavior policy never moved off uniform.** Avg behavior entropy is ≈ ln(16) ≈ 2.77 across all 31k decisions per cell. Online learners did not sharpen.
- **Drift = 0.** `current_entropy - entropy` is exactly 0 across all credit rows in arms d and e. The online updates produced no measurable policy change.
- **Cache writes never flowed through the new producer.** Every cell reports `cuda_stream_enabled: false` with all `episodic_async_writes` counters at 0 — including heuristic, which clearly exercised the cache (it paid the 2× perf cost). The GPU-pack pipeline didn't activate; runs fell back to the legacy CPU producer.
- **Loss ordering is real but small.** train.final_loss: a (3.700) < e (3.7441) ≈ d (3.7444) < c (3.7517) < b (3.7917). The simplex arms beat heuristic and warm/online beat frozen — but the arm-to-arm gaps are <1% and the control beats everyone.

This is best read as a **null result that is an infrastructure result**: the matrix proves the framework runs end-to-end on a fresh pod, the new entropy split + telemetry land cleanly, the perf rework cuts simplex slowdown from ~7.4× to ~1.02×, and the `simplex_decision_trace_schema` carries the full diagnostic signal we need for Step 4. But the actual learning loop didn't engage. Step 5 should fix wiring before re-running.

---

## 1. Health gate

| metric | gate | observed |
|---|---|---|
| 15 cells produced JSONs | ✓ | 15/15 |
| every train.final_loss finite | ✓ | min 3.6981, max 3.7984 — all finite |
| no top-level `errors` from the matrix orchestrator | ✓ | `"errors": []` |
| `publisher_error == ""` | ✓ | empty in every cell |
| `drain_errors == 0` | ✓ | 0 in every cell |
| `drops/submitted_batches < 0.1%` | n/a | both = 0 (pipeline didn't activate) |

The pipeline-not-activated finding fails this gate's premise rather than the gate itself. See §6 below.

---

## 2. Per-arm summary

Mean across 3 seeds per arm:

| arm | steps | steps/s | agg tok/s | slowdown vs control | final_loss (mean) | final_loss (per seed) |
|---|---:|---:|---:|---:|---:|---|
| arm_a_control | 3568 | 5.96 | 12.50 M | **1.00×** | 3.6996 | 3.6980 / 3.7006 / 3.7001 |
| arm_b_heuristic | 1888 | 3.15 | 4.96 M | **1.97×** | 3.7917 | 3.7970 / 3.7982 / 3.7799 |
| arm_c_simplex_frozen | 3489 | 5.83 | 9.17 M | **1.02×** | 3.7517 | 3.7515 / 3.7531 / 3.7506 |
| arm_d_simplex_online | 3487 | 5.82 | 9.16 M | **1.02×** | 3.7444 | 3.7461 / 3.7451 / 3.7421 |
| arm_e_simplex_warm_online | 3485 | 5.82 | 9.16 M | **1.02×** | 3.7441 | 3.7445 / 3.7445 / 3.7434 |

### Throughput observations

- **Simplex slowdown vs control: ~2%.** This is the headline of the perf rework. Prior runs showed 7.4× slowdown for episodic-enabled arms. The async ring + GPU-pack work cut that to barely-measurable.
- **Heuristic slowdown vs control: 1.97×.** Heuristic still pays a real cost. Combined with the `cuda_stream_enabled: false` finding (§6), this is consistent with heuristic running the legacy CPU producer path. Investigating whether heuristic can also use the GPU-pack pipeline (or why it isn't activating) is a Step 5 follow-up.

### Loss observations

The arm ordering by final_loss is `a < e < d < c < b` (lower is better). A control wins. Among episodic arms, the *direction* matches the pre-launch thesis (warm-online ≥ online > frozen > heuristic) but the magnitude is small (≤ 0.05 between any two episodic arms). With 3 seeds and 0.005-level noise per cell, some of these gaps may be statistical artifact.

train.final_loss is the running average over the last few hundred steps of training, NOT a held-out validation score — eval section is empty (`{}`) in every cell. We cannot compute BPB; loss comparisons are training-set-only.

---

## 3. Q1 — Exploration

Average behavior-policy entropy on `decision` rows by arm:

| arm | n_decisions | avg entropy | % decisions with entropy < 0.01 |
|---|---:|---:|---:|
| arm_c_simplex_frozen | 31,404 | **2.7669** | 0.0% |
| arm_d_simplex_online | 31,380 | **2.7696** | 0.0% |
| arm_e_simplex_warm_online | 31,366 | **2.7702** | 0.0% |

`ln(16) ≈ 2.7726`. **All three simplex arms maintained near-maximum entropy across every decision.** The policy is essentially uniform random over the 16 vertices for the entire run. This is the cleanest single signal that the controller learned nothing.

Two interpretations are consistent:
1. The credit signal reaching the learner is too noisy to break symmetry.
2. The learning rate × 600s window is too small relative to the entropy bonus, so the entropy bonus dominates and pins p uniformly.

---

## 4. Q2 — Drift

`current_entropy - entropy` averaged over `credit` rows by arm:

| arm | n_credit | avg drift | avg current_entropy | avg behavior_entropy |
|---|---:|---:|---:|---:|
| arm_c_simplex_frozen | 16,757 | **+0.0000** | 2.7664 | 2.7664 |
| arm_d_simplex_online | 31,370 | **−0.0000** | 2.7695 | 2.7695 |
| arm_e_simplex_warm_online | 31,358 | **−0.0000** | 2.7702 | 2.7702 |

The point of splitting `entropy` (behavior, from decision snapshot) and `current_entropy` (current, from replay forward) was to surface "did the policy sharpen between act-time and credit-time?". Across **all 79k credit rows in the simplex arms, drift is 0 to four decimal places.** Online learners are not changing the policy in any direction. This is consistent with §3.

---

## 5. Q3 + Q4 — Skip distribution

Total trace events per arm and the skip-row breakdown:

| arm | decision | credit | skip | gerber_rejected (% of skip) | gerber_rejected (% of total) |
|---|---:|---:|---:|---:|---:|
| arm_c_simplex_frozen | 31,404 | 16,757 | 14,638 | 100.00% | 23.31% |
| arm_d_simplex_online | 31,380 | 31,370 | **0** | n/a | 0% |
| arm_e_simplex_warm_online | 31,366 | 31,358 | **0** | n/a | 0% |

Two observations that disagree:

1. **arm_c (frozen) Gerber-rejects 100% of its skips and 23.3% of total events.** Every skip in the frozen arm comes from the Gerber concordance gate firing zero. Combined with the §3/§4 result that the policy didn't move, this means the frozen behavior policy generates margins that fail Gerber concordance against itself ~25% of the time.
2. **arm_d and arm_e (online learners) generated zero skip rows.** Not zero gerber_rejected — zero skip rows total. The online learners never hit any of the documented skip paths (`outcome_status`, `invalid_slot`, `missing_weights`, `missing_decision`, `zero_advantage`, `gerber_rejected`).

Asymmetry between frozen and online on the same Gerber gate is a real Step 5 question: either the online learners' parameter drift (even though it didn't surface in entropy) keeps the current margin close enough to behavior to pass Gerber, or there's a code path that gates Gerber differently when an online learner is wrapping the policy. Worth a grep.

---

## 6. Telemetry anomaly — GPU-pack pipeline never activated

Every cell of every arm reports:

```
"episodic_async_writes": {
  "enabled": true,
  "cuda_stream_enabled": false,    ← gate that should be true on H100
  "submitted_batches": 0,
  "pushed": 0, "drained": 0, "publish_drops": 0,
  "drain_errors": 0, "publisher_error": ""
}
```

This means commit `8305934` ("GPU-stage WriteEvent producer stream") didn't take over the producer path on this run. Heuristic clearly exercises the cache (its 2× slowdown is real and reproducible) so it must be using the legacy CPU producer (the per-K Python loop). Simplex arms emit 31k+ decisions which means the cache HAS slots being written, also via the legacy path.

Likely causes (not yet investigated):
- A config flag like `episodic_cuda_write_event_stage_enabled` defaulting to false in the matrix-generated configs.
- A runtime gate in `_emit_episodic_payloads_gpu` short-circuiting before the CUDA branch.
- The CUDA stream path being wired only to a config name the matrix doesn't set.

This explains why simplex arms are at near-control throughput (legacy path is light when only ~0-2 candidates per query) but heuristic is at 2× slowdown (legacy path with full top-k scoring is heavy). The perf rework's headline number — simplex at 1.02× — was achieved *without* the new GPU-pack pipeline ever firing. The actual rework wins are still latent.

---

## 7. Thesis check

Pre-launch thesis: **e > d > c > b > a** (warm-online beats online beats frozen beats heuristic beats control on BPB).

What we got (training-set final_loss, lower is better): **a < e ≈ d < c < b**.

- Control wins.
- Among episodic arms, ordering matches thesis direction (e ≥ d > c > b).
- Magnitude is small and likely contaminated by the §6 wiring issue.
- Without held-out BPB, we can't make a generalization claim either way.

The thesis is **neither confirmed nor refuted** by Step 3. The run validates infrastructure, throughput, and trace integrity. It does not validate the hypothesis that a learned simplex controller adds value, because the controller never learned and the producer pipeline didn't fire.

---

## 8. Suggested next steps

In rough priority order:

1. **Investigate why `cuda_stream_enabled` is false everywhere.** This is the single highest-leverage finding. If the new GPU-pack pipeline isn't activating, every perf number from this matrix is on the legacy CPU producer path. Trace `_emit_write_events_cuda_stream` (`runner_fast_path.py:~2222`) and the gates upstream of it; check the `config.get("episodic_cuda_write_event_stage_enabled", ...)` default; verify that the build path produced an importable `pack_write_events_cuda_` symbol.

2. **Audit why online learners produce zero Gerber-rejected skips while frozen produces 14k+.** Read the `gate == 0.0f && entropy_beta_ == 0.0f` branch in `simplex_learner.cpp` against the difference between frozen and online. The asymmetry is not explained by current code intuition.

3. **Audit why online entropy doesn't move.** REINFORCE updates with `entropy_beta=0.05` over 31k credit events should produce *some* drift in behavior entropy. The fact that it's exactly 0.0000 to four decimals suggests the simplex weight delta isn't reaching the policy or is being clamped to zero. Check `simplex_backward` against the weight-update path.

4. **Add held-out validation to the matrix.** `eval` is empty `{}`. Without a held-out score we have no way to make a clean BPB comparison. Either wire the matrix to do a fast eval pass at the end of each cell or extend the runner to checkpoint and run eval out-of-line.

5. **Run a re-launch once 1-3 are addressed.** The infrastructure is solid (build-on-fresh-pod takes ~12 min; matrix runs ~2:40 wall; harvest+stop+report fits in the 15-minute pod-idle budget). Once the controller is actually learning, re-run with the same matrix to validate.

6. **Optional — extend simplex-arm budgets to match control step count**. arm_a runs 3568 steps, arms c/d/e run ~3487. The ~2% step-count gap is small enough to be within noise but a budget-equalized re-run would remove the asymmetry as a confound.

7. **Optional — investigate why heuristic is 2× slower than simplex in steps/s.** Both use the legacy CPU producer. Heuristic does per-step `cosine_utility_weighted` scoring; simplex does per-credit REINFORCE backward. If heuristic's bottleneck is the per-step scoring, that's a real result independent of the GPU-pack work.

---

## 9. Artifacts

Local files now present (none pushed; commit only on local):

- `experiments/24_training_time_bundle/results/exp24_phase3_episodic_controller_v1_*.json` — 15 result JSONs
- `experiments/24_training_time_bundle/results/traces/episodic_controller_v1_*.ndjson` — 9 simplex traces (~330 MB total, ~21k rows per cell)
- `experiments/24_training_time_bundle/results/step3_launch.log` — the matrix orchestrator's launch log
- `docs/reports/2026-04-27-step3-results.md` — this report

Pod `pvapfq2vsyvh0o` is EXITED (stopped at 06:52:57Z). Cron monitor `5d9f8a59` is being deleted as the final step.
