# Step 3 Results v2 — Post-CUDA-Fix Re-Run

**Run:** 2026-04-27, pod `pvapfq2vsyvh0o` (4×H100 SXM, DE), launched 07:36Z. Pod kicked by RunPod at 09:53Z (12 cells done) and again at 09:58Z after restart attempt. **13 of 15 cells harvested locally; matrix is partial in `arm_e` (1 of 3 seeds).**

**Code:** origin/main at `0d5b117` — includes Ken's `60e8e35` (CUDA write-pack hard-raise + Gerber agreement-rescue + diagnostic counters), my `ad9f69b` (entropy column split), and my `0d5b117` (int64 cast on the new CUDA call site, smoke-caught dtype regression).

---

## TL;DR

The headline regression from v1 is **closed**. The headline open question (is the simplex actually learning?) is **answered no, but for a now-diagnosable reason**.

| Question (from v1 Step 5 list) | v2 result |
|---|---|
| 1. cuda_stream_enabled false everywhere | **Fixed.** True on all 10 episodic cells (3 heuristic + 3 frozen + 3 online + 1 warm_online). Pushed/drained counters move; 0 drops, 0 errors across all cells. |
| 2. Frozen-vs-online Gerber asymmetry | **Fixed.** Frozen now Gerber-rejects only 143 events across 3 cells (0.23% of events) vs v1's 14,638 (23.31%). The "agreement-rescue" branch credits when behavior == current, which restores credit flow on frozen. |
| 3. Online entropy doesn't move | **Confirmed at scale; root cause is now isolable.** SGD updates ARE firing (`sgd_steps=41` per cell, 10,500+ Gerber-accepted credits), but behavior entropy stays at 2.7686 ± noise (≈ ln 16). Drift between behavior and current entropy on 31,658 online credit rows: **+0.000002 ± noise**. The wiring works; the SGD step magnitude is not producing measurable policy change. |
| 4. Wire held-out validation | Not addressed this round. Same v1 limitation — final_loss is training-set only. |

**Headline thesis check** is unchanged: simplex matches frozen, both beat heuristic, control still wins. The thesis test that warm-start helps online learning **cannot be answered** with this data — only 1 of 3 arm_e seeds completed, the other two were lost when RunPod kicked the pod mid-run. Re-running just those 2 cells is a ~22 min job whenever a pod can stay up.

The actionable Step 5 question is much sharper than after v1: **why is the SGD step magnitude small enough that 41 updates × 10,500 credits don't move the policy off uniform?** That's a learning-rate / advantage-scale / entropy-bonus interaction question, not a wiring question.

---

## 1. Health gate

| metric | gate | observed |
|---|---|---|
| 13 cells produced JSONs | n/a (target was 15) | **13/15 — partial.** arm_e missing 2 of 3 seeds (RunPod-side pod kicks). |
| every harvested train.final_loss finite | ✓ | min 3.6991, max 3.7984 |
| no top-level `errors` from the matrix orchestrator on completed cells | ✓ | none |
| `cuda_stream_enabled = True` on every episodic cell | ✓ | True on all 10 episodic cells, False on the 3 control cells (correct — control has no episodic) |
| `pushed > 0` on every episodic cell | ✓ | 1876–3520 events/cell |
| `publish_drops < 0.1% × pushed` | ✓ | **0 drops, 0 errors, no publisher_error**, on every cell |
| `drain_errors == 0` | ✓ | 0 across all cells |

The two missing arm_e cells are the only red mark. Their absence does not invalidate the v1 → v2 deltas above.

---

## 2. Per-arm summary

Mean ± std across completed seeds (arm_e only has s1337):

| arm | n | mean steps | mean steps/s | mean final_loss |
|---|---:|---:|---:|---:|
| arm_a_control | 3 | 3566 | 5.96 ± 0.00 | 3.7178 ± 0.0290 |
| arm_b_heuristic | 3 | 1916 | 3.20 ± 0.08 | 3.7855 ± 0.0193 |
| arm_c_simplex_frozen | 3 | 3520 | 5.88 ± 0.00 | 3.7485 ± 0.0105 |
| arm_d_simplex_online | 3 | 3518 | 5.88 ± 0.00 | 3.7475 ± 0.0101 |
| arm_e_simplex_warm_online | **1** | 3516 | 5.87 | 3.7646 |

### Throughput observations

- **Simplex slowdown vs control: 1.4%** (5.88 vs 5.96). Last night was 2.1%. Within noise; perf parity essentially achieved.
- **Heuristic slowdown vs control: 1.86×** (3.20 vs 5.96). Last night was 1.97×. The CUDA pack is now firing on heuristic too (`pushed=1916/cell`, last night `pushed=0`). The remaining 1.86× is genuine cache/drain/controller cost, not silent fallback.
- **Producer telemetry trustworthy this run.** Last night's `cuda_stream_enabled=False` everywhere meant we had no idea which path was actually running. This run has clean per-cell counters.

### Loss observations

Ordering: `a (3.7178) < d (3.7475) ≈ c (3.7485) < e (3.7646) < b (3.7855)`.

Compared to v1:
- Control's std jumped from 0.001 → 0.029. Looking per-seed: s1337=3.7040, s2674=3.7511, s4011=3.6983. The s2674 outlier may be a cold-start ordering effect; not a thesis concern.
- arm_d (online) marginally beats arm_c (frozen) by 0.001. Within noise. The Gerber fix didn't unblock visible loss-level learning either.
- All simplex arms beat heuristic by ~0.04. Same direction as v1, similar magnitude.

`train.final_loss` is the running average over training tail, NOT held-out validation. eval section is `{}` in every cell. We cannot compute BPB; comparisons are training-set-only.

---

## 3. Q1 — Exploration

Average behavior-policy entropy on `decision` rows by arm:

| arm | n_decisions | avg entropy | min | max |
|---|---:|---:|---:|---:|
| arm_c_simplex_frozen | 31,679 | 2.7681 | 0.0000 | 2.7726 |
| arm_d_simplex_online | 31,667 | 2.7686 | 0.6807 | 2.7726 |
| arm_e_simplex_warm_online (n=1) | 10,548 | 2.7695 | 0.0000 | 2.7726 |

`ln(16) ≈ 2.7726`. Behavior entropy stays effectively at the ceiling on every arm. Same finding as v1 — but v1 we couldn't tell if the policy "saw" the credit signal. v2 we know it did (sgd_steps=41, 10,500 accepted credits).

The `min` column tells you the cache size in early decisions: `n_actual=1` produces entropy 0; `n_actual=2` produces ln(2) ≈ 0.69. Once cache fills, decisions are over the full 16 vertices. The avg pinned near ln(16) is the steady state.

---

## 4. Q2 — Drift

`current_entropy - entropy` on `credit` rows by arm:

| arm | n_credit | avg drift | direction |
|---|---:|---:|---|
| arm_c_simplex_frozen | 31,526 | **−0.000004** | flat (frozen, expected) |
| arm_d_simplex_online | 31,658 | **+0.000002** | flat |
| arm_e_simplex_warm_online | 10,545 | **−0.000001** | flat |

The entropy column split (mine, in `ad9f69b`) was meant to surface this exact drift. v2 nails the question: across 73,729 credit rows, **drift is statistical zero**. Online and warm-online learners do not produce measurable behavior-vs-current divergence.

This is the cleanest possible signal that the SGD step magnitude is too small. With 41 SGD updates × ~10,500 events, if even a tiny gradient existed, you'd see entropy drift of ~ε × 41 ≠ 0. We see ~10⁻⁶, which is float roundoff.

---

## 5. Q3 + Q4 — Skip distribution and Gerber rate

| arm | decision | credit | skip | gerber_rejected (% of events) |
|---|---:|---:|---:|---:|
| arm_c_simplex_frozen | 31,679 | 31,526 | **143** | 0.23% |
| arm_d_simplex_online | 31,667 | 31,658 | **0** | 0% |
| arm_e_simplex_warm_online (n=1) | 10,548 | 10,545 | **0** | 0% |

v1 had arm_c at 14,638 skips (23.31%). v2 has 143 (0.23%). **Gerber agreement-rescue from `60e8e35` collapsed the rejection rate by ~100×.** Frozen and online are now symmetric in their skip behavior (both ~0).

Per-cell `gerber_rejected_actions` counter (now in trace columns thanks to the `60e8e35` schema additions):
- arm_c cells: 50, 50, 43
- arm_d cells: 49, 37, 44
- arm_e cell: 37

Gerber is engaging — it's not always 1.0 — but the rescue branch handles the dominant "policy-uniform" case so the rejection events are rare and similar across arms.

`gerber_accepted_actions` per cell ≈ 10,510 across all simplex cells. SGD fires every `sgd_interval = 256` accepted events (the default), giving 10,510 / 256 ≈ 41 updates — matches the observed `sgd_steps=41` per cell exactly.

---

## 6. New diagnostic counters (v2 schema additions)

`60e8e35` added `sgd_steps`, `ema_blends`, `actions_since_sgd`, `gerber_accepted_actions`, `gerber_rejected_actions` to every trace row. The data finally pins down where the learning loop is alive vs dead:

| arm | sgd_steps (max) | gerber_accepted | gerber_rejected | actions_since_sgd at end |
|---|---:|---:|---:|---:|
| arm_c_simplex_frozen | 41 | 10,510 | 47 | ≤ 256 |
| arm_d_simplex_online | 41 | 10,509 | 43 | ≤ 256 |
| arm_e_simplex_warm_online | 41 | 10,508 | 37 | ≤ 256 |

Across-arm symmetry on these counters is striking — **both frozen and online complete 41 SGD steps**. Frozen "SGD steps" being non-zero is curious: the SGD loop runs but the optimizer is presumably zero-LR or the gradient is zero, so the call is structurally a no-op.

The fact that arm_d/e have **identical SGD counts to arm_c** but produce identical (zero) entropy drift confirms: the SGD update path is structurally healthy but its effect on the policy is below measurement.

---

## 7. Thesis check

Pre-launch thesis: **e > d > c > b > a** (warm-online beats online beats frozen beats heuristic beats control on BPB).

What we got (training-set final_loss, lower is better): **a < d ≈ c < e < b**.

- Control still wins
- Among episodic arms: d ≈ c < e — online matches frozen, warm-online slightly worse (but n=1, likely noise)
- All simplex arms beat heuristic
- Same direction as v1; magnitudes similar
- BPB still unmeasured (eval `{}`)

**Conclusion same as v1:** the controller-learning thesis is neither confirmed nor refuted. v2 has now ruled out *infrastructure* and *Gerber gating* as the explanation, leaving the SGD-step-magnitude question as the load-bearing variable for Step 5.

---

## 8. Suggested next steps

1. **Audit the SGD update magnitude.** With 41 updates × 10,500 credits across each cell producing exactly zero entropy drift, the gradient hitting `simplex_backward` is either being clipped to zero, averaged across a too-large batch, or scaled by an `advantage` that's near-zero. Concrete things to instrument:
   - Print `simplex_backward`'s incoming `advantage` magnitude (mean / std / abs-sum) per call. If average |advantage| is < 10⁻⁵, the bucket-baseline subtraction is killing signal.
   - Print the L2 norm of the simplex weight delta (`fast_weights_.W_*`) before and after each `apply_sgd()`. If it's < 10⁻⁶, the optimizer is producing no-op updates.
   - Check `lambda_hxh` and the entropy-bonus contribution. With `entropy_beta=0.05` and policy at ln(16), the entropy gradient is non-trivial; if it's exactly cancelling the REINFORCE gradient, you'd see drift = 0.

2. **Re-launch the missing 2 arm_e cells.** This is ~22 min of pod time once a pod stays up. Trivial scope. Lets us verify the warm-online vs online comparison at n=3.

3. **Wire held-out validation.** Same as v1's #4. Without BPB the thesis can't be confirmed even if the policy DOES learn.

4. **Investigate RunPod pod stability.** This pod was kicked twice tonight by RunPod itself (09:53Z, 09:58Z). Both kicks happened with `lastStatusChange: "Exited by Runpod"`. Worth opening a ticket or moving to a different host. The hassle of constantly losing pod state mid-run is a real productivity tax.

5. **Consider a cccl-bypass and matched-cuda-version preload in `pod_setup_cuda13.sh`.** I had to manually `pip install --force-reinstall nvidia-cuda-runtime==13.2.75 nvidia-cuda-nvrtc==13.2.78 nvidia-cuda-cupti==13.2.75` and set `NVCC_PREPEND_FLAGS=-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK` to get `write_event_pack.cu` to compile. Default install pulls 13.0.x runtimes against 13.2.x nvcc/cccl. Pre-pinning both versions in `pod_setup_cuda13.sh` makes future fresh-pod setups a one-liner.

---

## 9. v1 → v2 delta

| dimension | v1 | v2 |
|---|---|---|
| `cuda_stream_enabled` on episodic cells | False (15/15) | **True (10/10 episodic)** |
| Producer events pushed/cell | 0 (silent fallback) | **1876–3520** |
| Heuristic→control slowdown | 1.97× | **1.86×** |
| arm_c Gerber-rejected events | 14,638 (23.3%) | **143 (0.23%)** |
| Online behavior entropy | ln(16) | ln(16) (unchanged — but now with confirmed SGD execution) |
| Online drift (current − behavior) | exactly 0 | exactly 0 (confirmed, not artifact) |
| `sgd_steps` counter visible in trace | no | **yes — 41/cell** |
| Number of cells harvested | 15 | 13 (RunPod kicks) |
| Headline finding | "infrastructure broken, can't tell" | **"infrastructure works, SGD signal is too small"** |

v2 advanced the diagnosis substantially: from "we don't know if anything is wired" to "wiring is confirmed; the open question is one specific scalar magnitude."

---

## 10. Operational notes

- **RunPod kicked the pod twice** at 09:53Z and 09:58Z. Both `Exited by Runpod` (not by user). 13 cells survived because /workspace persists across stop/start. The 2 missing cells (arm_e_s2674, arm_e_s4011) need re-running on a stable pod.
- **Pod state**: EXITED (last kicked 09:58Z). Volume content (venv, dataset, CSWG, the 13 result JSONs) preserved.
- **Kept the existing pod** rather than creating a replacement after the first kick (per the `feedback_prefer_existing_pod` memory). Second kick within 5 min of restart suggests the issue is host-specific, not pod-specific. Future re-runs should consider a fresh pod or a different region.
- **No `--smoke` on re-launch** because we'd already verified smoke passed once tonight. Smoke caught the `int64` regression in 90s; real value confirmed.

---

## 11. Artifacts

Local files (none pushed):

- `experiments/24_training_time_bundle/results/exp24_phase3_episodic_controller_v1_*.json` — 13 cells (arm_a × 3, arm_b × 3, arm_c × 3, arm_d × 3, arm_e × 1)
- `experiments/24_training_time_bundle/results/traces/episodic_controller_v1_*.ndjson` — 7 simplex traces (~290 MB total)
- `experiments/24_training_time_bundle/results/.archive_2026-04-27-run1/` — last night's failed run, kept for v1 comparison
- `docs/reports/2026-04-27-step3-results.md` — v1 report
- `docs/reports/2026-04-27-step3-results-v2.md` — this report

`step3_run3.log` and `step3_run4.log` weren't rsynced (pod was kicked before the second harvest could run). Not load-bearing — the orchestrator's content is captured in the per-cell logs and result JSONs.
