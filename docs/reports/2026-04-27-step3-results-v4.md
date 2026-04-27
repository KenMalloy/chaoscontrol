# Step 3 v4 — Episodic controller matrix with full-val BPB

**Run:** 2026-04-27 17:08–21:05 UTC. Pod `3mlt2x4k1byzyh` (4× H100 SXM, US datacenter). Matrix `episodic_controller_v1` × 6 arms × 3 seeds = 18 cells. Train 600s + Exp 20 fast scorer over 50 000 docs per cell.

**Headline:** every simplex variant (arms c–f) beats the no-controller control (arm_a) by ~0.003 BPB. **arm_f (sharp T=0.2) wins on mean and on variance**: 1.4761 ± 0.0001 vs arm_d's 1.4771 ± 0.0005. Heuristic controller (arm_b) is destructive — 6× throughput loss + 0.16 BPB regression. The bootstrap-fix shows up on val BPB but **not via the predicted L2 channel** — diagnostic columns falsify the "bigger gradients" framing while the BPB confirms the win.

## Per-arm summary

n = 3 seeds (1337, 2674, 4011). Train budget 600s on 4×H100, eval over 50 000 docs.

| arm | mech | steps | initial loss | final loss | BPB mean ± std |
|---|---|---:|---:|---:|---:|
| arm_a | control (no episodic) | 3572 | 9.8649 | 3.7484 ± 0.0427 | **1.4794 ± 0.0012** |
| arm_b | heuristic | 604 | 9.8652 | 3.9620 ± 0.0134 | **1.6435 ± 0.0004** |
| arm_c | simplex frozen | 3503 | 9.8652 | 3.7522 ± 0.0122 | **1.4763 ± 0.0004** |
| arm_d | simplex online (T=1.0) | 3503 | 9.8652 | 3.7448 ± 0.0067 | **1.4771 ± 0.0005** |
| arm_e | simplex warm online | 3504 | 9.8652 | 3.7449 ± 0.0063 | **1.4764 ± 0.0005** |
| arm_f | simplex sharp online (T=0.2) | 3503 | 9.8652 | 3.7510 ± 0.0117 | **1.4761 ± 0.0001** |

Steps row for arm_b shows the throughput hole: ~604 train steps in 600s vs ~3503 for the simplex arms (5.8× slower). Same wall budget, fewer gradient updates → worse BPB.

## Bootstrap-fix A/B (arm_d vs arm_f)

| comparison | Δ BPB | n×σ vs arm_f std | n×σ vs arm_d std |
|---|---:|---:|---:|
| arm_f − arm_d | **−0.0010** | −10× | −2× |
| arm_f − arm_a (lift over control) | −0.0034 | −34× | — |
| arm_d − arm_a (lift over control) | −0.0024 | — | −5× |
| arm_b − arm_a (heuristic vs control) | **+0.1641** | — | — |

arm_f beats arm_d by ~2 std relative to arm_d's run-to-run variance. Tighter than that on arm_f's own variance (10×). On the actual val metric the bootstrap-fix is real but modest at this training budget.

The heuristic arm being +0.16 BPB worse than no-controller is the biggest single number in the table — telemetry-only routing without learned barycentric weights is actively harmful.

## Diagnostic L2 — the bootstrap-fix prediction is wrong

The pre-run hypothesis (validated locally on the microbench) was: arm_f's lower temperature unblocks REINFORCE, so per-event gradient direction is more aligned, accumulator grows faster, weights move more, BPB improves.

The data falsifies the L2 channel:

| metric | arm_c frozen | arm_d default T | arm_e warm | arm_f T=0.2 | f/d ratio |
|---|---:|---:|---:|---:|---:|
| `grad_logits_l2` mean | 0.758 | 0.762 | 0.763 | 0.757 | 0.99 |
| `grad_w_lh_l2` mean | 0.122 | 0.122 | 0.123 | 0.121 | 1.00 |
| `grad_w_lh_accum_l2` mean | 1.684 | **2.273** | 2.600 | **1.957** | **0.86** |
| `grad_w_lh_accum_l2` max | 11.66 | **76.37** | 79.71 | **19.24** | **0.25** |
| `w_lh_l2` std (drift) | 0.00139 | **0.00300** | 0.00191 | **0.00216** | **0.72** |

arm_f's accumulator is *smaller* than arm_d's — opposite of the prediction — and its tail is dramatically smaller (max 19 vs 76). arm_d (default T) has the largest weight drift of the four simplex arms, yet the worst BPB among them.

Reframe: a sharp policy doesn't increase per-event gradient magnitude; it reduces variance. arm_d's high-magnitude gradient outliers (the 76× tail) reflect noise from REINFORCE on near-uniform behavior, not signal — and that noise hurts when SGD applies it. arm_f trades large noisy steps for smaller consistent ones, and the val BPB rewards consistency.

This is also why arm_f's BPB std is 5× tighter than arm_d's (0.0001 vs 0.0005): same dynamic, expressed across seeds.

## Entropy drift

Average `current_entropy − behavior_entropy` on credit rows:

| arm | n credits | mean drift | mean behavior H | mean current H |
|---|---:|---:|---:|---:|
| arm_c frozen | 31 417 | −1e-6 | 2.7675 | 2.7675 |
| arm_d online | 31 514 | −2e-6 | 2.7640 | 2.7639 |
| arm_e warm online | 31 527 | +2e-6 | 2.7663 | 2.7663 |
| arm_f sharp online | 31 515 | −2e-6 | 2.7685 | 2.7685 |

Effectively zero entropy drift across all arms at the 600s budget. The policy is reordering ranks without changing entropy — so "entropy didn't move" is a poor proxy for "policy didn't move." Future SGD-debug should look at rank-order change or per-vertex p-shift, not entropy.

## Telemetry health

All simplex arms: `cuda_stream_enabled=True`, `publish_drops=0`, `drain_errors=0`, sgd batch counts 40–41. Async write path is healthy across the matrix; no controller-pipeline contribution to the BPB spread.

arm_a (control) has no episodic mechanism so push counts are 0. arm_b (heuristic) does push (~600 events) but no SGD because there's no learner.

## Step 5 — read

1. **arm_f is the new base.** T=0.2 wins on mean BPB and variance. Lock the temperature override into the simplex_v1 default config.
2. **arm_b can be deleted.** The heuristic mechanism is uniformly worse than no-controller. Stop running it and remove the arm from future matrices.
3. **arm_c (frozen) is within 1 std of arm_f.** A pretrained controller does most of the work; online RL contributes ≤0.0002 BPB on top. The arm_c-vs-control gap is 75% of arm_f-vs-control. Investment in better offline pretraining likely beats investment in tighter online dynamics.
4. **The diagnostic columns earned their keep.** They prevented us from claiming "bootstrap-fix confirmed via gradient L2" — empirically the channel is variance reduction, not magnitude amplification. Future debug-credit-and-update work should keep these columns.
5. **arm_d and arm_e are now redundant** under arm_f. Default T=1.0 is dominated; warm-online (arm_e) is within 1 std of arm_f and provides no separable mechanism story. Drop both from future runs unless we have a specific question.
6. **Open question for Step 5:** the simplex-vs-control gap is 0.003 BPB (~0.2%). Is that worth carrying the controller infrastructure cost into submission? If the goal is BPB minimization end-to-end, a 0.003 lift is real but small. The more interesting follow-up is **whether we can get a 10× larger lift** by either (a) longer training so the simplex policy has more steps to move, or (b) a better pretrained controller (Exp 25 follow-on) so arm_c's frozen ceiling rises.

## Reproducibility

Pod setup: `scripts/pod_bootstrap.sh` (one command — fetches shards, streams docs, builds val cache, smoke-imports). Matrix command (in `dispatch/from-claude/2026-04-27T1715Z-plan-a-launched-and-bootstrap-shipped.md`):

```bash
EPISODIC_CONTROLLER_V1_WEIGHTS_PATH=/workspace/chaoscontrol/experiments/25_controller_pretrain/simplex_policy_v1.cswg \
  python experiments/24_training_time_bundle/run_exp24.py \
  --matrix episodic_controller_v1 --limit 18 \
  --full-val-score --val-cache-dir /workspace/cache/exp23_val_16384 \
  --val-budget-seconds 600 --no-smoke
```

Diagnostic analysis: `python scripts/analyze_step3_v3_diagnostics.py`. Train results, eval summaries, and traces are committed under `experiments/24_training_time_bundle/results/`.
