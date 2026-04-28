# Simplex Decision Trace Schema

Status: locked for the Phase-S simplex controller instrumentation.

Purpose: make the controller trace directly ingestible by DuckDB without a
post-pod rewrite. Rows are NDJSON at write time; DuckDB readers should cast
using the column contract below and may partition files by `arm` and a coarse
`gpu_step` bucket.

Recommended path layout:

```text
decision_trace/arm=<arm>/gpu_step_bucket=<floor(gpu_step / 10000)>/part-*.ndjson
```

Required columns:

| Column | DuckDB type | Meaning |
| --- | --- | --- |
| `row_type` | `VARCHAR` | `decision`, `credit`, or `skip`. |
| `status` | `VARCHAR` | `ok` / `skipped`; analysis should not infer failure from missing rows. |
| `status_reason` | `VARCHAR` | Empty for normal rows; otherwise `outcome_status`, `invalid_slot`, `missing_weights`, `missing_decision`, `zero_advantage`, `gerber_rejected`, or `entropy_only` (`gerber_weight=0` but entropy bonus still produced a backward pass). |
| `gpu_step` | `BIGINT` | Producer GPU step for the query/decision; replay step on credit/skip rows. |
| `slot_id` | `UBIGINT` | Chosen cache slot id. |
| `arm` | `VARCHAR` | Matrix arm name; empty string when unavailable in-process. |
| `query_event_id` | `UBIGINT` | Rank-prefixed query id. |
| `replay_id` | `UBIGINT` | Replay/action id derived from query id and chosen rank until controller-issued ids land. |
| `source_write_id` | `UBIGINT` | Original WRITE_EVENT/candidate id for the chosen slot. |
| `selection_step` | `UBIGINT` | Producer step stamped on the decision; credit rows must match this against history. |
| `policy_version` | `UINTEGER` | Behavior policy version. |
| `outcome_status` | `UTINYINT` | Wire replay outcome status; zero on decision rows. |
| `flags` | `USMALLINT` | Reserved replay flags. |
| `write_bucket` | `TINYINT` | Token-frequency bucket for the admitted slot. |
| `slot_age_steps` | `UBIGINT` | `gpu_step - selection_step` on replay rows. |
| `n_actual` | `UINTEGER` | Number of real candidates before sentinel padding. |
| `chosen_idx` | `UTINYINT` | Chosen simplex vertex in the candidate set. |
| `selected_rank` | `UTINYINT` | Alias of `chosen_idx` for wire-event joins. |
| `p_chosen` | `FLOAT` | Behavior-policy probability assigned to the sampled vertex after valid-candidate renormalization. |
| `p_current_chosen` | `FLOAT` | Current-policy probability at replay attribution time; NaN on decision rows. |
| `p_behavior` | `FLOAT[]` | Full behavior distribution over the valid candidates. |
| `entropy` | `FLOAT` | Behavior-policy entropy `-sum(p_behavior * log(p_behavior))`. Always derived from the stored decision snapshot, so decision and credit rows for the same `(query_event_id, replay_id)` report the SAME value. Answers "how exploratory was the controller when it acted?". NaN on skip rows that have no decision snapshot (`outcome_status`, `invalid_slot`, `missing_weights`, `missing_decision`). |
| `current_entropy` | `FLOAT` | Current-policy entropy `-sum(p_current * log(p_current))` from the replay-time forward. NaN on `decision` rows. Populated on `credit` and on skip rows where the forward ran (`zero_advantage`, `gerber_rejected`). The drift `current_entropy - entropy` is a useful stability signal — if current entropy collapses relative to behavior entropy while credit still arrives late, the online learner may be hardening too fast. |
| `temperature` | `FLOAT` | Active simplex softmax temperature. |
| `entropy_beta` | `FLOAT` | Active entropy-bonus weight. |
| `sgd_steps` | `UBIGINT` | Number of learner SGD applications completed before this row was emitted. |
| `ema_blends` | `UBIGINT` | Number of slow-weight EMA blends completed before this row was emitted. |
| `actions_since_sgd` | `UINTEGER` | Credit/entropy-only actions accumulated toward the next SGD application. |
| `gerber_accepted_actions` | `UBIGINT` | Cumulative accepted Gerber-gated reward credits before this row. |
| `gerber_rejected_actions` | `UBIGINT` | Cumulative Gerber-zero events before this row; includes entropy-only updates when `entropy_beta > 0`. |
| `grad_logits_l2` | `FLOAT` | L2 norm of `g_logits` (immediate REINFORCE gradient on the simplex logits) for this event. NaN on decision and skip-without-fwd rows. Use to verify per-event gradient signal magnitude. |
| `grad_w_lh_l2` | `FLOAT` | L2 norm of this event's contribution to `dL/dW_lh` (the head onto policy logits). NaN on decision and skip-without-fwd rows. |
| `grad_w_lh_accum_l2` | `FLOAT` | L2 norm of `grad_weights_.W_lh` (the accumulator across the in-progress SGD batch) AFTER this event's contribution. Resets to ~0 after each `apply_sgd`. NaN on decision and skip-without-fwd rows. Use to verify accumulated gradient over a batch is large enough to actually move weights — the load-bearing diagnostic for the 2026-04-27 v2 "online entropy doesn't move" finding. |
| `w_lh_l2` | `FLOAT` | L2 norm of `fast_weights_.W_lh` (the current head weights). NaN on decision and skip-without-fwd rows. Use to verify cumulative weight drift over a run. |
| `teacher_score` | `FLOAT` | Heuristic score for the chosen slot, usually cosine times utility. |
| `chosen_score` | `FLOAT` | Heuristic candidate score at `chosen_idx`. Answers whether the learned/sample policy picked a high-utility candidate on the same simplex. |
| `chosen_score_gap_to_heuristic` | `FLOAT` | `heuristic_top_score - chosen_score`. Zero means the controller picked the heuristic top candidate; positive means it explored away from the heuristic. |
| `chosen_heuristic_rank` | `BIGINT` | Rank of the chosen candidate by heuristic score, where 0 is best. `-1` when candidate scores are absent. |
| `heuristic_top_idx` | `BIGINT` | Candidate index that the legacy heuristic would choose from this exact candidate set. |
| `heuristic_top_slot_id` | `UBIGINT` | Slot id at `heuristic_top_idx`; 0 only meaningful when `heuristic_top_idx >= 0`. |
| `heuristic_top_score` | `FLOAT` | Best heuristic score among the valid candidates. |
| `candidate_score_mean` | `FLOAT` | Mean heuristic score across valid candidates. |
| `candidate_score_stddev` | `FLOAT` | Standard deviation of heuristic scores across valid candidates. Low values mean the query simplex was nearly indifferent. |
| `candidate_score_margin` | `FLOAT` | Difference between the top two heuristic candidate scores. Small margin means choosing away from the top candidate is less informative. |
| `p_heuristic_top` | `FLOAT` | Behavior-policy probability assigned to the heuristic-top candidate at decision time. |
| `p_current_heuristic_top` | `FLOAT` | Current-policy probability assigned to the heuristic-top candidate at replay attribution time; NaN on decision rows. |
| `controller_logit` | `FLOAT` | Controller logit for the chosen vertex. |
| `ce_before_replay` | `FLOAT` | Replay CE before the optimizer step; NaN on decision-only rows. |
| `ce_after_replay` | `FLOAT` | Replay CE after the replay forward/step; NaN on decision-only rows. |
| `ce_delta_raw` | `FLOAT` | `ce_before_replay - ce_after_replay`; NaN on decision-only rows. |
| `bucket_baseline` | `FLOAT` | Running bucket EMA baseline. |
| `reward_shaped` | `FLOAT` | Raw CE delta minus bucket baseline from the wire event. |
| `gerber_weight` | `FLOAT` | Gerber gate applied at replay attribution; NaN on decision-only rows. |
| `advantage_raw` | `FLOAT` | `ce_delta_raw - bucket_baseline` before standardization; NaN on decision-only rows. |
| `advantage_standardized` | `FLOAT` | Per-bucket standardized advantage before recency/Gerber. |
| `recency_weight` | `FLOAT` | `gamma ** slot_age_steps`. |
| `advantage_pre_gerber` | `FLOAT` | Standardized and recency-decayed advantage before Gerber. |
| `advantage_final` | `FLOAT` | Advantage actually credited to policy after Gerber. |
| `gerber_threshold` | `FLOAT` | Bucket/type-local Gerber threshold. |
| `behavior_logprob_margin` | `FLOAT` | Behavior-policy categorical margin used by Gerber. |
| `current_logprob_margin` | `FLOAT` | Current-policy categorical margin used by Gerber. |
| `lambda_hxh` | `FLOAT` | Active HxH residual scale after warmup/clipping. |
| `selection_mode` | `VARCHAR` | `argmax`, `sample`, or equivalent mode. |
| `selection_seed` | `BIGINT` | CPU generator seed when stochastic selection is used; `-1` when unavailable. |
| `feature_manifest_hash` | `VARCHAR` | Stable hash of the V/E/simplex feature manifest. |

Optional sampled columns:

| Column | DuckDB type | Meaning |
| --- | --- | --- |
| `candidate_slot_ids` | `UBIGINT[]` | Valid candidate slot ids in simplex order. |
| `candidate_scores` | `FLOAT[]` | Heuristic scores in simplex order. |
| `logits` | `FLOAT[]` | Controller logits in simplex order. |
| `V_sample` | `FLOAT[]` | Flattened `(N, K_v)` vertex features, sampled every `simplex_trace_sample_stride` rows. |
| `E_sample` | `FLOAT[]` | Flattened edge features, sampled every `simplex_trace_sample_stride` rows. |

Notes:

- Rows are emitted through a bounded async writer. If the writer falls behind,
  rows drop and `SimplexLearnerTelemetry.simplex_trace_drops` increments; the
  controller/replay path must not wait on disk.
- Gerber statistics are maintained per `(bucket, action_type)`. A global
  statistic may be logged for diagnostics, but it is not the correction used
  by the learner.
- `p_behavior` is always over the valid candidates, not the zero-padded
  vertices. This keeps entropy and Gerber log-prob margins comparable across
  short candidate sets.
- `feature_manifest_hash` changes whenever the meaning or ordering of `V`,
  `E`, or `simplex_features` changes; it is the join key for analysis code
  that reconstructs controller inputs.
