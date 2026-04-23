# Exp 24 Phase 0b Entropy-Gated `log_a` Reread Preregistration

Frozen base references:

- Base-lock decision: `experiments/24_training_time_bundle/PHASE0_BASE_LOCK.md`
- Locked config: `experiments/24_training_time_bundle/configs/exp24_base.yaml`
- Confirm winner checkpoints:
  - `experiments/24_training_time_bundle/phase0_confirm_4x_20260423T044539Z/checkpoints/exp24_phase0_confirm_B_fs_i64a025_dw_c16i16_w010_s1337.pt`
  - `experiments/24_training_time_bundle/phase0_confirm_4x_20260423T044539Z/checkpoints/exp24_phase0_confirm_B_fs_i64a025_dw_c16i16_w010_s2674.pt`
  - `experiments/24_training_time_bundle/phase0_confirm_4x_20260423T044539Z/checkpoints/exp24_phase0_confirm_B_fs_i64a025_dw_c16i16_w010_s4011.pt`

This document freezes the Phase 0b thesis and evaluation contract before any
implementation. Phase 0b is not implemented or evaluated in the Phase 0
base-lock plan.

## Mechanism

At a chunk boundary, use only information available before the next chunk is
scored. If the previous/base predictive entropy exceeds a frozen threshold,
replay a short suffix with both a longer-memory and shorter-memory
`DeltaModulator(log_a_shift=s)`, choose or weight the direction causally, blend
the selected reread boundary state into the base boundary state, and score the
next chunk from that blended state.

Chunk-level gating is preferred for v0 because it makes leakage review simple.
Token-level gating is allowed only if the entropy used for token `t` is
computed before seeing the target token scored at `t`.

## Calibration Protocol

Calibration uses a held-out stream that is not the primary full-validation
scoring stream.

- Entropy statistic: raw mean entropy or z-scored entropy over the previous
  chunk.
- Fire-rate thresholds: choose from target rates `{0.05, 0.10, 0.20}`. This
  sets sensitivity, not a schedule.
- Suffix length `K`: choose from `{32, 64}` tokens. `128` is allowed only as a
  boundary diagnostic.
- Shift pair: choose a symmetric or near-symmetric pair, such as
  `(-0.1, +0.1)` or `(-0.2, +0.2)`. Do not choose a single primary direction.
- Blend policy: freeze a soft mix rule, such as constant `beta` or
  `beta = clamp((entropy_z - tau) / width, 0, beta_max)`.
- Direction policy: either a target-free candidate entropy/margin picker before
  scoring, or a causal next-step-loss update after scoring that affects only
  subsequent state.

Do not tune thresholds or shifts on the primary full-validation stream. If that
happens, mark the run exploratory and rerun with frozen settings.

## Compute-Matched Controls

Run the locked checkpoint under the addendum's compute-matched control set.
Budget `B_reread = N_fire x K x 2` is measured from the primary
`entropy_reread_bidirectional_blend` run first; each control then reproduces the
same total reread-token count.

1. `score_only`: no reread. `B_reread = 0`.
2. `scheduled_reread_compute_matched`: scheduled single-pass rereads at
   `2 x N_fire` fires x `K` tokens. Matches `B_reread`. Isolates gate type
   at matched compute.
3. `entropy_reread_shift0_compute_matched`: entropy gate at `N_fire` fires x
   2 redundant passes at `log_a_shift = 0.0`. Matches `B_reread`. Isolates
   decay-shift effect from refresh compute.
4. `entropy_reread_bidirectional_blend`: the primary. `N_fire` fires x 2
   passes (`+log_a_shift`, `-log_a_shift`) x `K` tokens = `B_reread`.
5. Ablation only, not budget-matched: fixed longer-memory shift and fixed
   shorter-memory shift. Single pass per fire. Run these only if the direction
   picker in control 4 underperforms.
6. Ablation only, not budget-matched: always-on best single shift. Run only if
   controls 2-4 all fail to promote.

## Legality Contract

The implementation must have a regression test proving current-chunk targets
cannot affect the decision for current-chunk scoring.

```python
# Legal: target-free confidence controls the next chunk.
gate_next = entropy(logits_for_previous_chunk) > threshold

# Legal: target-free candidate signal chooses the state before next scoring.
direction = choose_by_entropy_or_margin(long_state, short_state)
state_next = lerp(base_state, selected_reread_state, beta)

# Legal if lagged: current target loss may choose future carried state only
# after the current score has already been recorded.
winner_for_future = argmin(nll_long_current, nll_short_current)

# Invalid for primary reporting: realized target loss controls the chunk whose
# score includes that target.
gate_current = nll(logits_current, targets_current) > threshold
```

Surprise/loss may be logged for analysis and may feed a lagged direction
update, but it must not retroactively choose the score for the target that
produced that loss.

## Success Criteria

Promote `entropy_reread_bidirectional_blend` only if all four hold:

1. Beats `score_only` floor by `>= 0.015 BPB` mean over 3 seeds.
2. Beats `scheduled_reread_compute_matched` by `>= 0.005 BPB` mean over 3
   seeds.
3. Beats `entropy_reread_shift0_compute_matched` by `>= 0.005 BPB` mean over 3
   seeds.
4. Seed-to-seed stddev `< 0.01 BPB` across the 3 seeds.

Shelve and record consistent-with-Exp-20 if the primary is within `0.005 BPB`
of `score_only` either direction, or beats `scheduled_reread_compute_matched`
by `< 0.002 BPB`.

Mark ambiguous, do not promote, and design a sharper follow-up if the primary
beats `score_only` by `0.005` to `0.015 BPB` but fails any other promotion
gate.

## Operational Constraints

- Primary must fire on `<= 20%` of chunks unless the eval budget has obvious
  slack after a full pass.
- All controls and the primary must stay within the `600s` eval-time accounting
  rule. Any arm that exhausts the eval budget before finishing the stream is a
  failed feasibility check, not a result.
- Fixed-direction diagnostic arms do not gate promotion.
- Deferred-blend secondary is evaluated on the same four thresholds against its
  own matched-budget control set and promotes only if the primary also promotes.

Kill or park the mechanism if same-horizon reread matches it, if the best
always-on single shift is faster and better, if soft blending collapses to
replacement/zero-blend across most triggers, or if any leakage review finds
target-dependent current-chunk gating.
