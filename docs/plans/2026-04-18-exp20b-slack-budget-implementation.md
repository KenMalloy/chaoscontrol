# Exp 20b - Slack-Budget TTT Implementation Plan

**Goal:** Make Exp 20 measure the no-TTT eval floor first, then treat only the remaining wall-clock slack as the real TTT budget.

**Architecture:** Add lightweight budget accounting to the existing Exp 20 harness. `run_exp20_eval.py` times scoring and adaptation separately, writes a run-level summary JSON, and stops gradient TTT once the configured slack budget is exhausted.

**Status:** Implemented inline on 2026-04-18.

## Protocol

1. Run the final checkpoint in score-only mode:

```json
{
  "adapt_set": "none",
  "steps_per_chunk": 0,
  "budget_seconds": 600.0,
  "safety_margin_seconds": 30.0,
  "summary_path": "results/exp20b_score_floor_summary.json"
}
```

2. Read `score_floor_seconds` from the summary. In score-only mode this is the whole elapsed eval time, not only model forward time, so tokenization and driver overhead are included.

3. Run TTT candidates with the measured floor:

```json
{
  "score_floor_seconds": 410.0,
  "safety_margin_seconds": 30.0,
  "budget_seconds": 600.0
}
```

4. Compare candidates by bpb gain per slack second:

```text
usable_ttt_budget = budget_seconds - score_floor_seconds - safety_margin_seconds
```

## Files

- `src/chaoscontrol/eval_stream/budget.py`: budget math and summary generation.
- `src/chaoscontrol/eval_stream/types.py`: `RunConfig` fields for `score_floor_seconds`, `safety_margin_seconds`, and `summary_path`.
- `scripts/run_exp20_eval.py`: wall-clock timing, summary writing, and adaptation slack guard.
- `tests/test_eval_stream_budget.py`: unit coverage for slack-budget math.
- `tests/test_run_exp20_eval.py`: integration coverage for score-floor summaries and no-slack adaptation skips.

## Decision Rule

No TTT mechanism gets credit for using the scoring floor. The score-only floor is rent. The real comparison is:

```text
bpb_delta / ttt_budget_used_seconds
```

with `slack_remaining_seconds` kept positive enough to survive submission jitter.
