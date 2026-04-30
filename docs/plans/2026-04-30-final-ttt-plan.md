# Final TTT Plan

We are done with experiment sprawl. The final TTT surface should be a small
selector over the locked Exp26 substrate: keep the legal score-only floor,
make packet-clean adaptive carry the default TTT candidate, and use spare eval
budget only if Dreamworld proves both legal and useful.

## Scope

- In: Exp27 calc_type defaults, packet-clean eval-time state adaptation,
  source-order legality checks, focused tests, dry-runs, and Dreamworld
  working/falsification.
- Out: new training architecture, Exp26 substrate redesign, new pods without
  explicit approval, and broad ablation matrices.

## Action Items

- [x] Add `adaptive_carry` as a first-class calc_type.
- [x] Keep `score_only_reset` as the floor and make `adaptive_carry` the
  default Exp27 TTT candidate.
- [x] Ensure `adaptive_carry` uses `encode(memory_mode="packet")` plus
  `lm_head`, not `model.forward()`, so it cannot hit the legacy direct sidecar
  read path.
- [x] Add telemetry for horizon winners, per-head loss, final online weights,
  and actual hyperparameters.
- [x] Align Exp27's training entry with the locked Exp26 384d Adaptive
  Residual Memory substrate instead of the old 256d base config.
- [x] Decide Dreamworld by evidence: make it run in the calc_type harness with
  bounded defaults, or leave it registered but non-default with a written
  falsification reason.
- [x] Verify with focused pytest:
  `tests/test_exp27_calc_type_adaptive_carry.py`,
  `tests/test_exp27_calc_type_dreamworld_eval.py`,
  `tests/test_exp27_ttt_eval_foundation.py`,
  `tests/test_exp27_orchestrator.py`, and
  `tests/test_exp27_runner_dispatch.py`.
- [x] Dry-run Exp27 headline with `score_only_reset adaptive_carry` and, only
  if green, with `dreamworld_eval` explicitly included.
- [x] Commit once the tests and dry-runs are clean.
- [x] Run pod H100 smokes for the default TTT path and Dreamworld.

## Decisions

- `adaptive_carry` is the default TTT path. It is gradient-free but adaptive:
  source-ordered recurrent state and online head weights evolve causally from
  already-scored tokens.
- `dreamworld_eval` is not default until proven. It may use spare eval budget,
  but it must not become a submission-day ambiguity trap. On the 4xH100
  synthetic 256-doc smoke it ran successfully but slightly worsened the floor,
  so it remains registered and opt-in rather than default.
- Exp27 should not silently train the old 256d base trunk. If no checkpoint is
  supplied, it should build from Exp26's locked 384d ARM entry.
- Exp27 checkpoint loading is intentionally fail-loud until
  `runner_fast_path.py` has an explicit load path. A metadata-only
  `checkpoint_path` would be too easy to misread as "evaluated the winner."
- `score_only_reset`, `carry_state`, and `adaptive_carry` score through
  `encode(memory_mode="packet") -> final_norm -> lm_head`, not
  `model.forward()`. The floor and TTT candidates therefore use the same
  packet-clean residual lane as training.
- `run_exp27.py` owns the child `PYTHONPATH` needed by `torchrun`, and passes
  `--val-cache-dir` through `launch.run_matrix_entries` into
  `runner_fast_path.py`.

## Pod Evidence

Pod: `b6f45lkeda3juu`, 4x H100 SXM, torch `2.9.1+cu128`.

Focused pod tests:

```text
55 passed in 8.82s
```

Default adaptive smoke:

```text
result_dir: experiments/27_ttt_headline/results/pod_b6f45lkeda3juu_smoke_adaptive_45s_v4
steps: 64
aggregate_tokens_per_sec: 2013956.4
score_only_reset bpb: 5.2526163
adaptive_carry bpb: 5.2236075
adaptive online_final_weights: [1.0, 0.0, 0.0]
```

Combined 256-doc smoke:

```text
result_dir: experiments/27_ttt_headline/results/pod_b6f45lkeda3juu_smoke_all_10s_256docs
steps: 32
aggregate_tokens_per_sec: 4260050.0
score_only_reset bpb: 5.7405290
adaptive_carry bpb: 5.7289792
dreamworld_eval bpb: 5.7412683
```

The pod did not have `HF_TOKEN`, so these BPB numbers are synthetic ValCache
runtime evidence, not official validation evidence. The auth-gated official
`docs_selected.jsonl` cache still needs to be built before final scoring.

## Commands

```bash
.venv/bin/python -m pytest \
  tests/test_exp27_calc_type_adaptive_carry.py \
  tests/test_exp27_calc_type_carry_state.py \
  tests/test_exp27_calc_type_score_only_reset.py \
  tests/test_exp27_calc_type_dreamworld_eval.py \
  tests/test_exp27_ttt_eval_foundation.py \
  tests/test_exp27_orchestrator.py \
  tests/test_exp27_runner_dispatch.py -q -ra

.venv/bin/python experiments/27_ttt_headline/run_exp27.py \
  --stage headline \
  --dry-run \
  --seeds 1337 \
  --calc-types score_only_reset adaptive_carry
```
