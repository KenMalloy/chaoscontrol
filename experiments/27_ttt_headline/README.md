# Exp27 TTT Headline

Test-time-training headline measurement on the winning trunk from exp26.
A single trained trunk per seed runs three calc_types as separate 600s
eval sessions; the spread between sessions tells us which TTT strategies
clear the no-TTT floor.

## Three-stage discipline

**Stage 1 — calibrate.** Probe each calc_type on the winning trunk and
fit hyperparameters (`N`, `K`, `R`, `L`, `steps`) so each session lands
inside `600s - baseline_forward`. **Today this stage is stubbed**: the
probe routine is a follow-up that lands once exp26 settles on a winning
trunk. The stub records sensible defaults so the orchestrator runs
end-to-end.

**Stage 2 — analyze.** `calibrate.analyze` writes
`calibration/manifest.json` with one hyperparameter set per calc_type.
The stub analyzer emits the defaults below; the real analyzer will
read a probe trace and pick measured values.

**Stage 3 — headline.** `build_ttt_headline_matrix` reads the manifest
and emits one entry per seed. Each entry carries the calc_type list and
hyperparams in its config; `runner_fast_path.py` consumes those fields
and dispatches `evaluate_with_calc_types` over a `ValCache` loaded from
`--val-cache-dir`. The trunk lock and training contract carry over from
the exp26 winner.

## Calc_types

| Calc_type | What it does | Source order | Grad |
|---|---|---|---|
| `score_only_reset` | reset SSM state per doc, no params changed; the floor | no | no |
| `carry_state` | SSM state continues across docs (optional decay) | yes | no |
| `dreamworld_eval` | per-doc dream rollout + backward + SGD step | no | yes |

`requires_source_order` and `requires_grad` are the
`@register_calc_type` knobs in `chaoscontrol.eval.ttt_eval`.
`evaluate_with_calc_types` enforces `requires_source_order` via the
`source_order_preserved` kwarg: `ValCache` produced by `write_val_cache`
is always source-ordered (DocStreamer reads JSONL in order), so the
runner passes `source_order_preserved=True`. Future code that shuffles
the cache must pass `False` and the dispatcher will refuse to run any
order-sensitive calc_type. Each calc_type body manages its own grad
scope.

`dreamworld_eval` rejects `per_doc_reset=False` until a separate
continual variant is registered with `requires_source_order=True` —
the default `per_doc_reset=True` is the only supported mode.

A fourth calc_type, `state_replay_within_doc`, was prototyped and
removed because it broke causality: re-passing a doc with state carried
across the full pass leaks future tokens into past positions (pass r=2's
state at position t already integrated tokens t+1..T-1 from pass r=1's
encode). Same pattern as `carry_state` would be without source-order —
just with the doc as the time window. A real causal "depth thinking"
variant would need per-position iterative refinement; that's its own
design conversation, not part of this experiment.

## Reading the matrix

- `score_only_reset` -> `carry_state`: does state-carry across docs help?
- `score_only_reset` -> `dreamworld_eval`: does per-doc self-distill TTT
  help?

The floor is the same checkpoint each calc_type rides on, so deltas
isolate the eval-time strategy from anything training-time.

## Layout

```
experiments/27_ttt_headline/
  README.md
  exp27.py               # matrix builder
  calibrate.py           # stub analyzer + manifest writer
  run_exp27.py           # three-stage orchestrator
  calibration/
    manifest.json        # populated by stage 2 (stub today)
  results/
    matrix.json          # headline matrix
    exp27_ttt_headline_s*.json   # per-seed result with per-calc_type BPB
```

## Usage

```bash
# Full run on 4xH100 (calibrate stub -> analyze -> headline).
python experiments/27_ttt_headline/run_exp27.py --stage all

# Calibrate-only (stub today; previews the manifest the analyzer will write).
python experiments/27_ttt_headline/run_exp27.py --stage calibrate

# Analyze-only (writes calibration/manifest.json from the stub defaults).
python experiments/27_ttt_headline/run_exp27.py --stage analyze

# Headline only (requires a manifest from a prior analyze).
python experiments/27_ttt_headline/run_exp27.py --stage headline

# Restrict the headline to a subset of calc_types.
python experiments/27_ttt_headline/run_exp27.py --stage headline \
    --calc-types score_only_reset carry_state

# Target an existing checkpoint instead of training fresh per seed.
python experiments/27_ttt_headline/run_exp27.py --stage headline \
    --checkpoint-path /path/to/exp26_winner.pt

# Dry-run any stage; prints entries and returns without touching disk.
python experiments/27_ttt_headline/run_exp27.py --stage all --dry-run
```

## What's stubbed

`calibrate.analyze` is a stub. It does not probe any calc_type; it
writes a manifest of default hyperparameters and records `source_trace
= "stub"`. The defaults are:

- `score_only_reset`: `{}`
- `carry_state`: `{"decay": 1.0}`
- `dreamworld_eval`: `{"K": 8, "L": 64, "lr": 0.001, "steps": 1,
  "per_doc_reset": True, "dream_target_mode": "argmax",
  "dream_temperature": 1.0, "prefix_len": 16}`

A real probe routine — time each calc_type at the locked trunk, fit
N/K/R/L/steps inside the 600s eval budget minus the baseline forward,
write the chosen values into `calc_type_hyperparams` — is the follow-up
once exp26 picks a winning trunk.

## Result schema

When the runner is invoked with `--val-cache-dir` and a config that
sets `calc_types`, `result["eval"]` carries the per-calc_type dict
plus a backward-compatible headline:

```jsonc
{
  "eval": {
    "calc_types": {
      "score_only_reset": {"bpb": ..., "loss": ..., "docs_scored": ..., ...},
      "carry_state":      {"bpb": ..., "loss": ..., ...},
      "dreamworld_eval":  {"bpb": ..., "loss": ..., ...}
    },
    "headline_calc_type": "score_only_reset",
    "bpb": <score_only_reset's bpb>,
    "loss": <score_only_reset's loss>
  }
}
```

The headline mirrors the floor calc_type (`score_only_reset`) when it
was requested, otherwise the first calc_type in the requested list.
Downstream tooling that reads `result["eval"]["bpb"]` continues to
work; the calc_type breakdown is additive.

## Pre-launch checklist

The matrix builder, dispatcher, runner integration, and
`requires_source_order` tripwire are wired and tested. Before a real
launch:

1. **Build a `ValCache` on the pod.** `chaoscontrol.eval_stream.val_cache.write_val_cache`
   produces `tokens.npy` / `docs.npy` / `manifest.json` from the FineWeb
   JSONL + the SP model used by the trunk. The runner reads this via
   `--val-cache-dir <path>`.
2. **Build `_ssm_scan._C` on the pod** (see `feedback_nvcc_path_for_ssm_scan`)
   so the diagonal scan path runs at native speed; without it the
   Python fallback fires and a 1024-token doc takes tens of seconds.
3. **Numerical agreement check** against canonical `evaluate_bpb_sp`:
   `score_only_reset` should match the legacy single-BPB on the same
   checkpoint to within the float64 reduction tolerance.
