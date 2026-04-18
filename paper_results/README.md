# Paper Results Registry

Machine-readable index of run results tagged for paper-table eligibility.
Append-only JSONL at `paper_results/registry.jsonl`, one record per run,
gated on `confirmatory` vs `exploratory` status.

## Why

Per [`docs/plans/2026-04-17-paper-status.md`](../docs/plans/2026-04-17-paper-status.md)
§"Immediate next steps" item 4: a confirmatory paper table must not
silently mix in exploratory runs. Every result that could end up in a
table registers here first with an explicit status. A future
table-rendering script reads from this file and refuses to include
exploratory records without an explicit opt-in.

The loader lives at `src/chaoscontrol/paper_results.py`. The CLI is
`scripts/paper_results.py`.

## Schema

Each line of `registry.jsonl` is one JSON object:

| Field            | Type       | Who fills it | Description                                   |
|------------------|------------|--------------|-----------------------------------------------|
| `experiment`     | str        | caller       | `"exp19"`, `"exp20"`, `"exp21"`, ...          |
| `condition`      | str        | caller       | Cell name, e.g. `"C_ssm_random"`              |
| `seed`           | int        | caller       | RNG seed                                      |
| `status`         | enum       | caller       | `"confirmatory"` or `"exploratory"`           |
| `metrics`        | dict       | caller       | e.g. `{"bpb": 1.492, "wall_clock_s": 602.5}`  |
| `config_hash`    | str        | caller       | Hash of the full run config                   |
| `artifacts`      | list[str]  | caller       | Paths to checkpoints / result JSONs           |
| `extras`         | dict       | caller       | Experiment-specific fields                    |
| `git_sha`        | str        | auto         | Commit SHA at register time                   |
| `git_dirty`      | bool       | auto         | Working tree dirty at register time           |
| `timestamp`      | str        | auto         | ISO-8601 UTC                                  |
| `schema_version` | int        | auto         | Currently `1`                                 |

Uniqueness key: `(experiment, condition, seed, status)`. `verify`
refuses any registry with duplicate keys.

## Usage

From Python:

```python
from chaoscontrol.paper_results import register

register(
    experiment="exp21",
    condition="C_ssm_random",
    seed=1337,
    status="confirmatory",
    metrics={"bpb": 1.492, "wall_clock_s": 602.5},
    config_hash="sha256:abc123",
    artifacts=["experiments/21_sgns_tokenizer/results/four_cell_bpb.json"],
)
```

From the CLI:

```bash
python scripts/paper_results.py register \
    --experiment exp21 --condition C_ssm_random --seed 1337 \
    --status confirmatory --config-hash sha256:abc123 \
    --metrics '{"bpb": 1.492, "wall_clock_s": 602.5}'

python scripts/paper_results.py verify
python scripts/paper_results.py query --experiment exp21 --status confirmatory
```

## Rules

- `git_dirty=True` records are accepted but flagged by `verify`. A
  paper-final table rendering should refuse them.
- `status="exploratory"` is the right label for shake-out runs, LR
  sweeps that inform design but aren't reproduced for the paper, and
  anything from `experiments/09_revised_architecture/` per the
  exploratory-only stamp on those scripts.
- `status="confirmatory"` requires: the run config is frozen in the
  plan, the seed is in the pre-declared seed set, and the result is
  one we'd defend in print.
