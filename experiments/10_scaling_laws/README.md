# Experiment 10: SESSM Scaling Laws

Does the SESSM scale more parameter-efficiently than a transformer under the 16MB constraint?

## Conditions

| Config | Architecture | Purpose |
|--------|-------------|---------|
| `bare_ssm` | SSM (diag) | Pure SSM baseline |
| `full_ssm` | SSM (diag) + exp09 winners | Full bio stack |
| `our_tfm` | SimpleTransformerLM | Transformer baseline |
| `mamba2_ssm` | Mamba-2 | SSM-vs-SSM comparison |
| `comp_tfm` | Competition baseline | Reference (XL only) |

## Sizes

XS (64d) / S (128d) / M (256d) / L (384d) / XL (512d)

## Run

```bash
bash run.sh /path/to/fineweb --budget 600
```

## Analysis

```bash
python analyze_scaling.py
```

Produces 9 plots in `plots/`. See `docs/plans/2026-04-07-experiment-10-scaling-laws.md` for full design.

## Depends on

Experiment 09 results (winning component stack for `full_ssm`).
