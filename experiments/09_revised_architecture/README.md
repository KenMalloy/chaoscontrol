# Experiment 09: Revised Architecture

## Hypothesis

A layered test design isolates the contribution of each subsystem (gate mode,
memory, Wernicke routing, regret tracking) to compression quality.  By running
each layer with 3 seeds and picking winners statistically, we avoid
confirmation bias and identify which components are genuinely load-bearing.

The random-gate control fires with the same probability parameter as the
surprise threshold, not matched to the surprise gate's empirical rate.  The
analyzer reports both rates for comparison (via `gate_fired` and
`surprise_would_fire` fields in the step log).  This tests whether surprise
*timing* matters or whether the benefit comes purely from extra compute.

## Null hypothesis

No subsystem contributes beyond its compute cost.  The SSM baseline matches or
beats all gated variants once step counts are normalized.  Memory, Wernicke,
and CFR add parameters without improving bpb.

## Method

Five layers, run sequentially.  Each layer picks the winner (lowest mean bpb
across 3 seeds) and injects its settings into the next layer.

| Layer | Configs | Seeds | What it tests |
|-------|---------|-------|---------------|
| L1    | 7       | 3     | Gate modes: none, fork, MC, MCTS k4/k8, random control |
| L2    | 6       | 3     | Memory: none, episodic, episodic+semantic, full-sequence |
| L3    | 4       | 3     | Wernicke routing, CFR regret, compression consequence |
| L3.5  | 3       | 1     | Dark horses: cross-layer combos the layered design might miss |
| L4    | 4       | 1     | Scaling: dim 128/256/384 + transformer baseline |
| L5    | 3       | 1     | Full A-mode at 128/256/384 (30-minute budget) |
| L6    | 4       | 3     | Inference-time adaptation depth: which memory tiers help at eval |

Total: 28 config templates, 73 runs.

## Predictions

1. At least one gated variant beats both baselines (SSM and transformer)
2. Surprise-gated MCTS beats random-gate control (timing matters)
3. Episodic memory with warmup beats cold-start
4. Semantic tier adds measurable improvement over episodic-only
5. Wernicke with CFR beats plain Wernicke
6. Full stack scales better than transformer at dim=384
7. Deeper inference-time adaptation (episodic + semantic + latent) beats WM-only eval
8. Seeded LTM (compressed memories from training) outperforms cold-start warmup

## Dependencies

- FineWeb dataset (download via `baselines/parameter_golf/cached_challenge_fineweb.py`)
- `pyyaml` for config generation

## Running

```bash
# Full run (default 150s per config, 900s for L5)
./run.sh /path/to/fineweb_data

# Custom budget
./run.sh /path/to/fineweb_data --budget 120

# Resume from a specific layer
python run_layered.py --data-path /path/to/fineweb_data --start-layer 3
```

## Analysis

```bash
python analyze.py
```

Produces per-layer ranked tables, fork rate analysis, latent reactivation
counts, and cross-layer winner progression.

## Kill criteria

- If random-gate control matches surprise-gated: the gate timing claim is dead
- If no memory config beats no-memory: episodic/semantic memory is not load-bearing
- If transformer beats full stack at dim=384: the SSM architecture thesis fails
- If wm_only matches wm_plus_all in L6: deeper memory tiers don't help at inference
