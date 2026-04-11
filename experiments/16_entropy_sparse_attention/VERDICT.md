# Experiment 16 Verdict: Informative No-Go

## Result

Phase A probe completed. 42 runs across 6 conditions × 7 seeds.

**All four gates pass** for the best condition (oracle_buf64_k8):
selector mass@k ≥ 0.60, beats recent_k, beats token_keyed,
effective_connections ≤ 2k.

**But the central hypothesis is falsified:**

| Feature source | mean mass@k | eff_conn | interpretation |
|---|---|---|---|
| x_state | 0.735 | 1.27 | baseline (residual + recurrence) |
| x_only | 0.724 | 1.20 | **matches or beats x_state** |
| state_only | 0.636 | 25.4 | diffuse, much worse |

- x_only vs x_state: delta = -0.043 (p=0.017). Adding recurrence
  state slightly **hurts** the selector.
- state_only vs x_state: delta = +0.051 (p=0.009). Recurrence state
  alone is significantly worse.
- The selector works. The attention is naturally sparse. But the
  recurrence state adds no retrieval value beyond the residual stream.

## What We Learned

1. **Attention over SP8192 tokens is extremely concentrated.**
   Effective connections ~1.2 means the dense proxy puts almost all
   mass on 1-2 positions. 8 candidates capture 78% of mass.

2. **Token identity is not enough.** token_keyed captures only ~22%
   of mass. The selector needs content-addressed matching, not
   identity lookup. Exp 14's typed-buffer failure was not just about
   unstable keys — the retrieval problem is genuinely content-addressed.

3. **The residual stream is the retrieval signal.** Post-SSM features
   (embedding + feedforward path) carry enough information for
   retrieval. The recurrence state is too compressed — it remembers
   everything vaguely (eff_conn=25) and nothing precisely.

4. **The SSM backbone is now viable.** torch.compile on the diag
   recurrence gives ~32x speedup (14K steps in 600s vs 447 before).
   bare_bpb = 1.63, down from 1.97 with the old sequential loop.

## What This Means for the Architecture

- Stop investing in "SSM state as selector/oracle."
- The fast SSM backbone is a keeper.
- Short-range sparse retrieval is real and worth integrating, but
  the selector should use post-SSM residual features, not recurrence.
- The natural next step is local attention over a small window on
  top of the fast SSM, not state-oracle sparse retrieval.

## Phase B Decision

**No-go** for the originally planned Phase B (online sparse attention
using recurrence-state oracle). The oracle hypothesis is falsified.

Pivot to Exp 17: local attention hybrid on fast SP-SSM.
