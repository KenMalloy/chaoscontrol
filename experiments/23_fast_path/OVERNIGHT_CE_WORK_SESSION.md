# Exp23 Overnight CE Work Session

Date: 2026-04-21

## Goal

Push the training hot path toward the long-term moonshot target of 10M tokens/s
per H100, with CE/head work as the first target because it is still a dominant
and high-leverage component.

## Budget Guardrails

- 1xH100: allowed for development validation and profiling.
- 8xH100: hard ceiling of 3 hours total.
- 8xH100 runs require clean local tests, clean 1xH100 CUDA parity, and a
  measured 1xH100 speed improvement or a scaling-specific question that cannot
  be answered on 1x.
- Stop pods whenever not actively building, profiling, or running a gated
  measurement.

## Work Order

1. Refresh profile of the current `fused_streaming` path.
2. Implement CE improvements with test-first local coverage and H100 CUDA
   parity before speed claims.
3. Prefer exact-math fused paths: streaming CE backward, reduced allocator
   churn, and RMSNorm + linear CE integration.
4. Record every meaningful speed result under `experiments/23_fast_path/` and
   commit/push stable milestones.
5. Use Liger/Cut-CE as comparison or fallback only after our direct path stalls
   or needs validation.

## 8x Gate

Run 8x only if one of these is true:

- 1xH100 throughput improves by at least 8% over `fused_streaming`, or
- the code is unchanged but the measurement question is purely multi-GPU
  scaling, such as DDP communication overlap or 8x full Stage A timing.

If the first 8x smoke fails setup, hangs, or underperforms clearly, stop the pod
and bring back artifacts before trying a second run.
