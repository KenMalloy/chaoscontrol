"""Shared slot dtypes for the episodic-cache IPC rings.

Both Phase 1 Tasks 1.4 (producer, train-rank side) and 1.5 (consumer,
episodic-rank side) target these dtypes. Defined here as a single source
of truth so the parallel worktrees don't race on the file.

The dtypes are parameterized by `span_length` (cache value-span length)
and `key_rep_dim` (residual dimension; typically `model_dim`). Both come
from runner config; the dtypes are constructed once at runner init and
passed to `ShmRing.create(...)` / `ShmRing.attach(...)`.

See `docs/plans/2026-04-25-ring-contract-tasks-1-4-and-1-5.md` section
"Slot dtypes" for the canonical specification.
"""
from __future__ import annotations

import numpy as np


def make_write_payload_dtype(*, span_length: int, key_rep_dim: int) -> np.dtype:
    """Slot dtype for the per-rank write ring.

    Mirrors `chaoscontrol.optim.episodic_writer.WritePayload`:

      - ``key_fp``: int64 rolling-hash fingerprint of the preceding window
      - ``key_rep``: float32 [key_rep_dim] late-residual at write position
      - ``value_tok_ids``: int64 [span_length] next S target tokens
      - ``value_anchor_id``: int64 (= value_tok_ids[0])
    """
    if span_length <= 0:
        raise ValueError(f"span_length must be positive; got {span_length}")
    if key_rep_dim <= 0:
        raise ValueError(f"key_rep_dim must be positive; got {key_rep_dim}")
    return np.dtype([
        ("key_fp",          np.int64),
        ("key_rep",         np.float32, (key_rep_dim,)),
        ("value_tok_ids",   np.int64,   (span_length,)),
        ("value_anchor_id", np.int64),
    ])


def make_query_candidate_dtype(*, key_rep_dim: int) -> np.dtype:
    """Slot dtype for the per-rank query-candidate ring.

    Phase 2 controller will consume this; Phase 1 only fills it.

      - ``batch_index``: int64 (b in [0, B))
      - ``position``: int64 (t in [0, T))
      - ``pressure``: float32 (pressure × per_token_CE at this position)
      - ``residual``: float32 [key_rep_dim] late-layer residual at (b, t)
    """
    if key_rep_dim <= 0:
        raise ValueError(f"key_rep_dim must be positive; got {key_rep_dim}")
    return np.dtype([
        ("batch_index",  np.int64),
        ("position",     np.int64),
        ("pressure",     np.float32),
        ("residual",     np.float32, (key_rep_dim,)),
    ])
