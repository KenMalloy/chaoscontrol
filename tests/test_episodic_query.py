"""Tests for the cosine-utility-weighted top-K query helper (Phase 2 Task 2.3).

The query helper is the read side of the episodic cache. It runs on the
episodic rank's GPU and returns slot indices ranked by:

  - ``score_mode="cosine_utility_weighted"`` (Decision 0.2):
        score(i) = cosine_sim(query_residual, cache.key_rep[i]) * cache.utility_u[i]

  - ``score_mode="pressure_only"`` (Phase 3 Arm B' mechanism-specificity arm):
        score(i) = cache.utility_u[i]
        (EpisodicCache has no separate ``pressure_at_write`` field, so
        ``utility_u`` is the proxy. The Phase 3 falsifier matrix uses
        this arm to distinguish "memory persistence + similarity recall"
        from "any rare-grad-aligned retrieval policy.")

Tests pin both score modes against tiny hand-built caches with known
key_reps + utilities, plus the empty-cache edge case.
"""
from __future__ import annotations

import pytest
import torch

from chaoscontrol.episodic.query import query_topk
from chaoscontrol.optim.episodic_cache import EpisodicCache


def _build_cache_with_entries(
    *,
    capacity: int,
    key_rep_dim: int,
    span_length: int = 2,
    entries: list[dict] | None = None,
) -> EpisodicCache:
    """Build a cache and append the given entries.

    Each entry dict has keys ``key_fp``, ``key_rep`` (tensor [D]),
    ``utility_u`` (float, written directly into ``cache.utility_u`` after
    append since ``append`` always initializes utility to 1.0).
    """
    cache = EpisodicCache(
        capacity=capacity,
        span_length=span_length,
        key_rep_dim=key_rep_dim,
        grace_steps=10,
    )
    if entries:
        for i, e in enumerate(entries):
            cache.append(
                key_fp=int(e["key_fp"]),
                key_rep=e["key_rep"],
                value_tok_ids=torch.zeros(span_length, dtype=torch.int64),
                value_anchor_id=0,
                current_step=int(e.get("current_step", 0)),
                embedding_version=0,
            )
            # Override utility_u directly (append initializes it to 1.0
            # per Decision 0.2's cold-start fix).
            cache.utility_u[i] = float(e["utility_u"])
    return cache


def test_query_topk_cosine_utility_weighted():
    """Pin the score formula: ``cosine_sim(q, key_rep) * utility_u``.

    Cache has 3 entries with distinct key_reps and utilities. Query
    residual aligned with entry 1 (highest cosine) but entry 0 has
    higher utility — the score-weighted top-1 should be entry 0 if
    ``cosine_0 * util_0 > cosine_1 * util_1`` and entry 1 otherwise.
    Build the test so the formula is unambiguous.
    """
    D = 4
    # Three entries with orthogonal-ish key_reps. Use unit-norm vectors
    # so the cosine math is transparent.
    e0_key = torch.tensor([1.0, 0.0, 0.0, 0.0])
    e1_key = torch.tensor([0.0, 1.0, 0.0, 0.0])
    e2_key = torch.tensor([0.0, 0.0, 1.0, 0.0])
    cache = _build_cache_with_entries(
        capacity=8,
        key_rep_dim=D,
        entries=[
            {"key_fp": 100, "key_rep": e0_key, "utility_u": 0.9},
            {"key_fp": 200, "key_rep": e1_key, "utility_u": 0.5},
            {"key_fp": 300, "key_rep": e2_key, "utility_u": 0.1},
        ],
    )
    # Query aligned with e0: cosine_0 = 1.0, cosine_1 = 0, cosine_2 = 0.
    # Scores: 0.9, 0, 0. Top-1 should be slot 0 (which holds e0).
    q = torch.tensor([1.0, 0.0, 0.0, 0.0])
    out = query_topk(q, cache, k=3, score_mode="cosine_utility_weighted")
    assert out.dtype == torch.int64
    assert out.numel() == 3
    assert int(out[0].item()) == 0

    # Now make e1 win by aligning q with it: cosine_1 = 1.0, util = 0.5,
    # score = 0.5; cosine_0 = 0, score_0 = 0. Top-1 = slot 1.
    q2 = torch.tensor([0.0, 1.0, 0.0, 0.0])
    out2 = query_topk(q2, cache, k=3, score_mode="cosine_utility_weighted")
    assert int(out2[0].item()) == 1

    # Mixed alignment: q halfway between e0 and e1. cosine_0 = cosine_1
    # ≈ 0.707; scores ≈ 0.636 vs 0.354. Top-1 = slot 0 (higher utility).
    q3 = torch.tensor([1.0, 1.0, 0.0, 0.0])
    out3 = query_topk(q3, cache, k=3, score_mode="cosine_utility_weighted")
    assert int(out3[0].item()) == 0


def test_query_topk_pressure_only():
    """Pin the pressure-only score: ranks by ``utility_u`` alone.

    The cosine term is fully ignored. Even a query exactly aligned with
    the lowest-utility entry's key_rep returns the highest-utility slot
    first.
    """
    D = 4
    e0_key = torch.tensor([1.0, 0.0, 0.0, 0.0])
    e1_key = torch.tensor([0.0, 1.0, 0.0, 0.0])
    e2_key = torch.tensor([0.0, 0.0, 1.0, 0.0])
    cache = _build_cache_with_entries(
        capacity=8,
        key_rep_dim=D,
        entries=[
            {"key_fp": 100, "key_rep": e0_key, "utility_u": 0.1},
            {"key_fp": 200, "key_rep": e1_key, "utility_u": 0.9},
            {"key_fp": 300, "key_rep": e2_key, "utility_u": 0.5},
        ],
    )
    # Query exactly aligned with e0 — but e0 has the lowest utility.
    # pressure_only mode ignores cosine, so top-1 = slot 1 (utility 0.9).
    q = torch.tensor([1.0, 0.0, 0.0, 0.0])
    out = query_topk(q, cache, k=3, score_mode="pressure_only")
    assert int(out[0].item()) == 1
    assert int(out[1].item()) == 2
    assert int(out[2].item()) == 0


def test_query_topk_handles_empty_cache():
    """Empty cache returns an empty int64 tensor."""
    cache = EpisodicCache(capacity=4, span_length=2, key_rep_dim=4)
    q = torch.tensor([1.0, 0.0, 0.0, 0.0])
    for mode in ("cosine_utility_weighted", "pressure_only"):
        out = query_topk(q, cache, k=8, score_mode=mode)
        assert out.dtype == torch.int64
        assert out.numel() == 0


def test_query_topk_excludes_unoccupied_slots():
    """A cache with some occupied + some empty slots returns only the
    occupied ones. Pin the contract: ``k`` is the requested upper bound,
    not the guaranteed count.
    """
    D = 4
    # Capacity 8 but only 2 entries appended.
    cache = _build_cache_with_entries(
        capacity=8,
        key_rep_dim=D,
        entries=[
            {
                "key_fp": 1,
                "key_rep": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                "utility_u": 0.5,
            },
            {
                "key_fp": 2,
                "key_rep": torch.tensor([0.0, 1.0, 0.0, 0.0]),
                "utility_u": 0.8,
            },
        ],
    )
    q = torch.tensor([1.0, 0.0, 0.0, 0.0])
    out = query_topk(q, cache, k=8, score_mode="cosine_utility_weighted")
    # Only 2 occupied slots; output capped at 2.
    assert out.numel() == 2
    # All returned indices must be in {0, 1} (the occupied slots).
    for s in out.tolist():
        assert s in {0, 1}


def test_query_topk_k_smaller_than_occupied():
    """``k=1`` with 3 occupied slots returns exactly 1 index."""
    D = 4
    cache = _build_cache_with_entries(
        capacity=8,
        key_rep_dim=D,
        entries=[
            {
                "key_fp": 1,
                "key_rep": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                "utility_u": 0.5,
            },
            {
                "key_fp": 2,
                "key_rep": torch.tensor([0.0, 1.0, 0.0, 0.0]),
                "utility_u": 0.8,
            },
            {
                "key_fp": 3,
                "key_rep": torch.tensor([0.0, 0.0, 1.0, 0.0]),
                "utility_u": 0.3,
            },
        ],
    )
    q = torch.tensor([0.0, 1.0, 0.0, 0.0])  # aligned with entry 1
    out = query_topk(q, cache, k=1, score_mode="cosine_utility_weighted")
    assert out.numel() == 1
    assert int(out[0].item()) == 1


def test_query_topk_rejects_unknown_score_mode():
    """Unknown ``score_mode`` raises ValueError so typos surface
    immediately rather than silently degrading."""
    D = 4
    cache = _build_cache_with_entries(
        capacity=4,
        key_rep_dim=D,
        entries=[
            {
                "key_fp": 1,
                "key_rep": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                "utility_u": 0.5,
            },
        ],
    )
    q = torch.tensor([1.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="score_mode"):
        query_topk(q, cache, k=1, score_mode="not_a_real_mode")


def test_query_topk_returns_descending_score_order():
    """Top-K output is ordered by score, highest first."""
    D = 4
    cache = _build_cache_with_entries(
        capacity=8,
        key_rep_dim=D,
        entries=[
            {
                "key_fp": 1,
                "key_rep": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                "utility_u": 0.2,
            },
            {
                "key_fp": 2,
                "key_rep": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                "utility_u": 0.8,
            },
            {
                "key_fp": 3,
                "key_rep": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                "utility_u": 0.5,
            },
        ],
    )
    q = torch.tensor([1.0, 0.0, 0.0, 0.0])
    out = query_topk(q, cache, k=3, score_mode="cosine_utility_weighted")
    # All cosines = 1.0; ranking is by utility: slot 1 (0.8), slot 2
    # (0.5), slot 0 (0.2).
    assert [int(x.item()) for x in out] == [1, 2, 0]
