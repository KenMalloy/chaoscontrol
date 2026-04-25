"""Tests for the write-trigger logic (component 2 of the memory subsystem)."""
from __future__ import annotations

import pytest
import torch

from chaoscontrol.optim.episodic_writer import (
    WritePayload,
    build_write_payload,
    fingerprint_tokens,
    fingerprint_tokens_batch,
    select_top_p_positions,
    select_writes,
)


# ---------------------------------------------------------------------------
# fingerprint_tokens
# ---------------------------------------------------------------------------


def test_fingerprint_is_deterministic():
    a = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
    b = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
    assert fingerprint_tokens(a) == fingerprint_tokens(b)


def test_fingerprint_distinguishes_distinct_inputs():
    h1 = fingerprint_tokens(torch.tensor([1, 2, 3, 4], dtype=torch.int64))
    h2 = fingerprint_tokens(torch.tensor([1, 2, 3, 5], dtype=torch.int64))
    assert h1 != h2


def test_fingerprint_distinguishes_order():
    h1 = fingerprint_tokens(torch.tensor([1, 2, 3, 4], dtype=torch.int64))
    h2 = fingerprint_tokens(torch.tensor([4, 3, 2, 1], dtype=torch.int64))
    assert h1 != h2


def test_fingerprint_distinguishes_pads_from_zeros():
    """Token id 0 must produce a different hash from a shorter sequence —
    rolling hashes that just sum tokens conflate these. Our polynomial
    rolling hash adds 1 to each token to avoid this."""
    short = fingerprint_tokens(torch.tensor([1], dtype=torch.int64))
    padded = fingerprint_tokens(torch.tensor([0, 1], dtype=torch.int64))
    assert short != padded


def test_fingerprint_returns_non_negative_int64_range():
    h = fingerprint_tokens(torch.tensor([10, 20, 30, 40], dtype=torch.int64))
    assert isinstance(h, int)
    assert 0 <= h < (1 << 61) - 1


def test_fingerprint_rejects_wrong_dim():
    with pytest.raises(ValueError):
        fingerprint_tokens(torch.zeros(2, 3, dtype=torch.int64))


def test_fingerprint_batch_matches_per_row():
    rows = torch.tensor(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.int64,
    )
    batched = fingerprint_tokens_batch(rows)
    individual = [fingerprint_tokens(rows[i]) for i in range(rows.shape[0])]
    assert batched == individual


def test_fingerprint_batch_rejects_wrong_dim():
    with pytest.raises(ValueError):
        fingerprint_tokens_batch(torch.zeros(4, dtype=torch.int64))


# ---------------------------------------------------------------------------
# select_top_p_positions
# ---------------------------------------------------------------------------


def test_select_top_p_returns_highest_signal_position_first():
    signal = torch.tensor([
        [0.1, 0.5, 0.9],
        [0.3, 0.7, 0.4],
    ])
    positions = select_top_p_positions(signal, top_p=1.0 / 6.0)  # k=1
    assert positions.shape == (1, 2)
    assert int(positions[0, 0].item()) == 0
    assert int(positions[0, 1].item()) == 2


def test_select_top_p_returns_correct_count():
    signal = torch.rand(4, 8)
    positions = select_top_p_positions(signal, top_p=0.25)
    expected_k = max(1, int(round(4 * 8 * 0.25)))
    assert positions.shape == (expected_k, 2)


def test_select_top_p_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        select_top_p_positions(torch.zeros(4), top_p=0.1)
    with pytest.raises(ValueError):
        select_top_p_positions(torch.zeros(2, 3), top_p=0.0)
    with pytest.raises(ValueError):
        select_top_p_positions(torch.zeros(2, 3), top_p=1.5)


def test_select_top_p_minimum_one_position():
    """Even at very small top_p, at least one position must be returned —
    otherwise an unlucky run with very low signal could produce zero
    writes for a whole step, breaking the cache warm-up."""
    signal = torch.zeros(2, 4)
    positions = select_top_p_positions(signal, top_p=0.0001)
    assert positions.shape[0] >= 1


# ---------------------------------------------------------------------------
# build_write_payload
# ---------------------------------------------------------------------------


def _basic_inputs(B: int = 2, T: int = 16, key_rep_dim: int = 4):
    input_ids = torch.arange(B * T, dtype=torch.int64).reshape(B, T)
    target_ids = (input_ids + 1) % 100
    key_rep = torch.randn(B, T, key_rep_dim)
    return input_ids, target_ids, key_rep


def test_build_payload_returns_none_at_left_boundary():
    inputs, targets, rep = _basic_inputs()
    payload = build_write_payload(
        batch_index=0,
        position=2,
        input_ids=inputs,
        target_ids=targets,
        key_rep_per_position=rep,
        fingerprint_window=4,
        span_length=4,
    )
    assert payload is None


def test_build_payload_returns_none_at_right_boundary():
    inputs, targets, rep = _basic_inputs(T=16)
    # position=14 with span_length=4 needs positions 14..17, T=16 -> overflow.
    payload = build_write_payload(
        batch_index=0,
        position=14,
        input_ids=inputs,
        target_ids=targets,
        key_rep_per_position=rep,
        fingerprint_window=4,
        span_length=4,
    )
    assert payload is None


def test_build_payload_succeeds_when_in_bounds():
    inputs, targets, rep = _basic_inputs(B=2, T=16, key_rep_dim=4)
    payload = build_write_payload(
        batch_index=1,
        position=8,
        input_ids=inputs,
        target_ids=targets,
        key_rep_per_position=rep,
        fingerprint_window=4,
        span_length=4,
    )
    assert isinstance(payload, WritePayload)
    assert payload.batch_index == 1
    assert payload.position == 8
    # value_tok_ids = target_ids[1, 8:12]
    assert torch.equal(
        payload.value_tok_ids, targets[1, 8:12].to(torch.int64),
    )
    assert payload.value_anchor_id == int(targets[1, 8].item())
    # key_rep is the slice at (1, 8)
    assert torch.equal(payload.key_rep, rep[1, 8])
    # key_fp is reproducible from input_ids[1, 4:8]
    expected_fp = fingerprint_tokens(inputs[1, 4:8])
    assert payload.key_fp == expected_fp


def test_build_payload_validates_batch_and_position_bounds():
    inputs, targets, rep = _basic_inputs(B=2, T=16, key_rep_dim=4)
    with pytest.raises(IndexError):
        build_write_payload(
            batch_index=3, position=8,
            input_ids=inputs, target_ids=targets,
            key_rep_per_position=rep,
            fingerprint_window=4, span_length=4,
        )
    with pytest.raises(IndexError):
        build_write_payload(
            batch_index=0, position=20,
            input_ids=inputs, target_ids=targets,
            key_rep_per_position=rep,
            fingerprint_window=4, span_length=4,
        )


# ---------------------------------------------------------------------------
# select_writes (top-level)
# ---------------------------------------------------------------------------


def test_select_writes_picks_highest_signal_and_drops_boundaries():
    B, T, D = 2, 16, 4
    inputs, targets, rep = _basic_inputs(B=B, T=T, key_rep_dim=D)
    pressure = torch.zeros(B, T)
    ce = torch.zeros(B, T)
    # Make one strong-signal position deep in the interior, one near the
    # right boundary (will be dropped).
    pressure[0, 8] = 1.0
    ce[0, 8] = 5.0
    pressure[1, 14] = 1.0
    ce[1, 14] = 10.0  # higher signal but at the boundary
    payloads = select_writes(
        input_ids=inputs,
        target_ids=targets,
        pressure=pressure,
        per_token_ce=ce,
        key_rep_per_position=rep,
        top_p=2.0 / (B * T),  # k=2 -> both positions selected, one dropped
        fingerprint_window=4,
        span_length=4,
    )
    # Exactly one survives — the interior one.
    assert len(payloads) == 1
    assert payloads[0].batch_index == 0
    assert payloads[0].position == 8


def test_select_writes_returns_payloads_in_signal_descending_order():
    B, T, D = 1, 16, 4
    inputs, targets, rep = _basic_inputs(B=B, T=T, key_rep_dim=D)
    pressure = torch.zeros(B, T)
    ce = torch.zeros(B, T)
    pressure[0, 5] = 1.0
    ce[0, 5] = 1.0  # signal=1
    pressure[0, 8] = 1.0
    ce[0, 8] = 9.0  # signal=9
    pressure[0, 11] = 1.0
    ce[0, 11] = 4.0  # signal=4
    payloads = select_writes(
        input_ids=inputs,
        target_ids=targets,
        pressure=pressure,
        per_token_ce=ce,
        key_rep_per_position=rep,
        top_p=3.0 / (B * T),
        fingerprint_window=4,
        span_length=4,
    )
    # All three are in-bounds.
    positions = [p.position for p in payloads]
    assert positions == [8, 11, 5]


def test_select_writes_validates_shapes():
    B, T, D = 2, 8, 4
    inputs, targets, rep = _basic_inputs(B=B, T=T, key_rep_dim=D)
    with pytest.raises(ValueError, match="pressure"):
        select_writes(
            input_ids=inputs,
            target_ids=targets,
            pressure=torch.zeros(B, T + 1),
            per_token_ce=torch.zeros(B, T),
            key_rep_per_position=rep,
            top_p=0.1,
            fingerprint_window=4,
            span_length=4,
        )
    with pytest.raises(ValueError, match="target_ids"):
        select_writes(
            input_ids=inputs,
            target_ids=torch.zeros(B + 1, T, dtype=torch.int64),
            pressure=torch.zeros(B, T),
            per_token_ce=torch.zeros(B, T),
            key_rep_per_position=rep,
            top_p=0.1,
            fingerprint_window=4,
            span_length=4,
        )
    with pytest.raises(ValueError, match="key_rep_per_position"):
        select_writes(
            input_ids=inputs,
            target_ids=targets,
            pressure=torch.zeros(B, T),
            per_token_ce=torch.zeros(B, T),
            key_rep_per_position=torch.zeros(B, T),  # not 3-D
            top_p=0.1,
            fingerprint_window=4,
            span_length=4,
        )


def test_select_writes_handles_all_boundary_positions_gracefully():
    """Pathological case: all top-K signal happens to land in boundary
    regions. Function must return an empty list rather than crash."""
    B, T, D = 1, 8, 4
    inputs, targets, rep = _basic_inputs(B=B, T=T, key_rep_dim=D)
    pressure = torch.zeros(B, T)
    ce = torch.zeros(B, T)
    # Top-K all at position 0 and position T-1 -> all boundary, all dropped.
    pressure[0, 0] = 1.0
    ce[0, 0] = 10.0
    pressure[0, T - 1] = 1.0
    ce[0, T - 1] = 9.0
    payloads = select_writes(
        input_ids=inputs,
        target_ids=targets,
        pressure=pressure,
        per_token_ce=ce,
        key_rep_per_position=rep,
        top_p=2.0 / (B * T),
        fingerprint_window=4,
        span_length=4,
    )
    assert payloads == []
