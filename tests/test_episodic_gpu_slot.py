"""Tests for the GPU-resident episodic IPC slot format (Pass C.1).

Covers ``src/chaoscontrol/episodic/gpu_slot.py``:

  1. ``slot_dim(S, D)`` matches the design-doc layout exactly.
  2. ``pack_payload`` + ``unpack_payload`` round-trip every field with
     the right semantics (fp32 tolerances on float fields; exact match
     on int64-reinterpreted fields).
  3. INT64_MAX / INT64_MIN survive the fp32 reinterpret.
  4. ``make_slot_tensor`` returns a zero-initialized tensor so the
     valid_mask filter in the episodic-rank drain treats untouched rows
     as empty.

The slot format is the contract between the train-rank gather producer
and the episodic-rank gather consumer; any drift here breaks the IPC
end-to-end. These tests run on CPU only — the format and the int64
reinterpret are device-agnostic.
"""
from __future__ import annotations

import pytest
import torch

from chaoscontrol.episodic.gpu_slot import (
    SLOT_DIM_BASE,
    make_slot_tensor,
    pack_payload,
    slot_dim,
    unpack_payload,
)


# ---------------------------------------------------------------------------
# Test 1: slot_dim layout
# ---------------------------------------------------------------------------


def test_slot_dim_layout() -> None:
    """At the design defaults S=4, D=256 the slot is 526 fp32 cells.

    Layout: ``6 + 2*S + 2*D = 6 + 8 + 512 = 526``. If this number
    changes, every other test in this file is wrong AND the gather
    collective's per-rank tensor size on the runner is wrong, so it's
    pinned hard here.
    """
    assert slot_dim(span_length=4, key_rep_dim=256) == 526
    # Generic formula check at a few other (S, D) values.
    assert slot_dim(span_length=1, key_rep_dim=1) == SLOT_DIM_BASE + 2 + 2
    assert slot_dim(span_length=8, key_rep_dim=128) == SLOT_DIM_BASE + 16 + 256


def test_slot_dim_rejects_non_positive() -> None:
    """``slot_dim`` rejects non-positive S or D — there is no useful
    width that maps to a degenerate slot.
    """
    with pytest.raises(ValueError):
        slot_dim(span_length=0, key_rep_dim=4)
    with pytest.raises(ValueError):
        slot_dim(span_length=4, key_rep_dim=0)
    with pytest.raises(ValueError):
        slot_dim(span_length=-1, key_rep_dim=4)


# ---------------------------------------------------------------------------
# Test 2: pack/unpack round trip
# ---------------------------------------------------------------------------


def test_pack_unpack_round_trip() -> None:
    """Every field round-trips through pack -> unpack.

    fp32 fields use ``torch.allclose`` with a tight tolerance because
    ``.to(dtype=torch.float32)`` is a no-op on already-fp32 input.
    int64-reinterpreted fields use exact equality; they're round-tripped
    bit-for-bit through the view.
    """
    S = 4
    D = 8
    slot = torch.zeros(slot_dim(span_length=S, key_rep_dim=D), dtype=torch.float32)
    key_fp = 0x1234_5678_9ABC_DEF0
    anchor = 137
    tok_ids = torch.tensor([10, 20, 30, 40], dtype=torch.int64)
    key_rep = torch.arange(D, dtype=torch.float32) * 0.5
    residual = torch.arange(D, dtype=torch.float32) - 1.0
    pack_payload(
        slot,
        valid_mask=1.0,
        pressure=0.75,
        key_fp=key_fp,
        value_anchor_id=anchor,
        value_tok_ids=tok_ids,
        key_rep=key_rep,
        residual=residual,
        span_length=S,
        key_rep_dim=D,
    )
    out = unpack_payload(slot, span_length=S, key_rep_dim=D)
    assert out["valid_mask"] == 1.0
    assert out["pressure"] == pytest.approx(0.75)
    # int64 fields: exact match
    assert out["key_fp"] == key_fp
    assert out["value_anchor_id"] == anchor
    assert torch.equal(out["value_tok_ids"], tok_ids)
    # fp32 fields: tight allclose
    assert torch.allclose(out["key_rep"], key_rep)
    assert torch.allclose(out["residual"], residual)


def test_pack_unpack_clones_so_caller_can_reuse_slot() -> None:
    """``unpack_payload`` clones tensor outputs so re-packing the slot
    next step doesn't overwrite the unpacked dict's contents (which
    flow into ``cache.append`` and the controller queue).
    """
    S = 2
    D = 3
    slot = torch.zeros(slot_dim(span_length=S, key_rep_dim=D), dtype=torch.float32)
    pack_payload(
        slot,
        valid_mask=1.0,
        pressure=1.0,
        key_fp=42,
        value_anchor_id=1,
        value_tok_ids=torch.tensor([1, 2], dtype=torch.int64),
        key_rep=torch.tensor([1.0, 2.0, 3.0]),
        residual=torch.tensor([4.0, 5.0, 6.0]),
        span_length=S,
        key_rep_dim=D,
    )
    out = unpack_payload(slot, span_length=S, key_rep_dim=D)
    # Repack with all-different data. Unpacked outputs must NOT change.
    pack_payload(
        slot,
        valid_mask=0.0,
        pressure=0.0,
        key_fp=99,
        value_anchor_id=99,
        value_tok_ids=torch.tensor([99, 99], dtype=torch.int64),
        key_rep=torch.tensor([0.0, 0.0, 0.0]),
        residual=torch.tensor([0.0, 0.0, 0.0]),
        span_length=S,
        key_rep_dim=D,
    )
    assert out["key_fp"] == 42
    assert torch.equal(
        out["value_tok_ids"], torch.tensor([1, 2], dtype=torch.int64)
    )
    assert torch.equal(out["key_rep"], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(out["residual"], torch.tensor([4.0, 5.0, 6.0]))


# ---------------------------------------------------------------------------
# Test 3: INT64 reinterpret safety near the type boundaries
# ---------------------------------------------------------------------------


def test_int64_reinterpret_safety() -> None:
    """Values near INT64_MAX / INT64_MIN survive the fp32 view round-trip.

    The slot stores int64 fields by viewing two adjacent fp32 cells as
    one int64. fp32 has only 24 bits of mantissa, so a naive
    ``float(int64)`` cast would lose precision; the test pins that the
    int64 view sees the raw bytes, NOT a float-converted value.
    """
    INT64_MAX = (1 << 63) - 1
    INT64_MIN = -(1 << 63)

    S = 1
    D = 1
    slot = torch.zeros(slot_dim(span_length=S, key_rep_dim=D), dtype=torch.float32)

    for fp, anchor, tok in [
        (INT64_MAX, INT64_MAX, INT64_MAX),
        (INT64_MIN, INT64_MIN, INT64_MIN),
        (1 << 62, (1 << 62) + 1, 1 << 50),
        (-1, -2, -3),  # all-ones bit pattern in int64
        (0, 0, 0),
    ]:
        pack_payload(
            slot,
            valid_mask=1.0,
            pressure=0.0,
            key_fp=fp,
            value_anchor_id=anchor,
            value_tok_ids=torch.tensor([tok], dtype=torch.int64),
            key_rep=torch.zeros(D),
            residual=torch.zeros(D),
            span_length=S,
            key_rep_dim=D,
        )
        out = unpack_payload(slot, span_length=S, key_rep_dim=D)
        assert out["key_fp"] == fp, f"key_fp mismatch for {fp}: got {out['key_fp']}"
        assert out["value_anchor_id"] == anchor, (
            f"value_anchor_id mismatch for {anchor}: got {out['value_anchor_id']}"
        )
        assert int(out["value_tok_ids"][0].item()) == tok, (
            f"value_tok_ids mismatch for {tok}: got {out['value_tok_ids'][0].item()}"
        )


# ---------------------------------------------------------------------------
# Test 4: make_slot_tensor zero-initialization
# ---------------------------------------------------------------------------


def test_make_slot_tensor_zeroed() -> None:
    """Newly allocated slot tensor has ``valid_mask = 0`` everywhere.

    The episodic-rank drain filters by ``valid_mask > 0.5``; if the
    factory ever switched from ``zeros`` to ``empty`` we'd get
    nondeterministic "phantom" cache writes from uninitialized memory.
    """
    K_max = 16
    S = 4
    D = 256
    t = make_slot_tensor(
        k_max=K_max,
        span_length=S,
        key_rep_dim=D,
        device=torch.device("cpu"),
    )
    assert t.shape == (K_max, slot_dim(span_length=S, key_rep_dim=D))
    assert t.dtype == torch.float32
    # valid_mask is column 0 of every row.
    assert torch.all(t[:, 0] == 0.0)
    # Whole tensor is zero — defensive against a factory that only zeros
    # the valid_mask column.
    assert torch.all(t == 0.0)


def test_make_slot_tensor_rejects_bad_args() -> None:
    """Defensive: factory rejects non-positive K_max and non-fp32 dtype."""
    with pytest.raises(ValueError):
        make_slot_tensor(
            k_max=0, span_length=4, key_rep_dim=4, device=torch.device("cpu"),
        )
    with pytest.raises(ValueError):
        make_slot_tensor(
            k_max=4, span_length=4, key_rep_dim=4,
            device=torch.device("cpu"), dtype=torch.float16,
        )
