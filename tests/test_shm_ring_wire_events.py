"""Real wire-event-typed ShmRing instantiations (Phase A5 of CPU SSM
controller plan).

Three rings — WriteEvent (capacity 16384), QueryEvent (16384),
ReplayOutcome (8192). Each exposes the same create / attach / push /
pop / size / name / unlink API as A4's ShmRingU64x1024 but accepts and
returns Python dicts whose keys match the non-pad fields of the
corresponding wire-event struct in ``src/wire_events.h``.

Tests:
  - per-ring single-process round-trip (push 100 dicts, pop 100,
    field-by-field equality including arrays and special floats)
  - dict validation: extra key → KeyError, missing key → KeyError
  - capacity / REGION_BYTES expose the right values
  - NaN-safe round-trip for ReplayOutcome's grad_cos_* fields

Spec: docs/plans/2026-04-26-cpu-ssm-controller.md (Task A5).
"""
from __future__ import annotations

import math

import pytest

from chaoscontrol.kernels import _cpu_ssm_controller as _ext


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _force_unlink(cls, name: str) -> None:
    """Idempotent unlink — a previously failed run can leave the kernel
    region persistent; safe to call even if the region does not exist."""
    try:
        cls.unlink(name)
    except Exception:
        pass


def _sample_write_event() -> dict:
    return {
        "event_type": 1,
        "source_rank": 3,
        "write_bucket": 2,
        "candidate_id": (3 << 56) | 42,
        "gpu_step": 100,
        "key_fp": 0xDEADBEEFCAFEBABE,
        "key_rep": [i % 256 for i in range(256)],
        "value_tok_ids": [11, 22, 33, 44],
        "value_anchor_id": 7,
        "pressure_at_write": 1.25,
        "pre_write_ce": 2.34,
    }


_QE_SLOT_SENTINEL = (1 << 64) - 1


def _sample_query_event() -> dict:
    # Default fixture rides the V0 heuristic-only path: simplex
    # candidates sentinel-padded so the C++ controller will dispatch
    # to per-slot fallback. Tests that exercise the simplex path
    # override these two fields explicitly.
    return {
        "event_type": 2,
        "source_rank": 1,
        "bucket": 4,
        "query_id": 0xBADC0FFEE0DDF00D,
        "gpu_step": 200,
        "query_rep": [(i * 3) % 65536 for i in range(256)],
        "pressure": 0.875,
        "pre_query_ce": 3.5,
        "candidate_slot_ids": [_QE_SLOT_SENTINEL] * 16,
        "candidate_cosines": [0.0] * 16,
    }


def _sample_replay_outcome() -> dict:
    return {
        "event_type": 3,
        "selected_rank": 2,
        "outcome_status": 0,  # ok
        "replay_id": 0x1111_2222_3333_4444,
        "gpu_step": 300,
        "query_event_id": 0x5555_6666_7777_8888,
        "source_write_id": 0x9999_AAAA_BBBB_CCCC,
        "slot_id": 12345,
        "policy_version": 7,
        "selection_step": 999,
        "teacher_score": 0.5,
        "controller_logit": -1.25,
        "ce_before_replay": 4.1,
        "ce_after_replay": 3.9,
        "ce_delta_raw": -0.2,
        "bucket_baseline": 4.0,
        "reward_shaped": 0.05,
        "grad_cos_rare": float("nan"),   # NaN until Phase 4
        "grad_cos_total": float("nan"),  # NaN until Phase 4
        "flags": 0xABCD,
    }


# ---------------------------------------------------------------------------
# WriteEvent
# ---------------------------------------------------------------------------

def test_shm_ring_write_event_round_trip():
    """Push 100 distinct WriteEvent dicts, pop all 100, verify every
    field round-trips exactly. Pins the per-field push/pop conversion
    against the layout in wire_events.h."""
    name = "/cc_test_shm_write"
    _force_unlink(_ext.ShmRingWriteEvent, name)
    ring = _ext.ShmRingWriteEvent.create(name)
    try:
        assert ring.name() == name
        for i in range(100):
            ev = _sample_write_event()
            ev["candidate_id"] = i  # unique per push
            assert ring.push(ev), f"push {i} failed (size={ring.size()})"
        assert ring.size() == 100
        baseline_key_rep = [j % 256 for j in range(256)]
        for i in range(100):
            popped = ring.pop()
            assert popped is not None, f"pop {i} returned None"
            assert popped["candidate_id"] == i
            assert popped["event_type"] == 1
            assert popped["source_rank"] == 3
            assert popped["write_bucket"] == 2
            assert popped["gpu_step"] == 100
            assert popped["key_fp"] == 0xDEADBEEFCAFEBABE
            assert popped["key_rep"] == baseline_key_rep
            assert popped["value_tok_ids"] == [11, 22, 33, 44]
            assert popped["value_anchor_id"] == 7
            assert popped["pressure_at_write"] == pytest.approx(1.25)
            assert popped["pre_write_ce"] == pytest.approx(2.34)
        assert ring.pop() is None
    finally:
        _ext.ShmRingWriteEvent.unlink(name)


# ---------------------------------------------------------------------------
# QueryEvent
# ---------------------------------------------------------------------------

def test_shm_ring_query_event_round_trip():
    """Push 100 distinct QueryEvent dicts, pop all 100, verify every
    field round-trips exactly."""
    name = "/cc_test_shm_query"
    _force_unlink(_ext.ShmRingQueryEvent, name)
    ring = _ext.ShmRingQueryEvent.create(name)
    try:
        for i in range(100):
            ev = _sample_query_event()
            ev["query_id"] = i
            assert ring.push(ev), f"push {i} failed"
        assert ring.size() == 100
        baseline_query_rep = [(j * 3) % 65536 for j in range(256)]
        baseline_candidate_slot_ids = [_QE_SLOT_SENTINEL] * 16
        baseline_candidate_cosines = [0.0] * 16
        for i in range(100):
            popped = ring.pop()
            assert popped is not None
            assert popped["query_id"] == i
            assert popped["event_type"] == 2
            assert popped["source_rank"] == 1
            assert popped["bucket"] == 4
            assert popped["gpu_step"] == 200
            assert popped["query_rep"] == baseline_query_rep
            assert popped["pressure"] == pytest.approx(0.875)
            assert popped["pre_query_ce"] == pytest.approx(3.5)
            # Phase S3: simplex candidate arrays round-trip even on the
            # heuristic-only path (sentinel fill survives the wire).
            assert popped["candidate_slot_ids"] == baseline_candidate_slot_ids
            assert popped["candidate_cosines"] == baseline_candidate_cosines
        assert ring.pop() is None
    finally:
        _ext.ShmRingQueryEvent.unlink(name)


# ---------------------------------------------------------------------------
# QueryEvent simplex candidate set (Phase S3)
# ---------------------------------------------------------------------------

def test_query_event_roundtrip_with_simplex_candidates():
    """Push a QueryEvent dict with a populated 16-slot simplex candidate
    set; pop and verify the slot ids and cosines survive the wire byte-
    equal in order. Pins the Phase S3 schema bump end-to-end through the
    SpscRing<QueryEvent, 16384> instantiation."""
    name = "/cc_test_shm_query_simplex"
    _force_unlink(_ext.ShmRingQueryEvent, name)
    ring = _ext.ShmRingQueryEvent.create(name)
    try:
        ev = _sample_query_event()
        slot_ids = [10 * (i + 1) for i in range(16)]      # 10, 20, ..., 160
        cosines = [0.95 - 0.05 * i for i in range(16)]    # 0.95 .. 0.20
        ev["candidate_slot_ids"] = list(slot_ids)
        ev["candidate_cosines"] = list(cosines)
        assert ring.push(ev)
        popped = ring.pop()
        assert popped is not None
        assert popped["candidate_slot_ids"] == slot_ids
        for i, c in enumerate(cosines):
            assert popped["candidate_cosines"][i] == pytest.approx(c)
    finally:
        _ext.ShmRingQueryEvent.unlink(name)


def test_query_event_roundtrip_with_sentinel_candidates():
    """V0 heuristic-only path: producer fills both candidate arrays
    with sentinels (UINT64_MAX × 16 / 0.0 × 16). Verify the sentinels
    survive the wire and the popped dict can be matched against the
    sentinel constants for fallback dispatch."""
    name = "/cc_test_shm_query_sentinel"
    _force_unlink(_ext.ShmRingQueryEvent, name)
    ring = _ext.ShmRingQueryEvent.create(name)
    try:
        ev = _sample_query_event()
        ev["candidate_slot_ids"] = [_QE_SLOT_SENTINEL] * 16
        ev["candidate_cosines"] = [0.0] * 16
        assert ring.push(ev)
        popped = ring.pop()
        assert popped is not None
        assert popped["candidate_slot_ids"] == [_QE_SLOT_SENTINEL] * 16
        assert popped["candidate_cosines"] == [0.0] * 16
        # Dispatch contract: controller treats slot_id[0] == sentinel as
        # "no simplex; fall back to per-slot V0 path."
        assert popped["candidate_slot_ids"][0] == _QE_SLOT_SENTINEL
    finally:
        _ext.ShmRingQueryEvent.unlink(name)


def test_query_event_dict_validates_candidate_array_length():
    """The candidate arrays are uint64[16] / float[16] — wrong length is
    a programmer error (should have been sentinel-padded by the
    producer, not silently zero-filled by the wire glue)."""
    name = "/cc_test_shm_query_badlen"
    _force_unlink(_ext.ShmRingQueryEvent, name)
    ring = _ext.ShmRingQueryEvent.create(name)
    try:
        bad = _sample_query_event()
        bad["candidate_slot_ids"] = [1] * 15       # one short
        with pytest.raises((ValueError, KeyError)):
            ring.push(bad)
        bad = _sample_query_event()
        bad["candidate_cosines"] = [0.5] * 17      # one long
        with pytest.raises((ValueError, KeyError)):
            ring.push(bad)
    finally:
        _ext.ShmRingQueryEvent.unlink(name)


def test_query_event_dict_accepts_omitted_candidate_keys():
    """Backward-compat: a dict that predates Phase S3 (no candidate
    arrays) must still push. The C++ glue defaults each array to its
    sentinel fill so the popped dict carries the sentinel pattern."""
    name = "/cc_test_shm_query_omit"
    _force_unlink(_ext.ShmRingQueryEvent, name)
    ring = _ext.ShmRingQueryEvent.create(name)
    try:
        ev = _sample_query_event()
        del ev["candidate_slot_ids"]
        del ev["candidate_cosines"]
        assert ring.push(ev)
        popped = ring.pop()
        assert popped is not None
        assert popped["candidate_slot_ids"] == [_QE_SLOT_SENTINEL] * 16
        assert popped["candidate_cosines"] == [0.0] * 16
    finally:
        _ext.ShmRingQueryEvent.unlink(name)


# ---------------------------------------------------------------------------
# ReplayOutcome
# ---------------------------------------------------------------------------

def test_shm_ring_replay_outcome_round_trip_with_nan():
    """Round-trip the ReplayOutcome dict including NaN-valued
    grad_cos_* fields. NaN comparison uses math.isnan because
    `nan == nan` is False — a naive equality check would silently pass
    any garbage value back to the caller."""
    name = "/cc_test_shm_replay"
    _force_unlink(_ext.ShmRingReplayOutcome, name)
    ring = _ext.ShmRingReplayOutcome.create(name)
    try:
        for i in range(100):
            ev = _sample_replay_outcome()
            ev["replay_id"] = i
            assert ring.push(ev), f"push {i} failed"
        assert ring.size() == 100
        for i in range(100):
            popped = ring.pop()
            assert popped is not None
            assert popped["replay_id"] == i
            assert popped["event_type"] == 3
            assert popped["selected_rank"] == 2
            assert popped["outcome_status"] == 0
            assert popped["gpu_step"] == 300
            assert popped["query_event_id"] == 0x5555_6666_7777_8888
            assert popped["source_write_id"] == 0x9999_AAAA_BBBB_CCCC
            assert popped["slot_id"] == 12345
            assert popped["policy_version"] == 7
            assert popped["selection_step"] == 999
            assert popped["teacher_score"] == pytest.approx(0.5)
            assert popped["controller_logit"] == pytest.approx(-1.25)
            assert popped["ce_before_replay"] == pytest.approx(4.1)
            assert popped["ce_after_replay"] == pytest.approx(3.9)
            assert popped["ce_delta_raw"] == pytest.approx(-0.2)
            assert popped["bucket_baseline"] == pytest.approx(4.0)
            assert popped["reward_shaped"] == pytest.approx(0.05)
            assert math.isnan(popped["grad_cos_rare"])
            assert math.isnan(popped["grad_cos_total"])
            assert popped["flags"] == 0xABCD
        assert ring.pop() is None
    finally:
        _ext.ShmRingReplayOutcome.unlink(name)


def test_shm_ring_replay_outcome_round_trip_finite_grad_cos():
    """Once Phase 4 lands, grad_cos_* will carry finite values. Push a
    finite-valued ReplayOutcome separately from the NaN case so the
    finite-float survival check is not masked by NaN handling."""
    name = "/cc_test_shm_replay_finite"
    _force_unlink(_ext.ShmRingReplayOutcome, name)
    ring = _ext.ShmRingReplayOutcome.create(name)
    try:
        ev = _sample_replay_outcome()
        ev["grad_cos_rare"] = 0.123
        ev["grad_cos_total"] = -0.456
        assert ring.push(ev)
        popped = ring.pop()
        assert popped is not None
        assert popped["grad_cos_rare"] == pytest.approx(0.123)
        assert popped["grad_cos_total"] == pytest.approx(-0.456)
    finally:
        _ext.ShmRingReplayOutcome.unlink(name)


# ---------------------------------------------------------------------------
# dict validation
# ---------------------------------------------------------------------------

def test_shm_ring_write_event_rejects_extra_key():
    """Extra dict key → KeyError. Catches typos like 'pre_write_cee'
    that would otherwise silently pad a struct field with a default."""
    name = "/cc_test_shm_write_extra"
    _force_unlink(_ext.ShmRingWriteEvent, name)
    ring = _ext.ShmRingWriteEvent.create(name)
    try:
        bad = _sample_write_event()
        bad["unexpected_field"] = 0
        with pytest.raises(KeyError):
            ring.push(bad)
    finally:
        _ext.ShmRingWriteEvent.unlink(name)


def test_shm_ring_write_event_rejects_missing_key():
    """Missing dict key → KeyError. Caller can't accidentally rely on
    a zero-initialized field by omitting it."""
    name = "/cc_test_shm_write_missing"
    _force_unlink(_ext.ShmRingWriteEvent, name)
    ring = _ext.ShmRingWriteEvent.create(name)
    try:
        bad = _sample_write_event()
        del bad["pressure_at_write"]
        with pytest.raises(KeyError):
            ring.push(bad)
    finally:
        _ext.ShmRingWriteEvent.unlink(name)


def test_shm_ring_query_event_rejects_extra_key():
    name = "/cc_test_shm_query_extra"
    _force_unlink(_ext.ShmRingQueryEvent, name)
    ring = _ext.ShmRingQueryEvent.create(name)
    try:
        bad = _sample_query_event()
        bad["bonus"] = 1
        with pytest.raises(KeyError):
            ring.push(bad)
    finally:
        _ext.ShmRingQueryEvent.unlink(name)


def test_shm_ring_query_event_rejects_missing_key():
    name = "/cc_test_shm_query_missing"
    _force_unlink(_ext.ShmRingQueryEvent, name)
    ring = _ext.ShmRingQueryEvent.create(name)
    try:
        bad = _sample_query_event()
        del bad["pre_query_ce"]
        with pytest.raises(KeyError):
            ring.push(bad)
    finally:
        _ext.ShmRingQueryEvent.unlink(name)


def test_shm_ring_replay_outcome_rejects_extra_key():
    name = "/cc_test_shm_replay_extra"
    _force_unlink(_ext.ShmRingReplayOutcome, name)
    ring = _ext.ShmRingReplayOutcome.create(name)
    try:
        bad = _sample_replay_outcome()
        bad["something_else"] = 0.0
        with pytest.raises(KeyError):
            ring.push(bad)
    finally:
        _ext.ShmRingReplayOutcome.unlink(name)


def test_shm_ring_replay_outcome_rejects_missing_key():
    name = "/cc_test_shm_replay_missing"
    _force_unlink(_ext.ShmRingReplayOutcome, name)
    ring = _ext.ShmRingReplayOutcome.create(name)
    try:
        bad = _sample_replay_outcome()
        del bad["flags"]
        with pytest.raises(KeyError):
            ring.push(bad)
    finally:
        _ext.ShmRingReplayOutcome.unlink(name)


# ---------------------------------------------------------------------------
# array length validation
# ---------------------------------------------------------------------------

def test_shm_ring_write_event_rejects_wrong_key_rep_length():
    """key_rep is a uint16[256] — wrong length is a programmer error,
    not a silent truncation/zero-fill."""
    name = "/cc_test_shm_write_keylen"
    _force_unlink(_ext.ShmRingWriteEvent, name)
    ring = _ext.ShmRingWriteEvent.create(name)
    try:
        bad = _sample_write_event()
        bad["key_rep"] = [0] * 255  # one short
        with pytest.raises((ValueError, KeyError)):
            ring.push(bad)
    finally:
        _ext.ShmRingWriteEvent.unlink(name)


# ---------------------------------------------------------------------------
# capacity + REGION_BYTES
# ---------------------------------------------------------------------------

def test_shm_ring_capacities_match_design():
    assert _ext.ShmRingWriteEvent.capacity == 16384
    assert _ext.ShmRingQueryEvent.capacity == 16384
    assert _ext.ShmRingReplayOutcome.capacity == 8192


def test_shm_ring_region_bytes_at_least_slot_array():
    """REGION_BYTES is the static byte size of SpscRing<T, Capacity> —
    must be at least sizeof(slot_array). The two cacheline-padded
    indices (128B) plus alignment slack is the only overhead."""
    write_min = 568 * 16384
    # Phase S3: QueryEvent grew 544 → 736 to carry the simplex candidate set.
    query_min = 736 * 16384
    replay_min = 96 * 8192
    assert _ext.ShmRingWriteEvent.REGION_BYTES >= write_min
    assert _ext.ShmRingQueryEvent.REGION_BYTES >= query_min
    assert _ext.ShmRingReplayOutcome.REGION_BYTES >= replay_min
    # Sanity upper bound: never more than slot array + 1KB of
    # cacheline padding.
    assert _ext.ShmRingWriteEvent.REGION_BYTES <= write_min + 1024
    assert _ext.ShmRingQueryEvent.REGION_BYTES <= query_min + 1024
    assert _ext.ShmRingReplayOutcome.REGION_BYTES <= replay_min + 1024
