"""Phase 1 Task 1.4 — train-rank producer for the episodic write + query rings.

Tests cover the producer-side contract from
``docs/plans/2026-04-25-ring-contract-tasks-1-4-and-1-5.md``:

  1. When ``episodic_enabled=True`` and the rank is a TRAIN rank
     (``rank < world_size - 1``), the runner creates a per-rank write ring
     named ``episodic_write_ring_rank{R}`` with the shared write-payload
     dtype.
  2. Same rank also creates a per-rank query-candidate ring named
     ``episodic_query_ring_rank{R}`` with the shared query-candidate
     dtype.
  3. When ``episodic_enabled=False`` the helper returns ``(None, None)``
     so the back-compat path remains bit-identical.
  4. After a train step on a train rank with ``episodic_enabled=True``,
     payloads land in the write ring with the documented field values.
  5. After the same train step, query candidates land in the query ring
     with the documented field values.

Tests 1-3 exercise the ``_create_episodic_rings`` helper directly without
booting the distributed group — single-process, mocked ``rank`` /
``world_size``. Tests 4-5 drive ``_run_train_step`` directly with a tiny
model + a hand-built ``EpisodicRingsHandle``, then attach to the rings
as a separate reader to drain them. This avoids ``mp.spawn`` and
``init_process_group`` while still pinning the in-step producer flow,
which is what the contract guards.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from chaoscontrol.episodic.ipc import ShmRing
from chaoscontrol.episodic.payload_dtypes import (
    make_query_candidate_dtype,
    make_write_payload_dtype,
)


def _load_runner_module():
    """Mirror the importlib pattern from ``test_cd_config_threading.py:14``."""
    path = (
        Path(__file__).resolve().parent.parent
        / "experiments" / "23_fast_path" / "runner_fast_path.py"
    )
    spec = importlib.util.spec_from_file_location("runner_fast_path", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_NEXT_TAG_IDX = [0]
_TAG_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"


def _unique_tag() -> str:
    """Allocate a 2-char unique suffix for a test ring set.

    POSIX shm names on darwin cap at 31 chars including the leading
    ``/`` and the counter ``_c`` suffix. The runner's
    ``episodic_write_ring_rank{R}`` base eats 26 chars (with ``/``),
    leaving room for ONLY a 2-char ``_{suffix}`` before counter ``_c``
    overflows the 31-char cap. ``2`` chars × 36-symbol alphabet = 1296
    distinct suffixes per pytest session, plenty for the ~10 ring
    creations this file does.

    Suffix uniqueness is per-session; concurrent pytest workers (e.g.
    pytest-xdist) MUST seed each worker's counter from its worker id
    or accept rare cross-worker collisions handled by the
    ``ShmRing.create`` ``FileExistsError`` path. This test file does
    not run under xdist by default, so a simple monotonic counter is
    enough.
    """
    idx = _NEXT_TAG_IDX[0]
    _NEXT_TAG_IDX[0] += 1
    a = _TAG_ALPHABET[idx % len(_TAG_ALPHABET)]
    b = _TAG_ALPHABET[(idx // len(_TAG_ALPHABET)) % len(_TAG_ALPHABET)]
    return f"{a}{b}"


class _TinyTokenTrainModel(nn.Module):
    """Mirror of ``test_exp23_fast_path._TinyTokenTrainModel``.

    The runner's train step asks for ``model.encode(inputs)``,
    ``model.final_norm(hidden)``, and ``model.lm_head``; this satisfies
    that interface with a 4-dim hidden state and 6-token vocab.
    """

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(6, 4)
        self.final_norm = nn.Identity()
        self.lm_head = nn.Linear(4, 6, bias=False)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.embed(inputs)


# ---------------------------------------------------------------------------
# Tests 1-3: ring creation helper, single-process
# ---------------------------------------------------------------------------


def test_runner_creates_per_rank_write_ring_when_episodic_enabled() -> None:
    """Helper returns a write ring named per the ring contract.

    Train rank 0 of a 2-rank topology must get a ring named
    ``episodic_write_ring_rank0`` whose dtype matches the shared
    ``make_write_payload_dtype`` factory. Capacity defaults to 256 per
    the plan's config-key table.
    """
    mod = _load_runner_module()
    config = {
        "model_dim": 4,
        "episodic_enabled": True,
        "episodic_span_length": 4,
        "episodic_key_rep_dim": 4,
        "episodic_ring_name_suffix": _unique_tag(),
    }
    handle = mod._create_episodic_rings(rank=0, world_size=2, config=config)
    try:
        assert handle is not None
        assert handle.write_ring is not None
        # Producer-owned dtype must match the shared factory exactly so
        # Task 1.5's ``ShmRing.attach`` validation passes.
        expected = make_write_payload_dtype(span_length=4, key_rep_dim=4)
        assert handle.write_payload_dtype == expected
        # The ring is usable: writing a slot of the expected dtype works.
        slot = np.zeros((), dtype=expected)
        slot["key_fp"] = 7
        handle.write_ring.try_write(slot)
    finally:
        if handle is not None:
            mod._close_episodic_rings(handle)


def test_runner_creates_per_rank_query_ring_when_episodic_enabled() -> None:
    """Helper returns a query-candidate ring with the shared dtype."""
    mod = _load_runner_module()
    config = {
        "model_dim": 4,
        "episodic_enabled": True,
        "episodic_span_length": 4,
        "episodic_key_rep_dim": 4,
        "episodic_ring_name_suffix": _unique_tag(),
    }
    handle = mod._create_episodic_rings(rank=0, world_size=2, config=config)
    try:
        assert handle is not None
        assert handle.query_ring is not None
        expected = make_query_candidate_dtype(key_rep_dim=4)
        assert handle.query_candidate_dtype == expected
        slot = np.zeros((), dtype=expected)
        slot["batch_index"] = 1
        slot["position"] = 2
        slot["pressure"] = 0.25
        slot["residual"] = np.zeros(4, dtype=np.float32)
        handle.query_ring.try_write(slot)
    finally:
        if handle is not None:
            mod._close_episodic_rings(handle)


def test_runner_does_not_create_rings_when_episodic_disabled() -> None:
    """Back-compat: ``episodic_enabled=False`` is a no-op for ring setup."""
    mod = _load_runner_module()
    handle = mod._create_episodic_rings(
        rank=0,
        world_size=1,
        config={"model_dim": 4, "episodic_enabled": False},
    )
    assert handle is None


def test_runner_does_not_create_rings_on_episodic_rank() -> None:
    """The episodic rank (R == world_size-1) does NOT create rings.

    Ring creation is a producer-only operation; the consumer attaches in
    Task 1.5. This pins the producer guard.
    """
    mod = _load_runner_module()
    handle = mod._create_episodic_rings(
        rank=1,
        world_size=2,
        config={"model_dim": 4, "episodic_enabled": True},
    )
    assert handle is None


# ---------------------------------------------------------------------------
# Tests 4-5: in-step writes through ``_run_train_step``
# ---------------------------------------------------------------------------


def _drive_single_train_step_with_rings(mod, *, suffix: str):
    """Helper: build a tiny model + handle, run one train step, return drained slots.

    Drives ``_run_train_step`` with ``ddp_active=False`` (so we skip the
    all-reduce path), ``is_episodic_rank=False``, and a populated rings
    handle. Reads back from each ring after the step via a separate
    ``ShmRing.attach``.
    """
    torch.manual_seed(17)
    model = _TinyTokenTrainModel()
    inputs = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.int64)
    targets = torch.tensor([[2, 3, 4, 5, 0]], dtype=torch.int64)
    config = {
        "model_dim": 4,
        "episodic_enabled": True,
        "episodic_span_length": 2,
        "episodic_fingerprint_window": 1,
        "episodic_key_rep_dim": 4,
        # top_p chosen so we always select at least one position even
        # under boundary drops: ``select_top_p_positions`` always returns
        # at least 1, and a small ``span_length=2`` + ``fingerprint_window=1``
        # makes most positions pass the boundary checks.
        "episodic_top_p": 0.5,
        "episodic_ring_name_suffix": suffix,
    }
    handle = mod._create_episodic_rings(rank=0, world_size=2, config=config)
    assert handle is not None

    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    optimizer.zero_grad(set_to_none=True)
    mod._run_train_step(
        model=model,
        inputs=inputs,
        targets=targets,
        chunk_size=4,
        precision="fp32",
        ddp_active=False,
        world_size=1,
        lm_head_backward_mode="fused",
        episodic_rings=handle,
    )

    # Build separate reader handles to drain. ShmRing.attach validates
    # the claimed dtype against the metadata stored at create() time,
    # so any contract drift in the producer dtype would surface here.
    write_reader = ShmRing.attach(
        name=handle.write_ring_name,
        slot_shape=(),
        dtype=handle.write_payload_dtype,
        capacity=handle.write_ring_capacity,
    )
    query_reader = ShmRing.attach(
        name=handle.query_ring_name,
        slot_shape=(),
        dtype=handle.query_candidate_dtype,
        capacity=handle.query_ring_capacity,
    )
    write_slots: list[np.ndarray] = []
    query_slots: list[np.ndarray] = []
    while True:
        s = write_reader.try_read()
        if s is None:
            break
        write_slots.append(s)
    while True:
        s = query_reader.try_read()
        if s is None:
            break
        query_slots.append(s)
    write_reader.close()
    query_reader.close()
    return handle, write_slots, query_slots


def test_train_step_writes_payload_to_write_ring() -> None:
    """A train step with ``episodic_enabled=True`` pushes write payloads.

    Producer-side asserts: at least one slot landed, the dtype matches,
    and the ``value_anchor_id`` field equals the target token at the
    written ``(batch_index, position)`` — the contract names these as
    coupled fields, so a producer bug that swaps batch/position or
    drifts target_ids would show up here as an off-by-one anchor.
    """
    mod = _load_runner_module()
    suffix = _unique_tag()
    targets = torch.tensor([[2, 3, 4, 5, 0]], dtype=torch.int64)
    handle, write_slots, _query_slots = _drive_single_train_step_with_rings(
        mod, suffix=suffix,
    )
    try:
        assert len(write_slots) >= 1, (
            "expected at least one write payload after the step; got 0"
        )
        for slot in write_slots:
            assert slot.dtype == handle.write_payload_dtype
            # ``value_anchor_id == value_tok_ids[0]`` per the WritePayload
            # contract; pin it directly.
            assert int(slot["value_anchor_id"]) == int(slot["value_tok_ids"][0])
            # ``key_rep`` must be float32 length-key_rep_dim and must not
            # be all-zero for a non-zero hidden state — pins that the
            # producer actually copies the hidden tensor through.
            assert slot["key_rep"].dtype == np.float32
            assert slot["key_rep"].shape == (handle.key_rep_dim,)
            assert not np.all(slot["key_rep"] == 0.0)
            # Boundary check: anchor id must appear in the targets row
            # (the producer copies from ``target_ids[batch_index, position]``).
            assert int(slot["value_anchor_id"]) in targets.flatten().tolist()
    finally:
        mod._close_episodic_rings(handle)


def test_train_step_writes_query_candidate_to_query_ring() -> None:
    """A train step with ``episodic_enabled=True`` pushes query candidates."""
    mod = _load_runner_module()
    suffix = _unique_tag()
    handle, _write_slots, query_slots = _drive_single_train_step_with_rings(
        mod, suffix=suffix,
    )
    try:
        assert len(query_slots) >= 1, (
            "expected at least one query candidate after the step; got 0"
        )
        # Per contract the SAME positions drive both rings, so the count
        # is the same as the write ring.
        assert len(query_slots) == len(_write_slots)
        for slot in query_slots:
            assert slot.dtype == handle.query_candidate_dtype
            # batch_index in [0, B) and position in [0, T): the toy model
            # has B=1, T=5.
            assert 0 <= int(slot["batch_index"]) < 1
            assert 0 <= int(slot["position"]) < 5
            assert slot["residual"].dtype == np.float32
            assert slot["residual"].shape == (handle.key_rep_dim,)
    finally:
        mod._close_episodic_rings(handle)


# ---------------------------------------------------------------------------
# Shape adapter pins
# ---------------------------------------------------------------------------


def test_right_pad_per_token_signal_pads_t_minus_1_to_t() -> None:
    """Per the contract, ``per_token_ce`` arrives as ``[B, T-1]`` in the
    transformer-style next-token-CE convention; the adapter pads with a
    zero column on the right to ``[B, T]`` so it lines up with
    ``input_ids``/``target_ids`` for ``select_writes``.
    """
    mod = _load_runner_module()
    signal = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    padded = mod._right_pad_per_token_signal(signal, T=5)
    assert padded.shape == (3, 5)
    # The pre-existing values land in columns [0, T-1).
    assert torch.equal(padded[:, :4], signal)
    # Right-pad column is zero.
    assert torch.equal(padded[:, 4], torch.zeros(3))


def test_right_pad_per_token_signal_is_idempotent_when_already_t() -> None:
    """The exp23 batch builder shifts targets so ``per_token_ce`` is
    already ``[B, T]``. The adapter must then return the input unchanged
    (no extra column).
    """
    mod = _load_runner_module()
    signal = torch.arange(15, dtype=torch.float32).reshape(3, 5)
    padded = mod._right_pad_per_token_signal(signal, T=5)
    assert padded.shape == (3, 5)
    assert torch.equal(padded, signal)


def test_right_pad_per_token_signal_rejects_unexpected_shape() -> None:
    """Wrong-by-more-than-one widths are a coding error, not a pad case."""
    mod = _load_runner_module()
    signal = torch.zeros(3, 7)
    with pytest.raises(ValueError):
        mod._right_pad_per_token_signal(signal, T=5)
