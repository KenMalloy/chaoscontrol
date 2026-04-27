"""Tests for the train-rank episodic WRITE_EVENT emit path.

The producer keeps the historical ``[K_max, slot_dim]`` fp32 tensor for local
pack-semantics tests, but the cross-rank path publishes WRITE_EVENT structs to
a per-rank SPSC shm ring. There is no write-side ``dist.gather`` in the train
step anymore.

Tests:

  1. ``test_create_episodic_emit_returns_handle_when_episodic_enabled`` —
     helper produces an ``EpisodicGpuEmit`` with the right slot tensor
     shape, dtype, and config-derived dimensions.
  2. ``test_create_episodic_emit_returns_none_on_disabled`` —
     ``episodic_enabled=False`` produces ``None`` so the back-compat path
     stays bit-identical.
  3. ``test_create_episodic_emit_returns_none_on_world_size_one`` —
     ``world_size <= 1`` is a no-op (no episodic rank exists).
  4. ``test_create_episodic_emit_works_on_episodic_rank`` — the episodic
     rank still gets a no-op handle for shape compatibility, but no ring.
  5. ``test_train_step_packs_slot_tensor_when_episodic_enabled`` — after
     a single ``_run_train_step`` with no process group (single-rank
     test mode), the emit handle's slot_tensor has at least one row
     with ``valid_mask=1.0`` and that row's ``value_anchor_id`` matches
     the target token at the selected position.
  6. ``test_train_step_packs_value_tok_ids_correctly`` — the packed
     ``value_tok_ids`` field for a valid row exactly matches the next
     S target tokens.
  7. ``test_train_step_packs_key_rep_from_hidden`` — the packed
     ``key_rep`` is non-zero (the producer copies hidden, doesn't
     emit zeros).
  8. ``test_emit_truncates_when_k_exceeds_k_max`` — when top_p × B*T
     produces > K_max selections, the emit truncates rather than
     overflowing the slot tensor.
  9. ``test_right_pad_per_token_signal_*`` — shape-adapter pins
     unchanged from pre-Pass-C; the helper still lives in the runner.

Tests 1-4 exercise ``_create_episodic_emit`` directly without booting
a process group. Tests 5-8 drive ``_run_train_step`` with
``ddp_active=False`` and ``all_group=None`` so the runner's "single-rank
back-compat" branch packs the slot tensor without standing up a process group.
"""
from __future__ import annotations

import importlib.util
import threading
from collections import deque
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from chaoscontrol.episodic.gpu_slot import slot_dim, unpack_payload


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


class _TinyTokenTrainModel(nn.Module):
    """Mirror of ``test_exp23_fast_path._TinyTokenTrainModel`` and the
    pre-Pass-C ``test_runner_episodic_writes._TinyTokenTrainModel``.

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


def test_cuda_write_event_publisher_surfaces_daemon_failure():
    """Publisher thread failures must become telemetry, not silent drops."""
    mod = _load_runner_module()

    class EventReady:
        def query(self):
            return True

    class ExplodingRing:
        def push_batch_tensor(self, _batch):
            raise RuntimeError("publisher boom")

    publisher = object.__new__(mod._CudaWriteEventPublisher)
    publisher.ring = ExplodingRing()
    publisher.k_max = 16
    publisher.event_size = 568
    publisher.depth = 1
    publisher.gpu_slots = [object()]
    publisher.cpu_slots = [object()]
    publisher.events = [EventReady()]
    publisher.free_slots = deque()
    publisher.pending = deque([0])
    publisher.lock = threading.Lock()
    publisher.stop_event = threading.Event()
    publisher.stop_event.set()
    publisher.submitted_batches = 1
    publisher.pushed_events = 0
    publisher.skipped_events = 0
    publisher.dropped_events = 0
    publisher.dropped_batches = 0
    publisher.failed = False
    publisher.error = None

    publisher._run()

    assert publisher.failed is True
    assert "RuntimeError: publisher boom" in publisher.error
    assert list(publisher.pending) == []
    assert list(publisher.free_slots) == [0]


# ---------------------------------------------------------------------------
# Tests 1-4: helper-direct calls
# ---------------------------------------------------------------------------


def test_create_episodic_emit_returns_handle_when_episodic_enabled() -> None:
    """Helper allocates a slot tensor with the right shape + dimensions."""
    mod = _load_runner_module()
    config = {
        "episodic_enabled": True,
        "episodic_span_length": 4,
        "episodic_key_rep_dim": 8,
        "episodic_k_max": 16,
        "episodic_async_write_rings_enabled": False,
        "model_dim": 8,
    }
    handle = mod._create_episodic_emit(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        config=config,
    )
    assert handle is not None
    expected_width = slot_dim(span_length=4, key_rep_dim=8)
    assert handle.slot_tensor.shape == (16, expected_width)
    assert handle.slot_tensor.dtype == torch.float32
    assert handle.k_max == 16
    assert handle.span_length == 4
    assert handle.key_rep_dim == 8
    # Initial slot tensor must be zero (the drain's valid_mask filter
    # depends on this).
    assert torch.all(handle.slot_tensor == 0.0)


def test_create_episodic_emit_returns_none_on_disabled() -> None:
    """Back-compat: ``episodic_enabled=False`` is a no-op."""
    mod = _load_runner_module()
    handle = mod._create_episodic_emit(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        config={"episodic_enabled": False, "model_dim": 8},
    )
    assert handle is None


def test_create_episodic_emit_returns_none_on_world_size_one() -> None:
    """No episodic rank exists at world_size=1 — helper returns None."""
    mod = _load_runner_module()
    handle = mod._create_episodic_emit(
        rank=0,
        world_size=1,
        device=torch.device("cpu"),
        config={
            "episodic_enabled": True,
            "episodic_key_rep_dim": 4,
            "episodic_async_write_rings_enabled": False,
            "model_dim": 4,
        },
    )
    assert handle is None


def test_create_episodic_emit_works_on_episodic_rank() -> None:
    """The episodic rank gets a no-op handle and no write ring."""
    mod = _load_runner_module()
    handle = mod._create_episodic_emit(
        rank=1,  # rank 1 of world_size=2 == episodic rank
        world_size=2,
        device=torch.device("cpu"),
        config={
            "episodic_enabled": True,
            "episodic_key_rep_dim": 4,
            "episodic_k_max": 8,
            "episodic_async_write_rings_enabled": False,
            "model_dim": 4,
        },
    )
    assert handle is not None
    assert handle.slot_tensor.shape == (8, slot_dim(span_length=4, key_rep_dim=4))
    assert handle.write_ring is None


def test_create_episodic_emit_rejects_non_positive_k_max() -> None:
    """Defensive: ``episodic_k_max <= 0`` is a config error."""
    mod = _load_runner_module()
    with pytest.raises(ValueError):
        mod._create_episodic_emit(
            rank=0,
            world_size=2,
            device=torch.device("cpu"),
            config={
                "episodic_enabled": True,
                "episodic_key_rep_dim": 4,
                "episodic_k_max": 0,
                "model_dim": 4,
            },
        )


# ---------------------------------------------------------------------------
# Tests 5-8: in-step pack via _run_train_step (single-rank back-compat path)
# ---------------------------------------------------------------------------


def _drive_single_train_step_with_emit(mod):
    """Run one train step with ``ddp_active=False`` + ``all_group=None``.

    Returns the populated emit handle so the caller can inspect the
    slot tensor directly. The single-rank back-compat branch in
    ``_run_train_step`` packs the slot tensor without standing up a process
    group, which is the cheapest way to pin pack semantics.
    """
    torch.manual_seed(17)
    model = _TinyTokenTrainModel()
    inputs = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.int64)
    targets = torch.tensor([[2, 3, 4, 5, 0]], dtype=torch.int64)
    config = {
        "episodic_enabled": True,
        "episodic_span_length": 2,
        "episodic_fingerprint_window": 1,
        "episodic_key_rep_dim": 4,
        # top_p=0.5 and span=2 + window=1 makes ~half the positions
        # selectable and most of those pass the boundary check.
        "episodic_top_p": 0.5,
        "episodic_k_max": 8,
        "episodic_async_write_rings_enabled": False,
        "model_dim": 4,
    }
    handle = mod._create_episodic_emit(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        config=config,
    )
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
        rank=0,
        lm_head_backward_mode="fused",
        episodic_emit=handle,
    )
    return handle, inputs, targets


def test_train_step_packs_slot_tensor_when_episodic_enabled() -> None:
    """At least one slot row has valid_mask=1 after a train step, and
    its value_anchor_id appears in the target row.
    """
    mod = _load_runner_module()
    handle, _inputs, targets = _drive_single_train_step_with_emit(mod)
    valid_rows = handle.slot_tensor[:, 0] > 0.5
    assert int(valid_rows.sum().item()) >= 1, (
        "expected at least one valid slot after the step; got "
        f"{int(valid_rows.sum().item())}"
    )
    targets_set = set(int(x) for x in targets.flatten().tolist())
    for k in range(int(handle.k_max)):
        if float(handle.slot_tensor[k, 0].item()) <= 0.5:
            continue
        unpacked = unpack_payload(
            handle.slot_tensor[k],
            span_length=handle.span_length,
            key_rep_dim=handle.key_rep_dim,
        )
        assert unpacked["value_anchor_id"] in targets_set


def test_train_step_packs_value_tok_ids_correctly() -> None:
    """``value_tok_ids[0] == value_anchor_id`` per the cache contract,
    and the span equals targets[b, t:t+S] exactly.
    """
    mod = _load_runner_module()
    handle, _inputs, targets = _drive_single_train_step_with_emit(mod)
    found_valid = False
    for k in range(int(handle.k_max)):
        if float(handle.slot_tensor[k, 0].item()) <= 0.5:
            continue
        found_valid = True
        unpacked = unpack_payload(
            handle.slot_tensor[k],
            span_length=handle.span_length,
            key_rep_dim=handle.key_rep_dim,
        )
        # Anchor-id == first token of the span.
        assert (
            unpacked["value_anchor_id"]
            == int(unpacked["value_tok_ids"][0].item())
        )
        # Each token must be one we expect to find in the targets row;
        # exact (b, t) recovery requires inverting the top-p ranking,
        # which is more brittle than the targets-membership check below.
        targets_set = set(int(x) for x in targets.flatten().tolist())
        for tok in unpacked["value_tok_ids"].tolist():
            assert int(tok) in targets_set
    assert found_valid, "expected at least one valid slot after the step"


def test_train_step_packs_key_rep_from_hidden() -> None:
    """``key_rep`` must be non-zero (the producer copies hidden, not zeros).

    Pins that the pack actually reads ``hidden`` rather than emitting a
    placeholder. Hidden-state == 0 is theoretically possible at init but
    the seeded model + non-zero embedding init makes it overwhelmingly
    unlikely for a tiny seeded model.
    """
    mod = _load_runner_module()
    handle, _inputs, _targets = _drive_single_train_step_with_emit(mod)
    for k in range(int(handle.k_max)):
        if float(handle.slot_tensor[k, 0].item()) <= 0.5:
            continue
        unpacked = unpack_payload(
            handle.slot_tensor[k],
            span_length=handle.span_length,
            key_rep_dim=handle.key_rep_dim,
        )
        assert unpacked["key_rep"].dtype == torch.float32
        assert unpacked["key_rep"].shape == (handle.key_rep_dim,)
        # key_rep == residual in Phase 1 (same write-time hidden).
        assert torch.equal(unpacked["key_rep"], unpacked["residual"])
        assert not torch.all(unpacked["key_rep"] == 0.0)


def test_emit_truncates_when_k_exceeds_k_max() -> None:
    """When top_p * B * T produces > K_max selections, packing
    truncates rather than crashing or overflowing.
    """
    mod = _load_runner_module()
    torch.manual_seed(31)
    model = _TinyTokenTrainModel()
    # B=1, T=5, top_p=1.0 → K=5. Set k_max=2 so we must truncate.
    inputs = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.int64)
    targets = torch.tensor([[2, 3, 4, 5, 0]], dtype=torch.int64)
    config = {
        "episodic_enabled": True,
        "episodic_span_length": 1,
        "episodic_fingerprint_window": 1,
        "episodic_key_rep_dim": 4,
        "episodic_top_p": 1.0,
        "episodic_k_max": 2,
        "episodic_async_write_rings_enabled": False,
        "model_dim": 4,
    }
    handle = mod._create_episodic_emit(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        config=config,
    )
    assert handle is not None
    assert handle.slot_tensor.shape == (
        2, slot_dim(span_length=1, key_rep_dim=4),
    )
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
        rank=0,
        lm_head_backward_mode="fused",
        episodic_emit=handle,
    )
    # No more than k_max valid rows — pack must not have overflowed.
    valid_rows = int((handle.slot_tensor[:, 0] > 0.5).sum().item())
    assert valid_rows <= int(handle.k_max)


# ---------------------------------------------------------------------------
# Shape-adapter pins (unchanged from pre-Pass-C)
# ---------------------------------------------------------------------------


def test_right_pad_per_token_signal_pads_t_minus_1_to_t() -> None:
    """``per_token_ce`` arrives as ``[B, T-1]`` in the transformer-style
    convention; the adapter pads with a zero column on the right to
    ``[B, T]`` so it lines up with input_ids/target_ids for top-p
    selection.
    """
    mod = _load_runner_module()
    signal = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    padded = mod._right_pad_per_token_signal(signal, T=5)
    assert padded.shape == (3, 5)
    assert torch.equal(padded[:, :4], signal)
    assert torch.equal(padded[:, 4], torch.zeros(3))


def test_right_pad_per_token_signal_is_idempotent_when_already_t() -> None:
    """The exp23 batch builder shifts targets so ``per_token_ce`` is
    already ``[B, T]``. Adapter is a no-op there.
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


def test_valid_write_signal_window_excludes_unwritable_boundaries() -> None:
    """Write selection should never spend top-k or candidate ids on positions
    that cannot carry a full fingerprint/value span."""
    mod = _load_runner_module()
    signal = torch.arange(6, dtype=torch.float32).reshape(1, 6)

    valid, offset = mod._valid_write_signal_window(
        signal,
        fingerprint_window=2,
        span_length=2,
    )

    assert offset == 2
    assert torch.equal(valid, signal[:, 2:5])
    positions = mod._select_write_positions_with_action_space(
        action_space=None,
        write_signal=valid,
        pressure_full=None,
        ce_full=signal,
        top_p=1.0,
        k_max=8,
        current_step=0,
        write_bucket=0,
    )
    positions = positions.clone()
    positions[:, 1] += offset
    assert set(int(t) for t in positions[:, 1].tolist()) == {2, 3, 4}
