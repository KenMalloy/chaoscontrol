"""GPU-resident episodic IPC slot format (Perf Pass C).

A single fp32 tensor of shape ``[K_max, slot_dim]`` carries every
episodic-cache write payload AND query candidate from train ranks to the
episodic rank via a single ``dist.gather`` collective per training step.
Replaces the POSIX shm rings from Phase 1 Tasks 1.4 + 1.5.

Slot layout (canonical — see
``docs/plans/2026-04-25-perf-pass-c-gpu-resident-ipc.md`` for the design):

  ====== ===================== ========= ============================
  Offset Field                 Width     Interpretation
  ====== ===================== ========= ============================
  0      valid_mask            1         fp32 (0.0 = empty, 1.0 = valid)
  1      pressure              1         fp32
  2..3   key_fp                2         int64 reinterpret (rolling-hash fingerprint)
  4..5   value_anchor_id       2         int64 reinterpret (anchor token id)
  6..6+2S-1 value_tok_ids      2*S       int64 reinterpret of S elements
  6+2S.. key_rep               D         fp32 (write-time residual)
  6+2S+D.. residual            D         fp32 (query-time residual)
  ====== ===================== ========= ============================

  ``slot_dim = 6 + 2*S + 2*D``.

Int64 fields are reinterpreted via ``view(torch.int64)`` over an
even-aligned fp32 slice. The slot tensor is always contiguous fp32 so the
view is well-formed; tests pin this with INT64_MAX round-trips.

The collapsed write+query channel in this module subsumes both the Task
1.4 write-payload struct and the Task 1.5 query-candidate struct. Same
slot serves the cache writer and the controller queue; the episodic-rank
drain routes the unpacked dict to both.
"""
from __future__ import annotations

from typing import Any

import torch


SLOT_DIM_BASE = 6
"""Number of fp32 cells in the fixed prefix (valid_mask + pressure +
key_fp(2) + value_anchor_id(2)). The trailing ``2*S + 2*D`` cells
depend on ``span_length`` and ``key_rep_dim``."""


def slot_dim(*, span_length: int, key_rep_dim: int) -> int:
    """Total fp32 element count for one slot.

    ``slot_dim = SLOT_DIM_BASE + 2*S + 2*D``. At the design defaults
    ``S=4, D=256`` this is 526 fp32 = 2104 bytes. Both the train-rank
    emit tensor and the episodic-rank gather buffer use this width.
    """
    if span_length <= 0:
        raise ValueError(
            f"span_length must be positive; got {span_length}"
        )
    if key_rep_dim <= 0:
        raise ValueError(
            f"key_rep_dim must be positive; got {key_rep_dim}"
        )
    width = SLOT_DIM_BASE + 2 * int(span_length) + 2 * int(key_rep_dim)
    # ``slot_dim`` MUST stay even. Each row in a ``[K_max, slot_dim]``
    # tensor starts at offset ``k * slot_dim * 4`` bytes; for the int64
    # reinterpret (``slot[i:i+2].view(torch.int64)``) to land on an
    # 8-byte boundary at every ``k``, ``slot_dim`` must be even.
    # Adding a single fp32 field to the layout would silently misalign
    # odd-numbered slots — depending on GPU arch and PyTorch version,
    # this either reads garbage or hard-faults. If you ever extend the
    # slot, either add fields in pairs or pad to keep ``slot_dim`` even.
    if width % 2 != 0:
        raise AssertionError(
            f"slot_dim must be even (got {width}); int64 reinterpret "
            "requires every row to start on an 8-byte boundary inside "
            "the [K_max, slot_dim] tensor. Add fields in pairs or pad "
            "to keep the total even."
        )
    return width


def make_slot_tensor(
    *,
    k_max: int,
    span_length: int,
    key_rep_dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Allocate a zero-initialized ``[k_max, slot_dim]`` slot tensor on ``device``.

    Zero-initialized so every row's ``valid_mask`` (offset 0) starts at
    0.0 — the episodic-rank drain filters by ``valid_mask > 0.5``, so an
    untouched row from a step with fewer than ``k_max`` selected
    positions is correctly treated as empty. Caller is responsible for
    re-zeroing the tensor at the start of each step (or freshly
    allocating one) before packing.

    The dtype is pinned to fp32; the int64-via-view reinterpret only
    works on a 32-bit base type. Callers should not pass ``dtype=fp16``.
    """
    if k_max <= 0:
        raise ValueError(f"k_max must be positive; got {k_max}")
    if dtype != torch.float32:
        raise ValueError(
            "make_slot_tensor requires dtype=torch.float32 so the "
            "int64-via-view reinterpret aligns; got "
            f"dtype={dtype}"
        )
    width = slot_dim(span_length=span_length, key_rep_dim=key_rep_dim)
    return torch.zeros(int(k_max), int(width), device=device, dtype=dtype)


def pack_payload(
    slot: torch.Tensor,
    *,
    valid_mask: float,
    pressure: float,
    key_fp: int,
    value_anchor_id: int,
    value_tok_ids: torch.Tensor,
    key_rep: torch.Tensor,
    residual: torch.Tensor,
    span_length: int,
    key_rep_dim: int,
) -> None:
    """Pack one payload into a single slot tensor of shape ``[slot_dim]``.

    All writes are in-place. ``slot`` is expected to be a fp32 view into
    a row of a ``[K_max, slot_dim]`` parent tensor; it must be
    contiguous so the int64 reinterpret over the (2,) slices at offsets
    2/4/6..6+2S works.

    Args:
        slot: 1-D contiguous fp32 tensor of length ``slot_dim``.
        valid_mask: 0.0 or 1.0; the episodic-rank drain filters by
            ``valid_mask > 0.5``.
        pressure: scalar pressure × per-token CE at the write position.
        key_fp: int64 rolling-hash fingerprint of the preceding window.
        value_anchor_id: int64 anchor token id (== ``value_tok_ids[0]``).
        value_tok_ids: 1-D int64 tensor of length ``span_length`` —
            the next S target tokens. Will be ``.to(torch.int64)``-coerced
            via the int64 view.
        key_rep: 1-D fp32 tensor of length ``key_rep_dim`` — the write-
            time residual that doubles as the cache key.
        residual: 1-D fp32 tensor of length ``key_rep_dim`` — the query-
            time residual that goes to the controller queue. In Phase 1
            this is the SAME vector as ``key_rep`` (computed from the
            same source); kept as a separate arg so Phase 2+ can decouple.
        span_length: S, must match the value embedded in ``slot.shape``.
        key_rep_dim: D, must match the value embedded in ``slot.shape``.
    """
    expected_width = slot_dim(span_length=span_length, key_rep_dim=key_rep_dim)
    if slot.dim() != 1 or slot.shape[0] != expected_width:
        raise ValueError(
            f"slot must have shape ({expected_width},); got "
            f"{tuple(slot.shape)}"
        )
    if slot.dtype != torch.float32:
        raise ValueError(
            f"slot must be fp32; got dtype={slot.dtype}"
        )
    if not slot.is_contiguous():
        raise ValueError("slot must be contiguous for the int64 reinterpret")
    if value_tok_ids.shape != (int(span_length),):
        raise ValueError(
            f"value_tok_ids must have shape ({span_length},); got "
            f"{tuple(value_tok_ids.shape)}"
        )
    if key_rep.shape != (int(key_rep_dim),):
        raise ValueError(
            f"key_rep must have shape ({key_rep_dim},); got "
            f"{tuple(key_rep.shape)}"
        )
    if residual.shape != (int(key_rep_dim),):
        raise ValueError(
            f"residual must have shape ({key_rep_dim},); got "
            f"{tuple(residual.shape)}"
        )
    S = int(span_length)
    D = int(key_rep_dim)
    # fp32 fields: simple in-place writes.
    slot[0] = float(valid_mask)
    slot[1] = float(pressure)
    # int64 fields via view reinterpret on aligned (even-offset, 2-cell) slices.
    slot[2:4].view(torch.int64)[0] = int(key_fp)
    slot[4:6].view(torch.int64)[0] = int(value_anchor_id)
    # value_tok_ids: 2*S fp32 cells reinterpret as S int64s.
    slot[6:6 + 2 * S].view(torch.int64).copy_(
        value_tok_ids.detach().to(dtype=torch.int64)
    )
    # key_rep + residual: contiguous fp32 spans of width D each.
    slot[6 + 2 * S:6 + 2 * S + D].copy_(
        key_rep.detach().to(dtype=torch.float32)
    )
    slot[6 + 2 * S + D:6 + 2 * S + 2 * D].copy_(
        residual.detach().to(dtype=torch.float32)
    )


def unpack_payload(
    slot: torch.Tensor,
    *,
    span_length: int,
    key_rep_dim: int,
) -> dict[str, Any]:
    """Read a packed slot, returning a dict of cloned tensors / scalars.

    The caller is the episodic-rank drain. Returned tensors are
    ``.clone()``-d off the slot view so the slot buffer can be reused
    next step without aliasing the cache's internal storage. Scalar
    fields (``valid_mask``, ``pressure``, ``key_fp``, ``value_anchor_id``)
    are returned as Python ints/floats; ``.item()`` syncs are unavoidable
    since the cache schema and controller queue both want scalars, but
    K * 4 syncs per step at K_max=16 is microseconds.

    Args:
        slot: 1-D contiguous fp32 tensor of length ``slot_dim``.
        span_length: S, must match the value embedded in ``slot.shape``.
        key_rep_dim: D, must match the value embedded in ``slot.shape``.

    Returns:
        Dict with keys ``valid_mask`` (float), ``pressure`` (float),
        ``key_fp`` (int), ``value_anchor_id`` (int),
        ``value_tok_ids`` (int64 [S] tensor on ``slot.device``),
        ``key_rep`` (fp32 [D] tensor on ``slot.device``),
        ``residual`` (fp32 [D] tensor on ``slot.device``).
    """
    expected_width = slot_dim(span_length=span_length, key_rep_dim=key_rep_dim)
    if slot.dim() != 1 or slot.shape[0] != expected_width:
        raise ValueError(
            f"slot must have shape ({expected_width},); got "
            f"{tuple(slot.shape)}"
        )
    if slot.dtype != torch.float32:
        raise ValueError(
            f"slot must be fp32; got dtype={slot.dtype}"
        )
    if not slot.is_contiguous():
        raise ValueError("slot must be contiguous for the int64 reinterpret")
    S = int(span_length)
    D = int(key_rep_dim)
    return {
        "valid_mask": float(slot[0].item()),
        "pressure": float(slot[1].item()),
        "key_fp": int(slot[2:4].view(torch.int64)[0].item()),
        "value_anchor_id": int(slot[4:6].view(torch.int64)[0].item()),
        "value_tok_ids": slot[6:6 + 2 * S].view(torch.int64).clone(),
        "key_rep": slot[6 + 2 * S:6 + 2 * S + D].clone(),
        "residual": slot[6 + 2 * S + D:6 + 2 * S + 2 * D].clone(),
    }
