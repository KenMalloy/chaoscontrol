"""Slot maintenance commit records shared by memory-serving ranks.

The maintenance GPU owns expensive slot physics.  The packet-serving GPU owns
the low-latency cache that feeds residual packets.  A SlotCommit is the small
ordered record that lets the former update the latter without train-rank slot
reads or CPU tensor freight.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import torch

from .slot_table import SlotTable

class SlotCommitAction(IntEnum):
    APPEND = 0
    PRESERVE = 1
    DECAY = 2
    EVICT = 3
    REFRESH = 4
    QUARANTINE = 5
    DISTILL = 6
    RELEASE = 7

    def __str__(self) -> str:
        return self.name


SLOT_COMMIT_APPEND = SlotCommitAction.APPEND
SLOT_COMMIT_PRESERVE = SlotCommitAction.PRESERVE
SLOT_COMMIT_DECAY = SlotCommitAction.DECAY
SLOT_COMMIT_EVICT = SlotCommitAction.EVICT
SLOT_COMMIT_REFRESH = SlotCommitAction.REFRESH
SLOT_COMMIT_QUARANTINE = SlotCommitAction.QUARANTINE
SLOT_COMMIT_DISTILL = SlotCommitAction.DISTILL
SLOT_COMMIT_RELEASE = SlotCommitAction.RELEASE

SLOT_COMMIT_CODE_TO_ACTION = {int(action): action for action in SlotCommitAction}


class SlotCommitDType(IntEnum):
    FLOAT32 = 0
    FLOAT16 = 1
    BFLOAT16 = 2


SLOT_COMMIT_DTYPE_TO_CODE = {
    torch.float32: SlotCommitDType.FLOAT32,
    torch.float16: SlotCommitDType.FLOAT16,
    torch.bfloat16: SlotCommitDType.BFLOAT16,
}
SLOT_COMMIT_CODE_TO_DTYPE = {
    SlotCommitDType.FLOAT32: torch.float32,
    SlotCommitDType.FLOAT16: torch.float16,
    SlotCommitDType.BFLOAT16: torch.bfloat16,
}


def _coerce_action(action: SlotCommitAction | str | int) -> SlotCommitAction:
    if isinstance(action, SlotCommitAction):
        return action
    if isinstance(action, str):
        try:
            return SlotCommitAction[action]
        except KeyError as exc:
            raise ValueError(f"unknown SlotCommit action: {action}") from exc
    try:
        return SlotCommitAction(int(action))
    except ValueError as exc:
        raise ValueError(f"unknown SlotCommit action code: {action}") from exc


@dataclass(frozen=True)
class SlotCommit:
    """One ordered slot mutation from the maintenance rank to the packet rank."""

    slot_id: int
    action: SlotCommitAction | str | int
    step: int
    base_generation: int | None
    new_generation: int
    bucket_id: int = -1
    event_id: int = 0
    survival_factor: float = 1.0
    tensor: torch.Tensor | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        action = _coerce_action(self.action)
        base_generation = self.base_generation
        if action is SlotCommitAction.APPEND:
            if base_generation is not None and int(base_generation) >= 0:
                raise ValueError("APPEND SlotCommit must use base_generation=None")
            base_generation = None
        elif base_generation is None:
            raise ValueError(f"{action.name} SlotCommit requires base_generation")
        else:
            base_generation = int(base_generation)
        object.__setattr__(self, "slot_id", int(self.slot_id))
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "step", int(self.step))
        object.__setattr__(self, "base_generation", base_generation)
        object.__setattr__(self, "new_generation", int(self.new_generation))
        object.__setattr__(self, "bucket_id", int(self.bucket_id))
        object.__setattr__(self, "event_id", int(self.event_id))
        object.__setattr__(self, "survival_factor", float(self.survival_factor))
        object.__setattr__(self, "reason", str(self.reason))


def _resolve_commit_slot(table: SlotTable, commit: SlotCommit) -> tuple[int | None, str]:
    if table.record(commit.slot_id) is not None:
        return commit.slot_id, "slot_id"
    if commit.event_id == 0:
        return None, "missing_slot"
    for sid in table.active_slot_ids():
        rec = table.record(sid)
        if rec is None:
            continue
        if int(rec.event_id) != commit.event_id:
            continue
        if commit.bucket_id >= 0 and int(rec.bucket_id) != commit.bucket_id:
            continue
        return sid, "event_id"
    return None, "missing_slot"


def _set_generation(table: SlotTable, slot_id: int, generation: int) -> None:
    rec = table.record(int(slot_id))
    if rec is not None:
        rec.write_generation = int(generation)


def apply_slot_commit_to_model(model: Any, commit: SlotCommit) -> tuple[bool, str]:
    """Apply a maintenance commit to a packet-serving model mirror.

    Returns ``(accepted, reason)``.  Generation mismatch is a hard stale-drop:
    the packet rank keeps serving its latest complete cache instead of applying
    a mutation computed against a different slot version.
    """

    outer = model.outer_model
    table = outer.table
    if commit.action is SlotCommitAction.APPEND:
        raise ValueError("APPEND commits must use apply_append_slot_commit_to_model")
    if commit.base_generation is None:
        raise ValueError(f"{commit.action.name} commit requires base_generation")
    slot_id, resolve_reason = _resolve_commit_slot(table, commit)
    if slot_id is None:
        return False, resolve_reason
    rec = table.record(slot_id)
    if rec is None:
        return False, "missing_record"
    if int(rec.write_generation) != commit.base_generation:
        return False, "stale_generation"

    action = commit.action
    if action is SLOT_COMMIT_REFRESH:
        if commit.tensor is None:
            return False, "missing_tensor"
        table.replace_tensor(slot_id, commit.tensor, bump_generation=False)
        _set_generation(table, slot_id, commit.new_generation)
        return True, "refreshed"
    if action is SLOT_COMMIT_DECAY:
        table.scale_survival(slot_id, commit.survival_factor)
        _set_generation(table, slot_id, commit.new_generation)
        return True, "decayed"
    if action is SLOT_COMMIT_QUARANTINE:
        table.quarantine(slot_id)
        _set_generation(table, slot_id, commit.new_generation)
        return True, "quarantined"
    if action is SLOT_COMMIT_RELEASE:
        table.release(slot_id)
        _set_generation(table, slot_id, commit.new_generation)
        return True, "released"
    if action in {SLOT_COMMIT_EVICT, SLOT_COMMIT_DISTILL}:
        if action is SLOT_COMMIT_DISTILL and commit.tensor is not None:
            latent_traces = getattr(outer, "_latent_traces", None)
            if latent_traces is not None:
                latent_traces.append(
                    {
                        "bucket_id": int(commit.bucket_id),
                        "centroid_contrib": commit.tensor.detach().clone(),
                    }
                )
                max_latent = int(outer.max_slots or 0)
                while max_latent > 0 and len(latent_traces) > max_latent:
                    latent_traces.pop(0)
            prototypes = getattr(model, "bucket_prototypes_module", None)
            proto_buf = getattr(prototypes, "prototypes", None)
            if prototypes is not None and proto_buf is not None:
                bucket_id = int(commit.bucket_id)
                k_max = int(getattr(prototypes, "k_max", 0))
                if 0 <= bucket_id < k_max:
                    value = commit.tensor.detach().reshape(-1, commit.tensor.shape[-1])
                    value = value.to(device=proto_buf.device, dtype=proto_buf.dtype)
                    prototypes.update(bucket_id, value)
        table.retire(slot_id, reason="distilled" if action is SLOT_COMMIT_DISTILL else "evicted")
        _set_generation(table, slot_id, commit.new_generation)
        return True, "distilled" if action is SLOT_COMMIT_DISTILL else "evicted"
    if action is SLOT_COMMIT_PRESERVE:
        return True, "preserved"
    return False, "unknown_action"


def apply_append_slot_commit_to_model(model: Any, commit: SlotCommit) -> tuple[bool, str]:
    """Apply a packet-rank append commit to a maintenance-rank replica."""

    outer = model.outer_model
    table = outer.table
    if commit.action is not SLOT_COMMIT_APPEND:
        raise ValueError("apply_append_slot_commit_to_model only accepts APPEND commits")
    if commit.tensor is None:
        return False, "missing_tensor"
    existing = table.record(commit.slot_id)
    if existing is not None:
        if int(existing.write_generation) == commit.new_generation:
            return True, "already_present"
        return False, "slot_id_collision"
    max_slots = int(outer.max_slots or 0)
    if max_slots > 0 and len(table) >= max_slots:
        return False, "replica_capacity_full"
    table.append_with_id(
        commit.slot_id,
        commit.tensor,
        bucket_id=commit.bucket_id,
        event_id=commit.event_id,
        step=commit.step,
        survival=1.0,
        generation=commit.new_generation,
    )
    return True, "appended"


def slot_commit_dtype_code(dtype: torch.dtype) -> int:
    if dtype not in SLOT_COMMIT_DTYPE_TO_CODE:
        raise ValueError(f"unsupported SlotCommit tensor dtype: {dtype}")
    return int(SLOT_COMMIT_DTYPE_TO_CODE[dtype])


def slot_commit_dtype_from_code(code: int) -> torch.dtype:
    try:
        dtype_code = SlotCommitDType(int(code))
    except ValueError as exc:
        raise ValueError(f"unsupported SlotCommit tensor dtype code: {code}") from exc
    if dtype_code not in SLOT_COMMIT_CODE_TO_DTYPE:
        raise ValueError(f"unsupported SlotCommit tensor dtype code: {code}")
    return SLOT_COMMIT_CODE_TO_DTYPE[dtype_code]
