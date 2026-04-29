"""SlotTable: persistent identity and lifecycle management for episodic memory slots.

Physical indices shift when slots are retired. SlotIds never do.
The SlotTable owns the tensor storage and metadata, eliminating
reindex bugs by centralizing all slot mutations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

__all__ = ["SlotTable", "SlotRecord", "SlotId"]

SlotId = int

SLOT_WARMING = "WARMING"
SLOT_ACTIVE = "ACTIVE"
SLOT_SHARP = "SHARP"
SLOT_DECAYING = "DECAYING"
SLOT_QUARANTINED = "QUARANTINED"
SLOT_DISTILLING = "DISTILLING"
SLOT_RETIRED = "RETIRED"


@dataclass
class SlotRecord:
    slot_id: SlotId
    state: str = SLOT_WARMING
    created_step: int = 0
    last_scored_step: int = 0
    last_used_step: int = 0
    write_generation: int = 0
    bucket_id: int = -1
    event_id: int = 0

    utility_ema: float = 0.0
    marginal_gain_ema: float = 0.0
    sharpness_ema: float = 0.0
    activation_drift_ema: float = 0.0
    representation_drift_ema: float = 0.0
    semantic_drift_ema: float = 0.0
    contradiction_ema: float = 0.0
    retrieval_mass_ema: float = 0.0
    peak_utility: float = 0.0
    peak_sharpness: float = 0.0
    score_count: int = 0
    negative_streak: int = 0
    positive_streak: int = 0
    refresh_count: int = 0
    quarantine_count: int = 0
    last_action: str = "PRESERVE"
    retire_reason: str = ""


class SlotTable:
    """Centralized slot storage with persistent identity and lifecycle tracking."""

    def __init__(self) -> None:
        self._next_id: int = 0
        self._records: dict[SlotId, SlotRecord] = {}
        self._id_to_physical: dict[SlotId, int] = {}
        self._physical_to_id: dict[int, SlotId] = {}

        self._slots: list[torch.Tensor] = []
        self._survival: list[float] = []
        self._slot_buckets: list[int] = []
        self._slot_event_ids: list[int] = []
        self._priority: list[float] = []

    def __len__(self) -> int:
        return len(self._slots)

    @property
    def n_active(self) -> int:
        return len(self._slots)

    def _allocate_id(self) -> SlotId:
        sid = self._next_id
        self._next_id += 1
        return sid

    def append(
        self,
        tensor: torch.Tensor,
        *,
        bucket_id: int = -1,
        event_id: int = 0,
        step: int = 0,
        survival: float = 0.0,
    ) -> SlotId:
        sid = self._allocate_id()
        phys = len(self._slots)

        self._slots.append(tensor.detach())
        self._survival.append(survival)
        self._slot_buckets.append(bucket_id)
        self._slot_event_ids.append(event_id)
        self._priority.append(1.0)

        self._id_to_physical[sid] = phys
        self._physical_to_id[phys] = sid

        rec = SlotRecord(
            slot_id=sid,
            state=SLOT_WARMING,
            created_step=step,
            bucket_id=bucket_id,
            event_id=event_id,
        )
        self._records[sid] = rec
        return sid

    def retire(self, slot_id: SlotId, *, reason: str = "evicted") -> bool:
        if slot_id not in self._id_to_physical:
            return False
        phys = self._id_to_physical[slot_id]
        if phys < 0 or phys >= len(self._slots):
            return False

        del self._slots[phys]
        del self._survival[phys]
        del self._slot_buckets[phys]
        del self._slot_event_ids[phys]
        del self._priority[phys]

        del self._id_to_physical[slot_id]
        self._rebuild_physical_maps()

        if slot_id in self._records:
            self._records[slot_id].state = SLOT_RETIRED
            self._records[slot_id].retire_reason = reason

        return True

    def retire_many(self, slot_ids: list[SlotId], *, reason: str = "evicted") -> list[SlotId]:
        phys_indices = []
        valid_ids = []
        for sid in slot_ids:
            if sid in self._id_to_physical:
                phys_indices.append((self._id_to_physical[sid], sid))
                valid_ids.append(sid)

        for phys, sid in sorted(phys_indices, key=lambda x: x[0], reverse=True):
            if phys < len(self._slots):
                del self._slots[phys]
                del self._survival[phys]
                del self._slot_buckets[phys]
                del self._slot_event_ids[phys]
                del self._priority[phys]
            del self._id_to_physical[sid]
            if sid in self._records:
                self._records[sid].state = SLOT_RETIRED
                self._records[sid].retire_reason = reason

        self._rebuild_physical_maps()
        return valid_ids

    def _rebuild_physical_maps(self) -> None:
        sorted_sids = sorted(self._id_to_physical.keys())
        self._id_to_physical.clear()
        self._physical_to_id.clear()
        for phys, sid in enumerate(sorted_sids):
            self._id_to_physical[sid] = phys
            self._physical_to_id[phys] = sid

    def quarantine(self, slot_id: SlotId) -> bool:
        if slot_id not in self._id_to_physical:
            return False
        phys = self._id_to_physical[slot_id]
        self._priority[phys] = 0.0
        if slot_id in self._records:
            self._records[slot_id].state = SLOT_QUARANTINED
            self._records[slot_id].quarantine_count += 1
        return True

    def release(self, slot_id: SlotId) -> bool:
        if slot_id not in self._id_to_physical:
            return False
        phys = self._id_to_physical[slot_id]
        self._priority[phys] = 1.0
        if slot_id in self._records:
            self._records[slot_id].state = SLOT_ACTIVE
        return True

    def replace_tensor(self, slot_id: SlotId, tensor: torch.Tensor) -> bool:
        if slot_id not in self._id_to_physical:
            return False
        phys = self._id_to_physical[slot_id]
        self._slots[phys] = tensor.detach()
        return True

    def scale_survival(self, slot_id: SlotId, factor: float) -> bool:
        """Scale a slot's survival score without exposing table internals."""
        if slot_id not in self._id_to_physical:
            return False
        phys = self._id_to_physical[slot_id]
        self._survival[phys] *= float(factor)
        return True

    def get_tensor(self, slot_id: SlotId) -> torch.Tensor | None:
        if slot_id not in self._id_to_physical:
            return None
        return self._slots[self._id_to_physical[slot_id]]

    def record(self, slot_id: SlotId) -> SlotRecord | None:
        return self._records.get(slot_id)

    def active_records(self) -> list[SlotRecord]:
        return [
            self._records[sid]
            for sid in self._id_to_physical
            if sid in self._records
        ]

    def active_slot_ids(self) -> list[SlotId]:
        return list(self._id_to_physical.keys())

    def physical_to_slot_id(self, phys: int) -> SlotId | None:
        return self._physical_to_id.get(phys)

    def slot_id_to_physical(self, slot_id: SlotId) -> int | None:
        return self._id_to_physical.get(slot_id)

    def visible_indices(
        self,
        *,
        read_cutoff: int | None = None,
        bucket_id: int | None = None,
    ) -> list[int]:
        cutoff = None if read_cutoff is None else int(read_cutoff)
        indices: list[int] = []
        for i in range(len(self._slots)):
            if bucket_id is not None and self._slot_buckets[i] != bucket_id:
                continue
            if cutoff is not None and self._slot_event_ids[i] > cutoff:
                continue
            indices.append(i)
        return indices

    def slot_matrix(self, indices: list[int] | None = None) -> torch.Tensor:
        if indices is None:
            indices = list(range(len(self._slots)))
        if not indices:
            return torch.zeros(0, 0)
        return torch.cat([self._slots[i] for i in indices], dim=0)

    def priority_vector(self, indices: list[int] | None = None) -> torch.Tensor:
        if indices is None:
            indices = list(range(len(self._slots)))
        if not indices:
            return torch.zeros(0)
        return torch.tensor(
            [self._priority[i] for i in indices],
            dtype=torch.float32,
        )

    def state_dict(self) -> dict:
        records_ser = {}
        for sid, rec in self._records.items():
            if rec.state == SLOT_RETIRED:
                continue
            records_ser[sid] = {
                f.name: getattr(rec, f.name)
                for f in rec.__dataclass_fields__.values()
            }
        return {
            "next_id": self._next_id,
            "slots": [s.cpu() for s in self._slots],
            "survival": list(self._survival),
            "slot_buckets": list(self._slot_buckets),
            "slot_event_ids": list(self._slot_event_ids),
            "priority": list(self._priority),
            "id_to_physical": dict(self._id_to_physical),
            "records": records_ser,
        }

    def load_state_dict(self, d: dict, *, device: torch.device | None = None) -> None:
        self._next_id = d["next_id"]
        self._slots = [s.to(device) if device else s for s in d["slots"]]
        self._survival = list(d["survival"])
        self._slot_buckets = list(d["slot_buckets"])
        self._slot_event_ids = list(d["slot_event_ids"])
        self._priority = list(d.get("priority", [1.0] * len(self._slots)))

        self._id_to_physical = {int(k): v for k, v in d["id_to_physical"].items()}
        self._rebuild_physical_maps()

        self._records = {}
        for sid_str, rec_dict in d.get("records", {}).items():
            sid = int(sid_str)
            self._records[sid] = SlotRecord(**rec_dict)

    def purge_retired(self) -> int:
        retired = [sid for sid, rec in self._records.items() if rec.state == SLOT_RETIRED]
        for sid in retired:
            del self._records[sid]
        return len(retired)
