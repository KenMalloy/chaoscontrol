"""Episodic cache for the SSM memory subsystem (design: docs/plans/2026-04-25-memory-aware-optimizer-design.md).

Component 1 of the build order: the substrate that the optimizer-side append
hook writes to and the GPU-side snapshot copies from. A fixed-capacity store
of (key, value, utility) entries with content-addressable lookup by uint64
fingerprint and eviction by lowest retrieval-utility past a grace period.

Schema per entry (Phase 1, correctness-over-bytes; Phase 2 will tighten the
layout to 64-byte alignment for GPU snapshot efficiency):

    key_fp                  int64       rolling-hash fingerprint of preceding tokens
    key_rep                 float32[D]  projection of hidden state at write
                                        (CPU-rewritten on re-projection sweeps;
                                        absent in this phase, present as a
                                        placeholder field)
    value_tok_ids           int64[S]    the next S token IDs to inject on retrieval
    value_anchor_id         int64       focal token ID at write event
                                        (preserved across re-projections)
    utility_u               float32     EMA of retrieval-time CE delta
    last_fired_step         int64       most recent retrieval step
    write_step              int64       step at which this entry was written
    birth_embedding_version int64       embedding version at write time
                                        (drift detection)
    occupied                bool        slot in use

Eviction policy (Q1 + Q8 of design proposal): the loss function and the
eviction key are the same quantity — bottom-utility entries past their grace
period are evicted. Coupling loss to eviction prevents the policy/objective
drift that Titans Revisited flagged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class CacheEntry:
    """A single entry's contents, returned by query()."""

    slot: int
    key_fp: int
    key_rep: torch.Tensor  # [key_rep_dim] float32
    value_tok_ids: torch.Tensor  # [span_length] int64
    value_anchor_id: int
    utility_u: float
    last_fired_step: int
    write_step: int
    birth_embedding_version: int


class EpisodicCache:
    """Fixed-capacity content-addressable cache with utility-EMA eviction.

    All tensors live on CPU by default; ``snapshot_to(device)`` copies the
    field tensors to the requested device for GPU-side retrieval. The cache
    itself is the source of truth; the snapshot is read-only on the GPU.

    Thread-safety: this class is not internally synchronized. The intended
    threading model has one writer (the CPU controller draining the
    optimizer's append queue) and one reader (the snapshot publisher).
    Callers coordinate via the queue+snapshot protocol; the cache itself
    assumes single-writer access.
    """

    def __init__(
        self,
        *,
        capacity: int,
        span_length: int = 8,
        key_rep_dim: int = 16,
        grace_steps: int = 200,
        utility_ema_decay: float = 0.99,
        fingerprint_window: int = 8,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive; got {capacity}")
        if span_length <= 0:
            raise ValueError(f"span_length must be positive; got {span_length}")
        if key_rep_dim <= 0:
            raise ValueError(f"key_rep_dim must be positive; got {key_rep_dim}")
        if not 0.0 < utility_ema_decay < 1.0:
            raise ValueError(
                f"utility_ema_decay must be in (0, 1); got {utility_ema_decay}"
            )
        if fingerprint_window <= 0:
            raise ValueError(
                f"fingerprint_window must be positive; got {fingerprint_window}"
            )

        self.capacity = int(capacity)
        self.span_length = int(span_length)
        self.key_rep_dim = int(key_rep_dim)
        self.grace_steps = int(grace_steps)
        self.utility_ema_decay = float(utility_ema_decay)
        # Carried on the cache so the trainer's W choice rides into the
        # eval-time controller via the saved payload — preventing the
        # silent W-mismatch failure mode where the cache holds W=8
        # fingerprints but the controller queries with W=4 and scores
        # zero hits.
        self.fingerprint_window = int(fingerprint_window)

        # Storage. Parallel tensors keyed by slot index.
        self.key_fp = torch.zeros(self.capacity, dtype=torch.int64)
        self.key_rep = torch.zeros(
            self.capacity, self.key_rep_dim, dtype=torch.float32
        )
        self.value_tok_ids = torch.zeros(
            self.capacity, self.span_length, dtype=torch.int64
        )
        self.value_anchor_id = torch.zeros(self.capacity, dtype=torch.int64)
        self.utility_u = torch.zeros(self.capacity, dtype=torch.float32)
        self.last_fired_step = torch.full(
            (self.capacity,), -1, dtype=torch.int64,
        )
        self.write_step = torch.full((self.capacity,), -1, dtype=torch.int64)
        self.birth_embedding_version = torch.zeros(
            self.capacity, dtype=torch.int64,
        )
        self.occupied = torch.zeros(self.capacity, dtype=torch.bool)

        # Hash index for fast key_fp -> slot lookup. Rebuilt on every
        # mutation to keep this simple; if the rebuild cost shows up in
        # profiling, swap to incremental maintenance later.
        self._fp_index: dict[int, int] = {}

    # ---- size / introspection -------------------------------------------------

    def __len__(self) -> int:
        return int(self.occupied.sum().item())

    @property
    def is_full(self) -> bool:
        return len(self) >= self.capacity

    # ---- core operations ------------------------------------------------------

    def append(
        self,
        *,
        key_fp: int,
        key_rep: torch.Tensor,
        value_tok_ids: torch.Tensor,
        value_anchor_id: int,
        current_step: int,
        embedding_version: int,
    ) -> int:
        """Insert one entry. If the cache is full, evict the lowest-utility
        entry past its grace period; if no entry is past grace, evict by
        oldest write_step (so a saturated cache during the warm-up window
        rotates by FIFO rather than refusing all writes).

        Returns the slot index used.
        """
        if key_rep.shape != (self.key_rep_dim,):
            raise ValueError(
                f"key_rep must have shape ({self.key_rep_dim},); "
                f"got {tuple(key_rep.shape)}"
            )
        if value_tok_ids.shape != (self.span_length,):
            raise ValueError(
                f"value_tok_ids must have shape ({self.span_length},); "
                f"got {tuple(value_tok_ids.shape)}"
            )

        slot = self._allocate_slot(current_step=current_step)

        # If the slot was occupied, drop its old fingerprint from the index.
        if self.occupied[slot].item():
            old_fp = int(self.key_fp[slot].item())
            if self._fp_index.get(old_fp) == slot:
                self._fp_index.pop(old_fp, None)

        self.key_fp[slot] = int(key_fp)
        self.key_rep[slot] = key_rep.to(dtype=torch.float32)
        self.value_tok_ids[slot] = value_tok_ids.to(dtype=torch.int64)
        self.value_anchor_id[slot] = int(value_anchor_id)
        # utility_u initialized to 1.0 (not 0.0) so that retrieval-time
        # scoring `score = cosine_sim × utility_u` doesn't silently degenerate
        # to zero for fresh entries before any replay has updated utility.
        # New entries enter at full retrieval weight; subsequent replay-driven
        # update_utility calls drive utility down for entries that don't help.
        self.utility_u[slot] = 1.0
        self.last_fired_step[slot] = -1
        self.write_step[slot] = int(current_step)
        self.birth_embedding_version[slot] = int(embedding_version)
        self.occupied[slot] = True

        # Insert into hash index. If a duplicate fingerprint already exists
        # at a different slot, the new write wins (the older slot remains
        # occupied but is no longer reachable by hash; it will be evicted
        # naturally by utility/age).
        self._fp_index[int(key_fp)] = slot
        return slot

    def query(self, key_fp: int) -> CacheEntry | None:
        """Top-1 lookup by exact fingerprint match. Returns None on miss.

        Note: caller is responsible for invoking ``mark_fired`` if it
        retrieves and uses the entry — query is read-only and does not
        update last_fired_step or any utility state.
        """
        slot = self._fp_index.get(int(key_fp))
        if slot is None:
            return None
        if not self.occupied[slot].item():
            # Stale index entry; eject and miss.
            self._fp_index.pop(int(key_fp), None)
            return None
        return CacheEntry(
            slot=int(slot),
            key_fp=int(self.key_fp[slot].item()),
            key_rep=self.key_rep[slot].clone(),
            value_tok_ids=self.value_tok_ids[slot].clone(),
            value_anchor_id=int(self.value_anchor_id[slot].item()),
            utility_u=float(self.utility_u[slot].item()),
            last_fired_step=int(self.last_fired_step[slot].item()),
            write_step=int(self.write_step[slot].item()),
            birth_embedding_version=int(
                self.birth_embedding_version[slot].item()
            ),
        )

    def mark_fired(self, slot: int, current_step: int) -> None:
        """Record that this slot was retrieved at the given step. Does not
        change utility_u — utility update happens via update_utility once
        the downstream CE feedback is available."""
        self._check_slot(slot)
        if not self.occupied[slot].item():
            raise ValueError(f"slot {slot} is not occupied")
        self.last_fired_step[slot] = int(current_step)

    def update_utility(self, slot: int, ce_delta: float) -> None:
        """EMA update for the slot's utility. Higher ce_delta = more useful
        retrieval. Skip silently for unoccupied slots so feedback packets
        racing against eviction don't crash the controller."""
        self._check_slot(slot)
        if not self.occupied[slot].item():
            return
        decay = self.utility_ema_decay
        cur = float(self.utility_u[slot].item())
        self.utility_u[slot] = decay * cur + (1.0 - decay) * float(ce_delta)

    def evict(self, slot: int) -> None:
        """Mark a slot as unoccupied and remove it from the hash index."""
        self._check_slot(slot)
        if not self.occupied[slot].item():
            return
        fp = int(self.key_fp[slot].item())
        if self._fp_index.get(fp) == slot:
            self._fp_index.pop(fp, None)
        self.occupied[slot] = False
        self.last_fired_step[slot] = -1
        self.write_step[slot] = -1
        self.utility_u[slot] = 0.0

    def reset(self) -> None:
        """Return the cache to its post-construction state.

        Zeros every field tensor, clears the hash index, and re-applies the
        ``last_fired_step``/``write_step`` sentinel of -1. Capacity and the
        construction-time shape parameters (``span_length``, ``key_rep_dim``,
        ``grace_steps``, ``utility_ema_decay``, ``fingerprint_window``) are
        preserved.

        Used by the eval-time runner for per-doc reset semantics — each doc
        starts with a fresh cache so cross-document leakage in the
        retrieval index is structurally impossible. (Per-doc reset is opt-in
        via ``RunConfig.episodic_cache_reset_per_doc``; the default loaded-
        from-checkpoint path keeps the cache live across docs.)
        """
        self.key_fp.zero_()
        self.key_rep.zero_()
        self.value_tok_ids.zero_()
        self.value_anchor_id.zero_()
        self.utility_u.zero_()
        self.last_fired_step.fill_(-1)
        self.write_step.fill_(-1)
        self.birth_embedding_version.zero_()
        self.occupied.zero_()
        self._fp_index.clear()

    # ---- snapshot -------------------------------------------------------------

    def snapshot_to(self, device: torch.device) -> dict[str, torch.Tensor]:
        """Copy the field tensors to the target device. Returned dict is a
        consistent snapshot — caller treats it as read-only.

        Phase 1: full-tensor copy each call. Phase 2 will swap this for a
        double-buffered pinned-memory layout per IPC design (Q6).
        """
        return {
            "key_fp": self.key_fp.to(device, non_blocking=True),
            "key_rep": self.key_rep.to(device, non_blocking=True),
            "value_tok_ids": self.value_tok_ids.to(device, non_blocking=True),
            "value_anchor_id": self.value_anchor_id.to(
                device, non_blocking=True,
            ),
            "utility_u": self.utility_u.to(device, non_blocking=True),
            "last_fired_step": self.last_fired_step.to(
                device, non_blocking=True,
            ),
            "write_step": self.write_step.to(device, non_blocking=True),
            "birth_embedding_version": self.birth_embedding_version.to(
                device, non_blocking=True,
            ),
            "occupied": self.occupied.to(device, non_blocking=True),
        }

    # ---- save / load ----------------------------------------------------------

    # Construction parameters that round-trip through to_dict/from_dict.
    # These match the keyword arguments accepted by ``__init__``.
    _CONFIG_FIELDS: tuple[str, ...] = (
        "capacity",
        "span_length",
        "key_rep_dim",
        "grace_steps",
        "utility_ema_decay",
        "fingerprint_window",
    )
    # Per-slot tensor fields that round-trip as torch.Tensor (kept native
    # so the trainer's saved payload can copy_() into the new cache without
    # an extra dtype/shape conversion). The ``_fp_index`` Python dict is
    # carried separately under the ``fp_index`` key.
    _TENSOR_FIELDS: tuple[str, ...] = (
        "key_fp",
        "key_rep",
        "value_tok_ids",
        "value_anchor_id",
        "utility_u",
        "last_fired_step",
        "write_step",
        "birth_embedding_version",
        "occupied",
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the cache to a plain dict — symmetric with ``from_dict``.

        Construction params live as Python scalars; per-slot fields stay as
        torch tensors (clone() so callers can't mutate live cache state via
        the returned blob); the hash index ``fp_index`` rides along as a
        plain ``dict[int, int]`` so ``from_dict`` can rebuild content-
        addressable lookup without a re-scan.

        Used by the trainer to pack ``ckpt['episodic_cache']`` for save and
        by the eval-time loader to reconstruct on load. Both ends must speak
        the same schema; ``_CONFIG_FIELDS`` and ``_TENSOR_FIELDS`` are the
        canonical key list.
        """
        blob: dict[str, Any] = {name: getattr(self, name) for name in self._CONFIG_FIELDS}
        for name in self._TENSOR_FIELDS:
            blob[name] = getattr(self, name).clone()
        # dict() copy so consumers can't mutate the cache's live index via
        # the returned payload.
        blob["fp_index"] = dict(self._fp_index)
        return blob

    @classmethod
    def from_dict(cls, blob: dict[str, Any]) -> "EpisodicCache":
        """Reconstruct a cache from a ``to_dict`` payload.

        STRICT: every key in ``_CONFIG_FIELDS`` and ``_TENSOR_FIELDS`` plus
        ``fp_index`` MUST be present. A missing key raises KeyError with a
        message naming the field — silent defaults here are the failure
        mode where Arm B's cache shape silently diverges from the trainer's
        and the falsifier matrix's contrast collapses to noise.
        """
        missing = [
            name for name in (*cls._CONFIG_FIELDS, *cls._TENSOR_FIELDS, "fp_index")
            if name not in blob
        ]
        if missing:
            raise KeyError(
                f"EpisodicCache.from_dict missing required key(s): "
                f"{sorted(missing)}. The payload must carry every field "
                f"emitted by EpisodicCache.to_dict — silent defaults here "
                f"would let saved-vs-loaded cache shape diverge."
            )
        cache = cls(**{name: blob[name] for name in cls._CONFIG_FIELDS})
        for name in cls._TENSOR_FIELDS:
            getattr(cache, name).copy_(blob[name])
        # Cast keys/values to plain ``int`` so the loaded index behaves
        # like one constructed by ``append()`` — tensor element types
        # (e.g. numpy.int64 from a ckpt round-trip) compare unequal to
        # plain ints in dict lookups.
        cache._fp_index = {int(k): int(v) for k, v in blob["fp_index"].items()}
        return cache

    # ---- internals ------------------------------------------------------------

    def _check_slot(self, slot: int) -> None:
        if not 0 <= slot < self.capacity:
            raise IndexError(
                f"slot {slot} out of range for capacity {self.capacity}"
            )

    def _allocate_slot(self, *, current_step: int) -> int:
        """Pick a slot for a new write. Empty slot if any; otherwise the
        lowest-utility entry past its grace period; if no entry is past
        grace, the oldest by write_step (warm-up FIFO)."""
        empty = (~self.occupied).nonzero(as_tuple=True)[0]
        if empty.numel() > 0:
            return int(empty[0].item())

        age = current_step - self.write_step
        past_grace = age >= self.grace_steps
        if past_grace.any():
            # Among entries past grace, pick the lowest utility.
            utilities = torch.where(
                past_grace,
                self.utility_u,
                torch.full_like(self.utility_u, float("inf")),
            )
            return int(utilities.argmin().item())

        # All entries are within their grace period. Evict the oldest.
        return int(self.write_step.argmin().item())
