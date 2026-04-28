"""VRAM long-term memory modules: OuterModel and MultiSlotOuterModel.

OuterModel  -- single-slot lossy encode/decode with surprise-driven consolidation.
MultiSlotOuterModel -- multi-slot variant with cue-dependent retrieval, survival
                       scoring, typed compression, and checkpoint persistence.
"""
from __future__ import annotations

import random as _random
from collections import defaultdict
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .slot_table import SlotTable


class OuterModel(nn.Module):
    """VRAM long-term memory with lossy encode/decode and surprise-driven consolidation.

    The outer state lives in a different representational space (outer_dim)
    from the recurrence (model_dim). Encode != decode^-1 guarantees lossiness.
    Persists across sequences (state is a buffer, not reset).
    """

    def __init__(
        self,
        model_dim: int,
        outer_dim: int = 64,
        consolidation_mode: str = "symmetric",
        ema_decay: float = 0.99,
        trigger: str = "immediate",
        trigger_window: int = 8,
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.outer_dim = outer_dim
        self.consolidation_mode = consolidation_mode
        self.ema_decay = ema_decay
        self.trigger = trigger
        self.trigger_window = trigger_window

        # Trigger state for resolution/windowed modes
        self._spike_seen = False
        self._steps_since_spike = 0
        self._pre_spike_loss: float = 0.0

        # Encoder is a structural transform, not learnable via task loss.
        # The outer model's learning rule is self-supervised (surprise-driven).
        self.encoder = nn.Linear(model_dim, outer_dim, bias=False)
        self.encoder.weight.requires_grad_(False)

        # Decoder IS on the forward path and receives task-loss gradients.
        # This is the interface between outer model and the task — it should
        # learn to produce useful reconstructions.
        self.decoder = nn.Linear(outer_dim, model_dim, bias=False)

        # Inertia is structural (controls accumulation speed), not task-learnable.
        self.log_inertia = nn.Parameter(torch.tensor(2.0, dtype=torch.float32), requires_grad=False)

        if consolidation_mode == "learned":
            # Gradient-free: online adjustment based on whether pain or reward
            # consolidations tend to precede loss improvements.
            self.register_buffer("consolidation_w", torch.tensor(0.0))
            self.register_buffer("_last_signal_was_pain", torch.tensor(False))
            self.register_buffer("_last_loss", torch.tensor(0.0))
            self.register_buffer("_last_wrote", torch.tensor(False))

        # Persistent buffers (not reset between sequences)
        self.register_buffer("state", torch.zeros(1, outer_dim))
        self.register_buffer("loss_ema", torch.tensor([2.0]))

    def read(self, batch_size: int) -> torch.Tensor:
        """Lossy decode. Expand state to batch and project back to model_dim."""
        return self.decoder(self.state.expand(batch_size, -1).to(dtype=self.decoder.weight.dtype))

    def write(self, h: torch.Tensor, *, per_sample_weights: torch.Tensor | None = None) -> None:
        """Lossy encode. Weighted average across batch, accumulate with inertia.

        If per_sample_weights is provided (batch,), samples with higher weight
        contribute more to the stored memory — e.g. weight by per-sample surprise.
        """
        h_enc = h.detach().to(dtype=self.encoder.weight.dtype)
        encoded = torch.tanh(self.encoder(h_enc))
        if per_sample_weights is not None:
            w = per_sample_weights.detach()
            w = w / w.sum().clamp_min(1e-8)  # normalize to sum to 1
            encoded_agg = (w.unsqueeze(-1) * encoded).sum(dim=0, keepdim=True)
        else:
            encoded_agg = encoded.mean(dim=0, keepdim=True)
        inertia = torch.sigmoid(self.log_inertia)
        self.state = (inertia * self.state.detach() + (1 - inertia) * encoded_agg).detach()

    def compute_consolidation_signal(self, current_loss: float, running_avg: float) -> float:
        """Compute surprise magnitude.

        symmetric:    |loss - avg|
        pain_biased:  pain + 0.5 * reward
        learned:      sigmoid(w) * pain + (1 - sigmoid(w)) * reward
        """
        pain = max(current_loss - running_avg, 0.0)
        reward = max(running_avg - current_loss, 0.0)

        if self.consolidation_mode == "symmetric":
            return abs(current_loss - running_avg)
        elif self.consolidation_mode == "pain_biased":
            return pain + 0.5 * reward
        elif self.consolidation_mode == "learned":
            w = torch.sigmoid(self.consolidation_w).item()
            return w * pain + (1.0 - w) * reward
        else:
            raise ValueError(f"unsupported consolidation_mode: {self.consolidation_mode}")

    def consolidation_step(
        self,
        h: torch.Tensor,
        current_loss: float,
        per_sample_weights: torch.Tensor | None = None,
        bucket_id: int | None = None,
    ) -> float:
        """Full step: compute signal, gate write by surprise, update EMA.

        Only writes when signal/running_avg > 0.01 (skip boring moments).
        For "learned" mode: gradient-free update — if the last consolidation
        was pain-triggered and loss improved, nudge w toward pain; if reward-
        triggered and loss improved, nudge w toward reward.
        Returns the surprise signal value.
        """
        running_avg = self.loss_ema.item()
        signal = self.compute_consolidation_signal(current_loss, running_avg)
        is_pain = current_loss > running_avg

        # Gradient-free update for learned consolidation weight.
        # Only update w when the PREVIOUS step actually wrote memory,
        # so credit assignment is grounded in real consolidation events.
        if self.consolidation_mode == "learned" and self._last_wrote.item():
            loss_improved = current_loss < self._last_loss.item()
            if loss_improved:
                # Last consolidation type worked — nudge toward it
                if self._last_signal_was_pain.item():
                    self.consolidation_w = self.consolidation_w + 0.01
                else:
                    self.consolidation_w = self.consolidation_w - 0.01

        # Determine whether to write based on trigger mode
        surprise_threshold = running_avg > 0 and signal / running_avg > 0.01
        wrote = False

        if self.trigger == "immediate":
            # Write on surprise directly
            if surprise_threshold:
                self.write(h, per_sample_weights=per_sample_weights)
                wrote = True

        elif self.trigger == "resolution":
            # Spike = flag that accumulation phase should flush on settlement.
            # Always accumulating (overlapping). Spike marks "something happened."
            # Settlement (loss returns below running_avg after spike) = flush.
            if surprise_threshold and not self._spike_seen:
                self._spike_seen = True
                self._pre_spike_loss = running_avg
            if self._spike_seen:
                # Check for resolution: loss settled back below pre-spike level
                settled = current_loss <= self._pre_spike_loss
                if settled:
                    self.write(h, per_sample_weights=per_sample_weights)
                    wrote = True
                    self._spike_seen = False

        elif self.trigger == "windowed":
            # Spike opens a window. Flush after N steps.
            if surprise_threshold and not self._spike_seen:
                self._spike_seen = True
                self._steps_since_spike = 0
            if self._spike_seen:
                self._steps_since_spike += 1
                if self._steps_since_spike >= self.trigger_window:
                    self.write(h, per_sample_weights=per_sample_weights)
                    wrote = True
                    self._spike_seen = False

        # Track for learned mode's gradient-free update
        if self.consolidation_mode == "learned":
            self._last_signal_was_pain = torch.tensor(is_pain)
            self._last_loss = torch.tensor(current_loss)
            self._last_wrote = torch.tensor(wrote)

        # Update EMA
        self.loss_ema = self.ema_decay * self.loss_ema + (1 - self.ema_decay) * current_loss

        return signal


class MultiSlotOuterModel(nn.Module):
    """Multi-slot VRAM long-term memory with cue-dependent retrieval and compression.

    Each consolidation event writes a new slot. When max_slots is reached,
    the oldest slots are merged (lossy compression = forgetting). Retrieval
    uses dot-product similarity between the current hidden state and each slot.
    Per-slot survival scores track impact: how much does this slot help when cued?
    """

    def __init__(
        self,
        model_dim: int,
        outer_dim: int = 64,
        consolidation_mode: str = "symmetric",
        ema_decay: float = 0.99,
        trigger: str = "immediate",
        trigger_window: int = 8,
        max_slots: int = 64,
        compress_ratio: int = 2,
        compression_selection: str = "survival",
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.outer_dim = outer_dim
        self.consolidation_mode = consolidation_mode
        self.ema_decay = ema_decay
        self.trigger = trigger
        self.trigger_window = trigger_window
        self.max_slots = max_slots
        self.compress_ratio = max(2, compress_ratio)
        self.compression_selection = compression_selection
        self._compress_rng = _random.Random(42)  # seeded for reproducibility

        # Trigger state
        self._spike_seen = False
        self._steps_since_spike = 0
        self._pre_spike_loss: float = 0.0

        # Encoder: frozen structural transform
        self.encoder = nn.Linear(model_dim, outer_dim, bias=False)
        self.encoder.weight.requires_grad_(False)

        # Decoder: on forward path, receives task gradients
        self.decoder = nn.Linear(outer_dim, model_dim, bias=False)

        # Cue projection: project hidden state to outer_dim for similarity
        self.cue_proj = nn.Linear(model_dim, outer_dim, bias=False)

        if consolidation_mode == "learned":
            self.register_buffer("consolidation_w", torch.tensor(0.0))
            self.register_buffer("_last_signal_was_pain", torch.tensor(False))
            self.register_buffer("_last_loss", torch.tensor(0.0))
            self.register_buffer("_last_wrote", torch.tensor(False))

        self.register_buffer("loss_ema", torch.tensor([2.0]))

        # SlotTable: centralized slot storage with persistent identity
        self.table = SlotTable()
        self._retrieval_weights: torch.Tensor | None = None
        self._retrieval_indices: list[int] | None = None
        self._compression_consequences: list[tuple[int, float]] = []
        self._latent_traces: list[dict] = []

        self._bucket_affinity: torch.Tensor | None = None

    # Backward-compat properties — delegate to SlotTable internals
    @property
    def _slots(self) -> list[torch.Tensor]:
        return self.table._slots

    @_slots.setter
    def _slots(self, value: list[torch.Tensor]) -> None:
        self.table._slots = list(value)
        self._rebuild_table_identity()

    @property
    def _survival(self) -> list[float]:
        return self.table._survival

    @_survival.setter
    def _survival(self, value: list[float]) -> None:
        self.table._survival = list(value)

    @property
    def _slot_buckets(self) -> list[int]:
        return self.table._slot_buckets

    @_slot_buckets.setter
    def _slot_buckets(self, value: list[int]) -> None:
        self.table._slot_buckets = list(value)

    @property
    def _slot_event_ids(self) -> list[int]:
        return self.table._slot_event_ids

    @_slot_event_ids.setter
    def _slot_event_ids(self, value: list[int]) -> None:
        self.table._slot_event_ids = list(value)

    def _rebuild_table_identity(self) -> None:
        """Reset table identity mappings after bulk list replacement."""
        t = self.table
        n = len(t._slots)
        t._id_to_physical.clear()
        t._physical_to_id.clear()
        t._records.clear()
        t._next_id = n
        t._priority = [1.0] * n
        for i in range(n):
            t._id_to_physical[i] = i
            t._physical_to_id[i] = i

    def _ensure_slot_event_ids(self) -> None:
        n_slots = len(self.table._slots)
        n_eids = len(self.table._slot_event_ids)
        if n_eids < n_slots:
            self.table._slot_event_ids.extend([0] * (n_slots - n_eids))
        elif n_eids > n_slots:
            self.table._slot_event_ids[:] = self.table._slot_event_ids[:n_slots]

    def _visible_slot_indices(
        self,
        *,
        read_cutoff: int | None = None,
        bucket_id: int | None = None,
    ) -> list[int]:
        return self.table.visible_indices(read_cutoff=read_cutoff, bucket_id=bucket_id)

    def append_kv_batch(self, encoded_batch: torch.Tensor, bucket_ids: torch.Tensor) -> None:
        """Append multiple KV pairs at once as a batched operation.

        Avoids per-token Python loops and GPU syncs. One .tolist() call
        for bucket_ids, one torch.split for slots.

        Args:
            encoded_batch: (N, outer_dim) pre-encoded KV pairs.
            bucket_ids: (N,) integer bucket assignments for each pair.
        """
        self._append_kv_batch_committed(encoded_batch, bucket_ids, event_ids=None)

    def _append_kv_batch_committed(
        self,
        encoded_batch: torch.Tensor,
        bucket_ids: torch.Tensor,
        event_ids: torch.Tensor | None = None,
    ) -> None:
        """Append pre-encoded KV pairs with explicit MVCC event ids.

        Event id ``0`` is the legacy/default value: visible to every
        transaction.  CRCT's rank-3 oracle passes strictly newer ids so
        same-batch writes remain invisible to both oracle encode passes.
        """
        n = int(encoded_batch.shape[0])
        if n == 0:
            return
        # Single GPU→CPU transfer for all bucket ids
        bid_list = [int(b) for b in bucket_ids.detach().reshape(-1).tolist()]
        if len(bid_list) != n:
            raise ValueError(
                f"bucket_ids length {len(bid_list)} does not match encoded batch {n}"
            )
        if event_ids is None:
            event_list = [0] * n
        else:
            event_list = [int(e) for e in event_ids.detach().reshape(-1).tolist()]
            if len(event_list) != n:
                raise ValueError(
                    f"event_ids length {len(event_list)} does not match encoded batch {n}"
                )
        # Append through SlotTable to maintain identity mappings
        tensors = encoded_batch.detach().unsqueeze(1).unbind(0)
        for i, tensor in enumerate(tensors):
            self.table.append(
                tensor,
                bucket_id=bid_list[i],
                event_id=event_list[i],
                survival=1.0,
            )

        # Compress if capped
        if self.max_slots > 0 and len(self.table) > self.max_slots:
            self._compress()

    def get_extra_state(self) -> dict:
        """Persist slots, survival scores, bucket assignments, latent traces, and affinity."""
        state = {
            "slots": [s.cpu() for s in self.table._slots],
            "survival": list(self.table._survival),
            "slot_buckets": list(self.table._slot_buckets),
            "slot_event_ids": list(self.table._slot_event_ids),
            "table_state": self.table.state_dict(),
            "latent_traces": [
                {"bucket_id": t["bucket_id"], "centroid_contrib": t["centroid_contrib"].cpu()}
                for t in self._latent_traces
            ],
        }
        if self._bucket_affinity is not None:
            state["bucket_affinity"] = self._bucket_affinity.cpu()
        return state

    def set_extra_state(self, state: dict) -> None:
        """Restore slots, survival scores, bucket assignments, latent traces, and affinity."""
        device = self.decoder.weight.device
        if "table_state" in state:
            self.table.load_state_dict(state["table_state"], device=device)
        else:
            self.table._slots = [s.to(device) for s in state["slots"]]
            self.table._survival = list(state["survival"])
            self.table._slot_buckets = list(state.get("slot_buckets", [-1] * len(self.table._slots)))
            self.table._slot_event_ids = list(state.get("slot_event_ids", [0] * len(self.table._slots)))
            self.table._priority = [1.0] * len(self.table._slots)
            self._rebuild_table_identity()
        self._latent_traces = [
            {"bucket_id": t["bucket_id"], "centroid_contrib": t["centroid_contrib"].to(device)}
            for t in state.get("latent_traces", [])
        ]
        if "bucket_affinity" in state:
            self._bucket_affinity = state["bucket_affinity"].to(device)

    def read(
        self,
        batch_size: int,
        *,
        cue: torch.Tensor | None = None,
        read_cutoff: int | None = None,
    ) -> torch.Tensor:
        """Cue-dependent retrieval across slots.

        If no slots exist, returns zeros. If cue is provided (batch, model_dim),
        similarity between cue and each slot weights the decode.
        Retrieval weights are modulated by the control plane's priority vector.
        """
        if len(self.table) == 0:
            self._retrieval_weights = None
            self._retrieval_indices = None
            return torch.zeros(batch_size, self.model_dim, device=self.decoder.weight.device)
        indices = self.table.visible_indices(read_cutoff=read_cutoff)
        if not indices:
            self._retrieval_weights = None
            self._retrieval_indices = None
            return torch.zeros(batch_size, self.model_dim, device=self.decoder.weight.device)

        slot_matrix = self.table.slot_matrix(indices)

        if cue is not None:
            cue = cue.to(dtype=self.cue_proj.weight.dtype)
            cue_outer = self.cue_proj(cue)
            sim = torch.mm(cue_outer, slot_matrix.T)
            weights = F.softmax(sim, dim=-1)

            # Residual injector: control plane priority modulates retrieval
            priority = self.table.priority_vector(indices)
            weights = weights * priority.to(device=weights.device, dtype=weights.dtype).unsqueeze(0)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)

            self._retrieval_weights = weights.detach()
            self._retrieval_indices = indices
            retrieved = torch.mm(weights, slot_matrix)
        else:
            retrieved = slot_matrix.mean(dim=0, keepdim=True).expand(batch_size, -1)
            self._retrieval_weights = None
            self._retrieval_indices = None

        return self.decoder(retrieved.to(dtype=self.decoder.weight.dtype))

    def write(
        self,
        h: torch.Tensor,
        *,
        per_sample_weights: torch.Tensor | None = None,
        bucket_id: int | None = None,
    ) -> None:
        """Encode and append a new slot. Compress if at capacity.

        If bucket_id is provided, tags the slot with that bucket type from the
        Wernicke layer. Typed slots are only merged with same-type slots during
        compression.
        """
        h_enc = h.detach().to(dtype=self.encoder.weight.dtype)
        encoded = torch.tanh(self.encoder(h_enc))
        if per_sample_weights is not None:
            w = per_sample_weights.detach()
            w = w / w.sum().clamp_min(1e-8)
            slot = (w.unsqueeze(-1) * encoded).sum(dim=0, keepdim=True)
        else:
            slot = encoded.mean(dim=0, keepdim=True)

        self.table.append(
            slot.detach(),
            bucket_id=bucket_id if bucket_id is not None else -1,
        )

        # Compress if at capacity
        if len(self.table) > self.max_slots:
            self._compress()

    def write_sequence(
        self,
        h_seq: torch.Tensor,
        *,
        per_sample_weights: torch.Tensor | None = None,
        bucket_id: int | None = None,
    ) -> None:
        """Encode from full sequence hidden states (batch, seq, dim).

        Surprise-gated hippocampal encoding: promotes the hidden state
        trajectory to episodic storage when prediction error triggers
        consolidation. Recency-weighted to emphasize states closest to
        the surprising event. Uses exponentially-weighted mean with
        recency bias so later positions contribute more, preserving
        temporal order information that flat mean-pooling discards.
        """
        batch, seq, dim = h_seq.shape
        # Exponential recency weights: later positions matter more
        positions = torch.arange(seq, dtype=torch.float32, device=h_seq.device)
        weights = torch.exp(positions - positions[-1])  # exp-decay, last position = weight 1.0
        weights = weights / weights.sum()
        # Weighted sum over sequence: (batch, dim)
        h_pooled = (h_seq * weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)
        self.write(h_pooled, per_sample_weights=per_sample_weights, bucket_id=bucket_id)

    def read_bucket(
        self,
        batch_size: int,
        bucket_id: int,
        mode: str = "bucket_mean",
        k: int = 8,
        cue: torch.Tensor | None = None,
        read_cutoff: int | None = None,
    ) -> torch.Tensor:
        """Retrieve from a specific Wernicke bucket using the specified mode.

        All modes return tensors decoded to model_dim (via self.decoder),
        not outer_dim. This is critical -- the model stream operates in
        model_dim space.

        Modes:
          bucket_mean   -- mean of all entries in the bucket
          bucket_recent -- mean of last k entries in the bucket
          bucket_topk   -- softmax-weighted top-k by dot product with cue
          softmax_all   -- softmax over ALL slots (ignores bucket_id)
        """
        device = self.decoder.weight.device
        dtype = self.decoder.weight.dtype

        if not self._slots:
            return torch.zeros(batch_size, self.model_dim, device=device)

        if mode == "softmax_all":
            # Use all slots regardless of bucket -- baseline mode
            indices = self._visible_slot_indices(read_cutoff=read_cutoff)
            if not indices:
                return torch.zeros(batch_size, self.model_dim, device=device)
            slot_matrix = torch.cat([self._slots[i] for i in indices], dim=0).to(dtype=dtype)  # (n, outer_dim)
            if cue is not None:
                q = cue[0:1].to(dtype=dtype)  # (1, outer_dim)
                scores = (slot_matrix @ q.T).squeeze(-1)  # (n,)
                weights = torch.softmax(scores, dim=0)  # (n,)
                retrieved = (weights.unsqueeze(-1) * slot_matrix).sum(dim=0, keepdim=True)
            else:
                retrieved = slot_matrix.mean(dim=0, keepdim=True)
            decoded = self.decoder(retrieved)  # (1, model_dim)
            return decoded.expand(batch_size, -1)

        # Gather slots belonging to this bucket
        indices = self._visible_slot_indices(
            read_cutoff=read_cutoff,
            bucket_id=bucket_id,
        )
        if not indices:
            return torch.zeros(batch_size, self.model_dim, device=device)

        bucket_slots = torch.cat([self._slots[i] for i in indices], dim=0).to(dtype=dtype)  # (n, outer_dim)

        if mode == "bucket_mean":
            retrieved = bucket_slots.mean(dim=0, keepdim=True)  # (1, outer_dim)

        elif mode == "bucket_recent":
            recent = bucket_slots[-k:]  # last k entries
            retrieved = recent.mean(dim=0, keepdim=True)  # (1, outer_dim)

        elif mode == "bucket_topk":
            if cue is None:
                raise ValueError("bucket_topk requires a cue tensor")
            q = cue[0:1].to(dtype=dtype)  # (1, outer_dim)
            scores = (bucket_slots @ q.T).squeeze(-1)  # (n,)
            topk_idx = scores.topk(min(k, len(scores))).indices
            topk_slots = bucket_slots[topk_idx]  # (k, outer_dim)
            topk_scores = scores[topk_idx]
            weights = torch.softmax(topk_scores, dim=0)  # (k,)
            retrieved = (weights.unsqueeze(-1) * topk_slots).sum(dim=0, keepdim=True)

        else:
            raise ValueError(f"Unknown retrieval mode: {mode}")

        # Decode from outer_dim back to model_dim so the retrieved vector
        # can be added directly to the hidden stream (dimensional alignment)
        decoded = self.decoder(retrieved)  # (1, model_dim)
        return decoded.expand(batch_size, -1)

    def append_kv(self, encoded: torch.Tensor, bucket_id: int | None = None) -> None:
        """Unconditional append -- no surprise gating, no compression if unlimited.

        Used by the append_only buffer mode. Stores the pre-encoded vector
        directly (caller is responsible for encoding via self.encoder).
        """
        self.table.append(
            encoded.detach(),
            bucket_id=bucket_id if bucket_id is not None else -1,
            survival=1.0,
        )
        # Only compress if max_slots > 0 (0 = unlimited)
        if self.max_slots > 0 and len(self.table) > self.max_slots:
            self._compress()

    def update_survival(self, current_loss: float) -> None:
        """Update survival scores for slots that contributed to last retrieval.

        Impact = how much better is the model when this slot fires?
        survival_score += (running_avg - actual_loss) weighted by retrieval weight.
        """
        if self._retrieval_weights is None or len(self.table) == 0:
            return
        running_avg = self.loss_ema.item()
        impact = running_avg - current_loss
        mean_weights = self._retrieval_weights.mean(dim=0)
        retrieval_indices = self._retrieval_indices or list(range(mean_weights.size(0)))
        for pos, slot_idx in enumerate(retrieval_indices[:mean_weights.size(0)]):
            if 0 <= slot_idx < len(self.table._survival):
                self.table._survival[slot_idx] += impact * mean_weights[pos].item()

    def _compress(self) -> None:
        """Merge the oldest slots with lowest survival scores.

        Takes the oldest half, sorts by survival, merges the bottom
        compress_ratio into one averaged slot. High-survival old slots survive.
        When bucket types are present, only merges slots sharing the same bucket.

        Uses SlotTable retire/append to maintain persistent identity mappings.
        """
        n = len(self.table)
        if n <= self.compress_ratio:
            return

        n_old = n // 2
        if n_old < self.compress_ratio:
            return

        # Get slot IDs in insertion order (oldest first)
        all_sids = sorted(self.table.active_slot_ids())
        old_sids = all_sids[:n_old]

        # Snapshot old slot data before mutation
        old_tensors = [self.table.get_tensor(sid) for sid in old_sids]
        old_survival = [self.table._survival[self.table.slot_id_to_physical(sid)] for sid in old_sids]
        old_buckets = [self.table._slot_buckets[self.table.slot_id_to_physical(sid)] for sid in old_sids]
        old_event_ids = [self.table._slot_event_ids[self.table.slot_id_to_physical(sid)] for sid in old_sids]

        has_types = any(b != -1 for b in old_buckets)

        retire_sids: list[int] = []
        append_specs: list[tuple[torch.Tensor, float, int, int]] = []

        if has_types:
            bucket_groups: dict[int, list[int]] = defaultdict(list)
            for i in range(n_old):
                bucket_groups[old_buckets[i]].append(i)

            for bucket_id, group_indices in bucket_groups.items():
                if len(group_indices) >= self.compress_ratio:
                    if self.compression_selection == "random":
                        group_sorted = list(group_indices)
                        self._compress_rng.shuffle(group_sorted)
                    else:
                        group_sorted = sorted(group_indices, key=lambda i: old_survival[i])
                    merge_count = min(self.compress_ratio, len(group_sorted))
                    to_merge = group_sorted[:merge_count]

                    merged_tensors = [old_tensors[i] for i in to_merge]
                    merged_survivals = [old_survival[i] for i in to_merge]
                    weights = torch.tensor([max(s, 0.01) for s in merged_survivals])
                    weights = weights / weights.sum()
                    merged_slot = sum(w.item() * t for w, t in zip(weights, merged_tensors))
                    merged_score = sum(merged_survivals) / len(merged_survivals)
                    merged_event_id = max(old_event_ids[i] for i in to_merge)

                    original_mean = sum(merged_tensors) / len(merged_tensors)
                    quality_delta = -float((merged_slot - original_mean).norm())
                    self._compression_consequences.append((bucket_id, quality_delta))

                    for i in to_merge:
                        self._latent_traces.append({
                            "bucket_id": old_buckets[i],
                            "centroid_contrib": old_tensors[i].detach().clone(),
                        })
                        retire_sids.append(old_sids[i])

                    append_specs.append((merged_slot, merged_score, bucket_id, merged_event_id))
        else:
            if self.compression_selection == "random":
                indices = list(range(n_old))
                self._compress_rng.shuffle(indices)
            else:
                indices = sorted(range(n_old), key=lambda i: old_survival[i])
            merge_count = min(self.compress_ratio, n_old)
            to_merge_idx = indices[:merge_count]

            if to_merge_idx:
                merged_tensors = [old_tensors[i] for i in to_merge_idx]
                merged_survivals = [old_survival[i] for i in to_merge_idx]
                weights = torch.tensor([max(s, 0.01) for s in merged_survivals])
                weights = weights / weights.sum()
                merged_slot = sum(w.item() * t for w, t in zip(weights, merged_tensors))
                merged_score = sum(merged_survivals) / len(merged_survivals)
                merged_event_id = max(old_event_ids[i] for i in to_merge_idx)

                for i in to_merge_idx:
                    self._latent_traces.append({
                        "bucket_id": old_buckets[i],
                        "centroid_contrib": old_tensors[i].detach().clone(),
                    })
                    retire_sids.append(old_sids[i])

                append_specs.append((merged_slot, merged_score, -1, merged_event_id))

        # Execute through SlotTable to maintain identity mappings
        if retire_sids:
            self.table.retire_many(retire_sids, reason="compressed")
        for tensor, survival, bucket, event_id in append_specs:
            self.table.append(tensor, bucket_id=bucket, event_id=event_id, survival=survival)

        while len(self._latent_traces) > self.max_slots:
            self._latent_traces.pop(0)

    def try_reactivate(self, bucket_id: int | None, surprise: float, reactivation_threshold: float = 1.0) -> bool:
        """Attempt to reactivate a latent trace matching the given bucket.

        Returns True if a trace was reactivated (added back as a slot), False otherwise.
        Only fires when surprise exceeds threshold AND a matching latent trace exists.
        When bucket_id is None, matches any latent trace (untyped fallback).
        Reactivated memories are degraded (Gaussian noise added) reflecting the
        reconstructive nature of memory retrieval — consolidated memories are
        rebuilt, not replayed.
        """
        if surprise < reactivation_threshold:
            return False
        for i, trace in enumerate(self._latent_traces):
            if bucket_id is None or trace["bucket_id"] == bucket_id:
                reactivated_slot = trace["centroid_contrib"].clone()
                noise = torch.randn_like(reactivated_slot) * 0.1
                reactivated_slot = reactivated_slot + noise
                self.table.append(
                    reactivated_slot,
                    bucket_id=trace["bucket_id"],
                )
                self._latent_traces.pop(i)
                if len(self.table) > self.max_slots:
                    self._compress()
                return True
        return False

    # ------------------------------------------------------------------
    # Partition-scoped slot queries
    # ------------------------------------------------------------------

    def get_partition_slot_indices(self, partition: Any) -> list[int]:
        """Return indices of slots owned by this partition (bucket_id in partition.bucket_ids)."""
        return [
            i for i, b in enumerate(self._slot_buckets)
            if partition.owns_bucket(b)
        ]

    def partition_slot_count(self, partition: Any) -> int:
        """Count slots owned by this partition."""
        return len(self.get_partition_slot_indices(partition))

    def is_write_allowed(self, bucket_id: int, partitions: list[Any]) -> bool:
        """Check if any awake partition owns this bucket_id."""
        for p in partitions:
            if p.owns_bucket(bucket_id) and p.is_awake:
                return True
        return False

    # ------------------------------------------------------------------
    # Bucket affinity matrix
    # ------------------------------------------------------------------

    def _ensure_affinity(self, n_buckets: int) -> torch.Tensor:
        """Lazy-init or grow the affinity matrix to cover n_buckets."""
        if self._bucket_affinity is None:
            self._bucket_affinity = torch.eye(n_buckets)
        elif self._bucket_affinity.shape[0] < n_buckets:
            old = self._bucket_affinity
            self._bucket_affinity = torch.eye(n_buckets)
            self._bucket_affinity[:old.shape[0], :old.shape[1]] = old
        return self._bucket_affinity

    def bucket_affinity(self, a: int, b: int) -> float:
        """Get merge affinity between two buckets. 1.0 = same bucket, 0.0 = no affinity."""
        if self._bucket_affinity is None:
            return 1.0 if a == b else 0.0
        n = self._bucket_affinity.shape[0]
        if a >= n or b >= n:
            return 1.0 if a == b else 0.0
        return float(self._bucket_affinity[a, b].item())

    def update_affinity(self, a: int, b: int, delta: float, lr: float = 0.05) -> None:
        """Update affinity between buckets a and b.

        Positive delta = merge was good (increase affinity).
        Negative delta = merge was bad (decrease affinity).
        Diagonal (same bucket) is always 1.0.
        """
        if a == b:
            return  # same-bucket affinity is always 1.0
        n = max(a, b) + 1
        aff = self._ensure_affinity(n)
        aff[a, b] = max(0.0, min(1.0, aff[a, b] + lr * delta))
        aff[b, a] = aff[a, b]  # symmetric

    def affinity_clusters(self, threshold: float = 0.3) -> list[set[int]]:
        """Discover coarse bucket groups from the learned affinity matrix.

        Connected components where affinity > threshold form clusters.
        These are the emergent "coarse buckets" — hierarchical types
        discovered from data, not designed.
        """
        if self._bucket_affinity is None:
            return []
        n = self._bucket_affinity.shape[0]
        visited: set[int] = set()
        clusters: list[set[int]] = []
        for i in range(n):
            if i in visited:
                continue
            # BFS from bucket i
            cluster: set[int] = set()
            queue = [i]
            while queue:
                node = queue.pop()
                if node in visited:
                    continue
                visited.add(node)
                cluster.add(node)
                for j in range(n):
                    if j not in visited and self._bucket_affinity[node, j] > threshold:
                        queue.append(j)
            clusters.append(cluster)
        return clusters

    def compute_consolidation_signal(self, current_loss: float, running_avg: float) -> float:
        """Same as OuterModel — surprise magnitude."""
        pain = max(current_loss - running_avg, 0.0)
        reward = max(running_avg - current_loss, 0.0)
        if self.consolidation_mode == "symmetric":
            return abs(current_loss - running_avg)
        elif self.consolidation_mode == "pain_biased":
            return pain + 0.5 * reward
        elif self.consolidation_mode == "learned":
            w = torch.sigmoid(self.consolidation_w).item()
            return w * pain + (1.0 - w) * reward
        raise ValueError(f"unsupported consolidation_mode: {self.consolidation_mode}")

    def consolidation_step(
        self,
        h: torch.Tensor,
        current_loss: float,
        per_sample_weights: torch.Tensor | None = None,
        bucket_id: int | None = None,
    ) -> float:
        """Full step: trigger logic, write, survival update, EMA update."""
        running_avg = self.loss_ema.item()
        signal = self.compute_consolidation_signal(current_loss, running_avg)
        is_pain = current_loss > running_avg

        # Learned mode gradient-free update
        if self.consolidation_mode == "learned" and self._last_wrote.item():
            loss_improved = current_loss < self._last_loss.item()
            if loss_improved:
                if self._last_signal_was_pain.item():
                    self.consolidation_w = self.consolidation_w + 0.01
                else:
                    self.consolidation_w = self.consolidation_w - 0.01

        # Update survival scores based on last retrieval
        self.update_survival(current_loss)

        # Trigger logic (same as OuterModel)
        surprise_threshold = running_avg > 0 and signal / running_avg > 0.01
        wrote = False

        write_kw: dict[str, Any] = {"per_sample_weights": per_sample_weights}
        if bucket_id is not None:
            write_kw["bucket_id"] = bucket_id

        if self.trigger == "immediate":
            if surprise_threshold:
                self.write(h, **write_kw)
                wrote = True
        elif self.trigger == "resolution":
            if surprise_threshold and not self._spike_seen:
                self._spike_seen = True
                self._pre_spike_loss = running_avg
            if self._spike_seen:
                if current_loss <= self._pre_spike_loss:
                    self.write(h, **write_kw)
                    wrote = True
                    self._spike_seen = False
        elif self.trigger == "windowed":
            if surprise_threshold and not self._spike_seen:
                self._spike_seen = True
                self._steps_since_spike = 0
            if self._spike_seen:
                self._steps_since_spike += 1
                if self._steps_since_spike >= self.trigger_window:
                    self.write(h, **write_kw)
                    wrote = True
                    self._spike_seen = False

        if self.consolidation_mode == "learned":
            self._last_signal_was_pain = torch.tensor(is_pain)
            self._last_loss = torch.tensor(current_loss)
            self._last_wrote = torch.tensor(wrote)

        self.loss_ema = self.ema_decay * self.loss_ema + (1 - self.ema_decay) * current_loss
        return signal


class BucketPrototypes(nn.Module):
    """Per-bucket semantic priors. One prototype per Wernicke bucket type,
    EMA-updated from buffer entries. Ships in artifact for cold-start context.

    Stores prototypes in prototype_dim (typically == outer_dim) and decodes
    to model_dim on read. All memory reads must be decoded to model_dim
    before being added to the hidden stream (dimensional alignment).
    """

    def __init__(
        self,
        k_max: int,
        prototype_dim: int,
        model_dim: int,
        update_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.k_max = k_max
        self.prototype_dim = prototype_dim
        self.model_dim = model_dim
        self.update_rate = update_rate
        self.register_buffer("prototypes", torch.zeros(k_max, prototype_dim))
        # Decoder: prototype_dim -> model_dim (on forward path, receives task gradients)
        self.decoder = nn.Linear(prototype_dim, model_dim, bias=False)

    def read(self, batch_size: int, bucket_id: int) -> torch.Tensor:
        """Read prototype for bucket, decoded to model_dim."""
        proto = self.prototypes[bucket_id].unsqueeze(0)  # (1, prototype_dim)
        decoded = self.decoder(proto.to(dtype=self.decoder.weight.dtype))  # (1, model_dim)
        return decoded.expand(batch_size, -1)

    def update(self, bucket_id: int, value: torch.Tensor) -> None:
        """EMA update of prototype from new observation(s) in prototype_dim."""
        v = value.detach().mean(dim=0)  # (prototype_dim,)
        self.prototypes[bucket_id] = (
            (1 - self.update_rate) * self.prototypes[bucket_id]
            + self.update_rate * v
        )

    def update_batch(self, bucket_ids: torch.Tensor, values: torch.Tensor) -> None:
        """Batched EMA update: one GPU→CPU transfer, aggregate per bucket.

        Args:
            bucket_ids: (N,) integer bucket assignments.
            values: (N, prototype_dim) pre-encoded vectors.
        """
        bid_list = bucket_ids.tolist()
        vals = values.detach()
        # Group by bucket and average, then single EMA update per bucket
        seen: dict[int, list[int]] = {}
        for i, bid in enumerate(bid_list):
            seen.setdefault(bid, []).append(i)
        for bid, indices in seen.items():
            bucket_vals = vals[indices]  # (k, prototype_dim)
            v = bucket_vals.mean(dim=0)  # (prototype_dim,)
            self.prototypes[bid] = (
                (1 - self.update_rate) * self.prototypes[bid]
                + self.update_rate * v
            )


class SemanticTier(nn.Module):
    """Neocortical knowledge layer — always-on background bias.

    Stores slowly-updated basis vectors extracted from episodic experience.
    Not cue-dependent, not gated — shapes all processing as a persistent prior.

    The semantic tier is to the episodic tier what "waterfronts have workers"
    is to "I saw the fisherman on Tuesday." Gist, not episodes.
    """

    def __init__(self, model_dim: int, num_bases: int = 8, update_rate: float = 0.01) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.num_bases = num_bases
        self.update_rate = update_rate
        # Decoder is on the forward path — receives task gradients
        self.decoder = nn.Linear(num_bases, model_dim, bias=False)
        # Encoder is structural — does not receive task gradients
        self.encoder = nn.Linear(model_dim, num_bases, bias=False)
        self.encoder.weight.requires_grad_(False)
        # Basis vectors persist across sequences
        self.register_buffer("bases", torch.zeros(1, num_bases))

    def read(self, batch_size: int) -> torch.Tensor:
        """Always-on bias — added to recurrence at every step."""
        return self.decoder(self.bases.expand(batch_size, -1))

    def consolidate_from_episodes(self, episode_vectors: torch.Tensor) -> None:
        """Extract shared structure from episode vectors into bases.

        Args:
            episode_vectors: (N, model_dim) — recent episodic slot contents
        """
        encoded = self.encoder(episode_vectors.detach().to(dtype=self.encoder.weight.dtype)).mean(dim=0, keepdim=True)
        self.bases = ((1 - self.update_rate) * self.bases.detach() + self.update_rate * encoded).detach()
