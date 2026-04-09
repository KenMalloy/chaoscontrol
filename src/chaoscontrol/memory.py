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

        # Slot storage (not parameters — runtime state, persisted via extra_state)
        self._slots: list[torch.Tensor] = []        # each (1, outer_dim)
        self._survival: list[float] = []             # per-slot survival score
        self._slot_buckets: list[int] = []           # per-slot bucket type from Wernicke
        self._retrieval_weights: torch.Tensor | None = None  # cached from last read
        self._compression_consequences: list[tuple[int, float]] = []  # (bucket_id, quality_delta)
        self._latent_traces: list[dict] = []  # {bucket_id: int, centroid_contrib: Tensor}

        # Bucket affinity matrix: learned cross-type merge compatibility.
        # affinity[a, b] = how safe it is to merge slots from bucket a with bucket b.
        # Starts as identity (same-bucket = 1.0, cross-bucket = 0.0).
        # Updated during sleep: committed merges increase affinity, rejected decrease.
        self._bucket_affinity: torch.Tensor | None = None  # (n_buckets, n_buckets), lazy init

    def get_extra_state(self) -> dict:
        """Persist slots, survival scores, bucket assignments, latent traces, and affinity."""
        state = {
            "slots": [s.cpu() for s in self._slots],
            "survival": list(self._survival),
            "slot_buckets": list(self._slot_buckets),
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
        self._slots = [s.to(device) for s in state["slots"]]
        self._survival = list(state["survival"])
        self._slot_buckets = list(state.get("slot_buckets", [-1] * len(self._slots)))
        self._latent_traces = [
            {"bucket_id": t["bucket_id"], "centroid_contrib": t["centroid_contrib"].to(device)}
            for t in state.get("latent_traces", [])
        ]
        if "bucket_affinity" in state:
            self._bucket_affinity = state["bucket_affinity"].to(device)

    def read(self, batch_size: int, *, cue: torch.Tensor | None = None) -> torch.Tensor:
        """Cue-dependent retrieval across slots.

        If no slots exist, returns zeros. If cue is provided (batch, model_dim),
        similarity between cue and each slot weights the decode.
        """
        if not self._slots:
            return torch.zeros(batch_size, self.model_dim, device=self.decoder.weight.device)

        # Stack slots: (num_slots, outer_dim)
        slot_matrix = torch.cat(self._slots, dim=0)

        if cue is not None:
            # Project cue to outer space for similarity
            cue = cue.to(dtype=self.cue_proj.weight.dtype)
            cue_outer = self.cue_proj(cue)  # (batch, outer_dim)
            # Similarity: (batch, num_slots)
            sim = torch.mm(cue_outer, slot_matrix.T)
            weights = F.softmax(sim, dim=-1)  # (batch, num_slots)
            # Cache for survival scoring
            self._retrieval_weights = weights.detach()
            # Weighted retrieval: (batch, outer_dim)
            retrieved = torch.mm(weights, slot_matrix)
        else:
            # Uniform retrieval (no cue)
            retrieved = slot_matrix.mean(dim=0, keepdim=True).expand(batch_size, -1)
            self._retrieval_weights = None

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

        self._slots.append(slot.detach())
        self._survival.append(0.0)
        self._slot_buckets.append(bucket_id if bucket_id is not None else -1)

        # Compress if at capacity
        if len(self._slots) > self.max_slots:
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
        positions = torch.arange(seq, dtype=h_seq.dtype, device=h_seq.device)
        weights = torch.exp(positions - positions[-1])  # exp-decay, last position = weight 1.0
        weights = weights / weights.sum()
        # Weighted sum over sequence: (batch, dim)
        h_pooled = (h_seq * weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)
        self.write(h_pooled, per_sample_weights=per_sample_weights, bucket_id=bucket_id)

    def update_survival(self, current_loss: float) -> None:
        """Update survival scores for slots that contributed to last retrieval.

        Impact = how much better is the model when this slot fires?
        survival_score += (running_avg - actual_loss) weighted by retrieval weight.
        """
        if self._retrieval_weights is None or not self._slots:
            return
        running_avg = self.loss_ema.item()
        impact = running_avg - current_loss  # positive = model did better than avg
        # Weight impact by how much each slot contributed to retrieval
        # Average across batch
        mean_weights = self._retrieval_weights.mean(dim=0)  # (num_slots,)
        for i in range(min(len(self._survival), mean_weights.size(0))):
            self._survival[i] += impact * mean_weights[i].item()

    def _compress(self) -> None:
        """Merge the oldest slots with lowest survival scores.

        Takes the oldest half, sorts by survival, merges the bottom
        compress_ratio into one averaged slot. High-survival old slots survive.
        When bucket types are present, only merges slots sharing the same bucket.
        """
        if len(self._slots) <= self.compress_ratio:
            return

        # Target: merge the N oldest low-survival slots into N//compress_ratio
        n_old = len(self._slots) // 2
        if n_old < self.compress_ratio:
            return

        old_slots = self._slots[:n_old]
        old_survival = self._survival[:n_old]
        old_buckets = self._slot_buckets[:n_old]
        new_slots = self._slots[n_old:]
        new_survival = self._survival[n_old:]
        new_buckets = self._slot_buckets[n_old:]

        # Check if we have typed slots (any bucket != -1)
        has_types = any(b != -1 for b in old_buckets)

        if has_types:
            # Typed compression: only merge within same bucket
            # Group old slots by bucket
            bucket_groups: dict[int, list[int]] = defaultdict(list)
            for i in range(n_old):
                bucket_groups[old_buckets[i]].append(i)

            kept_slots = []
            kept_survival = []
            kept_buckets = []

            for bucket_id, group_indices in bucket_groups.items():
                if len(group_indices) >= self.compress_ratio:
                    # Sort by survival ascending (or random for ablation), merge the lowest
                    if self.compression_selection == "random":
                        group_sorted = list(group_indices)
                        self._compress_rng.shuffle(group_sorted)
                    else:
                        group_sorted = sorted(group_indices, key=lambda i: old_survival[i])
                    merge_count = min(self.compress_ratio, len(group_sorted))
                    to_merge = group_sorted[:merge_count]
                    to_keep = group_sorted[merge_count:]

                    # Merge
                    merged_tensors = [old_slots[i] for i in to_merge]
                    merged_survivals = [old_survival[i] for i in to_merge]
                    weights = torch.tensor([max(s, 0.01) for s in merged_survivals])
                    weights = weights / weights.sum()
                    merged_slot = sum(w.item() * t for w, t in zip(weights, merged_tensors))
                    merged_score = sum(merged_survivals) / len(merged_survivals)

                    # Compression consequence: how much info was lost?
                    original_mean = sum(merged_tensors) / len(merged_tensors)
                    quality_delta = -float((merged_slot - original_mean).norm())
                    self._compression_consequences.append((bucket_id, quality_delta))

                    # Record latent traces for the individual slots absorbed by the merge
                    for i in to_merge:
                        self._latent_traces.append({
                            "bucket_id": old_buckets[i],
                            "centroid_contrib": old_slots[i].detach().clone(),
                        })

                    for i in sorted(to_keep):
                        kept_slots.append(old_slots[i])
                        kept_survival.append(old_survival[i])
                        kept_buckets.append(old_buckets[i])
                    kept_slots.append(merged_slot)
                    kept_survival.append(merged_score)
                    kept_buckets.append(bucket_id)
                else:
                    # Not enough slots in this bucket to merge — keep them all
                    for i in sorted(group_indices):
                        kept_slots.append(old_slots[i])
                        kept_survival.append(old_survival[i])
                        kept_buckets.append(old_buckets[i])

            self._slots = kept_slots + new_slots
            self._survival = kept_survival + new_survival
            self._slot_buckets = kept_buckets + new_buckets
        else:
            # Untyped compression: sort by survival (or random for ablation)
            if self.compression_selection == "random":
                indices = list(range(n_old))
                self._compress_rng.shuffle(indices)
            else:
                indices = sorted(range(n_old), key=lambda i: old_survival[i])
            merge_count = min(self.compress_ratio, n_old)
            to_merge_idx = indices[:merge_count]
            to_keep_idx = indices[merge_count:]

            if to_merge_idx:
                merged_tensors = [old_slots[i] for i in to_merge_idx]
                merged_survivals = [old_survival[i] for i in to_merge_idx]
                weights = torch.tensor([max(s, 0.01) for s in merged_survivals])
                weights = weights / weights.sum()
                merged_slot = sum(w.item() * t for w, t in zip(weights, merged_tensors))
                merged_score = sum(merged_survivals) / len(merged_survivals)

                # Record latent traces for the individual slots absorbed by the merge
                for i in to_merge_idx:
                    self._latent_traces.append({
                        "bucket_id": old_buckets[i],
                        "centroid_contrib": old_slots[i].detach().clone(),
                    })

                kept_old = [(old_slots[i], old_survival[i], old_buckets[i]) for i in sorted(to_keep_idx)]
                self._slots = [s for s, _, _ in kept_old] + [merged_slot] + new_slots
                self._survival = [sc for _, sc, _ in kept_old] + [merged_score] + new_survival
                self._slot_buckets = [b for _, _, b in kept_old] + [-1] + new_buckets

        # Evict oldest latent traces if over capacity
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
                # Reactivate with degradation — reconstructed memories are imperfect
                reactivated_slot = trace["centroid_contrib"].clone()
                noise = torch.randn_like(reactivated_slot) * 0.1  # reconstruction noise
                reactivated_slot = reactivated_slot + noise
                self._slots.append(reactivated_slot)
                self._survival.append(0.0)
                self._slot_buckets.append(trace["bucket_id"])
                self._latent_traces.pop(i)
                # Compress if reactivation pushed past capacity
                if len(self._slots) > self.max_slots:
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
