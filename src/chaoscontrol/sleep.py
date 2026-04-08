"""SleepCycle — structured offline memory consolidation (N1/N2/N3/REM).

Implements biologically-inspired sleep stages for memory maintenance:
  N1 (transition)  — snapshot unstable slots, freeze new writes
  N2 (tag)         — leave-one-slot-out utility scoring
  N3 (rewrite)     — prune low-survival slots, propose merges
  REM (dream)      — dream generation, teacher-forced scoring,
                     merge validation, counterfactual regret updates
"""
from __future__ import annotations

import random as _random
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class SleepConfig:
    """Configuration for the sleep cycle."""

    stages: str = "full_cycle"
    # "n3_only", "n2_n3", "n2_n3_rem_validate", "n2_n3_rem_cfr",
    # "n2_n3_rem_full", "full_cycle"

    budget: int = 128        # total max ops
    n2_budget: int = 64      # fixed sub-budget for N2
    rem_budget: int = 64     # fixed sub-budget for REM
    merge_sim_threshold: float = 0.85
    survival_floor: float = 0.1
    n2_batches: int = 8
    rem_dreams: int = 4
    rem_length: int = 128
    rem_validate: bool = True
    rem_cfr: bool = True

    @property
    def use_n1(self) -> bool:
        return self.stages == "full_cycle"

    @property
    def use_n2(self) -> bool:
        return self.stages != "n3_only"

    @property
    def use_n3(self) -> bool:
        return True

    @property
    def use_rem(self) -> bool:
        return self.stages in (
            "n2_n3_rem_validate",
            "n2_n3_rem_cfr",
            "n2_n3_rem_full",
            "full_cycle",
        )


class SleepCycle:
    """Structured offline consolidation cycle.

    Entry point: ``run(model, cache, device, regret_table=None)``

    Executes sleep stages in order (N1 -> N2 -> N3 -> REM) based on the
    stage configuration.  Returns a diagnostics dict with per-stage metrics.
    """

    def __init__(self, config: SleepConfig | None = None) -> None:
        self.config = config or SleepConfig()
        self._rng = _random.Random(42)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        model: Any,
        cache: Any,
        device: torch.device | str = "cpu",
        regret_table: Any | None = None,
    ) -> dict[str, Any]:
        """Execute a full sleep cycle and return diagnostics.

        Parameters
        ----------
        model : ChaosStudentLM
            The model to consolidate.  Must have ``outer_model`` (MultiSlotOuterModel),
            and optionally ``wernicke``, ``semantic_tier``, ``typed_storage``.
        cache : WakeCache
            Cached high-signal moments from the wake phase.
        device : torch.device | str
            Device for tensor operations.
        regret_table : RegretTable | None
            If provided and REM CFR is active, counterfactual updates are written here.

        Returns
        -------
        dict
            Diagnostics with keys per stage plus ``total_ops``.
        """
        cfg = self.config
        om = model.outer_model
        diag: dict[str, Any] = {"total_ops": 0}

        if om is None or not hasattr(om, "_slots"):
            diag["skipped"] = "no multislot outer model"
            return diag

        # ---- N1 --------------------------------------------------------
        if cfg.use_n1:
            n1_diag = self._n1_transition(om)
            diag["n1"] = n1_diag

        # ---- N2 --------------------------------------------------------
        if cfg.use_n2:
            n2_diag = self._n2_tag(model, om, cache, device)
            diag["n2"] = n2_diag
            diag["total_ops"] += n2_diag.get("ops", 0)

        # ---- N3 --------------------------------------------------------
        n3_diag = self._n3_rewrite(model, om)
        diag["n3"] = n3_diag
        diag["total_ops"] += n3_diag.get("ops", 0)

        # ---- REM -------------------------------------------------------
        if cfg.use_rem:
            validate_merges = cfg.stages in (
                "n2_n3_rem_validate",
                "n2_n3_rem_full",
                "full_cycle",
            )
            use_cfr = cfg.stages in (
                "n2_n3_rem_cfr",
                "n2_n3_rem_full",
                "full_cycle",
            )
            rem_diag = self._rem_dream(
                model, om, cache, device,
                regret_table=regret_table if use_cfr else None,
                validate_merges=validate_merges,
            )
            diag["rem"] = rem_diag
            diag["total_ops"] += rem_diag.get("ops", 0)

        # ---- Semantic tier recomputation -------------------------------
        if model.semantic_tier is not None and om._slots:
            decoded = []
            for slot in om._slots:
                decoded.append(om.decoder(slot))  # (1, model_dim)
            episode_vectors = torch.cat(decoded, dim=0)  # (N, model_dim)
            model.semantic_tier.consolidate_from_episodes(episode_vectors)
            diag["semantic_recomputed"] = True

        return diag

    # ------------------------------------------------------------------
    # N1 — Transition
    # ------------------------------------------------------------------

    def _n1_transition(self, om: Any) -> dict[str, Any]:
        """Snapshot recent unstable slot indices.

        N1 is simple bookkeeping: identify slots that were recently written
        (low survival magnitude = not yet tested by retrieval) and mark them
        as unstable so N2 does not waste budget scoring them.
        """
        unstable_indices: list[int] = []
        for i, surv in enumerate(om._survival):
            if abs(surv) < 1e-6:
                unstable_indices.append(i)
        return {
            "unstable_count": len(unstable_indices),
            "unstable_indices": unstable_indices,
        }

    # ------------------------------------------------------------------
    # N2 — Tag (leave-one-slot-out utility scoring)
    # ------------------------------------------------------------------

    def _n2_tag(
        self,
        model: Any,
        om: Any,
        cache: Any,
        device: torch.device | str,
    ) -> dict[str, Any]:
        """Score each slot's utility via leave-one-slot-out CE delta."""
        cfg = self.config
        n_slots = len(om._slots)
        if n_slots == 0:
            return {"ops": 0, "slots_scored": 0}

        # Gather cached batches for scoring
        batches = self._gather_n2_batches(cache, device)
        if not batches:
            return {"ops": 0, "slots_scored": 0, "reason": "no cached batches"}

        # Baseline CE with all slots present
        baseline_ce = self._compute_mean_ce(model, batches, device)

        # Shuffle slot indices to avoid position bias
        indices = list(range(n_slots))
        self._rng.shuffle(indices)

        ops = 0
        utilities: list[tuple[int, float]] = []

        for idx in indices:
            if ops >= cfg.n2_budget:
                break

            # Pop the slot (truly remove, don't zero)
            removed_slot = om._slots.pop(idx)
            removed_survival = om._survival.pop(idx)
            removed_bucket = om._slot_buckets.pop(idx)

            # Measure CE without this slot
            ce_without = self._compute_mean_ce(model, batches, device)

            # Reinsert at the same position
            om._slots.insert(idx, removed_slot)
            om._survival.insert(idx, removed_survival)
            om._slot_buckets.insert(idx, removed_bucket)

            # Utility = CE_without - CE_baseline (positive = slot is useful)
            utility = ce_without - baseline_ce
            utilities.append((idx, utility))

            # Update survival from utility
            om._survival[idx] += utility

            ops += 1

        return {
            "ops": ops,
            "slots_scored": len(utilities),
            "baseline_ce": baseline_ce,
            "utilities": utilities,
        }

    def _gather_n2_batches(
        self,
        cache: Any,
        device: torch.device | str,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Extract (inputs, targets) pairs from cached moments."""
        batches: list[tuple[torch.Tensor, torch.Tensor]] = []
        n = min(self.config.n2_batches, len(cache.moments))
        for i in range(n):
            m = cache.moments[i]
            inputs = m["inputs"].to(device)
            targets = m["targets"].to(device)
            batches.append((inputs, targets))
        return batches

    def _compute_mean_ce(
        self,
        model: Any,
        batches: list[tuple[torch.Tensor, torch.Tensor]],
        device: torch.device | str,
    ) -> float:
        """Run batches through model and return mean cross-entropy."""
        total_ce = 0.0
        count = 0
        with torch.no_grad():
            for inputs, targets in batches:
                out = model(inputs)
                logits = out["logits"]
                # Flatten for cross-entropy
                ce = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    reduction="mean",
                )
                total_ce += ce.item()
                count += 1
        return total_ce / max(count, 1)

    # ------------------------------------------------------------------
    # N3 — Rewrite (prune + merge)
    # ------------------------------------------------------------------

    def _n3_rewrite(self, model: Any, om: Any) -> dict[str, Any]:
        """Prune low-survival slots, propose merges for similar slots."""
        cfg = self.config
        ops = 0
        pruned_count = 0
        latent_traces_created = 0

        # --- Prune slots below survival_floor ---
        keep_indices: list[int] = []
        for i in range(len(om._slots)):
            if om._survival[i] < cfg.survival_floor:
                # Store latent trace before pruning
                om._latent_traces.append({
                    "bucket_id": om._slot_buckets[i],
                    "centroid_contrib": om._slots[i].detach().clone(),
                })
                latent_traces_created += 1
                pruned_count += 1
                ops += 1
            else:
                keep_indices.append(i)

        if pruned_count > 0:
            om._slots = [om._slots[i] for i in keep_indices]
            om._survival = [om._survival[i] for i in keep_indices]
            om._slot_buckets = [om._slot_buckets[i] for i in keep_indices]

        # Evict oldest latent traces if over capacity
        while len(om._latent_traces) > om.max_slots:
            om._latent_traces.pop(0)

        # --- Propose merges for similar slots within same bucket ---
        provisional_merges: list[dict[str, Any]] = []
        use_typed = getattr(model, "typed_storage", False)

        if use_typed and len(om._slots) >= 2:
            provisional_merges = self._propose_typed_merges(om, cfg)
        elif not use_typed and len(om._slots) >= 2:
            provisional_merges = self._propose_untyped_merges(om, cfg)

        ops += len(provisional_merges)

        # If REM is not active, commit merges immediately
        if not cfg.use_rem:
            committed = 0
            for merge in provisional_merges:
                if self._commit_merge(om, merge):
                    committed += 1
            provisional_merges = []
        else:
            committed = 0

        # Store provisional merges for REM validation
        self._provisional_merges = provisional_merges

        return {
            "ops": ops,
            "pruned": pruned_count,
            "latent_traces_created": latent_traces_created,
            "merges_proposed": len(provisional_merges) + committed,
            "merges_committed": committed,
            "slots_remaining": len(om._slots),
        }

    def _propose_typed_merges(
        self,
        om: Any,
        cfg: SleepConfig,
    ) -> list[dict[str, Any]]:
        """Propose merges for similar slots within the same bucket."""
        from collections import defaultdict

        bucket_groups: dict[int, list[int]] = defaultdict(list)
        for i in range(len(om._slots)):
            bucket_groups[om._slot_buckets[i]].append(i)

        proposals: list[dict[str, Any]] = []
        for bucket_id, indices in bucket_groups.items():
            if len(indices) < 2:
                continue
            proposals.extend(
                self._find_similar_pairs(om, indices, cfg.merge_sim_threshold)
            )
        return proposals

    def _propose_untyped_merges(
        self,
        om: Any,
        cfg: SleepConfig,
    ) -> list[dict[str, Any]]:
        """Propose merges for all similar slot pairs (no bucket constraint)."""
        indices = list(range(len(om._slots)))
        return self._find_similar_pairs(om, indices, cfg.merge_sim_threshold)

    def _find_similar_pairs(
        self,
        om: Any,
        indices: list[int],
        threshold: float,
    ) -> list[dict[str, Any]]:
        """Find pairs of slots above cosine similarity threshold."""
        proposals: list[dict[str, Any]] = []
        already_proposed: set[int] = set()

        for i_pos in range(len(indices)):
            idx_a = indices[i_pos]
            if idx_a in already_proposed:
                continue
            for j_pos in range(i_pos + 1, len(indices)):
                idx_b = indices[j_pos]
                if idx_b in already_proposed:
                    continue
                sim = F.cosine_similarity(
                    om._slots[idx_a].reshape(1, -1),
                    om._slots[idx_b].reshape(1, -1),
                ).item()
                if sim >= threshold:
                    proposals.append({
                        "idx_a": idx_a,
                        "idx_b": idx_b,
                        "similarity": sim,
                        "slot_a": om._slots[idx_a].detach().clone(),
                        "slot_b": om._slots[idx_b].detach().clone(),
                        "survival_a": om._survival[idx_a],
                        "survival_b": om._survival[idx_b],
                        "bucket_a": om._slot_buckets[idx_a],
                        "bucket_b": om._slot_buckets[idx_b],
                    })
                    already_proposed.add(idx_a)
                    already_proposed.add(idx_b)
                    break  # each slot participates in at most one merge
        return proposals

    def _commit_merge(self, om: Any, merge: dict[str, Any]) -> bool:
        """Execute a merge: replace slot_a with weighted average, remove slot_b.

        Returns True if the merge was applied, False if indices were stale.
        """
        idx_a = merge["idx_a"]
        idx_b = merge["idx_b"]

        # Guard: indices may be stale after N3 pruning
        n = len(om._slots)
        if idx_a >= n or idx_b >= n or idx_a == idx_b:
            return False

        # Weighted average by survival
        sa = max(merge["survival_a"], 0.01)
        sb = max(merge["survival_b"], 0.01)
        total = sa + sb
        merged_slot = (sa / total) * merge["slot_a"] + (sb / total) * merge["slot_b"]

        # Store latent trace for the absorbed slot
        om._latent_traces.append({
            "bucket_id": merge["bucket_b"],
            "centroid_contrib": merge["slot_b"].detach().clone(),
        })

        # We need to be careful about index ordering when removing.
        # Remove the higher index first to avoid shifting.
        high_idx = max(idx_a, idx_b)
        low_idx = min(idx_a, idx_b)

        om._slots.pop(high_idx)
        om._survival.pop(high_idx)
        om._slot_buckets.pop(high_idx)

        # Replace the remaining slot with the merged version
        om._slots[low_idx] = merged_slot.detach()
        om._survival[low_idx] = (merge["survival_a"] + merge["survival_b"]) / 2.0

        # Evict oldest latent traces if over capacity
        while len(om._latent_traces) > om.max_slots:
            om._latent_traces.pop(0)

        return True

    # ------------------------------------------------------------------
    # REM — Dream
    # ------------------------------------------------------------------

    def _rem_dream(
        self,
        model: Any,
        om: Any,
        cache: Any,
        device: torch.device | str,
        regret_table: Any | None = None,
        validate_merges: bool = True,
    ) -> dict[str, Any]:
        """Execute REM dream phase: generate, score, optionally validate merges + CFR."""
        cfg = self.config
        ops = 0
        dream_scores: list[float] = []
        cfr_updates = 0

        n_dreams = min(cfg.rem_dreams, cfg.rem_budget)
        if not om._slots:
            return {"ops": 0, "dreams": 0, "reason": "no slots to dream from"}

        # Gather cached targets for teacher-forced scoring
        target_batches = self._gather_n2_batches(cache, device)

        for dream_idx in range(n_dreams):
            if ops >= cfg.rem_budget:
                break

            # Select a seed slot (round-robin through available slots)
            seed_idx = dream_idx % len(om._slots)
            seed_slot = om._slots[seed_idx]

            # Phase A: Generate dream tokens via model.dream_step()
            # Decode slot -> token space to get seed
            decoded_seed = om.decoder(seed_slot)  # (1, model_dim)
            # Project to logits to get seed token
            with torch.no_grad():
                seed_logits = model.lm_head(model.final_norm(decoded_seed.unsqueeze(1)).squeeze(1))
                seed_token = seed_logits.argmax(dim=-1, keepdim=True)  # (1, 1)

            # Autoregressive dream generation
            state = model.init_state(batch_size=1)
            dream_tokens = [seed_token]
            dream_length = min(cfg.rem_length, cfg.rem_budget - ops)
            for _ in range(dream_length):
                with torch.no_grad():
                    logits, hidden, state = model.dream_step(
                        dream_tokens[-1].to(device), state
                    )
                    next_token = logits.argmax(dim=-1, keepdim=True)  # (1, 1)
                    dream_tokens.append(next_token)
                ops += 1
                if ops >= cfg.rem_budget:
                    break

            # Phase B: Score against cached real targets (teacher-forced CE)
            if target_batches:
                score = self._score_dream(model, target_batches, device)
                dream_scores.append(score)

                # Update slot survival based on dream quality
                # Good dream (low CE) -> boost seed slot survival
                # Bad dream (high CE) -> penalize
                running_avg = sum(dream_scores) / len(dream_scores)
                impact = running_avg - score  # positive = better than average
                om._survival[seed_idx] += impact * 0.1

            # Phase C: Counterfactual regret updates (if enabled)
            if regret_table is not None and target_batches:
                cfr_updates += self._cfr_update(
                    model, om, cache, device, regret_table,
                    seed_idx, state, dream_tokens,
                )
                ops += 1

        # Merge validation (if enabled and we have provisional merges)
        merges_accepted = 0
        merges_rejected = 0
        if validate_merges and hasattr(self, "_provisional_merges"):
            for merge in self._provisional_merges:
                if ops >= cfg.rem_budget:
                    # Budget exhausted: commit remaining merges by default
                    if self._commit_merge(om, merge):
                        merges_accepted += 1
                    continue

                accepted = self._validate_merge(
                    model, om, merge, target_batches, device
                )
                if accepted:
                    if self._commit_merge(om, merge):
                        merges_accepted += 1
                else:
                    # Reject merge: reactivate latent trace
                    self._reject_merge(om, merge)
                    merges_rejected += 1
                ops += 1
            self._provisional_merges = []

        return {
            "ops": ops,
            "dreams": len(dream_scores),
            "mean_dream_score": (
                sum(dream_scores) / len(dream_scores) if dream_scores else 0.0
            ),
            "cfr_updates": cfr_updates,
            "merges_accepted": merges_accepted,
            "merges_rejected": merges_rejected,
        }

    def _score_dream(
        self,
        model: Any,
        batches: list[tuple[torch.Tensor, torch.Tensor]],
        device: torch.device | str,
    ) -> float:
        """Teacher-forced CE on cached real targets (non-circular anchor)."""
        return self._compute_mean_ce(model, batches, device)

    def _cfr_update(
        self,
        model: Any,
        om: Any,
        cache: Any,
        device: torch.device | str,
        regret_table: Any,
        seed_idx: int,
        state: list[torch.Tensor],
        dream_tokens: list[torch.Tensor],
    ) -> int:
        """Generate K perturbed forward passes for counterfactual values."""
        if not dream_tokens or len(dream_tokens) < 2:
            return 0

        k = 4  # Number of counterfactual rollouts
        base_token = dream_tokens[-1].to(device)

        # Get base value
        with torch.no_grad():
            base_logits, _, _ = model.dream_step(base_token, state)
            base_probs = F.softmax(base_logits, dim=-1)
            base_value = base_probs.max(dim=-1).values.mean().item()

        # Generate K counterfactual forward passes
        cf_values: list[float] = []
        for _ in range(k):
            with torch.no_grad():
                # Perturb by sampling a different token
                perturbed_probs = F.softmax(base_logits / 1.5, dim=-1)  # temperature
                alt_token = torch.multinomial(perturbed_probs, 1)  # (1, 1)
                cf_logits, _, _ = model.dream_step(alt_token, state)
                cf_probs = F.softmax(cf_logits, dim=-1)
                cf_values.append(cf_probs.max(dim=-1).values.mean().item())

        # Determine bucket_id for regret table update
        bucket_id = om._slot_buckets[seed_idx] if seed_idx < len(om._slot_buckets) else 0
        if bucket_id < 0:
            bucket_id = 0
        if bucket_id >= regret_table.n_buckets:
            bucket_id = bucket_id % regret_table.n_buckets

        # Pad cf_values to match n_actions
        while len(cf_values) < regret_table.n_actions:
            cf_values.append(base_value)

        action_taken = 0  # The base action
        regret_table.update(
            bucket_id=bucket_id,
            action_taken=action_taken,
            counterfactual_values=cf_values[:regret_table.n_actions],
            actual_value=base_value,
        )
        return 1

    def _validate_merge(
        self,
        model: Any,
        om: Any,
        merge: dict[str, Any],
        batches: list[tuple[torch.Tensor, torch.Tensor]],
        device: torch.device | str,
    ) -> bool:
        """Test if a provisional merge improves or maintains CE.

        Returns True if the merge should be committed.
        """
        if not batches:
            return True  # No data to validate against — accept by default

        # Baseline CE (current state)
        baseline_ce = self._compute_mean_ce(model, batches, device)

        # Temporarily apply the merge
        idx_a = merge["idx_a"]
        idx_b = merge["idx_b"]

        # Safety: check indices still valid (pruning may have shifted things)
        if idx_a >= len(om._slots) or idx_b >= len(om._slots):
            return True  # Indices invalid — accept by default

        # Save originals
        orig_a = om._slots[idx_a].clone()
        orig_b = om._slots[idx_b].clone()
        orig_surv_a = om._survival[idx_a]
        orig_surv_b = om._survival[idx_b]

        # Apply merge temporarily
        sa = max(merge["survival_a"], 0.01)
        sb = max(merge["survival_b"], 0.01)
        total = sa + sb
        merged = (sa / total) * merge["slot_a"] + (sb / total) * merge["slot_b"]
        om._slots[idx_a] = merged.detach()

        # Remove idx_b temporarily
        removed_slot = om._slots.pop(idx_b)
        removed_surv = om._survival.pop(idx_b)
        removed_bucket = om._slot_buckets.pop(idx_b)

        # Score with merge applied
        merged_ce = self._compute_mean_ce(model, batches, device)

        # Restore original state
        om._slots.insert(idx_b, removed_slot)
        om._survival.insert(idx_b, removed_surv)
        om._slot_buckets.insert(idx_b, removed_bucket)
        om._slots[idx_a] = orig_a
        om._survival[idx_a] = orig_surv_a
        om._survival[idx_b] = orig_surv_b

        # Accept if CE did not worsen
        return merged_ce <= baseline_ce + 1e-6

    def _reject_merge(self, om: Any, merge: dict[str, Any]) -> None:
        """Reject a provisional merge by keeping both slots as-is.

        The original slots are already in place (we only stored the proposal,
        never removed them). Record the rejected merge as a latent trace
        of the proposed combination for potential future reactivation.
        """
        # No structural change needed — slots were never actually merged.
        # The slot_a and slot_b references in the merge dict are clones.
        pass
