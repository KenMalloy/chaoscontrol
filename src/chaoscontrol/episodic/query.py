"""Cosine-utility-weighted top-K query helper for the episodic cache.

Phase 2 Task 2.3 of ``docs/plans/2026-04-25-memory-aware-optimizer-plan.md``.

The CPU controller drains the per-rank ``controller_query_queue`` and
calls this helper for each query candidate. The helper runs on the
episodic rank's GPU (where the cache lives) and returns the top-K slot
indices ranked by one of two scoring modes:

  ``"cosine_utility_weighted"`` (Decision 0.2 — the production read):
      ``score(i) = cosine_sim(query_residual, cache.key_rep[i]) * cache.utility_u[i]``

  ``"pressure_only"`` (Phase 3 Arm B' — the mechanism-specificity arm):
      ``score(i) = cache.utility_u[i]``
      The Phase 3 falsifier matrix uses this arm to distinguish "memory
      persistence + similarity recall" from "any rare-grad-aligned
      retrieval policy." ``EpisodicCache`` does not store a separate
      ``pressure_at_write`` field, so ``utility_u`` is the proxy: the
      utility EMA already encodes "did this entry produce gradients
      aligned with the live rare-grad direction" per Decision 0.10. If
      Phase 3 results suggest a true write-time pressure is needed, add
      a ``pressure_at_write`` tensor to the cache schema first.

Both modes exclude unoccupied slots from the candidate set; the empty
cache returns an empty int64 tensor without raising.
"""
from __future__ import annotations

import torch

from chaoscontrol.optim.episodic_cache import EpisodicCache

# Set of accepted ``score_mode`` strings. Centralized so any addition
# (e.g. a learned predictor mode in Phase 5+) flows through one place.
_VALID_SCORE_MODES = ("cosine_utility_weighted", "pressure_only")


def query_topk(
    query_residual: torch.Tensor,
    cache: EpisodicCache,
    k: int,
    score_mode: str = "cosine_utility_weighted",
) -> torch.Tensor:
    """Return the top-K occupied slot indices ranked by ``score_mode``.

    Args:
        query_residual: ``[D]`` fp32 tensor (the encode-output residual
            captured at write time, or the live query residual produced
            by the train rank). Device may be CPU or GPU; cache tensors
            are moved to ``query_residual.device`` for the score op.
        cache: the ``EpisodicCache`` to query against. Only occupied
            slots participate.
        k: requested upper bound on the output length. The actual count
            is ``min(k, num_occupied_slots)``; an empty cache returns an
            empty tensor regardless of ``k``.
        score_mode: one of ``"cosine_utility_weighted"`` (Decision 0.2)
            or ``"pressure_only"`` (Phase 3 Arm B').

    Returns:
        ``[k_eff]`` int64 tensor of slot indices into the original cache
        (NOT the post-occupied-filter array). Ordered by score, highest
        first. Output device matches ``query_residual.device``.

    Raises:
        ValueError: if ``score_mode`` is not one of the documented modes.
    """
    if score_mode not in _VALID_SCORE_MODES:
        raise ValueError(
            f"score_mode must be one of {_VALID_SCORE_MODES}; got "
            f"{score_mode!r}"
        )
    occupied = cache.occupied
    # Empty cache short-circuit. Returning an int64 tensor on the query
    # device matches the non-empty path so the caller never has to
    # branch on a None or check tensor dtype.
    if not occupied.any():
        return torch.empty(0, dtype=torch.int64, device=query_residual.device)

    device = query_residual.device
    occupied_dev = occupied.to(device)
    occupied_idx = occupied_dev.nonzero(as_tuple=True)[0]  # [N_occ]

    # Per-mode score tensor, both built on ``query_residual.device`` so
    # the returned tensor lands there too.
    if score_mode == "cosine_utility_weighted":
        # Pull only the occupied rows of key_rep + utility_u onto the
        # query device. For typical cache size (4096 * 256 fp32 = 4 MB)
        # this copy is negligible; if it shows up in profiling we'd
        # cache the device tensors per query cycle.
        keys = cache.key_rep.to(device)[occupied_idx]  # [N_occ, D]
        util = cache.utility_u.to(device)[occupied_idx]  # [N_occ]
        # Cosine similarity. Add a small epsilon to the norms so a
        # pathological zero-norm key (shouldn't happen post-Pass-C, but
        # defensive) doesn't divide by zero.
        q = query_residual / (query_residual.norm() + 1e-8)
        keys_n = keys / (keys.norm(dim=1, keepdim=True) + 1e-8)
        cosines = keys_n @ q  # [N_occ]
        scores = cosines * util  # [N_occ]
    else:  # pressure_only
        # TODO(task #101): when EpisodicCache gains a pressure_at_write
        # field (Phase 3 prereq), replace this proxy with
        # ``cache.pressure_at_write[occupied_idx]``. Until then,
        # ``utility_u`` is the closest available signal — but it's
        # semantically wrong for the Arm B' mechanism-specificity test
        # (spec wants Arm B' to ignore cosine AND utility entirely).
        # Phase 3 falsifier loses some discriminative power until #101
        # lands. Documented in the spec-review report.
        scores = cache.utility_u.to(device)[occupied_idx]  # [N_occ]

    k_eff = min(int(k), int(scores.numel()))
    if k_eff <= 0:
        return torch.empty(0, dtype=torch.int64, device=device)
    _, top = torch.topk(scores, k_eff, largest=True)
    # Map back from the occupied-only positions to the original cache
    # slot indices. ``occupied_idx`` is already int64 from ``nonzero``.
    return occupied_idx[top].to(dtype=torch.int64)
