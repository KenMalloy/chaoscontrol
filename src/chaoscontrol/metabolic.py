"""Metabolic fork: classical approximation of quantum generation+selection."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def metabolic_fork(
    model: Any,
    input_ids: torch.Tensor,
    *,
    k: int = 4,
    noise_std: float = 0.01,
    score_mode: str = "memory_consistency",
) -> dict[str, Any]:
    """Generate K candidate forward passes with perturbed embeddings, select best.

    This is the classical approximation of quantum generation+selection:
    K parallel rollouts with different perturbations, scored and selected.
    Called only when the metabolic gate opens (high surprise).

    The *model* argument is duck-typed — it must expose ``.embed``,
    ``.outer_model``, ``.layers``, ``.final_norm``, and ``.lm_head``.

    Args:
        model: A ChaosStudentLM (or compatible duck-typed object).
        input_ids: Token ids, shape ``(batch, seq)``.
        k: Number of candidate rollouts.
        noise_std: Std-dev of Gaussian perturbation added to embeddings.
        score_mode: One of ``"ensemble_agreement"``, ``"memory_consistency"``,
            or ``"loss_lookahead"``.

    Returns:
        The candidate dict with the highest score, containing keys
        ``"logits"`` and ``"hidden"``.
    """
    # Lazy import to avoid hard dependency on memory module until it is extracted
    try:
        from chaoscontrol.memory import MultiSlotOuterModel
    except ImportError:  # pragma: no cover – fallback while memory.py is not yet extracted
        MultiSlotOuterModel = None  # type: ignore[assignment,misc]

    x_base = model.embed(input_ids)
    batch, seq, dim = x_base.shape

    # Generate K candidates with different perturbations to embeddings
    candidates: list[dict[str, torch.Tensor]] = []
    for _ in range(k):
        noise = torch.randn_like(x_base) * noise_std
        x_perturbed = x_base + noise

        # Read from outer model (same for all candidates — it's the shared memory)
        if model.outer_model is not None:
            if MultiSlotOuterModel is not None and isinstance(
                model.outer_model, MultiSlotOuterModel
            ):
                cue = x_perturbed.detach().mean(dim=1)
                outer_read = model.outer_model.read(batch, cue=cue)
            else:
                outer_read = model.outer_model.read(batch)
            x_perturbed = x_perturbed + outer_read.unsqueeze(1)

        # Run through layers
        h = x_perturbed
        for layer in model.layers:
            h = layer(h)

        hidden = h
        h = model.final_norm(h)
        logits = model.lm_head(h)
        candidates.append({"logits": logits, "hidden": hidden})

    # Score candidates
    if score_mode == "ensemble_agreement":
        # Pick the candidate closest to the mean logits
        mean_logits = torch.stack([c["logits"] for c in candidates]).mean(dim=0)
        scores = [-F.mse_loss(c["logits"], mean_logits).item() for c in candidates]
    elif score_mode == "memory_consistency" and model.outer_model is not None:
        # Score by how consistent each candidate's hidden state is with memory
        if (
            MultiSlotOuterModel is not None
            and isinstance(model.outer_model, MultiSlotOuterModel)
            and model.outer_model._slots
        ):
            slot_matrix = torch.cat(model.outer_model._slots, dim=0)  # (num_slots, outer_dim)
            scores = []
            for c in candidates:
                h_last = c["hidden"][:, -1, :].detach()
                cue_outer = model.outer_model.cue_proj(h_last)  # (batch, outer_dim)
                sim = torch.mm(cue_outer, slot_matrix.T).mean().item()
                scores.append(sim)
        else:
            # Fallback: no slots or single-slot — use ensemble agreement
            mean_logits = torch.stack([c["logits"] for c in candidates]).mean(dim=0)
            scores = [-F.mse_loss(c["logits"], mean_logits).item() for c in candidates]
    elif score_mode == "loss_lookahead":
        # Score by CE loss against targets (requires targets, so use self-consistency)
        # Use each candidate's own prediction confidence as a proxy
        scores = []
        for c in candidates:
            probs = F.softmax(c["logits"], dim=-1)
            confidence = probs.max(dim=-1).values.mean().item()
            scores.append(confidence)
    else:
        # Default fallback
        mean_logits = torch.stack([c["logits"] for c in candidates]).mean(dim=0)
        scores = [-F.mse_loss(c["logits"], mean_logits).item() for c in candidates]

    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    return candidates[best_idx]
