"""Metabolic fork: classical approximation of quantum generation+selection.

Two approaches:
  metabolic_fork — pick-best: generate K candidates, score, return winner.
  metabolic_monte_carlo — distributional: generate K candidates, return
      ensemble statistics (mean, variance, entropy, uncertainty map) as
      learning signals. No winner is picked — the statistics ARE the signal.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def metabolic_fork(
    model: Any,
    input_ids: torch.Tensor,
    *,
    k: int = 4,
    noise_std: float = 0.01,
    score_mode: str = "memory_consistency",
    generation_mode: str = "noise",
    structured_proj: StructuredProjections | None = None,
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
        generation_mode: ``"noise"`` (default) adds Gaussian perturbation;
            ``"structured"`` uses learned projection heads via *structured_proj*.
        structured_proj: A :class:`StructuredProjections` instance. Required
            when *generation_mode* is ``"structured"``.

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

    # Generate K candidates
    candidates: list[dict[str, torch.Tensor]] = []

    if generation_mode == "structured" and structured_proj is not None:
        views = structured_proj(x_base)
        for view in views:
            x_candidate = view

            # Read from outer model (same for all candidates — it's the shared memory)
            if model.outer_model is not None:
                if MultiSlotOuterModel is not None and isinstance(
                    model.outer_model, MultiSlotOuterModel
                ):
                    cue = x_candidate.detach().mean(dim=1)
                    outer_read = model.outer_model.read(batch, cue=cue)
                else:
                    outer_read = model.outer_model.read(batch)
                x_candidate = x_candidate + outer_read.unsqueeze(1)

            # Run through layers
            h = x_candidate
            for layer in model.layers:
                h = layer(h)

            hidden = h
            h = model.final_norm(h)
            logits = model.lm_head(h)
            candidates.append({"logits": logits, "hidden": hidden})
    else:
        # Noise-based generation (default)
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


def metabolic_monte_carlo(
    model: Any,
    input_ids: torch.Tensor,
    *,
    k: int = 4,
    noise_std: float = 0.01,
    generation_mode: str = "noise",
    structured_proj: StructuredProjections | None = None,
) -> dict[str, Any]:
    """Monte Carlo metabolic gate: sample K candidates, return distributional statistics.

    Instead of picking a winner, treats K candidates as a Monte Carlo sample
    of the possibility space. The statistics of the sample — variance, entropy,
    agreement — are the signal, not any individual candidate.

    Returns:
        logits: Mean logits across K candidates (batch, seq, vocab).
        hidden: Mean hidden states across K candidates (batch, seq, dim).
        mc_stats: Dict of Monte Carlo statistics:
            - logits_var: Per-position variance across candidates (batch, seq).
              High = model is at a decision boundary.
            - entropy: Entropy of the mean distribution (batch, seq).
              High = uncertain about next token.
            - agreement: 1 - normalized variance. High = candidates agree.
              (batch, seq).
            - uncertainty_map: Combined uncertainty signal (batch, seq).
              Product of high variance AND high entropy = genuinely uncertain.
            - candidate_divergence: Mean KL divergence between each candidate
              and the ensemble mean (scalar). How spread out is the sample?
    """
    try:
        from chaoscontrol.memory import MultiSlotOuterModel
    except ImportError:
        MultiSlotOuterModel = None  # type: ignore[assignment,misc]

    x_base = model.embed(input_ids)
    batch, seq, dim = x_base.shape

    # Generate K candidates
    all_logits = []
    all_hidden = []

    if generation_mode == "structured" and structured_proj is not None:
        views = structured_proj(x_base)
        perturbations = views
    else:
        perturbations = [x_base + torch.randn_like(x_base) * noise_std for _ in range(k)]

    for x_candidate in perturbations:
        if model.outer_model is not None:
            if MultiSlotOuterModel is not None and isinstance(
                model.outer_model, MultiSlotOuterModel
            ):
                cue = x_candidate.detach().mean(dim=1)
                outer_read = model.outer_model.read(batch, cue=cue)
            else:
                outer_read = model.outer_model.read(batch)
            x_candidate = x_candidate + outer_read.unsqueeze(1)

        h = x_candidate
        for layer in model.layers:
            h = layer(h)

        hidden = h
        h = model.final_norm(h)
        logits = model.lm_head(h)
        all_logits.append(logits)
        all_hidden.append(hidden)

    # Stack: (K, batch, seq, vocab) and (K, batch, seq, dim)
    logits_stack = torch.stack(all_logits)
    hidden_stack = torch.stack(all_hidden)

    # Ensemble mean
    logits_mean = logits_stack.mean(dim=0)  # (batch, seq, vocab)
    hidden_mean = hidden_stack.mean(dim=0)  # (batch, seq, dim)

    # --- Monte Carlo statistics ---

    # Per-position variance across candidates (collapse vocab dim)
    logits_var = logits_stack.var(dim=0).mean(dim=-1)  # (batch, seq)

    # Entropy of the mean distribution
    mean_probs = F.softmax(logits_mean, dim=-1)  # (batch, seq, vocab)
    entropy = -(mean_probs * (mean_probs + 1e-10).log()).sum(dim=-1)  # (batch, seq)

    # Agreement: inverse of normalized variance
    var_normalized = logits_var / (logits_var.max() + 1e-8)
    agreement = 1.0 - var_normalized  # (batch, seq)

    # Uncertainty map: high variance AND high entropy = genuinely uncertain
    # (not just noisy but meaninglessly so)
    entropy_normalized = entropy / math.log(logits_mean.size(-1))  # normalize to [0, 1]
    uncertainty_map = var_normalized * entropy_normalized  # (batch, seq)

    # Candidate divergence: mean KL(candidate || ensemble) across all K
    log_mean_probs = (mean_probs + 1e-10).log()
    total_kl = 0.0
    for i in range(logits_stack.size(0)):
        cand_probs = F.softmax(logits_stack[i], dim=-1)
        kl = F.kl_div(log_mean_probs, cand_probs, reduction="batchmean", log_target=False)
        total_kl = total_kl + kl
    candidate_divergence = total_kl / logits_stack.size(0)

    return {
        "logits": logits_mean,
        "hidden": hidden_mean,
        "mc_stats": {
            "logits_var": logits_var,
            "entropy": entropy,
            "agreement": agreement,
            "uncertainty_map": uncertainty_map,
            "candidate_divergence": candidate_divergence,
        },
    }


def micro_mcts(
    model: Any,
    input_ids: torch.Tensor,
    *,
    n_rollouts: int = 4,
    horizon: int = 8,
    ucb_c: float = 1.41,
    value_proxy: str = "confidence",
) -> dict[str, Any]:
    """Deliberative planning gate: forward-only tree search triggered by surprise,
    analogous to System 2 / model-based reasoning. Uses the SSM recurrence as a
    world model for efficient O(1)-per-step rollouts.

    When surprise triggers the metabolic gate, runs N rollouts of depth H
    tokens from the current state, uses UCB to select which candidate branch
    to explore, backs up values, and returns root logits + search statistics.

    Args:
        model: Duck-typed model with embed, layers, final_norm, lm_head.
        input_ids: Token ids, shape (batch, seq).
        n_rollouts: Number of MCTS rollouts (0 = plain forward pass).
        horizon: Rollout depth in tokens.
        ucb_c: UCB exploration constant.
        value_proxy: "confidence" (max softmax prob) or "neg_ce" (negative CE).

    Returns:
        Dict with "logits", "hidden", and optionally "mcts_stats".
    """

    def _forward_pass(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run embedding tensor through layers -> norm -> lm_head."""
        h = x
        for layer in model.layers:
            h = layer(h)
        hidden = h
        h = model.final_norm(h)
        logits = model.lm_head(h)
        return logits, hidden

    # --- Fallback: n_rollouts=0 means plain forward pass ---
    if n_rollouts == 0:
        x = model.embed(input_ids)
        logits, hidden = _forward_pass(x)
        return {"logits": logits, "hidden": hidden}

    # --- Base forward pass to get root logits and hidden ---
    x_base = model.embed(input_ids)
    root_logits, root_hidden = _forward_pass(x_base)

    batch, seq = input_ids.shape

    # Get top-k candidate tokens from last-position logits as "actions"
    k = 8
    # root_logits shape: (batch, seq, vocab)
    last_logits = root_logits[:, -1, :]  # (batch, vocab)
    # Average across batch for action selection
    mean_last_logits = last_logits.mean(dim=0)  # (vocab,)
    _, top_actions = torch.topk(mean_last_logits, k)  # (k,)

    # MCTS bookkeeping
    visit_counts = torch.zeros(k, dtype=torch.long)
    value_sums = torch.zeros(k, dtype=torch.float)

    # Prior probabilities for PUCT: softmax of top-k logits
    top_logits = mean_last_logits[top_actions]  # (k,)
    prior_probs = F.softmax(top_logits, dim=0)  # (k,)

    total_visits = 0

    with torch.no_grad():
        # Build root recurrence state by stepping through input sequence.
        # This replaces O(N*H*seq) full forward passes with 1 sequential
        # setup pass + N*H cheap O(1) step() calls.
        state = model.init_state(batch)
        for t in range(seq):
            _, _, state = model.step(input_ids[:, t : t + 1], state)

        # state is now the recurrence state after the full input sequence
        root_state = [s.clone() for s in state]

        for _ in range(n_rollouts):
            # --- PUCT selection (AlphaZero-style) ---
            # UCB = Q(a) + c * P(a) * sqrt(N) / (1 + n(a))
            ucb_scores = torch.zeros(k)
            sqrt_total = math.sqrt(total_visits + 1)
            for a in range(k):
                q_val = (value_sums[a] / visit_counts[a]) if visit_counts[a] > 0 else 0.0
                exploration = ucb_c * prior_probs[a].item() * sqrt_total / (1 + visit_counts[a].item())
                ucb_scores[a] = q_val + exploration

            chosen_idx = ucb_scores.argmax().item()
            chosen_token = top_actions[chosen_idx]  # scalar token id

            # --- Rollout using step() interface: O(1) per token ---
            rollout_state = [s.clone() for s in root_state]
            current_token = chosen_token.unsqueeze(0).expand(batch).unsqueeze(-1)  # (batch, 1)

            for _step in range(horizon):
                step_logits, step_hidden, rollout_state = model.step(
                    current_token, rollout_state
                )
                # Sample next token from softmax distribution
                roll_probs = F.softmax(step_logits, dim=-1)
                current_token = torch.multinomial(roll_probs, 1)  # (batch, 1)

            # --- Value estimate from final step ---
            if value_proxy == "confidence":
                value = F.softmax(step_logits, dim=-1).max(dim=-1).values.mean().item()
            else:
                # neg_ce: negative cross-entropy (higher is better)
                value = F.softmax(step_logits, dim=-1).max(dim=-1).values.log().mean().item()

            # --- Back up ---
            visit_counts[chosen_idx] += 1
            value_sums[chosen_idx] += value
            total_visits += 1

    # Compute mean values per action
    mean_values = torch.zeros(k)
    for a in range(k):
        if visit_counts[a] > 0:
            mean_values[a] = value_sums[a] / visit_counts[a]

    # Root value: weighted average of action values by visit count
    if total_visits > 0:
        root_value = (mean_values * visit_counts.float()).sum().item() / total_visits
    else:
        root_value = 0.0

    # Blend MCTS search results into root logits at the last position.
    # Boost logits of tokens proportionally to their visit-weighted values.
    adjusted_logits = root_logits.clone()
    visit_probs = visit_counts.float() / max(total_visits, 1)
    for a in range(k):
        if visit_counts[a] > 0:
            token_idx = top_actions[a]
            boost = (visit_probs[a] * mean_values[a]).to(
                dtype=adjusted_logits.dtype, device=adjusted_logits.device
            )
            adjusted_logits[:, -1, token_idx] = (
                adjusted_logits[:, -1, token_idx] + boost
            )

    return {
        "logits": adjusted_logits,
        "hidden": root_hidden,
        "mcts_stats": {
            "visit_counts": visit_counts,
            "mean_values": mean_values,
            "root_value": root_value,
        },
    }


class StructuredProjections(nn.Module):
    """K learned projection heads — each emphasizes different features.

    'Choosing the question' — NFT-aligned generation mechanism.
    Instead of random noise perturbation, each projection head creates
    a structurally different view of the input. The system selects
    which view produced the best result.
    """

    def __init__(self, dim: int, k: int = 4) -> None:
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Linear(dim, dim, bias=False) for _ in range(k)
        ])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return K different views of x."""
        return [proj(x) for proj in self.projections]
