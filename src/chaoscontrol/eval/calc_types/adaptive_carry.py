"""``adaptive_carry`` — causal state-carry TTT with online horizon mixing.

This is the submission-day TTT candidate: keep the trunk weights frozen,
thread SSM state across source-ordered docs, and let a token-causal online
expert mixer choose among a small set of recurrent horizon shifts.  It is
gradient-free, but still test-time adaptive: the carried recurrent state and
the online head weights evolve only from already-scored tokens.

The implementation intentionally calls ``model.encode(memory_mode="packet")``
instead of ``model.forward()`` so eval stays on the same packet-clean trunk
lane as training.  If no episodic packet is supplied, packet mode is the
zero-residual no-op path.
"""
from __future__ import annotations

import inspect
import math
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F

from chaoscontrol.eval.ttt_eval import (
    CalcTypeContext,
    CalcTypeResult,
    register_calc_type,
)
from chaoscontrol.eval_stream.delta_mod import DeltaModulator
from chaoscontrol.eval_stream.temporal_heads import (
    _online_exp_weighted_logprob_mixture_with_final_weights,
    _token_nll_from_log_probs,
)


def _as_float_tuple(value: Any, *, name: str) -> tuple[float, ...]:
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        if not parts:
            raise ValueError(f"{name} must not be empty")
        return tuple(float(p) for p in parts)
    if isinstance(value, Sequence):
        if len(value) == 0:
            raise ValueError(f"{name} must not be empty")
        return tuple(float(v) for v in value)
    return (float(value),)


def _packet_encode(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    initial_states: list[torch.Tensor] | None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Encode via packet mode when available, returning hidden + final states."""
    encode = getattr(model, "encode")
    kwargs: dict[str, Any] = {
        "initial_states": initial_states,
        "return_final_states": True,
    }
    if "memory_mode" in inspect.signature(encode).parameters:
        out = encode(input_ids, memory_mode="packet", **kwargs)
    else:
        # Tiny test doubles predating packet mode still exercise the same
        # recurrent-state contract. Production CareStudentLM takes this path
        # only if packet mode was removed, which would fail dedicated tests.
        out = encode(input_ids, **kwargs)
    if isinstance(out, dict):
        hidden = out["hidden"]
        final_states = list(out["final_states"])
    else:
        hidden, final_states = out
    return hidden, [state.detach() for state in final_states]


def _decay_states(
    states: list[torch.Tensor],
    *,
    decay: float,
) -> list[torch.Tensor]:
    if decay == 1.0:
        return [state.detach() for state in states]
    return [state.detach() * decay for state in states]


def _lm_logits(model: torch.nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    final_norm = getattr(model, "final_norm", None)
    if final_norm is not None:
        hidden = final_norm(hidden)
    return model.lm_head(hidden)


@register_calc_type(
    "adaptive_carry",
    requires_source_order=True,
    requires_grad=False,
    description=(
        "Packet-clean state carry with online causal mixing over recurrent "
        "horizon shifts."
    ),
)
def adaptive_carry(ctx: CalcTypeContext) -> CalcTypeResult:
    """Source-ordered, gradient-free TTT over recurrent timescale heads.

    Hyperparameters:
        horizon_shifts: list/tuple/comma string of log_a shifts.  The default
            ``[-0.5, 0.0, 0.5]`` gives fast/base/slow heads.
        online_eta: token-causal exponential-weights learning rate.
        decay: cross-doc decay applied to carried states after each doc.
        online_initial_weights: optional positive initial head weights.

    No future-token leakage: the mixed distribution for token ``t`` is scored
    with weights learned only from tokens ``< t``.  The final online weights
    from doc N seed doc N+1, so adaptation survives across the source stream.
    """
    cfg = ctx.config
    horizon_shifts = _as_float_tuple(
        cfg.get("horizon_shifts", (-0.5, 0.0, 0.5)),
        name="horizon_shifts",
    )
    online_eta = float(cfg.get("online_eta", 1.0))
    decay = float(cfg.get("decay", 1.0))
    initial_weights_cfg = cfg.get("online_initial_weights")
    initial_weights = (
        None
        if initial_weights_cfg is None
        else _as_float_tuple(initial_weights_cfg, name="online_initial_weights")
    )
    if online_eta < 0.0:
        raise ValueError(f"online_eta must be non-negative, got {online_eta}")
    if len(horizon_shifts) < 1:
        raise ValueError("horizon_shifts must contain at least one head")
    if initial_weights is not None and len(initial_weights) != len(horizon_shifts):
        raise ValueError(
            "online_initial_weights length "
            f"{len(initial_weights)} does not match horizon_shifts length "
            f"{len(horizon_shifts)}"
        )

    model = ctx.model
    val_cache = ctx.val_cache
    device = ctx.device

    total_ce_nats = torch.zeros((), dtype=torch.float64)
    total_tokens_scored = 0
    total_raw_bytes = 0
    docs_scored = 0
    states_by_head: dict[float, list[torch.Tensor] | None] = {
        shift: None for shift in horizon_shifts
    }
    online_log_weights: torch.Tensor | None = None
    winner_counts = {str(shift): 0 for shift in horizon_shifts}
    per_head_loss_nats = {str(shift): 0.0 for shift in horizon_shifts}

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for doc in val_cache.iter_docs():
                if doc.token_len < 2:
                    continue
                tokens_np = val_cache.tokens_for_doc(doc)
                input_ids = torch.tensor(
                    tokens_np,
                    dtype=torch.long,
                    device=device,
                ).unsqueeze(0)
                targets = input_ids[:, 1:]

                log_probs: list[torch.Tensor] = []
                next_states_by_head: dict[float, list[torch.Tensor]] = {}
                for shift in horizon_shifts:
                    with DeltaModulator(model, log_a_shift=shift):
                        hidden, final_states = _packet_encode(
                            model,
                            input_ids,
                            initial_states=states_by_head.get(shift),
                        )
                    logits = _lm_logits(model, hidden)
                    head_log_probs = F.log_softmax(logits, dim=-1)
                    log_probs.append(head_log_probs)
                    next_states_by_head[shift] = _decay_states(
                        final_states,
                        decay=decay,
                    )
                    per_head_loss_nats[str(shift)] += float(
                        -head_log_probs[:, :-1]
                        .gather(-1, targets.unsqueeze(-1))
                        .squeeze(-1)
                        .sum()
                        .item()
                    )

                mixed_log_probs, online_log_weights = (
                    _online_exp_weighted_logprob_mixture_with_final_weights(
                        log_probs,
                        targets,
                        eta=online_eta,
                        initial_weights=initial_weights,
                        initial_log_weights=online_log_weights,
                    )
                )
                token_nll = _token_nll_from_log_probs(mixed_log_probs, targets)
                total_ce_nats += token_nll.sum().detach().to(
                    device="cpu",
                    dtype=torch.float64,
                )
                total_tokens_scored += int(targets.numel())
                total_raw_bytes += int(doc.raw_bytes)
                docs_scored += 1

                per_head_token_nll = torch.stack(
                    [_token_nll_from_log_probs(logp, targets) for logp in log_probs],
                    dim=0,
                )
                winners = per_head_token_nll.argmin(dim=0)
                for idx, shift in enumerate(horizon_shifts):
                    winner_counts[str(shift)] += int((winners == idx).sum().item())
                states_by_head = next_states_by_head
    finally:
        if was_training:
            model.train()

    ce_nats_f = float(total_ce_nats.item())
    bpb = 0.0 if total_raw_bytes <= 0 else ce_nats_f / total_raw_bytes / math.log(2.0)
    loss = ce_nats_f / max(total_tokens_scored, 1)
    final_weights: list[float] | None = None
    if online_log_weights is not None:
        final_weights = [
            float(v)
            for v in online_log_weights.exp().mean(dim=0).detach().cpu().tolist()
        ]

    return CalcTypeResult(
        bpb=bpb,
        loss=loss,
        docs_scored=docs_scored,
        tokens_scored=total_tokens_scored,
        raw_bytes=total_raw_bytes,
        hyperparams={
            "horizon_shifts": list(horizon_shifts),
            "online_eta": online_eta,
            "decay": decay,
            "online_initial_weights": (
                None if initial_weights is None else list(initial_weights)
            ),
        },
        extra={
            "winner_counts_by_shift": winner_counts,
            "per_head_loss_nats": per_head_loss_nats,
            "online_final_weights": final_weights,
        },
    )
