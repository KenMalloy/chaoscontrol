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


def _packet_encode_support(
    model: torch.nn.Module,
) -> tuple[bool, bool]:
    encode = getattr(model, "encode")
    params = inspect.signature(encode).parameters
    accepts_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    supports_memory_mode = "memory_mode" in params or accepts_kwargs
    supports_packet_args = accepts_kwargs or (
        "episodic_residual" in params and "episodic_gate" in params
    )
    return supports_memory_mode, supports_packet_args


def _packet_encode(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    initial_states: list[torch.Tensor] | None,
    episodic_residual: torch.Tensor | None = None,
    episodic_gate: torch.Tensor | None = None,
    packet_support: tuple[bool, bool] | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Encode via packet mode when available, returning hidden + final states."""
    encode = getattr(model, "encode")
    kwargs: dict[str, Any] = {
        "initial_states": initial_states,
        "return_final_states": True,
    }
    supports_memory_mode, supports_packet_args = (
        _packet_encode_support(model)
        if packet_support is None
        else packet_support
    )
    if supports_memory_mode:
        if (
            episodic_residual is not None
            and episodic_gate is not None
            and supports_packet_args
        ):
            kwargs["episodic_residual"] = episodic_residual
            kwargs["episodic_gate"] = episodic_gate
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


def _online_mix_equal_length(
    log_probs: list[torch.Tensor],
    targets: torch.Tensor,
    *,
    eta: float,
    initial_weights: tuple[float, ...] | None,
    initial_log_weights: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Causal exponential-weights mixture for externally shifted chunks.

    ``adaptive_carry`` evaluates chunks as ``input=x[t-1:u-1]`` and
    ``target=x[t:u]`` so it can carry recurrent state exactly without
    re-processing the overlap token. The shared helper in
    ``temporal_heads`` uses the conventional in-tensor shift and therefore
    expects ``targets.shape[1] == log_probs.shape[1] - 1``; this local variant
    keeps the same score-before-update rule for equal-length target slices.
    """
    if not log_probs:
        raise ValueError("log_probs requires at least one tensor")
    if len(log_probs) == 1:
        return log_probs[0], None
    shape = log_probs[0].shape
    if len(shape) != 3:
        raise ValueError("log_probs tensors must have shape [batch, seq, vocab]")
    if any(logp.shape != shape for logp in log_probs):
        raise ValueError("all log_probs tensors must have the same shape")
    batch, seq_len, _vocab = shape
    if targets.shape != (batch, seq_len):
        raise ValueError(
            f"targets shape {tuple(targets.shape)} does not match "
            f"expected {(batch, seq_len)}"
        )
    if eta < 0.0:
        raise ValueError("eta must be non-negative")

    if initial_log_weights is not None:
        log_weights = initial_log_weights.to(
            device=log_probs[0].device,
            dtype=log_probs[0].dtype,
        )
        if log_weights.shape != (batch, len(log_probs)):
            raise ValueError(
                f"initial_log_weights shape {tuple(log_weights.shape)} "
                f"does not match expected {(batch, len(log_probs))}"
            )
    else:
        if initial_weights is None:
            initial_weights = tuple(1.0 for _ in log_probs)
        w = torch.tensor(
            initial_weights,
            device=log_probs[0].device,
            dtype=log_probs[0].dtype,
        )
        if len(w) != len(log_probs):
            raise ValueError(
                f"initial_weights length {len(w)} does not match "
                f"log_probs length {len(log_probs)}"
            )
        if torch.any(w <= 0):
            raise ValueError("all initial online weights must be positive")
        log_weights = (w / w.sum()).log().unsqueeze(0).expand(batch, -1).clone()

    stacked = torch.stack(log_probs, dim=1)  # (B, H, T, V)
    head_count = len(log_probs)
    mixed_positions: list[torch.Tensor] = []
    for pos in range(seq_len):
        mixed_positions.append(
            torch.logsumexp(
                stacked[:, :, pos, :] + log_weights.unsqueeze(-1),
                dim=1,
            )
        )
        token_log_probs = torch.stack(
            [
                stacked[:, head_idx, pos, :]
                .gather(-1, targets[:, pos].unsqueeze(-1))
                .squeeze(-1)
                for head_idx in range(head_count)
            ],
            dim=1,
        )
        log_weights = torch.log_softmax(
            log_weights + float(eta) * token_log_probs,
            dim=1,
        )
    return torch.stack(mixed_positions, dim=1), log_weights


def _token_nll_equal_length(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    if targets.shape != log_probs.shape[:2]:
        raise ValueError(
            f"targets shape {tuple(targets.shape)} does not match "
            f"log_probs prefix {tuple(log_probs.shape[:2])}"
        )
    return -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def _outer_slot_count(model: torch.nn.Module) -> int:
    outer = getattr(model, "outer_model", None)
    if outer is None:
        return 0
    table = getattr(outer, "table", None)
    if table is not None:
        try:
            return int(len(table))
        except Exception:
            pass
    slots = getattr(outer, "_slots", None)
    if slots is not None:
        try:
            return int(len(slots))
        except Exception:
            pass
    return 0


def _episodic_packet_from_prefix(
    model: torch.nn.Module,
    *,
    states: list[torch.Tensor] | None,
    batch_size: int,
    gate_value: float,
) -> tuple[torch.Tensor | None, torch.Tensor | None, bool]:
    """Read a compact residual from already-committed episodic memory.

    The cue is the recurrent state at the chunk boundary, i.e. strict-prefix
    information. No token from the chunk being scored participates in the
    cache query, so the packet is legal for every target in that chunk.
    """
    if gate_value <= 0.0 or _outer_slot_count(model) <= 0:
        return None, None, False
    outer = getattr(model, "outer_model", None)
    read = getattr(outer, "read", None)
    if read is None or not callable(read):
        return None, None, False
    cue = None
    if states:
        cue = states[-1].detach()
        if cue.dim() > 2:
            cue = cue.reshape(cue.shape[0], -1)
    residual = read(int(batch_size), cue=cue)
    if not isinstance(residual, torch.Tensor):
        return None, None, False
    gate = torch.full(
        (int(batch_size),),
        float(gate_value),
        device=residual.device,
        dtype=residual.dtype,
    )
    return residual.detach(), gate.detach(), True


def _append_online_episodic_memory(
    model: torch.nn.Module,
    hidden: torch.Tensor,
    *,
    score: torch.Tensor | None,
    max_tokens: int,
) -> bool:
    append = getattr(model, "append_memory_from_hidden", None)
    if append is None or not callable(append) or int(max_tokens) <= 0:
        return False
    kwargs: dict[str, Any] = {"max_tokens": int(max_tokens)}
    if score is not None:
        kwargs["score"] = score.detach()
    return bool(append(hidden.detach(), **kwargs))


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
    episodic_chunk_tokens = int(cfg.get("online_episodic_chunk_tokens", 256))
    episodic_write_tokens = int(
        cfg.get("online_episodic_write_tokens_per_chunk", 16)
    )
    episodic_gate = float(cfg.get("online_episodic_gate", 1.0))
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
    if episodic_chunk_tokens < 1:
        raise ValueError("online_episodic_chunk_tokens must be >= 1")
    if episodic_write_tokens < 0:
        raise ValueError("online_episodic_write_tokens_per_chunk must be >= 0")
    if episodic_gate < 0.0:
        raise ValueError("online_episodic_gate must be non-negative")
    if initial_weights is not None and len(initial_weights) != len(horizon_shifts):
        raise ValueError(
            "online_initial_weights length "
            f"{len(initial_weights)} does not match horizon_shifts length "
            f"{len(horizon_shifts)}"
        )

    model = ctx.model
    val_cache = ctx.val_cache
    device = ctx.device
    packet_support = _packet_encode_support(model)

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
    episodic_reads = 0
    episodic_writes = 0
    episodic_chunks = 0

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
                score_pos = 1
                last_base_hidden: torch.Tensor | None = None
                while score_pos < int(input_ids.shape[1]):
                    target_end = min(
                        int(input_ids.shape[1]),
                        score_pos + int(episodic_chunk_tokens),
                    )
                    chunk_inputs = input_ids[:, score_pos - 1 : target_end - 1]
                    targets = input_ids[:, score_pos:target_end]
                    if chunk_inputs.numel() == 0 or targets.numel() == 0:
                        break

                    log_probs: list[torch.Tensor] = []
                    next_states_by_head: dict[float, list[torch.Tensor]] = {}
                    base_hidden_for_write: torch.Tensor | None = None
                    base_shift = 0.0 if 0.0 in horizon_shifts else horizon_shifts[0]
                    for shift in horizon_shifts:
                        residual, gate, read_hit = _episodic_packet_from_prefix(
                            model,
                            states=states_by_head.get(shift),
                            batch_size=int(chunk_inputs.shape[0]),
                            gate_value=episodic_gate,
                        )
                        if read_hit:
                            episodic_reads += 1
                        with DeltaModulator(model, log_a_shift=shift):
                            hidden, final_states = _packet_encode(
                                model,
                                chunk_inputs,
                                initial_states=states_by_head.get(shift),
                                episodic_residual=residual,
                                episodic_gate=gate,
                                packet_support=packet_support,
                            )
                        logits = _lm_logits(model, hidden)
                        head_log_probs = F.log_softmax(logits, dim=-1)
                        log_probs.append(head_log_probs)
                        next_states_by_head[shift] = final_states
                        head_nll = _token_nll_equal_length(head_log_probs, targets)
                        per_head_loss_nats[str(shift)] += float(
                            head_nll.sum().item()
                        )
                        if shift == base_shift:
                            base_hidden_for_write = hidden

                    mixed_log_probs, online_log_weights = _online_mix_equal_length(
                        log_probs,
                        targets,
                        eta=online_eta,
                        initial_weights=initial_weights,
                        initial_log_weights=online_log_weights,
                    )
                    token_nll = _token_nll_equal_length(mixed_log_probs, targets)
                    total_ce_nats += token_nll.sum().detach().to(
                        device="cpu",
                        dtype=torch.float64,
                    )
                    total_tokens_scored += int(targets.numel())

                    per_head_token_nll = torch.stack(
                        [_token_nll_equal_length(logp, targets) for logp in log_probs],
                        dim=0,
                    )
                    winners = per_head_token_nll.argmin(dim=0)
                    for idx, shift in enumerate(horizon_shifts):
                        winner_counts[str(shift)] += int((winners == idx).sum().item())

                    states_by_head = {
                        shift: _decay_states(states, decay=decay)
                        for shift, states in next_states_by_head.items()
                    }
                    if base_hidden_for_write is not None:
                        wrote = _append_online_episodic_memory(
                            model,
                            base_hidden_for_write,
                            score=token_nll,
                            max_tokens=episodic_write_tokens,
                        )
                        episodic_writes += int(wrote)
                        last_base_hidden = base_hidden_for_write
                    episodic_chunks += 1
                    score_pos = target_end

                # The final token has been scored as a target but has not
                # been fed as an input by the shifted chunks above. Commit it
                # to recurrent state after its score is fixed so cross-doc
                # carry/cache state represents the full observed stream.
                tail = input_ids[:, -1:]
                next_states_by_head = {}
                base_shift = 0.0 if 0.0 in horizon_shifts else horizon_shifts[0]
                for shift in horizon_shifts:
                    residual, gate, read_hit = _episodic_packet_from_prefix(
                        model,
                        states=states_by_head.get(shift),
                        batch_size=int(tail.shape[0]),
                        gate_value=episodic_gate,
                    )
                    if read_hit:
                        episodic_reads += 1
                    with DeltaModulator(model, log_a_shift=shift):
                        hidden, final_states = _packet_encode(
                            model,
                            tail,
                            initial_states=states_by_head.get(shift),
                            episodic_residual=residual,
                            episodic_gate=gate,
                            packet_support=packet_support,
                        )
                    next_states_by_head[shift] = _decay_states(
                        final_states,
                        decay=decay,
                    )
                    if shift == base_shift:
                        last_base_hidden = hidden
                states_by_head = next_states_by_head
                if last_base_hidden is not None:
                    episodic_writes += int(
                        _append_online_episodic_memory(
                            model,
                            last_base_hidden,
                            score=None,
                            max_tokens=1 if episodic_write_tokens > 0 else 0,
                        )
                    )
                total_raw_bytes += int(doc.raw_bytes)
                docs_scored += 1
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
            "online_episodic_chunk_tokens": episodic_chunk_tokens,
            "online_episodic_write_tokens_per_chunk": episodic_write_tokens,
            "online_episodic_gate": episodic_gate,
            "online_initial_weights": (
                None if initial_weights is None else list(initial_weights)
            ),
        },
        extra={
            "winner_counts_by_shift": winner_counts,
            "per_head_loss_nats": per_head_loss_nats,
            "online_final_weights": final_weights,
            "online_episodic_reads": episodic_reads,
            "online_episodic_writes": episodic_writes,
            "online_episodic_chunks": episodic_chunks,
            "online_episodic_slots_final": _outer_slot_count(model),
        },
    )
