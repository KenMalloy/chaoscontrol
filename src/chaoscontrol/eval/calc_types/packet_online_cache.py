"""``packet_online_cache`` — submission-facing eval calc_type.

Episodic memory is part of the model. Each chunk:

  1. Read a prefix-safe cue from carried recurrent state.
  2. Score the chunk through the packet-clean encode lane with the cue as
     ``episodic_residual`` + ``episodic_gate``.
  3. Commit the just-scored hidden states (and their token NLLs) to the
     episodic cache so future chunks can read them.

No horizon mixer, no DeltaModulator, no per-doc reset of cache state.  The
cache arrives at eval-time from the training checkpoint (via
``serialize_artifact``) and grows monotonically over the source-ordered
stream.  A fresh-cache variant ("empty") is selected by setting
``seeded=False`` in the calc_type config; the cache is cleared on entry.
"""
from __future__ import annotations

import inspect
import math
from typing import Any

import torch
import torch.nn.functional as F

from chaoscontrol.eval.ttt_eval import (
    CalcTypeContext,
    CalcTypeResult,
    register_calc_type,
)


def _packet_encode_support(model: torch.nn.Module) -> tuple[bool, bool]:
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
    episodic_residual: torch.Tensor | None,
    episodic_gate: torch.Tensor | None,
    packet_support: tuple[bool, bool],
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    encode = getattr(model, "encode")
    kwargs: dict[str, Any] = {
        "initial_states": initial_states,
        "return_final_states": True,
    }
    supports_memory_mode, supports_packet_args = packet_support
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


def _clear_outer_cache(model: torch.nn.Module) -> None:
    """Drop every active slot before eval. Used by the empty/no-seed variant."""
    outer = getattr(model, "outer_model", None)
    if outer is None:
        return
    table = getattr(outer, "table", None)
    if table is not None:
        active = list(table.active_slot_ids())
        if active:
            table.retire_many(active, reason="eval_reset")
    for attr in ("_slots", "_survival", "_slot_buckets", "_slot_event_ids"):
        seq = getattr(outer, attr, None)
        if isinstance(seq, list):
            seq.clear()


def _episodic_packet_from_prefix(
    model: torch.nn.Module,
    *,
    states: list[torch.Tensor] | None,
    batch_size: int,
    gate_value: float,
) -> tuple[torch.Tensor | None, torch.Tensor | None, bool]:
    """Read a residual from already-committed memory.

    The cue is the recurrent state at the chunk boundary: strict-prefix
    information only.  No token from the chunk being scored participates in
    the cue, so the residual is legal for every target in that chunk.
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
) -> int:
    """Commit ``max_tokens`` highest-scoring hidden states to the cache.

    Returns the number of slot-append calls (0 or 1 in practice).  Score is the
    per-token NLL produced by THIS chunk's scoring pass; appending happens AFTER
    the chunk's loss has been accumulated, so the cache reflects only
    already-scored evidence.
    """
    append = getattr(model, "append_memory_from_hidden", None)
    if append is None or not callable(append) or int(max_tokens) <= 0:
        return 0
    kwargs: dict[str, Any] = {"max_tokens": int(max_tokens)}
    if score is not None:
        kwargs["score"] = score.detach()
    return int(bool(append(hidden.detach(), **kwargs)))


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


@register_calc_type(
    "packet_online_cache",
    requires_source_order=True,
    requires_grad=False,
    description=(
        "Submission path: prefix-safe cue + packet-lane score + post-score "
        "cache write.  Episodic memory is part of the model."
    ),
)
def packet_online_cache(ctx: CalcTypeContext) -> CalcTypeResult:
    """Packet-lane online episodic eval.

    Hyperparameters:
        chunk_tokens: chunk size for the read-score-write loop (default 256).
        write_tokens_per_chunk: top-K tokens (by NLL) committed to cache
            after each chunk's score is finalized (default 16).
        gate_value: residual gate magnitude (default 1.0).  ``0.0`` disables
            the residual entirely.
        decay: cross-doc decay applied to carried recurrent state (default 1.0).
        seeded: if False, clear ``model.outer_model`` before eval to isolate
            online accumulation from training-seeded cache (default True).
        max_docs: optional source-order doc cap for smoke tests. Omitted or
            <=0 means score the full cache.
    """
    cfg = ctx.config
    chunk_tokens = int(cfg.get("chunk_tokens", 256))
    write_tokens_per_chunk = int(cfg.get("write_tokens_per_chunk", 16))
    gate_value = float(cfg.get("gate_value", 1.0))
    decay = float(cfg.get("decay", 1.0))
    seeded = bool(cfg.get("seeded", True))
    max_docs = int(cfg.get("max_docs", 0) or 0)
    if chunk_tokens < 1:
        raise ValueError("chunk_tokens must be >= 1")
    if write_tokens_per_chunk < 0:
        raise ValueError("write_tokens_per_chunk must be >= 0")
    if gate_value < 0.0:
        raise ValueError("gate_value must be non-negative")

    model = ctx.model
    val_cache = ctx.val_cache
    device = ctx.device
    packet_support = _packet_encode_support(model)

    if not seeded:
        _clear_outer_cache(model)
    initial_slot_count = _outer_slot_count(model)

    total_ce_nats = torch.zeros((), dtype=torch.float64)
    total_tokens_scored = 0
    total_raw_bytes = 0
    docs_scored = 0
    states: list[torch.Tensor] | None = None
    episodic_reads = 0
    episodic_writes = 0
    chunks_scored = 0

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for doc in val_cache.iter_docs():
                if max_docs > 0 and docs_scored >= max_docs:
                    break
                if doc.token_len < 2:
                    continue
                tokens_np = val_cache.tokens_for_doc(doc)
                input_ids = torch.tensor(
                    tokens_np,
                    dtype=torch.long,
                    device=device,
                ).unsqueeze(0)
                score_pos = 1
                while score_pos < int(input_ids.shape[1]):
                    target_end = min(
                        int(input_ids.shape[1]),
                        score_pos + chunk_tokens,
                    )
                    chunk_inputs = input_ids[:, score_pos - 1 : target_end - 1]
                    targets = input_ids[:, score_pos:target_end]
                    if chunk_inputs.numel() == 0 or targets.numel() == 0:
                        break

                    slot_count_at_score = _outer_slot_count(model)
                    residual, gate, read_hit = _episodic_packet_from_prefix(
                        model,
                        states=states,
                        batch_size=int(chunk_inputs.shape[0]),
                        gate_value=gate_value,
                    )
                    if read_hit:
                        episodic_reads += 1
                    hidden, final_states = _packet_encode(
                        model,
                        chunk_inputs,
                        initial_states=states,
                        episodic_residual=residual,
                        episodic_gate=gate,
                        packet_support=packet_support,
                    )
                    log_probs = F.log_softmax(_lm_logits(model, hidden), dim=-1)
                    token_nll = _token_nll_equal_length(log_probs, targets)
                    if _outer_slot_count(model) != slot_count_at_score:
                        raise RuntimeError(
                            "score-before-write violated: cache grew from "
                            f"{slot_count_at_score} to {_outer_slot_count(model)} "
                            "between cue read and score accumulation"
                        )
                    total_ce_nats += token_nll.sum().detach().to(
                        device="cpu",
                        dtype=torch.float64,
                    )
                    total_tokens_scored += int(targets.numel())

                    states = _decay_states(final_states, decay=decay)
                    episodic_writes += _append_online_episodic_memory(
                        model,
                        hidden,
                        score=token_nll,
                        max_tokens=write_tokens_per_chunk,
                    )
                    chunks_scored += 1
                    score_pos = target_end

                # Commit the trailing token to recurrent state so cross-doc
                # carry reflects the full observed stream.  No new score is
                # produced here (the tail target was already scored above).
                tail = input_ids[:, -1:]
                residual, gate, read_hit = _episodic_packet_from_prefix(
                    model,
                    states=states,
                    batch_size=int(tail.shape[0]),
                    gate_value=gate_value,
                )
                if read_hit:
                    episodic_reads += 1
                _, final_states = _packet_encode(
                    model,
                    tail,
                    initial_states=states,
                    episodic_residual=residual,
                    episodic_gate=gate,
                    packet_support=packet_support,
                )
                states = _decay_states(final_states, decay=decay)

                total_raw_bytes += int(doc.raw_bytes)
                docs_scored += 1
    finally:
        if was_training:
            model.train()

    ce_nats_f = float(total_ce_nats.item())
    bpb = (
        0.0
        if total_raw_bytes <= 0
        else ce_nats_f / total_raw_bytes / math.log(2.0)
    )
    loss = ce_nats_f / max(total_tokens_scored, 1)

    return CalcTypeResult(
        bpb=bpb,
        loss=loss,
        docs_scored=docs_scored,
        tokens_scored=total_tokens_scored,
        raw_bytes=total_raw_bytes,
        hyperparams={
            "chunk_tokens": chunk_tokens,
            "write_tokens_per_chunk": write_tokens_per_chunk,
            "gate_value": gate_value,
            "decay": decay,
            "seeded": seeded,
            "max_docs": max_docs,
        },
        extra={
            "episodic_reads": episodic_reads,
            "episodic_writes": episodic_writes,
            "chunks_scored": chunks_scored,
            "slot_count_initial": initial_slot_count,
            "slot_count_final": _outer_slot_count(model),
        },
    )
