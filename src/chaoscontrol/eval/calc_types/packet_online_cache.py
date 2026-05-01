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
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from chaoscontrol.eval.ttt_eval import (
    CalcTypeContext,
    CalcTypeResult,
    register_calc_type,
)
from chaoscontrol.eval_stream.val_cache import CachedDoc, ValCache


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


def _outer_slot_matrix_and_indices(
    model: torch.nn.Module,
) -> tuple[torch.Tensor | None, list[int]]:
    outer = getattr(model, "outer_model", None)
    if outer is None:
        return None, []
    table = getattr(outer, "table", None)
    if table is not None:
        visible = getattr(table, "visible_indices", None)
        slot_matrix = getattr(table, "slot_matrix", None)
        if callable(visible) and callable(slot_matrix):
            indices = [int(i) for i in visible()]
            if not indices:
                return None, []
            return slot_matrix(indices), indices
    slots = getattr(outer, "_slots", None)
    if not slots:
        return None, []
    matrix = torch.cat([slot.detach().reshape(1, -1) for slot in slots], dim=0)
    return matrix, list(range(int(matrix.shape[0])))


def _outer_slot_survival(
    model: torch.nn.Module,
    *,
    indices: list[int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    outer = getattr(model, "outer_model", None)
    survival = getattr(outer, "_survival", None)
    if isinstance(survival, list) and survival:
        vals = [
            float(survival[i]) if 0 <= int(i) < len(survival) else 1.0
            for i in indices
        ]
        return torch.tensor(vals, device=device, dtype=dtype)
    table = getattr(outer, "table", None)
    table_survival = getattr(table, "_survival", None)
    if isinstance(table_survival, list) and table_survival:
        vals = [
            float(table_survival[i]) if 0 <= int(i) < len(table_survival) else 1.0
            for i in indices
        ]
        return torch.tensor(vals, device=device, dtype=dtype)
    return torch.ones(len(indices), device=device, dtype=dtype)


def _controller_slot_mask_from_prefix(
    model: torch.nn.Module,
    *,
    cue: torch.Tensor | None,
    batch_size: int,
    topk: int,
    score_mode: str,
) -> tuple[torch.Tensor | None, int]:
    """Select a bounded slot set before packet read.

    This is the eval-side controller toggle: a strict-prefix cue, when
    available, ranks memory slots before ``outer.read`` forms the residual.
    The direct packet path still owns the final decode; the controller only
    narrows *which* slots are allowed to participate.
    """
    outer = getattr(model, "outer_model", None)
    if outer is None or int(topk) <= 0:
        return None, 0
    slot_matrix, indices = _outer_slot_matrix_and_indices(model)
    if slot_matrix is None or not indices:
        return None, 0
    slot_device = slot_matrix.device
    slot_dtype = slot_matrix.dtype
    k_eff = min(int(topk), len(indices))
    mode = str(score_mode).strip().lower()
    valid_modes = {
        "cosine",
        "cosine_survival",
        "cosine_utility_weighted",
        "dot",
        "dot_survival",
    }
    if mode not in valid_modes:
        raise ValueError(
            f"controller_score_mode must be one of {sorted(valid_modes)}, "
            f"got {score_mode!r}"
        )
    survival = _outer_slot_survival(
        model,
        indices=indices,
        device=slot_device,
        dtype=slot_dtype,
    )
    if cue is None:
        scores = survival.unsqueeze(0).expand(int(batch_size), -1)
    else:
        cue = cue.detach()
        if cue.dim() > 2:
            cue = cue.reshape(cue.shape[0], -1)
        if cue.shape[0] == 1 and int(batch_size) != 1:
            cue = cue.expand(int(batch_size), -1)
        elif cue.shape[0] != int(batch_size):
            cue = cue[:1].expand(int(batch_size), -1)
        cue_proj = getattr(outer, "cue_proj", None)
        if cue_proj is not None:
            cue_outer = cue_proj(
                cue.to(device=slot_device, dtype=cue_proj.weight.dtype)
            ).to(dtype=slot_dtype)
        else:
            cue_outer = cue.to(device=slot_device, dtype=slot_dtype)
            if cue_outer.shape[-1] != slot_matrix.shape[-1]:
                cue_outer = cue_outer[..., : slot_matrix.shape[-1]]
                if cue_outer.shape[-1] < slot_matrix.shape[-1]:
                    cue_outer = torch.nn.functional.pad(
                        cue_outer,
                        (0, slot_matrix.shape[-1] - cue_outer.shape[-1]),
                    )
        if mode in {"cosine", "cosine_survival", "cosine_utility_weighted"}:
            q = cue_outer / (cue_outer.norm(dim=-1, keepdim=True) + 1e-8)
            keys = slot_matrix / (slot_matrix.norm(dim=-1, keepdim=True) + 1e-8)
            scores = q @ keys.T
        else:
            scores = cue_outer @ slot_matrix.T
        if mode in {"cosine_survival", "dot_survival", "cosine_utility_weighted"}:
            scores = scores * survival.unsqueeze(0)
    top = torch.topk(scores, k=k_eff, dim=-1, largest=True, sorted=False).indices
    physical = torch.tensor(indices, device=slot_device, dtype=torch.long)[top]
    mask_width = max(indices) + 1
    slot_mask = torch.zeros(
        int(batch_size),
        int(mask_width),
        device=slot_device,
        dtype=torch.bool,
    )
    slot_mask.scatter_(1, physical, True)
    return slot_mask, int(physical.numel())


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
    controller_read_enabled: bool = False,
    controller_topk_k: int = 16,
    controller_score_mode: str = "cosine_survival",
) -> tuple[torch.Tensor | None, torch.Tensor | None, bool, int]:
    """Read a residual from already-committed memory.

    The cue is the recurrent state at the chunk boundary: strict-prefix
    information only.  No token from the chunk being scored participates in
    the cue, so the residual is legal for every target in that chunk.
    """
    if gate_value <= 0.0 or _outer_slot_count(model) <= 0:
        return None, None, False, 0
    outer = getattr(model, "outer_model", None)
    read = getattr(outer, "read", None)
    if read is None or not callable(read):
        return None, None, False, 0
    cue = None
    if states:
        cue = states[-1].detach()
        if cue.dim() > 2:
            cue = cue.reshape(cue.shape[0], -1)
    slot_mask = None
    controller_selected = 0
    if controller_read_enabled:
        slot_mask, controller_selected = _controller_slot_mask_from_prefix(
            model,
            cue=cue,
            batch_size=int(batch_size),
            topk=int(controller_topk_k),
            score_mode=str(controller_score_mode),
        )
    read_kwargs: dict[str, Any] = {"cue": cue}
    if slot_mask is not None:
        read_kwargs["slot_mask"] = slot_mask
    residual = read(int(batch_size), **read_kwargs)
    if not isinstance(residual, torch.Tensor):
        return None, None, False, controller_selected
    gate = torch.full(
        (int(batch_size),),
        float(gate_value),
        device=residual.device,
        dtype=residual.dtype,
    )
    return residual.detach(), gate.detach(), True, controller_selected


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


def _iter_score_docs(
    val_cache: ValCache,
    *,
    max_docs: int,
) -> Iterable[CachedDoc]:
    docs_scored = 0
    for doc in val_cache.iter_docs():
        if max_docs > 0 and docs_scored >= max_docs:
            break
        if doc.token_len < 2:
            continue
        yield doc
        docs_scored += 1


def _iter_doc_microbatches(
    val_cache: ValCache,
    *,
    max_docs: int,
    batch_docs: int,
    batch_token_budget: int,
) -> Iterable[list[CachedDoc]]:
    """Yield source-ordered doc groups for score-first batched eval.

    The budget is measured in padded input tokens (B * max_len), matching the
    forward/LM-head work the batch will actually materialize.
    """
    batch: list[CachedDoc] = []
    max_input_len = 0
    for doc in _iter_score_docs(val_cache, max_docs=max_docs):
        input_len = int(doc.token_len) - 1
        next_count = len(batch) + 1
        next_max = max(max_input_len, input_len)
        over_docs = batch_docs > 0 and next_count > batch_docs
        over_tokens = (
            batch_token_budget > 0
            and bool(batch)
            and next_count * next_max > batch_token_budget
        )
        if over_docs or over_tokens:
            yield batch
            batch = []
            max_input_len = 0
        batch.append(doc)
        max_input_len = max(max_input_len, input_len)
    if batch:
        yield batch


def _score_doc_microbatch(
    model: torch.nn.Module,
    val_cache: ValCache,
    docs: list[CachedDoc],
    *,
    device: torch.device,
    packet_support: tuple[bool, bool],
    gate_value: float,
    write_tokens_per_chunk: int,
    controller_read_enabled: bool,
    controller_topk_k: int,
    controller_score_mode: str,
) -> tuple[float, int, int, int, int, int, int]:
    """Score a source-ordered microbatch, then append its evidence.

    This is prequential at the microbatch boundary: no hidden state from the
    microbatch is written until every token in the microbatch has already had
    its loss accumulated.  The model may ignore some available strict-prefix
    information within the microbatch, but it never uses future information.
    """
    if not docs:
        return 0.0, 0, 0, 0, 0, 0, 0
    token_arrays: list[np.ndarray] = [val_cache.tokens_for_doc(doc) for doc in docs]
    lengths = [int(arr.shape[0]) - 1 for arr in token_arrays]
    max_len = max(lengths)
    batch = len(docs)
    input_ids = torch.zeros((batch, max_len), dtype=torch.long, device=device)
    targets = torch.zeros((batch, max_len), dtype=torch.long, device=device)
    mask = torch.zeros((batch, max_len), dtype=torch.bool, device=device)
    for row, arr in enumerate(token_arrays):
        length = lengths[row]
        ids = torch.tensor(arr, dtype=torch.long, device=device)
        input_ids[row, :length] = ids[:-1]
        targets[row, :length] = ids[1:]
        mask[row, :length] = True

    slot_count_at_score = _outer_slot_count(model)
    residual, gate, read_hit, controller_selected = _episodic_packet_from_prefix(
        model,
        states=None,
        batch_size=batch,
        gate_value=gate_value,
        controller_read_enabled=controller_read_enabled,
        controller_topk_k=controller_topk_k,
        controller_score_mode=controller_score_mode,
    )
    hidden, _final_states = _packet_encode(
        model,
        input_ids,
        initial_states=None,
        episodic_residual=residual,
        episodic_gate=gate,
        packet_support=packet_support,
    )
    logits = _lm_logits(model, hidden)
    token_nll = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    ).reshape(batch, max_len)
    if _outer_slot_count(model) != slot_count_at_score:
        raise RuntimeError(
            "score-before-write violated: cache grew from "
            f"{slot_count_at_score} to {_outer_slot_count(model)} "
            "between batched cue read and score accumulation"
        )

    valid_nll = token_nll.masked_select(mask)
    ce_nats = float(valid_nll.detach().to(device="cpu", dtype=torch.float64).sum().item())
    tokens_scored = int(mask.sum().item())
    raw_bytes = int(sum(int(doc.raw_bytes) for doc in docs))
    scores_for_write = token_nll.masked_fill(~mask, float("-inf"))
    write_limit = min(max(0, int(write_tokens_per_chunk)) * batch, tokens_scored)
    writes = _append_online_episodic_memory(
        model,
        hidden,
        score=scores_for_write,
        max_tokens=write_limit,
    )
    return (
        ce_nats,
        tokens_scored,
        raw_bytes,
        int(bool(read_hit)),
        writes,
        1,
        int(controller_selected),
    )


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
        batch_docs: optional microbatch doc count. ``1`` keeps the exact
            chunk-by-chunk online path; values >1 score a source-ordered
            microbatch before committing its evidence.
        batch_token_budget: padded-token budget for batched eval. ``0`` means
            only ``batch_docs`` limits the microbatch.
        controller_read_enabled: when True, a strict-prefix controller selector
            picks a bounded top-K slot mask before packet residual formation.
        controller_topk_k: slot budget for the controller selector.
        controller_score_mode: ``cosine_survival`` (default), ``cosine``, or
            ``dot_survival``.
    """
    cfg = ctx.config
    chunk_tokens = int(cfg.get("chunk_tokens", 256))
    write_tokens_per_chunk = int(cfg.get("write_tokens_per_chunk", 16))
    gate_value = float(cfg.get("gate_value", 1.0))
    decay = float(cfg.get("decay", 1.0))
    seeded = bool(cfg.get("seeded", True))
    max_docs = int(cfg.get("max_docs", 0) or 0)
    batch_docs = int(cfg.get("batch_docs", 1))
    batch_token_budget = int(cfg.get("batch_token_budget", 0) or 0)
    controller_read_enabled = bool(cfg.get("controller_read_enabled", False))
    controller_topk_k = int(cfg.get("controller_topk_k", 16))
    controller_score_mode = str(cfg.get("controller_score_mode", "cosine_survival"))
    if chunk_tokens < 1:
        raise ValueError("chunk_tokens must be >= 1")
    if write_tokens_per_chunk < 0:
        raise ValueError("write_tokens_per_chunk must be >= 0")
    if gate_value < 0.0:
        raise ValueError("gate_value must be non-negative")
    if batch_docs < 1:
        raise ValueError("batch_docs must be >= 1")
    if batch_token_budget < 0:
        raise ValueError("batch_token_budget must be >= 0")
    if controller_topk_k < 1:
        raise ValueError("controller_topk_k must be >= 1")
    valid_controller_modes = {
        "cosine",
        "cosine_survival",
        "cosine_utility_weighted",
        "dot",
        "dot_survival",
    }
    if (
        controller_read_enabled
        and controller_score_mode.strip().lower() not in valid_controller_modes
    ):
        raise ValueError(
            "controller_score_mode must be one of "
            f"{sorted(valid_controller_modes)}, got {controller_score_mode!r}"
        )

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
    controller_reads = 0
    controller_selected_slots = 0

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            if batch_docs > 1 or batch_token_budget > 0:
                for docs in _iter_doc_microbatches(
                    val_cache,
                    max_docs=max_docs,
                    batch_docs=batch_docs,
                    batch_token_budget=batch_token_budget,
                ):
                    (
                        ce_nats,
                        tokens_scored,
                        raw_bytes,
                        reads,
                        writes,
                        chunks,
                        selected_slots,
                    ) = _score_doc_microbatch(
                        model,
                        val_cache,
                        docs,
                        device=device,
                        packet_support=packet_support,
                        gate_value=gate_value,
                        write_tokens_per_chunk=write_tokens_per_chunk,
                        controller_read_enabled=controller_read_enabled,
                        controller_topk_k=controller_topk_k,
                        controller_score_mode=controller_score_mode,
                    )
                    total_ce_nats += torch.tensor(ce_nats, dtype=torch.float64)
                    total_tokens_scored += int(tokens_scored)
                    total_raw_bytes += int(raw_bytes)
                    docs_scored += len(docs)
                    episodic_reads += int(reads)
                    episodic_writes += int(writes)
                    chunks_scored += int(chunks)
                    if selected_slots > 0:
                        controller_reads += 1
                        controller_selected_slots += int(selected_slots)
            else:
                for doc in _iter_score_docs(val_cache, max_docs=max_docs):
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
                        (
                            residual,
                            gate,
                            read_hit,
                            selected_slots,
                        ) = _episodic_packet_from_prefix(
                            model,
                            states=states,
                            batch_size=int(chunk_inputs.shape[0]),
                            gate_value=gate_value,
                            controller_read_enabled=controller_read_enabled,
                            controller_topk_k=controller_topk_k,
                            controller_score_mode=controller_score_mode,
                        )
                        if read_hit:
                            episodic_reads += 1
                        if selected_slots > 0:
                            controller_reads += 1
                            controller_selected_slots += int(selected_slots)
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
                    (
                        residual,
                        gate,
                        read_hit,
                        selected_slots,
                    ) = _episodic_packet_from_prefix(
                        model,
                        states=states,
                        batch_size=int(tail.shape[0]),
                        gate_value=gate_value,
                        controller_read_enabled=controller_read_enabled,
                        controller_topk_k=controller_topk_k,
                        controller_score_mode=controller_score_mode,
                    )
                    if read_hit:
                        episodic_reads += 1
                    if selected_slots > 0:
                        controller_reads += 1
                        controller_selected_slots += int(selected_slots)
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
            "batch_docs": batch_docs,
            "batch_token_budget": batch_token_budget,
            "controller_read_enabled": controller_read_enabled,
            "controller_topk_k": controller_topk_k,
            "controller_score_mode": controller_score_mode,
        },
        extra={
            "episodic_reads": episodic_reads,
            "episodic_writes": episodic_writes,
            "chunks_scored": chunks_scored,
            "controller_reads": controller_reads,
            "controller_selected_slots": controller_selected_slots,
            "slot_count_initial": initial_slot_count,
            "slot_count_final": _outer_slot_count(model),
        },
    )
