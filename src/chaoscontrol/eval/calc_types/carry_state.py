"""``carry_state`` — SSM state continues across doc boundaries.

The model's ``encode(initial_states=..., return_final_states=True)``
threads recurrent state from the end of doc N into the start of doc
N+1. Optional ``decay`` factor (default 1.0) attenuates the carried
state. Order-sensitive — the orchestrator must load ``ValCache`` with
``source_order`` ordering.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from chaoscontrol.eval.ttt_eval import (
    CalcTypeContext,
    CalcTypeResult,
    register_calc_type,
)


@register_calc_type(
    "carry_state",
    requires_source_order=True,
    requires_grad=False,
    description="SSM state carries across docs with optional decay.",
)
def carry_state(ctx: CalcTypeContext) -> CalcTypeResult:
    """Cross-doc state carry with optional decay.

    Hyperparameters:
        decay (float): scale applied to every carried state tensor at
            each doc boundary. ``1.0`` (default) carries unchanged;
            ``0.0`` is equivalent to per-doc reset.

    Per-doc loop: call ``model.encode(input_ids, initial_states=prev,
    return_final_states=True)`` to get ``(hidden, final_states)``, push
    ``hidden`` through ``model.lm_head`` to score, then scale
    ``final_states`` by ``decay`` and use them as ``prev`` for the next
    doc. Docs with ``token_len < 2`` are skipped without disturbing the
    carried state.
    """
    model = ctx.model
    val_cache = ctx.val_cache
    device = ctx.device

    decay = float(ctx.config.get("decay", 1.0))

    was_training = model.training
    model.eval()
    try:
        total_ce_nats = torch.zeros((), dtype=torch.float64)
        total_tokens_scored = 0
        total_raw_bytes = 0
        docs_scored = 0
        prev_states: list[torch.Tensor] | None = None
        with torch.no_grad():
            for doc in val_cache.iter_docs():
                if doc.token_len < 2:
                    continue
                tokens_np = val_cache.tokens_for_doc(doc)
                input_ids = torch.tensor(
                    tokens_np, dtype=torch.long, device=device
                ).unsqueeze(0)
                hidden, final_states = model.encode(
                    input_ids,
                    initial_states=prev_states,
                    return_final_states=True,
                )
                logits = model.lm_head(hidden)
                ce_sum = F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    input_ids[:, 1:].reshape(-1),
                    reduction="sum",
                )
                total_ce_nats += ce_sum.detach().to(device="cpu", dtype=torch.float64)
                total_tokens_scored += int(input_ids.size(1) - 1)
                total_raw_bytes += int(doc.raw_bytes)
                docs_scored += 1

                if decay == 1.0:
                    prev_states = [s.detach() for s in final_states]
                else:
                    prev_states = [s.detach() * decay for s in final_states]
    finally:
        if was_training:
            model.train()

    ce_nats_f = float(total_ce_nats.item())
    if total_raw_bytes <= 0:
        bpb = 0.0
    else:
        bpb = ce_nats_f / total_raw_bytes / math.log(2.0)
    loss = ce_nats_f / max(total_tokens_scored, 1)

    return CalcTypeResult(
        bpb=bpb,
        loss=loss,
        docs_scored=docs_scored,
        tokens_scored=total_tokens_scored,
        raw_bytes=total_raw_bytes,
        hyperparams={"decay": decay},
        extra={},
    )
