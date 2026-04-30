"""``score_only_reset`` — the floor calc_type.

Reset SSM state per doc, no parameter updates, no extra forward passes.
Mirrors the per-doc loop used by ``scripts/run_exp20_fast_score.py``;
this is the reference Param-Golf-legal eval contract.
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


def _model_logits(out: object) -> torch.Tensor:
    """Extract logits from either a tensor or a dict-shaped model output."""
    if isinstance(out, dict):
        return out["logits"]  # type: ignore[return-value]
    return out  # type: ignore[return-value]


@register_calc_type(
    "score_only_reset",
    requires_source_order=False,
    requires_grad=False,
    description="Reset SSM state per doc; no params changed. The floor.",
)
def score_only_reset(ctx: CalcTypeContext) -> CalcTypeResult:
    """Per-doc reset eval.

    For each doc in ``ctx.val_cache.iter_docs()``, run the model once with
    a fresh SSM state, compute next-token cross-entropy (in nats) over
    every scoreable position, and accumulate. BPB is the standard
    ``total_ce_nats / total_raw_bytes / ln(2)`` aggregator (matches
    :func:`chaoscontrol.evaluation.compute_bpb`).
    """
    model = ctx.model
    val_cache = ctx.val_cache
    device = ctx.device

    was_training = model.training
    model.eval()
    try:
        total_ce_nats = torch.zeros((), dtype=torch.float64)
        total_tokens_scored = 0
        total_raw_bytes = 0
        docs_scored = 0
        with torch.no_grad():
            for doc in val_cache.iter_docs():
                if doc.token_len < 2:
                    continue
                tokens_np = val_cache.tokens_for_doc(doc)
                input_ids = torch.tensor(
                    tokens_np, dtype=torch.long, device=device
                ).unsqueeze(0)
                out = model(input_ids)
                logits = _model_logits(out)
                # Predict token i+1 from tokens 0..i: use logits[:, :-1] vs targets[:, 1:].
                ce_sum = F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    input_ids[:, 1:].reshape(-1),
                    reduction="sum",
                )
                total_ce_nats += ce_sum.detach().to(device="cpu", dtype=torch.float64)
                total_tokens_scored += int(input_ids.size(1) - 1)
                total_raw_bytes += int(doc.raw_bytes)
                docs_scored += 1
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
        hyperparams={},
        extra={},
    )
