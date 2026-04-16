"""Lean training path for the bare-SSM regime.

This module is the throughput-first training entry point introduced on
2026-04-16 to support Exp 19 and the paper's large-batch pre-training
experiments. It is intentionally narrower than ``training.py``:

- Only supports the bare-SSM architecture (no metabolic forks, no MC
  sampling, no criticality regularization, no typed buffer, no wernicke,
  no alignment/commit/recon losses).
- Uses chunked LM-head backward so the full ``(B, T, V)`` logits
  gradient is never materialized, which unlocks bs×seq regimes that
  OOM under ``training.py``'s path (see
  ``project_chunked_lm_backward_design_2026-04-16.md``).

``training.py`` remains the path for every experiment that needs any of
the above features. The two paths co-exist by design — the repo is a
paper testbed, and every prior experiment must remain reproducible.
"""
from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from chaoscontrol.data import batch_from_starts
from chaoscontrol.distributed import (
    allreduce_grads,
    broadcast_params,
    resolve_ddp_context,
    should_stop_now,
)


def chunked_lm_head_backward(
    hidden: torch.Tensor,
    final_norm: nn.Module,
    lm_head: nn.Linear,
    targets: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Run LM-head forward + CE + backward in time-chunks, without
    materializing the full ``(B, T, V)`` logits tensor.

    The caller is responsible for detaching ``hidden`` from the encoder
    graph and setting ``requires_grad=True`` before this call. On
    return, ``hidden.grad``, ``lm_head.weight.grad``, and
    ``final_norm.weight.grad`` have been populated in-place via
    per-chunk ``backward()``. The caller then propagates
    ``hidden.grad`` into the encoder graph (typically via
    ``torch.autograd.backward([enc_hidden, aux_loss], [hidden.grad, None])``).

    Args:
        hidden: ``(batch, seq, dim)`` detached leaf tensor with
            ``requires_grad=True``.
        final_norm: RMSNorm-or-similar module applied per-token before
            the projection.
        lm_head: ``nn.Linear(dim, vocab_size, bias=False)``.
        targets: ``(batch, seq)`` int64 token targets.
        chunk_size: time-axis chunk size. At ``chunk_size >= seq`` the
            loop runs once and matches a naive non-chunked backward.

    Returns:
        Total CE loss as an fp32 scalar (detached — already backprop'd),
        suitable for logging.

    Notes:
        Each chunk's loss is computed as ``CE(reduction='sum') /
        total_tokens`` so the accumulated gradient across chunks equals
        the gradient of a single mean-reduced loss over all tokens. The
        fp32 upcast matches the contract of ``training.chunked_cross_entropy``.
    """
    if not hidden.requires_grad:
        raise ValueError(
            "chunked_lm_head_backward: ``hidden`` must have requires_grad=True "
            "so per-chunk backward can accumulate into hidden.grad. Call "
            "hidden.detach().requires_grad_(True) before passing it here."
        )
    batch, seq, _ = hidden.shape
    vocab = lm_head.out_features
    total_tokens = batch * seq

    loss_accum = hidden.new_zeros((), dtype=torch.float32)

    start = 0
    while start < seq:
        end = min(start + chunk_size, seq)
        # Slice is a view on ``hidden``; gradient from downstream ops
        # flows back into ``hidden.grad`` at the corresponding positions.
        h_chunk = hidden[:, start:end, :]
        logits_chunk = lm_head(final_norm(h_chunk))  # (B, chunk_T, V)
        tgt_chunk = targets[:, start:end]
        # ``reduction='sum'`` on fp32-upcast logits, then divide by the
        # global token count. Summed across chunks this equals mean CE
        # over all (B×T) tokens. Using sum + global-N avoids the
        # per-chunk mean that would otherwise miscount when the last
        # chunk is shorter than ``chunk_size``.
        chunk_loss = F.cross_entropy(
            logits_chunk.reshape(-1, vocab).float(),
            tgt_chunk.reshape(-1),
            reduction="sum",
        ) / total_tokens
        chunk_loss.backward()
        loss_accum = loss_accum + chunk_loss.detach()
        start = end

    return loss_accum


_UNSUPPORTED_ATTRS: tuple[tuple[str, str], ...] = (
    ("wernicke", "wernicke layer"),
    ("outer_model", "outer_model / typed buffer"),
    ("semantic_tier", "semantic_tier bias"),
    ("posterior", "posterior correction module"),
    ("bucket_prototypes_module", "bucket_prototypes"),
)


def _reject_unsupported(model: torch.nn.Module) -> None:
    """Fail fast when ``train_ssm`` is called on a config it doesn't support.

    ``train_ssm`` is intentionally narrower than ``training.py``; it
    only understands the bare-SSM path. Silently ignoring extra
    modules would produce gradients that don't match the old loop and
    invalidate experiment reproducibility. Better to error at the
    entry point.
    """
    for attr, label in _UNSUPPORTED_ATTRS:
        if getattr(model, attr, None) is not None:
            raise ValueError(
                f"train_ssm does not support the {label} path "
                f"(model.{attr} is not None). Use "
                f"chaoscontrol.training.train_chaoscontrol_for_budget "
                f"for configs that enable this feature."
            )


def train_ssm_step(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    chunk_size: int,
    *,
    ddp_active: bool = False,
    world_size: int = 1,
) -> torch.Tensor:
    """One forward + chunked-backward cycle for a bare-SSM model.

    Runs ``model.encode(inputs)`` to produce the hidden state, then
    loops chunked forward/backward through ``final_norm + lm_head + CE``
    accumulating gradients into the detached hidden tensor. A single
    encoder backward propagates the accumulated hidden gradient into
    encoder parameters. When DDP is active, per-parameter gradients
    are all-reduced (AVG) before return.

    The caller owns ``optimizer.step()`` and ``optimizer.zero_grad()``
    — this function only computes gradients and leaves them on the
    parameters.

    Returns the total CE loss as an fp32 scalar (already backprop'd).
    """
    _reject_unsupported(model)

    # Encoder forward — hidden has requires_grad via model params.
    hidden = model.encode(inputs)

    # Detach to form a boundary between the chunked LM-head graph and
    # the encoder graph. The chunked backward accumulates gradient
    # into hidden_for_ce.grad; that gradient is then injected into the
    # encoder by ``hidden.backward(gradient=hidden_for_ce.grad)``.
    hidden_for_ce = hidden.detach().requires_grad_(True)
    loss = chunked_lm_head_backward(
        hidden=hidden_for_ce,
        final_norm=model.final_norm,
        lm_head=model.lm_head,
        targets=targets,
        chunk_size=chunk_size,
    )

    # Single encoder backward — ``hidden_for_ce.grad`` is the complete
    # downstream gradient that ``hidden`` should receive.
    hidden.backward(gradient=hidden_for_ce.grad)

    if ddp_active and world_size > 1:
        allreduce_grads(model, world_size)

    return loss


def train_ssm_for_budget(
    model: torch.nn.Module,
    *,
    train_tokens: torch.Tensor,
    train_starts: list[int],
    seq_len: int,
    batch_size: int,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    budget_seconds: float,
    chunk_size: int,
    grad_clip_norm: float = 0.0,
    rank: int | None = None,
    world_size: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Bare-SSM wall-clock training loop.

    Mirrors ``training.train_chaoscontrol_for_budget`` at the level
    that applies to a bare-SSM config: no metabolic fork, no MC, no
    criticality, no typed buffer, no wernicke, no alignment, no
    tokenizer losses. See ``train_ssm_step`` for the per-step math.

    DDP is optional — single-GPU is the ``world_size == 1`` path. When
    multi-rank, initial params are broadcast from rank 0 and the
    stop decision is synchronized across ranks via ``should_stop_now``.
    """
    _reject_unsupported(model)

    rank_, world_size_ = resolve_ddp_context(rank, world_size)
    ddp_active = world_size_ > 1
    if ddp_active:
        broadcast_params(model)

    import random
    rng = random.Random(seed + rank_)

    model.train()
    history: list[dict[str, Any]] = []
    steps = 0
    start_time = time.perf_counter()

    while True:
        elapsed = time.perf_counter() - start_time
        local_stop = elapsed >= budget_seconds and steps > 0
        if should_stop_now(local_stop, device, ddp_active):
            break

        batch_starts = [
            train_starts[rng.randrange(len(train_starts))]
            for _ in range(batch_size)
        ]
        inputs, targets = batch_from_starts(
            train_tokens, batch_starts, seq_len, device,
        )

        optimizer.zero_grad(set_to_none=True)
        loss = train_ssm_step(
            model=model,
            inputs=inputs,
            targets=targets,
            chunk_size=chunk_size,
            ddp_active=ddp_active,
            world_size=world_size_,
        )
        if grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        history.append({"step": float(steps), "loss": float(loss.detach().cpu())})
        steps += 1

    return {
        "history": history,
        "steps": steps,
        "elapsed_seconds": time.perf_counter() - start_time,
        "rank": rank_,
        "world_size": world_size_,
    }
