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

import functools
import time
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from chaoscontrol.data import batch_from_starts
from chaoscontrol.distributed import (
    allreduce_grads,
    broadcast_params,
    clip_grad_norm_fused,
    resolve_ddp_context,
    should_stop_now,
)
from chaoscontrol.kernels._lm_head_loss import fused_linear_cross_entropy, fused_rms_norm
from chaoscontrol.precision import autocast_context


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
    if chunk_size <= 0:
        raise ValueError(
            f"chunked_lm_head_backward: chunk_size must be positive, got "
            f"{chunk_size}. A non-positive value would wedge the time-chunk "
            f"loop indefinitely rather than failing cleanly."
        )
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


def full_lm_head_backward(
    hidden: torch.Tensor,
    final_norm: nn.Module,
    lm_head: nn.Linear,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Run final norm + LM head + CE as one autograd graph.

    This is the fast path for shapes where the full ``(B, T, V)`` logits
    tensor fits in memory. Unlike ``chunked_lm_head_backward``, it does
    not detach ``hidden`` or inject gradients through a second backward;
    a single ``loss.backward()`` propagates through the head and encoder.
    """
    vocab = lm_head.out_features
    logits = lm_head(final_norm(hidden))
    loss = F.cross_entropy(
        logits.reshape(-1, vocab).float(),
        targets.reshape(-1),
        reduction="mean",
    )
    loss.backward()
    return loss.detach()


def fused_lm_head_backward(
    hidden: torch.Tensor,
    final_norm: nn.Module,
    lm_head: nn.Linear,
    targets: torch.Tensor,
    *,
    backend: str = "auto",
    tile_size: int = 1024,
) -> torch.Tensor:
    """LM-head backward using native fused pieces where available.

    This path is the single-backward fast hook for the Exp23 head/loss
    hot spot: native RMSNorm where eligible, then the fused linear+CE
    API. On dev machines or unsupported tensor layouts the CE API falls
    back to the exact PyTorch ``linear -> cross_entropy`` expression.
    """
    weight = getattr(final_norm, "weight", None)
    eps = float(getattr(final_norm, "eps", 1e-6))
    if weight is None:
        return full_lm_head_backward(hidden, final_norm, lm_head, targets)

    normed = fused_rms_norm(hidden, weight, eps=eps)
    loss = fused_linear_cross_entropy(
        normed,
        lm_head.weight,
        targets,
        reduction="mean",
        backend=backend,
        tile_size=int(tile_size),
    )
    loss.backward()
    return loss.detach()


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


def _encode_only(
    model: torch.nn.Module,
    inputs: torch.Tensor,
) -> torch.Tensor:
    """Compile target: encoder forward through ``model.encode``.

    This is the narrowed compile scope used by ``train_ssm_step`` when
    ``compile_full_path=True``. Everything downstream of the encoder
    (detach + ``requires_grad_``, chunked LM-head forward/backward,
    encoder backward, DDP allreduce) is intentionally excluded because
    dynamo under ``fullgraph=True`` rejects both ``Tensor.requires_grad_``
    calls and in-graph ``.backward()`` — the two operations the chunked
    CE algorithm's memory architecture requires at the detach boundary.
    Compiling the encoder alone still captures the dominant cost of the
    step (encoder matmuls + SSM recurrences) while leaving the
    autograd-orchestration fragment eager.

    Autocast is entered by the eager caller so this function inherits
    whatever precision policy the outer ``autocast_context`` selected;
    dynamo traces the matmuls inside it correctly.

    Nested-compile fix: the encoder path has two cached lazy
    ``torch.compile`` wrappers (``core._diag_recurrence_impl`` at
    ``core.py:152``, ``core_fused._post_scan_impl`` at
    ``core_fused.py:180``). Without mitigation, the outer
    ``torch.compile(fullgraph=True)`` here would trace INTO those
    wrappers and produce a double-compile — the 1A-4 microbench saw
    a −62.77% tok/s regression from exactly that pattern. Both
    dispatchers now early-return the uncompiled inner when
    ``torch.compiler.is_compiling()`` reports outer tracing, so the
    outer gets a single unified graph and eager callers still benefit
    from the inner compile.
    """
    return model.encode(inputs)


@functools.cache
def _compiled_step_fn() -> Callable[..., torch.Tensor]:
    """Compiled wrapper over ``_encode_only``, memoized so
    ``torch.compile`` is invoked once per process rather than once per step.

    ``fullgraph=True`` is load-bearing: it raises on any graph break
    instead of silently falling back to eager, so the "zero graph breaks
    on the encoder forward" claim this flag gates is genuine.
    ``dynamic=False`` opts into static shapes for larger inductor
    speedup; dynamo will recompile on shape change.

    Scope note: the compiled region is the encoder forward only. The
    detach + ``requires_grad_(True)`` boundary that sets up chunked
    CE, every per-chunk ``.backward()`` inside
    ``chunked_lm_head_backward``, and the single ``hidden.backward()``
    that propagates accumulated gradient into the encoder all run eager
    by design — dynamo rejects both ``Tensor.requires_grad_`` and
    in-graph ``.backward()`` under ``fullgraph=True``, so the algorithm
    cannot live inside the compiled graph. Expected inductor speedup
    is therefore bounded by the encoder fraction of step time
    (~1.5-2% at Phase 1 shapes) rather than the full-path 5% that a
    hypothetical end-to-end compile would have delivered.

    Caveat on cache semantics: ``@cache`` returns the same
    ``OptimizedModule`` regardless of which model is passed at call
    time. dynamo itself keys its internal guard cache on argument
    structure (including model identity), so different models correctly
    trigger internal recompiles — but the cache here stores a single
    wrapper. Fine for the intended one-model-per-process training flow;
    revisit if a future eval/debug path instantiates multiple models in
    the same process.
    """
    return torch.compile(
        _encode_only, fullgraph=True, dynamic=False,
    )


def train_ssm_step(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    chunk_size: int,
    *,
    ddp_active: bool = False,
    world_size: int = 1,
    precision: str = "bf16",
    compile_full_path: bool = False,
    lm_head_backward_mode: str = "chunked",
) -> torch.Tensor:
    """One forward + chunked-backward cycle for a bare-SSM model.

    Runs ``model.encode(inputs)`` to produce the hidden state. In the
    default ``lm_head_backward_mode="chunked"`` path it then loops chunked
    forward/backward through ``final_norm + lm_head + CE`` accumulating
    gradients into a detached hidden tensor before one encoder backward.
    In ``"single"`` mode it materializes full logits and calls
    ``loss.backward()`` once, which removes the detached-hidden bridge at
    the cost of higher peak memory. ``"fused"`` is the Exp23 native path:
    currently fused RMSNorm + PyTorch/cuBLAS LM head + CE. When DDP is
    active, per-parameter gradients are all-reduced (AVG) before return.

    The caller owns ``optimizer.step()`` and ``optimizer.zero_grad()``
    — this function only computes gradients and leaves them on the
    parameters.

    ``precision`` selects the autocast policy around the forward +
    backward; see ``chaoscontrol.precision.autocast_context``. The
    default "bf16" preserves the pre-2026-04-16 behavior (existing
    callers don't pass the kwarg). "fp8" requires transformer_engine
    and requires the caller to have pre-promoted nn.Linear -> te.Linear
    via ``maybe_promote_linears_to_te``.

    ``compile_full_path`` wraps the encoder forward
    (``model.encode``) under ``torch.compile(fullgraph=True,
    dynamic=False)``. The name is retained for backward compatibility
    with callers and configs, but the compiled region is narrower than
    the flag suggests: the detach + ``requires_grad_(True)`` boundary,
    the chunked LM-head forward/backward loop, the encoder backward,
    and the DDP allreduce all stay eager because dynamo under
    ``fullgraph=True`` rejects both ``Tensor.requires_grad_`` and
    in-graph ``.backward()``. ``fullgraph=True`` still gates a genuine
    "zero graph breaks on the encoder forward" guarantee —
    it raises on any break in that region rather than silently falling
    back to eager. Expected inductor speedup is bounded by the encoder
    fraction of step time at Phase 1 shapes.

    Returns the total CE loss as an fp32 scalar (already backprop'd).
    """
    _reject_unsupported(model)

    # One autocast block covering BOTH forward and backward. For bf16
    # this is a documentation-level nicety (backward restores saved
    # tensors to forward dtype regardless), but for fp8+TE the
    # fp8_autocast context IS required during backward so TE's recipe
    # tracker sees the backward-pass matmuls. The compiled ``_encode_only``
    # (when enabled) runs inside this context and inherits the policy.
    device_type = inputs.device.type
    with autocast_context(precision, device_type=device_type):
        if compile_full_path:
            hidden = _compiled_step_fn()(model, inputs)
        else:
            hidden = model.encode(inputs)

        if lm_head_backward_mode == "single":
            loss = full_lm_head_backward(
                hidden=hidden,
                final_norm=model.final_norm,
                lm_head=model.lm_head,
                targets=targets,
            )
        elif lm_head_backward_mode == "fused":
            loss = fused_lm_head_backward(
                hidden=hidden,
                final_norm=model.final_norm,
                lm_head=model.lm_head,
                targets=targets,
            )
        elif lm_head_backward_mode == "chunked":
            # Detach to form a boundary between the chunked LM-head graph and
            # the encoder graph. The chunked backward accumulates gradient
            # into hidden_for_ce.grad; that gradient is then injected into the
            # encoder by ``hidden.backward(gradient=hidden_for_ce.grad)``.
            # This step is eager by design — ``requires_grad_`` is unsupported
            # under ``fullgraph=True``, as is the per-chunk ``.backward()``
            # inside ``chunked_lm_head_backward``.
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
        else:
            raise ValueError(
                "lm_head_backward_mode must be 'chunked', 'single', or "
                f"'fused', got {lm_head_backward_mode!r}"
            )

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
    fused_grad_clip: bool = False,
    rank: int | None = None,
    world_size: int | None = None,
    seed: int = 0,
    time_fn: Callable[[], float] = time.perf_counter,
    precision: str = "bf16",
    compile_full_path: bool = False,
    max_steps: int | None = None,
    lm_head_backward_mode: str = "chunked",
) -> dict[str, Any]:
    """Bare-SSM wall-clock training loop.

    Mirrors ``training.train_chaoscontrol_for_budget`` at the level
    that applies to a bare-SSM config: no metabolic fork, no MC, no
    criticality, no typed buffer, no wernicke, no alignment, no
    tokenizer losses. See ``train_ssm_step`` for the per-step math.

    DDP is optional — single-GPU is the ``world_size == 1`` path. When
    multi-rank, initial params are broadcast from rank 0 and the
    stop decision is synchronized across ranks via ``should_stop_now``.
    ``time_fn`` defaults to ``time.perf_counter`` and exists so tests
    can drive the budget loop deterministically without depending on
    machine speed.

    ``precision`` is threaded through to every ``train_ssm_step`` call;
    see the step docstring for the autocast policy it selects.
    """
    _reject_unsupported(model)

    rank_, world_size_ = resolve_ddp_context(rank, world_size)
    ddp_active = world_size_ > 1
    if ddp_active:
        broadcast_params(model)

    import random
    rng = random.Random(seed + rank_)

    model.train()
    # GPU-side loss accumulator: detached scalar tensors appended per step,
    # stacked + transferred to CPU once at loop exit. Eliminates the per-step
    # `.cpu()` sync that was forcing the Python dispatcher to wait on GPU
    # completion every iteration. The single end-of-loop transfer lets the
    # CPU dispatcher run ahead of the GPU during training.
    loss_tensors: list[torch.Tensor] = []
    steps = 0
    start_time = time_fn()

    # Training-only peak-memory tracking: reset before the loop so the
    # returned peak reflects training allocations only, not any prior
    # setup or later eval. Matches training.py's contract so callers
    # can read train_result["peak_vram_mb"] uniformly across both paths.
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    while True:
        elapsed = time_fn() - start_time
        budget_reached = elapsed >= budget_seconds and steps > 0
        max_steps_reached = max_steps is not None and steps >= max_steps
        local_stop = budget_reached or max_steps_reached
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
            precision=precision,
            compile_full_path=compile_full_path,
            lm_head_backward_mode=lm_head_backward_mode,
        )
        if grad_clip_norm > 0.0:
            if fused_grad_clip:
                clip_grad_norm_fused(model.parameters(), grad_clip_norm)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        # Post-step amax bookkeeping for the bespoke fp8 path. Each
        # FusedFP8Linear holds (x/w/gy) amax-history rings + pending
        # buffers; the per-forward/backward kernels atomicMax-fold into
        # the pending slots, and this call rolls them into the history
        # ring + recomputes the live scale. Skipping it leaves scales
        # stale (correctness-preserving; just loses the recent-amax
        # trajectory). The helper is a no-op when the model contains no
        # FusedFP8Linear submodules, so the unconditional call is safe
        # under any precision setting.
        if precision == "fp8_fused":
            from chaoscontrol.kernels.fp8_linear import fused_fp8_flush_all
            fused_fp8_flush_all(model)

        loss_tensors.append(loss.detach())
        steps += 1

    # DDP teardown barrier — same role as training.py:903. Without this, a
    # faster rank can start eval / tear down the process group while a
    # slower rank is still iterating, and the next collective blocks on
    # mismatched state. Matches the frozen path's contract.
    if ddp_active:
        dist.barrier()

    # Single GPU->CPU transfer for the whole trajectory — collapses N per-step
    # syncs into one stack+copy, preserving the {"step", "loss"} history
    # contract consumers rely on.
    if loss_tensors:
        stacked_losses = torch.stack(loss_tensors).cpu()
        history: list[dict[str, Any]] = [
            {"step": float(i), "loss": float(stacked_losses[i])}
            for i in range(len(loss_tensors))
        ]
    else:
        history = []

    peak_vram_mb = 0.0
    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    return {
        "history": history,
        "steps": steps,
        "elapsed_s": time_fn() - start_time,
        "rank": rank_,
        "world_size": world_size_,
        "peak_vram_mb": peak_vram_mb,
    }
