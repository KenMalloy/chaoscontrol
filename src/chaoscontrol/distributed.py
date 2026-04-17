"""DDP primitives for the lean training path.

Canonical home for ``_broadcast_params``, ``_allreduce_grads``,
``should_stop_now``, and ``resolve_ddp_context`` going forward.
``training.py`` retains its own verbatim copies for reproducibility of
every prior experiment — that module is frozen. New code
(``train_ssm``, Exp 19 and later) should import from here instead.

The gradient-sync design here is the outcome of three DDP bugs fixed on
2026-04-16 (see project_ddp_manual_allreduce_2026-04-16.md):

1. ``DistributedDataParallel``'s autograd-hook bucket-readiness mechanism
   desyncs across ranks when ``chunked_cross_entropy`` creates K
   independent backward sub-graphs. Fix: remove the wrapper, all-reduce
   grads explicitly after ``backward()``.

2. Per-rank wall-clock stop-decision desyncs the training loop. Fix:
   ``all_reduce(MAX)`` the stop flag so any rank wanting to stop causes
   all ranks to stop on the same step.

3. Stop-flag branch divergence (stopping rank takes one path, continuing
   rank takes another). Fix: ALL ranks execute the same
   ``all_reduce(MAX)`` unconditionally when DDP is active.
"""
from __future__ import annotations

import os
from typing import Iterable

import torch
import torch.distributed as dist


def broadcast_params(model: torch.nn.Module) -> None:
    """Broadcast all parameters from rank 0 so every rank starts identical."""
    for p in model.parameters():
        dist.broadcast(p.data, src=0)


def allreduce_grads(model: torch.nn.Module, world_size: int) -> None:
    """Average all parameter gradients across DDP ranks.

    Replaces ``DistributedDataParallel``'s autograd-hook gradient sync
    with a single explicit pass after ``loss.backward()``. DDP's hooks
    fire per-bucket during backward, and the bucket-readiness ordering
    can diverge across ranks when chunked backward creates multiple
    independent sub-graphs — leading to NCCL collective count
    mismatches and deadlocks. A post-backward all-reduce avoids the
    ordering problem entirely.

    Coalesces every grad into a single flat buffer for one
    ``dist.all_reduce`` call. N per-param NCCL launches would each pay
    ~100-500µs of fixed-cost overhead regardless of tensor size; a
    single coalesced call pays that cost once and reduces the whole
    flattened buffer in one collective. Uses the flatten/unflatten
    helpers torch's own ``DistributedDataParallel`` relies on.
    """
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        return
    flat = torch._utils._flatten_dense_tensors(grads)
    dist.all_reduce(flat, op=dist.ReduceOp.AVG)
    synced = torch._utils._unflatten_dense_tensors(flat, grads)
    for g, s in zip(grads, synced):
        g.copy_(s)


def clip_grad_norm_fused(
    parameters: Iterable[torch.nn.Parameter],
    max_norm: float,
    norm_type: float = 2.0,
) -> torch.Tensor:
    """Coalesced L2 grad clip — numerical parity with torch.nn.utils.clip_grad_norm_ at fp32 tolerance.

    Reduction order differs (one flat norm vs per-tensor norms stacked and
    re-reduced), so results agree via allclose, not torch.equal. See
    tests/test_distributed.py::TestClipGradNormFused for the tolerance
    contract. The rest of the function body — flatten, norm, clip factor,
    unflatten + in-place copy back — matches allreduce_grads' 2026-04-17
    pattern and preserves grad tensor identity (.data_ptr()) so optimizer
    references stay valid.

    Flattens every grad into one buffer, computes the global L2 norm via
    a single kernel, and applies one in-place multiplicative clip factor
    across the buffer. Avoids the per-param norm + stack + global-norm
    dance the stdlib helper walks through in Python. The clip multiply
    is unconditional — multiplying by ``clip_coef_clamped == 1.0`` when
    the norm is under threshold is trivially cheap, while the
    ``if clip_coef_clamped < 1.0:`` guard would force a GPU->CPU sync
    every step (see torch's own ``_clip_grads_with_norm_`` for the same
    motivation).
    """
    if float(norm_type) != 2.0:
        raise NotImplementedError(
            "clip_grad_norm_fused only implements L2 (norm_type=2.0); "
            "fall back to torch.nn.utils.clip_grad_norm_ for other norms."
        )
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return torch.zeros(())
    flat = torch._utils._flatten_dense_tensors(grads)
    total_norm = flat.norm(p=2)
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    flat.mul_(clip_coef_clamped)
    synced = torch._utils._unflatten_dense_tensors(flat, grads)
    for g, s in zip(grads, synced):
        g.copy_(s)
    return total_norm


def should_stop_now(
    local_should_stop: bool,
    device: torch.device,
    ddp_active: bool,
) -> bool:
    """Synchronize the per-step stop decision across DDP ranks.

    Under DDP all ranks call the same ``all_reduce(MAX)`` unconditionally
    so the result is "any rank wants to stop ⇒ all ranks stop together."
    Divergent code paths around the stop flag were the root cause of
    bug #3 in the 2026-04-16 DDP rewrite — every rank must take the
    same collective-communication path.

    Single-process (``ddp_active=False``) just returns the local flag.
    """
    if ddp_active:
        stop_flag = torch.tensor(
            [1.0 if local_should_stop else 0.0], device=device,
        )
        dist.all_reduce(stop_flag, op=dist.ReduceOp.MAX)
        return stop_flag.item() > 0.5
    return local_should_stop


def resolve_ddp_context(
    rank: int | None,
    world_size: int | None,
) -> tuple[int, int]:
    """Resolve (rank, world_size) from explicit args or env vars.

    Precedence:
        1. Explicit args (both must be provided together).
        2. torch.distributed if initialized.
        3. Env vars RANK / WORLD_SIZE (set by torchrun).
        4. Fallback (0, 1) — single device.

    Returns (rank, world_size) where world_size == 1 means the
    single-device path. In that case no DDP wrapping or barriers
    happen.
    """
    if rank is not None and world_size is not None:
        return int(rank), int(world_size)
    if rank is not None or world_size is not None:
        raise ValueError(
            "rank and world_size must both be provided, or both be None. "
            f"Got rank={rank}, world_size={world_size}."
        )
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank()), int(dist.get_world_size())
    env_rank = os.environ.get("RANK")
    env_world = os.environ.get("WORLD_SIZE")
    if env_rank is not None and env_world is not None:
        return int(env_rank), int(env_world)
    return 0, 1
