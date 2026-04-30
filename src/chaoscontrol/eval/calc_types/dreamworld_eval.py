"""``dreamworld_eval`` — per-doc self-distillation TTT.

For each doc:
1. Snapshot trainable params.
2. Generate K dream rollouts of length L from the current SSM state,
   conditioned on the doc's prefix tokens. Rollouts are sampled
   autoregressively under ``no_grad``; sampling does not require autograd
   and the sampled token IDs are integer "targets" with no gradient path.
3. Compute self-distillation loss: teacher-force the rollout sequence
   through ``model.encode`` + ``lm_head``, CE-loss against the rollout's
   own argmax (or temperature-softmax) targets. Backward; SGD step(s).
4. Score the doc via the same per-token CE accumulation pattern as
   ``scripts/run_exp20_fast_score.py::_score_doc``.
5. Restore params (when ``per_doc_reset=True``, the default).

Design notes
------------
The existing ``experiments/23_fast_path/dreamworld.py::dreamworld_replay_backward``
expects a pre-built ``DreamReplayEntry`` with fixed replay tokens (it
was built for training-time replay over batches). For per-doc eval we
need to *generate* rollouts autoregressively from a doc-prefix state,
so we implement a focused per-doc rollout helper inline rather than
stretching the existing function. The teacher-forced backward shares
the same ``encode → final_norm → lm_head → CE`` formula that the
fallback path of ``dreamworld_replay_backward`` uses.

When ``per_doc_reset=False`` (continual mode), params accumulate across
docs and the calc_type is order-sensitive — the orchestrator must
load ``ValCache`` with ``source_order`` ordering. The default
``per_doc_reset=True`` is order-invariant.

BPB aggregation matches ``chaoscontrol.evaluation.compute_bpb``:
``total_ce_nats / total_raw_bytes / log(2)``.
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


def _packet_encode(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    initial_states: list[torch.Tensor] | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Encode on the packet-clean lane, falling back for tiny test doubles."""
    encode = getattr(model, "encode")
    kwargs: dict[str, Any] = {
        "initial_states": initial_states,
        "return_final_states": True,
    }
    if "memory_mode" in inspect.signature(encode).parameters:
        out = encode(input_ids, memory_mode="packet", **kwargs)
    else:
        out = encode(input_ids, **kwargs)
    if isinstance(out, dict):
        return out["hidden"], list(out["final_states"])
    hidden, final_states = out
    return hidden, list(final_states)


def _packet_logits_and_states(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    initial_states: list[torch.Tensor] | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    hidden, final_states = _packet_encode(
        model,
        input_ids,
        initial_states=initial_states,
    )
    final_norm = getattr(model, "final_norm", None)
    if final_norm is not None:
        hidden = final_norm(hidden)
    logits = model.lm_head(hidden)
    return logits, final_states


def _snapshot_params(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Detached clones of every trainable parameter, keyed by name."""
    return {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if p.requires_grad
    }


def _restore_params(model: torch.nn.Module, snapshot: dict[str, torch.Tensor]) -> None:
    """Copy snapshot values back into params in-place.

    Uses ``p.data.copy_(...)`` rather than reassignment so optimizer /
    buffer references that may hold ``p`` stay valid.
    """
    with torch.no_grad():
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name not in snapshot:
                continue
            p.data.copy_(snapshot[name])


def _expand_states(states: list[torch.Tensor], k: int) -> list[torch.Tensor]:
    """Expand each state's leading batch dim from 1 to ``k``.

    The prefix encode runs at batch=1; for K rollouts we replicate that
    state K times so the rollout forward / backward can run as one
    batch.
    """
    out: list[torch.Tensor] = []
    for state in states:
        if state.shape[0] == k:
            out.append(state)
            continue
        if state.shape[0] != 1:
            raise ValueError(
                f"expected state batch dim 1 before expand, got {state.shape[0]}"
            )
        # ``expand`` is a view; ``contiguous`` to break aliasing so
        # subsequent in-place ops on the rollout path are safe.
        expanded = state.expand(k, *state.shape[1:]).contiguous()
        out.append(expanded)
    return out


def _generate_dream_rollouts(
    model: torch.nn.Module,
    *,
    prefix_tokens: torch.Tensor,
    k: int,
    rollout_len: int,
    target_mode: str,
    temperature: float,
    device: torch.device,
) -> torch.Tensor:
    """Sample K autoregressive rollouts of length ``rollout_len + 1``.

    Returns an ``(K, rollout_len + 1)`` int64 tensor whose first column
    is the prefix's last token (the "anchor" so teacher-forcing has a
    valid first input/target pair) and the remaining columns are
    sampled from the model.

    Sampling runs under ``torch.no_grad``: rollout tokens are integer
    targets; we never want gradients flowing through the sampling path.
    The teacher-forced backward in :func:`_dream_backward` regenerates
    the graph from the same inputs.
    """
    if rollout_len < 1:
        raise ValueError(f"rollout_len must be >= 1, got {rollout_len}")
    with torch.no_grad():
        prefix = prefix_tokens.to(device=device, dtype=torch.long).view(1, -1)
        # Encode the prefix once at batch=1 to get the conditioning state.
        _, prefix_states = _packet_encode(
            model,
            prefix,
            initial_states=None,
        )
        # Replicate prefix state across K rollouts.
        states = _expand_states(prefix_states, k)
        # Anchor token: each rollout starts from the prefix's final
        # token so teacher-forcing in the backward pass has a valid
        # first input (the (input, target) pair (anchor, sample_0)).
        anchor = prefix[0, -1]
        cur = anchor.view(1, 1).expand(k, 1).contiguous()
        sampled: list[torch.Tensor] = [cur]
        for _ in range(rollout_len):
            # Use encode + lm_head explicitly so the post-token recurrent
            # state is always captured. Calling model(cur, initial_states=)
            # would route through forward(), which on the production
            # bare-SSM model returns logits-only and gives no path to
            # propagate state across rollout tokens — leaving rollouts
            # autoregressive in token space but frozen in state space.
            hidden, states = _packet_encode(
                model,
                cur,
                initial_states=states,
            )
            final_norm = getattr(model, "final_norm", None)
            if final_norm is not None:
                hidden = final_norm(hidden)
            logits = model.lm_head(hidden)
            # ``logits`` shape: (K, 1, V).
            step_logits = logits[:, -1, :].to(torch.float32)
            if target_mode == "argmax":
                next_tok = step_logits.argmax(dim=-1, keepdim=True)
            elif target_mode == "softmax":
                probs = F.softmax(step_logits / max(float(temperature), 1e-6), dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
            else:
                raise ValueError(
                    f"dream_target_mode must be 'argmax' or 'softmax', got {target_mode!r}"
                )
            sampled.append(next_tok)
            cur = next_tok
    return torch.cat(sampled, dim=1).to(torch.long)


def _dream_backward(
    model: torch.nn.Module,
    *,
    rollouts: torch.Tensor,
    prefix_states: list[torch.Tensor],
) -> torch.Tensor:
    """Teacher-force the rollouts and run backward; return scalar loss.

    ``rollouts`` shape: ``(K, L+1)``; we use ``[:, :-1]`` as input and
    ``[:, 1:]`` as targets. ``prefix_states`` is the conditioning state
    expanded to batch K (callers pass the already-expanded list).
    """
    inputs = rollouts[:, :-1].contiguous()
    targets = rollouts[:, 1:].contiguous()
    # Detach the conditioning state — backward updates params via the
    # rollout's own logits; we do NOT propagate into the prefix encode.
    cond_states = [s.detach() for s in prefix_states]
    logits, _final_states = _packet_logits_and_states(
        model,
        inputs,
        initial_states=cond_states,
    )
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)).to(torch.float32),
        targets.reshape(-1),
        reduction="mean",
    )
    loss.backward()
    return loss.detach()


def _sgd_step(model: torch.nn.Module, lr: float) -> None:
    """Vanilla SGD on grads currently sitting on params; zero grads after."""
    with torch.no_grad():
        for p in model.parameters():
            if not p.requires_grad or p.grad is None:
                continue
            p.data.add_(p.grad, alpha=-float(lr))
            p.grad = None


def _score_doc_ce(
    model: torch.nn.Module,
    *,
    tokens: torch.Tensor,
    device: torch.device,
) -> tuple[float, int]:
    """Forward-only doc score: returns ``(ce_nats, tokens_scored)``.

    Mirrors the per-token CE accumulation in
    ``scripts/run_exp20_fast_score.py::_score_doc`` for the simple
    single-chunk case. Eval calc_types in this harness operate at
    per-doc granularity; we keep doc-internal chunking out of scope
    here and let the harness pass us whole docs.
    """
    if tokens.numel() < 2:
        return 0.0, 0
    chunk = tokens.to(device=device, dtype=torch.long).view(1, -1)
    with torch.no_grad():
        logits, _final_states = _packet_logits_and_states(model, chunk)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)).to(torch.float32),
            chunk[:, 1:].reshape(-1),
            reduction="sum",
        )
    return float(loss.item()), int(chunk.size(1) - 1)


@register_calc_type(
    "dreamworld_eval",
    requires_source_order=False,
    requires_grad=True,
    description="Per-doc dream-rollout + backward + SGD; default per-doc reset.",
)
def dreamworld_eval(ctx: CalcTypeContext) -> CalcTypeResult:
    """Per-doc dream TTT via self-distillation.

    Reads hyperparameters from ``ctx.config`` with the following
    defaults (also baked into the returned ``hyperparams`` dict):

    - ``K``: 8 — dream rollouts per doc
    - ``L``: 64 — rollout length in tokens
    - ``lr``: 1e-3 — SGD learning rate
    - ``steps``: 1 — SGD steps per doc
    - ``per_doc_reset``: True — restore params after each doc
    - ``dream_target_mode``: ``"argmax"`` — or ``"softmax"``
    - ``dream_temperature``: 1.0 — sampling temperature for softmax
    - ``prefix_len``: 16 — doc-prefix tokens used to condition rollouts
    """
    cfg = ctx.config
    k = int(cfg.get("K", 8))
    rollout_len = int(cfg.get("L", 64))
    lr = float(cfg.get("lr", 1e-3))
    steps = int(cfg.get("steps", 1))
    per_doc_reset = bool(cfg.get("per_doc_reset", True))
    target_mode = str(cfg.get("dream_target_mode", "argmax"))
    temperature = float(cfg.get("dream_temperature", 1.0))
    prefix_len = int(cfg.get("prefix_len", 16))

    if k < 1:
        raise ValueError(f"K must be >= 1, got {k}")
    if rollout_len < 1:
        raise ValueError(f"L must be >= 1, got {rollout_len}")
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if prefix_len < 1:
        raise ValueError(f"prefix_len must be >= 1, got {prefix_len}")
    if target_mode not in {"argmax", "softmax"}:
        raise ValueError(
            f"dream_target_mode must be 'argmax' or 'softmax', got {target_mode!r}"
        )
    if not per_doc_reset:
        # Continual mode is order-sensitive but the registry advertises
        # this calc_type as ``requires_source_order=False`` because the
        # default per-doc-reset path is order-invariant. Allowing the
        # continual path through ``calc_type_configs`` would silently
        # violate Param Golf order semantics. If continual eval is ever
        # wanted, it must be a separate calc_type registered with the
        # source-order flag.
        raise ValueError(
            "per_doc_reset=False is not supported for dreamworld_eval; "
            "continual mode is order-sensitive and the registered "
            "metadata advertises requires_source_order=False. Use the "
            "default per_doc_reset=True, or register a separate "
            "continual calc_type with requires_source_order=True."
        )

    model = ctx.model
    device = ctx.device

    total_ce_nats = 0.0
    total_tokens_scored = 0
    total_raw_bytes = 0
    docs_scored = 0

    was_training = model.training
    model.eval()
    try:
        for doc in ctx.val_cache.iter_docs():
            doc_tokens_np = ctx.val_cache.tokens_for_doc(doc)
            token_len = int(doc.token_len)
            if token_len < 2:
                # Nothing to score; respect raw_bytes accounting still.
                total_raw_bytes += int(doc.raw_bytes)
                continue

            # Clamp prefix so scoring still has at least one CE-bearing
            # token left after the prefix — this matters for tiny test
            # docs and is harmless on full-length val docs.
            effective_prefix = min(prefix_len, token_len - 1)
            prefix_tokens = torch.from_numpy(
                doc_tokens_np[:effective_prefix].astype("int64")
            ).to(device=device)
            doc_tokens = torch.from_numpy(
                doc_tokens_np.astype("int64")
            ).to(device=device)

            snapshot = _snapshot_params(model) if per_doc_reset else None

            # --- Dream phase: rollout under no_grad, backward under enable_grad. ---
            with torch.enable_grad():
                for _ in range(steps):
                    rollouts = _generate_dream_rollouts(
                        model,
                        prefix_tokens=prefix_tokens,
                        k=k,
                        rollout_len=rollout_len,
                        target_mode=target_mode,
                        temperature=temperature,
                        device=device,
                    )
                    # Re-encode the prefix to get a fresh, grad-tracked
                    # state for the teacher-forced backward. We detach
                    # it inside ``_dream_backward`` so backward only
                    # touches params via the rollout logits, not via
                    # the prefix encode graph.
                    _, prefix_states = _packet_encode(
                        model,
                        prefix_tokens.view(1, -1),
                        initial_states=None,
                    )
                    expanded = _expand_states(prefix_states, k)
                    _dream_backward(
                        model,
                        rollouts=rollouts,
                        prefix_states=expanded,
                    )
                    _sgd_step(model, lr)

            # --- Score phase: forward-only over doc tokens. ---
            ce_nats, tokens_scored = _score_doc_ce(
                model,
                tokens=doc_tokens,
                device=device,
            )

            total_ce_nats += ce_nats
            total_tokens_scored += tokens_scored
            total_raw_bytes += int(doc.raw_bytes)
            docs_scored += 1

            if per_doc_reset and snapshot is not None:
                _restore_params(model, snapshot)
    finally:
        if was_training:
            model.train()

    if total_raw_bytes <= 0:
        bpb = 0.0
    else:
        bpb = float(total_ce_nats / total_raw_bytes / math.log(2.0))
    if total_tokens_scored > 0:
        mean_loss = float(total_ce_nats / total_tokens_scored)
    else:
        mean_loss = 0.0

    hyperparams = {
        "K": k,
        "L": rollout_len,
        "lr": lr,
        "steps": steps,
        "per_doc_reset": per_doc_reset,
        "dream_target_mode": target_mode,
        "dream_temperature": temperature,
        "prefix_len": prefix_len,
    }

    return CalcTypeResult(
        bpb=bpb,
        loss=mean_loss,
        docs_scored=docs_scored,
        tokens_scored=total_tokens_scored,
        raw_bytes=total_raw_bytes,
        hyperparams=hyperparams,
    )
