"""Rank-3 oracle utility scoring for the CRCT (Cache-Reweighted
Continuation Training) architecture.

This module computes a per-token utility signal — NLL without episodic
memory minus NLL with episodic memory — and converts it into:

* a controller-head probability target (with optional scarcity-aware
  shadow pricing on the read budget),
* a positive-only language-model loss reweighting, and
* per-entry credit/debit accumulators for memory housekeeping.

The whole module runs on rank 3, which is otherwise idle during training.
It assumes ``model.encode(memory_mode=...)`` and a transactional cache
with a monotone read-cutoff clock — both land on parallel branches and
will be wired into ``runner_fast_path.py`` in a follow-up task.
"""
from __future__ import annotations

import contextlib
import math
from typing import Any

import torch
import torch.nn.functional as F


__all__ = [
    "alpha_ramp",
    "assign_memory_credit",
    "chunked_nll_from_hidden",
    "positive_only_lm_weight",
    "rank3_score_batch_causal",
    "ScarcityAwareMemoryOptimizer",
]


# ---------------------------------------------------------------------------
# Per-token LM loss weighting.
# ---------------------------------------------------------------------------


def positive_only_lm_weight(
    utility: torch.Tensor,
    mask: torch.Tensor,
    *,
    tau: float,
    strength: float,
    w_max: float,
) -> torch.Tensor:
    """Per-token language-model loss weight that never downweights.

    Tokens where memory helped (utility > 0) get a soft upweight via
    ``1 + strength * sigmoid(utility / tau)``; tokens where memory hurt
    (utility < 0) bottom out at ``1.0`` because ``sigmoid → 0``. The
    raw weight is clamped to ``[1.0, w_max]`` as a safety bound, then
    mean-1 normalized over the valid positions so the total gradient
    magnitude across the batch is unchanged.

    Invalid positions (``mask == False``) are set to zero — they do
    not contribute to the mean and do not flow gradients through the
    LM head.
    """
    mask_bool = mask.bool()
    raw = 1.0 + strength * torch.sigmoid(utility.float() / tau)
    weights = raw.clamp(min=1.0, max=w_max)

    if mask_bool.any():
        mean_w = weights[mask_bool].mean().clamp(min=1e-8)
        weights = weights / mean_w

    return weights * mask_bool.float()


# ---------------------------------------------------------------------------
# Chunked per-token NLL through the LM head.
# ---------------------------------------------------------------------------


@torch.inference_mode()
def chunked_nll_from_hidden(
    model: Any,
    hidden_states: torch.Tensor,
    targets: torch.Tensor,
    *,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Per-token negative log-likelihood ``(B, T)`` from encoder hidden
    states, computed in time-axis chunks to bound peak memory.

    Mirrors the ``final_norm → lm_head → cross_entropy`` ordering that
    ``train_ssm.chunked_lm_head_backward`` uses, but returns the raw
    per-token NLL (``reduction='none'``) instead of a scalar loss —
    rank-3 scoring needs the per-position signal so it can compute
    utility deltas pointwise.
    """
    if chunk_size <= 0:
        raise ValueError(
            f"chunked_nll_from_hidden: chunk_size must be positive, got {chunk_size}"
        )
    batch, seq, _ = hidden_states.shape
    final_norm = model.final_norm
    lm_head = model.lm_head
    vocab = lm_head.out_features

    out = hidden_states.new_zeros((batch, seq), dtype=torch.float32)
    start = 0
    while start < seq:
        end = min(start + chunk_size, seq)
        h_chunk = hidden_states[:, start:end, :]
        logits_chunk = lm_head(final_norm(h_chunk))
        tgt_chunk = targets[:, start:end]
        nll_flat = F.cross_entropy(
            logits_chunk.reshape(-1, vocab).float(),
            tgt_chunk.reshape(-1),
            reduction="none",
        )
        out[:, start:end] = nll_flat.reshape(batch, end - start)
        start = end
    return out


# ---------------------------------------------------------------------------
# Alpha ramp schedule.
# ---------------------------------------------------------------------------


def alpha_ramp(step: int, total_steps: int, *, alpha_max: float) -> float:
    """Sigmoid ramp ``alpha_max * sigmoid(8 * (step/total - 0.3))``.

    Bootstraps the loss-reweighting strength: ~0.083 * alpha_max at
    step 0, exactly 0.5 * alpha_max at 30% through training, and
    ~0.996 * alpha_max at the end. Guards ``total_steps == 0``.
    """
    if total_steps <= 0:
        progress = 1.0
    else:
        progress = step / float(total_steps)
    return alpha_max / (1.0 + math.exp(-8.0 * (progress - 0.3)))


# ---------------------------------------------------------------------------
# Scarcity-aware controller targeting.
# ---------------------------------------------------------------------------


class ScarcityAwareMemoryOptimizer:
    """Tracks shadow prices for memory reads and writes via primal-dual
    updates. The controller is targeted on the *net* utility (utility
    minus the current shadow price), so memory only fires when its
    expected NLL gain exceeds its budgeted cost.

    The dual variable rises when the actual read/write rate exceeds
    the target rate (penalize over-use) and falls otherwise (encourage
    use until the rate hits target). EMA smoothing on the rate
    estimates damps the dual oscillation.
    """

    def __init__(
        self,
        *,
        tau: float = 0.10,
        target_read_rate: float = 0.25,
        target_write_rate: float = 0.10,
        dual_lr: float = 0.01,
        ema_beta: float = 0.95,
        max_price: float = 0.50,
    ) -> None:
        self.tau = float(tau)
        self.target_read_rate = float(target_read_rate)
        self.target_write_rate = float(target_write_rate)
        self.dual_lr = float(dual_lr)
        self.ema_beta = float(ema_beta)
        self.max_price = float(max_price)

        self.read_price: float = 0.0
        self.write_price: float = 0.0
        self.read_rate_ema: float = float(target_read_rate)
        self.write_rate_ema: float = float(target_write_rate)

    @torch.no_grad()
    def controller_target(
        self, utility: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(target, confidence)`` for the controller BCE head.

        ``target`` is the bid probability — where memory is worth
        firing given the shadow price. ``confidence`` is ``tanh(|net|/tau)``
        so the BCE loss can downweight ambiguous tokens (net ≈ 0)
        relative to clear wins/losses.
        """
        net = utility.float() - float(self.read_price)
        target = torch.sigmoid(net / self.tau).clamp(0.05, 0.95)
        confidence = torch.tanh(net.abs() / self.tau)
        return target.detach(), confidence.detach()

    # Spec wording in the design update calls this method
    # ``make_controller_target``; the alias keeps both names live so
    # the wiring task can use either form.
    make_controller_target = controller_target

    @torch.no_grad()
    def write_target(
        self, write_utility: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Same shape as :meth:`controller_target` but uses the write
        shadow price. Used by the write-admission head, not the read
        controller.
        """
        net = write_utility.float() - float(self.write_price)
        target = torch.sigmoid(net / self.tau).clamp(0.05, 0.95)
        confidence = torch.tanh(net.abs() / self.tau)
        return target.detach(), confidence.detach()

    @torch.no_grad()
    def dual_step(
        self,
        *,
        actual_read_rate: float,
        actual_write_rate: float | None = None,
    ) -> None:
        """Single primal-dual update of the shadow prices.

        ``actual_read_rate`` is the empirical read rate this step
        (controller mean over the batch). The EMA tracks it; the price
        is nudged by ``dual_lr * (ema - target)`` and clamped to
        ``[0, max_price]``.
        """
        self.read_rate_ema = (
            self.ema_beta * self.read_rate_ema
            + (1.0 - self.ema_beta) * float(actual_read_rate)
        )
        read_error = self.read_rate_ema - self.target_read_rate
        new_read_price = self.read_price + self.dual_lr * read_error
        self.read_price = float(min(max(new_read_price, 0.0), self.max_price))

        if actual_write_rate is not None:
            self.write_rate_ema = (
                self.ema_beta * self.write_rate_ema
                + (1.0 - self.ema_beta) * float(actual_write_rate)
            )
            write_error = self.write_rate_ema - self.target_write_rate
            new_write_price = self.write_price + self.dual_lr * write_error
            self.write_price = float(
                min(max(new_write_price, 0.0), self.max_price)
            )


# ---------------------------------------------------------------------------
# Per-entry credit assignment.
# ---------------------------------------------------------------------------


@torch.no_grad()
def assign_memory_credit(
    entry_credit: torch.Tensor,
    entry_debit: torch.Tensor,
    entry_ids: torch.Tensor,
    weights: torch.Tensor,
    utility: torch.Tensor,
) -> None:
    """Accumulate per-entry credit and debit from a batch of utility signals.

    ``entry_ids`` and ``weights`` are ``(B, T, K)`` — for each target
    token the controller picked ``K`` cache entries and routed them
    through the encoder with attention weights ``weights``. ``utility``
    is ``(B, T)``. Positive utility flows to ``entry_credit``;
    negative utility flows to ``entry_debit``. Both accumulators are
    1-D tensors of length ``num_entries`` and are mutated in place.
    """
    pos = torch.relu(utility).unsqueeze(-1).float()
    neg = torch.relu(-utility).unsqueeze(-1).float()
    weights_f = weights.float()
    credit = (weights_f * pos).reshape(-1)
    debit = (weights_f * neg).reshape(-1)
    flat_ids = entry_ids.reshape(-1).long()
    entry_credit.scatter_add_(0, flat_ids, credit)
    entry_debit.scatter_add_(0, flat_ids, debit)


# ---------------------------------------------------------------------------
# Top-level rank-3 scoring entry point.
# ---------------------------------------------------------------------------


def _autocast_for(device_type: str) -> Any:
    if device_type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if device_type == "cpu":
        # CPU autocast is supported but is a no-op for most ops we use;
        # entering the context is still cheaper than branching at every
        # call site and keeps semantics aligned with the GPU path.
        return torch.autocast("cpu", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@torch.inference_mode()
def rank3_score_batch_causal(
    *,
    model: Any,
    cache: Any,
    input_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    scarcity_optimizer: ScarcityAwareMemoryOptimizer | None = None,
    tau: float = 0.10,
    strength: float = 0.10,
    w_max: float = 1.15,
) -> dict[str, torch.Tensor]:
    """Score a batch by comparing memory-on vs memory-off NLL.

    The cache transaction wraps the whole compare so both encode passes
    see the same read-cutoff snapshot — utility deltas are not poisoned
    by mid-batch cache writes from peer ranks.

    Returns a dict with:

    * ``utility``: ``(B, T-1)`` per-token NLL_off − NLL_mem (zeroed at
      invalid positions).
    * ``controller_target``: ``(B, T-1)`` clamped probability for the
      controller BCE head.
    * ``confidence``: ``(B, T-1)`` ``tanh(|net|/tau)`` so the
      controller loss can de-emphasise ambiguous tokens.
    * ``loss_weight``: ``(B, T-1)`` mean-1 LM-loss reweighting that
      never goes below 1.0 in the raw form.
    """
    txn = cache.begin_batch()
    x = input_ids[:, :-1]
    y = input_ids[:, 1:]
    mask = valid_mask[:, 1:].bool()

    with _autocast_for(input_ids.device.type):
        h_off = model.encode(
            x, memory_mode="off", cache_read_cutoff=txn.read_cutoff
        )
        h_mem = model.encode(
            x, memory_mode="force_on", cache_read_cutoff=txn.read_cutoff
        )

    nll_off = chunked_nll_from_hidden(model, h_off, y)
    nll_mem = chunked_nll_from_hidden(model, h_mem, y)

    utility = (nll_off - nll_mem) * mask.float()

    if scarcity_optimizer is None:
        net = utility.float()
        controller_target = torch.sigmoid(net / tau).clamp(0.05, 0.95)
        confidence = torch.tanh(net.abs() / tau)
    else:
        controller_target, confidence = scarcity_optimizer.controller_target(
            utility
        )

    # Mask out invalid positions on every per-token output so the wiring
    # task can multiply BCE × confidence without leaking gradient through
    # padding (and so utility, target, weight all share one truth).
    mask_f = mask.float()
    controller_target = controller_target * mask_f
    confidence = confidence * mask_f

    loss_weight = positive_only_lm_weight(
        utility, mask, tau=tau, strength=strength, w_max=w_max
    )

    cache.commit(txn)
    return {
        "utility": utility,
        "controller_target": controller_target,
        "confidence": confidence,
        "loss_weight": loss_weight,
    }
