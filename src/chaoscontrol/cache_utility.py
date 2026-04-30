"""Rank-3 oracle utility scoring for the CRCT (Cache-Reweighted
Continuation Training) architecture.

This module computes a per-token utility signal — NLL without episodic
memory minus NLL with episodic memory — and converts it into:

* a controller-head probability target (with optional scarcity-aware
  shadow pricing on the read budget),
* a positive-only language-model loss reweighting, and
* per-entry credit/debit accumulators for memory housekeeping.

The whole module runs on rank 3, which is otherwise idle during training.
It uses ``model.encode(memory_mode=..., cache_read_cutoff=...)`` and a
transactional cache with a monotone event-id clock so scoring sees a stable
memory snapshot and same-batch writes become visible only after scoring.
"""
from __future__ import annotations

import contextlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


__all__ = [
    "alpha_ramp",
    "assign_memory_credit",
    "chunked_nll_from_hidden",
    "plasticity_budget_from_hidden_delta",
    "positive_only_lm_weight",
    "rank3_score_batch_causal",
    "ScarcityAwareMemoryOptimizer",
    "CrctGradientConflictMonitor",
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
    ``1 + strength * relu(tanh(utility / tau))``. Tokens where memory
    hurt (utility <= 0) bottom out at ``1.0`` exactly, so a neutral
    utility carries no hidden upweight before normalization. The raw
    weight is clamped to ``[1.0, w_max]`` as a safety bound, then mean-1
    normalized over the valid positions so the total gradient magnitude
    across the batch is unchanged.

    Invalid positions (``mask == False``) are set to zero — they do
    not contribute to the mean and do not flow gradients through the
    LM head.
    """
    mask_bool = mask.bool()
    positive_utility = torch.relu(torch.tanh(utility.float() / tau))
    raw = 1.0 + strength * positive_utility
    weights = raw.clamp(min=1.0, max=w_max)

    if mask_bool.any():
        mean_w = weights[mask_bool].mean().clamp(min=1e-8)
        weights = weights / mean_w

    return weights * mask_bool.float()


# ---------------------------------------------------------------------------
# Chunked per-token NLL through the LM head.
# ---------------------------------------------------------------------------


_NLL_CHUNK_BUDGET_BYTES = 1 << 30  # 1 GiB peak per-chunk logits — GPU0-2 default.

# Rank-3 maintenance/oracle paths run on a GPU that does not carry the trunk
# optimizer state, so a larger logits buffer is safely affordable. The 2 GiB
# budget keeps launch overhead manageable at the gathered B≈3072, V=16384 scale
# while leaving plenty of headroom on H100/A100 ranks; callers on those paths
# pass it explicitly via ``chunk_budget_bytes``.
_RANK3_NLL_CHUNK_BUDGET_BYTES = 2 << 30


@torch.inference_mode()
def chunked_nll_from_hidden(
    model: Any,
    hidden_states: torch.Tensor,
    targets: torch.Tensor,
    *,
    chunk_size: int = 1024,
    chunk_budget_bytes: int | None = None,
) -> torch.Tensor:
    """Per-token negative log-likelihood ``(B, T)`` from encoder hidden
    states, computed in time-axis chunks to bound peak memory.

    Mirrors the ``final_norm → lm_head → cross_entropy`` ordering that
    ``train_ssm.chunked_lm_head_backward`` uses, but returns the raw
    per-token NLL (``reduction='none'``) instead of a scalar loss —
    rank-3 scoring needs the per-position signal so it can compute
    utility deltas pointwise.

    ``chunk_size`` is clamped against ``chunk_budget_bytes`` (default
    ``_NLL_CHUNK_BUDGET_BYTES``) so the per-chunk allocation
    ``batch * chunk_size * vocab * 4`` (fp32 logits) stays bounded
    regardless of the value a caller passes; otherwise an over-large
    ``chunk_size`` (or one that exceeds ``seq`` and skips chunking
    entirely) materialises the full logits tensor in one shot. Rank-3
    callers raise the budget to amortise per-chunk launch overhead at
    large gathered batch sizes; GPU0-2 keeps the conservative default.
    """
    if chunk_size <= 0:
        raise ValueError(
            f"chunked_nll_from_hidden: chunk_size must be positive, got {chunk_size}"
        )
    batch, seq, _ = hidden_states.shape
    final_norm = model.final_norm
    lm_head = model.lm_head
    vocab = lm_head.out_features

    budget_bytes = (
        int(chunk_budget_bytes)
        if chunk_budget_bytes is not None
        else _NLL_CHUNK_BUDGET_BYTES
    )
    if budget_bytes <= 0:
        raise ValueError(
            f"chunked_nll_from_hidden: chunk_budget_bytes must be positive, got {budget_bytes}"
        )
    budget_chunk = max(
        1,
        budget_bytes // max(1, int(batch) * int(vocab) * 4),
    )
    effective_chunk = min(int(chunk_size), int(budget_chunk))

    out = hidden_states.new_zeros((batch, seq), dtype=torch.float32)
    start = 0
    while start < seq:
        end = min(start + effective_chunk, seq)
        h_chunk = hidden_states[:, start:end, :]
        head_dtype = lm_head.weight.dtype
        if h_chunk.dtype != head_dtype:
            h_chunk = h_chunk.to(dtype=head_dtype)
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
# Gradient-conflict sensing for write admission.
# ---------------------------------------------------------------------------


class CrctGradientConflictMonitor:
    """Rank-3 write-admission sensor for conflicting memory candidates.

    This is deliberately not a second controller.  It computes a compact
    LM-head-gradient sketch for tokens that would otherwise be written to
    memory, compares each sketch to an EMA of recent accepted sketches,
    and returns an adjusted write score plus diagnostics.  The controller
    and CRCT teacher targets still own normal behavior; the optional hard
    threshold is a circuit breaker for catastrophic anti-alignment.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        ema_beta: float = 0.95,
        catastrophic_threshold: float = -0.90,
        soft_gate_strength: float = 0.0,
        soft_gate_floor: float = 0.05,
        trace_path: str | None = None,
        trace_stride: int = 1,
        trace_max_rows: int = 0,
        trace_flush_rows: int = 256,
        eps: float = 1e-8,
    ) -> None:
        self.enabled = bool(enabled)
        self.ema_beta = float(ema_beta)
        self.catastrophic_threshold = float(catastrophic_threshold)
        self.soft_gate_strength = float(soft_gate_strength)
        self.soft_gate_floor = float(soft_gate_floor)
        self.trace_path = None if trace_path in (None, "") else Path(str(trace_path))
        self.trace_stride = max(1, int(trace_stride))
        self.trace_max_rows = max(0, int(trace_max_rows))
        self.trace_flush_rows = max(1, int(trace_flush_rows))
        self.eps = float(eps)
        self._ema: torch.Tensor | None = None
        self._trace_buffer: list[str] = []
        self._diag: dict[str, Any] = {
            "enabled": self.enabled,
            "ema_beta": self.ema_beta,
            "catastrophic_threshold": self.catastrophic_threshold,
            "soft_gate_strength": self.soft_gate_strength,
            "soft_gate_floor": self.soft_gate_floor,
            "trace_enabled": self.trace_path is not None,
            "trace_path": "" if self.trace_path is None else str(self.trace_path),
            "trace_stride": self.trace_stride,
            "trace_max_rows": self.trace_max_rows,
            "trace_flush_rows": self.trace_flush_rows,
            "trace_rows_written": 0,
            "trace_rows_dropped": 0,
            "trace_rows_buffered": 0,
            "trace_errors": 0,
            "last_trace_error": "",
            "calls": 0,
            "cold_start_calls": 0,
            "candidates_seen": 0,
            "candidates_compared": 0,
            "admitted_candidates": 0,
            "guardrail_suppressed_candidates": 0,
            "soft_gated_candidates": 0,
            "ema_updates": 0,
            "mean_conflict_sum": 0.0,
            "min_conflict": 1.0,
            "max_conflict": -1.0,
            "last_conflict_mean": 0.0,
            "last_conflict_min": 0.0,
            "last_conflict_max": 0.0,
            "last_gate_mean": 1.0,
            "last_suppressed": 0,
            "last_admitted": 0,
            "last_write_token_limit": None,
            "last_reason": "",
        }

    @torch.no_grad()
    def apply_to_write_scores(
        self,
        *,
        model: Any,
        hidden: torch.Tensor,
        targets: torch.Tensor,
        utility: torch.Tensor,
        mask: torch.Tensor,
        max_tokens: int | None,
        step: int | None = None,
    ) -> tuple[torch.Tensor, int | None]:
        """Return ``(write_score, write_token_limit)`` for memory append.

        Only the append-side score is adjusted.  ``utility`` itself,
        controller targets, confidence, and LM loss weights are left alone.
        ``write_token_limit`` can be lower than ``max_tokens`` when the
        hard guardrail suppresses candidates and leaves fewer safe writes.
        """
        if not self.enabled:
            return utility, max_tokens

        self._diag["calls"] += 1
        selected = self._select_candidate_indices(
            utility=utility,
            mask=mask,
            max_tokens=max_tokens,
        )
        n = int(selected.numel())
        self._diag["candidates_seen"] += n
        if n == 0:
            self._diag["last_reason"] = "no_valid_candidates"
            self._diag["last_write_token_limit"] = 0 if max_tokens is not None else None
            return utility, 0 if max_tokens is not None else max_tokens

        sketches = self._lm_head_gradient_sketches(
            model=model,
            hidden=hidden,
            targets=targets,
            selected=selected,
        )
        if sketches.numel() == 0:
            self._diag["last_reason"] = "empty_sketch"
            return utility, max_tokens

        write_score = utility.detach().clone()
        gate = torch.ones(n, device=utility.device, dtype=torch.float32)
        suppressed = torch.zeros(n, device=utility.device, dtype=torch.bool)
        had_reference = self._ema is not None

        if self._ema is None:
            self._diag["cold_start_calls"] += 1
            self._diag["last_reason"] = "cold_start"
            conflict = torch.zeros(n, device=utility.device, dtype=torch.float32)
        else:
            ref = F.normalize(
                self._ema.to(device=sketches.device, dtype=torch.float32),
                dim=0,
                eps=self.eps,
            )
            conflict = (sketches * ref.unsqueeze(0)).sum(dim=-1).clamp(-1.0, 1.0)
            self._diag["candidates_compared"] += n
            suppressed = conflict < self.catastrophic_threshold
            if self.soft_gate_strength > 0.0:
                severity = torch.relu(-conflict).clamp(0.0, 1.0)
                gate = (1.0 - self.soft_gate_strength * severity).clamp(
                    min=self.soft_gate_floor,
                    max=1.0,
                )
                self._diag["soft_gated_candidates"] += int((gate < 1.0).sum().item())
            flat_score = write_score.reshape(-1)
            selected_score = flat_score.index_select(0, selected)
            selected_score = selected_score * gate.to(
                device=selected_score.device,
                dtype=selected_score.dtype,
            )
            selected_score = torch.where(
                suppressed.to(device=selected_score.device),
                torch.full_like(selected_score, -torch.inf),
                selected_score,
            )
            flat_score.index_copy_(0, selected, selected_score)

        admitted_mask = ~suppressed
        admitted = int(admitted_mask.sum().item())
        suppressed_n = int(suppressed.sum().item())
        self._diag["admitted_candidates"] += admitted
        self._diag["guardrail_suppressed_candidates"] += suppressed_n
        self._diag["last_gate_mean"] = float(gate.mean().item())
        self._update_conflict_stats(conflict)
        if admitted > 0:
            self._update_ema(sketches[admitted_mask])
        elif self._ema is None:
            self._update_ema(sketches)

        next_limit = admitted if max_tokens is None else min(int(max_tokens), admitted)
        self._diag["last_write_token_limit"] = next_limit
        self._diag["last_suppressed"] = suppressed_n
        self._diag["last_admitted"] = admitted
        if suppressed_n:
            self._diag["last_reason"] = "guardrail_suppressed"
        elif self.soft_gate_strength > 0.0 and bool((gate < 1.0).any().item()):
            self._diag["last_reason"] = "soft_gated"
        else:
            self._diag["last_reason"] = "observed"
        self._maybe_trace_rows(
            step=step,
            selected=selected,
            targets=targets,
            utility=utility,
            conflict=conflict,
            gate=gate,
            suppressed=suppressed,
            reason=str(self._diag["last_reason"]),
            max_tokens=max_tokens,
            had_reference=had_reference,
        )
        return write_score, next_limit

    def diagnostics(self) -> dict[str, Any]:
        self.flush_trace()
        out = dict(self._diag)
        out["trace_rows_buffered"] = len(self._trace_buffer)
        calls = int(out.get("calls", 0))
        compared = int(out.get("candidates_compared", 0))
        if calls:
            out["mean_conflict_per_call"] = (
                float(out["mean_conflict_sum"]) / float(calls)
            )
        else:
            out["mean_conflict_per_call"] = 0.0
        if compared == 0:
            out["min_conflict"] = 0.0
            out["max_conflict"] = 0.0
        out["has_reference"] = self._ema is not None
        return out

    def flush_trace(self) -> None:
        if self.trace_path is None or not self._trace_buffer:
            return
        try:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            with self.trace_path.open("a", encoding="utf-8") as fh:
                fh.write("".join(self._trace_buffer))
            self._trace_buffer.clear()
            self._diag["trace_rows_buffered"] = 0
        except Exception as exc:  # pragma: no cover - filesystem failures are host-specific
            self._diag["trace_errors"] += 1
            self._diag["last_trace_error"] = f"{type(exc).__name__}: {exc}"

    def _select_candidate_indices(
        self,
        *,
        utility: torch.Tensor,
        mask: torch.Tensor,
        max_tokens: int | None,
    ) -> torch.Tensor:
        flat_mask = mask.reshape(-1).bool()
        valid = torch.nonzero(flat_mask, as_tuple=False).reshape(-1)
        if valid.numel() == 0:
            return valid
        if max_tokens is None or int(max_tokens) <= 0 or int(max_tokens) >= valid.numel():
            return valid
        flat_utility = utility.detach().reshape(-1).float()
        valid_scores = flat_utility.index_select(0, valid)
        local = torch.topk(valid_scores, k=int(max_tokens), largest=True, sorted=False).indices
        return valid.index_select(0, local)

    def _lm_head_gradient_sketches(
        self,
        *,
        model: Any,
        hidden: torch.Tensor,
        targets: torch.Tensor,
        selected: torch.Tensor,
    ) -> torch.Tensor:
        dim = int(hidden.shape[-1])
        h = hidden.detach().reshape(-1, dim).index_select(0, selected)
        y = targets.detach().reshape(-1).index_select(0, selected).long()
        h_norm = model.final_norm(h)
        logits = model.lm_head(h_norm).float()
        probs = torch.softmax(logits, dim=-1)
        probs[torch.arange(probs.shape[0], device=probs.device), y] -= 1.0
        weight = model.lm_head.weight.detach().to(device=probs.device, dtype=torch.float32)
        sketches = probs @ weight
        return F.normalize(sketches, dim=-1, eps=self.eps)

    def _maybe_trace_rows(
        self,
        *,
        step: int | None,
        selected: torch.Tensor,
        targets: torch.Tensor,
        utility: torch.Tensor,
        conflict: torch.Tensor,
        gate: torch.Tensor,
        suppressed: torch.Tensor,
        reason: str,
        max_tokens: int | None,
        had_reference: bool,
    ) -> None:
        if self.trace_path is None:
            return
        call_index = int(self._diag["calls"])
        if (call_index - 1) % self.trace_stride != 0:
            return
        selected_cpu = selected.detach().cpu().tolist()
        conflict_cpu = conflict.detach().cpu().tolist()
        gate_cpu = gate.detach().cpu().tolist()
        suppressed_cpu = suppressed.detach().cpu().tolist()
        utility_flat = utility.detach().reshape(-1).float().cpu()
        targets_flat = targets.detach().reshape(-1).long().cpu()
        seq_len = int(targets.shape[1]) if targets.ndim >= 2 else int(targets.numel())
        for i, flat_idx in enumerate(selected_cpu):
            if self.trace_max_rows > 0 and int(self._diag["trace_rows_written"]) >= self.trace_max_rows:
                self._diag["trace_rows_dropped"] += 1
                continue
            idx = int(flat_idx)
            batch_idx = idx // max(1, seq_len)
            token_pos = idx % max(1, seq_len)
            row = {
                "row_type": "crct_gradient_conflict_candidate",
                "step": None if step is None else int(step),
                "call_index": call_index,
                "candidate_rank": int(i),
                "candidate_flat_index": idx,
                "batch_index": int(batch_idx),
                "token_pos": int(token_pos),
                "token_id": int(targets_flat[idx].item()),
                "utility": float(utility_flat[idx].item()),
                "conflict_cos": float(conflict_cpu[i]),
                "gate": float(gate_cpu[i]),
                "suppressed": bool(suppressed_cpu[i]),
                "reason": reason if bool(suppressed_cpu[i]) else "admitted",
                "max_tokens": None if max_tokens is None else int(max_tokens),
                "catastrophic_threshold": self.catastrophic_threshold,
                "soft_gate_strength": self.soft_gate_strength,
                "has_reference": bool(had_reference),
            }
            self._trace_buffer.append(json.dumps(row, separators=(",", ":")) + "\n")
            self._diag["trace_rows_written"] += 1
        self._diag["trace_rows_buffered"] = len(self._trace_buffer)
        if len(self._trace_buffer) >= self.trace_flush_rows:
            self.flush_trace()

    def _update_ema(self, sketches: torch.Tensor) -> None:
        mean = F.normalize(sketches.float().mean(dim=0), dim=0, eps=self.eps)
        if self._ema is None:
            self._ema = mean.detach().cpu()
        else:
            cur = self._ema.to(device=mean.device, dtype=torch.float32)
            nxt = self.ema_beta * cur + (1.0 - self.ema_beta) * mean
            self._ema = F.normalize(nxt, dim=0, eps=self.eps).detach().cpu()
        self._diag["ema_updates"] += 1

    def _update_conflict_stats(self, conflict: torch.Tensor) -> None:
        if conflict.numel() == 0:
            return
        mean = float(conflict.mean().item())
        cmin = float(conflict.min().item())
        cmax = float(conflict.max().item())
        self._diag["mean_conflict_sum"] += mean
        self._diag["min_conflict"] = min(float(self._diag["min_conflict"]), cmin)
        self._diag["max_conflict"] = max(float(self._diag["max_conflict"]), cmax)
        self._diag["last_conflict_mean"] = mean
        self._diag["last_conflict_min"] = cmin
        self._diag["last_conflict_max"] = cmax


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


def _compact_memory_packet_residual(
    residual: torch.Tensor,
    *,
    batch_size: int,
) -> torch.Tensor:
    """Normalize rank-3 memory packets to the trunk's compact residual lane."""
    if residual.dim() == 2:
        if int(residual.shape[0]) != int(batch_size):
            raise ValueError(
                "memory_residual packet batch mismatch: "
                f"{tuple(residual.shape)} for batch {batch_size}"
            )
        return residual.unsqueeze(1)
    if residual.dim() == 3:
        if int(residual.shape[0]) != int(batch_size) or int(residual.shape[1]) != 1:
            raise ValueError(
                "memory_residual packets must be compact with shape "
                f"(B, D) or (B, 1, D); got {tuple(residual.shape)}"
            )
        return residual
    raise ValueError(
        "memory_residual packets must be compact with shape (B, D) or (B, 1, D); "
        f"got {tuple(residual.shape)}"
    )


def plasticity_budget_from_hidden_delta(
    *,
    h_off: torch.Tensor,
    h_mem: torch.Tensor,
    utility: torch.Tensor,
    mask: torch.Tensor,
    tau: float,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Per-channel evidence that episodic memory is doing useful work.

    ``coverage[c]`` is the signed correlation, over valid batch tokens,
    between residual magnitude ``|h_mem - h_off|[..., c]`` and positive
    memory utility ``relu(nll_off - nll_mem)``.  Positive coverage says
    channel ``c`` moves more when memory actually reduces NLL; negative
    coverage says motion is anti-correlated with help.

    ``confidence[c]`` downweights low-signal batches using the same
    utility scale as the controller target.  ``budget[c]`` is the
    optimizer-facing positive gate: trusted positive coverage only.
    """
    if h_off.shape != h_mem.shape:
        raise ValueError(
            f"h_off and h_mem must have identical shape, got "
            f"{tuple(h_off.shape)} and {tuple(h_mem.shape)}"
        )
    if h_off.dim() != 3:
        raise ValueError(f"hidden tensors must be (B, T, D), got {tuple(h_off.shape)}")
    if utility.shape != h_off.shape[:2] or mask.shape != h_off.shape[:2]:
        raise ValueError(
            "utility and mask must match hidden batch/time shape; got "
            f"utility={tuple(utility.shape)} mask={tuple(mask.shape)} "
            f"hidden={tuple(h_off.shape)}"
        )

    # Mask is 0/1 so ``w == w**2``; multiplying ``x`` by ``w`` once gives a
    # masked-residual tensor we can reuse for every weighted reduction below.
    # That keeps peak fp32 memory at one (B, T, D) buffer instead of the
    # 4-5 large intermediates the centered formulation needs. The fp32
    # subtraction order matches the original implementation so cancellation
    # in tightly coupled hidden states is not amplified.
    w = mask.detach().float()
    n = w.sum().clamp_min(1.0)
    y = torch.relu(utility.detach().float()) * w  # (B, T) — already mask-zeroed.

    x_w = (h_mem.detach().float() - h_off.detach().float()).abs_()
    x_w.mul_(w.unsqueeze(-1))

    # Flattened views for matmul/reduction ops; no extra (B, T, D) allocs.
    x_w_flat = x_w.flatten(0, 1)  # (N, D), view
    y_flat = y.flatten()  # (N,)
    d = int(h_off.shape[-1])
    n_total = int(x_w_flat.shape[0])

    x_sum = x_w_flat.sum(dim=0)  # (D,)
    xy_sum = torch.mv(x_w_flat.t(), y_flat)  # (D,)

    # ``xx_sum[d] = sum_n x_w[n, d]^2``. Done in N-axis chunks so the
    # transient ``x_w[chunk] ** 2`` buffer is bounded (~100 MiB at the
    # default 65 536-row chunk and D ≤ 4096) instead of the 4-5 GiB the
    # elementwise (x.square() * w3) reduction would peak at on the
    # gathered rank-3 batch. Same O(D·N) FLOPs as the original.
    xx_sum = torch.zeros(d, device=x_w_flat.device, dtype=x_w_flat.dtype)
    chunk = 65_536
    for start in range(0, n_total, chunk):
        end = min(n_total, start + chunk)
        x_chunk = x_w_flat[start:end]  # view
        xx_sum.add_(x_chunk.square().sum(dim=0))

    y_sum = y.sum()  # already mask-zeroed.
    yy_sum = (y_flat * y_flat).sum()  # cheap, (N,) ops.

    x_mean = x_sum / n
    y_mean = y_sum / n
    cov = xy_sum / n - x_mean * y_mean
    # Naive form ``E[X^2] - E[X]^2`` can dip slightly negative under
    # rounding; clamp before sqrt.
    x_var = (xx_sum / n - x_mean.square()).clamp_min(0.0)
    y_var = (yy_sum / n - y_mean.square()).clamp_min(0.0)

    coverage = cov / torch.sqrt((x_var * y_var).clamp_min(float(eps)))
    coverage = coverage.clamp(min=-1.0, max=1.0)

    # A high correlation on a batch with no useful memory signal should
    # not open the optimizer gate. The scalar energy term is shared across
    # channels; correlation quality stays per-channel.
    utility_energy = torch.sqrt(yy_sum / n)
    utility_conf = torch.tanh(utility_energy / max(float(tau), float(eps)))
    confidence = coverage.abs() * utility_conf
    confidence = confidence.clamp(min=0.0, max=1.0)
    budget = torch.relu(coverage) * confidence

    return {
        "plasticity_coverage": coverage.detach(),
        "plasticity_confidence": confidence.detach(),
        "plasticity_budget": budget.detach(),
    }


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
    update_model_memory_after: bool = False,
    memory_write_tokens: int | None = None,
    gradient_conflict_monitor: CrctGradientConflictMonitor | None = None,
    step: int | None = None,
    record_stage_seconds: dict[str, float] | None = None,
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

    ``record_stage_seconds``, if provided, is populated with per-stage
    elapsed seconds (``encode_off``, ``encode_force_on``, ``nll_off``,
    ``nll_mem``, ``plasticity``, ``append_memory``). Timing uses CUDA
    events on GPU paths so only one synchronize fires at the end of the
    call — no per-stage stalls when the dict is omitted.
    """
    txn = cache.begin_batch()
    x = input_ids[:, :-1]
    y = input_ids[:, 1:]
    mask = valid_mask[:, 1:].bool()

    use_cuda_timing = (
        record_stage_seconds is not None
        and input_ids.device.type == "cuda"
        and torch.cuda.is_available()
    )
    use_cpu_timing = record_stage_seconds is not None and not use_cuda_timing
    cuda_events: dict[str, torch.cuda.Event] = {}
    cpu_marks: dict[str, float] = {}

    def _mark(label: str) -> None:
        if use_cuda_timing:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            cuda_events[label] = ev
        elif use_cpu_timing:
            cpu_marks[label] = time.perf_counter()

    _mark("start")

    memory_meta: dict[str, Any] | None = None
    with _autocast_for(input_ids.device.type):
        paired_encode = getattr(model, "encode_paired_for_score", None)
        if callable(paired_encode):
            h_off, h_mem, memory_meta = paired_encode(
                x,
                cache_read_cutoff=txn.read_cutoff,
            )
            # The paired path is one recurrent pass over stacked off/on
            # streams. Attribute all encode time to ``encode_off`` and leave
            # ``encode_force_on`` near-zero so existing encode_sum telemetry
            # remains meaningful without inventing a new stage name.
            _mark("after_encode_off")
            _mark("after_encode_force_on")
        else:
            h_off = model.encode(
                x, memory_mode="off", cache_read_cutoff=txn.read_cutoff
            )
            _mark("after_encode_off")
            try:
                h_mem_out = model.encode(
                    x,
                    memory_mode="force_on",
                    cache_read_cutoff=txn.read_cutoff,
                    return_memory_meta=True,
                )
            except TypeError:
                h_mem_out = model.encode(
                    x, memory_mode="force_on", cache_read_cutoff=txn.read_cutoff
                )
            if isinstance(h_mem_out, dict):
                h_mem = h_mem_out["hidden"]
                meta = h_mem_out.get("memory_meta")
                if isinstance(meta, dict):
                    memory_meta = meta
            else:
                h_mem = h_mem_out
            _mark("after_encode_force_on")

    nll_off = chunked_nll_from_hidden(
        model, h_off, y, chunk_budget_bytes=_RANK3_NLL_CHUNK_BUDGET_BYTES
    )
    _mark("after_nll_off")
    nll_mem = chunked_nll_from_hidden(
        model, h_mem, y, chunk_budget_bytes=_RANK3_NLL_CHUNK_BUDGET_BYTES
    )
    _mark("after_nll_mem")

    utility = (nll_off - nll_mem) * mask.float()
    plasticity = plasticity_budget_from_hidden_delta(
        h_off=h_off,
        h_mem=h_mem,
        utility=utility,
        mask=mask,
        tau=float(tau),
    )
    _mark("after_plasticity")

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

    memory_residual = None
    if memory_meta is not None:
        maybe_residual = memory_meta.get("memory_residual")
        if isinstance(maybe_residual, torch.Tensor):
            memory_residual = _compact_memory_packet_residual(
                maybe_residual.detach(),
                batch_size=int(x.shape[0]),
            )

    if update_model_memory_after:
        append_fn = getattr(model, "append_memory_from_hidden", None)
        if append_fn is None:
            raise ValueError(
                "rank3_score_batch_causal(update_model_memory_after=True) "
                "requires model.append_memory_from_hidden(...)"
            )
        write_score = utility.detach()
        write_limit = memory_write_tokens
        if gradient_conflict_monitor is not None:
            write_score, write_limit = gradient_conflict_monitor.apply_to_write_scores(
                model=model,
                hidden=h_off,
                targets=y,
                utility=utility,
                mask=mask,
                max_tokens=memory_write_tokens,
                step=step,
            )
        if write_limit is not None and int(write_limit) <= 0:
            cache.commit(txn)
            out = {
                "utility": utility,
                "controller_target": controller_target,
                "confidence": confidence,
                "loss_weight": loss_weight,
            }
            if memory_residual is not None:
                out["memory_residual"] = memory_residual
                out["memory_gate"] = controller_target.detach()
            out.update(plasticity)
            if gradient_conflict_monitor is not None:
                out["gradient_conflict"] = torch.zeros_like(utility)
                out["write_score"] = write_score.detach()
            _mark("after_append")
            _flush_stage_seconds(
                record_stage_seconds,
                use_cuda_timing=use_cuda_timing,
                use_cpu_timing=use_cpu_timing,
                cuda_events=cuda_events,
                cpu_marks=cpu_marks,
            )
            return out
        event_ids = None
        reserve_event_ids = getattr(cache, "reserve_event_ids", None)
        if callable(reserve_event_ids):
            n_write = int(h_off.shape[0] * h_off.shape[1])
            if write_limit is not None:
                n_write = min(n_write, max(0, int(write_limit)))
            event_ids = reserve_event_ids(
                n_write,
                device=h_off.device,
            )
        append_kwargs = {
            "score": write_score.detach(),
            "max_tokens": write_limit,
            "event_ids": event_ids,
        }
        wrote = bool(append_fn(h_off.detach(), **append_kwargs))
        if not wrote:
            raise ValueError(
                "rank3_score_batch_causal(update_model_memory_after=True) "
                "requires append-only multislot memory; the teacher would "
                "otherwise keep comparing against an empty memory path."
            )
    _mark("after_append")

    cache.commit(txn)
    out = {
        "utility": utility,
        "controller_target": controller_target,
        "confidence": confidence,
        "loss_weight": loss_weight,
        **(
            {
                "write_score": write_score.detach()
                if "write_score" in locals()
                else utility.detach()
            }
            if gradient_conflict_monitor is not None
            else {}
        ),
    }
    if memory_residual is not None:
        out["memory_residual"] = memory_residual
        out["memory_gate"] = controller_target.detach()
    out.update(plasticity)
    _flush_stage_seconds(
        record_stage_seconds,
        use_cuda_timing=use_cuda_timing,
        use_cpu_timing=use_cpu_timing,
        cuda_events=cuda_events,
        cpu_marks=cpu_marks,
    )
    return out


_STAGE_LABELS: tuple[tuple[str, str, str], ...] = (
    ("encode_off", "start", "after_encode_off"),
    ("encode_force_on", "after_encode_off", "after_encode_force_on"),
    ("nll_off", "after_encode_force_on", "after_nll_off"),
    ("nll_mem", "after_nll_off", "after_nll_mem"),
    ("plasticity", "after_nll_mem", "after_plasticity"),
    ("append_memory", "after_plasticity", "after_append"),
)


def _flush_stage_seconds(
    record_stage_seconds: dict[str, float] | None,
    *,
    use_cuda_timing: bool,
    use_cpu_timing: bool,
    cuda_events: dict[str, "torch.cuda.Event"],
    cpu_marks: dict[str, float],
) -> None:
    """Resolve recorded marks into per-stage seconds.

    CUDA events are queued on the same stream as the work, so a single
    ``synchronize()`` here is the only mid-call host stall — and it only
    fires when the caller asked for timing.  CPU-side marks need no sync.
    """
    if record_stage_seconds is None:
        return
    if use_cuda_timing:
        if cuda_events:
            torch.cuda.synchronize()
        for label, start_key, end_key in _STAGE_LABELS:
            start_ev = cuda_events.get(start_key)
            end_ev = cuda_events.get(end_key)
            if start_ev is None or end_ev is None:
                continue
            record_stage_seconds[label] = (
                record_stage_seconds.get(label, 0.0)
                + float(start_ev.elapsed_time(end_ev)) * 1e-3
            )
        return
    if use_cpu_timing:
        for label, start_key, end_key in _STAGE_LABELS:
            t0 = cpu_marks.get(start_key)
            t1 = cpu_marks.get(end_key)
            if t0 is None or t1 is None:
                continue
            record_stage_seconds[label] = (
                record_stage_seconds.get(label, 0.0) + max(0.0, float(t1 - t0))
            )
