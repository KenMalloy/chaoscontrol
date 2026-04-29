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
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


__all__ = [
    "alpha_ramp",
    "assign_memory_credit",
    "chunked_nll_from_hidden",
    "CpuMemoryScorer",
    "CpuMemoryScorerLanePool",
    "CpuMemoryScorerWeights",
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


_NLL_CHUNK_BUDGET_BYTES = 1 << 30  # 1 GiB peak per-chunk logits


@dataclass
class CpuMemoryScorerTelemetry:
    """Runtime counters for the CPU-resident memory scorer."""

    backend_requested: str
    backend_active: str
    fallback_reason: str = ""
    calls: int = 0
    rows_scored: int = 0
    tokens_scored: int = 0
    seconds_total: float = 0.0
    seconds_last: float = 0.0
    vocab_tile_size: int = 0
    row_chunk_size: int = 0
    weights_version: int = 0
    weights_shared: bool = False

    def as_dict(self) -> dict[str, Any]:
        rows_per_second = (
            self.rows_scored / self.seconds_total
            if self.seconds_total > 0.0
            else 0.0
        )
        return {
            "backend_requested": self.backend_requested,
            "backend_active": self.backend_active,
            "fallback_reason": self.fallback_reason,
            "calls": int(self.calls),
            "rows_scored": int(self.rows_scored),
            "tokens_scored": int(self.tokens_scored),
            "seconds_total": float(self.seconds_total),
            "seconds_last": float(self.seconds_last),
            "rows_per_second": float(rows_per_second),
            "vocab_tile_size": int(self.vocab_tile_size),
            "row_chunk_size": int(self.row_chunk_size),
            "weights_version": int(self.weights_version),
            "weights_shared": bool(self.weights_shared),
        }


@dataclass
class CpuMemoryScorerWeights:
    """Shared CPU-resident scorer weights for all memory lanes.

    This is the controller-plane weight bank: one CPU snapshot, many worker
    views. ``share_memory=True`` promotes the CPU tensors into torch shared
    memory so forked/scoring worker processes can read the same storage instead
    of cloning the LM head per lane.
    """

    norm_weight: torch.Tensor
    lm_head_weight: torch.Tensor
    eps: float = 1e-6
    version: int = 0
    shared: bool = False

    def __post_init__(self) -> None:
        self.norm_weight = self.norm_weight.detach().cpu().contiguous().clone()
        self.lm_head_weight = self.lm_head_weight.detach().cpu().contiguous().clone()
        self.eps = float(self.eps)
        self.version = int(self.version)
        self.shared = bool(self.shared)
        self._amx_packed_head: torch.Tensor | None = None
        self._amx_tile_version: int | None = None
        self._validate()
        if self.shared:
            self.share_memory_()

    def _validate(self) -> None:
        if self.norm_weight.dim() != 1:
            raise ValueError("CpuMemoryScorerWeights norm_weight must be 1D")
        if self.lm_head_weight.dim() != 2:
            raise ValueError("CpuMemoryScorerWeights lm_head_weight must be 2D")
        if self.lm_head_weight.shape[1] != self.norm_weight.shape[0]:
            raise ValueError(
                "CpuMemoryScorerWeights dim mismatch: lm_head_weight.shape[1] "
                f"({self.lm_head_weight.shape[1]}) != norm_weight.shape[0] "
                f"({self.norm_weight.shape[0]})"
            )

    @classmethod
    def from_model(
        cls,
        model: Any,
        *,
        version: int = 0,
        share_memory: bool = True,
    ) -> "CpuMemoryScorerWeights":
        final_norm = getattr(model, "final_norm", None)
        lm_head = getattr(model, "lm_head", None)
        if final_norm is None or lm_head is None:
            raise ValueError(
                "CpuMemoryScorerWeights.from_model requires final_norm and lm_head"
            )
        norm_weight = getattr(final_norm, "weight", None)
        lm_head_weight = getattr(lm_head, "weight", None)
        if norm_weight is None or lm_head_weight is None:
            raise ValueError(
                "CpuMemoryScorerWeights.from_model requires final_norm.weight "
                "and lm_head.weight"
            )
        return cls(
            norm_weight=norm_weight,
            lm_head_weight=lm_head_weight,
            eps=float(getattr(final_norm, "eps", 1e-6)),
            version=int(version),
            shared=bool(share_memory),
        )

    def share_memory_(self) -> "CpuMemoryScorerWeights":
        self.norm_weight.share_memory_()
        self.lm_head_weight.share_memory_()
        self.shared = True
        return self

    @torch.inference_mode()
    def refresh_from_model(self, model: Any, *, version: int | None = None) -> None:
        """Refresh the shared CPU snapshot in place from a live model."""
        final_norm = getattr(model, "final_norm", None)
        lm_head = getattr(model, "lm_head", None)
        if final_norm is None or lm_head is None:
            raise ValueError("refresh_from_model requires final_norm and lm_head")
        self.norm_weight.copy_(final_norm.weight.detach().cpu())
        self.lm_head_weight.copy_(lm_head.weight.detach().cpu())
        self.eps = float(getattr(final_norm, "eps", self.eps))
        self.version = int(self.version + 1 if version is None else version)
        self._amx_packed_head = None
        self._amx_tile_version = None

    def amx_packed_head(self) -> torch.Tensor:
        """Return one VNNI-packed shared LM-head snapshot.

        The C++ scorer consumes the whole head as a single
        ``[D_pad/2, 2*V]`` BF16 VNNI matrix, so every scorer lane reads the
        same CPU storage and the hot path never repacks per vocabulary tile.
        """
        if (
            self._amx_packed_head is not None
            and self._amx_tile_version == self.version
        ):
            return self._amx_packed_head
        from chaoscontrol.kernels import _cpu_ssm_controller as _ext

        weight_t = self.lm_head_weight.to(dtype=torch.bfloat16).t().contiguous()
        dim = int(weight_t.shape[0])
        k_pad = ((dim + 31) // 32) * 32
        if k_pad != dim:
            padded = torch.zeros(
                (k_pad, int(weight_t.shape[1])),
                dtype=torch.bfloat16,
                device=weight_t.device,
            )
            padded[:dim, :] = weight_t
            weight_t = padded
        self._amx_packed_head = _ext.amx_pack_b_vnni(weight_t).contiguous()
        if self.shared:
            self._amx_packed_head.share_memory_()
        self._amx_tile_version = self.version
        return self._amx_packed_head

    def diagnostics(self) -> dict[str, Any]:
        return {
            "version": int(self.version),
            "shared": bool(self.shared),
            "dim": int(self.norm_weight.numel()),
            "vocab": int(self.lm_head_weight.shape[0]),
            "dtype": str(self.lm_head_weight.dtype).replace("torch.", ""),
            "amx_head_prepacked": self._amx_packed_head is not None,
            "amx_packed_shape": (
                []
                if self._amx_packed_head is None
                else [int(x) for x in self._amx_packed_head.shape]
            ),
        }


class CpuMemoryScorer:
    """CPU-resident ``final_norm -> lm_head -> per-token NLL`` scorer.

    The scorer is a lightweight worker view over ``CpuMemoryScorerWeights``.
    The weight bank is the shared CPU snapshot; scorer lanes must not clone the
    LM head privately unless a unit test explicitly constructs private weights.
    ``backend='amx_bf16'`` is fail-loud: if the extension was not built with
    AMX BF16 support or the runtime OS state is unavailable, construction
    raises instead of silently falling back to PyTorch. ``backend='auto'`` is
    explicit telemetry mode: it uses AMX when available and records the reason
    when it must fall back to ``torch_cpu``.
    """

    _VALID_BACKENDS = {"auto", "torch_cpu", "amx_bf16"}

    def __init__(
        self,
        *,
        weights: CpuMemoryScorerWeights | None = None,
        norm_weight: torch.Tensor | None = None,
        lm_head_weight: torch.Tensor | None = None,
        eps: float = 1e-6,
        backend: str = "auto",
        vocab_tile_size: int = 512,
        row_chunk_size: int = 8192,
    ) -> None:
        backend = str(backend)
        if backend not in self._VALID_BACKENDS:
            raise ValueError(
                "CpuMemoryScorer backend must be one of "
                f"{sorted(self._VALID_BACKENDS)}, got {backend!r}"
            )
        if weights is None:
            if norm_weight is None or lm_head_weight is None:
                raise ValueError(
                    "CpuMemoryScorer requires either weights=... or both "
                    "norm_weight=... and lm_head_weight=..."
                )
            weights = CpuMemoryScorerWeights(
                norm_weight=norm_weight,
                lm_head_weight=lm_head_weight,
                eps=float(eps),
                shared=False,
            )
        self.weights = weights
        self.vocab_tile_size = max(1, int(vocab_tile_size))
        self.row_chunk_size = max(1, int(row_chunk_size))

        active, reason = self._resolve_backend(backend)
        self.backend_requested = backend
        self.backend_active = active
        self.telemetry = CpuMemoryScorerTelemetry(
            backend_requested=backend,
            backend_active=active,
            fallback_reason=reason,
            vocab_tile_size=self.vocab_tile_size,
            row_chunk_size=self.row_chunk_size,
            weights_version=int(self.weights.version),
            weights_shared=bool(self.weights.shared),
        )

    @property
    def norm_weight(self) -> torch.Tensor:
        return self.weights.norm_weight

    @property
    def lm_head_weight(self) -> torch.Tensor:
        return self.weights.lm_head_weight

    @property
    def eps(self) -> float:
        return self.weights.eps

    @classmethod
    def from_model(
        cls,
        model: Any,
        *,
        backend: str = "auto",
        vocab_tile_size: int = 512,
        row_chunk_size: int = 8192,
    ) -> "CpuMemoryScorer":
        weights = CpuMemoryScorerWeights.from_model(model, share_memory=True)
        return cls(
            weights=weights,
            backend=backend,
            vocab_tile_size=vocab_tile_size,
            row_chunk_size=row_chunk_size,
        )

    @staticmethod
    def _amx_status() -> tuple[bool, str, Any | None]:
        try:
            from chaoscontrol.kernels import _cpu_ssm_controller as _ext
        except Exception as exc:
            return False, f"extension import failed: {exc.__class__.__name__}: {exc}", None
        try:
            compiled = bool(_ext.amx_bf16_kernel_available())
        except Exception as exc:
            return False, f"AMX availability query failed: {exc}", _ext
        if not compiled:
            return False, "AMX BF16 kernel not compiled into extension", _ext
        try:
            runtime = bool(_ext.has_amx_bf16())
        except Exception as exc:
            return False, f"AMX runtime query failed: {exc}", _ext
        if not runtime:
            return False, "AMX BF16 hardware/OS state unavailable", _ext
        return True, "", _ext

    def _resolve_backend(self, requested: str) -> tuple[str, str]:
        if requested == "torch_cpu":
            return "torch_cpu", ""
        available, reason, _ext = self._amx_status()
        if requested == "amx_bf16":
            if not available:
                raise RuntimeError(
                    "CpuMemoryScorer backend='amx_bf16' requested but AMX is "
                    f"unavailable: {reason}"
                )
            if self.lm_head_weight.shape[1] % 2 != 0:
                raise RuntimeError(
                    "CpuMemoryScorer AMX backend requires even hidden dim for "
                    "BF16 dot pairs"
                )
            return "amx_bf16", ""
        if available and self.lm_head_weight.shape[1] % 2 == 0:
            return "amx_bf16", ""
        fallback = reason or "hidden dim is odd"
        return "torch_cpu", fallback

    def diagnostics(self) -> dict[str, Any]:
        self.telemetry.weights_version = int(self.weights.version)
        self.telemetry.weights_shared = bool(self.weights.shared)
        out = self.telemetry.as_dict()
        out["weights"] = self.weights.diagnostics()
        return out

    @torch.inference_mode()
    def score_hidden(
        self,
        hidden_states: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-token NLL for CPU hidden states.

        This method intentionally rejects CUDA tensors. Moving data across the
        PCIe boundary is a scheduler decision, not a hidden side effect inside
        the scorer.
        """
        if hidden_states.device.type != "cpu" or targets.device.type != "cpu":
            raise ValueError(
                "CpuMemoryScorer.score_hidden expects CPU tensors; move/copy "
                "at the caller boundary so transport cost is visible"
            )
        if hidden_states.dim() != 3:
            raise ValueError("CpuMemoryScorer hidden_states must be (B, T, D)")
        if targets.shape != hidden_states.shape[:2]:
            raise ValueError(
                "CpuMemoryScorer targets must have shape matching hidden_states[:2]"
            )
        if hidden_states.shape[-1] != self.lm_head_weight.shape[1]:
            raise ValueError(
                "CpuMemoryScorer hidden dim mismatch: "
                f"{hidden_states.shape[-1]} != {self.lm_head_weight.shape[1]}"
            )

        t0 = time.perf_counter()
        if self.backend_active == "amx_bf16":
            out = self._score_hidden_amx(hidden_states, targets)
        else:
            out = self._score_hidden_torch(hidden_states, targets)
        elapsed = time.perf_counter() - t0

        rows = int(hidden_states.shape[0] * hidden_states.shape[1])
        self.telemetry.calls += 1
        self.telemetry.rows_scored += rows
        self.telemetry.tokens_scored += rows
        self.telemetry.seconds_total += float(elapsed)
        self.telemetry.seconds_last = float(elapsed)
        return out

    def _score_hidden_torch(
        self,
        hidden_states: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq, dim = hidden_states.shape
        vocab = self.lm_head_weight.shape[0]
        budget_chunk = max(
            1,
            _NLL_CHUNK_BUDGET_BYTES // max(1, int(batch) * int(vocab) * 4),
        )
        effective_chunk = min(int(seq), int(budget_chunk))
        out = torch.empty((batch, seq), dtype=torch.float32)
        weight = self.lm_head_weight
        norm_weight = self.norm_weight
        start = 0
        while start < seq:
            end = min(start + effective_chunk, seq)
            h_chunk = hidden_states[:, start:end, :].to(dtype=weight.dtype)
            normed = F.rms_norm(h_chunk.float(), (dim,), eps=self.eps).to(
                h_chunk.dtype
            )
            normed = normed * norm_weight.to(dtype=normed.dtype)
            if normed.dtype != weight.dtype:
                normed = normed.to(dtype=weight.dtype)
            logits = F.linear(normed, weight).float()
            nll = F.cross_entropy(
                logits.reshape(-1, vocab),
                targets[:, start:end].reshape(-1).long(),
                reduction="none",
            )
            out[:, start:end] = nll.reshape(batch, end - start)
            start = end
        return out

    def _score_hidden_amx(
        self,
        hidden_states: torch.Tensor,
        targets: torch.Tensor,
        *,
        lanes: int = 1,
    ) -> torch.Tensor:
        from chaoscontrol.kernels import _cpu_ssm_controller as _ext

        return _ext.amx_bf16_nll(
            hidden_states.float().contiguous(),
            targets.long().contiguous(),
            self.norm_weight.float().contiguous(),
            self.weights.amx_packed_head(),
            float(self.eps),
            int(self.row_chunk_size),
            int(max(1, lanes)),
        )


class CpuMemoryScorerLanePool:
    """Parallel CPU scorer lanes over one shared scorer weight snapshot."""

    def __init__(
        self,
        *,
        weights: CpuMemoryScorerWeights,
        lanes: int = 8,
        backend: str = "auto",
        vocab_tile_size: int = 512,
        row_chunk_size: int = 8192,
        parallel_threshold_rows: int = 2048,
    ) -> None:
        self.weights = weights
        self.lanes = max(1, int(lanes))
        self.parallel_threshold_rows = max(1, int(parallel_threshold_rows))
        self.scorers = [
            CpuMemoryScorer(
                weights=weights,
                backend=backend,
                vocab_tile_size=vocab_tile_size,
                row_chunk_size=row_chunk_size,
            )
            for _ in range(self.lanes)
        ]
        self._executor: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=self.lanes, thread_name_prefix="cc-cpu-scorer")
            if self.lanes > 1
            else None
        )
        self.calls = 0
        self.parallel_calls = 0
        self.seconds_total = 0.0
        self.seconds_last = 0.0
        self.rows_scored = 0

    @classmethod
    def from_model(
        cls,
        model: Any,
        *,
        lanes: int = 8,
        backend: str = "auto",
        vocab_tile_size: int = 512,
        row_chunk_size: int = 8192,
        parallel_threshold_rows: int = 2048,
    ) -> "CpuMemoryScorerLanePool":
        return cls(
            weights=CpuMemoryScorerWeights.from_model(model, share_memory=True),
            lanes=lanes,
            backend=backend,
            vocab_tile_size=vocab_tile_size,
            row_chunk_size=row_chunk_size,
            parallel_threshold_rows=parallel_threshold_rows,
        )

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def __enter__(self) -> "CpuMemoryScorerLanePool":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @torch.inference_mode()
    def score_hidden(
        self,
        hidden_states: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.device.type != "cpu" or targets.device.type != "cpu":
            raise ValueError("CpuMemoryScorerLanePool expects CPU tensors")
        if hidden_states.dim() != 3:
            raise ValueError("CpuMemoryScorerLanePool hidden_states must be (B, T, D)")
        if targets.shape != hidden_states.shape[:2]:
            raise ValueError("CpuMemoryScorerLanePool targets shape mismatch")

        t0 = time.perf_counter()
        batch, seq, dim = hidden_states.shape
        rows = int(batch * seq)
        self.calls += 1
        self.rows_scored += rows
        if self.scorers[0].backend_active == "amx_bf16":
            out = self.scorers[0]._score_hidden_amx(
                hidden_states,
                targets,
                lanes=self.lanes,
            )
            self.parallel_calls += int(self.lanes > 1 and rows >= self.parallel_threshold_rows)
            rows_by_lane = [0 for _ in range(self.lanes)]
            chunk = (rows + self.lanes - 1) // self.lanes
            for lane_idx in range(self.lanes):
                start = lane_idx * chunk
                end = min(rows, start + chunk)
                if start < rows:
                    rows_by_lane[lane_idx] = end - start
            for scorer, lane_rows in zip(self.scorers, rows_by_lane, strict=False):
                scorer.telemetry.calls += 1 if lane_rows > 0 else 0
                scorer.telemetry.rows_scored += int(lane_rows)
                scorer.telemetry.tokens_scored += int(lane_rows)
        elif (
            self._executor is None
            or rows < self.parallel_threshold_rows
            or self.lanes <= 1
        ):
            out = self.scorers[0].score_hidden(hidden_states, targets)
        else:
            hidden_flat = hidden_states.reshape(rows, dim)
            target_flat = targets.reshape(rows)
            ranges: list[tuple[int, int]] = []
            chunk = (rows + self.lanes - 1) // self.lanes
            for start in range(0, rows, chunk):
                ranges.append((start, min(start + chunk, rows)))

            def _score_range(args: tuple[int, int, int]) -> tuple[int, torch.Tensor]:
                lane_idx, start, end = args
                h = hidden_flat[start:end].reshape(end - start, 1, dim)
                y = target_flat[start:end].reshape(end - start, 1)
                return start, self.scorers[lane_idx].score_hidden(h, y).reshape(-1)

            futures = [
                self._executor.submit(_score_range, (i, start, end))
                for i, (start, end) in enumerate(ranges)
            ]
            out_flat = torch.empty(rows, dtype=torch.float32)
            for fut in futures:
                start, values = fut.result()
                out_flat[start : start + values.numel()] = values
            out = out_flat.reshape(batch, seq)
            self.parallel_calls += 1
        elapsed = time.perf_counter() - t0
        self.seconds_last = float(elapsed)
        self.seconds_total += float(elapsed)
        return out

    def diagnostics(self) -> dict[str, Any]:
        lane_diags = [scorer.diagnostics() for scorer in self.scorers]
        pin_raw = os.environ.get("CHAOSCONTROL_AMX_SCORER_PIN_THREADS", "1")
        return {
            "lanes": int(self.lanes),
            "amx_thread_pinning_enabled": pin_raw not in {"0", "false", "False", "off"},
            "parallel_threshold_rows": int(self.parallel_threshold_rows),
            "calls": int(self.calls),
            "parallel_calls": int(self.parallel_calls),
            "rows_scored": int(self.rows_scored),
            "seconds_total": float(self.seconds_total),
            "seconds_last": float(self.seconds_last),
            "rows_per_second": (
                float(self.rows_scored / self.seconds_total)
                if self.seconds_total > 0.0
                else 0.0
            ),
            "weights": self.weights.diagnostics(),
            "lane_backends": sorted({d["backend_active"] for d in lane_diags}),
            "lane_rows_scored": [int(d["rows_scored"]) for d in lane_diags],
        }


@torch.inference_mode()
def chunked_nll_from_hidden(
    model: Any,
    hidden_states: torch.Tensor,
    targets: torch.Tensor,
    *,
    chunk_size: int = 1024,
    cpu_scorer: CpuMemoryScorer | CpuMemoryScorerLanePool | None = None,
) -> torch.Tensor:
    """Per-token negative log-likelihood ``(B, T)`` from encoder hidden
    states, computed in time-axis chunks to bound peak memory.

    Mirrors the ``final_norm → lm_head → cross_entropy`` ordering that
    ``train_ssm.chunked_lm_head_backward`` uses, but returns the raw
    per-token NLL (``reduction='none'``) instead of a scalar loss —
    rank-3 scoring needs the per-position signal so it can compute
    utility deltas pointwise.

    ``chunk_size`` is clamped against ``_NLL_CHUNK_BUDGET_BYTES`` so the
    per-chunk allocation ``batch * chunk_size * vocab * 4`` (fp32 logits)
    stays bounded regardless of the value a caller passes; otherwise an
    over-large ``chunk_size`` (or one that exceeds ``seq`` and skips
    chunking entirely) materialises the full logits tensor in one shot.
    """
    if chunk_size <= 0:
        raise ValueError(
            f"chunked_nll_from_hidden: chunk_size must be positive, got {chunk_size}"
        )
    if cpu_scorer is not None:
        return cpu_scorer.score_hidden(hidden_states, targets)
    batch, seq, _ = hidden_states.shape
    final_norm = model.final_norm
    lm_head = model.lm_head
    vocab = lm_head.out_features

    budget_chunk = max(
        1,
        _NLL_CHUNK_BUDGET_BYTES // max(1, int(batch) * int(vocab) * 4),
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
            if gradient_conflict_monitor is not None:
                out["gradient_conflict"] = torch.zeros_like(utility)
                out["write_score"] = write_score.detach()
            return out
        event_ids = None
        reserve_event_ids = getattr(cache, "reserve_event_ids", None)
        if callable(reserve_event_ids):
            event_ids = reserve_event_ids(
                int(h_off.shape[0] * h_off.shape[1]),
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

    cache.commit(txn)
    return {
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
