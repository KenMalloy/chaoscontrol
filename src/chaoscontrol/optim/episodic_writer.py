"""Write-trigger logic for the episodic cache memory subsystem.

Component 2 of the build order. Selects positions in a training batch that
deserve a cache write, then builds the write payloads. Pure functions
operating on CPU/GPU tensors — no runner / IPC plumbing here. The runner
calls ``select_writes`` once per training step with the signals it has
already computed (per-token CE, ScOpt pressure, hidden states), and gets
back a list of payloads ready to enqueue for the CPU controller.

Selection rule (Q3 of the design proposal): the write signal is
``pressure × per_token_ce``. Positions in the top ``top_p`` quantile of
the per-batch distribution get written. Quantile thresholding adapts to
absolute scale across training and gives a predictable write rate.

Payload contents (Q2):

  - ``key_fp``: rolling fingerprint of the ``fingerprint_window`` input
    tokens preceding the write position
  - ``key_rep``: a projected hidden-state vector (the runner supplies a
    pre-computed projection; this module just extracts the slice)
  - ``value_tok_ids``: the next ``span_length`` target tokens
  - ``value_anchor_id``: the target token at the write position

Edge handling: if a candidate position doesn't have ``fingerprint_window``
preceding tokens or doesn't have a full ``span_length`` window of
following tokens, it is dropped silently. Boundary writes would carry
incomplete payloads and skew the cache; cleanest to skip them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Sequence

import torch


# Polynomial-rolling-hash modulus. 2^61 - 1 is a Mersenne prime; the modular
# multiply stays in int64 with no overflow for window sizes up to ~30 tokens
# at vocab sizes up to 32 bits.
_FINGERPRINT_MODULUS = (1 << 61) - 1
_FINGERPRINT_BASE = 1_000_003

# Per-rank monotonically-increasing sequence used to mint candidate_id for the
# admission trace (Phase D1 of the CPU SSM controller plan). The sequence is
# module-level rather than thread-local because the runner's writer call sits
# inside a single training step on each rank's process; cross-rank uniqueness
# is supplied by packing source_rank into the high byte of candidate_id. The
# trained controller (Phase B1) will replace this Python-side counter with a
# proper rank_seq driven from the runner; for D1 the trace just needs unique
# rank-local ids so downstream pretrain joins behave.
_admission_trace_seq: int = 0


def _reset_admission_trace_seq() -> None:
    """Test-only hook to zero the admission-trace counter."""
    global _admission_trace_seq
    _admission_trace_seq = 0


def _next_admission_trace_seq() -> int:
    global _admission_trace_seq
    seq = _admission_trace_seq
    _admission_trace_seq += 1
    return seq


@dataclass(frozen=True)
class WritePayload:
    """A single ready-to-enqueue write decision."""

    batch_index: int
    position: int
    key_fp: int
    key_rep: torch.Tensor  # [key_rep_dim] float32
    value_tok_ids: torch.Tensor  # [span_length] int64
    value_anchor_id: int


def fingerprint_tokens(tokens: torch.Tensor) -> int:
    """Stable polynomial rolling hash over a 1-D sequence of token IDs.

    The output is a non-negative int64 in ``[0, _FINGERPRINT_MODULUS)``.
    Identical inputs produce identical fingerprints across calls and
    processes (no Python ``hash()`` randomization).
    """
    if tokens.dim() != 1:
        raise ValueError(
            f"fingerprint_tokens expects a 1-D tensor; got shape "
            f"{tuple(tokens.shape)}"
        )
    h = 0
    base = _FINGERPRINT_BASE
    mod = _FINGERPRINT_MODULUS
    for tok in tokens.detach().to(dtype=torch.int64).cpu().tolist():
        h = (h * base + int(tok) + 1) % mod
    return int(h)


def fingerprint_tokens_batch(tokens: torch.Tensor) -> list[int]:
    """Vectorized rolling hash over ``[N, W]`` token sequences. Returns a
    Python list of ``N`` int64 fingerprints. Used by the runner to compute
    fingerprints for all candidate positions in one shot.
    """
    if tokens.dim() != 2:
        raise ValueError(
            f"fingerprint_tokens_batch expects 2-D [N, W]; got shape "
            f"{tuple(tokens.shape)}"
        )
    N, _W = tokens.shape
    return [fingerprint_tokens(tokens[i]) for i in range(N)]


def select_top_p_positions(
    write_signal: torch.Tensor, top_p: float,
) -> torch.Tensor:
    """Pick the top ``top_p`` fraction of positions by ``write_signal``.

    Args:
        write_signal: ``[B, T]`` non-negative scalar per position.
            Typically ``pressure × per_token_ce``. NaN/inf-free.
        top_p: target fraction of positions to keep, in (0, 1].

    Returns:
        ``[K, 2]`` int64 tensor of (batch_index, position) pairs, where
        ``K = max(1, round(B * T * top_p))``. Returned in descending
        signal order; ties broken arbitrarily by torch.topk's tie rule.
    """
    if write_signal.dim() != 2:
        raise ValueError(
            f"write_signal must be 2-D [B, T]; got shape "
            f"{tuple(write_signal.shape)}"
        )
    if not 0.0 < top_p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1]; got {top_p}")
    B, T = write_signal.shape
    total = B * T
    k = max(1, int(round(total * float(top_p))))
    flat = write_signal.reshape(-1)
    _, flat_idx = torch.topk(flat, k=k, largest=True)
    rows = (flat_idx // T).to(dtype=torch.int64)
    cols = (flat_idx % T).to(dtype=torch.int64)
    return torch.stack([rows, cols], dim=1)


def build_write_payload(
    *,
    batch_index: int,
    position: int,
    input_ids: torch.Tensor,  # [B, T] int64
    target_ids: torch.Tensor,  # [B, T] int64
    key_rep_per_position: torch.Tensor,  # [B, T, key_rep_dim] float32
    fingerprint_window: int,
    span_length: int,
) -> WritePayload | None:
    """Construct a single write payload for one (b, t) position. Returns
    None if the position is too close to either boundary to support a
    full fingerprint window or a full target span.

    Boundary rule:
      - need ``position >= fingerprint_window`` so the rolling hash sees
        a complete preceding window
      - need ``position + span_length <= T`` so the target span is full

    Both bounds must hold; otherwise the position is silently skipped.
    """
    B, T = input_ids.shape
    if not 0 <= batch_index < B:
        raise IndexError(
            f"batch_index {batch_index} out of range for B={B}"
        )
    if not 0 <= position < T:
        raise IndexError(
            f"position {position} out of range for T={T}"
        )
    if position < fingerprint_window:
        return None
    if position + span_length > T:
        return None

    fp_window = input_ids[
        batch_index, position - fingerprint_window:position
    ]
    key_fp = fingerprint_tokens(fp_window)
    span = target_ids[batch_index, position:position + span_length].clone()
    anchor = int(target_ids[batch_index, position].item())
    rep = key_rep_per_position[batch_index, position].clone()
    return WritePayload(
        batch_index=int(batch_index),
        position=int(position),
        key_fp=int(key_fp),
        key_rep=rep,
        value_tok_ids=span.to(dtype=torch.int64),
        value_anchor_id=anchor,
    )


def select_writes(
    *,
    input_ids: torch.Tensor,  # [B, T] int64
    target_ids: torch.Tensor,  # [B, T] int64
    pressure: torch.Tensor,  # [B, T] float32 (ScOpt or equivalent)
    per_token_ce: torch.Tensor,  # [B, T] float32 (per-token cross-entropy)
    key_rep_per_position: torch.Tensor,  # [B, T, key_rep_dim] float32
    top_p: float,
    fingerprint_window: int,
    span_length: int,
    source_rank: int = 0,
    gpu_step: int = 0,
    write_bucket: int = 0,
    admission_trace_path: str | None = None,
) -> list[WritePayload]:
    """Select and build write payloads for one training step.

    Top-level entry point: the runner passes its already-computed
    per-token signals; this function returns a list of
    ``WritePayload``s ready to enqueue for the CPU controller. Boundary
    positions are dropped, so the returned list may be shorter than
    ``top_p × B × T``.

    Order of returned payloads is descending write_signal (highest
    surprise first), so a downstream queue with limited capacity will
    keep the most informative writes if it has to drop on overflow.

    Optional admission-trace logging (Phase D1 of the CPU SSM controller
    plan): when ``admission_trace_path`` is set, every top-K candidate —
    admit AND boundary-reject — is appended as one NDJSON row matching
    the WRITE_EVENT wire-schema column order. The trace is a side-effect
    only; the returned payload list is bit-identical to a call with the
    path unset. ``source_rank``, ``gpu_step`` and ``write_bucket`` are
    plumbed through onto each row so the offline pretrain pipeline can
    join admissions back to the WRITE_EVENT ring (once Phase B1 lands)
    via ``candidate_id``. The first call inside a process should be
    preceded by ``_reset_admission_trace_seq()`` if the runner wants to
    pin the rank-local sequence to its own counter; otherwise the
    sequence simply continues across calls.
    """
    if pressure.shape != per_token_ce.shape:
        raise ValueError(
            f"pressure {tuple(pressure.shape)} and per_token_ce "
            f"{tuple(per_token_ce.shape)} must match"
        )
    if pressure.shape != input_ids.shape:
        raise ValueError(
            f"pressure {tuple(pressure.shape)} must match input_ids "
            f"{tuple(input_ids.shape)}"
        )
    if target_ids.shape != input_ids.shape:
        raise ValueError(
            f"target_ids {tuple(target_ids.shape)} must match input_ids "
            f"{tuple(input_ids.shape)}"
        )
    if key_rep_per_position.dim() != 3:
        raise ValueError(
            f"key_rep_per_position must be 3-D [B, T, D]; got shape "
            f"{tuple(key_rep_per_position.shape)}"
        )
    if key_rep_per_position.shape[:2] != input_ids.shape:
        raise ValueError(
            f"key_rep_per_position [B, T] {tuple(key_rep_per_position.shape[:2])} "
            f"must match input_ids {tuple(input_ids.shape)}"
        )
    if not 0 <= int(source_rank) < 256:
        raise ValueError(
            f"source_rank must fit in u8; got {source_rank}"
        )

    write_signal = pressure.detach().to(dtype=torch.float32) * per_token_ce.detach().to(
        dtype=torch.float32
    )
    positions = select_top_p_positions(write_signal, top_p=top_p)
    payloads: list[WritePayload] = []
    trace_rows: list[dict] = [] if admission_trace_path is not None else []
    for i in range(positions.shape[0]):
        b = int(positions[i, 0].item())
        t = int(positions[i, 1].item())
        payload = build_write_payload(
            batch_index=b,
            position=t,
            input_ids=input_ids,
            target_ids=target_ids,
            key_rep_per_position=key_rep_per_position,
            fingerprint_window=fingerprint_window,
            span_length=span_length,
        )
        if payload is not None:
            payloads.append(payload)
        if admission_trace_path is not None:
            # Reject path: ``payload is None`` means the candidate was
            # boundary-trimmed — no fingerprint window or no full target
            # span. The signal slice is still meaningful (pressure, CE,
            # key_rep, anchor are all valid at every position), so we
            # log them. ``key_fp`` is set to 0 on rejects because the
            # rolling hash needs a complete preceding window; downstream
            # consumers should filter on ``decision`` before joining on
            # key_fp.
            if payload is not None:
                key_fp = int(payload.key_fp)
                anchor_id = int(payload.value_anchor_id)
            else:
                key_fp = 0
                anchor_id = int(target_ids[b, t].item())
            key_rep_slice = key_rep_per_position[b, t]
            l2 = float(torch.linalg.vector_norm(key_rep_slice).item())
            rank_seq = _next_admission_trace_seq()
            candidate_id = (int(source_rank) << 56) | (
                rank_seq & ((1 << 56) - 1)
            )
            trace_rows.append({
                "candidate_id": candidate_id,
                "decision": 1 if payload is not None else 0,
                "gpu_step": int(gpu_step),
                "source_rank": int(source_rank),
                "key_fp": key_fp,
                "key_rep_l2": l2,
                "value_anchor_id": anchor_id,
                "pressure_at_write": float(pressure[b, t].item()),
                "pre_write_ce": float(per_token_ce[b, t].item()),
                "write_bucket": int(write_bucket),
            })

    if admission_trace_path is not None and trace_rows:
        with open(admission_trace_path, "a") as fh:
            for row in trace_rows:
                fh.write(json.dumps(row, separators=(",", ":")))
                fh.write("\n")

    return payloads


def fingerprints_match(a: Sequence[int], b: Sequence[int]) -> bool:
    """Convenience for tests / debug — element-wise int equality."""
    return list(a) == list(b)
