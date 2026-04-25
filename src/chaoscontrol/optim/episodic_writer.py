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

from dataclasses import dataclass
from typing import Sequence

import torch


# Polynomial-rolling-hash modulus. 2^61 - 1 is a Mersenne prime; the modular
# multiply stays in int64 with no overflow for window sizes up to ~30 tokens
# at vocab sizes up to 32 bits.
_FINGERPRINT_MODULUS = (1 << 61) - 1
_FINGERPRINT_BASE = 1_000_003


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

    write_signal = pressure.detach().to(dtype=torch.float32) * per_token_ce.detach().to(
        dtype=torch.float32
    )
    positions = select_top_p_positions(write_signal, top_p=top_p)
    payloads: list[WritePayload] = []
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
    return payloads


def fingerprints_match(a: Sequence[int], b: Sequence[int]) -> bool:
    """Convenience for tests / debug — element-wise int equality."""
    return list(a) == list(b)
