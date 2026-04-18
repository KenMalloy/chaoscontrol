from __future__ import annotations
import torch
import torch.nn as nn
from typing import Callable


class LeakDetectedError(RuntimeError):
    pass


class LegalityController:
    """Enforces Issue #1017 score-before-update rule structurally.

    Contract:
      score_chunk(chunk)    : forward-only under current weights; returns scalar loss.
                              Records chunk's token hash in _scored_chunks.
      adapt_on_chunk(chunk) : runs optimizer step(s) on `chunk`. Records chunk's
                              token hash in _adapted_chunks.

    Leak detection (optional, for contract testing):
      If a chunk is score_chunk'd AFTER being adapt_on_chunk'd, LeakDetectedError.
    """

    def __init__(self, model: nn.Module, *, loss_fn: Callable,
                 leak_detection: bool = False):
        self.model = model
        self.loss_fn = loss_fn
        self.leak_detection = leak_detection
        self._adapted_chunks: set[bytes] = set()

    @staticmethod
    def _chunk_hash(chunk: torch.Tensor) -> bytes:
        # Stable hash across processes and PYTHONHASHSEED values. Python's
        # builtin `hash(bytes)` is randomized per-process — fine for the
        # in-run set, but we want reproducible diagnostics across launches.
        import hashlib
        return hashlib.blake2b(
            chunk.detach().cpu().numpy().tobytes(), digest_size=8
        ).digest()

    def score_chunk(self, chunk: torch.Tensor) -> float:
        # Empty-CE guard: CE needs at least one target token, i.e. chunk length
        # >= 2 after shifting. Callers must have already filtered these, but a
        # silent NaN here would poison the stability gate.
        if chunk.size(1) < 2:
            raise ValueError(
                f"score_chunk needs chunk length >= 2 for teacher-forcing CE; "
                f"got shape {tuple(chunk.shape)}."
            )
        h = self._chunk_hash(chunk)
        if self.leak_detection and h in self._adapted_chunks:
            raise LeakDetectedError(
                f"Chunk hash {h.hex()} was adapt_on_chunk'd before score_chunk: "
                "Issue #1017 violation."
            )
        self.model.eval()
        with torch.no_grad():
            out = self.model(chunk)
            logits = out["logits"] if isinstance(out, dict) else out
            loss = self.loss_fn(logits[:, :-1], chunk[:, 1:])
        return float(loss.item())

    def adapt_on_chunk(
        self, chunk: torch.Tensor, *, optimizer, steps: int = 1,
    ) -> float | None:
        """Runs `steps` gradient updates on the chunk. Returns final loss, or None if steps==0."""
        if steps <= 0:
            return None
        h = self._chunk_hash(chunk)
        self._adapted_chunks.add(h)
        self.model.train()
        final_loss = None
        for _ in range(steps):
            optimizer.zero_grad()
            out = self.model(chunk)
            logits = out["logits"] if isinstance(out, dict) else out
            loss = self.loss_fn(logits[:, :-1], chunk[:, 1:])
            loss.backward()
            optimizer.step()
            final_loss = float(loss.item())
        return final_loss

    def mark_new_epoch(self) -> None:
        """Reset chunk-reuse tracking at doc boundary — chunks are per-doc."""
        self._adapted_chunks.clear()
