from __future__ import annotations
import torch
import torch.nn as nn
from typing import Callable

from chaoscontrol.optim.episodic_cache import CacheEntry, EpisodicCache
from chaoscontrol.optim.episodic_writer import fingerprint_tokens


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

    Cache-aware mode (optional, opt-in via ``cache=`` constructor arg):
      When an ``EpisodicCache`` is passed, ``score_chunk`` additionally
      computes a fingerprint over the chunk's last ``fingerprint_window``
      tokens and queries the cache for matching entries — query-only, no
      weight mutation. Hits are stashed on ``self._pending_cache_hits`` and
      consumed by the next ``adapt_on_chunk`` call as extra replay-driven
      gradient steps. The no-cache path (``cache=None``, default) is
      bit-identical to the pre-cache controller.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        loss_fn: Callable,
        leak_detection: bool = False,
        cache: EpisodicCache | None = None,
        fingerprint_window: int = 4,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.leak_detection = leak_detection
        self.cache = cache
        self.fingerprint_window = int(fingerprint_window)
        self._adapted_chunks: set[bytes] = set()
        # Hits stashed by score_chunk for the immediately-following adapt.
        # mark_new_epoch() drains this between docs so retrieval bias from
        # one doc never crosses into the next.
        self._pending_cache_hits: list[CacheEntry] = []

    @staticmethod
    def _chunk_hash(chunk: torch.Tensor) -> bytes:
        # Stable hash across processes and PYTHONHASHSEED values. Python's
        # builtin `hash(bytes)` is randomized per-process — fine for the
        # in-run set, but we want reproducible diagnostics across launches.
        import hashlib
        return hashlib.blake2b(
            chunk.detach().cpu().numpy().tobytes(), digest_size=8
        ).digest()

    def _query_cache_for_chunk(self, chunk: torch.Tensor) -> list[CacheEntry]:
        """Compute the chunk-tail fingerprint and look up matching entries.

        Phase 1 retrieval: top-1 exact match on the trailing
        ``fingerprint_window`` tokens of the chunk's first row. Returns an
        empty list when the chunk is shorter than the fingerprint window
        (no fingerprint defined) or when the cache has no hit.
        """
        if self.cache is None or self.fingerprint_window <= 0:
            return []
        if chunk.dim() != 2 or chunk.size(1) < self.fingerprint_window:
            return []
        # Single-row contract for the eval driver. Multi-row would require a
        # batched fingerprint helper — out of scope until the runner emits
        # multi-doc batches.
        fp_tokens = chunk[0, -self.fingerprint_window:]
        fp = fingerprint_tokens(fp_tokens)
        entry = self.cache.query(int(fp))
        return [entry] if entry is not None else []

    def score_chunk(
        self,
        chunk: torch.Tensor,
        *,
        initial_states: list[torch.Tensor] | None = None,
    ) -> tuple[float, list[torch.Tensor]]:
        """Forward-only score under current weights.

        Returns ``(loss, final_states)``. ``final_states`` is the list of
        per-physical-block recurrent states at the end of the chunk, suitable
        for threading into the next chunk via ``persistence_mode=carry_state``.
        Returns ``[]`` when the model's forward doesn't expose ``final_states``
        (non-SSM blocks).

        Empty-CE guard: CE needs at least one target token, i.e. chunk length
        >= 2 after shifting. Callers must have already filtered these, but a
        silent NaN here would poison the stability gate.

        Cache-aware: when ``self.cache`` is set, the chunk's tail fingerprint
        is queried and any hit is stashed on ``self._pending_cache_hits`` for
        the next ``adapt_on_chunk`` to replay. The forward+loss path is
        unchanged — query-only, no weight mutation.
        """
        if chunk.size(1) < 2:
            raise ValueError(
                f"score_chunk needs chunk length >= 2 for teacher-forcing CE; "
                f"got shape {tuple(chunk.shape)}."
            )
        if self.leak_detection:
            h = self._chunk_hash(chunk)
            if h in self._adapted_chunks:
                raise LeakDetectedError(
                    f"Chunk hash {h.hex()} was adapt_on_chunk'd before score_chunk: "
                    "Issue #1017 violation."
                )
        self.model.eval()
        # StateManager returns [] before start_doc; pass None rather than an
        # empty list so the model's length-mismatch guard doesn't misfire.
        kwargs: dict = {}
        if initial_states:
            kwargs["initial_states"] = initial_states
        with torch.no_grad():
            out = self.model(chunk, **kwargs)
            logits = out["logits"] if isinstance(out, dict) else out
            loss = self.loss_fn(logits[:, :-1], chunk[:, 1:])
            final_states = (
                list(out["final_states"])
                if isinstance(out, dict) and "final_states" in out
                else []
            )
        # Cache query AFTER the forward+loss so any exception during scoring
        # can't leave stale hits stashed on the controller. Always overwrite —
        # an empty result on a miss must clear prior hits, otherwise the
        # next adapt would replay the *previous* chunk's stashed slot.
        if self.cache is not None:
            self._pending_cache_hits = self._query_cache_for_chunk(chunk)
        return float(loss.item()), final_states

    def adapt_on_chunk(
        self,
        chunk: torch.Tensor,
        *,
        optimizer,
        steps: int = 1,
        initial_states: list[torch.Tensor] | None = None,
        cache_replay_steps: int = 1,
    ) -> float | None:
        """Runs ``steps`` gradient updates on ``chunk``.

        ``initial_states`` must match the state context the chunk was just
        scored under. For ``carry_state`` modes that keeps the adaptation pass
        legally aligned with the preceding score-before-update forward instead
        of silently resetting to zeros.

        ``cache_replay_steps``: when the cache is enabled and the preceding
        score_chunk stashed hits on ``_pending_cache_hits``, run up to this
        many additional optimizer steps on the cached value spans. Hits are
        consumed (drained) regardless of step count, so a missed-hit chunk
        and a 0-replay-step chunk both leave the next adapt with no carryover.
        """
        if steps <= 0 and not self._pending_cache_hits:
            return None
        if self.leak_detection:
            h = self._chunk_hash(chunk)
            self._adapted_chunks.add(h)
        self.model.train()
        final_loss = None
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            kwargs: dict = {}
            if initial_states:
                kwargs["initial_states"] = initial_states
            out = self.model(chunk, **kwargs)
            logits = out["logits"] if isinstance(out, dict) else out
            loss = self.loss_fn(logits[:, :-1], chunk[:, 1:])
            loss.backward()
            optimizer.step()
            final_loss = loss.detach()
        # Cache replay AFTER the chunk-driven steps so the chunk-CE update
        # lands first; replay then biases the result toward retrieved spans.
        # ``_replay_from_cache_at_eval`` takes one slot per call; consume up
        # to ``cache_replay_steps`` hits this turn and leave the rest (none,
        # in the Phase 1 top-1 design) for next chunk.
        #
        # The return value of adapt_on_chunk is the *chunk-CE* loss from the
        # last chunk-driven step. Replay losses are bonus work whose units
        # (CE on the value span) don't compose with the chunk-CE loss the
        # MetricsCollector logs as ``loss_after``. Keeping ``final_loss``
        # pointing at the last chunk step preserves the metric's meaning
        # whether or not the cache hit on this chunk.
        if self.cache is not None and self._pending_cache_hits:
            replay_budget = max(0, int(cache_replay_steps))
            consumed = 0
            for hit in list(self._pending_cache_hits):
                if consumed >= replay_budget:
                    break
                _replay_from_cache_at_eval(
                    model=self.model,
                    cache=self.cache,
                    entry=hit,
                    optimizer=optimizer,
                    loss_fn=self.loss_fn,
                )
                consumed += 1
            # Drain unconditionally — partial consumption + leftover hits
            # would replay across chunks and break the score-before-update
            # association the controller is meant to enforce.
            self._pending_cache_hits = []
        return None if final_loss is None else float(final_loss.item())

    def mark_new_epoch(self) -> None:
        """Reset chunk-reuse tracking at doc boundary — chunks are per-doc.

        Also drains any pending cache hits so retrieval candidates from the
        previous doc do not leak into the next doc's first adapt step.
        """
        self._adapted_chunks.clear()
        self._pending_cache_hits = []


def _replay_from_cache_at_eval(
    *,
    model: nn.Module,
    cache: EpisodicCache,
    entry: CacheEntry,
    optimizer,
    loss_fn: Callable,
) -> torch.Tensor | None:
    """Replay one cached entry's value-token span as an extra TTT step.

    Eval-time twin of Y's training-time
    ``dreamworld_replay_from_cache_entry`` (lives on
    ``experiments/23_fast_path/dreamworld.py`` on the ``task/Y-replay-path``
    branch). The two helpers should converge after both branches merge.

    Inlined here rather than imported because Y's branch is not yet in
    ``main`` and this worktree must stand alone for the W test suite.
    Behavior contract (matches Y's training-time version intent):
      - Build a teacher-forcing input on ``entry.value_tok_ids`` (need at
        least 2 tokens to score CE; the min span_length=2 in the cache
        constructor enforces this).
      - Forward → CE loss → backward → optimizer.step on the value span.
      - mark_fired(slot, current_step=-1) so the cache records the retrieval
        even though the eval-time controller doesn't carry a step counter.
        Utility-EMA updates remain a future extension; the W task only
        wires the replay path, not the feedback loop.
    Returns the detached loss tensor, or None when the value span is too
    short for teacher forcing.
    """
    value_tokens = entry.value_tok_ids
    if value_tokens.numel() < 2:
        return None
    # Restore device alignment — the cache lives on CPU, the model on CUDA
    # (or CPU for tests). Don't keep cache state in the graph.
    target_device = next(model.parameters()).device
    chunk = value_tokens.detach().to(
        device=target_device, dtype=torch.long,
    ).unsqueeze(0)
    optimizer.zero_grad(set_to_none=True)
    out = model(chunk)
    logits = out["logits"] if isinstance(out, dict) else out
    loss = loss_fn(logits[:, :-1], chunk[:, 1:])
    loss.backward()
    optimizer.step()
    # current_step=-1 because the eval-time controller doesn't hold a
    # step counter today. The cache only uses last_fired_step for grace
    # / eviction, both of which are no-ops in the eval path (cache is
    # not being grown).
    cache.mark_fired(entry.slot, current_step=-1)
    return loss.detach()
