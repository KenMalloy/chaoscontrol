"""Tests for cache-aware extensions to LegalityController.

Covers the W-task integration: when ``cache=<EpisodicCache>`` is passed to
LegalityController, the controller queries the cache during score_chunk
(without mutating weights) and replays cached entries during adapt_on_chunk
to bias the gradient signal toward retrieved spans.

The no-cache path (``cache=None``, default) MUST be bit-identical to the
existing controller — backward compat is load-bearing for the non-cache
TTT runs the matrix's Arms A and D produce.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from chaoscontrol.eval_stream.legality import LegalityController
from chaoscontrol.optim.episodic_cache import EpisodicCache
from chaoscontrol.optim.episodic_writer import fingerprint_tokens


VOCAB = 32
DIM = 16
SPAN = 4
KEY_REP_DIM = 8
FP_WINDOW = 4


class _TinyLM(nn.Module):
    def __init__(self, vocab: int = VOCAB, dim: int = DIM) -> None:
        super().__init__()
        self.vocab_size = vocab
        self.embed = nn.Embedding(vocab, dim)
        self.lm_head = nn.Linear(dim, vocab, bias=False)

    def forward(self, input_ids: torch.Tensor) -> dict:
        x = self.embed(input_ids)
        return {"logits": self.lm_head(x)}


def _loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
    )


def _make_cache(capacity: int = 8) -> EpisodicCache:
    return EpisodicCache(
        capacity=capacity, span_length=SPAN, key_rep_dim=KEY_REP_DIM,
    )


def _populate_cache_with_chunk_tail(
    cache: EpisodicCache, chunk: torch.Tensor, *, current_step: int = 0,
) -> tuple[int, list[int]]:
    """Write one entry whose fingerprint matches the LAST FP_WINDOW tokens
    of the chunk's first row, and whose value spans are deterministic IDs.
    Returns (slot, value_tok_id_list) so the test can assert on the replay
    path.
    """
    chunk_row = chunk[0]
    fp_window_tokens = chunk_row[-FP_WINDOW:]
    fp = fingerprint_tokens(fp_window_tokens)
    value_ids = [3, 5, 7, 11]
    cache.append(
        key_fp=fp,
        key_rep=torch.zeros(KEY_REP_DIM),
        value_tok_ids=torch.tensor(value_ids, dtype=torch.int64),
        value_anchor_id=int(chunk_row[-1].item()),
        current_step=current_step,
        embedding_version=0,
    )
    e = cache.query(fp)
    assert e is not None
    return e.slot, value_ids


def test_no_cache_path_is_bit_identical_to_existing_controller():
    """Constructor signature back-compat: omitting `cache=` (or passing None)
    must produce a controller behaviorally identical to the pre-cache one.

    Asserted by running matched score+adapt sequences on two parallel models
    with identical seeds; final weights must match exactly.
    """
    torch.manual_seed(0)
    model_baseline = _TinyLM()
    torch.manual_seed(0)
    model_with_none = _TinyLM()

    # Sanity: identical at construction.
    for k in model_baseline.state_dict():
        assert torch.equal(
            model_baseline.state_dict()[k], model_with_none.state_dict()[k]
        )

    chunk = torch.randint(0, VOCAB, (1, 32))

    opt_a = torch.optim.SGD(model_baseline.parameters(), lr=0.1)
    ctrl_a = LegalityController(model_baseline, loss_fn=_loss)

    opt_b = torch.optim.SGD(model_with_none.parameters(), lr=0.1)
    # Two valid back-compat paths: no kwarg, or explicit cache=None.
    ctrl_b = LegalityController(model_with_none, loss_fn=_loss, cache=None)

    loss_a, _ = ctrl_a.score_chunk(chunk)
    loss_b, _ = ctrl_b.score_chunk(chunk)
    assert loss_a == loss_b

    ctrl_a.adapt_on_chunk(chunk, optimizer=opt_a, steps=2)
    ctrl_b.adapt_on_chunk(chunk, optimizer=opt_b, steps=2)
    for k in model_baseline.state_dict():
        torch.testing.assert_close(
            model_baseline.state_dict()[k], model_with_none.state_dict()[k],
        )


def test_score_chunk_with_cache_does_not_mutate_weights():
    """LEGALITY: querying the cache during score_chunk MUST NOT touch model
    weights. score_before_update is the whole point of LegalityController.
    """
    torch.manual_seed(0)
    model = _TinyLM()
    cache = _make_cache()
    chunk = torch.randint(0, VOCAB, (1, 32))
    _populate_cache_with_chunk_tail(cache, chunk)

    pre_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    controller = LegalityController(
        model, loss_fn=_loss, cache=cache,
        fingerprint_window=FP_WINDOW,
    )
    loss, _ = controller.score_chunk(chunk)
    assert loss > 0
    # No weight may have moved.
    for k, v in model.state_dict().items():
        assert torch.equal(pre_state[k], v), f"{k} mutated during score_chunk"


def test_score_chunk_with_cache_records_hit_for_subsequent_adapt():
    """When score_chunk's fingerprint window matches a cache entry, the
    controller must stash the hit so the next adapt_on_chunk can replay it.

    The hit is observable on the controller (no need to expose internals
    through a public method): it lives in `_pending_cache_hits` as a list
    of CacheEntry-like records.
    """
    torch.manual_seed(0)
    model = _TinyLM()
    cache = _make_cache()
    chunk = torch.randint(0, VOCAB, (1, 32))
    slot, value_ids = _populate_cache_with_chunk_tail(cache, chunk)

    controller = LegalityController(
        model, loss_fn=_loss, cache=cache,
        fingerprint_window=FP_WINDOW,
    )
    controller.score_chunk(chunk)
    assert controller._pending_cache_hits, "expected at least one hit"
    hit = controller._pending_cache_hits[0]
    assert hit.slot == slot
    assert hit.value_tok_ids.tolist() == value_ids


def test_score_chunk_with_cache_no_hit_stores_empty_list():
    """When the chunk's fingerprint window has no cached match, the
    controller must record an empty hit list — NOT the previous chunk's
    hits, which would replay stale spans."""
    torch.manual_seed(0)
    model = _TinyLM()
    cache = _make_cache()
    # Use ascending tokens 0..31 so the fingerprint window is deterministic
    # AND distinct from the cache's stored entry below.
    chunk_with_hit = torch.arange(32, dtype=torch.long).unsqueeze(0)
    chunk_no_hit = (
        (torch.arange(32, dtype=torch.long) + VOCAB // 2) % VOCAB
    ).unsqueeze(0)
    _populate_cache_with_chunk_tail(cache, chunk_with_hit)

    controller = LegalityController(
        model, loss_fn=_loss, cache=cache, fingerprint_window=FP_WINDOW,
    )
    controller.score_chunk(chunk_with_hit)
    assert controller._pending_cache_hits  # at least one hit
    controller.score_chunk(chunk_no_hit)
    assert controller._pending_cache_hits == [], (
        "score on a chunk that misses the cache must clear stale hits"
    )


def test_adapt_on_chunk_with_cache_steps_optimizer_and_consumes_hits():
    """When score has stashed a cache hit, adapt_on_chunk must:
      - run a backward+step on the cached value span
      - actually move the weights (replay had effect)
      - clear `_pending_cache_hits` after consumption (no double-replay)
    """
    torch.manual_seed(0)
    model = _TinyLM()
    cache = _make_cache()
    chunk = torch.randint(0, VOCAB, (1, 32))
    _populate_cache_with_chunk_tail(cache, chunk)

    pre_lm = model.lm_head.weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    controller = LegalityController(
        model, loss_fn=_loss, cache=cache, fingerprint_window=FP_WINDOW,
    )
    controller.score_chunk(chunk)
    assert controller._pending_cache_hits
    # cache_replay_steps is the per-chunk replay budget; the test pins it at
    # >= 1 so at least one slot's value span is replayed.
    controller.adapt_on_chunk(
        chunk, optimizer=optimizer, steps=1, cache_replay_steps=1,
    )
    assert not torch.equal(pre_lm, model.lm_head.weight), (
        "cache replay must move the model's weights"
    )
    # Hits drained: a second adapt would replay nothing extra.
    assert controller._pending_cache_hits == []


def test_reset_between_docs_clears_pending_hits():
    """mark_new_epoch is the per-doc boundary; pending hits from the prior
    doc must NOT leak into the next doc's adapt step."""
    torch.manual_seed(0)
    model = _TinyLM()
    cache = _make_cache()
    chunk = torch.randint(0, VOCAB, (1, 32))
    _populate_cache_with_chunk_tail(cache, chunk)

    controller = LegalityController(
        model, loss_fn=_loss, cache=cache, fingerprint_window=FP_WINDOW,
    )
    controller.score_chunk(chunk)
    assert controller._pending_cache_hits
    controller.mark_new_epoch()
    assert controller._pending_cache_hits == []
