"""Phase B5 — true pre/post replay CE pair computation.

The B3 REPLAY_OUTCOME producer cannot fill ``ce_after_replay`` /
``ce_delta_raw`` / ``reward_shaped`` on its own: the helper computes
``replay_loss`` BEFORE ``optimizer.step()``, so every reward field stays
NaN until a second forward runs on the post-step weights. B5 wires that
second forward, gated on the new ``episodic_compute_replay_ce_pair``
flag, and patches the in-place REPLAY_OUTCOME dict the in-step drain
emitted.

Two contracts pin here:

1. **Flag ON** — drain emits NaN reward placeholders, post-step pass
   patches finite values into ``ce_before_replay`` (override =
   helper's ``replay_loss``), ``ce_after_replay`` (post-step CE on the
   same value tokens), ``ce_delta_raw``, ``bucket_baseline``,
   ``reward_shaped``, AND advances the per-bucket EMA on the real delta.
2. **Flag OFF** (B3 default) — bit-identical to the producer test:
   ``ce_after_replay`` stays NaN, EMA never updates, no pending list is
   allocated. Tested in ``tests/test_replay_outcome_producer.py``; this
   file pins the off-path no-op of the post-step helper as a
   defense-in-depth check.

Sanity bound: with a non-trivial gradient (eval_lr scale, weight=1.0,
real backward + step on the same value tokens) and a fresh tiny model,
``ce_delta_raw > 0`` should hold for the majority of replays — the
optimizer step on the same tokens is exactly the descent direction the
post-step CE should drop along. We pin a soft majority bound (>= 60%)
so a single noisy seed doesn't flake; the per-replay magnitudes also
land on the test's printout for diagnostic visibility.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from chaoscontrol.optim.episodic_cache import EpisodicCache

REPO = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO / "experiments" / "23_fast_path" / "runner_fast_path.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("runner_b5", RUNNER_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class _TinyTokenModel(nn.Module):
    """Same surface as ``test_replay_outcome_producer.py``'s helper."""

    def __init__(self, vocab: int = 16, dim: int = 8) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.final_norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab, bias=False)

    def encode(
        self,
        inputs: torch.Tensor,
        *,
        initial_states: list[torch.Tensor] | None = None,
        return_final_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        h = self.embed(inputs.to(torch.long))
        if return_final_states:
            return h, []
        return h


def _consumer(
    mod,
    *,
    event_log_enabled: bool,
    cache: EpisodicCache,
    compute_replay_ce_pair: bool,
):
    return mod._EpisodicConsumerState(
        cache=cache,
        heartbeat=[0],
        controller_query_queue=[],
        episodic_event_log_enabled=event_log_enabled,
        compute_replay_ce_pair=compute_replay_ce_pair,
    )


def _append_slot(
    cache: EpisodicCache,
    *,
    key_fp: int,
    write_bucket: int,
    source_write_id: int,
    span_tokens: list[int] | None = None,
) -> int:
    return cache.append(
        key_fp=key_fp,
        key_rep=torch.zeros(cache.key_rep_dim, dtype=torch.float32),
        value_tok_ids=torch.tensor(
            span_tokens or [3, 5, 7, 9], dtype=torch.int64,
        ),
        value_anchor_id=2,
        current_step=10,
        embedding_version=0,
        pressure_at_write=1.25,
        source_write_id=source_write_id,
        write_bucket=write_bucket,
    )


def _drain(mod, *, consumer, model, weight: float = 1.0) -> int:
    return mod._run_episodic_replay_from_tagged_queue(
        consumer=consumer,
        model=model,
        current_step=17,
        weight=weight,
        lm_head_backward_mode="single",
        lm_head_tile_size=1024,
        logger=None,
        max_replays_per_step=0,
    )


def _step_optimizer_and_finalize(mod, *, consumer, model, optimizer):
    """Mirror the runner's outer-loop sequencing for the test."""
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return mod._run_post_step_replay_ce(consumer=consumer, model=model)


def test_compute_replay_ce_pair_off_keeps_b3_defaults():
    """Flag OFF: no pending list, no EMA update, reward fields NaN."""
    mod = _load_runner()
    torch.manual_seed(7)
    model = _TinyTokenModel()
    cache = EpisodicCache(capacity=4, span_length=4, key_rep_dim=8)
    slot = _append_slot(
        cache, key_fp=42, write_bucket=2, source_write_id=1200,
        span_tokens=[3, 5, 7, 9],
    )
    consumer = _consumer(
        mod,
        event_log_enabled=True,
        cache=cache,
        compute_replay_ce_pair=False,
    )
    assert consumer.compute_replay_ce_pair is False
    assert consumer.pending_post_step_replays is None

    consumer.tagged_replay_queue.append({
        "slot": int(slot),
        "replay_id": 700,
        "query_event_id": 600,
        "source_write_id": 1200,
        "selection_step": 16,
        "policy_version": 3,
        "selected_rank": 0,
        "teacher_score": 0.75,
        "controller_logit": 0.25,
        "ce_before_replay": 4.0,
    })

    replayed = _drain(mod, consumer=consumer, model=model)
    assert replayed == 1
    assert consumer.replay_outcome_log is not None
    event = consumer.replay_outcome_log[0]
    # B3 default: entry-supplied ce_before_replay survives, everything
    # downstream of it stays NaN until B5 fills it.
    assert event["ce_before_replay"] == pytest.approx(4.0)
    assert math.isnan(event["ce_after_replay"])
    assert math.isnan(event["ce_delta_raw"])
    assert math.isnan(event["reward_shaped"])
    # No pending list, no EMA mutation regardless of optimizer step.
    patched = mod._run_post_step_replay_ce(consumer=consumer, model=model)
    assert patched == 0
    assert consumer.bucket_baseline_ema == [0.0, 0.0, 0.0, 0.0]


def test_compute_replay_ce_pair_on_finalizes_reward_fields_after_step():
    """Flag ON: drain stages, post-step pass patches finite values."""
    mod = _load_runner()
    torch.manual_seed(7)
    model = _TinyTokenModel()
    cache = EpisodicCache(capacity=4, span_length=4, key_rep_dim=8)
    slot = _append_slot(
        cache, key_fp=42, write_bucket=2, source_write_id=1200,
        span_tokens=[3, 5, 7, 9],
    )
    consumer = _consumer(
        mod,
        event_log_enabled=True,
        cache=cache,
        compute_replay_ce_pair=True,
    )
    assert consumer.compute_replay_ce_pair is True
    assert consumer.pending_post_step_replays == []

    consumer.tagged_replay_queue.append({
        "slot": int(slot),
        "replay_id": 700,
        "query_event_id": 600,
        "source_write_id": 1200,
        "selection_step": 16,
        "policy_version": 3,
        "selected_rank": 0,
        "teacher_score": 0.75,
        "controller_logit": 0.25,
        # Upstream-supplied placeholder; the override path replaces it
        # with the helper's ``replay_loss``.
        "ce_before_replay": 99.0,
    })

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
    optimizer.zero_grad(set_to_none=True)
    replayed = _drain(mod, consumer=consumer, model=model)
    assert replayed == 1
    # Drain emitted the dict already; reward fields are NaN until the
    # post-step pass runs.
    assert len(consumer.replay_outcome_log) == 1
    event = consumer.replay_outcome_log[0]
    assert math.isnan(event["ce_after_replay"])
    assert math.isnan(event["ce_delta_raw"])
    assert math.isnan(event["reward_shaped"])
    # ce_before_replay was overridden to the helper's replay_loss
    # (finite mean CE) — NOT the upstream 99.0 placeholder.
    assert math.isfinite(event["ce_before_replay"])
    assert event["ce_before_replay"] != pytest.approx(99.0)
    assert len(consumer.pending_post_step_replays) == 1

    patched = _step_optimizer_and_finalize(
        mod, consumer=consumer, model=model, optimizer=optimizer,
    )
    assert patched == 1
    # Pending list is drained — next step starts clean.
    assert consumer.pending_post_step_replays == []
    # Reward fields are now finite and self-consistent.
    assert math.isfinite(event["ce_before_replay"])
    assert math.isfinite(event["ce_after_replay"])
    expected_delta = (
        float(event["ce_before_replay"]) - float(event["ce_after_replay"])
    )
    assert event["ce_delta_raw"] == pytest.approx(expected_delta)
    assert event["bucket_baseline"] == pytest.approx(0.0)
    assert event["reward_shaped"] == pytest.approx(expected_delta)
    # EMA absorbed the finite delta on this bucket.
    assert consumer.bucket_baseline_ema is not None
    assert consumer.bucket_baseline_ema[2] == pytest.approx(
        0.05 * expected_delta
    )


def test_post_step_pass_is_noop_with_empty_pending_list():
    """Flag ON but no replay drained: helper returns 0, no mutation."""
    mod = _load_runner()
    model = _TinyTokenModel()
    cache = EpisodicCache(capacity=2, span_length=4, key_rep_dim=8)
    consumer = _consumer(
        mod,
        event_log_enabled=True,
        cache=cache,
        compute_replay_ce_pair=True,
    )
    assert consumer.pending_post_step_replays == []
    patched = mod._run_post_step_replay_ce(consumer=consumer, model=model)
    assert patched == 0
    assert consumer.bucket_baseline_ema == [0.0, 0.0, 0.0, 0.0]


def test_compute_replay_ce_pair_majority_positive_delta_sanity():
    """Soft sanity: descent on the same tokens drops post-step CE.

    With a fresh model, a non-zero LR, and the optimizer stepping on the
    same value tokens we just replayed, the post-step CE should drop
    below the pre-step CE for the majority of replays. This is a
    floor-check, not a tight invariant — flagging if it ever fails would
    surface a wrong-direction sign in the post-step forward (wrong
    tokens, wrong reduction, optimizer not actually applying grads).
    """
    mod = _load_runner()
    torch.manual_seed(31)
    model = _TinyTokenModel(vocab=24, dim=12)
    optimizer = torch.optim.SGD(model.parameters(), lr=2e-1)
    cache = EpisodicCache(capacity=64, span_length=6, key_rep_dim=12)

    rng = torch.Generator().manual_seed(101)
    deltas: list[float] = []
    for replay_idx in range(20):
        # Distinct token spans per replay so the test isn't pinned on
        # one particular row that the model overfits in step 1.
        tokens = torch.randint(
            0, 24, (6,), generator=rng, dtype=torch.int64
        ).tolist()
        slot = _append_slot(
            cache,
            key_fp=1000 + replay_idx,
            write_bucket=replay_idx % 4,
            source_write_id=2000 + replay_idx,
            span_tokens=tokens,
        )
        consumer = _consumer(
            mod,
            event_log_enabled=True,
            cache=cache,
            compute_replay_ce_pair=True,
        )
        consumer.tagged_replay_queue.append({
            "slot": int(slot),
            "replay_id": 700 + replay_idx,
            "query_event_id": 600 + replay_idx,
            "source_write_id": 2000 + replay_idx,
            "selection_step": replay_idx,
            "policy_version": 0,
            "selected_rank": 0,
            "teacher_score": 0.0,
            "controller_logit": 0.0,
            "ce_before_replay": 0.0,
        })
        optimizer.zero_grad(set_to_none=True)
        replayed = _drain(mod, consumer=consumer, model=model, weight=1.0)
        assert replayed == 1
        patched = _step_optimizer_and_finalize(
            mod, consumer=consumer, model=model, optimizer=optimizer,
        )
        assert patched == 1
        event = consumer.replay_outcome_log[0]
        delta = float(event["ce_delta_raw"])
        assert math.isfinite(delta), (
            f"replay {replay_idx}: ce_delta_raw not finite: {delta!r}"
        )
        deltas.append(delta)

    positive = sum(1 for d in deltas if d > 0.0)
    fraction_positive = positive / len(deltas)
    mean_delta = sum(deltas) / len(deltas)
    median_delta = sorted(deltas)[len(deltas) // 2]
    print(
        f"\n[B5 sanity] n={len(deltas)} fraction_positive={fraction_positive:.2f} "
        f"mean={mean_delta:+.4f} median={median_delta:+.4f}"
    )
    assert fraction_positive >= 0.6, (
        "post-step CE should drop on the majority of replays under a "
        "non-zero LR on the same value tokens; the wrong-sign delta is "
        "the canary for a buggy second forward (wrong tokens, wrong "
        "reduction, or the optimizer didn't actually update the weights)"
    )
