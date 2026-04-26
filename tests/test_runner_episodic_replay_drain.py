"""Phase 3.1+3.2 runner integration: replay drain + diagnostic log.

End state: when X's controller fills ``tagged_replay_queue`` on the
episodic-rank consumer, the episodic step body drains the queue, runs
replay backward through ``dreamworld_replay_from_cache_entry``, and
emits one NDJSON row per replay event. With ``episodic_enabled=False``
the runner stays bit-identical to pre-Phase-3 main (no log file, no
drain).

Tests are single-process (no mp.spawn / no NCCL) — the gather +
all-reduce path is exercised by the existing
``test_runner_episodic_gpu_drain.py::test_4rank_gloo_end_to_end``.
What this file pins is the per-step Python control flow: queue drain,
log emission, utility update, and the back-compat skip.

Tests:

* ``test_runner_drains_tagged_replay_queue_on_episodic_rank`` —
  pre-populate ``tagged_replay_queue``, run one ``_run_train_step``,
  verify the queue empties, the cache utility EMA moves, and the
  diagnostic log file has the expected rows.
* ``test_runner_back_compat_when_episodic_disabled`` — with
  ``episodic_consumer=None`` (the no-op shape) the step body
  short-circuits and no log file is created.
* ``test_runner_episodic_step_skips_when_no_logger_wired`` — the
  episodic-rank step still drains and runs replay even when no
  logger is wired (Phase 1 default until run_dir threading lands).
"""
from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.nn as nn

from chaoscontrol.episodic.diagnostics import (
    REPLAY_LOG_SCHEMA,
    DiagnosticsLogger,
)
from chaoscontrol.optim.episodic_cache import EpisodicCache


REPO = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO / "experiments" / "23_fast_path" / "runner_fast_path.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "runner_fast_path_phase3_test", RUNNER_PATH,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _TinyTokenModel(nn.Module):
    """Token-in / hidden-out model. Same shape as the
    `_TinyTokenTrainModel` used by `test_runner_episodic_gpu_drain.py`
    so the encode call signature matches the production runner's
    expectations.
    """

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


def _populate_cache_slot(
    cache: EpisodicCache,
    *,
    slot_value_anchor: int,
    span_tokens: torch.Tensor,
    key_fp: int,
    write_step: int,
    pressure_at_write: float = 0.0,
    write_bucket: int = -1,
) -> int:
    return cache.append(
        key_fp=key_fp,
        key_rep=torch.zeros(cache.key_rep_dim, dtype=torch.float32),
        value_tok_ids=span_tokens.to(dtype=torch.int64),
        value_anchor_id=slot_value_anchor,
        current_step=write_step,
        embedding_version=0,
        pressure_at_write=pressure_at_write,
        write_bucket=write_bucket,
    )


class _StubConsumerWithTaggedQueue:
    """Minimal consumer state stand-in that carries a cache + a
    ``tagged_replay_queue`` field BEFORE X's controller-spawn lane
    merges. The real ``_EpisodicConsumerState`` will gain this field
    in Phase 2; until then, ``_get_tagged_replay_queue`` reads
    whatever attribute is present via ``getattr`` so this stub is
    indistinguishable from the post-merge shape from Y's side.
    """

    __slots__ = (
        "cache",
        "heartbeat",
        "controller_query_queue",
        "controller_query_enabled",
        "tagged_replay_queue",
    )

    def __init__(self, cache: EpisodicCache) -> None:
        self.cache = cache
        self.heartbeat = [0]
        self.controller_query_queue: list = []
        self.controller_query_enabled = False
        self.tagged_replay_queue: list[dict] = []


class TestEpisodicReplayDrain(unittest.TestCase):

    def setUp(self) -> None:
        self.mod = _load_runner()
        torch.manual_seed(0)

    def test_runner_drains_tagged_replay_queue_on_episodic_rank(self):
        """Pre-populate ``tagged_replay_queue`` with two slot indices,
        run the episodic-rank step body once, verify:

        * the queue ended up empty (drained destructively),
        * the diagnostic log file exists with one row per replay,
        * each row carries every Decision-0.9 column,
        * cache utility EMA moved on each replayed slot,
        * grads accumulated on the model (replay backward fired).
        """
        model = _TinyTokenModel(vocab=16, dim=8)
        cache = EpisodicCache(capacity=4, span_length=4, key_rep_dim=8)
        slot_a = _populate_cache_slot(
            cache,
            slot_value_anchor=2,
            span_tokens=torch.tensor([3, 5, 7, 9], dtype=torch.int64),
            key_fp=42,
            write_step=10,
            pressure_at_write=1.25,
            write_bucket=2,
        )
        slot_b = _populate_cache_slot(
            cache,
            slot_value_anchor=4,
            span_tokens=torch.tensor([1, 2, 3, 4], dtype=torch.int64),
            key_fp=99,
            write_step=11,
            pressure_at_write=2.50,
            write_bucket=3,
        )
        utility_pre_a = float(cache.utility_u[slot_a].item())
        utility_pre_b = float(cache.utility_u[slot_b].item())
        self.assertEqual(utility_pre_a, 1.0)
        self.assertEqual(utility_pre_b, 1.0)

        consumer = _StubConsumerWithTaggedQueue(cache)
        consumer.tagged_replay_queue.append({
            "step": 11, "slot": int(slot_a), "score": 0.7, "selected_at": 11,
        })
        consumer.tagged_replay_queue.append({
            "step": 11, "slot": int(slot_b), "score": 0.3, "selected_at": 11,
        })

        # Inputs/targets just need to be valid shapes for the
        # ``is_episodic_rank=True`` branch — that branch skips the
        # main forward, so the inputs aren't consumed by the SSM.
        # Use a tiny [1, 4] batch.
        inputs = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)
        targets = torch.tensor([[2, 3, 4, 5]], dtype=torch.int64)

        with TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "episodic_replay_log_rank3.ndjson"
            logger = DiagnosticsLogger(log_path)

            # Single-process episodic-rank step. ddp_active=False
            # short-circuits the gather + all-reduce, exercising only
            # the in-process drain + replay path.
            self.mod._run_train_step(
                model=model,
                inputs=inputs,
                targets=targets,
                chunk_size=4,
                precision="fp32",
                ddp_active=False,
                world_size=1,
                rank=0,
                lm_head_backward_mode="single",
                grad_allreduce_mode="bulk",
                is_episodic_rank=True,
                all_group=None,
                episodic_emit=None,
                episodic_consumer=consumer,
                episodic_replay_logger=logger,
                current_step=11,
                embedding_version=0,
                dreamworld_weight=1.0,
            )
            logger.close()

            # Queue drained destructively.
            self.assertEqual(consumer.tagged_replay_queue, [])

            # Diagnostic log has one row per replay event.
            lines = log_path.read_text().splitlines()
            self.assertEqual(len(lines), 2)
            for line in lines:
                row = json.loads(line)
                # All 16 documented columns present.
                for col in REPLAY_LOG_SCHEMA:
                    self.assertIn(col, row)
                self.assertEqual(row["step"], 11)
                self.assertIn(row["slot"], (slot_a, slot_b))
                if row["slot"] == slot_a:
                    self.assertEqual(row["write_pressure"], 1.25)
                    self.assertEqual(row["write_bucket"], 2)
                else:
                    self.assertEqual(row["write_pressure"], 2.5)
                    self.assertEqual(row["write_bucket"], 3)
                self.assertEqual(row["utility_pre"], 1.0)
                # Phase 1 NaN policy: cosines + utility_signal_raw
                # serialize as null.
                self.assertIsNone(row["replay_grad_cos_rare"])
                self.assertIsNone(row["replay_grad_cos_common"])
                self.assertIsNone(row["replay_grad_cos_total"])
                self.assertIsNone(row["utility_signal_raw"])
                # utility_signal_transformed = 0.0 deterministic.
                self.assertEqual(row["utility_signal_transformed"], 0.0)

            # Cache utility EMA stayed at the init value because the
            # raw signal was NaN and update_utility was skipped per the
            # Phase 1 NaN-skip invariant (Y reviewer fix #3). Without
            # the skip, replayed entries would decay toward 0 on every
            # replay, getting evicted faster than unused entries —
            # collapsing Arm B to "cosine × anti-frequency-of-replay."
            # See test_runner_skips_utility_update_when_signal_is_nan
            # for the standalone pin.
            self.assertEqual(float(cache.utility_u[slot_a].item()), utility_pre_a)
            self.assertEqual(float(cache.utility_u[slot_b].item()), utility_pre_b)

        # Replay backward fired → at least one param now has a grad.
        any_nonzero = False
        for p in model.parameters():
            if p.grad is not None and p.grad.abs().sum().item() > 0.0:
                any_nonzero = True
                break
        self.assertTrue(any_nonzero)

    def test_runner_skips_replay_when_consumer_is_none(self):
        """``episodic_consumer=None`` short-circuits the entire
        Phase-3 path on the episodic-rank side. No log file created,
        no grads accumulated — this is the no-op shape that
        ``_attach_episodic_consumer`` returns for
        ``episodic_enabled=False`` as well, so the back-compat
        invariant for the disabled case is covered transitively. The
        ``train rank`` (else) branch is untouched by Phase 3.1+3.2;
        its bit-identical-to-main behavior is pinned by the existing
        regression tests (``test_runner_episodic_gpu_drain.py`` end-
        to-end + ``test_runner.py``)."""
        model = _TinyTokenModel()
        inputs = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)
        targets = torch.tensor([[2, 3, 4, 5]], dtype=torch.int64)

        with TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "episodic_replay_log_rank3.ndjson"
            # No log file before the call.
            self.assertFalse(log_path.exists())

            self.mod._run_train_step(
                model=model,
                inputs=inputs,
                targets=targets,
                chunk_size=4,
                precision="fp32",
                ddp_active=False,
                world_size=1,
                rank=0,
                lm_head_backward_mode="single",
                grad_allreduce_mode="bulk",
                is_episodic_rank=True,
                all_group=None,
                episodic_emit=None,
                episodic_consumer=None,
                episodic_replay_logger=None,
                current_step=0,
                embedding_version=0,
            )
            # No log file created.
            self.assertFalse(log_path.exists())
        # No grads on the model — episodic-rank skip-main + no replay
        # = no backward = no param.grad.
        for p in model.parameters():
            self.assertIsNone(p.grad)

    def test_runner_episodic_step_handles_no_logger_wired(self):
        """Even with a ``tagged_replay_queue`` populated, if the runner
        didn't wire a logger (Phase 1 transition state — X has merged
        but the logger config-key isn't set), replay backward still
        fires and the queue still drains. We just don't get a
        diagnostic log file. Phase 3 production runs always wire the
        logger; this test pins the graceful fallback."""
        model = _TinyTokenModel()
        cache = EpisodicCache(capacity=2, span_length=4, key_rep_dim=8)
        slot = _populate_cache_slot(
            cache,
            slot_value_anchor=2,
            span_tokens=torch.tensor([3, 5, 7, 9], dtype=torch.int64),
            key_fp=42,
            write_step=10,
        )
        consumer = _StubConsumerWithTaggedQueue(cache)
        consumer.tagged_replay_queue.append({
            "step": 11, "slot": int(slot), "score": 0.7, "selected_at": 11,
        })

        inputs = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)
        targets = torch.tensor([[2, 3, 4, 5]], dtype=torch.int64)

        self.mod._run_train_step(
            model=model,
            inputs=inputs,
            targets=targets,
            chunk_size=4,
            precision="fp32",
            ddp_active=False,
            world_size=1,
            rank=0,
            lm_head_backward_mode="single",
            grad_allreduce_mode="bulk",
            is_episodic_rank=True,
            all_group=None,
            episodic_emit=None,
            episodic_consumer=consumer,
            # No logger wired — should still drain + replay.
            episodic_replay_logger=None,
            current_step=11,
            embedding_version=0,
            dreamworld_weight=1.0,
        )
        # Queue drained.
        self.assertEqual(consumer.tagged_replay_queue, [])
        # Replay backward fired.
        any_nonzero = False
        for p in model.parameters():
            if p.grad is not None and p.grad.abs().sum().item() > 0.0:
                any_nonzero = True
                break
        self.assertTrue(any_nonzero)

    def test_get_tagged_replay_queue_returns_empty_list_for_no_op_consumer(self):
        """The no-op consumer (episodic disabled OR non-episodic rank)
        carries an empty tagged_replay_queue post-X-merge; the helper
        returns it as []. None consumer also returns []."""
        no_op_consumer = self.mod._attach_episodic_consumer(
            episodic_enabled=False,
            is_episodic_rank=False,
            world_size=2,
            config={},
            model_dim=4,
            all_group=None,
        )
        result = self.mod._get_tagged_replay_queue(no_op_consumer)
        self.assertEqual(result, [])
        self.assertEqual(self.mod._get_tagged_replay_queue(None), [])

    def test_runner_skips_utility_update_when_signal_is_nan(self):
        """Phase 1 NaN-skip invariant (Y reviewer fix #3).

        With no live rare-grad direction in scope (the default until
        Phase 4 wires the post-allreduce EMA snapshot), the replay's
        utility_signal_raw is NaN. update_utility MUST be skipped in
        that case — otherwise feeding 0.0 into the EMA decays utility
        toward zero on every replay, replayed entries get evicted faster
        than unused ones, and Arm B collapses to "cosine ×
        anti-frequency-of-replay."

        This test pins the skip: replay a slot once with NaN raw signal,
        verify utility_u stays at its init value (1.0).
        """
        model = _TinyTokenModel()
        cache = EpisodicCache(capacity=2, span_length=4, key_rep_dim=8)
        slot = _populate_cache_slot(
            cache,
            slot_value_anchor=2,
            span_tokens=torch.tensor([3, 5, 7, 9], dtype=torch.int64),
            key_fp=42,
            write_step=10,
        )
        # Sanity: cache is fresh, utility_u starts at the documented init.
        self.assertEqual(float(cache.utility_u[slot].item()), 1.0)

        consumer = _StubConsumerWithTaggedQueue(cache)
        consumer.tagged_replay_queue.append({
            "step": 11, "slot": int(slot), "score": 0.5, "selected_at": 11,
        })

        with TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "episodic_replay_log_rank3.ndjson"
            logger = DiagnosticsLogger(log_path)

            self.mod._run_train_step(
                model=model,
                inputs=torch.tensor([[1, 2, 3, 4]], dtype=torch.int64),
                targets=torch.tensor([[2, 3, 4, 5]], dtype=torch.int64),
                chunk_size=4,
                precision="fp32",
                ddp_active=False,
                world_size=1,
                rank=0,
                lm_head_backward_mode="single",
                grad_allreduce_mode="bulk",
                is_episodic_rank=True,
                all_group=None,
                episodic_emit=None,
                episodic_consumer=consumer,
                episodic_replay_logger=logger,
                current_step=11,
                embedding_version=0,
                dreamworld_weight=1.0,
            )
            logger.close()

            # The replay fired (queue drained, log row written),
            # but utility_u stayed at the init value because the raw
            # signal was NaN and update_utility was skipped.
            self.assertEqual(consumer.tagged_replay_queue, [])
            log_lines = log_path.read_text().strip().splitlines()
            self.assertEqual(len(log_lines), 1)
            # Utility unchanged.
            self.assertEqual(float(cache.utility_u[slot].item()), 1.0)

    def test_runner_skips_replay_for_evicted_slot(self):
        """A slot can be evicted between the controller's tag and our
        drain (cache eviction races against queue drains). The drain
        must skip without crashing or logging a row for the evicted
        slot."""
        model = _TinyTokenModel()
        cache = EpisodicCache(capacity=2, span_length=4, key_rep_dim=8)
        slot = _populate_cache_slot(
            cache,
            slot_value_anchor=2,
            span_tokens=torch.tensor([3, 5, 7, 9], dtype=torch.int64),
            key_fp=42,
            write_step=10,
        )
        cache.evict(slot)  # Controller tagged it before we drained.

        consumer = _StubConsumerWithTaggedQueue(cache)
        consumer.tagged_replay_queue.append({
            "step": 11, "slot": int(slot), "score": 0.5, "selected_at": 11,
        })

        with TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "episodic_replay_log_rank3.ndjson"
            logger = DiagnosticsLogger(log_path)

            self.mod._run_train_step(
                model=model,
                inputs=torch.tensor([[1, 2, 3, 4]], dtype=torch.int64),
                targets=torch.tensor([[2, 3, 4, 5]], dtype=torch.int64),
                chunk_size=4,
                precision="fp32",
                ddp_active=False,
                world_size=1,
                rank=0,
                lm_head_backward_mode="single",
                grad_allreduce_mode="bulk",
                is_episodic_rank=True,
                all_group=None,
                episodic_emit=None,
                episodic_consumer=consumer,
                episodic_replay_logger=logger,
                current_step=11,
                embedding_version=0,
                dreamworld_weight=1.0,
            )
            logger.close()

            # Queue drained but no log row written for the evicted slot.
            self.assertEqual(consumer.tagged_replay_queue, [])
            self.assertEqual(log_path.read_text(), "")

    def test_runner_replay_drain_respects_per_step_budget(self):
        """Replay fanout is part of the treatment dose.

        A top-k controller can enqueue many tags per training step. The runner
        must cap how many it replays in one step and leave the remainder queued
        for later rather than turning one write burst into dozens of backwards.
        """
        model = _TinyTokenModel()
        cache = EpisodicCache(capacity=4, span_length=4, key_rep_dim=8)
        slots = [
            _populate_cache_slot(
                cache,
                slot_value_anchor=2,
                span_tokens=torch.tensor([3, 5, 7, 9], dtype=torch.int64),
                key_fp=40 + i,
                write_step=10 + i,
            )
            for i in range(3)
        ]
        consumer = _StubConsumerWithTaggedQueue(cache)
        for slot in slots:
            consumer.tagged_replay_queue.append({
                "step": 11,
                "slot": int(slot),
                "score": 0.7,
                "selected_at": 11,
            })

        replayed = self.mod._run_episodic_replay_from_tagged_queue(
            consumer=consumer,
            model=model,
            current_step=11,
            weight=1.0,
            lm_head_backward_mode="single",
            lm_head_tile_size=1024,
            logger=None,
            max_replays_per_step=1,
        )

        self.assertEqual(replayed, 1)
        self.assertEqual(len(consumer.tagged_replay_queue), 2)
        self.assertEqual(
            [entry["slot"] for entry in consumer.tagged_replay_queue],
            slots[1:],
        )

    def test_runner_replay_drain_unbounded_when_budget_is_zero(self):
        """Pre-landing behavior was an unbounded `while tagged:` drain.
        max_replays_per_step=0 must restore that — drain everything in
        one call so a controller-driven config doesn't silently lose
        treatment dose by inheriting a per-step cap."""
        model = _TinyTokenModel()
        cache = EpisodicCache(capacity=8, span_length=4, key_rep_dim=8)
        slots = [
            _populate_cache_slot(
                cache,
                slot_value_anchor=2,
                span_tokens=torch.tensor([3, 5, 7, 9], dtype=torch.int64),
                key_fp=40 + i,
                write_step=10 + i,
            )
            for i in range(5)
        ]
        consumer = _StubConsumerWithTaggedQueue(cache)
        for slot in slots:
            consumer.tagged_replay_queue.append({
                "step": 11,
                "slot": int(slot),
                "score": 0.7,
                "selected_at": 11,
            })

        replayed = self.mod._run_episodic_replay_from_tagged_queue(
            consumer=consumer,
            model=model,
            current_step=11,
            weight=1.0,
            lm_head_backward_mode="single",
            lm_head_tile_size=1024,
            logger=None,
            max_replays_per_step=0,
        )

        self.assertEqual(replayed, 5)
        self.assertEqual(consumer.tagged_replay_queue, [])


if __name__ == "__main__":
    unittest.main()
