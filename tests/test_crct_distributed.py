"""Mocked-primitive coverage for crct_distributed.

End-to-end gloo-spawn 4-rank coverage rides on the existing
test_runner_3plus1_skip_main.py pattern as a follow-up integration test;
this file unit-tests the coordination layer with FakeProcessGroup so the
suite stays CPU-only and fast.
"""
import unittest
import torch
import torch.nn as nn
from chaoscontrol import crct_distributed as crct


class FakeWork:
    def __init__(self, completed=False):
        self._completed = completed
    def is_completed(self):
        return self._completed
    def wait(self):
        self._completed = True


class FakeProcessGroup:
    def __init__(self, ranks):
        self.ranks = list(ranks)
        self.broadcasts = []
        self.gathers = []
    def __repr__(self):
        return f"FakePG({self.ranks})"


class TestModuleImports(unittest.TestCase):
    def test_module_exposes_documented_api(self):
        for name in (
            "create_crct_process_groups",
            "WeightMailbox",
            "TeacherResultMailbox",
            "TeacherPayload",
            "Rank3MemoryCoprocessor",
            "rank3_coprocessor_loop",
            "fail_open_controller_term",
        ):
            self.assertTrue(
                hasattr(crct, name),
                f"crct_distributed must expose {name}",
            )


class TestCreateProcessGroups(unittest.TestCase):
    def setUp(self):
        self.calls = []
        import torch.distributed as dist
        self._real = dist.new_group
        dist.new_group = lambda ranks=None, **_: (
            self.calls.append(list(ranks)) or FakeProcessGroup(ranks)
        )

    def tearDown(self):
        import torch.distributed as dist
        dist.new_group = self._real

    def test_world_size_4_returns_three_groups_in_canonical_order(self):
        train_pg, crct_pg, sync_pg = crct.create_crct_process_groups(4)
        self.assertEqual(self.calls, [[0, 1, 2], [0, 1, 2, 3], [0, 3]])
        self.assertEqual(train_pg.ranks, [0, 1, 2])
        self.assertEqual(crct_pg.ranks, [0, 1, 2, 3])
        self.assertEqual(sync_pg.ranks, [0, 3])

    def test_world_size_below_4_returns_none_groups(self):
        train_pg, crct_pg, sync_pg = crct.create_crct_process_groups(2)
        self.assertIsNone(train_pg)
        self.assertIsNone(crct_pg)
        self.assertIsNone(sync_pg)


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2, bias=True)


class TestWeightMailboxBasics(unittest.TestCase):
    def test_post_weights_skips_when_step_not_aligned(self):
        m = _TinyModel()
        pg = FakeProcessGroup([0, 3])
        mb = crct.WeightMailbox(
            m, pg, sync_interval_steps=50, src_rank=0, my_rank=0,
            broadcast_fn=lambda *a, **kw: FakeWork(completed=False),
        )
        mb.post_weights(global_step=49)
        self.assertEqual(mb.snapshots_posted, 0)

    def test_post_weights_snapshots_at_interval(self):
        m = _TinyModel()
        pg = FakeProcessGroup([0, 3])
        broadcasts = []
        mb = crct.WeightMailbox(
            m, pg, sync_interval_steps=50, src_rank=0, my_rank=0,
            broadcast_fn=lambda buf, src, group, async_op: (
                broadcasts.append((buf.shape, src, async_op)) or FakeWork(completed=True)
            ),
        )
        mb.post_weights(global_step=50)
        self.assertEqual(mb.snapshots_posted, 1)
        self.assertEqual(len(broadcasts), 1)
        self.assertTrue(broadcasts[0][2])

    def test_maybe_sync_idempotent_without_new_post(self):
        m = _TinyModel()
        pg = FakeProcessGroup([0, 3])
        mb = crct.WeightMailbox(
            m, pg, sync_interval_steps=50, src_rank=0, my_rank=3,
            broadcast_fn=lambda *a, **kw: FakeWork(completed=False),
        )
        v0 = mb.snapshot_version
        mb.maybe_sync(50)
        mb.maybe_sync(51)
        self.assertEqual(mb.snapshot_version, v0)


class TestWeightMailboxNonBlocking(unittest.TestCase):
    def test_post_weights_does_not_await_broadcast_handle(self):
        m = _TinyModel()
        pg = FakeProcessGroup([0, 3])
        slow_work = FakeWork(completed=False)
        broadcasts_started = []
        mb = crct.WeightMailbox(
            m, pg, sync_interval_steps=10, src_rank=0, my_rank=0,
            broadcast_fn=lambda *a, **kw: (
                broadcasts_started.append(1) or slow_work
            ),
        )
        mb.post_weights(global_step=10)
        self.assertEqual(len(broadcasts_started), 1)
        self.assertFalse(slow_work.is_completed())
        self.assertEqual(mb.snapshots_posted, 1)


class TestTeacherResultMailbox(unittest.TestCase):
    def test_try_get_returns_none_when_work_incomplete(self):
        pg = FakeProcessGroup([0, 1, 2, 3])
        pending = FakeWork(completed=False)
        mb = crct.TeacherResultMailbox(
            pg, my_rank=0, num_train_ranks=3,
            payload_shape=(2, 8, 16),
            broadcast_fn=lambda *a, **kw: pending,
        )
        self.assertIsNone(mb.try_get(step=100))

    def test_try_get_returns_payload_when_work_complete(self):
        pg = FakeProcessGroup([0, 1, 2, 3])
        done = FakeWork(completed=True)
        mb = crct.TeacherResultMailbox(
            pg, my_rank=0, num_train_ranks=3,
            payload_shape=(2, 8, 16),
            broadcast_fn=lambda *a, **kw: done,
        )
        mb._meta_buf[0] = 0.75
        mb._meta_buf[1] = 100.0
        mb._meta_buf[2] = 7.0

        first = mb.try_get(step=100)
        payload = first if first is not None else mb.try_get(step=100)
        self.assertIsNotNone(payload)
        self.assertIsInstance(payload, crct.TeacherPayload)
        self.assertEqual(payload.step_id, 100)
        self.assertEqual(payload.target.shape, (2, 8, 16))
        self.assertEqual(payload.conf.shape, (2, 8))
        self.assertAlmostEqual(payload.loss_weight, 0.75)
        self.assertEqual(payload.snapshot_version, 7)

    def test_post_result_drop_when_inflight(self):
        pg = FakeProcessGroup([0, 1, 2, 3])
        pending = FakeWork(completed=False)
        mb = crct.TeacherResultMailbox(
            pg, my_rank=3, num_train_ranks=3,
            payload_shape=(2, 8, 16),
            broadcast_fn=lambda *a, **kw: pending,
        )
        target = torch.zeros(2, 8, 16, dtype=torch.float16)
        conf = torch.zeros(2, 8, dtype=torch.float16)
        mb.post_result(step_id=10, target=target, conf=conf, loss_weight=1.0)
        mb.post_result(step_id=11, target=target, conf=conf, loss_weight=1.0)
        self.assertEqual(mb.posts_attempted, 2)
        self.assertEqual(mb.posts_dropped, 1)


class TestFailOpenContract(unittest.TestCase):
    def test_fail_open_keeps_controller_head_in_autograd_graph(self):
        controller_head = nn.Linear(8, 4)
        x = torch.randn(2, 8)
        controller_logits = controller_head(x)
        lm_loss = torch.tensor(1.0, requires_grad=True)
        loss = crct.fail_open_controller_term(
            payload=None,
            controller_logits=controller_logits,
            lm_loss=lm_loss,
            lambda_ctrl=0.5,
            controller_loss_fn=lambda payload, logits: torch.zeros(()),
        )
        loss.backward()
        self.assertIsNotNone(controller_head.weight.grad)
        self.assertTrue(torch.equal(
            controller_head.weight.grad,
            torch.zeros_like(controller_head.weight),
        ))


class _StubMailbox:
    posts_attempted = 0
    posts_dropped = 0

    def __init__(self, events, name):
        self._events = events
        self.name = name

    def post_result(self, *a, **kw):
        self._events.append("teacher_post")

    def try_get(self, *a, **kw):
        return None

    def maybe_sync(self, *a, **kw):
        self._events.append("weight_sync")

    def post_weights(self, *a, **kw):
        self._events.append("weight_post")


def _make_copro(events, *, gather_returns=None):
    return crct.Rank3MemoryCoprocessor(
        model_copy=_TinyModel(),
        weight_mailbox=_StubMailbox(events, "weights"),
        result_mailbox=_StubMailbox(events, "results"),
        score_fn=lambda *a, **kw: events.append("score") or None,
        episodic_drain_fn=lambda: events.append("episodic_drain"),
        controller_tick_fn=lambda: events.append("controller_tick"),
        crct_pg=FakeProcessGroup([0, 1, 2, 3]),
        sync_pg=FakeProcessGroup([0, 3]),
        gather_fn=lambda *a, **kw: (
            events.append("gather") or (gather_returns or object())
        ),
    )


class TestCoprocessorCausalOrder(unittest.TestCase):
    def test_empty_start_order_is_gather_drain_tick_sync_score(self):
        events = []
        copro = _make_copro(events)
        copro.step_once(global_step=10)
        self.assertEqual(
            events,
            ["gather", "episodic_drain", "controller_tick",
             "weight_sync", "score"],
        )
        self.assertEqual(copro.metrics["scores_run"], 1)
        self.assertEqual(copro.metrics["teacher_drops"], 0)

    def test_inflight_completed_reaps_first_then_enqueues_new(self):
        events = []
        copro = _make_copro(events)
        copro._inflight_score = FakeWork(completed=True)
        copro.step_once(global_step=10)
        self.assertEqual(
            events,
            ["teacher_post", "gather", "episodic_drain",
             "controller_tick", "weight_sync", "score"],
        )

    def test_inflight_incomplete_skips_score_and_increments_drops(self):
        events = []
        copro = _make_copro(events)
        copro._inflight_score = FakeWork(completed=False)
        copro.step_once(global_step=10)
        copro.step_once(global_step=11)
        self.assertNotIn("score", events)
        self.assertEqual(events.count("gather"), 2)
        self.assertEqual(copro.metrics["scores_run"], 0)
        self.assertEqual(copro.metrics["teacher_drops"], 2)


class TestRank3Loop(unittest.TestCase):
    def test_loop_calls_step_once_until_stop_flag(self):
        ticks = []

        class _StubCopro:
            def step_once(self, step):
                ticks.append(step)

        flag = {"stop": False}

        def step_gen():
            for s in range(3):
                yield s
            flag["stop"] = True

        crct.rank3_coprocessor_loop(
            _StubCopro(),
            stop_flag=lambda: flag["stop"],
            step_iter=step_gen(),
        )
        self.assertEqual(ticks, [0, 1, 2])

    def test_loop_no_op_when_world_size_below_4(self):
        crct.rank3_coprocessor_loop(
            coprocessor=None,
            stop_flag=lambda: False,
            step_iter=iter(range(100)),
        )


class TestSynchronousFallback(unittest.TestCase):
    def test_try_get_with_none_pg_invokes_inline_score_fn(self):
        score_calls = []

        def synchronous_score(input_ids, valid_mask, step):
            del input_ids, valid_mask
            score_calls.append(step)
            return crct.TeacherPayload(
                step_id=step,
                target=torch.zeros(2, 8, 16, dtype=torch.float16),
                conf=torch.zeros(2, 8, dtype=torch.float16),
                loss_weight=1.0,
                snapshot_version=0,
            )

        mb = crct.TeacherResultMailbox(
            crct_pg=None, my_rank=0, num_train_ranks=1,
            payload_shape=(2, 8, 16),
            inline_score_fn=synchronous_score,
        )
        payload = mb.try_get_with_input(
            step=42,
            input_ids=torch.zeros(2, 8, dtype=torch.long),
            valid_mask=torch.ones(2, 8, dtype=torch.bool),
        )
        self.assertEqual(score_calls, [42])
        self.assertEqual(payload.step_id, 42)

    def test_fallback_init_sets_all_attrs_uniformly(self):
        mb = crct.TeacherResultMailbox(
            crct_pg=None, my_rank=0, num_train_ranks=1,
            payload_shape=(2, 8, 16),
            inline_score_fn=lambda *a, **kw: None,
        )
        for attr in (
            "posts_attempted", "posts_dropped", "_target_buf",
            "_conf_buf", "_meta_buf", "_inflight", "num_train_ranks",
            "payload_shape", "dtype",
        ):
            self.assertTrue(hasattr(mb, attr), f"missing {attr}")
        self.assertEqual(mb.posts_attempted, 0)
        self.assertEqual(mb.posts_dropped, 0)
        self.assertIsNone(mb._inflight)
