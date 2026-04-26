"""Pin bit-identity of the train step when episodic is OFF (I4).

The substrate landing extended ``EpisodicCache`` with new schema fields
(``pressure_at_write``, ``source_write_id``, ``write_bucket``,
``slot_state``, ``simplex_edge_slot``, ``simplex_edge_weight``) and the
runner gained the ``_EpisodicConsumerState`` shape with placeholder
event-log lists and a ``tagged_replay_queue``. All of these default to
empty / no-op shapes when the controller isn't wired
(``episodic_consumer=None``, ``episodic_emit=None``).

This file pins that the train rank's ``_run_train_step`` is
bit-identical between two parallel-seeded runs with episodic OFF — the
substrate's schema additions and consumer-state placeholders MUST NOT
leak into the gradient path. Mirrors W's eval-side pin in
``test_eval_stream_legality_with_cache.py::
test_no_cache_path_is_bit_identical_to_existing_controller``.

Per the W code-review I4 task: closes the regression gap that future
schema additions could creep into the train gradient path silently.

Tests:

* ``test_episodic_off_train_step_is_bit_identical`` — two parallel-
  seeded models run one ``_run_train_step`` with ``episodic_consumer=
  None`` and ``episodic_emit=None``. Loss tensor, every ``param.grad``,
  and the post-``optimizer.step()`` state dict must match
  bit-identically.
* ``test_episodic_enabled_but_unwired_train_step_is_bit_identical`` —
  stronger invariant: even with the ``_EpisodicConsumerState`` shape
  attached on the train rank (``cache=None``, both queues empty,
  ``controller_query_enabled=False``), the train step is still
  bit-identical because no ``episodic_emit`` is wired.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import torch.nn as nn


REPO = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO / "experiments" / "23_fast_path" / "runner_fast_path.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "runner_fast_path_i4_test", RUNNER_PATH,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _TinyTokenModel(nn.Module):
    """Token-in / hidden-out model. Same shape as the
    ``_TinyTokenModel`` in ``test_runner_episodic_replay_drain.py`` so
    the encode call signature matches the production runner's
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


def _assert_state_dicts_equal(a: nn.Module, b: nn.Module, *, where: str) -> None:
    sd_a = a.state_dict()
    sd_b = b.state_dict()
    assert sd_a.keys() == sd_b.keys(), (
        f"state_dict keys diverged at {where}: {sd_a.keys()} vs {sd_b.keys()}"
    )
    for key in sd_a:
        assert torch.equal(sd_a[key], sd_b[key]), (
            f"state_dict mismatch on {key} at {where}"
        )


def _assert_grads_equal(a: nn.Module, b: nn.Module) -> None:
    params_a = dict(a.named_parameters())
    params_b = dict(b.named_parameters())
    assert params_a.keys() == params_b.keys()
    for name, pa in params_a.items():
        pb = params_b[name]
        if pa.grad is None and pb.grad is None:
            continue
        assert pa.grad is not None and pb.grad is not None, (
            f"grad presence mismatch on {name}: "
            f"{pa.grad is not None} vs {pb.grad is not None}"
        )
        assert torch.equal(pa.grad, pb.grad), f"grad mismatch on {name}"


def test_episodic_off_train_step_is_bit_identical():
    """With ``episodic_consumer=None`` and ``episodic_emit=None`` (the
    default off-state), one ``_run_train_step`` on parallel-seeded
    models must produce a bit-identical loss tensor, bit-identical
    ``param.grad`` for every parameter, and bit-identical state dicts
    after one ``optimizer.step()``.

    Pins that the substrate's cache-schema additions
    (``pressure_at_write``, ``source_write_id``, ``write_bucket``,
    ``slot_state``, ``simplex_edge_*``) and the placeholder
    ``tagged_replay_queue`` / ``controller_query_queue`` lists do NOT
    leak into the gradient path when the controller is off.

    Note: this is a determinism canary on the disabled path, not a
    diff against a frozen pre-substrate reference. Any new
    non-determinism in the substrate's no-op path will trip this.

    Closes I4 from the W code-review.
    """
    runner = _load_runner()

    torch.manual_seed(1337)
    model_a = _TinyTokenModel(vocab=16, dim=8)
    torch.manual_seed(1337)
    model_b = _TinyTokenModel(vocab=16, dim=8)

    # Sanity: matched init.
    _assert_state_dicts_equal(model_a, model_b, where="construction")

    inputs = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)
    targets = torch.tensor([[2, 3, 4, 5]], dtype=torch.int64)

    loss_a = runner._run_train_step(
        model=model_a,
        inputs=inputs,
        targets=targets,
        chunk_size=4,
        precision="fp32",
        ddp_active=False,
        world_size=1,
        rank=0,
        lm_head_backward_mode="single",
        grad_allreduce_mode="bulk",
        is_episodic_rank=False,
        all_group=None,
        episodic_emit=None,
        episodic_consumer=None,
        episodic_replay_logger=None,
        current_step=0,
        embedding_version=0,
    )
    loss_b = runner._run_train_step(
        model=model_b,
        inputs=inputs,
        targets=targets,
        chunk_size=4,
        precision="fp32",
        ddp_active=False,
        world_size=1,
        rank=0,
        lm_head_backward_mode="single",
        grad_allreduce_mode="bulk",
        is_episodic_rank=False,
        all_group=None,
        episodic_emit=None,
        episodic_consumer=None,
        episodic_replay_logger=None,
        current_step=0,
        embedding_version=0,
    )

    # Loss tensor: bit-identical scalar.
    assert torch.equal(loss_a, loss_b), (
        f"loss diverged: {loss_a.item()} vs {loss_b.item()}"
    )

    # param.grad: bit-identical for every parameter.
    _assert_grads_equal(model_a, model_b)

    # state_dict after one optimizer step: bit-identical end-to-end.
    # _run_train_step doesn't step the optimizer; do it here so the
    # state_dict comparison actually exercises a full training step.
    opt_a = torch.optim.SGD(model_a.parameters(), lr=0.1)
    opt_b = torch.optim.SGD(model_b.parameters(), lr=0.1)
    opt_a.step()
    opt_b.step()
    _assert_state_dicts_equal(model_a, model_b, where="after optimizer.step()")


def test_episodic_enabled_but_unwired_train_step_is_bit_identical():
    """Stronger invariant: even when ``episodic_enabled=True`` but the
    train rank's ``_EpisodicConsumerState`` is the no-op shape
    (``cache=None``, ``controller_query_enabled=False``, both queues
    empty) AND no ``episodic_emit`` is wired, the train step is still
    bit-identical to the off case.

    This pins that the consumer-state placeholder lists
    (``controller_query_queue``, ``tagged_replay_queue``) and the
    ``heartbeat`` field don't leak into the gradient path on train
    ranks. The episodic emit is the only train-rank touchpoint that
    can affect the gradient path; with it ``None``, all the substrate
    additions stay quarantined.

    Closes I4 from the W code-review (optional second test).
    """
    runner = _load_runner()

    # Build the no-op consumer shape that ``_attach_episodic_consumer``
    # returns on a train rank when episodic is ENABLED but the rank
    # isn't the episodic one. cache=None, both queues empty, heartbeat
    # at zero (per test_consumer_state_back_compat_with_disabled_episodic
    # invariant).
    train_rank_consumer = runner._attach_episodic_consumer(
        episodic_enabled=True,
        is_episodic_rank=False,
        world_size=4,
        config={
            "episodic_capacity": 16,
            "episodic_span_length": 4,
            "episodic_key_rep_dim": 8,
            "controller_query_enabled": False,
        },
        model_dim=8,
        all_group=None,
    )
    # Sanity: this is the no-op shape.
    assert train_rank_consumer.cache is None
    assert train_rank_consumer.controller_query_queue == []
    assert train_rank_consumer.tagged_replay_queue == []

    torch.manual_seed(1337)
    model_a = _TinyTokenModel(vocab=16, dim=8)
    torch.manual_seed(1337)
    model_b = _TinyTokenModel(vocab=16, dim=8)
    _assert_state_dicts_equal(model_a, model_b, where="construction")

    inputs = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)
    targets = torch.tensor([[2, 3, 4, 5]], dtype=torch.int64)

    # Run A: episodic OFF (consumer=None). This is the ground truth.
    loss_a = runner._run_train_step(
        model=model_a,
        inputs=inputs,
        targets=targets,
        chunk_size=4,
        precision="fp32",
        ddp_active=False,
        world_size=1,
        rank=0,
        lm_head_backward_mode="single",
        grad_allreduce_mode="bulk",
        is_episodic_rank=False,
        all_group=None,
        episodic_emit=None,
        episodic_consumer=None,
        episodic_replay_logger=None,
        current_step=0,
        embedding_version=0,
    )

    # Run B: episodic ENABLED (consumer attached) but unwired
    # (emit=None, controller_query_enabled=False). Must match Run A
    # bit-identically because the train-rank gradient path doesn't
    # touch the consumer state when emit is None.
    loss_b = runner._run_train_step(
        model=model_b,
        inputs=inputs,
        targets=targets,
        chunk_size=4,
        precision="fp32",
        ddp_active=False,
        world_size=1,
        rank=0,
        lm_head_backward_mode="single",
        grad_allreduce_mode="bulk",
        is_episodic_rank=False,
        all_group=None,
        episodic_emit=None,
        episodic_consumer=train_rank_consumer,
        episodic_replay_logger=None,
        current_step=0,
        embedding_version=0,
    )

    assert torch.equal(loss_a, loss_b), (
        f"loss diverged when consumer attached on train rank: "
        f"{loss_a.item()} vs {loss_b.item()}"
    )
    _assert_grads_equal(model_a, model_b)

    opt_a = torch.optim.SGD(model_a.parameters(), lr=0.1)
    opt_b = torch.optim.SGD(model_b.parameters(), lr=0.1)
    opt_a.step()
    opt_b.step()
    _assert_state_dicts_equal(model_a, model_b, where="after optimizer.step()")

    # Consumer state must remain the untouched no-op shape — train
    # rank should never write to it.
    assert train_rank_consumer.cache is None
    assert train_rank_consumer.controller_query_queue == []
    assert train_rank_consumer.tagged_replay_queue == []
