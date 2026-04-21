import torch
import torch.nn as nn
import pytest
from chaoscontrol.eval_stream.legality import LegalityController, LeakDetectedError


class _TinyLM(nn.Module):
    def __init__(self, vocab=32, dim=16):
        super().__init__()
        self.vocab_size = vocab
        self.embed = nn.Embedding(vocab, dim)
        self.lm_head = nn.Linear(dim, vocab, bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        return {"logits": self.lm_head(x)}


class _StateAwareTinyLM(_TinyLM):
    def __init__(self, vocab=32, dim=16):
        super().__init__(vocab=vocab, dim=dim)
        self.initial_state_history: list[list[torch.Tensor] | None] = []

    def forward(self, input_ids, initial_states=None):
        recorded = None
        if initial_states is not None:
            recorded = [s.detach().clone() for s in initial_states]
        self.initial_state_history.append(recorded)
        x = self.embed(input_ids)
        final_states = (
            [initial_states[0] + 1.0]
            if initial_states is not None
            else [x.mean(dim=1)]
        )
        return {"logits": self.lm_head(x), "final_states": final_states}


def _loss(logits, targets):
    import torch.nn.functional as F
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def test_score_is_under_pre_update_weights():
    torch.manual_seed(0)
    model = _TinyLM()
    tokens = torch.randint(0, 32, (1, 64))
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    controller = LegalityController(model, loss_fn=_loss)
    # Score chunk of 32 tokens under frozen weights, then adapt on them
    chunk = tokens[:, :32]
    loss_before, _ = controller.score_chunk(chunk)
    # Capture FULL state snapshot — SGD on model.parameters() updates every
    # param (embed + lm_head), so rolling back only lm_head would leave embed
    # perturbed and the re-forward loss would not match the pre-update score.
    full_snap = {k: v.detach().clone() for k, v in model.state_dict().items()}
    lm_snap = model.lm_head.weight.detach().clone()
    controller.adapt_on_chunk(chunk, optimizer=opt, steps=1)
    # Weights must have CHANGED (we adapted)
    assert not torch.allclose(lm_snap, model.lm_head.weight)
    # The score is the score under the OLD weights — this is validated by
    # re-forwarding under a rolled-back snapshot and checking equality.
    with torch.no_grad():
        model.load_state_dict(full_snap)
        logits = model(chunk)["logits"]
        loss_rollback = _loss(logits[:, :-1], chunk[:, 1:])
    torch.testing.assert_close(loss_before, loss_rollback.item(), rtol=0, atol=1e-6)


def test_leak_detected_when_scoring_under_updated_weights():
    """CONTRACT TEST: a forced leak MUST be detected. If this fails, the harness is invalid."""
    torch.manual_seed(0)
    model = _TinyLM()
    tokens = torch.randint(0, 32, (1, 32))
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    controller = LegalityController(model, loss_fn=_loss, leak_detection=True)

    # Intentional leak: update weights FIRST, then score the same chunk
    controller.adapt_on_chunk(tokens, optimizer=opt, steps=1)
    with pytest.raises(LeakDetectedError):
        # Attempting to score a chunk that was already adapted-on is a leak
        controller.score_chunk(tokens)


def test_leak_detection_disabled_does_not_hash_chunks(monkeypatch):
    """The production path should not CPU-sync chunks for leak hashing.

    Leak detection is a contract-test mode. When it is disabled, score/adapt
    order is enforced by the driver, so hashing every CUDA chunk is pure
    overhead.
    """
    torch.manual_seed(0)
    model = _TinyLM()
    tokens = torch.randint(0, 32, (1, 32))
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    controller = LegalityController(model, loss_fn=_loss, leak_detection=False)

    def _unexpected_hash(_chunk):
        raise AssertionError("hashing should be skipped when leak_detection=False")

    monkeypatch.setattr(LegalityController, "_chunk_hash", staticmethod(_unexpected_hash))

    controller.score_chunk(tokens)
    controller.adapt_on_chunk(tokens, optimizer=opt, steps=1)


def test_empty_chunk_raises_valueerror():
    m = _TinyLM()
    controller = LegalityController(m, loss_fn=_loss)
    with pytest.raises(ValueError):
        controller.score_chunk(torch.randint(0, 32, (1, 1)))  # length < 2


def test_cross_doc_re_adapt_same_bytes_is_ok_after_mark_new_epoch():
    """Same chunk content appearing in two different docs is not a leak.
    mark_new_epoch() clears tracking; re-adapting then re-scoring the same
    bytes in the next doc must succeed without raising.
    """
    torch.manual_seed(0)
    m = _TinyLM()
    opt = torch.optim.SGD(m.parameters(), lr=0.01)
    controller = LegalityController(m, loss_fn=_loss, leak_detection=True)

    chunk = torch.randint(0, 32, (1, 16))
    controller.adapt_on_chunk(chunk, optimizer=opt, steps=1)
    controller.mark_new_epoch()
    # Next doc: same bytes reappear, legitimate score-then-adapt order.
    _, _ = controller.score_chunk(chunk)
    controller.adapt_on_chunk(chunk, optimizer=opt, steps=1)  # must not raise


def test_adapt_on_chunk_with_zero_steps_returns_none():
    """adapt_on_chunk(steps=0) is a no-op; returns None and does not touch
    the optimizer, the model, or _adapted_chunks tracking.

    Driver uses this path when `adapt_set="none"` or `steps_per_chunk=0`,
    so a silent side-effect here would corrupt Phase B (which relies on
    adapt_set=none producing zero weight change).
    """
    torch.manual_seed(0)
    m = _TinyLM()
    pre_weights = {k: v.detach().clone() for k, v in m.state_dict().items()}
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    controller = LegalityController(m, loss_fn=_loss, leak_detection=True)

    chunk = torch.randint(0, 32, (1, 32))
    result = controller.adapt_on_chunk(chunk, optimizer=opt, steps=0)
    assert result is None
    # Model must be bit-identical — no optimizer step happened.
    for k, v in m.state_dict().items():
        assert torch.equal(pre_weights[k], v), f"{k} mutated on steps=0"
    # And the chunk must not have been marked adapted — leak detection stays
    # armed for a real future adapt.
    controller.score_chunk(chunk)  # must not raise


def test_adapt_on_chunk_zeros_grad_to_none_and_syncs_loss_once(monkeypatch):
    torch.manual_seed(0)
    model = _TinyLM()
    controller = LegalityController(model, loss_fn=_loss)
    chunk = torch.randint(0, 32, (1, 32))
    zero_grad_args: list[bool] = []

    class _RecordingOptimizer:
        def zero_grad(self, *, set_to_none: bool = False):
            zero_grad_args.append(set_to_none)
            for p in model.parameters():
                p.grad = None

        def step(self):
            pass

    original_item = torch.Tensor.item
    item_calls = 0

    def _counting_item(self):
        nonlocal item_calls
        item_calls += 1
        return original_item(self)

    monkeypatch.setattr(torch.Tensor, "item", _counting_item)

    result = controller.adapt_on_chunk(
        chunk,
        optimizer=_RecordingOptimizer(),
        steps=3,
    )

    assert result is not None
    assert zero_grad_args == [True, True, True]
    assert item_calls == 1


def test_score_chunk_returns_tuple_with_loss_as_first_element():
    """Contract pin: score_chunk returns (loss, ...). The driver unpacks
    the first element as the scalar loss. A refactor that swaps the
    tuple order would silently produce garbage bpb values.
    """
    m = _TinyLM()
    controller = LegalityController(m, loss_fn=_loss)
    chunk = torch.randint(0, 32, (1, 32))
    result = controller.score_chunk(chunk)
    # Returned tuple must have at least one element; first is a Python float
    # (cross-entropy summed over the chunk).
    assert isinstance(result, tuple)
    loss = result[0]
    assert isinstance(loss, float)
    # Cross-entropy of a random uniform lm_head over random tokens should be
    # close to log(vocab_size) = log(32) ≈ 3.47 per token; with 31 targets,
    # the summed value is ≈ 107 (SGD not run here, just checking sanity).
    # Just pin it's a positive finite number.
    assert loss > 0 and loss < 1e6


def test_adapt_on_chunk_threads_same_initial_states_as_score():
    """Carry-state TTT must adapt in the same recurrent context it scored.

    Regression target: adapt_on_chunk() used to ignore ``initial_states`` and
    silently re-run the chunk from zero state, which makes carry-state
    score-before-update results incomparable to the subsequent gradient step.
    """
    torch.manual_seed(0)
    model = _StateAwareTinyLM()
    controller = LegalityController(model, loss_fn=_loss)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    chunk = torch.randint(0, 32, (1, 32))
    init_state = [torch.randn(1, 16)]

    controller.score_chunk(chunk, initial_states=init_state)
    controller.adapt_on_chunk(
        chunk,
        optimizer=opt,
        steps=1,
        initial_states=init_state,
    )

    assert len(model.initial_state_history) == 2
    scored_state, adapted_state = model.initial_state_history
    assert scored_state is not None and adapted_state is not None
    torch.testing.assert_close(scored_state[0], init_state[0])
    torch.testing.assert_close(adapted_state[0], init_state[0])
