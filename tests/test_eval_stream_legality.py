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
    loss_before = controller.score_chunk(chunk)
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
    _ = controller.score_chunk(chunk)
    controller.adapt_on_chunk(chunk, optimizer=opt, steps=1)  # must not raise
