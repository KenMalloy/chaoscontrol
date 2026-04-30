import torch
from chaoscontrol.eval_stream.persistence import (
    StateManager, attach_trainable_h0, detach_trainable_h0,
)
from chaoscontrol.model import CareStudentLM


def _tiny_ssm_lm():
    return CareStudentLM(vocab_size=32, dim=16, num_layers=2, block_type="ssm", a_mode="diag")


def test_reset_mode_zeros_state():
    m = _tiny_ssm_lm()
    sm = StateManager(m, persistence_mode="reset")
    sm.start_doc(doc_id=1, batch_size=1)
    state = sm.get_state()
    assert all(torch.all(s == 0) for s in state)


def test_carry_mode_preserves_state_across_docs():
    m = _tiny_ssm_lm()
    sm = StateManager(m, persistence_mode="carry_state")
    sm.start_doc(doc_id=0, batch_size=1)
    sm.set_state([torch.randn(1, 16), torch.randn(1, 16)])
    prev = [s.clone() for s in sm.get_state()]
    sm.start_doc(doc_id=1, batch_size=1)  # should NOT reset
    for a, b in zip(prev, sm.get_state()):
        torch.testing.assert_close(a, b)


def test_trainable_h0_is_learnable():
    m = _tiny_ssm_lm()
    attach_trainable_h0(m)
    # h0 should appear in model.named_parameters()
    names = [n for n, _ in m.named_parameters()]
    h0_names = [n for n in names if "trainable_h0" in n]
    assert len(h0_names) == 2  # 2 layers
    detach_trainable_h0(m)
    names = [n for n, _ in m.named_parameters()]
    assert not any("trainable_h0" in n for n in names)


def test_detach_trainable_h0_clears_state_dict():
    """After detach, state_dict must not contain any _trainable_h0 keys —
    otherwise saving/loading checkpoints leaks eval-only state.
    """
    m = _tiny_ssm_lm()
    attach_trainable_h0(m)
    assert any("_trainable_h0" in k for k in m.state_dict().keys())
    detach_trainable_h0(m)
    assert not any("_trainable_h0" in k for k in m.state_dict().keys())


def test_trainable_h0_receives_gradient():
    """Pre-req for Task 3.5 integration: with h0 threaded through
    initial_states, loss.backward() must accumulate grad on _trainable_h0.
    """
    m = _tiny_ssm_lm()
    attach_trainable_h0(m)
    ids = torch.randint(0, 32, (1, 16))
    sm = StateManager(m, persistence_mode="trainable_h0")
    sm.start_doc(doc_id=0, batch_size=1)
    out = m(ids, initial_states=sm.get_state())
    loss = out["logits"].sum()
    loss.backward()
    for core in m.modules():
        from chaoscontrol.core import CareSSMCore
        if isinstance(core, CareSSMCore):
            assert core._trainable_h0.grad is not None
            assert core._trainable_h0.grad.abs().sum() > 0
