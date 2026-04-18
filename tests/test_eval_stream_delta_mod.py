import torch
from chaoscontrol.eval_stream.delta_mod import DeltaModulator
from chaoscontrol.core import ChaosSSMCore


def test_delta_scale_identity_when_one():
    core = ChaosSSMCore(dim=16, a_mode="diag")
    x = torch.randn(2, 4, 16)
    y_base = core.forward(x)
    with DeltaModulator(core, delta_scale=1.0, log_a_shift=0.0):
        y_mod = core.forward(x)
    torch.testing.assert_close(y_base, y_mod)


def test_delta_scale_changes_output():
    core = ChaosSSMCore(dim=16, a_mode="diag")
    x = torch.randn(2, 4, 16)
    y_base = core.forward(x)
    with DeltaModulator(core, delta_scale=2.0, log_a_shift=0.0):
        y_mod = core.forward(x)
    assert not torch.allclose(y_base, y_mod, atol=1e-4)


def test_hooks_removed_after_context():
    core = ChaosSSMCore(dim=16, a_mode="diag")
    assert len(core.delta_proj._forward_hooks) == 0
    with DeltaModulator(core, delta_scale=2.0):
        assert len(core.delta_proj._forward_hooks) == 1
    assert len(core.delta_proj._forward_hooks) == 0


import pytest


def test_log_a_shift_restores_on_exit():
    """log_a must bit-identical revert after __exit__."""
    torch.manual_seed(0)
    core = ChaosSSMCore(dim=16, a_mode="diag")
    orig = core.log_a.detach().clone()
    with DeltaModulator(core, delta_scale=1.0, log_a_shift=0.5):
        pass
    assert torch.equal(core.log_a, orig), "log_a not restored"


def test_log_a_shift_changes_output():
    """Nonzero log_a_shift must change forward output."""
    torch.manual_seed(0)
    core = ChaosSSMCore(dim=16, a_mode="diag")
    x = torch.randn(2, 4, 16)
    y_base = core.forward(x)
    with DeltaModulator(core, delta_scale=1.0, log_a_shift=1.0):
        y_shifted = core.forward(x)
    assert not torch.allclose(y_base, y_shifted, atol=1e-4)


@pytest.mark.parametrize("bad_hint", ["log_a", "log_a+delta_proj", "all"])
def test_log_a_shift_raises_on_log_a_adapt_hint(bad_hint):
    """DeltaModulator refuses to shift log_a when caller is also adapting it."""
    core = ChaosSSMCore(dim=16, a_mode="diag")
    modulator = DeltaModulator(
        core, delta_scale=1.0, log_a_shift=0.5, adapt_set_hint=bad_hint,
    )
    with pytest.raises(ValueError, match="log_a_shift"):
        modulator.__enter__()


def test_log_a_shift_zero_does_not_raise_even_with_log_a_hint():
    """Shift of 0.0 is a no-op; the guard should not fire."""
    core = ChaosSSMCore(dim=16, a_mode="diag")
    with DeltaModulator(core, delta_scale=1.0, log_a_shift=0.0, adapt_set_hint="log_a"):
        pass  # no raise
