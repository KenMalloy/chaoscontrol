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
