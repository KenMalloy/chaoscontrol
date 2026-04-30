import torch
from chaoscontrol.eval_stream.delta_mod import DeltaModulator
from chaoscontrol.core import CareSSMCore


def test_delta_scale_identity_when_one():
    core = CareSSMCore(dim=16, a_mode="diag")
    x = torch.randn(2, 4, 16)
    y_base = core.forward(x)
    with DeltaModulator(core, delta_scale=1.0, log_a_shift=0.0):
        y_mod = core.forward(x)
    torch.testing.assert_close(y_base, y_mod)


def test_delta_scale_changes_output():
    core = CareSSMCore(dim=16, a_mode="diag")
    x = torch.randn(2, 4, 16)
    y_base = core.forward(x)
    with DeltaModulator(core, delta_scale=2.0, log_a_shift=0.0):
        y_mod = core.forward(x)
    assert not torch.allclose(y_base, y_mod, atol=1e-4)


def test_hooks_removed_after_context():
    core = CareSSMCore(dim=16, a_mode="diag")
    assert len(core.delta_proj._forward_hooks) == 0
    with DeltaModulator(core, delta_scale=2.0):
        assert len(core.delta_proj._forward_hooks) == 1
    assert len(core.delta_proj._forward_hooks) == 0


import pytest


def test_log_a_shift_restores_on_exit():
    """log_a must bit-identical revert after __exit__."""
    torch.manual_seed(0)
    core = CareSSMCore(dim=16, a_mode="diag")
    orig = core.log_a.detach().clone()
    with DeltaModulator(core, delta_scale=1.0, log_a_shift=0.5):
        pass
    assert torch.equal(core.log_a, orig), "log_a not restored"


def test_log_a_shift_changes_output():
    """Nonzero log_a_shift must change forward output."""
    torch.manual_seed(0)
    core = CareSSMCore(dim=16, a_mode="diag")
    x = torch.randn(2, 4, 16)
    y_base = core.forward(x)
    with DeltaModulator(core, delta_scale=1.0, log_a_shift=1.0):
        y_shifted = core.forward(x)
    assert not torch.allclose(y_base, y_shifted, atol=1e-4)


@pytest.mark.parametrize("bad_hint", ["log_a", "log_a+delta_proj", "all"])
def test_log_a_shift_raises_on_log_a_adapt_hint(bad_hint):
    """DeltaModulator refuses to shift log_a when caller is also adapting it."""
    core = CareSSMCore(dim=16, a_mode="diag")
    modulator = DeltaModulator(
        core, delta_scale=1.0, log_a_shift=0.5, adapt_set_hint=bad_hint,
    )
    with pytest.raises(ValueError, match="log_a_shift"):
        modulator.__enter__()


def test_log_a_shift_zero_does_not_raise_even_with_log_a_hint():
    """Shift of 0.0 is a no-op; the guard should not fire."""
    core = CareSSMCore(dim=16, a_mode="diag")
    with DeltaModulator(core, delta_scale=1.0, log_a_shift=0.0, adapt_set_hint="log_a"):
        pass  # no raise


def test_multi_core_model_all_cores_get_hooks():
    """DeltaModulator walks `module.modules()` to find CareSSMCores.
    For a full model with N layers, all N cores must be modulated, not
    just the first. Otherwise the Phase D sweep partially modulates and
    the result is a lie about Δ-rescaling.
    """
    from chaoscontrol.model import CareStudentLM
    torch.manual_seed(0)
    m = CareStudentLM(
        vocab_size=32, dim=16, num_layers=3, block_type="ssm", a_mode="diag",
    )
    # Collect the 3 cores.
    cores = [c for c in m.modules() if isinstance(c, CareSSMCore)]
    assert len(cores) == 3

    # Without modulator, no hooks.
    for c in cores:
        assert len(c.delta_proj._forward_hooks) == 0

    with DeltaModulator(m, delta_scale=2.0):
        for c in cores:
            assert len(c.delta_proj._forward_hooks) == 1, \
                f"core {id(c)} missing hook"

    # After exit, all cleaned up.
    for c in cores:
        assert len(c.delta_proj._forward_hooks) == 0


def test_multi_core_log_a_shift_applied_to_all():
    """Every CareSSMCore's log_a gets shifted on enter and restored on exit."""
    from chaoscontrol.model import CareStudentLM
    torch.manual_seed(0)
    m = CareStudentLM(
        vocab_size=32, dim=16, num_layers=3, block_type="ssm", a_mode="diag",
    )
    cores = [c for c in m.modules() if isinstance(c, CareSSMCore)]
    originals = [c.log_a.detach().clone() for c in cores]

    with DeltaModulator(m, delta_scale=1.0, log_a_shift=0.5):
        # All cores must be shifted.
        for c, orig in zip(cores, originals):
            assert torch.allclose(c.log_a, orig + 0.5, atol=1e-6), \
                "log_a not shifted"

    # After exit, all cores restored.
    for c, orig in zip(cores, originals):
        assert torch.equal(c.log_a, orig), "log_a not restored"


def test_module_without_cores_is_no_op():
    """Graceful no-op on a module that contains no CareSSMCore.

    Regression pin: `_find_cores` uses isinstance filtering, so a module
    hierarchy with no SSM cores must not crash or produce spurious hooks —
    it should just enter/exit as a no-op. Matters for any test harness
    that wraps a pure-attention baseline.
    """
    import torch.nn as nn
    m = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))
    # No CareSSMCore anywhere.
    with DeltaModulator(m, delta_scale=2.0, log_a_shift=0.5):
        pass  # must not raise
