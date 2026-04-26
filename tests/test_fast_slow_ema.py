import pytest
import torch

from chaoscontrol.kernels import _cpu_ssm_controller as _ext


def test_fast_slow_ema_blends_only_on_positive_interval_boundary():
    ema = _ext.FastSlowEma(0.25, 64)

    assert ema.event_count == 0
    assert ema.should_blend() is False

    for expected_count in range(1, 64):
        ema.tick_event()
        assert ema.event_count == expected_count
        assert ema.should_blend() is False

    ema.tick_event()

    assert ema.event_count == 64
    assert ema.should_blend() is True


def test_fast_slow_ema_blends_single_value_in_place():
    ema = _ext.FastSlowEma(0.25, 64)
    slow = torch.tensor([1.0], dtype=torch.float32)
    fast = torch.tensor([2.0], dtype=torch.float32)

    ema.blend(slow, fast)

    assert torch.allclose(slow, torch.tensor([1.25], dtype=torch.float32))


def test_fast_slow_ema_blends_vector_in_place():
    ema = _ext.FastSlowEma(0.5, 2)
    slow = torch.tensor([1.0, 3.0, -1.0], dtype=torch.float32)
    fast = torch.tensor([3.0, 1.0, 1.0], dtype=torch.float32)

    ema.blend(slow, fast)

    assert torch.allclose(
        slow,
        torch.tensor([2.0, 2.0, 0.0], dtype=torch.float32),
    )


@pytest.mark.parametrize("alpha", [-0.01, 1.01])
def test_fast_slow_ema_rejects_invalid_alpha(alpha):
    with pytest.raises(ValueError, match="alpha"):
        _ext.FastSlowEma(alpha, 64)


def test_fast_slow_ema_rejects_zero_interval():
    with pytest.raises(ValueError, match="interval"):
        _ext.FastSlowEma(0.25, 0)


def test_fast_slow_ema_rejects_mismatched_tensor_shapes():
    ema = _ext.FastSlowEma(0.25, 64)
    slow = torch.ones(2, dtype=torch.float32)
    fast = torch.ones(3, dtype=torch.float32)

    with pytest.raises(RuntimeError, match="same shape"):
        ema.blend(slow, fast)
