import torch
import pytest
from chaoscontrol.core import ChaosSSMCore


def _make_core(dim: int = 8) -> ChaosSSMCore:
    torch.manual_seed(0)
    return ChaosSSMCore(dim=dim, a_mode="diag")


def test_capture_states_context_manager_records_shape_via_helper():
    core = _make_core(dim=8)
    x = torch.randn(2, 5, 8, requires_grad=True)
    with core.capture_states() as get_states:
        _ = core._forward_diag_scan(x)
        captured = get_states()
    assert captured is not None, "capture_states must populate states"
    assert captured.shape == (2, 5, 8)
    # Both checks needed: requires_grad=False is necessary but not sufficient —
    # a detached view also has no grad_fn, which is the stronger invariant.
    assert captured.requires_grad is False, "captured states must be detached"
    assert captured.grad_fn is None, "captured states must have no grad_fn"


def test_capture_states_context_clears_after_exit():
    core = _make_core(dim=8)
    x = torch.randn(2, 5, 8)
    with core.capture_states() as get_states:
        _ = core._forward_diag_scan(x)
    assert core._captured_states is None
    assert core._capture_states_enabled is False


def test_capture_states_clears_when_block_raises():
    core = _make_core(dim=8)
    x = torch.randn(2, 5, 8)

    class _Sentinel(Exception):
        pass

    with pytest.raises(_Sentinel):
        with core.capture_states() as _get_states:
            _ = core._forward_diag_scan(x)
            raise _Sentinel()

    # finally must have run
    assert core._captured_states is None
    assert core._capture_states_enabled is False


def test_capture_via_top_level_forward_diag_fast_path():
    """Production model.encode() path routes through forward()'s inlined
    diag fast-path, not _forward_diag_scan. Must capture there too."""
    core = _make_core(dim=8)
    x = torch.randn(2, 5, 8, requires_grad=True)
    with core.capture_states() as get_states:
        _ = core(x)  # forward, not _forward_diag_scan
        captured = get_states()
    assert captured is not None, "forward() diag fast-path must capture too"
    assert captured.shape == (2, 5, 8)
    assert captured.requires_grad is False, "captured states must be detached"
    assert captured.grad_fn is None, "captured states must have no grad_fn"


def test_capture_is_disabled_by_default_no_overhead_path():
    """Capture is off by default; no attribute should be populated without
    the context manager."""
    core = _make_core(dim=8)
    x = torch.randn(2, 5, 8)
    _ = core(x)
    assert core._captured_states is None
    assert core._capture_states_enabled is False
