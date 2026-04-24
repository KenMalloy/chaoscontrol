import torch
import pytest
from chaoscontrol.core import ChaosSSMCore


def _make_core(dim: int = 8) -> ChaosSSMCore:
    torch.manual_seed(0)
    return ChaosSSMCore(dim=dim, a_mode="diag")


def test_capture_states_context_manager_records_shape_via_helper():
    core = _make_core(dim=8)
    x = torch.randn(2, 5, 8)
    with core.capture_states() as get_states:
        _ = core._forward_diag_scan(x)
        captured = get_states()
    assert captured is not None, "capture_states must populate states"
    assert captured.shape == (2, 5, 8)
    assert captured.requires_grad is False, "captured states must be detached"


def test_capture_states_context_clears_after_exit():
    core = _make_core(dim=8)
    x = torch.randn(2, 5, 8)
    with core.capture_states() as get_states:
        _ = core._forward_diag_scan(x)
    assert core._captured_states is None
    assert core._capture_states_enabled is False
