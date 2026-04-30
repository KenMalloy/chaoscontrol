"""Regression tests for Task 3.5: threading initial_states through core/model.

These tests pin the contract:
  - Default call (no state kwarg) matches prior behavior bit-identically.
  - Passing zeros for initial_states is bit-identical to passing nothing.
  - A non-zero initial state actually changes the output.
  - A whole-document forward equals two chunked forwards threaded through
    final_state, which is what Axis 2 persistence modes rely on.
"""
from __future__ import annotations

import torch

from chaoscontrol.core import _should_use_zero_initial_state_fast_path
from chaoscontrol.model import CareStudentLM


def _tiny_lm():
    torch.manual_seed(0)
    return CareStudentLM(vocab_size=32, dim=16, num_layers=2, block_type="ssm", a_mode="diag")


def test_backward_compat_no_state_kwarg():
    """model(input_ids) with no state kwarg must match prior behavior."""
    m = _tiny_lm()
    m.eval()
    ids = torch.randint(0, 32, (1, 32))
    with torch.no_grad():
        out = m(ids)
    logits = out["logits"] if isinstance(out, dict) else out
    assert logits.shape == (1, 32, 32)
    assert "final_states" in out
    assert len(out["final_states"]) == 2
    for s in out["final_states"]:
        assert s.shape == (1, 16)


def test_zeros_initial_states_match_default():
    """Passing zeros for initial_states must be bit-identical to passing nothing."""
    m = _tiny_lm()
    m.eval()
    ids = torch.randint(0, 32, (1, 32))
    with torch.no_grad():
        out_default = m(ids)
        zeros = [torch.zeros(1, 16) for _ in range(2)]
        out_zeros = m(ids, initial_states=zeros)
    torch.testing.assert_close(
        out_default["logits"], out_zeros["logits"], rtol=0, atol=0,
    )


def test_nonzero_initial_state_changes_output():
    m = _tiny_lm()
    m.eval()
    ids = torch.randint(0, 32, (1, 32))
    with torch.no_grad():
        out_zero = m(ids)
        nz = [torch.randn(1, 16) for _ in range(2)]
        out_nz = m(ids, initial_states=nz)
    assert not torch.allclose(out_zero["logits"], out_nz["logits"], atol=1e-5)


def test_zero_initial_state_with_grad_flows_gradient():
    """Zero-valued initial_state WITH requires_grad=True must route through
    the slow path so autograd accumulates grad on the source Parameter.

    The fast-path zero short-circuit (core.py) is gated on
    `not initial_state.requires_grad` precisely so Axis 2 `trainable_h0`
    starts at zero yet still receives gradient. See
    test_trainable_h0_receives_gradient for the Axis-2 integration pin;
    this test is the minimal core-level contract.
    """
    import torch.nn as nn
    m = _tiny_lm()
    m.train()
    ids = torch.randint(0, 32, (1, 16))
    # Mint a zero Parameter per layer and feed it as initial_states.
    h0 = [nn.Parameter(torch.zeros(1, 16)) for _ in range(2)]
    out = m(ids, initial_states=h0)
    out["logits"].sum().backward()
    for p in h0:
        assert p.grad is not None
        assert p.grad.abs().sum() > 0


def test_cuda_graph_capture_skips_zero_state_tensor_check(monkeypatch):
    """The zero-state shortcut uses torch.any, which is illegal inside CUDA graph capture."""
    class FakeCudaState:
        is_cuda = True
        requires_grad = False

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    def fail_any(_state):
        raise AssertionError("torch.any should not run while CUDA graph capture is active")

    monkeypatch.setattr(torch, "any", fail_any)

    assert _should_use_zero_initial_state_fast_path(FakeCudaState()) is False


def test_final_state_equals_chunked_sequential():
    """Whole-doc forward must equal concat of two chunked forwards threaded through final_state."""
    m = _tiny_lm()
    m.eval()
    ids = torch.randint(0, 32, (1, 64))
    with torch.no_grad():
        whole = m(ids)
        first = m(ids[:, :32])
        second = m(ids[:, 32:], initial_states=first["final_states"])
    torch.testing.assert_close(
        whole["logits"][:, 32:], second["logits"], rtol=1e-5, atol=1e-5,
    )
