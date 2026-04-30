"""Equivalence tests for ``CareStudentLM.encode()`` vs ``forward()``.

``encode()`` is the new entry point used by ``train_ssm`` — it runs
every pre-LM-head computation (embed, wernicke, outer-model reads,
layers, posterior) and returns the hidden ``(B, T, dim)`` tensor plus
whatever auxiliary outputs the encoder produced. The existing
``forward()`` is frozen for reproducibility of every prior experiment,
so ``encode()`` must produce byte-identical hidden states on every
config that ``forward()`` already supports.

These tests exercise the bare-SSM config (the only config ``train_ssm``
uses) because that's what the new path has to get right. If a future
experiment wires ``encode()`` into a wernicke- or posterior-enabled
config, this file is the place to add the coverage for it.
"""
from __future__ import annotations

import pytest
import torch

from chaoscontrol.model import CareStudentLM


@pytest.fixture
def bare_ssm_model() -> CareStudentLM:
    torch.manual_seed(123)
    model = CareStudentLM(
        vocab_size=64,
        dim=16,
        num_layers=2,
        ff_mult=2,
        a_mode="diag",
    )
    model.eval()  # deterministic path — no dropout surprises
    return model


def _make_input(batch: int, seq: int, vocab: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (batch, seq), generator=g)


class TestEncodeBareSSMEquivalence:
    """On the bare-SSM config, encode + manual decode must equal forward."""

    def test_logits_bit_identical_to_forward(self, bare_ssm_model: CareStudentLM) -> None:
        # Build identical inputs; run both paths; compare logits exactly.
        model = bare_ssm_model
        inputs = _make_input(batch=2, seq=16, vocab=64, seed=1)

        with torch.no_grad():
            forward_out = model(inputs)
            hidden = model.encode(inputs)
            logits_from_encode = model.lm_head(model.final_norm(hidden))

        assert torch.equal(forward_out["logits"], logits_from_encode), (
            "logits from encode() + final_norm + lm_head must bit-match forward()"
        )

    def test_encode_returns_hidden_tensor(self, bare_ssm_model: CareStudentLM) -> None:
        # encode() returns the pre-final_norm hidden state. forward()["hidden"]
        # in the bare-SSM path is the same tensor (see model.py:1051).
        model = bare_ssm_model
        inputs = _make_input(batch=2, seq=16, vocab=64, seed=2)

        with torch.no_grad():
            forward_out = model(inputs)
            hidden = model.encode(inputs)

        assert torch.equal(forward_out["hidden"], hidden), (
            "encode() must return the same hidden tensor that forward() "
            "surfaces in out['hidden']"
        )

    def test_encode_preserves_grad_flow(self, bare_ssm_model: CareStudentLM) -> None:
        # Gradients from encode() output must propagate back to model
        # parameters the same way forward()['hidden'] does.
        model = bare_ssm_model
        inputs = _make_input(batch=2, seq=8, vocab=64, seed=3)

        # Capture reference param grads via forward()
        model.zero_grad(set_to_none=True)
        forward_out = model(inputs)
        # Sum over hidden gives a scalar that backprops through the full encoder.
        forward_out["hidden"].sum().backward()
        ref_grads = {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None}

        model.zero_grad(set_to_none=True)
        hidden = model.encode(inputs)
        hidden.sum().backward()
        new_grads = {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None}

        # Same set of parameters get gradients
        assert set(ref_grads.keys()) == set(new_grads.keys()), (
            f"parameter grad coverage differs: "
            f"forward-only: {set(ref_grads) - set(new_grads)}, "
            f"encode-only: {set(new_grads) - set(ref_grads)}"
        )
        # Same gradient values (bit-identical — same graph)
        for name in ref_grads:
            assert torch.equal(ref_grads[name], new_grads[name]), (
                f"param {name!r} gradient differs between forward() and encode()"
            )

    def test_encode_can_return_final_states(self, bare_ssm_model: CareStudentLM) -> None:
        model = bare_ssm_model
        inputs = _make_input(batch=2, seq=16, vocab=64, seed=4)

        with torch.no_grad():
            forward_out = model(inputs)
            hidden, final_states = model.encode(inputs, return_final_states=True)

        assert torch.equal(forward_out["hidden"], hidden)
        assert len(forward_out["final_states"]) == len(final_states)
        for from_forward, from_encode in zip(forward_out["final_states"], final_states):
            assert torch.equal(from_forward, from_encode)

    def test_encode_accepts_initial_states(self, bare_ssm_model: CareStudentLM) -> None:
        model = bare_ssm_model
        inputs = _make_input(batch=2, seq=16, vocab=64, seed=5)
        init_states = [torch.full((2, 16), 2.0) for _ in range(len(model.layers))]

        with torch.no_grad():
            forward_out = model(inputs, initial_states=init_states)
            encoded_hidden, encoded_final_states = model.encode(
                inputs,
                initial_states=init_states,
                return_final_states=True,
            )

        assert torch.equal(forward_out["hidden"], encoded_hidden)
        assert len(forward_out["final_states"]) == len(encoded_final_states)
        for from_forward, from_encode in zip(forward_out["final_states"], encoded_final_states):
            assert torch.equal(from_forward, from_encode)
