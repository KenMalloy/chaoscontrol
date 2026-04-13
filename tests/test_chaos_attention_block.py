"""Tests for ChaosAttentionBlock — Exp 19 Phase 2 scientific control.

ChaosAttentionBlock is a causal multi-head self-attention sibling to
ChaosSSMBlock. It exists so Exp 19 Phase 2 can run the same training stack
with attention substituted for SSM and measure per-token learning efficiency.

These tests cover the block contract:
  - Shape parity with ChaosSSMBlock (forward and return_jacobian_stats).
  - Causal masking: outputs at position t must not depend on inputs at
    positions > t.
  - Backward pass produces non-zero gradients on every learnable tensor.
  - Shape interchangeability with ChaosSSMBlock inside ChaosStudentLM via
    the `block_type="attention"` constructor flag.
  - No Flash Attention imports — SDPA only.
"""
from __future__ import annotations

import importlib
import sys
import unittest

import torch
import torch.nn.functional as F

from chaoscontrol.model import (
    ChaosAttentionBlock,
    ChaosSSMBlock,
    ChaosStudentLM,
)


class TestChaosAttentionBlockForwardShape(unittest.TestCase):
    def test_forward_shape_matches_input(self) -> None:
        block = ChaosAttentionBlock(dim=32, ff_mult=2, num_heads=4)
        x = torch.randn(3, 11, 32)
        out = block(x)
        assert out.shape == (3, 11, 32), f"shape mismatch: {out.shape}"

    def test_forward_dtype_preserved(self) -> None:
        block = ChaosAttentionBlock(dim=32, ff_mult=2, num_heads=4)
        x = torch.randn(2, 8, 32)
        out = block(x)
        assert out.dtype == x.dtype

    def test_return_jacobian_stats_shape_contract(self) -> None:
        """Match ChaosSSMBlock's return contract when return_jacobian_stats=True."""
        block = ChaosAttentionBlock(dim=32, ff_mult=2, num_heads=4)
        x = torch.randn(2, 8, 32)
        result = block(x, return_jacobian_stats=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        out, stats = result
        assert out.shape == (2, 8, 32)
        assert isinstance(stats, dict)
        assert "lambda_max" in stats
        assert "sv_log_var" in stats

    def test_init_signature_shape_matches_ssm_block(self) -> None:
        """ChaosAttentionBlock(dim, ff_mult=...) mirrors ChaosSSMBlock(dim, ff_mult=...)."""
        dim = 32
        attn = ChaosAttentionBlock(dim, ff_mult=2, num_heads=4)
        ssm = ChaosSSMBlock(dim, ff_mult=2, a_mode="diag", rich_b_mode="none")
        x = torch.randn(2, 8, dim)
        attn_out = attn(x)
        ssm_out = ssm(x)
        assert attn_out.shape == ssm_out.shape

    def test_dim_not_divisible_by_heads_raises(self) -> None:
        with self.assertRaises(ValueError):
            ChaosAttentionBlock(dim=32, ff_mult=2, num_heads=5)


class TestChaosAttentionBlockCausality(unittest.TestCase):
    """A causal block's output at position t must be independent of inputs at
    positions > t. We test this empirically: mutate inputs at positions > t
    and check that outputs at positions <= t are unchanged.
    """

    def _make_block(self) -> ChaosAttentionBlock:
        torch.manual_seed(0)
        block = ChaosAttentionBlock(dim=16, ff_mult=2, num_heads=4)
        block.eval()  # disable dropout for deterministic comparison
        return block

    def test_future_tokens_do_not_affect_past(self) -> None:
        block = self._make_block()
        seq_len = 8
        x = torch.randn(2, seq_len, 16)
        with torch.no_grad():
            out_ref = block(x)

        # For every position t in [0, seq_len - 2], zero out positions > t
        # and confirm that outputs at positions [0..t] are byte-identical to
        # the reference run. Everything in the (future) region we perturbed is
        # of course expected to differ.
        for t in range(seq_len - 1):
            x_mut = x.clone()
            x_mut[:, t + 1 :, :] = 0.0
            with torch.no_grad():
                out_mut = block(x_mut)
            past = out_mut[:, : t + 1, :]
            past_ref = out_ref[:, : t + 1, :]
            # Allow tight tolerance for any floating-point reassociation SDPA
            # might do, but the causal invariant is strict; use a very small
            # atol to catch real causality bugs.
            assert torch.allclose(past, past_ref, atol=1e-6, rtol=1e-5), (
                f"causality violated at t={t}: "
                f"max diff = {(past - past_ref).abs().max().item():.3e}"
            )

    def test_random_future_noise_does_not_affect_past(self) -> None:
        """Stronger variant: replace future tokens with fresh random noise.
        This probes attention leakage more aggressively than zeroing.
        """
        block = self._make_block()
        seq_len = 6
        x = torch.randn(2, seq_len, 16)
        with torch.no_grad():
            out_ref = block(x)

        for t in range(seq_len - 1):
            x_mut = x.clone()
            x_mut[:, t + 1 :, :] = torch.randn_like(x_mut[:, t + 1 :, :])
            with torch.no_grad():
                out_mut = block(x_mut)
            past = out_mut[:, : t + 1, :]
            past_ref = out_ref[:, : t + 1, :]
            assert torch.allclose(past, past_ref, atol=1e-6, rtol=1e-5), (
                f"causality violated at t={t} (random perturbation): "
                f"max diff = {(past - past_ref).abs().max().item():.3e}"
            )


class TestChaosAttentionBlockBackward(unittest.TestCase):
    def test_backward_produces_nonzero_grads_on_every_learnable(self) -> None:
        torch.manual_seed(0)
        block = ChaosAttentionBlock(dim=32, ff_mult=2, num_heads=4)
        x = torch.randn(2, 8, 32, requires_grad=True)
        out = block(x)
        loss = out.pow(2).mean()
        loss.backward()

        # Inputs get gradient.
        assert x.grad is not None
        assert x.grad.abs().sum().item() > 0

        # Every named parameter of the block has non-zero gradient.
        offenders: list[str] = []
        for name, p in block.named_parameters():
            if p.grad is None:
                offenders.append(f"{name}: grad is None")
                continue
            if p.grad.abs().sum().item() == 0.0:
                offenders.append(f"{name}: grad is zero")
        assert not offenders, "parameters with missing/zero grad: " + ", ".join(offenders)

    def test_qkv_and_out_proj_receive_nonzero_grads_explicitly(self) -> None:
        """Sanity check on the named projections called out in the task spec."""
        torch.manual_seed(0)
        block = ChaosAttentionBlock(dim=32, ff_mult=2, num_heads=4)
        x = torch.randn(2, 8, 32)
        loss = block(x).pow(2).mean()
        loss.backward()
        assert block.qkv_proj.weight.grad is not None
        assert block.qkv_proj.weight.grad.abs().sum().item() > 0
        assert block.out_proj.weight.grad is not None
        assert block.out_proj.weight.grad.abs().sum().item() > 0
        assert block.ff.fc.weight.grad is not None
        assert block.ff.fc.weight.grad.abs().sum().item() > 0
        assert block.ff.proj.weight.grad is not None
        assert block.ff.proj.weight.grad.abs().sum().item() > 0


class TestChaosStudentLMWithAttentionBlock(unittest.TestCase):
    """Interchangeability: build ChaosStudentLM with ChaosAttentionBlock layers
    and verify the forward contract is identical to the SSM path.
    """

    def test_student_lm_attention_forward_shape_matches_ssm(self) -> None:
        vocab_size, dim, num_layers = 64, 32, 2
        ssm_model = ChaosStudentLM(
            vocab_size=vocab_size,
            dim=dim,
            num_layers=num_layers,
            ff_mult=2,
            block_type="ssm",
            a_mode="diag",
            rich_b_mode="none",
        )
        attn_model = ChaosStudentLM(
            vocab_size=vocab_size,
            dim=dim,
            num_layers=num_layers,
            ff_mult=2,
            block_type="attention",
            attention_num_heads=4,
        )
        ids = torch.randint(0, vocab_size, (2, 16))
        ssm_out = ssm_model(ids)
        attn_out = attn_model(ids)
        assert ssm_out["logits"].shape == attn_out["logits"].shape
        assert attn_out["logits"].shape == (2, 16, vocab_size)
        assert attn_out["hidden"].shape == (2, 16, dim)

    def test_student_lm_attention_uses_attention_block(self) -> None:
        model = ChaosStudentLM(
            vocab_size=64,
            dim=32,
            num_layers=2,
            ff_mult=2,
            block_type="attention",
            attention_num_heads=4,
        )
        for layer in model.layers:
            assert isinstance(layer, ChaosAttentionBlock)
        assert model.block_type == "attention"

    def test_student_lm_attention_gradients_flow(self) -> None:
        model = ChaosStudentLM(
            vocab_size=64,
            dim=32,
            num_layers=2,
            ff_mult=2,
            block_type="attention",
            attention_num_heads=4,
        )
        ids = torch.randint(0, 64, (2, 16))
        out = model(ids)
        loss = F.cross_entropy(out["logits"].reshape(-1, 64), ids.reshape(-1))
        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in model.parameters()
        )
        assert has_grad

    def test_unknown_block_type_raises(self) -> None:
        with self.assertRaises(ValueError):
            ChaosStudentLM(
                vocab_size=64, dim=32, num_layers=2, ff_mult=2,
                block_type="lstm",
            )

    def test_student_lm_attention_causality(self) -> None:
        """Full LM is causal when ChaosAttentionBlock is used.

        Perturb a single future input embedding (by replacing the token id at
        position t+1) and check that logits at positions <= t are unchanged.
        This ties the block-level causality test to the end-to-end model path.
        """
        torch.manual_seed(0)
        model = ChaosStudentLM(
            vocab_size=64,
            dim=32,
            num_layers=2,
            ff_mult=2,
            block_type="attention",
            attention_num_heads=4,
        )
        model.eval()
        ids = torch.randint(0, 64, (1, 8))
        with torch.no_grad():
            out_ref = model(ids)["logits"]

        for t in range(6):
            ids_mut = ids.clone()
            # Replace the token at position t+1 with a different id.
            ids_mut[0, t + 1] = (ids[0, t + 1].item() + 1) % 64
            with torch.no_grad():
                out_mut = model(ids_mut)["logits"]
            past_ref = out_ref[:, : t + 1, :]
            past_mut = out_mut[:, : t + 1, :]
            assert torch.allclose(past_mut, past_ref, atol=1e-5, rtol=1e-5), (
                f"end-to-end causality violated at t={t}: "
                f"max diff = {(past_mut - past_ref).abs().max().item():.3e}"
            )


class TestNoFlashAttentionImport(unittest.TestCase):
    """Task constraint: ChaosAttentionBlock must not depend on Flash Attention.
    We verify by asserting no flash_attn module is loaded after importing
    chaoscontrol.model.
    """

    def test_no_flash_attn_in_sys_modules_after_import(self) -> None:
        # Force a clean import of the model module.
        if "chaoscontrol.model" in sys.modules:
            del sys.modules["chaoscontrol.model"]
        importlib.import_module("chaoscontrol.model")
        flash_modules = [
            name for name in sys.modules if name.startswith("flash_attn")
        ]
        assert flash_modules == [], (
            f"ChaosAttentionBlock transitively imported flash_attn modules: "
            f"{flash_modules}"
        )


if __name__ == "__main__":
    unittest.main()
