"""Parity tests for FusedChaosSSMBlock in chaoscontrol.core_fused.

Exp 18 Test 8: kernel fusion for the block's post-scan hot path.

Load-bearing: if any of these parity checks drifts, the fusion is wrong
and must be scrapped. Template follows
`tests/test_core.py::TestChunkedDiagScan::test_chunked_backend_gradients_match_loop`.

The fused block shares its ChaosSSMCore and submodules with the unfused
ChaosSSMBlock, so forward outputs and backward gradients must agree to
within numerical noise across float32 and bf16 for both single-forward
parity and multi-step training trajectories.

The tests force `CHAOSCONTROL_POST_SCAN_BACKEND=eager` and
`CHAOSCONTROL_DIAG_SCAN_BACKEND=python` so we never touch Inductor
during the parity test — the eager fused function is written to be
algebraically identical to the unfused block, so any drift would
indicate an algorithmic bug in the fused path, not a compiler
numerics regression. (Inductor numerics vs eager are a separate
concern handled by PyTorch's own testing.)
"""
from __future__ import annotations

import os
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F


def _force_eager_backends() -> None:
    os.environ["CHAOSCONTROL_POST_SCAN_BACKEND"] = "eager"
    os.environ["CHAOSCONTROL_DIAG_SCAN_BACKEND"] = "python"


class TestFusedBlockForwardParity(unittest.TestCase):
    """Forward-pass parity between ChaosSSMBlock and FusedChaosSSMBlock."""

    def setUp(self) -> None:
        _force_eager_backends()
        # Import after env setup so the resolver picks up eager backend.
        from chaoscontrol.core_fused import FusedChaosSSMBlock  # noqa: F401
        from chaoscontrol.model import ChaosSSMBlock  # noqa: F401

    def test_forward_parity_float32(self) -> None:
        from chaoscontrol.core_fused import FusedChaosSSMBlock
        from chaoscontrol.model import ChaosSSMBlock

        torch.manual_seed(0)
        block = ChaosSSMBlock(dim=32, ff_mult=2, a_mode="diag")
        fused = FusedChaosSSMBlock.from_unfused(block)

        x = torch.randn(4, 16, 32)
        y_base = block(x)
        y_fused = fused(x)
        max_diff = (y_base - y_fused).abs().max().item()
        # Pure float32 eager execution of algebraically identical code
        # should be bit-exact.
        assert max_diff < 1e-6, f"float32 forward drift: {max_diff:.2e}"
        assert y_base.dtype == y_fused.dtype
        assert y_base.shape == y_fused.shape

    def test_forward_parity_bf16(self) -> None:
        from chaoscontrol.core_fused import FusedChaosSSMBlock
        from chaoscontrol.model import ChaosSSMBlock

        torch.manual_seed(1)
        block = ChaosSSMBlock(dim=32, ff_mult=2, a_mode="diag").to(torch.bfloat16)
        fused = FusedChaosSSMBlock.from_unfused(block)
        assert fused.ff.fc.weight.dtype == torch.bfloat16, (
            "from_unfused should inherit source block dtype"
        )

        x = torch.randn(4, 16, 32, dtype=torch.bfloat16)
        y_base = block(x)
        y_fused = fused(x)
        assert y_base.dtype == torch.bfloat16
        assert y_fused.dtype == torch.bfloat16
        max_diff = (y_base.float() - y_fused.float()).abs().max().item()
        # Algebraically identical ops, so even bf16 should be bit-exact
        # (same rounding at every op).
        assert max_diff < 1e-3, f"bf16 forward drift: {max_diff:.2e}"

    def test_forward_shapes_various_configs(self) -> None:
        """Run a few (batch, seq, dim, ff_mult) configs."""
        from chaoscontrol.core_fused import FusedChaosSSMBlock
        from chaoscontrol.model import ChaosSSMBlock

        cases = [
            (1, 8, 16, 2),
            (2, 32, 32, 4),
            (3, 17, 24, 2),  # non-multiple-of-chunk seq
        ]
        for B, T, D, M in cases:
            torch.manual_seed(B * 1000 + T)
            block = ChaosSSMBlock(dim=D, ff_mult=M, a_mode="diag")
            fused = FusedChaosSSMBlock.from_unfused(block)
            x = torch.randn(B, T, D)
            y_base = block(x)
            y_fused = fused(x)
            assert y_fused.shape == y_base.shape
            max_diff = (y_base - y_fused).abs().max().item()
            assert max_diff < 1e-6, (
                f"(B={B},T={T},D={D},M={M}) forward drift: {max_diff:.2e}"
            )


class TestFusedBlockGradientParity(unittest.TestCase):
    """Backward-pass parity between ChaosSSMBlock and FusedChaosSSMBlock.

    Load-bearing: if gradient magnitudes or directions diverge here the
    fusion changes the computational graph and cannot be used for
    training. Template matches
    `TestChunkedDiagScan::test_chunked_backend_gradients_match_loop`.
    """

    def setUp(self) -> None:
        _force_eager_backends()

    def _compare_grads(
        self,
        m_base: nn.Module,
        m_fused: nn.Module,
        x_base: torch.Tensor,
        x_fused: torch.Tensor,
        tol_input: float,
        tol_param: float,
    ) -> None:
        input_diff = (x_base.grad - x_fused.grad).abs().max().item()
        input_scale = x_base.grad.abs().max().item() + 1e-12
        input_rel = input_diff / input_scale
        assert input_rel < tol_input, (
            f"input grad drift: abs={input_diff:.2e}, rel={input_rel:.2e}"
        )

        base_params = dict(m_base.named_parameters())
        fused_params = dict(m_fused.named_parameters())
        assert set(base_params.keys()) == set(fused_params.keys()), (
            f"param names differ: base={sorted(base_params.keys())}, "
            f"fused={sorted(fused_params.keys())}"
        )
        for name, p_base in base_params.items():
            p_fused = fused_params[name]
            if p_base.grad is None and p_fused.grad is None:
                continue
            assert p_base.grad is not None, f"base grad None for {name}"
            assert p_fused.grad is not None, f"fused grad None for {name}"
            diff = (p_base.grad - p_fused.grad).abs().max().item()
            scale = p_base.grad.abs().max().item() + 1e-12
            rel = diff / scale
            assert rel < tol_param, (
                f"{name} grad drift: abs={diff:.2e}, rel={rel:.2e}"
            )

    def test_gradient_parity_float32(self) -> None:
        from chaoscontrol.core_fused import FusedChaosSSMBlock
        from chaoscontrol.model import ChaosSSMBlock

        torch.manual_seed(2)
        block = ChaosSSMBlock(dim=32, ff_mult=2, a_mode="diag")
        fused = FusedChaosSSMBlock.from_unfused(block)

        x_base = torch.randn(4, 16, 32, requires_grad=True)
        x_fused = x_base.detach().clone().requires_grad_(True)

        y_base = block(x_base)
        y_fused = fused(x_fused)
        (y_base ** 2).sum().backward()
        (y_fused ** 2).sum().backward()

        self._compare_grads(
            block, fused, x_base, x_fused, tol_input=1e-6, tol_param=1e-6,
        )

    def test_gradient_parity_bf16(self) -> None:
        from chaoscontrol.core_fused import FusedChaosSSMBlock
        from chaoscontrol.model import ChaosSSMBlock

        torch.manual_seed(3)
        block = ChaosSSMBlock(dim=32, ff_mult=2, a_mode="diag").to(torch.bfloat16)
        fused = FusedChaosSSMBlock.from_unfused(block)

        x_base = torch.randn(4, 16, 32, dtype=torch.bfloat16, requires_grad=True)
        x_fused = x_base.detach().clone().requires_grad_(True)

        y_base = block(x_base)
        y_fused = fused(x_fused)
        # Compute loss in float32 to reduce noise in the .backward()
        # gradient magnitudes — we're testing parity of the *path*, not
        # the numerical accuracy of bf16 arithmetic.
        (y_base.float() ** 2).sum().backward()
        (y_fused.float() ** 2).sum().backward()

        # bf16 allows a bit more slack than float32 but algebraically
        # identical ops should still be effectively bit-exact.
        self._compare_grads(
            block, fused, x_base, x_fused, tol_input=1e-3, tol_param=1e-3,
        )


class TestFusedBlockTrainingParity(unittest.TestCase):
    """End-to-end toy LM training parity.

    Single-forward parity can pass while multi-step training diverges
    if an accumulation or autograd ordering subtlety exists. This test
    runs 30 optimizer steps on a toy LM and asserts loss trajectories
    agree throughout.
    """

    def setUp(self) -> None:
        _force_eager_backends()

    def _make_toy_lm(self, use_fused: bool, *, dim: int, vocab: int) -> nn.Module:
        from chaoscontrol.core_fused import FusedChaosSSMBlock
        from chaoscontrol.model import ChaosSSMBlock

        cls = FusedChaosSSMBlock if use_fused else ChaosSSMBlock

        class ToyLM(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed = nn.Embedding(vocab, dim)
                self.block = cls(dim=dim, ff_mult=2, a_mode="diag")
                self.head = nn.Linear(dim, vocab, bias=False)

            def forward(self, ids: torch.Tensor) -> torch.Tensor:
                return self.head(self.block(self.embed(ids)))

        return ToyLM()

    def _run_toy_training(
        self, *, dtype: torch.dtype, steps: int = 30
    ) -> tuple[list[float], list[float]]:
        vocab = 32
        dim = 16
        seq = 12
        batch = 3

        torch.manual_seed(0)
        m_base = self._make_toy_lm(use_fused=False, dim=dim, vocab=vocab)
        torch.manual_seed(0)
        m_fused = self._make_toy_lm(use_fused=True, dim=dim, vocab=vocab)
        # Force weight parity regardless of init order.
        m_fused.load_state_dict(m_base.state_dict())

        if dtype == torch.bfloat16:
            m_base = m_base.to(torch.bfloat16)
            m_fused = m_fused.to(torch.bfloat16)

        torch.manual_seed(100)
        tokens = torch.randint(0, vocab, (batch, seq + 1))

        opt_base = torch.optim.AdamW(m_base.parameters(), lr=1e-3)
        opt_fused = torch.optim.AdamW(m_fused.parameters(), lr=1e-3)

        losses_base: list[float] = []
        losses_fused: list[float] = []
        for _ in range(steps):
            inp = tokens[:, :-1]
            tgt = tokens[:, 1:]

            logits_base = m_base(inp)
            logits_fused = m_fused(inp)
            loss_base = F.cross_entropy(
                logits_base.reshape(-1, vocab).float(), tgt.reshape(-1)
            )
            loss_fused = F.cross_entropy(
                logits_fused.reshape(-1, vocab).float(), tgt.reshape(-1)
            )

            opt_base.zero_grad(set_to_none=True)
            opt_fused.zero_grad(set_to_none=True)
            loss_base.backward()
            loss_fused.backward()
            opt_base.step()
            opt_fused.step()

            losses_base.append(loss_base.item())
            losses_fused.append(loss_fused.item())

        return losses_base, losses_fused

    def test_toy_training_float32_bit_exact(self) -> None:
        losses_base, losses_fused = self._run_toy_training(dtype=torch.float32, steps=30)
        max_diff = max(abs(a - b) for a, b in zip(losses_base, losses_fused))
        assert max_diff < 1e-6, (
            f"float32 toy training loss drift across 30 steps: {max_diff:.2e}\n"
            f"base:  {losses_base}\nfused: {losses_fused}"
        )

    def test_toy_training_bf16_within_noise(self) -> None:
        losses_base, losses_fused = self._run_toy_training(dtype=torch.bfloat16, steps=30)
        max_diff = max(abs(a - b) for a, b in zip(losses_base, losses_fused))
        # bf16 is noisier but algebraically identical ops should still
        # agree to well under a millibit.
        assert max_diff < 1e-2, (
            f"bf16 toy training loss drift across 30 steps: {max_diff:.2e}\n"
            f"base:  {losses_base}\nfused: {losses_fused}"
        )


class TestFusedBlockFallbackPaths(unittest.TestCase):
    """Non-fast-path configurations still return correct results.

    FusedChaosSSMBlock is supposed to be a drop-in replacement: any
    configuration it doesn't support (jacobian stats, non-diag mode)
    should dispatch to the unfused path and produce correct outputs.
    """

    def setUp(self) -> None:
        _force_eager_backends()

    def test_return_jacobian_stats_falls_back(self) -> None:
        from chaoscontrol.core_fused import FusedChaosSSMBlock
        from chaoscontrol.model import ChaosSSMBlock

        torch.manual_seed(4)
        block = ChaosSSMBlock(dim=16, ff_mult=2, a_mode="diag")
        fused = FusedChaosSSMBlock.from_unfused(block)

        x = torch.randn(2, 8, 16)
        y_base, stats_base = block(x, return_jacobian_stats=True)
        y_fused, stats_fused = fused(x, return_jacobian_stats=True)
        assert "lambda_max" in stats_fused
        assert "sv_log_var" in stats_fused
        max_diff = (y_base - y_fused).abs().max().item()
        assert max_diff < 1e-6, f"jacobian-stats fallback drift: {max_diff:.2e}"


class TestPostScanFusedFunction(unittest.TestCase):
    """Direct tests on the standalone _post_scan_fused_eager function.

    The function should match a hand-rolled reference that spells out
    every op. This catches subtle rewriting bugs in the fused function
    body that could drift from ChaosSSMBlock semantics.
    """

    def setUp(self) -> None:
        _force_eager_backends()

    def test_post_scan_function_matches_reference(self) -> None:
        from chaoscontrol.core_fused import _post_scan_fused_eager

        torch.manual_seed(5)
        B, T, D, M = 2, 12, 16, 2
        x = torch.randn(B, T, D)
        scan_out = torch.randn(B, T, D)
        w_norm = torch.randn(D)
        eps = 1e-6
        w_fc = torch.randn(D * M, D)
        w_proj = torch.randn(D, D * M)

        out_fused = _post_scan_fused_eager(x, scan_out, w_norm, eps, w_fc, w_proj)

        # Hand-rolled reference
        x_r = x + scan_out
        normed = F.rms_norm(x_r.float(), (D,), eps=eps)
        normed = normed.to(x_r.dtype) * w_norm
        hidden = F.linear(normed, w_fc)
        activated = F.silu(hidden)
        ff_out = F.linear(activated, w_proj)
        out_ref = x_r + ff_out

        max_diff = (out_fused - out_ref).abs().max().item()
        assert max_diff < 1e-6, f"post_scan_fused vs reference: {max_diff:.2e}"


if __name__ == "__main__":
    unittest.main()
