"""Tests for the LAMB optimizer (chaoscontrol.optim.lamb).

These tests run on CPU only and are designed to be cheap (< 1s total).
They cover the trust-ratio math, Adam-parity when the trust ratio is
disabled, state_dict round-tripping, deterministic single-step updates,
reproducibility across seed-matched instances, and LAMB's headline
property: stability at very large effective learning rates where plain
Adam diverges.
"""
from __future__ import annotations

import copy
import math
import unittest

import torch

from chaoscontrol.optim.lamb import LAMB


def _toy_linear(seed: int = 0) -> torch.nn.Linear:
    torch.manual_seed(seed)
    # 2-D weight so the trust ratio is actually applied (1-D tensors are
    # skipped by default, matching the NVIDIA/timm convention).
    return torch.nn.Linear(4, 3, bias=False)


class TestTrustRatioMath(unittest.TestCase):
    """Hand-computed trust-ratio cases for LAMB._trust_ratio."""

    def test_basic_ratio(self) -> None:
        # ||w|| = 5, ||r|| = 1 => phi = clip(5/1, 0, 10) = 5.0
        w_norm = torch.tensor(5.0)
        r_norm = torch.tensor(1.0)
        phi = LAMB._trust_ratio(w_norm, r_norm, trust_clip=10.0)
        assert math.isclose(phi.item(), 5.0, abs_tol=1e-7), phi

    def test_ratio_clipped(self) -> None:
        # ||w|| = 100, ||r|| = 1, clip = 10 => phi = 10.0
        w_norm = torch.tensor(100.0)
        r_norm = torch.tensor(1.0)
        phi = LAMB._trust_ratio(w_norm, r_norm, trust_clip=10.0)
        assert math.isclose(phi.item(), 10.0, abs_tol=1e-7), phi

    def test_zero_weight_norm_falls_back_to_one(self) -> None:
        # w=0 => phi = 1 (not 0), so the step still moves the param.
        phi = LAMB._trust_ratio(torch.tensor(0.0), torch.tensor(3.0), trust_clip=10.0)
        assert math.isclose(phi.item(), 1.0, abs_tol=1e-7), phi

    def test_zero_update_norm_falls_back_to_one(self) -> None:
        # r=0 => phi = 1 (not inf), so we never divide by zero.
        phi = LAMB._trust_ratio(torch.tensor(3.0), torch.tensor(0.0), trust_clip=10.0)
        assert math.isclose(phi.item(), 1.0, abs_tol=1e-7), phi

    def test_fractional_ratio(self) -> None:
        # ||w|| = 3, ||r|| = 4 => phi = 0.75 (sub-unit ratios are legal,
        # they mean "shrink the update because the weight is small").
        phi = LAMB._trust_ratio(torch.tensor(3.0), torch.tensor(4.0), trust_clip=10.0)
        assert math.isclose(phi.item(), 0.75, abs_tol=1e-7), phi


class TestAdamParity(unittest.TestCase):
    """With trust_ratio forced to 1.0 and weight_decay=0, LAMB == Adam."""

    def test_lamb_matches_adam_when_trust_ratio_forced_one(self) -> None:
        torch.manual_seed(0)
        model_lamb = _toy_linear(seed=0)
        model_adam = _toy_linear(seed=0)
        # Parity check on the fresh models before we touch grads.
        for p_l, p_a in zip(model_lamb.parameters(), model_adam.parameters()):
            assert torch.equal(p_l.data, p_a.data)

        # LAMB with trust ratio pinned to 1.0 and WD = 0 should match
        # torch.optim.Adam step-for-step (both in state and parameters).
        # NB: we pass eps=1e-8 to BOTH optimizers because LAMB's default
        # (1e-6) differs from torch.optim.Adam's default (1e-8). Leaving
        # either side on defaults would silently break parity on eps.
        lamb = LAMB(
            model_lamb.parameters(),
            lr=1e-2,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            trust_ratio_override=1.0,
            always_adapt=True,  # irrelevant because override is set, but explicit
        )
        adam = torch.optim.Adam(
            model_adam.parameters(),
            lr=1e-2,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
        )

        torch.manual_seed(123)
        for _ in range(5):
            x = torch.randn(8, 4)
            y = torch.randn(8, 3)
            # Same input to both — same loss, same grads.
            loss_l = ((model_lamb(x) - y) ** 2).mean()
            loss_a = ((model_adam(x) - y) ** 2).mean()
            lamb.zero_grad()
            adam.zero_grad()
            loss_l.backward()
            loss_a.backward()
            lamb.step()
            adam.step()

        for p_l, p_a in zip(model_lamb.parameters(), model_adam.parameters()):
            max_diff = (p_l.data - p_a.data).abs().max().item()
            assert max_diff < 1e-6, f"LAMB(trust=1)/Adam diverge: max diff {max_diff:.2e}"


class TestStateDictRoundTrip(unittest.TestCase):
    def test_save_load_preserves_state(self) -> None:
        torch.manual_seed(0)
        model = _toy_linear(seed=0)
        opt = LAMB(model.parameters(), lr=1e-2)

        # Build up optimizer state with a couple of steps.
        for _ in range(3):
            x = torch.randn(4, 4)
            y = torch.randn(4, 3)
            opt.zero_grad()
            ((model(x) - y) ** 2).mean().backward()
            opt.step()

        checkpoint = copy.deepcopy(opt.state_dict())

        # Fresh model+optimizer, load state, compare internals.
        model2 = _toy_linear(seed=0)
        opt2 = LAMB(model2.parameters(), lr=1e-2)
        opt2.load_state_dict(checkpoint)

        for group, group2 in zip(opt.param_groups, opt2.param_groups):
            assert group["lr"] == group2["lr"]
            assert group["betas"] == group2["betas"]
        for p1, p2 in zip(opt.param_groups[0]["params"], opt2.param_groups[0]["params"]):
            s1 = opt.state[p1]
            s2 = opt2.state[p2]
            assert s1["step"] == s2["step"]
            assert torch.equal(s1["exp_avg"], s2["exp_avg"])
            assert torch.equal(s1["exp_avg_sq"], s2["exp_avg_sq"])


class TestDeterministicStep(unittest.TestCase):
    """One LAMB step on a toy model with a fixed seed must be deterministic."""

    def test_single_step_is_deterministic(self) -> None:
        def run() -> torch.Tensor:
            torch.manual_seed(42)
            model = _toy_linear(seed=42)
            opt = LAMB(model.parameters(), lr=5e-3, weight_decay=0.01)
            torch.manual_seed(99)
            x = torch.randn(16, 4)
            y = torch.randn(16, 3)
            opt.zero_grad()
            ((model(x) - y) ** 2).mean().backward()
            opt.step()
            return next(model.parameters()).detach().clone()

        w1 = run()
        w2 = run()
        assert torch.equal(w1, w2), "LAMB step is not deterministic across reruns"

    def test_two_instances_same_seed_agree(self) -> None:
        """Two independent LAMB instances seeded identically must stay identical."""
        torch.manual_seed(7)
        model_a = _toy_linear(seed=7)
        model_b = _toy_linear(seed=7)
        opt_a = LAMB(model_a.parameters(), lr=1e-3, weight_decay=1e-4)
        opt_b = LAMB(model_b.parameters(), lr=1e-3, weight_decay=1e-4)

        torch.manual_seed(777)
        for _ in range(4):
            x = torch.randn(8, 4)
            y = torch.randn(8, 3)
            opt_a.zero_grad()
            opt_b.zero_grad()
            ((model_a(x) - y) ** 2).mean().backward()
            ((model_b(x) - y) ** 2).mean().backward()
            opt_a.step()
            opt_b.step()

        for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
            assert torch.equal(p_a.data, p_b.data)


class TestLargeBatchStability(unittest.TestCase):
    """LAMB's raison d'etre: per-tensor step norm is bounded and grad-scale-invariant.

    The mathematical guarantee LAMB provides, for a 2-D parameter W on a
    single step, is:

        ||W_new - W|| = lr * phi * ||r|| = lr * clip(||W||/||r||, 0, C) * ||r||

    When ||W||/||r|| <= C (the typical regime once ||r|| is large),
    this simplifies to

        ||W_new - W|| = lr * ||W||                          (1)

    i.e. the step norm is *exactly proportional* to the weight norm and
    completely independent of how big the raw Adam update ||r|| gets.
    That is the property that keeps LAMB stable at very large effective
    batch sizes: scaling up the batch changes ||r|| but leaves the LAMB
    step invariant. Plain Adam's step has no such invariance.

    We verify (1) directly and also check that doubling the injected
    gradient does not change LAMB's step.
    """

    def test_lamb_step_norm_matches_closed_form(self) -> None:
        torch.manual_seed(0)
        model = torch.nn.Linear(8, 4, bias=False)
        w0 = model.weight.data.clone()
        w0_norm = w0.norm(p=2).item()

        lr = 1e-2
        trust_clip = 1e9  # effectively unclamped
        opt = LAMB(
            model.parameters(),
            lr=lr,
            eps=1e-12,
            weight_decay=0.0,
            trust_clip=trust_clip,
            always_adapt=True,
        )
        # Huge gradient so ||r|| >> ||w|| and we are NOT in the clamp.
        model.weight.grad = torch.full_like(model.weight, 1e6)
        opt.step()

        step_norm = (model.weight.data - w0).norm(p=2).item()
        expected = lr * w0_norm
        rel = abs(step_norm - expected) / expected
        assert rel < 1e-4, (
            f"LAMB step norm {step_norm:.6e} did not match closed-form "
            f"lr * ||w|| = {expected:.6e} (rel diff {rel:.2e})"
        )

    def test_lamb_scale_invariance_under_gradient_burst(self) -> None:
        """Doubling the gradient must leave LAMB's parameter update unchanged.

        This is the core mathematical reason LAMB is stable at very large
        effective batch sizes: the trust ratio `||w|| / ||r||` absorbs any
        multiplicative change in the raw Adam direction, so the actual
        parameter update depends only on ||w||. Plain Adam does not have
        this property (its update grows with ||g|| until the second
        moment catches up).
        """
        torch.manual_seed(0)
        model_a = torch.nn.Linear(8, 4, bias=False)
        model_b = torch.nn.Linear(8, 4, bias=False)
        model_b.weight.data.copy_(model_a.weight.data)

        opt_a = LAMB(
            model_a.parameters(),
            lr=1e-2,
            eps=1e-12,
            weight_decay=0.0,
            trust_clip=1e9,
            always_adapt=True,
        )
        opt_b = LAMB(
            model_b.parameters(),
            lr=1e-2,
            eps=1e-12,
            weight_decay=0.0,
            trust_clip=1e9,
            always_adapt=True,
        )

        g = torch.randn_like(model_a.weight) * 1e3
        model_a.weight.grad = g.clone()
        model_b.weight.grad = g.clone() * 1e6  # one million times larger

        opt_a.step()
        opt_b.step()

        # Scale invariance: both updates should produce (nearly) the
        # same weights. The only disagreement comes from eps in the
        # denominator, which is negligible here.
        max_diff = (model_a.weight.data - model_b.weight.data).abs().max().item()
        assert max_diff < 1e-4, (
            f"LAMB updates diverged under a 1e6x gradient scaling "
            f"(max diff {max_diff:.2e}); scale invariance broken."
        )

    def test_lamb_survives_gradient_burst_without_nan(self) -> None:
        """A mid-training gradient spike must not leave parameters non-finite."""
        torch.manual_seed(0)
        model = torch.nn.Linear(16, 8, bias=False)
        opt = LAMB(
            model.parameters(),
            lr=1e-2,
            eps=1e-8,
            weight_decay=1e-2,
            trust_clip=10.0,
            always_adapt=True,
        )
        # Normal gradient step.
        model.weight.grad = torch.randn_like(model.weight) * 0.1
        opt.step()
        # Gradient burst — 6 orders of magnitude larger.
        model.weight.grad = torch.randn_like(model.weight) * 1e5
        opt.step()
        # Back to normal.
        model.weight.grad = torch.randn_like(model.weight) * 0.1
        opt.step()

        assert torch.isfinite(model.weight).all(), (
            "LAMB produced non-finite parameters after gradient burst"
        )


class TestBasicInterface(unittest.TestCase):
    """Sanity checks on the torch.optim.Optimizer contract."""

    def test_zero_grad_clears_grads(self) -> None:
        model = _toy_linear(seed=0)
        opt = LAMB(model.parameters(), lr=1e-3)
        x = torch.randn(4, 4)
        y = torch.randn(4, 3)
        ((model(x) - y) ** 2).mean().backward()
        assert next(model.parameters()).grad is not None
        opt.zero_grad(set_to_none=True)
        assert next(model.parameters()).grad is None

    def test_rejects_bad_hyperparameters(self) -> None:
        model = _toy_linear(seed=0)
        with self.assertRaises(ValueError):
            LAMB(model.parameters(), lr=-1.0)
        with self.assertRaises(ValueError):
            LAMB(model.parameters(), betas=(1.5, 0.999))
        with self.assertRaises(ValueError):
            LAMB(model.parameters(), eps=-1e-9)
        with self.assertRaises(ValueError):
            LAMB(model.parameters(), weight_decay=-0.1)
        with self.assertRaises(ValueError):
            LAMB(model.parameters(), trust_clip=0.0)


if __name__ == "__main__":
    unittest.main()
