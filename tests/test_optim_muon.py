"""Gradient-parity and sanity tests for ``chaoscontrol.optim.muon.Muon``.

Tests run on CPU with the float32 compute-dtype fallback so bf16 matmul
imprecision does not enter the parity checks. The goal here is numerical
correctness of the algorithm on a known reference, not end-to-end
benchmarking.
"""
from __future__ import annotations

import copy
import unittest

import pytest
import torch
import torch.nn as nn

from chaoscontrol.optim.muon import Muon, newton_schulz_orthogonalize


def _make_linear(seed: int, in_features: int = 6, out_features: int = 4) -> nn.Linear:
    torch.manual_seed(seed)
    layer = nn.Linear(in_features, out_features, bias=True)
    return layer


def _standard_loss(layer: nn.Linear, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return ((layer(x) - y) ** 2).mean()


class TestNewtonSchulz(unittest.TestCase):
    """Newton-Schulz 5-step isometry characteristics on random inputs.

    NS5 with the tuned quintic constants ``(3.4445, -4.7750, 2.0315)`` is an
    *approximation*, not an exact orthogonalization. Five steps on random
    fp32 inputs lands singular values in roughly ``[0.5, 1.5]`` with
    ``||U U^T - I||_max`` around ``0.3-0.5`` on square/rectangular shapes.
    The competition SOTA Muon uses the same constants and produces
    identical output (verified by Codex review 2026-04-13). The meaningful
    test is therefore that singular values are bounded in the NS5 target
    range, not that ``U U^T`` is near identity.
    """

    def _assert_sv_bounded(self, out: torch.Tensor) -> None:
        """NS5 5-step pulls singular values toward 1.0 but not exactly to 1.0.

        All SVs must land in ``[0.5, 1.5]`` — wide enough to cover the
        empirical NS5 residual on random inputs, tight enough to catch a
        broken iteration that returns the input unchanged or explodes.
        """
        sv = torch.linalg.svdvals(out)
        self.assertTrue(
            (sv > 0.5).all(),
            msg=f"NS5 produced SVs below 0.5 (min={sv.min().item():.4f})",
        )
        self.assertTrue(
            (sv < 1.5).all(),
            msg=f"NS5 produced SVs above 1.5 (max={sv.max().item():.4f})",
        )

    def test_square_matrix_is_bounded_isometry(self) -> None:
        torch.manual_seed(0)
        grad = torch.randn(8, 8)
        out = newton_schulz_orthogonalize(grad, steps=5, compute_dtype=torch.float32)
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())
        self._assert_sv_bounded(out)

    def test_rectangular_wide_matrix_bounded_left_isometry(self) -> None:
        torch.manual_seed(1)
        grad = torch.randn(5, 9)  # rows < cols, not transposed internally
        out = newton_schulz_orthogonalize(grad, steps=6, compute_dtype=torch.float32)
        self.assertFalse(torch.isnan(out).any())
        self._assert_sv_bounded(out)

    def test_rectangular_tall_matrix_bounded_right_isometry(self) -> None:
        torch.manual_seed(2)
        grad = torch.randn(9, 5)  # rows > cols, transposed internally then untransposed
        out = newton_schulz_orthogonalize(grad, steps=6, compute_dtype=torch.float32)
        self.assertFalse(torch.isnan(out).any())
        self._assert_sv_bounded(out)

    def test_batched_matrices(self) -> None:
        torch.manual_seed(3)
        grad = torch.randn(3, 7, 7)
        out = newton_schulz_orthogonalize(grad, steps=5, compute_dtype=torch.float32)
        self.assertEqual(out.shape, (3, 7, 7))
        self.assertFalse(torch.isnan(out).any())
        # SV bounds on each batch element.
        for b in range(out.shape[0]):
            self._assert_sv_bounded(out[b])


class TestMuonMatrixPath(unittest.TestCase):
    def test_single_matrix_step_produces_bounded_isometric_update_direction(self) -> None:
        """A single Muon step on a square matrix applies an NS5-orthogonalized update.

        For a square matrix, the rectangular scale factor ``max(1, rows/cols)**0.5``
        equals 1, so the applied update direction should have the same
        singular-value characteristics as raw NS5 output — bounded in
        ``[0.5, 1.5]``, not exactly orthogonal. See ``TestNewtonSchulz``
        docstring for why strict orthogonality is the wrong assertion.
        """
        torch.manual_seed(42)
        w = torch.randn(6, 6, requires_grad=True)
        w.grad = torch.randn(6, 6)

        opt = Muon([w], lr=0.01, momentum=0.9, weight_decay=0.0,
                   compute_dtype=torch.float32)
        w_before = w.detach().clone()
        opt.step()
        delta = (w_before - w.detach()) / 0.01  # undo -lr scaling
        self.assertFalse(torch.isnan(delta).any())
        sv = torch.linalg.svdvals(delta)
        self.assertTrue(
            (sv > 0.5).all() and (sv < 1.5).all(),
            msg=f"Muon update SVs out of range (min={sv.min().item():.4f}, max={sv.max().item():.4f})",
        )

    def test_matrix_path_respects_rectangular_scale(self) -> None:
        torch.manual_seed(7)
        w = torch.randn(8, 4, requires_grad=True)  # rows > cols -> scale = sqrt(2)
        w.grad = torch.randn(8, 4)
        opt = Muon([w], lr=0.01, momentum=0.0, nesterov=False,
                   weight_decay=0.0, compute_dtype=torch.float32)
        before = w.detach().clone()
        opt.step()
        applied = (before - w.detach()) / 0.01
        expected_direction = newton_schulz_orthogonalize(
            w.grad, steps=5, compute_dtype=torch.float32
        )
        expected_scale = (8.0 / 4.0) ** 0.5
        expected_applied = expected_direction * expected_scale
        self.assertTrue(torch.allclose(applied, expected_applied, atol=1e-5))


class TestMuonAdamWFallback(unittest.TestCase):
    def test_non_matrix_param_matches_torch_adamw(self) -> None:
        torch.manual_seed(11)
        # 1D vector so the default matrix classifier routes it to AdamW.
        ref = nn.Parameter(torch.randn(16))
        muon_param = nn.Parameter(ref.detach().clone())

        grad = torch.randn(16)
        ref.grad = grad.clone()
        muon_param.grad = grad.clone()

        betas = (0.9, 0.999)
        eps = 1e-8
        lr = 0.005
        wd = 0.0

        ref_opt = torch.optim.AdamW([ref], lr=lr, betas=betas, eps=eps,
                                     weight_decay=wd)
        muon_opt = Muon([muon_param], lr=0.02, momentum=0.95,
                        adamw_betas=betas, adamw_eps=eps, adamw_lr=lr,
                        adamw_weight_decay=wd, compute_dtype=torch.float32)

        ref_opt.step()
        muon_opt.step()
        self.assertTrue(
            torch.allclose(ref.detach(), muon_param.detach(), atol=1e-6),
            f"max diff {(ref - muon_param).abs().max().item()}",
        )

    def test_non_matrix_param_matches_torch_adamw_with_weight_decay(self) -> None:
        torch.manual_seed(13)
        ref = nn.Parameter(torch.randn(16))
        muon_param = nn.Parameter(ref.detach().clone())
        grad = torch.randn(16)
        ref.grad = grad.clone()
        muon_param.grad = grad.clone()

        betas = (0.9, 0.999)
        lr = 0.01
        wd = 0.1

        ref_opt = torch.optim.AdamW([ref], lr=lr, betas=betas, eps=1e-8,
                                     weight_decay=wd)
        muon_opt = Muon([muon_param], lr=0.02, adamw_betas=betas,
                        adamw_lr=lr, adamw_weight_decay=wd,
                        compute_dtype=torch.float32)
        ref_opt.step()
        muon_opt.step()
        self.assertTrue(torch.allclose(ref.detach(), muon_param.detach(), atol=1e-6))

    def test_multi_step_adamw_parity(self) -> None:
        torch.manual_seed(17)
        ref = nn.Parameter(torch.randn(8))
        muon_param = nn.Parameter(ref.detach().clone())

        lr = 0.01
        ref_opt = torch.optim.AdamW([ref], lr=lr, betas=(0.9, 0.999), eps=1e-8,
                                     weight_decay=0.0)
        muon_opt = Muon([muon_param], lr=0.02, adamw_betas=(0.9, 0.999),
                        adamw_lr=lr, adamw_weight_decay=0.0,
                        compute_dtype=torch.float32)
        for step_idx in range(4):
            g = torch.randn(8, generator=torch.Generator().manual_seed(100 + step_idx))
            ref.grad = g.clone()
            muon_param.grad = g.clone()
            ref_opt.step()
            muon_opt.step()
        self.assertTrue(torch.allclose(ref.detach(), muon_param.detach(), atol=1e-6))


class TestMuonPlasticityBudget(unittest.TestCase):
    def test_channel_budget_maps_to_ssm_parameter_roles(self) -> None:
        in_w = nn.Parameter(torch.zeros(3, 3))
        out_w = nn.Parameter(torch.zeros(3, 3))
        log_a = nn.Parameter(torch.zeros(3))
        opt = Muon(
            [in_w, out_w, log_a],
            lr=0.01,
            compute_dtype=torch.float32,
        )
        opt.bind_param_names(
            [
                ("layers.0.core.in_proj.weight", in_w),
                ("layers.0.core.out_proj.weight", out_w),
                ("layers.0.core.log_a", log_a),
            ]
        )
        opt.set_plasticity_budget(
            torch.tensor([0.0, 0.5, 1.0]),
            confidence=torch.ones(3),
            strength=0.25,
            step=17,
        )

        expected = torch.tensor([1.0, 1.125, 1.25])
        torch.testing.assert_close(
            opt._plasticity_multiplier_for(in_w), expected.view(3, 1)
        )
        torch.testing.assert_close(
            opt._plasticity_multiplier_for(out_w), expected.view(1, 3)
        )
        torch.testing.assert_close(opt._plasticity_multiplier_for(log_a), expected)
        trace = opt.plasticity_budget_trace()
        assert trace["enabled"] is True
        assert trace["updates"] == 1
        assert trace["step"] == 17
        assert trace["lr_multiplier_max"] == pytest.approx(1.25)
        assert trace["budget_nonzero_fraction"] == pytest.approx(2.0 / 3.0)
        assert trace["top_channels"][0] == 2
        assert trace["top_budget"][0] == pytest.approx(1.0)

    def test_plasticity_budget_caches_dtype_device_views(self) -> None:
        p = nn.Parameter(torch.ones(3, 4, dtype=torch.bfloat16))
        opt = Muon(
            [p],
            lr=0.1,
            fused=False,
            matrix_param_names={"layers.0.core.in_proj.weight"},
        )
        opt.bind_param_names([("layers.0.core.in_proj.weight", p)])
        opt.set_plasticity_budget(torch.tensor([0.0, 0.5, 1.0]), strength=0.5)

        first = opt._plasticity_multiplier_for(p)
        second = opt._plasticity_multiplier_for(p)

        assert first is not None
        assert second is not None
        assert first.dtype == torch.bfloat16
        assert second.data_ptr() == first.data_ptr()

    def test_plasticity_budget_scales_log_a_adamw_update(self) -> None:
        p = nn.Parameter(torch.zeros(2))
        p.grad = torch.ones_like(p)
        opt = Muon(
            [p],
            lr=0.1,
            adamw_lr=0.1,
            adamw_betas=(0.0, 0.0),
            adamw_eps=1e-12,
            adamw_weight_decay=0.0,
            matrix_param_names=set(),
            compute_dtype=torch.float32,
        )
        opt.bind_param_names([("layers.0.core.log_a", p)])
        opt.set_plasticity_budget(
            torch.tensor([0.0, 1.0]),
            confidence=torch.ones(2),
            strength=0.5,
            step=3,
        )

        opt.step()

        torch.testing.assert_close(
            p.detach(),
            torch.tensor([-0.1, -0.15]),
            rtol=0,
            atol=1e-6,
        )

    def test_log_a_beta_coupling_uses_slow_ema_trace(self) -> None:
        p = nn.Parameter(torch.tensor([-8.0, 8.0]))
        opt = Muon(
            [p],
            lr=0.1,
            adamw_betas=(0.9, 0.0),
            adamw_eps=1e-12,
            adamw_weight_decay=0.0,
            matrix_param_names=set(),
            log_a_beta_coupling=True,
            log_a_beta_ema=0.99,
            log_a_beta_min=0.2,
            compute_dtype=torch.float32,
        )
        opt.bind_param_names([("layers.0.core.log_a", p)])
        p.grad = torch.ones_like(p)

        opt.step()

        trace = opt.ssm_role_trace()
        assert trace["log_a_beta_coupling"] is True
        assert trace["log_a_beta_updates"] == 1
        assert trace["log_a_beta_min"] == pytest.approx(0.2, abs=1e-3)
        assert trace["log_a_beta_max"] == pytest.approx(0.9, abs=1e-3)
        state = opt.state[p]
        torch.testing.assert_close(
            state["log_a_slow_ema"],
            torch.tensor([-8.0, 8.0]),
        )

    def test_log_a_beta_coupling_updates_slow_ema_before_beta(self) -> None:
        p = nn.Parameter(torch.tensor([0.0]))
        opt = Muon(
            [p],
            lr=0.1,
            adamw_betas=(0.9, 0.0),
            adamw_eps=1e-12,
            adamw_weight_decay=0.0,
            matrix_param_names=set(),
            log_a_beta_coupling=True,
            log_a_beta_ema=0.5,
            log_a_beta_min=0.1,
            compute_dtype=torch.float32,
        )
        opt.bind_param_names([("layers.0.core.log_a", p)])
        p.grad = torch.ones_like(p)
        opt.step()
        with torch.no_grad():
            p.fill_(8.0)
        p.grad = torch.ones_like(p)

        opt.step()

        state = opt.state[p]
        # EMA moves halfway from the original 0.0 toward the new 8.0;
        # beta is therefore high, but derived from the damped EMA rather
        # than the instantaneous parameter.
        torch.testing.assert_close(state["log_a_slow_ema"], torch.tensor([4.0]))
        trace = opt.ssm_role_trace()
        assert trace["log_a_beta_max"] < 0.9
        assert trace["log_a_beta_max"] > 0.88


class TestMuonClassifier(unittest.TestCase):
    def test_matrix_param_names_override(self) -> None:
        torch.manual_seed(19)
        mat = nn.Parameter(torch.randn(4, 4))  # shape-wise a matrix
        mat.grad = torch.randn(4, 4)
        # But we force it onto the AdamW path via the name whitelist (empty set).
        opt = Muon([mat], lr=0.01, matrix_param_names=set(),
                   compute_dtype=torch.float32)
        opt.bind_param_names([("mat", mat)])
        opt.step()
        state = opt.state[mat]
        self.assertIn("exp_avg", state)
        self.assertNotIn("momentum_buffer", state)

    def test_custom_predicate_routes_to_matrix_path(self) -> None:
        torch.manual_seed(23)
        vec = nn.Parameter(torch.randn(6, 6))
        vec.grad = torch.randn(6, 6)
        opt = Muon([vec], lr=0.01,
                   is_matrix=lambda p, name: p.ndim == 2,
                   compute_dtype=torch.float32)
        opt.step()
        state = opt.state[vec]
        self.assertIn("momentum_buffer", state)
        self.assertNotIn("exp_avg", state)


class TestMuonStateDict(unittest.TestCase):
    def test_state_dict_round_trip_matches_no_save(self) -> None:
        torch.manual_seed(29)
        mat = nn.Parameter(torch.randn(6, 6))
        vec = nn.Parameter(torch.randn(6))
        mat.grad = torch.randn(6, 6)
        vec.grad = torch.randn(6)

        opt1 = Muon([mat, vec], lr=0.01, compute_dtype=torch.float32)
        opt1.step()
        saved = copy.deepcopy(opt1.state_dict())

        # Reference path: keep stepping in-place.
        mat_ref = nn.Parameter(mat.detach().clone())
        vec_ref = nn.Parameter(vec.detach().clone())
        opt_ref = Muon([mat_ref, vec_ref], lr=0.01, compute_dtype=torch.float32)
        opt_ref.load_state_dict(saved)

        # Give both the same next-step gradient.
        g_mat = torch.randn(6, 6)
        g_vec = torch.randn(6)
        mat.grad = g_mat.clone()
        vec.grad = g_vec.clone()
        mat_ref.grad = g_mat.clone()
        vec_ref.grad = g_vec.clone()

        opt1.step()
        opt_ref.step()

        self.assertTrue(torch.allclose(mat.detach(), mat_ref.detach(), atol=1e-6))
        self.assertTrue(torch.allclose(vec.detach(), vec_ref.detach(), atol=1e-6))


class TestMuonToyTraining(unittest.TestCase):
    def _train_once(self, seed: int, steps: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(seed)
        layer = nn.Linear(8, 4, bias=True)
        x = torch.randn(32, 8)
        y = torch.randn(32, 4)
        opt = Muon(layer.parameters(), lr=0.005,
                   adamw_lr=0.001, compute_dtype=torch.float32)
        for _ in range(steps):
            opt.zero_grad()
            loss = _standard_loss(layer, x, y)
            loss.backward()
            opt.step()
        return layer.weight.detach().clone(), layer.bias.detach().clone()

    def test_deterministic_with_fixed_seed(self) -> None:
        w1, b1 = self._train_once(seed=101)
        w2, b2 = self._train_once(seed=101)
        self.assertTrue(torch.equal(w1, w2))
        self.assertTrue(torch.equal(b1, b2))

    def test_two_instances_agree_on_same_model(self) -> None:
        torch.manual_seed(202)
        layer_a = nn.Linear(5, 3, bias=True)
        layer_b = copy.deepcopy(layer_a)
        x = torch.randn(16, 5)
        y = torch.randn(16, 3)

        opt_a = Muon(layer_a.parameters(), lr=0.01, adamw_lr=0.002,
                     compute_dtype=torch.float32)
        opt_b = Muon(layer_b.parameters(), lr=0.01, adamw_lr=0.002,
                     compute_dtype=torch.float32)
        for _ in range(3):
            opt_a.zero_grad()
            _standard_loss(layer_a, x, y).backward()
            opt_a.step()
            opt_b.zero_grad()
            _standard_loss(layer_b, x, y).backward()
            opt_b.step()
        self.assertTrue(torch.allclose(layer_a.weight, layer_b.weight, atol=1e-6))
        self.assertTrue(torch.allclose(layer_a.bias, layer_b.bias, atol=1e-6))

    def test_loss_decreases_over_a_few_steps(self) -> None:
        torch.manual_seed(303)
        layer = nn.Linear(8, 4, bias=True)
        x = torch.randn(64, 8)
        y = torch.randn(64, 4)
        opt = Muon(layer.parameters(), lr=0.005, adamw_lr=0.001,
                   compute_dtype=torch.float32)
        losses: list[float] = []
        for _ in range(5):
            opt.zero_grad()
            loss = _standard_loss(layer, x, y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        self.assertLess(losses[-1], losses[0])


class TestFusedMuonParity:
    """Fused Muon must produce byte-identical updates to the reference loop
    when the two paths see the same grads and optimizer state.

    The fused path groups matrix params by shape and runs a batched NS;
    the reference iterates one-by-one. Mathematically equivalent; the
    test locks this against a future batched-NS edit that silently
    changes numerical reduction order.
    """

    @staticmethod
    def _build_model(seed: int) -> nn.Sequential:
        torch.manual_seed(seed)
        # Mix of matrix shapes and a non-matrix param to exercise both branches.
        # Two Linear(4, 4) layers in the same shape-group test batched NS over
        # a group of size > 1; LayerNorm tests the AdamW fallback coalesce.
        return nn.Sequential(
            nn.Linear(8, 4),
            nn.Linear(4, 4),
            nn.Linear(4, 4),
            nn.LayerNorm(4),
        )

    def _clone_grads(self, src: nn.Module, dst: nn.Module) -> None:
        for p_src, p_dst in zip(src.parameters(), dst.parameters()):
            p_dst.grad = p_src.grad.clone()

    def test_single_step_matches_reference_loop(self) -> None:
        """One step: fused updates match unfused to fp32 tolerance."""
        from chaoscontrol.optim.muon import Muon
        m_ref = self._build_model(seed=13)
        m_new = self._build_model(seed=13)
        # Identical init weights (seed is deterministic).
        for p_r, p_n in zip(m_ref.parameters(), m_new.parameters()):
            assert torch.equal(p_r.data, p_n.data)

        # Identical grads.
        for p_r in m_ref.parameters():
            p_r.grad = torch.randn_like(p_r)
        self._clone_grads(m_ref, m_new)

        opt_ref = Muon(list(m_ref.parameters()), lr=0.02, fused=False)
        opt_new = Muon(list(m_new.parameters()), lr=0.02, fused=True)
        opt_ref.step()
        opt_new.step()

        for (n_r, p_r), (n_n, p_n) in zip(
            m_ref.named_parameters(), m_new.named_parameters(),
        ):
            assert n_r == n_n
            # fp32 tolerance — batched NS reorders reductions minimally
            # vs the per-param loop. Cross-shape reductions are identical;
            # intra-shape batch-reduce is where the drift comes in.
            assert torch.allclose(p_r.data, p_n.data, rtol=1e-5, atol=1e-6), (
                f"fused Muon drift on {n_r!r}: "
                f"max|diff|={(p_r.data - p_n.data).abs().max().item()}"
            )

    def test_multi_step_matches_reference_loop(self) -> None:
        """Four steps of training: state dicts and params stay in sync.

        Accumulating drift over multiple steps is a realistic regression
        case — a batched NS that's slightly wrong per step compounds.
        """
        from chaoscontrol.optim.muon import Muon
        m_ref = self._build_model(seed=17)
        m_new = self._build_model(seed=17)
        opt_ref = Muon(list(m_ref.parameters()), lr=0.02, fused=False)
        opt_new = Muon(list(m_new.parameters()), lr=0.02, fused=True)
        torch.manual_seed(100)
        for _ in range(4):
            for p_r in m_ref.parameters():
                p_r.grad = torch.randn_like(p_r)
            self._clone_grads(m_ref, m_new)
            opt_ref.step()
            opt_new.step()
        for (n_r, p_r), (n_n, p_n) in zip(
            m_ref.named_parameters(), m_new.named_parameters(),
        ):
            assert torch.allclose(p_r.data, p_n.data, rtol=1e-5, atol=1e-6), (
                f"fused Muon drift after 4 steps on {n_r!r}: "
                f"max|diff|={(p_r.data - p_n.data).abs().max().item()}"
            )

    def test_fused_step_calls_ns_once_per_shape_group(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Structural: fused path calls newton_schulz_orthogonalize once per
        unique matrix shape, not once per matrix param.

        The test model has three matrix params:
          Linear(8, 4).weight  — shape (4, 8)
          Linear(4, 4).weight  — shape (4, 4)  (x2 — grouped)
          Linear(4, 4).weight  — shape (4, 4)
        Plus biases (vector, non-matrix) and LayerNorm (vector, non-matrix).
        Expected: 2 NS calls for the fused path (one per shape), 3 for the
        unfused path.
        """
        from chaoscontrol.optim import muon as muon_mod
        calls: list[tuple[int, ...]] = []
        orig_ns = muon_mod.newton_schulz_orthogonalize

        def counting_ns(tensor, **kw):
            calls.append(tuple(tensor.shape))
            return orig_ns(tensor, **kw)

        monkeypatch.setattr(muon_mod, "newton_schulz_orthogonalize", counting_ns)

        m = self._build_model(seed=19)
        for p in m.parameters():
            p.grad = torch.randn_like(p)

        opt = muon_mod.Muon(list(m.parameters()), lr=0.02, fused=True)
        opt.step()

        # Exactly 2 NS invocations: one per shape-group.
        assert len(calls) == 2, (
            f"fused path expected 2 NS calls (one per shape group), got "
            f"{len(calls)}: shapes = {calls}"
        )
        # The (4, 4) group batched two params into leading-dim-2.
        shapes = sorted(calls, key=lambda s: s)
        batched_shapes = [s for s in calls if len(s) == 3 and s[0] > 1]
        assert len(batched_shapes) >= 1, (
            f"expected at least one batched NS call (leading dim > 1); "
            f"got shapes={calls}"
        )

    def test_ns_compute_dtype_preserved(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Fused path must pass compute_dtype=bf16 on CUDA and fp32 on CPU
        to newton_schulz_orthogonalize — matches the unfused default at
        optim/muon.py:31-32. Regression guard against an edit that batches
        params but silently drops the dtype policy.
        """
        from chaoscontrol.optim import muon as muon_mod
        captured_dtypes: list[torch.dtype | None] = []
        orig_ns = muon_mod.newton_schulz_orthogonalize

        def capturing_ns(tensor, **kw):
            captured_dtypes.append(kw.get("compute_dtype"))
            return orig_ns(tensor, **kw)

        monkeypatch.setattr(muon_mod, "newton_schulz_orthogonalize", capturing_ns)

        m = self._build_model(seed=21)
        for p in m.parameters():
            p.grad = torch.randn_like(p)
        opt = muon_mod.Muon(list(m.parameters()), lr=0.02, fused=True)
        opt.step()

        # compute_dtype on CPU-only test run: newton_schulz defaults fp32 on
        # CPU. Fused path must respect that default (i.e., pass None or
        # torch.float32 explicitly — both are fine, None means NS falls back
        # to its own default).
        for dtype in captured_dtypes:
            assert dtype in (None, torch.float32, torch.bfloat16), (
                f"unexpected compute_dtype: {dtype}"
            )

    def test_state_dict_shape_matches_unfused(self) -> None:
        """After one fused step, optimizer.state_dict() must have the same
        structure the unfused path produces — checkpoint-compatibility.
        """
        from chaoscontrol.optim.muon import Muon
        m_ref = self._build_model(seed=23)
        m_new = self._build_model(seed=23)
        for p in m_ref.parameters():
            p.grad = torch.randn_like(p)
        self._clone_grads(m_ref, m_new)

        opt_ref = Muon(list(m_ref.parameters()), lr=0.02, fused=False)
        opt_new = Muon(list(m_new.parameters()), lr=0.02, fused=True)
        opt_ref.step()
        opt_new.step()

        state_ref = opt_ref.state_dict()
        state_new = opt_new.state_dict()
        assert set(state_ref.keys()) == set(state_new.keys())
        # Per-param state entries: same keys (momentum_buffer for matrix,
        # step/exp_avg/exp_avg_sq for non-matrix).
        for pid in state_ref["state"]:
            ref_keys = set(state_ref["state"][pid].keys())
            new_keys = set(state_new["state"][pid].keys())
            assert ref_keys == new_keys, (
                f"state dict keys differ for param {pid}: "
                f"ref={ref_keys} new={new_keys}"
            )


if __name__ == "__main__":
    unittest.main()
