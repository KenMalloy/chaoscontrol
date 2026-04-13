"""Gradient-parity and sanity tests for ``chaoscontrol.optim.muon.Muon``.

Tests run on CPU with the float32 compute-dtype fallback so bf16 matmul
imprecision does not enter the parity checks. The goal here is numerical
correctness of the algorithm on a known reference, not end-to-end
benchmarking.
"""
from __future__ import annotations

import copy
import unittest

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
    def test_square_matrix_is_nearly_orthogonal(self) -> None:
        torch.manual_seed(0)
        grad = torch.randn(8, 8)
        out = newton_schulz_orthogonalize(grad, steps=5, compute_dtype=torch.float32)
        eye = torch.eye(8)
        # NS5 in float32 lands very close to an isometry.
        max_dev = (out @ out.mT - eye).abs().max().item()
        self.assertLess(max_dev, 1e-3)

    def test_rectangular_wide_matrix_left_isometry(self) -> None:
        torch.manual_seed(1)
        grad = torch.randn(5, 9)  # rows < cols, not transposed internally
        out = newton_schulz_orthogonalize(grad, steps=6, compute_dtype=torch.float32)
        eye = torch.eye(5)
        max_dev = (out @ out.mT - eye).abs().max().item()
        self.assertLess(max_dev, 1e-3)

    def test_rectangular_tall_matrix_right_isometry(self) -> None:
        torch.manual_seed(2)
        grad = torch.randn(9, 5)  # rows > cols, transposed internally then untransposed
        out = newton_schulz_orthogonalize(grad, steps=6, compute_dtype=torch.float32)
        eye = torch.eye(5)
        # For tall matrices, U.T @ U ≈ I.
        max_dev = (out.mT @ out - eye).abs().max().item()
        self.assertLess(max_dev, 1e-3)

    def test_batched_matrices(self) -> None:
        torch.manual_seed(3)
        grad = torch.randn(3, 7, 7)
        out = newton_schulz_orthogonalize(grad, steps=5, compute_dtype=torch.float32)
        self.assertEqual(out.shape, (3, 7, 7))
        eye = torch.eye(7).expand_as(out)
        max_dev = (out @ out.mT - eye).abs().max().item()
        self.assertLess(max_dev, 1e-3)


class TestMuonMatrixPath(unittest.TestCase):
    def test_single_matrix_step_produces_near_orthogonal_update_direction(self) -> None:
        torch.manual_seed(42)
        w = torch.randn(6, 6, requires_grad=True)
        w.grad = torch.randn(6, 6)

        opt = Muon([w], lr=0.01, momentum=0.9, weight_decay=0.0,
                   compute_dtype=torch.float32)
        w_before = w.detach().clone()
        opt.step()
        delta = (w_before - w.detach()) / 0.01  # undo -lr scaling
        # The applied update should itself be a near-orthogonal factor (scale=1 for square).
        eye = torch.eye(6)
        max_dev = (delta @ delta.mT - eye).abs().max().item()
        self.assertLess(max_dev, 5e-3)

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


if __name__ == "__main__":
    unittest.main()
