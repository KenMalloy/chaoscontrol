"""Tests for weight-tied depth recurrence in ChaosStudentLM.

Exp 19 Phase 1 architectural lever. Depth recurrence runs a contiguous shared
group of physical layers N times in the forward pass without duplicating
parameters. At count=1 the behavior must be bit-identical to the non-recurrent
path; at count>=2 outputs must differ (otherwise the feature is a no-op).
"""
from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from chaoscontrol.model import ChaosStudentLM


def _make_model(**overrides) -> ChaosStudentLM:
    kwargs = dict(
        vocab_size=64,
        dim=16,
        num_layers=4,
        ff_mult=2,
        a_mode="diag",
        rich_b_mode="none",
        outer_model_dim=0,
    )
    kwargs.update(overrides)
    return ChaosStudentLM(**kwargs)


class TestDepthRecurrenceParity(unittest.TestCase):
    """Load-bearing test: count=1 must be bit-identical to non-recurrent."""

    def test_count_one_bit_identical_to_baseline(self) -> None:
        torch.manual_seed(0)
        baseline = _make_model()

        torch.manual_seed(0)
        recurrent = _make_model(
            depth_recurrence_shared_layers=[1, 2],
            depth_recurrence_count=1,
        )

        # Same seed -> same params, but copy state_dict explicitly to make
        # the invariant load-bearing on parameter equality, not init order.
        recurrent.load_state_dict(baseline.state_dict())

        baseline.eval()
        recurrent.eval()

        ids = torch.randint(0, 64, (2, 16))
        with torch.no_grad():
            out_base = baseline(ids)["logits"]
            out_recur = recurrent(ids)["logits"]

        self.assertTrue(
            torch.equal(out_base, out_recur),
            "count=1 must produce bit-identical outputs to the non-recurrent baseline",
        )

    def test_virtual_sequence_matches_range_at_count_one(self) -> None:
        model = _make_model(
            depth_recurrence_shared_layers=[1, 2],
            depth_recurrence_count=1,
        )
        self.assertEqual(model._virtual_layer_indices, [0, 1, 2, 3])

    def test_virtual_sequence_empty_shared_is_identity(self) -> None:
        model = _make_model()
        self.assertEqual(model._virtual_layer_indices, [0, 1, 2, 3])
        self.assertEqual(model.depth_recurrence_shared_layers, [])
        self.assertEqual(model.depth_recurrence_count, 1)


class TestDepthRecurrenceForward(unittest.TestCase):
    def test_forward_shape_count_two(self) -> None:
        model = _make_model(
            depth_recurrence_shared_layers=[1, 2],
            depth_recurrence_count=2,
        )
        ids = torch.randint(0, 64, (2, 16))
        out = model(ids)
        self.assertEqual(out["logits"].shape, (2, 16, 64))
        self.assertEqual(out["hidden"].shape, (2, 16, 16))

    def test_forward_shape_count_three(self) -> None:
        model = _make_model(
            depth_recurrence_shared_layers=[1, 2],
            depth_recurrence_count=3,
        )
        ids = torch.randint(0, 64, (2, 16))
        out = model(ids)
        self.assertEqual(out["logits"].shape, (2, 16, 64))

    def test_virtual_sequence_count_three(self) -> None:
        model = _make_model(
            depth_recurrence_shared_layers=[1, 2],
            depth_recurrence_count=3,
        )
        # prefix [0] + shared [1,2]*3 + suffix [3]
        self.assertEqual(
            model._virtual_layer_indices, [0, 1, 2, 1, 2, 1, 2, 3]
        )

    def test_full_stack_shared(self) -> None:
        """All layers shared, no prefix/suffix."""
        model = _make_model(
            num_layers=3,
            depth_recurrence_shared_layers=[0, 1, 2],
            depth_recurrence_count=2,
        )
        self.assertEqual(
            model._virtual_layer_indices, [0, 1, 2, 0, 1, 2]
        )
        ids = torch.randint(0, 64, (2, 8))
        out = model(ids)
        self.assertEqual(out["logits"].shape, (2, 8, 64))

    def test_single_layer_shared(self) -> None:
        """Group of one layer replayed — corner case."""
        model = _make_model(
            depth_recurrence_shared_layers=[2],
            depth_recurrence_count=3,
        )
        self.assertEqual(
            model._virtual_layer_indices, [0, 1, 2, 2, 2, 3]
        )


class TestDepthRecurrenceGradients(unittest.TestCase):
    def test_gradient_flow_count_two(self) -> None:
        torch.manual_seed(0)
        model = _make_model(
            depth_recurrence_shared_layers=[1, 2],
            depth_recurrence_count=2,
        )
        ids = torch.randint(0, 64, (2, 16))
        out = model(ids)
        loss = F.cross_entropy(out["logits"].reshape(-1, 64), ids.reshape(-1))
        loss.backward()

        # Shared-layer parameters must have non-zero, non-NaN gradients.
        for layer_idx in (1, 2):
            layer = model.layers[layer_idx]
            found_grad = False
            for name, p in layer.named_parameters():
                self.assertIsNotNone(
                    p.grad, f"layer {layer_idx} param {name} has no grad"
                )
                self.assertFalse(
                    torch.isnan(p.grad).any(),
                    f"layer {layer_idx} param {name} grad contains NaN",
                )
                if p.grad.abs().sum() > 0:
                    found_grad = True
            self.assertTrue(
                found_grad,
                f"layer {layer_idx} has no non-zero gradients",
            )

    def test_shared_layer_grad_accumulates_from_multiple_passes(self) -> None:
        """Verify grads on a shared layer reflect contributions from every pass.

        With count=2, the shared layer is traversed twice. Autograd should
        accumulate gradients from both traversals onto the same physical
        parameters. A single-pass model with the same weights should produce
        a strictly smaller gradient norm on the shared layer (because only
        one pass contributes).
        """
        torch.manual_seed(0)
        model_single = _make_model(
            depth_recurrence_shared_layers=[1, 2],
            depth_recurrence_count=1,
        )
        torch.manual_seed(0)
        model_double = _make_model(
            depth_recurrence_shared_layers=[1, 2],
            depth_recurrence_count=2,
        )
        model_double.load_state_dict(model_single.state_dict())

        ids = torch.randint(0, 64, (2, 16))

        for m in (model_single, model_double):
            out = m(ids)
            loss = out["logits"].sum()
            loss.backward()

        # Compare grad norms on layer 1 (shared in both).
        grad_single = sum(
            p.grad.pow(2).sum().item()
            for p in model_single.layers[1].parameters()
            if p.grad is not None
        )
        grad_double = sum(
            p.grad.pow(2).sum().item()
            for p in model_double.layers[1].parameters()
            if p.grad is not None
        )
        self.assertGreater(
            grad_double,
            grad_single,
            "Recurrent grad on shared layer should be larger than single-pass grad",
        )


class TestDepthRecurrenceParamCount(unittest.TestCase):
    def test_param_count_invariant(self) -> None:
        baseline = _make_model()
        recurrent = _make_model(
            depth_recurrence_shared_layers=[1, 2],
            depth_recurrence_count=4,
        )
        base_count = sum(p.numel() for p in baseline.parameters())
        rec_count = sum(p.numel() for p in recurrent.parameters())
        self.assertEqual(
            base_count,
            rec_count,
            "Depth recurrence must not change parameter count",
        )

    def test_artifact_bytes_invariant(self) -> None:
        baseline = _make_model()
        recurrent = _make_model(
            depth_recurrence_shared_layers=[0, 1, 2, 3],
            depth_recurrence_count=3,
        )
        self.assertEqual(
            baseline.artifact_bytes(),
            recurrent.artifact_bytes(),
            "Depth recurrence must not change artifact size",
        )


class TestDepthRecurrenceDistinctFromCountOne(unittest.TestCase):
    def test_count_two_output_differs_from_count_one(self) -> None:
        torch.manual_seed(0)
        m1 = _make_model(
            depth_recurrence_shared_layers=[1, 2],
            depth_recurrence_count=1,
        )
        torch.manual_seed(0)
        m2 = _make_model(
            depth_recurrence_shared_layers=[1, 2],
            depth_recurrence_count=2,
        )
        m2.load_state_dict(m1.state_dict())

        m1.eval()
        m2.eval()

        ids = torch.randint(0, 64, (2, 16))
        with torch.no_grad():
            out1 = m1(ids)["logits"]
            out2 = m2(ids)["logits"]

        self.assertFalse(
            torch.equal(out1, out2),
            "count=2 output must differ from count=1 (otherwise recurrence is a no-op)",
        )
        # Sanity: they should also not be near-zero-difference.
        diff = (out1 - out2).abs().max().item()
        self.assertGreater(diff, 1e-6)

    def test_count_three_output_differs_from_count_two(self) -> None:
        torch.manual_seed(0)
        m2 = _make_model(
            depth_recurrence_shared_layers=[1, 2],
            depth_recurrence_count=2,
        )
        torch.manual_seed(0)
        m3 = _make_model(
            depth_recurrence_shared_layers=[1, 2],
            depth_recurrence_count=3,
        )
        m3.load_state_dict(m2.state_dict())

        m2.eval()
        m3.eval()

        ids = torch.randint(0, 64, (2, 16))
        with torch.no_grad():
            out2 = m2(ids)["logits"]
            out3 = m3(ids)["logits"]

        self.assertFalse(torch.equal(out2, out3))


class TestDepthRecurrenceValidation(unittest.TestCase):
    def test_invalid_count_zero_raises(self) -> None:
        with self.assertRaises(ValueError):
            _make_model(
                depth_recurrence_shared_layers=[1, 2],
                depth_recurrence_count=0,
            )

    def test_out_of_range_index_raises(self) -> None:
        with self.assertRaises(ValueError):
            _make_model(
                depth_recurrence_shared_layers=[1, 99],
                depth_recurrence_count=2,
            )

    def test_non_contiguous_raises(self) -> None:
        with self.assertRaises(ValueError):
            _make_model(
                depth_recurrence_shared_layers=[0, 2],
                depth_recurrence_count=2,
            )

    def test_unsorted_raises(self) -> None:
        with self.assertRaises(ValueError):
            _make_model(
                depth_recurrence_shared_layers=[2, 1],
                depth_recurrence_count=2,
            )


if __name__ == "__main__":
    unittest.main()
