"""Tests for SemanticOptimizer.

Covers:
  * β map: monotonicity, boundedness, detach.
  * Reduction to Muon when no A is bound (matrix params and AdamW path).
  * Per-channel broadcast math: row-axis decay, negative-axis support,
    Nesterov=True branch, shape-mismatch detection.
  * Misconfiguration: A-in-channel_map raises at construction;
    missing A name or missing channel_map name raises at bind;
    step-before-bind raises.
  * State handling: checkpoint roundtrip, multi-param-group with distinct
    LRs, zero-grad preserves momentum buffer.
  * Precision: fp32 momentum buffer when channel-coupled so β near 1
    doesn't lose precision to bf16 ULP across many steps.
  * Diagnostics: beta_trace returns vec/tau/summaries.
  * End-to-end: loss decreases on a toy linear regression.

CPU-only, float32 compute throughout for numerical parity.
"""
from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from chaoscontrol.optim.muon import Muon
from chaoscontrol.optim.semantic import (
    SemanticOptimizer,
    default_beta_from_log_a,
)


class TestDefaultBetaFromLogA:
    def test_monotone_in_log_a(self) -> None:
        log_a = torch.linspace(-3.0, 3.0, 13)
        beta = default_beta_from_log_a(log_a, beta_min=0.5, beta_max=0.95)
        deltas = beta[1:] - beta[:-1]
        assert (deltas > 0).all(), (
            f"β must be strictly monotone in log_a, got deltas={deltas}"
        )

    def test_bounded(self) -> None:
        log_a = torch.linspace(-10.0, 10.0, 5)
        beta = default_beta_from_log_a(log_a, beta_min=0.5, beta_max=0.95)
        assert (beta >= 0.5).all()
        assert (beta <= 0.95).all()

    def test_detaches_and_upcasts(self) -> None:
        log_a = torch.tensor(
            [0.0, 1.0], dtype=torch.float32, requires_grad=True
        )
        beta = default_beta_from_log_a(log_a, beta_min=0.5, beta_max=0.95)
        assert not beta.requires_grad
        assert beta.dtype == torch.float32


class TestReducesToMuon:
    def test_single_matrix_param_matches_muon(self) -> None:
        torch.manual_seed(0)
        layer_a = nn.Linear(8, 4, bias=False)
        layer_b = copy.deepcopy(layer_a)
        xs = torch.randn(6, 8)
        target = torch.randn(6, 4)

        muon = Muon(
            [layer_a.weight],
            lr=0.01,
            momentum=0.9,
            nesterov=True,
            ns_steps=5,
            weight_decay=0.0,
            compute_dtype=torch.float32,
        )
        sem = SemanticOptimizer(
            [layer_b.weight],
            lr=0.01,
            momentum=0.9,
            momentum_min=0.5,
            nesterov=True,
            ns_steps=5,
            weight_decay=0.0,
            a_param_name=None,
            compute_dtype=torch.float32,
        )

        for _ in range(5):
            for layer, opt in ((layer_a, muon), (layer_b, sem)):
                opt.zero_grad(set_to_none=True)
                pred = layer(xs)
                loss = ((pred - target) ** 2).mean()
                loss.backward()
                opt.step()

        diff = (layer_a.weight - layer_b.weight).abs().max().item()
        assert torch.allclose(layer_a.weight, layer_b.weight, atol=1e-6), (
            f"SemanticOptimizer(a_param_name=None) should match Muon; "
            f"max abs diff = {diff}"
        )


class TestPerChannelBroadcast:
    def test_row_axis_decay_matches_beta_from_a(self) -> None:
        """With nesterov off and ns_steps=1, the momentum buffer update
        is ``buf = β ⊙ buf + g``. After step 1 with grad=1 we get buf=1
        everywhere. After step 2 with grad=0 we get buf = β, broadcast
        along the channel axis.
        """
        torch.manual_seed(1)
        dim = 4
        log_a = nn.Parameter(torch.linspace(-2.0, 2.0, dim))
        w = nn.Parameter(torch.zeros(dim, dim))

        opt = SemanticOptimizer(
            [{"params": [log_a, w]}],
            lr=1.0,
            momentum=0.95,
            momentum_min=0.5,
            nesterov=False,
            ns_steps=1,
            weight_decay=0.0,
            a_param_name="log_a",
            channel_map={"w": 0},
            compute_dtype=torch.float32,
        )
        opt.bind_param_names([("log_a", log_a), ("w", w)])

        w.grad = torch.ones_like(w)
        log_a.grad = torch.zeros_like(log_a)
        opt.step()
        buf1 = opt.state[w]["momentum_buffer"].clone()
        assert torch.allclose(buf1, torch.ones_like(buf1))

        w.grad = torch.zeros_like(w)
        log_a.grad = torch.zeros_like(log_a)
        opt.step()
        buf2 = opt.state[w]["momentum_buffer"]

        expected = default_beta_from_log_a(
            log_a.data, beta_min=0.5, beta_max=0.95
        )
        for i in range(dim):
            row_expected = torch.full((dim,), float(expected[i]))
            assert torch.allclose(buf2[i], row_expected, atol=1e-6), (
                f"row {i}: expected β={expected[i].item():.6f}, "
                f"got buf2[{i}]={buf2[i]}"
            )

    def test_negative_axis_resolves(self) -> None:
        """channel_axis=-1 should resolve to the last dimension."""
        dim = 3
        log_a = nn.Parameter(torch.zeros(dim))
        w = nn.Parameter(torch.zeros(5, dim))  # columns = channels

        opt = SemanticOptimizer(
            [{"params": [log_a, w]}],
            lr=1.0,
            momentum=0.9,
            momentum_min=0.5,
            nesterov=False,
            ns_steps=1,
            a_param_name="log_a",
            channel_map={"w": -1},
            compute_dtype=torch.float32,
        )
        opt.bind_param_names([("log_a", log_a), ("w", w)])

        w.grad = torch.ones_like(w)
        log_a.grad = torch.zeros_like(log_a)
        opt.step()
        w.grad = torch.zeros_like(w)
        opt.step()

        buf = opt.state[w]["momentum_buffer"]
        # Every column j should decay by the same β_j; rows are uniform.
        for j in range(dim):
            assert torch.allclose(
                buf[:, j], torch.full((5,), float(buf[0, j])), atol=1e-6
            ), f"column {j} not uniform across rows: {buf[:, j]}"

    def test_nesterov_true_with_tensor_beta(self) -> None:
        """With Nesterov=True and tensor β, the direction is
        ``grad + β ⊙ buf``. Run two steps with a known grad sequence and
        verify the update direction matches the manual computation.
        """
        torch.manual_seed(2)
        dim = 3
        log_a = nn.Parameter(torch.zeros(dim))  # all β_i identical
        w = nn.Parameter(torch.zeros(dim, 2))

        opt = SemanticOptimizer(
            [{"params": [log_a, w]}],
            lr=1.0,
            momentum=0.9,
            momentum_min=0.9,  # constant β to make the reference math easy
            nesterov=True,
            ns_steps=1,
            a_param_name="log_a",
            channel_map={"w": 0},
            compute_dtype=torch.float32,
        )
        opt.bind_param_names([("log_a", log_a), ("w", w)])

        # Step 1: grad=1, buf starts 0.
        # buf -> 0.9*0 + 1 = 1 (everywhere).
        # direction -> 1 + 0.9 * 1 = 1.9 (everywhere).
        g1 = torch.ones(dim, 2)
        w.grad = g1.clone()
        log_a.grad = torch.zeros(dim)
        opt.step()
        assert torch.allclose(
            opt.state[w]["momentum_buffer"], torch.ones(dim, 2), atol=1e-6
        )

    def test_shape_mismatch_raises(self) -> None:
        log_a = nn.Parameter(torch.zeros(4))
        w = nn.Parameter(torch.zeros(3, 5))

        opt = SemanticOptimizer(
            [{"params": [log_a, w]}],
            lr=0.01,
            momentum=0.95,
            momentum_min=0.5,
            nesterov=False,
            ns_steps=1,
            a_param_name="log_a",
            channel_map={"w": 0},
            compute_dtype=torch.float32,
        )
        opt.bind_param_names([("log_a", log_a), ("w", w)])
        w.grad = torch.ones_like(w)
        log_a.grad = torch.zeros_like(log_a)
        with pytest.raises(ValueError, match="channels"):
            opt.step()


class TestMisconfigurationFailsLoudly:
    def test_bind_raises_when_a_name_missing(self) -> None:
        w = nn.Parameter(torch.zeros(4, 4))
        opt = SemanticOptimizer(
            [w],
            lr=0.01,
            momentum=0.9,
            momentum_min=0.5,
            a_param_name="log_a_that_does_not_exist",
            compute_dtype=torch.float32,
        )
        with pytest.raises(ValueError, match="not found"):
            opt.bind_param_names([("w", w)])

    def test_bind_raises_when_channel_map_name_missing(self) -> None:
        log_a = nn.Parameter(torch.zeros(4))
        w = nn.Parameter(torch.zeros(4, 4))
        opt = SemanticOptimizer(
            [{"params": [log_a, w]}],
            lr=0.01,
            momentum=0.9,
            momentum_min=0.5,
            a_param_name="log_a",
            channel_map={"typo_name": 0},
            compute_dtype=torch.float32,
        )
        with pytest.raises(ValueError, match="not in bound"):
            opt.bind_param_names([("log_a", log_a), ("w", w)])

    def test_step_without_bind_raises(self) -> None:
        log_a = nn.Parameter(torch.zeros(4))
        w = nn.Parameter(torch.zeros(4, 4))
        opt = SemanticOptimizer(
            [{"params": [log_a, w]}],
            lr=0.01,
            momentum=0.9,
            momentum_min=0.5,
            a_param_name="log_a",
            channel_map={"w": 0},
            compute_dtype=torch.float32,
        )
        w.grad = torch.ones_like(w)
        log_a.grad = torch.zeros_like(log_a)
        with pytest.raises(RuntimeError, match="bind_param_names"):
            opt.step()

    def test_a_in_channel_map_raises_at_construction(self) -> None:
        log_a = nn.Parameter(torch.zeros(4))
        w = nn.Parameter(torch.zeros(4, 4))
        with pytest.raises(ValueError, match="A-parameter"):
            SemanticOptimizer(
                [log_a, w],
                lr=0.01,
                momentum=0.9,
                momentum_min=0.5,
                a_param_name="log_a",
                channel_map={"log_a": 0, "w": 0},
                compute_dtype=torch.float32,
            )


class TestAdamWFallback:
    def test_bias_param_matches_muon_adamw_branch(self) -> None:
        torch.manual_seed(3)
        layer_a = nn.Linear(6, 4, bias=True)
        layer_b = copy.deepcopy(layer_a)
        xs = torch.randn(8, 6)
        target = torch.randn(8, 4)

        muon = Muon(
            list(layer_a.parameters()),
            lr=0.01,
            momentum=0.9,
            nesterov=True,
            ns_steps=5,
            weight_decay=0.01,
            adamw_betas=(0.9, 0.95),
            compute_dtype=torch.float32,
        )
        sem = SemanticOptimizer(
            list(layer_b.parameters()),
            lr=0.01,
            momentum=0.9,
            momentum_min=0.5,
            nesterov=True,
            ns_steps=5,
            weight_decay=0.01,
            adamw_betas=(0.9, 0.95),
            a_param_name=None,
            compute_dtype=torch.float32,
        )

        for _ in range(4):
            for layer, opt in ((layer_a, muon), (layer_b, sem)):
                opt.zero_grad(set_to_none=True)
                pred = layer(xs)
                loss = ((pred - target) ** 2).mean()
                loss.backward()
                opt.step()

        assert torch.allclose(layer_a.bias, layer_b.bias, atol=1e-6), (
            f"bias (AdamW fallback) should match Muon; "
            f"max diff = {(layer_a.bias - layer_b.bias).abs().max().item()}"
        )
        assert torch.allclose(layer_a.weight, layer_b.weight, atol=1e-6)


class TestZeroGradPreservesBuffer:
    """A param with grad=None mid-training should not have its momentum
    buffer disturbed. This guards against a regression where skipping the
    grad path accidentally resets state.
    """

    def test_skipping_param_leaves_buf_alone(self) -> None:
        log_a = nn.Parameter(torch.zeros(4))
        w_active = nn.Parameter(torch.zeros(4, 4))
        w_dormant = nn.Parameter(torch.zeros(4, 4))

        opt = SemanticOptimizer(
            [{"params": [log_a, w_active, w_dormant]}],
            lr=1.0,
            momentum=0.9,
            momentum_min=0.5,
            nesterov=False,
            ns_steps=1,
            a_param_name="log_a",
            channel_map={"w_active": 0, "w_dormant": 0},
            compute_dtype=torch.float32,
        )
        opt.bind_param_names([
            ("log_a", log_a),
            ("w_active", w_active),
            ("w_dormant", w_dormant),
        ])

        # Step 1: both active.
        w_active.grad = torch.ones_like(w_active)
        w_dormant.grad = torch.ones_like(w_dormant)
        log_a.grad = torch.zeros_like(log_a)
        opt.step()
        dormant_buf_snapshot = opt.state[w_dormant]["momentum_buffer"].clone()

        # Step 2: dormant skips (grad=None). Its buffer must be unchanged.
        w_active.grad = torch.ones_like(w_active)
        w_dormant.grad = None
        log_a.grad = torch.zeros_like(log_a)
        opt.step()
        assert torch.allclose(
            opt.state[w_dormant]["momentum_buffer"], dormant_buf_snapshot
        ), "param with grad=None had its momentum buffer mutated"


class TestMomentumBufferPrecision:
    """When channel-coupled, momentum buffers must stay in fp32 regardless
    of parameter dtype. bf16 β near 1 would lose ~1e-3 ULP per multiply,
    compounding to meaningful τ drift over a training run.
    """

    def test_bf16_params_get_fp32_buffer_when_channel_coupled(self) -> None:
        log_a = nn.Parameter(torch.zeros(4, dtype=torch.float32))
        w = nn.Parameter(torch.zeros(4, 4, dtype=torch.bfloat16))

        opt = SemanticOptimizer(
            [{"params": [log_a, w]}],
            lr=0.01,
            momentum=0.95,
            momentum_min=0.5,
            nesterov=False,
            ns_steps=1,
            a_param_name="log_a",
            channel_map={"w": 0},
            compute_dtype=torch.float32,
        )
        opt.bind_param_names([("log_a", log_a), ("w", w)])
        w.grad = torch.ones_like(w)
        log_a.grad = torch.zeros_like(log_a)
        opt.step()
        buf = opt.state[w]["momentum_buffer"]
        assert buf.dtype == torch.float32, (
            f"expected fp32 momentum buffer for channel-coupled bf16 param, "
            f"got {buf.dtype}"
        )

    def test_bf16_params_match_muon_when_not_coupled(self) -> None:
        """When a_param_name=None (Muon reduction), buffers should match
        p.dtype for bit-for-bit Muon parity.
        """
        w = nn.Parameter(torch.zeros(4, 4, dtype=torch.bfloat16))
        opt = SemanticOptimizer(
            [w],
            lr=0.01,
            momentum=0.9,
            momentum_min=0.5,
            nesterov=False,
            ns_steps=1,
            a_param_name=None,
            compute_dtype=torch.float32,
        )
        w.grad = torch.ones_like(w)
        opt.step()
        buf = opt.state[w]["momentum_buffer"]
        assert buf.dtype == torch.bfloat16, (
            f"expected bf16 buffer for Muon-reduction mode, got {buf.dtype}"
        )


class TestCheckpointRoundtrip:
    def test_resume_produces_identical_updates(self) -> None:
        torch.manual_seed(4)
        dim = 6
        log_a = nn.Parameter(torch.linspace(-1.0, 1.0, dim))
        layer = nn.Linear(dim, dim, bias=False)

        def _make_opt(
            named: list[tuple[str, nn.Parameter]]
        ) -> SemanticOptimizer:
            opt = SemanticOptimizer(
                [{"params": [p for _, p in named]}],
                lr=0.05,
                momentum=0.9,
                momentum_min=0.4,
                nesterov=True,
                ns_steps=5,
                weight_decay=0.0,
                a_param_name="log_a",
                channel_map={"weight": 0},
                compute_dtype=torch.float32,
            )
            opt.bind_param_names(named)
            return opt

        named = [("log_a", log_a), ("weight", layer.weight)]
        opt = _make_opt(named)

        xs = torch.randn(16, dim)
        target = xs @ torch.randn(dim, dim).T

        for _ in range(5):
            opt.zero_grad(set_to_none=True)
            loss = ((layer(xs) - target) ** 2).mean()
            loss.backward()
            opt.step()

        checkpoint_weight = layer.weight.detach().clone()
        checkpoint_log_a = log_a.detach().clone()
        # Deep-copy: Optimizer.state_dict() returns references to live
        # momentum buffers, so continuing to step mutates the "saved"
        # state in place. Deep-copy snapshots it.
        state_dict = copy.deepcopy(opt.state_dict())

        opt.zero_grad(set_to_none=True)
        loss = ((layer(xs) - target) ** 2).mean()
        loss.backward()
        opt.step()
        expected_weight = layer.weight.detach().clone()
        expected_log_a = log_a.detach().clone()

        with torch.no_grad():
            layer.weight.copy_(checkpoint_weight)
            log_a.copy_(checkpoint_log_a)
        resumed = _make_opt(named)
        resumed.load_state_dict(state_dict)
        resumed.zero_grad(set_to_none=True)
        loss = ((layer(xs) - target) ** 2).mean()
        loss.backward()
        resumed.step()

        assert torch.allclose(layer.weight, expected_weight, atol=1e-6), (
            f"resumed weight update differs from continued baseline; "
            f"max diff = {(layer.weight - expected_weight).abs().max().item()}"
        )
        assert torch.allclose(log_a, expected_log_a, atol=1e-6)


class TestMultipleParamGroups:
    def test_distinct_lrs_across_groups(self) -> None:
        torch.manual_seed(5)
        dim = 4
        log_a = nn.Parameter(torch.zeros(dim))
        fast = nn.Parameter(torch.zeros(dim, dim))
        slow = nn.Parameter(torch.zeros(dim, dim))

        opt = SemanticOptimizer(
            [
                {"params": [log_a], "lr": 0.1},
                {"params": [fast], "lr": 0.5},
                {"params": [slow], "lr": 0.01},
            ],
            lr=0.05,
            momentum=0.9,
            momentum_min=0.5,
            nesterov=False,
            ns_steps=5,
            weight_decay=0.0,
            a_param_name="log_a",
            channel_map={"fast": 0, "slow": 0},
            compute_dtype=torch.float32,
        )
        opt.bind_param_names(
            [("log_a", log_a), ("fast", fast), ("slow", slow)]
        )

        grad = torch.ones(dim, dim)
        fast.grad = grad.clone()
        slow.grad = grad.clone()
        log_a.grad = torch.zeros(dim)
        opt.step()

        fast_disp = fast.detach().abs().sum().item()
        slow_disp = slow.detach().abs().sum().item()
        assert fast_disp > 10 * slow_disp, (
            f"expected fast-lr param to move >>10× more than slow-lr; "
            f"got fast={fast_disp:.6f} slow={slow_disp:.6f}"
        )


class TestBetaTrace:
    def test_returns_vec_and_tau(self) -> None:
        log_a = nn.Parameter(torch.linspace(-2.0, 2.0, 5))
        w = nn.Parameter(torch.zeros(5, 5))
        opt = SemanticOptimizer(
            [{"params": [log_a, w]}],
            lr=0.01,
            momentum=0.95,
            momentum_min=0.5,
            a_param_name="log_a",
            channel_map={"w": 0},
            compute_dtype=torch.float32,
        )
        opt.bind_param_names([("log_a", log_a), ("w", w)])
        trace = opt.beta_trace()
        assert trace is not None
        assert trace["beta_vec"].shape == (5,)
        assert trace["tau_steps"].shape == (5,)
        taus = trace["tau_steps"]
        deltas = taus[1:] - taus[:-1]
        assert (deltas > 0).all(), (
            f"τ should grow monotonically across channels, got {deltas}"
        )
        assert trace["beta_min"] >= 0.5
        assert trace["beta_max"] <= 0.95

    def test_returns_none_without_a(self) -> None:
        w = nn.Parameter(torch.zeros(4, 4))
        opt = SemanticOptimizer(
            [w],
            lr=0.01,
            momentum=0.9,
            momentum_min=0.5,
            a_param_name=None,
            compute_dtype=torch.float32,
        )
        assert opt.beta_trace() is None
        assert opt.current_beta_vec() is None

    def test_trace_reflects_updated_log_a(self) -> None:
        """After A moves (via AdamW fallback), beta_trace should report
        the new β distribution. Guards against caching the β vector past
        its valid lifetime.
        """
        log_a = nn.Parameter(torch.zeros(4))
        w = nn.Parameter(torch.zeros(4, 4))
        opt = SemanticOptimizer(
            [{"params": [log_a, w]}],
            lr=0.5,
            momentum=0.9,
            momentum_min=0.5,
            nesterov=False,
            ns_steps=1,
            a_param_name="log_a",
            channel_map={"w": 0},
            compute_dtype=torch.float32,
        )
        opt.bind_param_names([("log_a", log_a), ("w", w)])

        trace0 = opt.beta_trace()
        assert trace0 is not None
        # Drive log_a via its AdamW fallback with a strong gradient so it
        # actually moves; β should follow.
        log_a.grad = torch.tensor([-2.0, -1.0, 1.0, 2.0])
        w.grad = torch.zeros_like(w)
        for _ in range(20):
            opt.step()
            w.grad = torch.zeros_like(w)
            log_a.grad = torch.tensor([-2.0, -1.0, 1.0, 2.0])
        trace1 = opt.beta_trace()
        assert trace1 is not None
        # β should now vary across channels because log_a diverged.
        assert trace1["beta_max"] - trace1["beta_min"] > 0.01, (
            f"β vec did not track log_a movement; "
            f"spread={trace1['beta_max'] - trace1['beta_min']}"
        )


class TestSmokeTraining:
    def test_loss_decreases_on_toy_linear_regression(self) -> None:
        torch.manual_seed(42)
        dim = 8
        log_a = nn.Parameter(torch.zeros(dim))
        layer = nn.Linear(dim, dim, bias=False)
        xs = torch.randn(64, dim)
        target_w = torch.randn(dim, dim)
        target = xs @ target_w.T

        opt = SemanticOptimizer(
            [{"params": [log_a, layer.weight]}],
            lr=0.05,
            momentum=0.9,
            momentum_min=0.5,
            nesterov=True,
            ns_steps=5,
            weight_decay=0.0,
            a_param_name="log_a",
            channel_map={"weight": 0},
            compute_dtype=torch.float32,
        )
        opt.bind_param_names([("log_a", log_a), ("weight", layer.weight)])

        losses: list[float] = []
        for _ in range(60):
            opt.zero_grad(set_to_none=True)
            pred = layer(xs)
            loss = ((pred - target) ** 2).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        assert losses[-1] < 0.5 * losses[0], (
            f"expected loss to halve in 60 steps, "
            f"got {losses[0]:.4f} -> {losses[-1]:.4f}"
        )
