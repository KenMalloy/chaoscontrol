"""Unit tests for ``chaoscontrol.distributed``.

Covers helpers that don't require a live process group:
``resolve_ddp_context`` decodes (explicit args, env vars, fallback) and
``allreduce_grads``'s coalesced flatten/reduce/unflatten path vs the prior
per-param loop. The collective helpers' multi-rank behavior is exercised
end-to-end by ``test_ddp_integration.py`` and the pod smoke runs; here
we lock in local-process invariants a future refactor could silently
break (in-place grad update, flatten round-trip equivalence, numerical
parity with the reference per-param path under a mocked collective).
"""
from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from chaoscontrol.distributed import allreduce_grads, resolve_ddp_context


class TestClipGradNormFused:
    """Numerical parity with torch.nn.utils.clip_grad_norm_ at fp32 tolerance.

    Reduction order differs — stdlib uses per-tensor norms stacked and
    re-reduced, fused path uses one flat norm over the concatenated
    grads. Same math, same value to ~fp32 epsilon. Tests use allclose
    (not torch.equal) for clipped-regime comparisons.
    """

    @staticmethod
    def _build_model_with_mixed_grads(seed: int) -> nn.Module:
        torch.manual_seed(seed)
        model = nn.Sequential(nn.Linear(8, 4), nn.Linear(4, 2))
        for i, p in enumerate(model.parameters()):
            p.grad = torch.randn_like(p) * (i + 1) + float(i)
        return model

    def test_matches_reference_on_clipping_regime(self) -> None:
        """Grads well above max_norm — clipping must fire on both paths.

        The stdlib path computes total_norm as
        ``vector_norm(stack([norm(g) for g in grads]))`` — per-tensor
        norms then one final reduction over the stack. The fused path
        computes ``flat.norm()`` directly on the flattened buffer —
        one global reduction. Both are mathematically equivalent L2
        norms over the same values, but the reduction orders differ
        so the results agree to fp32 tolerance, not bit-identity.
        The clip coefficient inherits that tolerance, so clipped grads
        match via allclose (not torch.equal).
        """
        from chaoscontrol.distributed import clip_grad_norm_fused
        model_ref = self._build_model_with_mixed_grads(seed=7)
        model_new = self._build_model_with_mixed_grads(seed=7)
        max_norm = 0.1  # chosen to fire clipping on these grads
        ref_total = torch.nn.utils.clip_grad_norm_(
            model_ref.parameters(), max_norm,
        )
        new_total = clip_grad_norm_fused(model_new.parameters(), max_norm)
        assert torch.allclose(ref_total, new_total, rtol=1e-6, atol=0.0), (
            f"total_norm mismatch: ref={ref_total} new={new_total}"
        )
        for (n1, p1), (n2, p2) in zip(
            model_ref.named_parameters(), model_new.named_parameters(),
        ):
            assert n1 == n2
            assert torch.allclose(p1.grad, p2.grad, rtol=1e-6, atol=1e-7), (
                f"clipped grad mismatch on {n1!r}: "
                f"max abs diff = {(p1.grad - p2.grad).abs().max().item()}"
            )

    def test_matches_reference_below_clip_threshold(self) -> None:
        """Grads below max_norm — clip is an identity scaling on both paths.

        The fused path unconditionally multiplies by clip_coef_clamped (= 1.0
        when the total norm is below threshold) to avoid a GPU->CPU sync; the
        multiply is mathematically a no-op but the flatten/unflatten round-trip
        inherits the fp32 reduction-order drift from the norm computation, so
        allclose rather than torch.equal. Tolerance matches the clipping-regime
        test so a real regression (wrong clip coef, wrong slice) still fails.
        """
        from chaoscontrol.distributed import clip_grad_norm_fused
        model_ref = self._build_model_with_mixed_grads(seed=11)
        model_new = self._build_model_with_mixed_grads(seed=11)
        max_norm = 1e6  # above any realistic grad norm
        torch.nn.utils.clip_grad_norm_(model_ref.parameters(), max_norm)
        clip_grad_norm_fused(model_new.parameters(), max_norm)
        for (n1, p1), (n2, p2) in zip(
            model_ref.named_parameters(), model_new.named_parameters(),
        ):
            assert n1 == n2
            assert torch.allclose(p1.grad, p2.grad, rtol=1e-6, atol=1e-7), (
                f"below-threshold drift on {n1!r}: "
                f"max|diff|={(p1.grad - p2.grad).abs().max().item()}"
            )

    def test_empty_params_is_no_op(self) -> None:
        from chaoscontrol.distributed import clip_grad_norm_fused
        total = clip_grad_norm_fused([], max_norm=1.0)
        assert float(total) == 0.0

    def test_grad_identity_preserved(self) -> None:
        """p.grad tensor identity must survive — optimizer holds refs."""
        from chaoscontrol.distributed import clip_grad_norm_fused
        model = self._build_model_with_mixed_grads(seed=23)
        ptrs_before = [p.grad.data_ptr() for p in model.parameters()]
        clip_grad_norm_fused(model.parameters(), max_norm=0.1)
        ptrs_after = [p.grad.data_ptr() for p in model.parameters()]
        assert ptrs_before == ptrs_after

    def test_nonfinite_grad_does_not_raise_and_matches_stdlib(self) -> None:
        """inf / nan grads are not raised; total_norm is nonfinite; clip scales to ~0.

        Matches stdlib's default error_if_nonfinite=False behavior. The training
        loop's first bad batch hits this path — must not crash, must match the
        stdlib's nonfinite handling so a trainer can drop the step the same way
        regardless of which clip path is active.
        """
        from chaoscontrol.distributed import clip_grad_norm_fused
        # Build two identical models, both with one infinite grad and one normal.
        torch.manual_seed(41)
        model_ref = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
        model_new = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
        # Copy weights so grads we set are the only source of divergence.
        for pr, pn in zip(model_ref.parameters(), model_new.parameters()):
            pn.data.copy_(pr.data)
        params_ref = list(model_ref.parameters())
        params_new = list(model_new.parameters())
        for i, (pr, pn) in enumerate(zip(params_ref, params_new)):
            if i == 0:
                pr.grad = torch.full_like(pr, float("inf"))
                pn.grad = torch.full_like(pn, float("inf"))
            else:
                g = torch.randn_like(pr)
                pr.grad = g.clone()
                pn.grad = g.clone()
        ref_total = torch.nn.utils.clip_grad_norm_(params_ref, max_norm=1.0)
        new_total = clip_grad_norm_fused(params_new, max_norm=1.0)
        # Both paths report nonfinite total_norm — the training loop's bad-batch
        # detector reads this field; behavior must agree across paths.
        assert not torch.isfinite(ref_total)
        assert not torch.isfinite(new_total)

    def test_nan_grad_does_not_raise_and_matches_stdlib(self) -> None:
        """Separate test for NaN grads — same contract, different nonfinite path."""
        from chaoscontrol.distributed import clip_grad_norm_fused
        torch.manual_seed(42)
        model_ref = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
        model_new = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
        for pr, pn in zip(model_ref.parameters(), model_new.parameters()):
            pn.data.copy_(pr.data)
        params_ref = list(model_ref.parameters())
        params_new = list(model_new.parameters())
        for i, (pr, pn) in enumerate(zip(params_ref, params_new)):
            if i == 0:
                pr.grad = torch.full_like(pr, float("nan"))
                pn.grad = torch.full_like(pn, float("nan"))
            else:
                g = torch.randn_like(pr)
                pr.grad = g.clone()
                pn.grad = g.clone()
        ref_total = torch.nn.utils.clip_grad_norm_(params_ref, max_norm=1.0)
        new_total = clip_grad_norm_fused(params_new, max_norm=1.0)
        assert not torch.isfinite(ref_total)
        assert not torch.isfinite(new_total)


class TestResolveDDPContext:
    def test_explicit_args_passed_through(self) -> None:
        assert resolve_ddp_context(rank=2, world_size=4) == (2, 4)

    def test_one_of_explicit_raises(self) -> None:
        with pytest.raises(ValueError, match="both"):
            resolve_ddp_context(rank=1, world_size=None)
        with pytest.raises(ValueError, match="both"):
            resolve_ddp_context(rank=None, world_size=2)

    def test_env_vars_picked_up(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RANK", "3")
        monkeypatch.setenv("WORLD_SIZE", "8")
        assert resolve_ddp_context(rank=None, world_size=None) == (3, 8)

    def test_fallback_single_device(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Strip any inherited torchrun-style env so the fallback actually fires.
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        assert resolve_ddp_context(rank=None, world_size=None) == (0, 1)


def _ref_allreduce_grads_per_param(model: nn.Module) -> None:
    """Reference impl matching the pre-2026-04-17 per-param all-reduce loop.

    Kept local to this test file so a future edit of the production path
    cannot accidentally change the reference, and vice versa.
    """
    for p in model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)


class _MockAllReduce:
    """Deterministic stand-in for ``dist.all_reduce`` under monkeypatch.

    Simulates a 2-rank AVG that mutates the input tensor in place. The
    transform ``x -> 1.5*x + 0.05`` is chosen so identity input produces a
    non-trivial, position-independent output — any accidental identity
    elision (e.g., coalesced path silently skipping the copy-back) surfaces
    as a value mismatch, and any position-dependent bug (e.g., wrong stride
    on unflatten) surfaces as a per-tensor mismatch against the reference.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], int]] = []

    def __call__(
        self,
        tensor: torch.Tensor,
        op: object = None,
        group: object = None,
    ) -> None:
        self.calls.append((tuple(tensor.shape), int(tensor.numel())))
        tensor.mul_(1.5).add_(0.05)


class TestAllreduceGradsCoalesced:
    """Lock in the coalesced flatten/reduce/unflatten path against regression.

    Infrastructure rewrite to a private torch API (``torch._utils.
    _flatten_dense_tensors``) landed with no regression test — exactly the
    contamination pattern the 2026-04-16 DDP rewrite history warns
    against. These tests pin the public contract: one collective call,
    in-place grad updates, numerical equivalence with the per-param loop.
    """

    @staticmethod
    def _build_model_with_mixed_grads(seed: int) -> nn.Module:
        torch.manual_seed(seed)
        model = nn.Sequential(
            nn.Linear(8, 4),          # 2D weight + 1D bias
            nn.Linear(4, 2),
        )
        # Non-uniform grads so per-tensor differences show up if unflatten
        # slices the wrong segments.
        for i, p in enumerate(model.parameters()):
            p.grad = torch.randn_like(p) * (i + 1) + float(i)
        return model

    def test_flatten_unflatten_round_trips_values(self) -> None:
        """The flatten/unflatten helpers preserve every element.

        This is the smallest possible regression check on the private
        torch API we depend on — if a torch upgrade ever breaks these
        helpers' value contract, we want a unit test to catch it, not a
        bpb drift on a live pod.
        """
        tensors = [
            torch.randn(3, 5),
            torch.randn(7),
            torch.randn(2, 2, 2),
            torch.tensor(1.25),  # scalar
        ]
        flat = torch._utils._flatten_dense_tensors(tensors)
        round_tripped = torch._utils._unflatten_dense_tensors(flat, tensors)
        assert len(round_tripped) == len(tensors)
        for orig, rt in zip(tensors, round_tripped):
            assert torch.equal(orig, rt), "flatten/unflatten lost values"

    def test_coalesced_issues_one_collective_call(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """N parameters must produce exactly ONE all_reduce call.

        The whole point of coalescing is amortizing launch overhead; any
        future edit that re-introduces per-param calls should fail here
        before it hits a pod's NCCL dispatcher.
        """
        mock = _MockAllReduce()
        monkeypatch.setattr(dist, "all_reduce", mock)

        model = self._build_model_with_mixed_grads(seed=7)
        n_params_with_grad = sum(
            1 for p in model.parameters() if p.grad is not None
        )
        assert n_params_with_grad == 4  # 2 linears × (weight, bias)

        allreduce_grads(model, world_size=2)
        assert len(mock.calls) == 1, (
            f"expected 1 coalesced all_reduce call, got {len(mock.calls)}"
        )

    def test_grads_updated_in_place_preserves_data_ptr(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``p.grad`` identity (.data_ptr()) must survive the coalesced path.

        Optimizers and downstream hooks hold references to the exact grad
        tensor. If the coalesced path replaced ``p.grad`` with a new
        tensor, those references would dangle — every downstream optimizer
        step would then silently operate on pre-reduction grads.
        """
        monkeypatch.setattr(dist, "all_reduce", _MockAllReduce())
        model = self._build_model_with_mixed_grads(seed=11)
        ptrs_before = [p.grad.data_ptr() for p in model.parameters()]
        allreduce_grads(model, world_size=2)
        ptrs_after = [p.grad.data_ptr() for p in model.parameters()]
        assert ptrs_before == ptrs_after, (
            "coalesced all-reduce replaced grad tensors instead of updating "
            "them in place"
        )

    def test_numerical_parity_with_per_param_reference(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Coalesced path must produce byte-identical grads vs the per-param
        loop, under the same deterministic collective.

        Two independent models with identical starting grads — one runs
        the production coalesced path, one runs the reference per-param
        loop. Both call through the same mocked ``all_reduce``. Any
        divergence is a real regression in the coalescing rewrite.
        """
        monkeypatch.setattr(dist, "all_reduce", _MockAllReduce())

        model_coalesced = self._build_model_with_mixed_grads(seed=23)
        model_reference = self._build_model_with_mixed_grads(seed=23)
        # Sanity: same seed means same starting grads on both models.
        for pc, pr in zip(
            model_coalesced.parameters(), model_reference.parameters(),
        ):
            assert torch.equal(pc.grad, pr.grad)

        allreduce_grads(model_coalesced, world_size=2)
        _ref_allreduce_grads_per_param(model_reference)

        for (name_c, pc), (name_r, pr) in zip(
            model_coalesced.named_parameters(),
            model_reference.named_parameters(),
        ):
            assert name_c == name_r
            assert torch.equal(pc.grad, pr.grad), (
                f"coalesced vs per-param grad mismatch on {name_c!r}: "
                f"max abs diff = {(pc.grad - pr.grad).abs().max().item()}"
            )

    def test_empty_model_is_no_op(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A model with no grads must issue zero collective calls.

        Edge case the flatten helper would reject (empty list), so
        ``allreduce_grads`` has to short-circuit before calling it.
        """
        mock = _MockAllReduce()
        monkeypatch.setattr(dist, "all_reduce", mock)
        model = nn.Linear(2, 2)
        # No backward has run; p.grad is None for every parameter.
        assert all(p.grad is None for p in model.parameters())
        allreduce_grads(model, world_size=4)
        assert mock.calls == []
