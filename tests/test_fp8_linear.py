"""Bespoke fp8 Linear numerical validation against stock TE.

The bespoke path must produce arithmetically-equivalent output to stock
``te.Linear`` for every input shape we use in training. "Equivalent"
here means within fp8 representation granularity — the two paths run
the same math but may use different amax scaling strategies. Tolerance
is tuned per test and documented with its rationale.

Scaffold note (Task 1B-1): ``FusedFP8Linear`` currently delegates to
``te.Linear`` internally, so parity tests pass trivially. The harness
exists so Task 1B-2 (real ``_scaled_mm`` fused kernel) has a ready
numerical reference to validate against.

Skip semantics:
    * Module-level ``pytest.importorskip("transformer_engine")`` skips
      the entire file on TE-less hosts (dev macs).
    * ``cuda_required`` fixture additionally skips tests that need a
      live GPU (everything from test 4 downward).
    * On a CPU-only host WITH TE installed (unusual), tests 1-3 would
      run since they only construct + inspect parameters; tests 4-7
      would skip at the fixture. The FusedFP8Linear __init__ itself
      does not require CUDA.
"""
from __future__ import annotations

import pytest

pytest.importorskip(
    "transformer_engine",
    reason="bespoke fp8 tests require TE as reference implementation",
)
pytest.importorskip("torch")

import torch
import torch.nn as nn
import transformer_engine.pytorch as te  # type: ignore[import-not-found]

from chaoscontrol.kernels.fp8_linear import FusedFP8Linear


@pytest.fixture
def cuda_required() -> None:
    """Skip a test if CUDA is not available on the host."""
    if not torch.cuda.is_available():
        pytest.skip("bespoke fp8 requires CUDA")


# ---------------------------------------------------------------------------
# Construction + weight-adoption tests (CPU-safe).
# ---------------------------------------------------------------------------

def test_constructor_shapes() -> None:
    """Bare ``FusedFP8Linear(256, 512)`` exposes the expected parameter
    shapes and dtype. No CUDA required — TE construction alone is enough
    to exercise the scaffold's parameter layout."""
    layer = FusedFP8Linear(256, 512)
    assert layer.weight.shape == (512, 256)
    assert layer.weight.dtype == torch.bfloat16
    assert layer.bias is not None
    assert layer.bias.shape == (512,)
    assert layer.bias.dtype == torch.bfloat16


def test_from_nn_linear_copies_weights() -> None:
    """``from_nn_linear`` produces an instance whose weight + bias data
    is byte-equal to the source ``nn.Linear``'s parameters."""
    torch.manual_seed(0)
    src = nn.Linear(64, 128, bias=True, dtype=torch.bfloat16)
    # Randomize to something non-default so the copy is observable.
    with torch.no_grad():
        src.weight.data.normal_(mean=0.0, std=0.1)
        src.bias.data.normal_(mean=0.0, std=0.1)

    adopted = FusedFP8Linear.from_nn_linear(src)

    assert torch.equal(adopted.weight.data, src.weight.data)
    assert adopted.bias is not None
    assert torch.equal(adopted.bias.data, src.bias.data)


def test_bias_False() -> None:
    """``bias=False`` results in no bias parameter and forward still works
    (forward only exercised on CUDA; here we just check the structural
    contract)."""
    layer = FusedFP8Linear(256, 256, bias=False)
    assert layer.bias is None
    # ``bias`` must not appear in the parameter list either.
    param_names = [name for name, _ in layer.named_parameters()]
    assert "bias" not in param_names


# ---------------------------------------------------------------------------
# Parity tests — need CUDA because fp8 matmul is a GPU-only path.
# ---------------------------------------------------------------------------

def test_forward_matches_te_on_submission_regime_shape(cuda_required) -> None:
    """dim=256, batch=32 — the Phase 1 submission-regime target shape.

    Ref = stock ``te.Linear`` under ``te.fp8_autocast``. Bespoke =
    ``FusedFP8Linear.from_nn_linear(nn.Linear(...))`` with weights
    copied in from the reference. At scaffold stage the bespoke path
    IS stock TE, so parity is trivial; the ``rtol=atol=3e-2`` bound
    comes from the Task 1B-2 calibration in the implementation plan
    (3x the worst per-element fp8 error observed on a calibration run)
    and is the bar the fused kernel will be held to.
    """
    torch.manual_seed(0)
    ref = te.Linear(256, 256, device="cuda", params_dtype=torch.bfloat16)
    bespoke = FusedFP8Linear.from_nn_linear(
        nn.Linear(256, 256, device="cuda", dtype=torch.bfloat16),
    )
    with torch.no_grad():
        bespoke.weight.data.copy_(ref.weight.data)
        if ref.bias is not None and bespoke.bias is not None:
            bespoke.bias.data.copy_(ref.bias.data)

    x = torch.randn(32, 256, device="cuda", dtype=torch.bfloat16)
    with te.fp8_autocast(enabled=True):
        y_ref = ref(x)
    y_new = bespoke(x)

    assert torch.allclose(y_ref, y_new, rtol=3e-2, atol=3e-2), (
        "bespoke fp8 output drift: max abs diff = "
        f"{(y_ref - y_new).abs().max().item()}"
    )


# Batch sizes must satisfy TE's fp8 alignment contract: prod(shape[:-1])
# divisible by 8 and last dim divisible by 16. At dim=256 (div by 16)
# we only need batch % 8 == 0. batch=1 fails TE's pre-kernel assert at
# transformer_engine/pytorch/utils.py:443 on 2.13.0; don't exercise
# alignment-violating shapes in parity tests.
@pytest.mark.parametrize("batch", [8, 32, 1024])
def test_forward_matches_te_on_multiple_batch_sizes(
    cuda_required, batch: int,
) -> None:
    """Parity holds across fp8-aligned batch sizes — 8 is the smallest
    permitted by TE at dim=256, 1024 is the submission-regime batch."""
    torch.manual_seed(0)
    ref = te.Linear(256, 256, device="cuda", params_dtype=torch.bfloat16)
    bespoke = FusedFP8Linear.from_nn_linear(
        nn.Linear(256, 256, device="cuda", dtype=torch.bfloat16),
    )
    with torch.no_grad():
        bespoke.weight.data.copy_(ref.weight.data)
        if ref.bias is not None and bespoke.bias is not None:
            bespoke.bias.data.copy_(ref.bias.data)

    x = torch.randn(batch, 256, device="cuda", dtype=torch.bfloat16)
    with te.fp8_autocast(enabled=True):
        y_ref = ref(x)
    y_new = bespoke(x)

    assert torch.allclose(y_ref, y_new, rtol=3e-2, atol=3e-2), (
        f"batch={batch}: max abs diff = "
        f"{(y_ref - y_new).abs().max().item()}"
    )


def test_forward_matches_te_on_non_square(cuda_required) -> None:
    """Not every Linear in the model is square — exercise a 256 -> 768
    shape (the kind that shows up in MLP expansion layers)."""
    torch.manual_seed(0)
    ref = te.Linear(256, 768, device="cuda", params_dtype=torch.bfloat16)
    bespoke = FusedFP8Linear.from_nn_linear(
        nn.Linear(256, 768, device="cuda", dtype=torch.bfloat16),
    )
    with torch.no_grad():
        bespoke.weight.data.copy_(ref.weight.data)
        if ref.bias is not None and bespoke.bias is not None:
            bespoke.bias.data.copy_(ref.bias.data)

    x = torch.randn(32, 256, device="cuda", dtype=torch.bfloat16)
    with te.fp8_autocast(enabled=True):
        y_ref = ref(x)
    y_new = bespoke(x)

    assert torch.allclose(y_ref, y_new, rtol=3e-2, atol=3e-2), (
        "non-square (256->768) max abs diff = "
        f"{(y_ref - y_new).abs().max().item()}"
    )


def test_backward_produces_weight_grad(cuda_required) -> None:
    """Backward through ``FusedFP8Linear`` must produce a non-NaN weight
    gradient, matching stock TE's behavior. At scaffold stage this was
    trivial (delegate); post-1B-2 this rides on ``_scaled_mm``'s native
    autograd registration."""
    torch.manual_seed(0)
    bespoke = FusedFP8Linear.from_nn_linear(
        nn.Linear(256, 256, device="cuda", dtype=torch.bfloat16),
    )
    x = torch.randn(32, 256, device="cuda", dtype=torch.bfloat16)
    y = bespoke(x)
    loss = y.float().pow(2).mean()
    loss.backward()

    assert bespoke.weight.grad is not None, "weight.grad not populated"
    assert torch.isfinite(bespoke.weight.grad).all(), (
        "weight.grad contains NaN or Inf"
    )
    if bespoke.bias is not None:
        assert bespoke.bias.grad is not None, "bias.grad not populated"
        assert torch.isfinite(bespoke.bias.grad).all(), (
            "bias.grad contains NaN or Inf"
        )


# ---------------------------------------------------------------------------
# Task 1B-2 additions — fused-path structural and performance checks.
# ---------------------------------------------------------------------------


def test_scaled_mm_path_used() -> None:
    """Task 1B-2 removes the ``te.Linear`` scaffold delegate. The instance
    must NOT expose ``_te_delegate`` anymore — that's our structural
    proof that forward is running through ``torch._scaled_mm`` and not
    silently falling back to the scaffold path."""
    layer = FusedFP8Linear(256, 256)
    assert not hasattr(layer, "_te_delegate"), (
        "FusedFP8Linear still has a _te_delegate attribute — the Task 1B-2 "
        "fused path should have removed it."
    )


def test_amax_buffers_registered() -> None:
    """Amax history ring buffers must be persistent buffers so
    checkpoints capture the scale-factor trajectory."""
    layer = FusedFP8Linear(256, 256)
    sd = layer.state_dict()
    assert "x_amax_history" in sd, "x_amax_history missing from state_dict"
    assert "w_amax_history" in sd, "w_amax_history missing from state_dict"
    # Default length 16, fp32.
    assert sd["x_amax_history"].shape == (16,)
    assert sd["x_amax_history"].dtype == torch.float32
    assert sd["w_amax_history"].shape == (16,)
    assert sd["w_amax_history"].dtype == torch.float32


def test_forward_still_matches_te_post_fusion(cuda_required) -> None:
    """Post-fusion parity check — same bar as
    ``test_forward_matches_te_on_submission_regime_shape`` but named
    explicitly to make the Task 1B-2 success gate visible. If this
    fails the scale-direction convention in ``_scaled_mm`` is almost
    certainly flipped — verify empirically by constructing an identity
    weight + unit input and inspecting the output before touching
    anything else."""
    torch.manual_seed(0)
    ref = te.Linear(256, 256, device="cuda", params_dtype=torch.bfloat16)
    bespoke = FusedFP8Linear.from_nn_linear(
        nn.Linear(256, 256, device="cuda", dtype=torch.bfloat16),
    )
    with torch.no_grad():
        bespoke.weight.data.copy_(ref.weight.data)
        if ref.bias is not None and bespoke.bias is not None:
            bespoke.bias.data.copy_(ref.bias.data)

    x = torch.randn(32, 256, device="cuda", dtype=torch.bfloat16)
    with te.fp8_autocast(enabled=True):
        y_ref = ref(x)
    y_new = bespoke(x)

    assert torch.allclose(y_ref, y_new, rtol=3e-2, atol=3e-2), (
        "post-fusion parity drift: max abs diff = "
        f"{(y_ref - y_new).abs().max().item()}"
    )


@pytest.mark.slow
def test_forward_no_sync_per_call(cuda_required) -> None:
    """Throughput microbench — plan step 6.

    Fused path must beat stock TE by at least 30% (``bespoke_time <
    0.7 * te_time``) on the submission-regime shape. If this fails the
    fusion isn't buying anything and the engineering isn't worth it.
    Marked ``@pytest.mark.slow`` so CI skips it by default; runs
    explicitly on the pod via ``pytest -m slow``.
    """
    import time

    dim = 256
    batch = 1024
    iters = 200
    warmup = 20

    torch.manual_seed(0)
    ref = te.Linear(dim, dim, device="cuda", params_dtype=torch.bfloat16)
    bespoke = FusedFP8Linear.from_nn_linear(
        nn.Linear(dim, dim, device="cuda", dtype=torch.bfloat16),
    )
    with torch.no_grad():
        bespoke.weight.data.copy_(ref.weight.data)
        if ref.bias is not None and bespoke.bias is not None:
            bespoke.bias.data.copy_(ref.bias.data)

    x = torch.randn(batch, dim, device="cuda", dtype=torch.bfloat16)

    # Warmup — compile, autotune, amax history fill.
    for _ in range(warmup):
        with te.fp8_autocast(enabled=True):
            _ = ref(x)
        _ = bespoke(x)
    torch.cuda.synchronize()

    # Time stock TE.
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        with te.fp8_autocast(enabled=True):
            _ = ref(x)
    torch.cuda.synchronize()
    te_time = time.perf_counter() - t0

    # Time bespoke.
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = bespoke(x)
    torch.cuda.synchronize()
    bespoke_time = time.perf_counter() - t0

    assert bespoke_time < 0.7 * te_time, (
        f"fused path not fast enough: bespoke={bespoke_time:.4f}s "
        f"te={te_time:.4f}s ratio={bespoke_time / te_time:.3f} "
        "(need < 0.7×)"
    )
