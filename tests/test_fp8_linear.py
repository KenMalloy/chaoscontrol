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

from chaoscontrol.kernels.fp8_linear import (
    FusedFP8Linear,
    fused_fp8_flush_all,
)


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


# ---------------------------------------------------------------------------
# Task 1B-3 additions — deferred amax + cuBLASLt-fork swap.
# ---------------------------------------------------------------------------


def test_flush_amax_history_exists_and_noop_without_call() -> None:
    """``flush_amax_history`` exists at the expected name. Skipping it
    between steps is permitted: scales stay stale but forward/backward
    still produce finite output. No CUDA required for the API check;
    a skip_if_cuda live-path test exercises the flush on a real tensor."""
    layer = FusedFP8Linear(64, 64)
    assert hasattr(layer, "flush_amax_history"), (
        "FusedFP8Linear must expose flush_amax_history() for the training "
        "loop to call once per optimizer step"
    )
    # Pending + scale buffers are part of the module state.
    sd = layer.state_dict()
    for name in ("x_amax_pending", "w_amax_pending",
                 "gy_amax_pending", "gx_amax_pending",
                 "x_scale", "w_scale", "gy_scale"):
        assert name in sd, f"{name} missing from state_dict"
    # Scales default to 1.0 — cold-start path.
    assert float(sd["x_scale"].item()) == 1.0
    assert float(sd["w_scale"].item()) == 1.0
    assert float(sd["gy_scale"].item()) == 1.0


def test_flush_amax_history_updates_scales(cuda_required) -> None:
    """After a forward + flush, the scales reflect the observed amax.

    This is the real correctness check on deferred amax: the first
    forward folds its amax into ``x_amax_pending`` via ``torch.maximum``;
    the flush rolls pending into history and recomputes the scale. A
    second forward at different magnitude must see the UPDATED scale.
    """
    torch.manual_seed(0)
    layer = FusedFP8Linear.from_nn_linear(
        nn.Linear(64, 64, device="cuda", dtype=torch.bfloat16),
    )
    # Forward with a known amax (~3.0).
    x = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16) * 3.0
    _ = layer(x)

    # Before flush: scales still at cold-start 1.0.
    assert float(layer.x_scale.item()) == 1.0
    # Pending: populated to something > 0.
    assert float(layer.x_amax_pending.item()) > 0.0

    # Flush — pending rolls into history, scale recomputes to
    # max(history) / 448.
    layer.flush_amax_history()
    new_scale = float(layer.x_scale.item())
    assert new_scale > 0.0, "x_scale zero after flush"
    # Expected: roughly the observed amax / 448 ~ 3*sqrt(2*ln(64*32))/448.
    # Just check it's in a sane range (>0, <1).
    assert 0.0 < new_scale < 1.0, f"x_scale out of expected range: {new_scale}"
    # Pending zeroed.
    assert float(layer.x_amax_pending.item()) == 0.0


def test_fused_fp8_flush_all_walks_model(cuda_required) -> None:
    """``fused_fp8_flush_all`` finds every FusedFP8Linear in a model
    tree and flushes each. Returns the count so training loops can
    sanity-check the walk."""
    model = nn.Sequential(
        FusedFP8Linear(32, 32, device="cuda"),
        nn.ReLU(),
        FusedFP8Linear(32, 32, device="cuda"),
    ).cuda()
    x = torch.randn(8, 32, device="cuda", dtype=torch.bfloat16)
    _ = model(x)
    n = fused_fp8_flush_all(model)
    assert n == 2, f"expected 2 flushes, got {n}"
    # Scales on both submodules now non-default.
    for sub in model.modules():
        if isinstance(sub, FusedFP8Linear):
            assert float(sub.x_amax_pending.item()) == 0.0


@pytest.mark.slow
def test_forward_backward_report_bench(cuda_required, capsys) -> None:
    """Whole-stack fwd+bwd benchmark — reported number only.

    Our FusedFP8Linear.forward + .backward() combined vs stock TE
    te.Linear + fp8_autocast fwd+bwd at dim=256 batch=1024. The ratio
    is PRINTED for the report but NOT asserted: the gate named in the
    task spec is forward-only (``test_forward_no_sync_per_call``, the
    0.7× TE bar). Backward on our side still materializes two
    transposes (weight -> col-major, x_fp8 -> col-major) that TE
    elides by caching fp8 operands across the fwd→bwd boundary; that's
    a follow-up optimization, not a 1B-3 blocker.
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

    def run_te() -> None:
        x = torch.randn(batch, dim, device="cuda", dtype=torch.bfloat16,
                        requires_grad=True)
        with te.fp8_autocast(enabled=True):
            y = ref(x)
        y.float().pow(2).mean().backward()

    def run_bespoke() -> None:
        x = torch.randn(batch, dim, device="cuda", dtype=torch.bfloat16,
                        requires_grad=True)
        y = bespoke(x)
        y.float().pow(2).mean().backward()
        bespoke.flush_amax_history()

    for _ in range(warmup):
        run_te()
        run_bespoke()
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        run_te()
    torch.cuda.synchronize()
    te_time = time.perf_counter() - t0

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        run_bespoke()
    torch.cuda.synchronize()
    bespoke_time = time.perf_counter() - t0

    with capsys.disabled():
        print(
            f"\n[fused-fp8 fwd+bwd bench] dim={dim} batch={batch} iters={iters}\n"
            f"  stock TE  = {te_time * 1e3:.2f} ms   ({te_time * 1e6 / iters:.2f} us/iter)\n"
            f"  bespoke   = {bespoke_time * 1e3:.2f} ms   "
            f"({bespoke_time * 1e6 / iters:.2f} us/iter)\n"
            f"  ratio     = {bespoke_time / te_time:.3f}\n"
        )
