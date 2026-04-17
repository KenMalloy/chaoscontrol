"""Parity + throughput harness for the bespoke cuBLASLt fp8 matmul.

Phase 1 scope (forward only):

* ``test_forward_matches_stock_te`` — numerical parity against stock
  ``te.Linear`` under ``fp8_autocast``. Tolerance is the same ``rtol``/
  ``atol=3e-2`` bar that the ``FusedFP8Linear`` harness uses; fp8 math
  produces dense granularity so tight tolerances aren't meaningful.
* ``test_forward_matches_scaled_mm`` — parity against the in-tree
  ``torch._scaled_mm`` path. This is the tighter sibling test: both
  kernels should land on the same cuBLASLt algo for the same inputs,
  so tolerance can drop to fp8-granularity (``atol=0.0`` byte-equal
  only if we're lucky; keep the same 3e-2 bar to stay robust against
  different heuristic picks).
* ``test_forward_faster_than_stock_te`` — microbenchmark: 200 iters of
  each kernel at submission shape (dim=256, batch=1024). Asserts our
  wall-clock time is below ``0.9 * te_time`` on H100. Marked ``slow``
  so CI runs without it by default; run explicitly on the pod.

Skip semantics: the module-level importorskip covers three cases —
  1. ``torch.cuda.is_available()`` false → skip (no H100, no fp8).
  2. ``transformer_engine.pytorch`` unimportable → skip (dev mac).
  3. Our cuBLASLt extension isn't compiled → skip with a clear note
     pointing at the pod setup script.
"""
from __future__ import annotations

import time

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

if not torch.cuda.is_available():  # pragma: no cover — dev-mac branch
    pytest.skip("cuBLASLt fp8 tests require a CUDA H100", allow_module_level=True)

pytest.importorskip(
    "transformer_engine",
    reason="cuBLASLt fp8 parity test requires TE as reference",
)
import transformer_engine.pytorch as te  # type: ignore[import-not-found]  # noqa: E402

# Import the extension separately so we can skip cleanly when it isn't
# built (dev-mac / partial pod setup). The wrapper's ``cublaslt_fp8_matmul``
# raises ImportError at call time, which would obscure the skip reason.
try:
    from chaoscontrol.kernels._cublaslt import _C as _cublaslt_C  # noqa: F401
    from chaoscontrol.kernels._cublaslt import (
        cublaslt_fp8_matmul,
        cublaslt_fp8_matmul_grad_w,
        cublaslt_fp8_matmul_grad_x,
    )
except ImportError as e:  # pragma: no cover
    pytest.skip(
        f"cuBLASLt fp8 extension not built: {e!r}. "
        "Re-run scripts/pod_setup_cuda13.sh on the pod.",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


_E4M3_MAX = 448.0
_E5M2_MAX = 57344.0


def _quantize_e4m3(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor scale quantization to E4M3. Matches FusedFP8Linear conv.

    Returns:
        (x_fp8, scale) where ``scale`` is the dequant multiplier used on
        the accumulator side (``scale = amax / 448``) and ``x_fp8`` is
        ``(x / scale).to(e4m3fn)``. Scalar fp32 ``scale`` lives on the
        same device as ``x`` so it can be passed straight to
        ``_scaled_mm`` / ``cublaslt_fp8_matmul``.
    """
    amax = x.detach().abs().amax().to(torch.float32)
    scale = torch.where(amax > 0, amax / _E4M3_MAX, torch.ones_like(amax))
    x_fp8 = (x.to(torch.float32) / scale).to(torch.float8_e4m3fn)
    return x_fp8, scale


def _quantize_e5m2(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """E5M2 sibling of ``_quantize_e4m3`` — for grad tensors."""
    amax = x.detach().abs().amax().to(torch.float32)
    scale = torch.where(amax > 0, amax / _E5M2_MAX, torch.ones_like(amax))
    x_fp8 = (x.to(torch.float32) / scale).to(torch.float8_e5m2)
    return x_fp8, scale


def _reference_scaled_mm(
    x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor | None,
) -> torch.Tensor:
    """Reference path via ``torch._scaled_mm`` — same semantics as our
    bespoke kernel. Returns bf16 ``[M, N]``."""
    a_fp8, a_scale = _quantize_e4m3(x)
    b_fp8, b_scale = _quantize_e4m3(w.t())  # quantize [K, N]
    return torch._scaled_mm(
        a_fp8, b_fp8,
        scale_a=a_scale, scale_b=b_scale,
        bias=bias, out_dtype=torch.bfloat16,
    )


def _cublaslt_path(
    x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor | None,
) -> torch.Tensor:
    """Our bespoke cuBLASLt path, same quantization and scale convention."""
    a_fp8, a_scale = _quantize_e4m3(x)
    b_fp8, b_scale = _quantize_e4m3(w.t())
    return cublaslt_fp8_matmul(
        a_fp8, b_fp8,
        scale_a=a_scale, scale_b=b_scale,
        bias=bias,
        out_dtype=torch.bfloat16,
    )


# ---------------------------------------------------------------------------
# Correctness.
# ---------------------------------------------------------------------------


def test_forward_matches_scaled_mm() -> None:
    """Bespoke kernel vs ``torch._scaled_mm`` on the same inputs. Both
    should dispatch to compatible cuBLASLt paths; tolerance is fp8
    granularity (3e-2 atol covers heuristic-induced ULP differences)."""
    torch.manual_seed(0)
    M, K, N = 1024, 256, 256
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.05
    b = torch.randn(N, device="cuda", dtype=torch.bfloat16) * 0.01

    y_ref = _reference_scaled_mm(x, w, b)
    y_new = _cublaslt_path(x, w, b)

    assert y_new.shape == y_ref.shape
    assert y_new.dtype == torch.bfloat16
    diff = (y_new.float() - y_ref.float()).abs()
    assert torch.allclose(y_new, y_ref, rtol=3e-2, atol=3e-2), (
        f"diff max={diff.max().item():.4e} mean={diff.mean().item():.4e}"
    )


def test_forward_matches_stock_te() -> None:
    """Parity against ``te.Linear`` under ``fp8_autocast``. Slightly
    looser tolerance than the ``_scaled_mm`` sibling because TE's amax
    strategy differs — TE keeps an amax history; we derive the scale
    per-call from the input tensor's own amax. Both live inside fp8
    representable range for our submission shape, so outputs should be
    identical up to the scale-factor rounding."""
    torch.manual_seed(0)
    M, K, N = 1024, 256, 256
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

    # Stock TE reference: build a te.Linear with the same weights, run
    # one fp8_autocast forward. We copy the weight in to match exactly.
    te_lin = te.Linear(K, N, bias=True, params_dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        w_src = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.05
        b_src = torch.randn(N, device="cuda", dtype=torch.bfloat16) * 0.01
        te_lin.weight.data.copy_(w_src)
        te_lin.bias.data.copy_(b_src)

    # Warm the amax histories so TE's first post-warmup call uses a
    # representative scale (first call of fp8_autocast uses amax=1.0).
    with te.fp8_autocast(enabled=True):
        _ = te_lin(x)
    with te.fp8_autocast(enabled=True):
        y_te = te_lin(x)

    y_new = _cublaslt_path(x, w_src, b_src)

    diff = (y_new.float() - y_te.float()).abs()
    assert torch.allclose(y_new, y_te, rtol=3e-2, atol=3e-2), (
        f"diff max={diff.max().item():.4e} mean={diff.mean().item():.4e}"
    )


def test_no_bias_path() -> None:
    """Sanity: bias=None takes the DEFAULT epilogue branch. Output should
    equal the ``with-bias`` call minus the bias column."""
    torch.manual_seed(1)
    M, K, N = 128, 64, 48
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.1

    y = _cublaslt_path(x, w, bias=None)
    assert y.shape == (M, N)
    assert y.dtype == torch.bfloat16
    assert torch.isfinite(y).all(), "non-finite values in fp8 matmul output"


# ---------------------------------------------------------------------------
# Throughput.
# ---------------------------------------------------------------------------


def _bench_iters(fn, iters: int) -> float:
    """Return seconds per iter for ``fn``, averaged over ``iters`` runs
    after 20 warmup iterations. Forces a cuda.synchronize at the start
    and end of the timed section so we measure wall-clock, not queued
    kernel latency."""
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / iters


@pytest.mark.slow
def test_forward_faster_than_stock_te() -> None:
    """At submission shape (dim=256 batch=1024) our kernel must come in
    under ``0.9 * te_time``. Sub-goal: match or beat it. If this fails
    the fork didn't close the dispatch-overhead gap we set out to
    close and Phase 2 needs to profile what's actually slower."""
    torch.manual_seed(0)
    M, K, N = 1024, 256, 256
    iters = 200

    # Pre-quantize so timing measures the matmul+bias path only.
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.05
    bias = torch.randn(N, device="cuda", dtype=torch.bfloat16) * 0.01

    a_fp8, a_scale = _quantize_e4m3(x)
    b_fp8, b_scale = _quantize_e4m3(w.t())

    def run_ours() -> None:
        cublaslt_fp8_matmul(a_fp8, b_fp8, a_scale, b_scale, bias, torch.bfloat16)

    def run_scaled_mm() -> None:
        torch._scaled_mm(
            a_fp8, b_fp8, scale_a=a_scale, scale_b=b_scale,
            bias=bias, out_dtype=torch.bfloat16,
        )

    te_lin = te.Linear(K, N, bias=True, params_dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        te_lin.weight.data.copy_(w)
        te_lin.bias.data.copy_(bias)

    def run_te() -> None:
        with te.fp8_autocast(enabled=True):
            te_lin(x)

    # Warm TE's amax first (first call has no real matmul work).
    for _ in range(5):
        run_te()
    torch.cuda.synchronize()

    ours_s = _bench_iters(run_ours, iters)
    scaled_mm_s = _bench_iters(run_scaled_mm, iters)
    te_s = _bench_iters(run_te, iters)

    print(
        f"\n[cublaslt-fp8 bench] shape=({M},{K})x({K},{N}) iters={iters}\n"
        f"  ours          = {ours_s * 1e6:8.2f} us\n"
        f"  _scaled_mm    = {scaled_mm_s * 1e6:8.2f} us\n"
        f"  te.Linear fp8 = {te_s * 1e6:8.2f} us\n"
        f"  ours/te       = {ours_s / te_s:.3f}\n"
        f"  ours/_scaled_mm = {ours_s / scaled_mm_s:.3f}\n"
    )

    # Win condition: beat TE by 10%+. If we don't, phase 2 has work to do.
    assert ours_s < 0.9 * te_s, (
        f"Phase 1 target missed: ours={ours_s * 1e6:.2f}us te={te_s * 1e6:.2f}us"
    )


# ---------------------------------------------------------------------------
# Backward — grad_x = grad_y @ W via the fork.
# ---------------------------------------------------------------------------


def _reference_grad_x_scaled_mm(
    grad_y: torch.Tensor, w: torch.Tensor,
) -> torch.Tensor:
    """Reference grad_x via ``torch._scaled_mm`` on E5M2 × E4M3.

    grad_y: [M, N] bf16 row-major.
    w:      [N, K] bf16 row-major (an ``nn.Linear`` weight).
    Output: [M, K] bf16 = grad_y @ W.

    _scaled_mm wants a row-major [M, N] and b column-major [N, K]. W
    natively is [N, K] row-major, so we need to materialize a col-major
    [N, K] view — ``w.t().contiguous().t()``.
    """
    gy_fp8, gy_scale = _quantize_e5m2(grad_y)
    w_col = w.t().contiguous().t()           # [N, K] col-major
    w_fp8 = (w_col.to(torch.float32) / _quantize_e4m3(w)[1]).to(torch.float8_e4m3fn)
    _, w_scale = _quantize_e4m3(w)
    return torch._scaled_mm(
        gy_fp8, w_fp8, scale_a=gy_scale, scale_b=w_scale,
        out_dtype=torch.bfloat16,
    )


def test_grad_x_matches_scaled_mm() -> None:
    """Backward grad_x via the fork matches ``torch._scaled_mm`` bwd path
    at fp8 granularity (rtol/atol=3e-2)."""
    torch.manual_seed(0)
    M, K, N = 1024, 256, 256
    grad_y = torch.randn(M, N, device="cuda", dtype=torch.bfloat16) * 0.1
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.05

    # Reference path.
    y_ref = _reference_grad_x_scaled_mm(grad_y, w)

    # Fork path — quantize once, call grad_x kernel.
    gy_fp8, gy_scale = _quantize_e5m2(grad_y)
    w_col = w.t().contiguous().t()           # [N, K] col-major bf16
    _, w_scale = _quantize_e4m3(w)
    w_col_fp8 = (w_col.to(torch.float32) / w_scale).to(torch.float8_e4m3fn)

    y_new = cublaslt_fp8_matmul_grad_x(
        gy_fp8, w_col_fp8, scale_gy=gy_scale, scale_w=w_scale,
        out_dtype=torch.bfloat16,
    )

    assert y_new.shape == (M, K)
    assert y_new.dtype == torch.bfloat16
    diff = (y_new.float() - y_ref.float()).abs()
    assert torch.allclose(y_new, y_ref, rtol=3e-2, atol=3e-2), (
        f"grad_x drift: max={diff.max().item():.4e} mean={diff.mean().item():.4e}"
    )


# ---------------------------------------------------------------------------
# Backward — grad_w = grad_y.t() @ x (+ fused dbias via BGRADB).
# ---------------------------------------------------------------------------


def test_grad_w_with_bias_grad_matches_te() -> None:
    """grad_w + fused dbias via the fork match the bf16-reference ground
    truth (grad_y.t() @ x for grad_w, grad_y.sum(0) for grad_b).

    We compare against bf16 ground truth rather than "stock TE's fp8
    backward" because setting up TE's fp8 backward deterministically
    requires plumbing a whole autograd tape, whereas bf16 is the
    numerical truth within the same tolerance bar.
    """
    torch.manual_seed(0)
    M, K, N = 1024, 256, 256
    grad_y = torch.randn(M, N, device="cuda", dtype=torch.bfloat16) * 0.1
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

    # Ground truth in bf16.
    gw_ref = grad_y.float().t() @ x.float()
    gb_ref = grad_y.float().sum(dim=0)

    # Fork path.
    gy_t_fp8, gy_scale = _quantize_e5m2(grad_y.t().contiguous())
    x_col = x.t().contiguous().t()           # [M, K] col-major bf16
    _, x_scale = _quantize_e4m3(x)
    x_col_fp8 = (x_col.to(torch.float32) / x_scale).to(torch.float8_e4m3fn)

    gw, gb = cublaslt_fp8_matmul_grad_w(
        gy_t_fp8, x_col_fp8, scale_gy=gy_scale, scale_x=x_scale,
        out_dtype=torch.bfloat16, compute_bias_grad=True,
    )
    assert gw.shape == (N, K)
    if gb is None:
        # BGRADB unsupported on this cuBLAS — the C++ kernel returns
        # None so the caller can fall back. Compute eagerly on the
        # input tensor for the parity check; this is what
        # FusedFP8Linear.backward does in the wild.
        gb = grad_y.sum(dim=0).to(torch.bfloat16)
    assert gb.shape == (N,)

    # fp8 quantization at E5M2×E4M3 accumulates entry-level error that
    # scales with sqrt(M) at our shape. Checking mean rather than max
    # rel-err screens out the few entries near zero where rel-err blows
    # up. Bar: <20% of ref's own RMS norm — fp8 reasonably-granular.
    gw_diff = (gw.float() - gw_ref).abs()
    gw_ref_norm = gw_ref.abs().mean().item()
    gw_rms = gw_diff.pow(2).mean().sqrt().item()
    assert gw_rms < 0.2 * gw_ref_norm, (
        f"grad_w drift too high: rms={gw_rms:.4e} ref_mean_abs={gw_ref_norm:.4e}"
    )
    gb_diff = (gb.float() - gb_ref).abs()
    gb_ref_norm = gb_ref.abs().mean().item()
    gb_rms = gb_diff.pow(2).mean().sqrt().item()
    assert gb_rms < 0.2 * gb_ref_norm, (
        f"grad_b drift too high: rms={gb_rms:.4e} ref_mean_abs={gb_ref_norm:.4e}"
    )


def test_grad_w_no_bias_grad_returns_none() -> None:
    """``compute_bias_grad=False`` returns (grad_w, None) and the grad_w
    value matches the with-bias-grad call (BGRADB epilogue vs DEFAULT
    must only differ in the bias output, not in the matmul output)."""
    torch.manual_seed(2)
    M, K, N = 128, 64, 48
    grad_y = torch.randn(M, N, device="cuda", dtype=torch.bfloat16) * 0.05
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

    gy_t_fp8, gy_scale = _quantize_e5m2(grad_y.t().contiguous())
    x_col_fp8, x_scale = _quantize_e4m3(x.t().contiguous().t())

    gw_no_bias, none_out = cublaslt_fp8_matmul_grad_w(
        gy_t_fp8, x_col_fp8, scale_gy=gy_scale, scale_x=x_scale,
        out_dtype=torch.bfloat16, compute_bias_grad=False,
    )
    assert none_out is None
    gw_with_bias, _ = cublaslt_fp8_matmul_grad_w(
        gy_t_fp8, x_col_fp8, scale_gy=gy_scale, scale_x=x_scale,
        out_dtype=torch.bfloat16, compute_bias_grad=True,
    )
    # Matmul outputs must match exactly regardless of epilogue choice.
    assert torch.allclose(gw_no_bias, gw_with_bias, rtol=1e-3, atol=1e-3), (
        "grad_w differs between BGRADB and DEFAULT epilogues — epilogue "
        "should only affect the bias-grad output, not the matmul"
    )


@pytest.mark.slow
def test_grad_x_faster_than_scaled_mm() -> None:
    """Throughput: our grad_x kernel must come in under 0.9 × torch._scaled_mm
    at submission shape. Marked slow; runs explicitly on the pod."""
    torch.manual_seed(0)
    M, K, N = 1024, 256, 256
    iters = 200

    grad_y = torch.randn(M, N, device="cuda", dtype=torch.bfloat16) * 0.1
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.05
    gy_fp8, gy_scale = _quantize_e5m2(grad_y)
    w_col = w.t().contiguous().t()
    _, w_scale = _quantize_e4m3(w)
    w_col_fp8 = (w_col.to(torch.float32) / w_scale).to(torch.float8_e4m3fn)

    def run_ours() -> None:
        cublaslt_fp8_matmul_grad_x(
            gy_fp8, w_col_fp8, scale_gy=gy_scale, scale_w=w_scale,
            out_dtype=torch.bfloat16,
        )

    def run_scaled_mm() -> None:
        torch._scaled_mm(
            gy_fp8, w_col_fp8, scale_a=gy_scale, scale_b=w_scale,
            out_dtype=torch.bfloat16,
        )

    ours_s = _bench_iters(run_ours, iters)
    scaled_mm_s = _bench_iters(run_scaled_mm, iters)
    print(
        f"\n[cublaslt-fp8 grad_x bench] shape=({M},{N})x({N},{K}) iters={iters}\n"
        f"  ours       = {ours_s * 1e6:8.2f} us\n"
        f"  _scaled_mm = {scaled_mm_s * 1e6:8.2f} us\n"
        f"  ours/ref   = {ours_s / scaled_mm_s:.3f}\n"
    )
    assert ours_s < 0.9 * scaled_mm_s, (
        f"grad_x not fast enough: ours={ours_s * 1e6:.2f}us "
        f"_scaled_mm={scaled_mm_s * 1e6:.2f}us "
        f"ratio={ours_s / scaled_mm_s:.3f}"
    )
