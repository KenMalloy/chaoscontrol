"""Phase S1 — simplex policy forward kernel.

Pinned by `docs/plans/2026-04-26-simplex-controller-design.md` (Forward
architecture) and the S1 scope in `docs/plans/2026-04-26-simplex-controller.md`.
The pure-NumPy reference here is the load-bearing piece — it mirrors the C++
forward step-by-step so any future change to the C++ kernel that drifts from
the spec lights this test up.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from chaoscontrol.kernels import _cpu_ssm_controller as _ext


def _make_weights(
    *,
    K_v: int = 16,
    K_e: int = 1,
    K_s: int = 4,
    H: int = 32,
    N: int = 16,
    seed: int = 0,
    scale: float = 1.0,
) -> "_ext.SimplexWeights":
    """Build a SimplexWeights with deterministic random values."""
    rng = np.random.default_rng(seed)
    w = _ext.SimplexWeights()
    w.K_v = K_v
    w.K_e = K_e
    w.K_s = K_s
    w.H = H
    w.N = N
    w.W_vp = (rng.standard_normal((K_v, H)).astype(np.float32) * scale).flatten().tolist()
    w.b_vp = rng.standard_normal((H,)).astype(np.float32).tolist()
    w.W_lh = rng.standard_normal((H,)).astype(np.float32).tolist()
    w.b_lh = float(rng.standard_normal())
    w.W_sb = rng.standard_normal((K_s,)).astype(np.float32).tolist()
    w.alpha = float(rng.standard_normal())
    w.temperature = 1.0
    # bucket_embed stays a placeholder; S1 stores it without consuming.
    w.bucket_embed = rng.standard_normal((8, 8)).astype(np.float32).flatten().tolist()
    return w


def _make_inputs(
    *,
    N: int = 16,
    K_v: int = 16,
    K_s: int = 4,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 1000)
    V = rng.standard_normal((N, K_v)).astype(np.float32)
    # E should be a symmetric-ish cosine matrix with diagonal=1 in the
    # production wire path, but the kernel doesn't enforce structure — any
    # finite (N, N) input is valid.
    E_raw = rng.standard_normal((N, N)).astype(np.float32)
    E = 0.5 * (E_raw + E_raw.T)
    np.fill_diagonal(E, 1.0)
    s = rng.standard_normal((K_s,)).astype(np.float32)
    return V, E, s


def _stable_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def _exact_gelu(x: np.ndarray) -> np.ndarray:
    # Same form as the C++ kernel: 0.5 * x * (1 + erf(x / sqrt(2))).
    # math.erf only takes scalars; vectorize so we don't drag scipy into the
    # test dependency surface (the venv on this worktree doesn't ship it).
    erf_v = np.vectorize(math.erf, otypes=[np.float32])
    return 0.5 * x * (1.0 + erf_v(x / math.sqrt(2.0)))


def _python_simplex_forward(
    weights: "_ext.SimplexWeights",
    V: np.ndarray,
    E: np.ndarray,
    s: np.ndarray,
) -> dict:
    """Pure-NumPy reference. Mirror the C++ kernel one layer at a time."""
    N = weights.N
    H = weights.H
    K_v = weights.K_v
    K_s = weights.K_s

    W_vp = np.asarray(weights.W_vp, dtype=np.float32).reshape(K_v, H)
    b_vp = np.asarray(weights.b_vp, dtype=np.float32)
    W_lh = np.asarray(weights.W_lh, dtype=np.float32)
    b_lh = float(weights.b_lh)
    W_sb = np.asarray(weights.W_sb, dtype=np.float32)
    alpha = float(weights.alpha)
    T = float(weights.temperature)

    # Layer 1.
    vertex_h = _exact_gelu(V @ W_vp + b_vp)  # [N, H]

    # Layer 2.
    dots = vertex_h @ vertex_h.T  # [N, N]
    attn_logits = dots / math.sqrt(H) + alpha * E  # [N, N]
    attn = _stable_softmax(attn_logits, axis=-1)  # [N, N]
    mixed_h = attn @ vertex_h + vertex_h  # post-residual

    # Layer 3.
    logits = mixed_h @ W_lh + b_lh  # [N]
    sb = float(s @ W_sb)
    p = _stable_softmax((logits + sb) / T, axis=-1)
    return {
        "logits": logits,
        "p": p,
        "vertex_h": vertex_h,
        "mixed_h": mixed_h,
        "attn": attn,
    }


@pytest.fixture(autouse=True)
def _require_extension():
    if _ext._C is None:
        pytest.skip("chaoscontrol._cpu_ssm_controller._C is not built")


def _call_simplex_forward(
    weights: "_ext.SimplexWeights",
    V: np.ndarray,
    E: np.ndarray,
    s: np.ndarray,
):
    return _ext.simplex_forward(
        weights,
        V.flatten().astype(np.float32).tolist(),
        E.flatten().astype(np.float32).tolist(),
        s.astype(np.float32).tolist(),
    )


def test_simplex_forward_output_shapes():
    weights = _make_weights()
    V, E, s = _make_inputs()
    out = _call_simplex_forward(weights, V, E, s)
    assert len(out.logits) == 16
    assert len(out.p) == 16
    p = np.asarray(out.p, dtype=np.float64)
    assert math.isclose(float(p.sum()), 1.0, abs_tol=1e-5)
    assert np.all(p >= 0.0)
    assert np.all(p <= 1.0)


def test_simplex_forward_softmax_stability():
    # 100x scaling on W_vp pushes Layer-1 outputs (and downstream attention
    # logits + final logits) into ranges that would overflow naive softmax.
    # The stable-softmax form (subtract max) must keep p in [0, 1] and
    # summing to 1 regardless.
    weights = _make_weights(scale=100.0)
    V, E, s = _make_inputs()
    out = _call_simplex_forward(weights, V, E, s)
    p = np.asarray(out.p, dtype=np.float64)
    assert np.all(np.isfinite(p)), "softmax output must be finite even with huge logits"
    assert np.all(p >= 0.0)
    assert np.all(p <= 1.0)
    assert math.isclose(float(p.sum()), 1.0, abs_tol=1e-5)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_simplex_forward_matches_python_reference(seed: int):
    # Tight tolerance is the at::matmul-path contract. On hosts where the
    # AMX BF16 dispatch is live, the three forward GEMMs round-trip through
    # bf16 and lose ~1e-2 of precision; the AMX-path parity test below
    # carries the looser tolerance. To keep this test honest as the strict
    # reference, skip it when the dispatch would route through AMX.
    if _ext.amx_bf16_kernel_available() and _ext.has_amx_bf16():
        pytest.skip(
            "AMX BF16 dispatch is live; tight 1e-4 tolerance is covered by "
            "the bf16-loose AMX-path parity test"
        )

    weights = _make_weights(seed=seed)
    V, E, s = _make_inputs(seed=seed)
    cpp = _call_simplex_forward(weights, V, E, s)
    ref = _python_simplex_forward(weights, V, E, s)

    cpp_logits = np.asarray(cpp.logits, dtype=np.float32)
    cpp_p = np.asarray(cpp.p, dtype=np.float32)
    cpp_vertex_h = np.asarray(cpp.vertex_h, dtype=np.float32).reshape(weights.N, weights.H)
    cpp_mixed_h = np.asarray(cpp.mixed_h, dtype=np.float32).reshape(weights.N, weights.H)
    cpp_attn = np.asarray(cpp.attn, dtype=np.float32).reshape(weights.N, weights.N)

    np.testing.assert_allclose(cpp_logits, ref["logits"], atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(cpp_p, ref["p"], atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(cpp_vertex_h, ref["vertex_h"], atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(cpp_mixed_h, ref["mixed_h"], atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(cpp_attn, ref["attn"], atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_simplex_forward_matches_python_reference_amx_bf16_path(seed: int):
    """Hardware-gated parity for the AMX BF16 dispatch path.

    The forward GEMMs (V@W_vp, vh@vh.T, mh@W_lh) bf16-cast inputs before
    handing them to the tiled AMX kernel. The fp32 NumPy reference does
    the same multiply at full precision, so the worst-case drift is the
    bf16 round-trip plus the chained accumulation across 16- to 32-wide
    K dimensions. atol/rtol of 1e-2 is the bf16 ULP-equivalent slack
    that mirrors `tests/test_amx_matmul.py`'s tolerance for the same
    kernel — both target Sapphire Rapids.

    Skipped on arm64 / non-AMX builds where the dispatch falls through
    to at::matmul; the strict 1e-4 test above covers that path.
    """
    if not (_ext.amx_bf16_kernel_available() and _ext.has_amx_bf16()):
        pytest.skip("AMX BF16 hardware/OS state and compiled kernel are required")

    weights = _make_weights(seed=seed)
    V, E, s = _make_inputs(seed=seed)
    cpp = _call_simplex_forward(weights, V, E, s)
    ref = _python_simplex_forward(weights, V, E, s)

    cpp_logits = np.asarray(cpp.logits, dtype=np.float32)
    cpp_p = np.asarray(cpp.p, dtype=np.float32)
    cpp_vertex_h = np.asarray(cpp.vertex_h, dtype=np.float32).reshape(weights.N, weights.H)
    cpp_mixed_h = np.asarray(cpp.mixed_h, dtype=np.float32).reshape(weights.N, weights.H)
    cpp_attn = np.asarray(cpp.attn, dtype=np.float32).reshape(weights.N, weights.N)

    np.testing.assert_allclose(cpp_logits, ref["logits"], atol=1e-2, rtol=1e-2)
    np.testing.assert_allclose(cpp_p, ref["p"], atol=1e-2, rtol=1e-2)
    np.testing.assert_allclose(cpp_vertex_h, ref["vertex_h"], atol=1e-2, rtol=1e-2)
    np.testing.assert_allclose(cpp_mixed_h, ref["mixed_h"], atol=1e-2, rtol=1e-2)
    np.testing.assert_allclose(cpp_attn, ref["attn"], atol=1e-2, rtol=1e-2)


def test_simplex_forward_alpha_zero_recovers_per_vertex_softmax():
    # Sanity check: when alpha=0 and biases are zero AND vertex_h dot products
    # are dominated by self-attention (orthogonal vertex_h rows + self-dots
    # large vs cross-dots), Layer 2 attention concentrates on the diagonal,
    # so mixed_h ≈ ~2 * vertex_h (residual + ~self-attended copy). The final
    # logits are then ~2 * vertex_h @ W_lh; softmax preserves ranking with
    # any positive scaling, so argmax(p) == argmax(per-vertex linear).
    weights = _make_weights(K_v=16, H=32, N=16, seed=0)
    weights.alpha = 0.0
    weights.b_vp = [0.0] * weights.H
    weights.b_lh = 0.0
    weights.W_sb = [0.0] * weights.K_s

    # Force vertex_h rows to have large self-dot vs cross-dot: pick V to
    # make GeLU(V @ W_vp) approximately a scaled identity-like matrix.
    # Easiest: set W_vp = identity-aligned, V = scaled identity.
    K_v = weights.K_v
    H = weights.H
    N = weights.N
    assert K_v == 16 and H == 32 and N == 16
    W_vp = np.zeros((K_v, H), dtype=np.float32)
    # Place a strong positive value on the i-th column so vertex_h_pre[i, i] is
    # large positive (post-GeLU stays large) and off-diagonal entries are 0
    # (post-GeLU = 0 since GeLU(0)=0). This makes vertex_h rows pairwise near-
    # orthogonal with strong self-norms.
    for i in range(K_v):
        W_vp[i, i] = 5.0
    weights.W_vp = W_vp.flatten().tolist()

    V = np.eye(N, K_v, dtype=np.float32)  # one-hot rows
    E = np.eye(N, dtype=np.float32)
    s = np.zeros((weights.K_s,), dtype=np.float32)

    cpp = _call_simplex_forward(weights, V, E, s)
    cpp_p = np.asarray(cpp.p, dtype=np.float32)
    cpp_attn = np.asarray(cpp.attn, dtype=np.float32).reshape(N, N)

    # Attention should concentrate on the diagonal — each row's self-weight
    # dominates. Loose tolerance because the cross-row dot products are not
    # exactly zero (GeLU isn't quite linear at our scale) but they are tiny.
    diag_mass = np.diag(cpp_attn)
    assert np.all(diag_mass > 0.5), (
        "with self-dominated dots, attn should concentrate on diagonal; got "
        f"min diag mass {diag_mass.min()}"
    )

    # Argmax(p) should match argmax of the per-vertex linear logits computed
    # naively from vertex_h alone (no mixing).
    ref = _python_simplex_forward(weights, V, E, s)
    vertex_h = ref["vertex_h"]
    W_lh = np.asarray(weights.W_lh, dtype=np.float32)
    per_vertex_logits = vertex_h @ W_lh
    per_vertex_p = _stable_softmax(per_vertex_logits)
    assert int(np.argmax(cpp_p)) == int(np.argmax(per_vertex_p)), (
        "argmax(p) must match per-vertex softmax argmax under the loose-sanity setup"
    )


def test_simplex_forward_savedstate_shapes_for_backward():
    # S2 (REINFORCE backward) reads vertex_h, mixed_h, attn from the forward
    # output. Pin shapes here so a refactor that drops one of them lights up
    # this test before S2's gradient math diverges silently.
    weights = _make_weights()
    V, E, s = _make_inputs()
    out = _call_simplex_forward(weights, V, E, s)

    N, H = weights.N, weights.H
    assert len(out.vertex_h) == N * H
    assert len(out.mixed_h) == N * H
    assert len(out.attn) == N * N

    # All saved tensors must be finite — NaN-poisoning at decision time would
    # corrupt downstream replay.
    for name, buf in (
        ("vertex_h", out.vertex_h),
        ("mixed_h", out.mixed_h),
        ("attn", out.attn),
        ("logits", out.logits),
        ("p", out.p),
    ):
        arr = np.asarray(buf, dtype=np.float32)
        assert np.all(np.isfinite(arr)), f"{name} contains non-finite values"
