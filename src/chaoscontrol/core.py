"""Core SSM components: ChaosSSMCore (diag/paired/full), RMSNorm, FeedForward, criticality_loss."""
from __future__ import annotations

import math
import os
import warnings
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _diag_recurrence_inner(decay: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
    """Sequential linear recurrence: state_t = decay_t * state_{t-1} + update_t.

    Args:
        decay: (batch, seq, dim)
        update: (batch, seq, dim)

    Returns:
        states: (batch, seq, dim) — all intermediate states
    """
    batch, seq, dim = decay.shape
    state = torch.zeros(batch, dim, dtype=decay.dtype, device=decay.device)
    outputs = []
    for t in range(seq):
        state = decay[:, t] * state + update[:, t]
        outputs.append(state)
    return torch.stack(outputs, dim=1)


_DEFAULT_CHUNK_SIZE = 32


def _should_use_zero_initial_state_fast_path(initial_state: torch.Tensor) -> bool:
    if initial_state.requires_grad:
        return False
    if initial_state.is_cuda and torch.cuda.is_current_stream_capturing():
        return False
    return not torch.any(initial_state)


def _diag_recurrence_chunked(
    decay: torch.Tensor,
    update: torch.Tensor,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> torch.Tensor:
    """Chunked vectorized diag scan.

    Implements the same first-order linear recurrence as _diag_recurrence_inner,
    but uses torch.cumprod + torch.cumsum within chunks and a short Python loop
    across chunks. The chunking bounds the cumprod range so we stay safely
    within float64 dynamic range regardless of decay magnitude.

    Internals run in float64 for numerical stability; the result is cast back
    to the input dtype. The chunk-loop Python overhead is O(T/chunk_size)
    iterations instead of O(T), and each iteration is a parallel kernel on
    GPU, so this is typically 5-20x faster than the serial loop on long
    sequences while producing identical output to float32 precision.

    Args:
        decay: (batch, seq, dim) — all values in (0, 1]
        update: (batch, seq, dim)
        chunk_size: chunk length for local exact scan. Smaller = more
                   numerically robust but more Python overhead. Default 32.

    Returns:
        states: (batch, seq, dim)
    """
    orig_dtype = decay.dtype
    B, T, D = decay.shape

    # Pad to multiple of chunk_size with identity (decay=1, update=0)
    pad_len = (chunk_size - T % chunk_size) % chunk_size
    if pad_len > 0:
        decay_pad = torch.cat(
            [decay, torch.ones(B, pad_len, D, dtype=decay.dtype, device=decay.device)],
            dim=1,
        )
        update_pad = torch.cat(
            [update, torch.zeros(B, pad_len, D, dtype=update.dtype, device=update.device)],
            dim=1,
        )
    else:
        decay_pad = decay
        update_pad = update

    T_padded = decay_pad.shape[1]
    num_chunks = T_padded // chunk_size

    # (B, num_chunks, chunk_size, D) in float64
    d = decay_pad.view(B, num_chunks, chunk_size, D).to(torch.float64)
    u = update_pad.view(B, num_chunks, chunk_size, D).to(torch.float64)

    # Within each chunk, compute local states assuming chunk_start_state = 0:
    # D_partial[t] = prod_{j<=t} decay[j]       (running cumprod)
    # local[t] = D_partial[t] * cumsum(update / D_partial)[t]
    D_partial = torch.cumprod(d, dim=2)
    weighted = u / D_partial
    cum_w = torch.cumsum(weighted, dim=2)
    local_states = D_partial * cum_w  # (B, num_chunks, K, D)

    # Combine chunks serially: propagate end-of-chunk state as carry into next chunk
    #   state[chunk i, t] = D_partial[i, t] * carry_i + local_states[i, t]
    #   carry_{i+1}      = D_partial[i, -1] * carry_i + local_states[i, -1]
    carry = torch.zeros(B, D, dtype=torch.float64, device=decay.device)
    outputs = []
    for i in range(num_chunks):
        partial_decay_i = D_partial[:, i]  # (B, K, D)
        local_i = local_states[:, i]        # (B, K, D)
        chunk_out = partial_decay_i * carry.unsqueeze(1) + local_i
        outputs.append(chunk_out)
        carry = partial_decay_i[:, -1] * carry + local_i[:, -1]

    result = torch.cat(outputs, dim=1)[:, :T]  # trim padding
    return result.to(orig_dtype)


_diag_recurrence_impl = None
_diag_recurrence_backend = "python"
_diag_recurrence_note = "fallback"


def _resolve_diag_recurrence_impl():
    """Resolve the fastest available diag recurrence backend.

    Backends (selectable via CHAOSCONTROL_DIAG_SCAN_BACKEND env var):
        "python"   — sequential Python loop (_diag_recurrence_inner)
        "compile"  — torch.compile'd Python loop (default, fast on CUDA)
        "chunked"  — chunked vectorized scan (cumprod+cumsum, Exp 18 Test 1)
        "ssm_scan" — hand-written CUDA kernels for forward AND backward
                     (per-lane serial scan, fp32 accumulator). Phase 1B-4
                     / Phase 2 follow-up to the torch.compile regression.

    We avoid compiling at import time so a mismatched Inductor/CUDA/toolchain
    stack does not make the whole package fail before argument parsing.
    ``ssm_scan`` falls through to ``chunked`` if the extension is not
    importable (dev mac, non-CUDA env, partial pod setup).
    """
    global _diag_recurrence_impl, _diag_recurrence_backend, _diag_recurrence_note
    if _diag_recurrence_impl is not None:
        return _diag_recurrence_impl

    requested = os.environ.get("CHAOSCONTROL_DIAG_SCAN_BACKEND", "").strip().lower()

    # Legacy flag: CHAOSCONTROL_DISABLE_TORCH_COMPILE=1 forces python backend
    if os.environ.get("CHAOSCONTROL_DISABLE_TORCH_COMPILE", "").strip() == "1":
        requested = "python"

    if requested == "python":
        _diag_recurrence_impl = _diag_recurrence_inner
        _diag_recurrence_backend = "python"
        _diag_recurrence_note = "explicit python backend"
        return _diag_recurrence_impl

    if requested == "chunked":
        _diag_recurrence_impl = _diag_recurrence_chunked
        _diag_recurrence_backend = "chunked"
        _diag_recurrence_note = f"vectorized cumprod+cumsum, chunk_size={_DEFAULT_CHUNK_SIZE}"
        return _diag_recurrence_impl

    if requested == "ssm_scan":
        # Two levels of availability: the Python package always imports
        # (it's pure Python), but the C extension ``_C`` is only present
        # when ``pip install -e .`` built it. Fall back to chunked on
        # either missing — the package fully imports on dev macs where
        # the extension cannot compile.
        _ssm_scan_fn = None
        _ssm_scan_err: Exception | None = None
        try:
            import chaoscontrol.kernels._ssm_scan as _ssm_ext

            if getattr(_ssm_ext, "_C", None) is None:
                _ssm_scan_err = ImportError(
                    "chaoscontrol.kernels._ssm_scan._C not built"
                )
            else:
                # Use getattr so a deleted attribute produces a clean
                # ImportError-shaped fallback (see monkeypatch test).
                _ssm_scan_fn = getattr(_ssm_ext, "ssm_scan", None)
                if _ssm_scan_fn is None:
                    _ssm_scan_err = ImportError(
                        "chaoscontrol.kernels._ssm_scan.ssm_scan unavailable"
                    )
        except ImportError as exc:
            _ssm_scan_err = exc

        if _ssm_scan_fn is not None:
            _diag_recurrence_impl = _ssm_scan_fn
            _diag_recurrence_backend = "ssm_scan"
            _diag_recurrence_note = (
                "hand-written CUDA kernels for forward and backward "
                "(per-lane serial scan with fp32 accumulator; fp32 state "
                "snapshot saved for backward to avoid bf16-quantized grads). "
                "At submission shape B=1024/T=512/D=256 bf16 on 1xH100: "
                "2.20 ms fwd+bwd, ~35x vs chunked (the honest baseline; "
                "`compile` baseline is pessimistic because compile's own "
                "backward is broken). See benchmarks/bench_ssm_scan.py "
                "for the full table."
            )
        else:
            _diag_recurrence_impl = _diag_recurrence_chunked
            _diag_recurrence_backend = "chunked"
            _diag_recurrence_note = (
                f"ssm_scan requested but extension not importable "
                f"({type(_ssm_scan_err).__name__}: {_ssm_scan_err}); "
                f"falling back to chunked backend"
            )
            warnings.warn(
                "CHAOSCONTROL_DIAG_SCAN_BACKEND=ssm_scan requested but "
                f"the extension is not importable ({_ssm_scan_err}). "
                "Falling back to the chunked backend.",
                RuntimeWarning,
                stacklevel=2,
            )
        return _diag_recurrence_impl

    # Default: try torch.compile, fall back to Python
    try:
        _diag_recurrence_impl = torch.compile(_diag_recurrence_inner, dynamic=False)
        _diag_recurrence_backend = "compile"
        _diag_recurrence_note = "torch.compile(dynamic=False)"
    except Exception as exc:  # pragma: no cover - only triggers on mismatched stacks
        _diag_recurrence_impl = _diag_recurrence_inner
        _diag_recurrence_backend = "python"
        _diag_recurrence_note = f"compile unavailable: {exc.__class__.__name__}: {exc}"
        warnings.warn(
            "torch.compile unavailable for diag recurrence; falling back to the Python loop. "
            f"Reason: {_diag_recurrence_note}",
            RuntimeWarning,
            stacklevel=2,
        )
    return _diag_recurrence_impl


def get_diag_recurrence_backend() -> dict[str, str]:
    """Report which diag recurrence backend is active."""
    _resolve_diag_recurrence_impl()
    return {
        "backend": _diag_recurrence_backend,
        "note": _diag_recurrence_note,
    }


def verify_diag_recurrence(device: torch.device | None = None) -> dict[str, str]:
    """Resolve backend AND execute a tiny forward pass to confirm it works.

    Returns the backend info after execution, so the reported backend
    reflects reality (e.g. "python" if compile fell back at runtime).
    """
    dev = device if device is not None else torch.device("cpu")
    probe_decay = torch.zeros(1, 4, 8, device=dev)
    probe_update = torch.ones(1, 4, 8, device=dev)
    _diag_recurrence(probe_decay, probe_update)
    return get_diag_recurrence_backend()


def _diag_recurrence(decay: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
    # When an outer ``torch.compile`` is tracing this call (e.g. via
    # ``train_ssm._compiled_step_fn``), route through the uncompiled
    # Python loop so the outer gets a single unified graph. Skipping the
    # inner ``torch.compile`` wrapper here is what un-parks
    # ``compile_full_path=True``: the nested-compile pattern produced a
    # −62.77% regression in the 1A-4 microbench. Dynamo evaluates
    # ``is_compiling()`` at trace time, so this branch resolves to a
    # single-path graph — no runtime overhead for the eager caller either
    # (constant-folded to False by the CPython interpreter).
    if torch.compiler.is_compiling():
        return _diag_recurrence_inner(decay, update)
    global _diag_recurrence_impl, _diag_recurrence_backend, _diag_recurrence_note
    impl = _resolve_diag_recurrence_impl()
    try:
        return impl(decay, update)
    except Exception as exc:  # pragma: no cover - only triggers on mismatched stacks
        if impl is _diag_recurrence_inner:
            raise
        # Only the `compile` backend should be demoted on a runtime
        # failure — that's the one known to fail mid-run on mismatched
        # Inductor/CUDA stacks. Other backends (ssm_scan, chunked)
        # raise meaningful exceptions that callers need to see; do NOT
        # silently rewrite the global backend cache to "python" and
        # discard the error. Fix #2: `ssm_scan` on a CPU tensor already
        # falls back gracefully inside the wrapper, so this path is
        # never entered for that case — but guarding the cache demote
        # on backend == "compile" is belt-and-suspenders.
        if _diag_recurrence_backend != "compile":
            raise
        _diag_recurrence_impl = _diag_recurrence_inner
        _diag_recurrence_backend = "python"
        _diag_recurrence_note = f"compile runtime failure: {exc.__class__.__name__}: {exc}"
        warnings.warn(
            "torch.compile failed during diag recurrence execution; falling back to the Python loop. "
            f"Reason: {_diag_recurrence_note}",
            RuntimeWarning,
            stacklevel=2,
        )
        return _diag_recurrence_inner(decay, update)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = F.rms_norm(x.float(), (x.size(-1),), eps=self.eps)
        return normed.to(x.dtype) * self.weight


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int) -> None:
        super().__init__()
        hidden = dim * mult
        self.fc = nn.Linear(dim, hidden, bias=False)
        self.proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.silu(self.fc(x)))


class ChaosSSMCore(nn.Module):
    """SSM recurrence with three A parameterizations: diag, paired, full."""

    def __init__(
        self,
        dim: int,
        *,
        a_mode: str = "diag",
        a_full_rank: int = 8,
        a_full_gamma: float = 0.05,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.a_mode = a_mode
        self.a_full_rank = a_full_rank

        # Shared projections across all modes
        self.in_proj = nn.Linear(dim, dim, bias=False)
        self.select_proj = nn.Linear(dim, dim, bias=False)
        self.gate_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        if a_mode == "diag":
            self.delta_proj = nn.Linear(dim, dim, bias=False)
            self.log_a = nn.Parameter(torch.zeros((dim,), dtype=torch.float32))

        elif a_mode == "paired":
            assert dim % 2 == 0, f"paired mode requires even dim, got {dim}"
            n_pairs = dim // 2
            self.delta_proj = nn.Linear(dim, dim, bias=False)
            self.log_r = nn.Parameter(torch.zeros((n_pairs,), dtype=torch.float32))
            self.theta = nn.Parameter(
                torch.linspace(0.0, math.pi, n_pairs, dtype=torch.float32)
            )

        elif a_mode == "full":
            # delta_proj outputs scalar step size
            self.delta_proj = nn.Linear(dim, 1, bias=False)
            # Skew-symmetric S: upper triangle params
            n_skew = dim * (dim - 1) // 2
            self.skew_params = nn.Parameter(torch.zeros((n_skew,), dtype=torch.float32))
            # Damping
            self.log_gamma = nn.Parameter(
                torch.full((), math.log(a_full_gamma), dtype=torch.float32)
            )
            # Non-normal low-rank part UV^T
            rank = min(a_full_rank, dim)
            self.U = nn.Parameter(torch.randn(dim, rank) * 0.01)
            self.V = nn.Parameter(torch.randn(dim, rank) * 0.01)

        else:
            raise ValueError(f"unsupported a_mode: {a_mode}")

    def _build_skew_symmetric(self) -> torch.Tensor:
        """Build a dim x dim skew-symmetric matrix from upper-triangle params."""
        S = torch.zeros(self.dim, self.dim, dtype=self.skew_params.dtype,
                        device=self.skew_params.device)
        idx = torch.triu_indices(self.dim, self.dim, offset=1)
        S[idx[0], idx[1]] = self.skew_params
        S = S - S.T
        return S

    def _get_A_full(self) -> torch.Tensor:
        """Build the continuous-time A matrix for full mode: S - gamma*I + UV^T."""
        S = self._build_skew_symmetric()
        gamma = torch.exp(self.log_gamma)
        A_c = S - gamma * torch.eye(self.dim, device=S.device, dtype=S.dtype) + self.U @ self.V.T
        return A_c

    def _diag_terms(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute per-timestep decay, update, and output gate terms for diag mode.

        This helper is used by the vectorized scan backend for the common
        `rich_b is None` path. The recurrence is:

            state_t = decay_t * state_{t-1} + update_t

        with `state_{-1} = 0`.
        """
        a_base = torch.sigmoid(self.log_a).to(dtype=x.dtype)[None, None, :]
        delta = F.softplus(self.delta_proj(x)).clamp_min(1e-4)
        decay = torch.exp(-delta * a_base)
        select = torch.sigmoid(self.select_proj(x))
        candidate = torch.tanh(self.in_proj(x))
        update = select * candidate
        gate = torch.sigmoid(self.gate_proj(x))
        return decay, update, gate

    def _forward_diag_scan(self, x: torch.Tensor) -> torch.Tensor:
        """Compiled sequential diag recurrence.

        Uses the same elementwise recurrence as the sequential loop
        but processes decay/update/gate in a single batched projection
        pass, then runs the state update loop on pre-computed terms.
        torch.compile fuses the per-step kernels on CUDA.
        """
        decay, update, gate = self._diag_terms(x)
        states = _diag_recurrence(decay, update)
        out = gate * states
        return self.out_proj(out)

    def step(
        self,
        inp: torch.Tensor,
        state: torch.Tensor,
        *,
        rich_b: Any = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-token recurrence step.

        Args:
            inp: (batch, dim) — single token embedding (already projected if needed)
            state: (batch, dim) — previous recurrence state

        Returns:
            (output, new_state) — both (batch, dim)
        """
        if self.a_mode == "diag":
            a_base = torch.sigmoid(self.log_a).to(dtype=inp.dtype)[None, :]
            delta = F.softplus(self.delta_proj(inp)).clamp_min(1e-4)
            decay = torch.exp(-delta * a_base)
            if rich_b is not None:
                update = rich_b(inp, state)
            else:
                select = torch.sigmoid(self.select_proj(inp))
                candidate = torch.tanh(self.in_proj(inp))
                update = select * candidate
            new_state = decay * state + update
            out = torch.sigmoid(self.gate_proj(inp)) * new_state
            return self.out_proj(out), new_state

        elif self.a_mode == "paired":
            batch = inp.shape[0]
            dim = self.dim
            n_pairs = dim // 2
            delta = F.softplus(self.delta_proj(inp)).clamp_min(1e-4)
            r = torch.exp(-F.softplus(self.log_r)).to(dtype=inp.dtype)
            cos_t = torch.cos(self.theta).to(dtype=inp.dtype)
            sin_t = torch.sin(self.theta).to(dtype=inp.dtype)
            delta_pairs = (delta[:, 0::2] + delta[:, 1::2]) * 0.5
            effective_r = torch.exp(-delta_pairs * (1.0 - r[None, :]))
            s = state.view(batch, n_pairs, 2)
            s0 = s[:, :, 0]
            s1 = s[:, :, 1]
            new_s0 = effective_r * (cos_t[None, :] * s0 - sin_t[None, :] * s1)
            new_s1 = effective_r * (sin_t[None, :] * s0 + cos_t[None, :] * s1)
            decayed = torch.stack([new_s0, new_s1], dim=-1).view(batch, dim)
            if rich_b is not None:
                update = rich_b(inp, state)
            else:
                select = torch.sigmoid(self.select_proj(inp))
                candidate = torch.tanh(self.in_proj(inp))
                update = select * candidate
            new_state = decayed + update
            out = torch.sigmoid(self.gate_proj(inp)) * new_state
            return self.out_proj(out), new_state

        elif self.a_mode == "full":
            A_c = self._get_A_full().to(dtype=inp.dtype)
            delta = F.softplus(self.delta_proj(inp)).clamp(1e-4, 2.0)
            d = delta.mean()
            A_d = torch.matrix_exp(d * A_c)
            proposed = state @ A_d.T
            if rich_b is not None:
                update = rich_b(inp, state)
            else:
                select = torch.sigmoid(self.select_proj(inp))
                candidate = torch.tanh(self.in_proj(inp))
                update = select * candidate
            new_state = proposed + update
            out = torch.sigmoid(self.gate_proj(inp)) * new_state
            return self.out_proj(out), new_state

        else:
            raise ValueError(f"unsupported a_mode: {self.a_mode}")

    def forward(
        self,
        x: torch.Tensor,
        *,
        rich_b: Any = None,
        return_jacobian_stats: bool = False,
        initial_state: torch.Tensor | None = None,
        return_final_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Run the SSM recurrence over a sequence.

        Args:
            x: (batch, seq, dim) input.
            rich_b: optional RichB module that computes per-step update.
            return_jacobian_stats: if True, append a per-seq Jacobian stats dict
                to the return (only meaningful in ``full`` mode).
            initial_state: (batch, dim) — optional non-zero seed for the
                recurrence. When ``None`` (default), state is zero-initialized
                exactly like before, so the default call is bit-identical to
                prior behavior.
            return_final_state: if True, append ``final_state`` (shape
                ``(batch, dim)``) to the return. Caller must destructure.

        Returns:
            Depending on the return_* kwargs:
                y                                     (both False)
                (y, stats)                            (stats=True)
                (y, final_state)                      (final_state=True)
                (y, stats, final_state)               (both True)
        """
        batch, seq, dim = x.shape
        if initial_state is not None:
            # Cast to match x's dtype/device so the recurrence stays well-typed.
            state = initial_state.to(dtype=x.dtype, device=x.device)
            # LOAD-BEARING: this zero-detection is what keeps the rtol=0, atol=0
            # bit-identity invariant in test_zeros_initial_states_match_default.
            # Without it, explicit zeros would take the sequential Python-loop
            # branch below while None takes the parallel _diag_recurrence scan,
            # and the two paths accumulate in different orders — the outputs
            # match to ~1e-7 but NOT bit-exact. Downstream Axis 2 persistence
            # modes rely on "pass zeros == pass nothing" as a clean identity.
            # Do not remove without also collapsing the fast/slow paths to one.
            # Cost: O(B*D), negligible vs. (B, T, D) forward; real persisted
            # non-zero states skip this branch via the short-circuit.
            #
            # `requires_grad` gate: a trainable `h0` (Task 7 / Axis 2) may
            # legitimately start at zero but needs the autograd edge through
            # the slow path so gradient accumulates on the source Parameter.
            # Training's zero-init buffer is `x.new_zeros` with grad disabled,
            # so the fast path still fires there. See
            # test_trainable_h0_receives_gradient for the regression pin.
            if _should_use_zero_initial_state_fast_path(initial_state):
                initial_state = None
        else:
            state = x.new_zeros((batch, dim))
        outputs = []

        def _bundle(y: torch.Tensor, stats: dict | None, final_state: torch.Tensor):
            """Assemble the return tuple per the return_* kwargs."""
            if return_jacobian_stats and return_final_state:
                return y, stats, final_state
            if return_jacobian_stats:
                return y, stats
            if return_final_state:
                return y, final_state
            return y

        if self.a_mode == "diag":
            # Fast path: parallel scan. Requires zero-initial-state AND no rich_b.
            # When an initial state is supplied we fall back to the sequential
            # loop so the seed threads through correctly — the chunked scan's
            # carry is hardcoded to zeros and plumbing a non-zero carry would
            # touch the scan kernels unnecessarily. This fallback is tested by
            # test_final_state_equals_chunked_sequential.
            if rich_b is None and initial_state is None:
                decay, update, gate = self._diag_terms(x)
                states = _diag_recurrence(decay, update)
                out = gate * states
                y = self.out_proj(out)
                stats = (
                    {"lambda_max": torch.tensor(0.0), "sv_log_var": torch.tensor(0.0)}
                    if return_jacobian_stats else None
                )
                final_state = states[:, -1] if return_final_state else None
                return _bundle(y, stats, final_state)
            a_base = torch.sigmoid(self.log_a).to(dtype=x.dtype)[None, :]
            for idx in range(seq):
                inp = x[:, idx, :]
                delta = F.softplus(self.delta_proj(inp)).clamp_min(1e-4)
                decay = torch.exp(-delta * a_base)
                if rich_b is not None:
                    update = rich_b(inp, state)
                else:
                    select = torch.sigmoid(self.select_proj(inp))
                    candidate = torch.tanh(self.in_proj(inp))
                    update = select * candidate
                state = decay * state + update
                out = torch.sigmoid(self.gate_proj(inp)) * state
                outputs.append(self.out_proj(out))
            y = torch.stack(outputs, dim=1)
            stats = (
                {"lambda_max": torch.tensor(0.0), "sv_log_var": torch.tensor(0.0)}
                if return_jacobian_stats else None
            )
            return _bundle(y, stats, state)

        elif self.a_mode == "paired":
            n_pairs = dim // 2
            for idx in range(seq):
                inp = x[:, idx, :]
                delta = F.softplus(self.delta_proj(inp)).clamp_min(1e-4)
                # Build per-pair rotation+decay
                r = torch.exp(-F.softplus(self.log_r)).to(dtype=x.dtype)  # (n_pairs,)
                cos_t = torch.cos(self.theta).to(dtype=x.dtype)  # (n_pairs,)
                sin_t = torch.sin(self.theta).to(dtype=x.dtype)  # (n_pairs,)
                # Modulate decay by delta (average over pair dims)
                delta_pairs = (delta[:, 0::2] + delta[:, 1::2]) * 0.5  # (batch, n_pairs)
                effective_r = torch.exp(-delta_pairs * (1.0 - r[None, :]))  # (batch, n_pairs)
                # Apply rotation: state is reshaped to (batch, n_pairs, 2)
                s = state.view(batch, n_pairs, 2)
                s0 = s[:, :, 0]  # (batch, n_pairs)
                s1 = s[:, :, 1]  # (batch, n_pairs)
                new_s0 = effective_r * (cos_t[None, :] * s0 - sin_t[None, :] * s1)
                new_s1 = effective_r * (sin_t[None, :] * s0 + cos_t[None, :] * s1)
                decayed = torch.stack([new_s0, new_s1], dim=-1).view(batch, dim)
                if rich_b is not None:
                    update = rich_b(inp, state)
                else:
                    select = torch.sigmoid(self.select_proj(inp))
                    candidate = torch.tanh(self.in_proj(inp))
                    update = select * candidate
                state = decayed + update
                out = torch.sigmoid(self.gate_proj(inp)) * state
                outputs.append(self.out_proj(out))
            y = torch.stack(outputs, dim=1)
            stats = (
                {"lambda_max": torch.tensor(0.0), "sv_log_var": torch.tensor(0.0)}
                if return_jacobian_stats else None
            )
            return _bundle(y, stats, state)

        elif self.a_mode == "full":
            A_c = self._get_A_full().to(dtype=x.dtype)
            sv_log_maxes = []
            sv_log_vars = []
            for idx in range(seq):
                inp = x[:, idx, :]
                delta = F.softplus(self.delta_proj(inp)).clamp(1e-4, 2.0)
                d = delta.mean()
                A_d = torch.matrix_exp(d * A_c)
                proposed = state @ A_d.T
                if rich_b is not None:
                    update = rich_b(inp, state)
                else:
                    select = torch.sigmoid(self.select_proj(inp))
                    candidate = torch.tanh(self.in_proj(inp))
                    update = select * candidate
                state = proposed + update
                out = torch.sigmoid(self.gate_proj(inp)) * state
                outputs.append(self.out_proj(out))
                if return_jacobian_stats:
                    svs = torch.linalg.svdvals(A_d.detach())
                    log_svs = torch.log(svs.clamp_min(1e-8))
                    sv_log_maxes.append(log_svs[0])
                    sv_log_vars.append(log_svs.var())
            y = torch.stack(outputs, dim=1)
            stats = None
            if return_jacobian_stats:
                stats = {
                    "lambda_max": torch.stack(sv_log_maxes).mean(),
                    "sv_log_var": torch.stack(sv_log_vars).mean(),
                }
            return _bundle(y, stats, state)

        else:
            raise ValueError(f"unsupported a_mode: {self.a_mode}")


def criticality_loss(
    jacobian_stats: dict[str, torch.Tensor],
    *,
    alpha: float = 0.01,
    beta: float = 0.001,
    target_log_sv: float = -0.13,  # log(0.88) ~ -0.13, slightly subcritical
) -> torch.Tensor:
    """Penalize drift from near-critical regime.

    Targets slightly subcritical dynamics (~0.88 coupling) rather than
    exactly critical (1.0). lambda_max is the mean top log singular value
    of the per-step Jacobian; we penalize its distance from the target.
    L_crit = alpha * (lambda_max - target)^2 + beta * Var(log singular_values)
    """
    lam = jacobian_stats["lambda_max"]
    sv_var = jacobian_stats["sv_log_var"]
    return alpha * (lam - target_log_sv).pow(2) + beta * sv_var
