"""SemanticOptimizer: Muon with SSM-channel-coupled momentum time constants.

Intuition: the optimizer's temporal time constant should match each SSM
channel's forward-pass time constant. Channels with slow recurrence
(``sigmoid(log_a)`` near 1) integrate gradients over long horizons;
channels with fast recurrence (near 0) should forget gradients quickly.
Per-channel momentum β coupled to the current channel decay implements
this match.

Mechanics: we fork Muon — Newton-Schulz orthogonalized momentum on matrix
params, AdamW fallback on non-matrix — and replace the scalar momentum
decay with a channel-broadcast tensor on parameters the user flags as
channel-coupled. The NS readout and its ``max(1, rows/cols)**0.5``
rectangular-shape scaling stay unchanged: spatial semantics (all singular
directions move equally, rescaled once per matrix) composes cleanly with
temporal semantics (per-channel β). The *spatial* geometry remains
per-matrix; only the *temporal* geometry becomes per-channel.

A→A self-coupling is intentionally not supported in v1. A is 1D
(``log_a`` has shape ``(dim,)``) so it falls through to the AdamW
fallback, which has no momentum buffer to broadcast β across. A proper
self-coupling path would need a distinct vector-momentum update rule
(per-element β, no NS) and is deferred to v2 — the feedback loop
"A_i → β_i → ΔA_i → A_i" wants explicit damping before it's safe to
ship. Listing the A-parameter in ``channel_map`` does nothing in v1 by
design, and the constructor raises if you try it.

Precision note: when channel-coupling is active, matrix momentum buffers
are kept in float32 regardless of parameter dtype. bf16 representation
of β values near 0.999 (slow channels) loses ~1e-3 of precision per
multiply, and the compounding error across thousands of steps materially
shifts the effective time constant. fp32 buffers cost 2× memory on the
buffer, negligible in total VRAM, and preserve the β semantics the user
configures. When a_param_name is None (reduces to Muon), buffers match
parameter dtype for bit-for-bit Muon parity.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable

import torch
from torch import Tensor

from chaoscontrol.optim.muon import newton_schulz_orthogonalize


def default_beta_from_log_a(
    log_a: Tensor,
    *,
    beta_min: float,
    beta_max: float,
) -> Tensor:
    """Map the SSM's log-decay parameter to a per-channel momentum β.

    The SSM's base decay is ``sigmoid(log_a) ∈ (0, 1)``, values near 1 for
    slow channels. β is linearly interpolated in ``[beta_min, beta_max]``
    with the same ordering: slow channels get long optimizer memory, fast
    channels get short memory.

    Output is detached and upcast to float32 so the optimizer never
    back-props through the β map into A.
    """
    a_base = torch.sigmoid(log_a.detach().float())
    return beta_min + (beta_max - beta_min) * a_base


def _default_is_matrix(param: Tensor, name: str | None) -> bool:  # noqa: ARG001
    return param.ndim == 2


class SemanticOptimizer(torch.optim.Optimizer):
    """Muon variant with per-channel momentum coupled to an SSM's diagonal A.

    Key arguments beyond Muon's:
      ``a_param_name``: name of the parameter to read for the channel
        time-constant signal (the SSM's ``log_a``). If ``None``, falls
        back to scalar momentum everywhere, reducing to straight Muon.
      ``channel_map``: ``{param_name: channel_axis}``. Params listed here
        receive per-channel β broadcast along the named axis. Params not
        listed use scalar β.
      ``beta_from_a``: override the default mapping from A to per-channel
        β. Signature ``(a_param: Tensor) -> Tensor``; output length must
        equal the channel dimension.

    Call :meth:`bind_param_names` after construction so the optimizer can
    resolve ``a_param_name`` and ``channel_map`` entries by name. Bind
    validates that every configured name exists in the bindings and
    raises otherwise — misconfigured names fail loudly instead of
    silently reducing to Muon.

    After ``load_state_dict`` you do NOT need to re-bind — state dicts
    preserve parameter identity because they load into the same Python
    tensors. A rebind is only needed if you rebuild the model or move
    parameters across optimizer instances.

    v1 constraints:
      * Unfused path only. The fused Muon path batches NS across
        shape-groups; folding per-channel β into that batching needs
        care around per-param channel axes (they can differ within a
        shape group) and is deferred to v2.
      * Matrix params listed in ``channel_map`` go through NS exactly as
        in Muon — the channel axis only controls the momentum decay.
      * The A parameter itself is always updated via the AdamW fallback
        (since it's 1D). Listing it in ``channel_map`` raises, because
        there is no matrix-path momentum buffer for per-channel β to
        broadcast across.

    Wiring example for an SSM with ``self.log_a`` on the recurrence
    block, and ``in_proj``/``gate_proj``/``out_proj`` Linear layers whose
    state axis aligns with ``log_a``::

        opt = SemanticOptimizer(
            list(model.parameters()),
            lr=0.032,
            momentum=0.95,
            momentum_min=0.5,
            a_param_name="ssm.log_a",
            channel_map={
                "ssm.in_proj.weight":   0,   # (out=state, in=model)
                "ssm.gate_proj.weight": 0,
                "ssm.out_proj.weight":  1,   # (out=model, in=state)
            },
        )
        opt.bind_param_names(list(model.named_parameters()))
    """

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        *,
        lr: float = 0.02,
        momentum: float = 0.95,
        momentum_min: float = 0.5,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
        adamw_lr: float | None = None,
        adamw_weight_decay: float | None = None,
        a_param_name: str | None = None,
        channel_map: dict[str, int] | None = None,
        beta_from_a: Callable[[Tensor], Tensor] | None = None,
        matrix_param_names: set[str] | None = None,
        is_matrix: Callable[[Tensor, str | None], bool] | None = None,
        compute_dtype: torch.dtype | None = None,
    ) -> None:
        if lr <= 0.0 or ns_steps <= 0:
            raise ValueError(f"invalid hparams: lr={lr} ns_steps={ns_steps}")
        if not 0.0 <= momentum_min <= momentum < 1.0:
            raise ValueError(
                f"require 0 <= momentum_min ({momentum_min}) "
                f"<= momentum ({momentum}) < 1"
            )
        if (
            a_param_name is not None
            and channel_map is not None
            and a_param_name in channel_map
        ):
            raise ValueError(
                f"A-parameter {a_param_name!r} cannot be in channel_map in v1: "
                f"A is 1D and uses the AdamW fallback, which has no matrix "
                f"momentum buffer for per-channel β to broadcast across. "
                f"A-self-coupling needs a distinct vector-momentum update "
                f"rule (with damping) and is deferred to v2."
            )
        defaults = dict(
            lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
            weight_decay=weight_decay, adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            adamw_lr=lr if adamw_lr is None else adamw_lr,
            adamw_weight_decay=(
                weight_decay if adamw_weight_decay is None else adamw_weight_decay
            ),
        )
        super().__init__(params, defaults)
        self._a_param_name = a_param_name
        self._channel_map = dict(channel_map or {})
        self._matrix_param_names = (
            set(matrix_param_names) if matrix_param_names is not None else None
        )
        self._is_matrix_fn = (
            is_matrix if is_matrix is not None else _default_is_matrix
        )
        self._momentum_max = float(momentum)
        self._momentum_min = float(momentum_min)
        self._beta_from_a = beta_from_a or (
            lambda a: default_beta_from_log_a(
                a, beta_min=self._momentum_min, beta_max=self._momentum_max
            )
        )
        self._compute_dtype = compute_dtype
        self._param_name_by_id: dict[int, str] = {}
        # Cached A-parameter resolved once at bind time. None when
        # a_param_name is None (Muon-reducing mode) or when bind hasn't
        # happened yet.
        self._a_param_ref: Tensor | None = None

    def bind_param_names(
        self, named_params: Iterable[tuple[str, Tensor]]
    ) -> None:
        """Attach (name, param) pairs so channel/A lookup can resolve by name.

        Validates that ``a_param_name`` and every ``channel_map`` key is
        present in the bindings. Misconfigured names raise here rather
        than silently degrading to Muon at step time.
        """
        named_list = list(named_params)
        self._param_name_by_id = {id(p): n for n, p in named_list}
        bound_names = {n for n, _ in named_list}

        if self._a_param_name is not None:
            if self._a_param_name not in bound_names:
                raise ValueError(
                    f"a_param_name {self._a_param_name!r} not found in "
                    f"bound named_params; cannot compute per-channel β"
                )
            # Resolve A tensor now so hot-path step doesn't scan every call.
            self._a_param_ref = None
            for group in self.param_groups:
                for p in group["params"]:
                    if self._param_name_by_id.get(id(p)) == self._a_param_name:
                        self._a_param_ref = p
                        break
                if self._a_param_ref is not None:
                    break
            if self._a_param_ref is None:
                raise ValueError(
                    f"a_param_name {self._a_param_name!r} is in bindings but "
                    f"not in any param_group — did you forget to pass it to "
                    f"the optimizer?"
                )

        missing = [n for n in self._channel_map if n not in bound_names]
        if missing:
            raise ValueError(
                f"channel_map names not in bound named_params: {missing}"
            )

    def current_beta_vec(self) -> Tensor | None:
        """Return the current per-channel β vector, or None if no A is bound.

        Detached fp32 tensor on the A parameter's device. Safe to read
        from training-loop diagnostics without perturbing state.
        """
        if self._a_param_ref is None:
            return None
        return self._beta_from_a(self._a_param_ref).detach()

    def beta_trace(self) -> dict[str, Any] | None:
        """Snapshot the current per-channel β and derived time constants.

        Returns None when the optimizer has no A parameter bound;
        otherwise returns a dict suitable for logging:

        * ``beta_vec``: current β vector (detached fp32, on CPU)
        * ``tau_steps``: per-channel effective time constant,
          ``-1/ln(β_i)`` in optimizer steps. β=0.9 → τ≈9.5, β=0.99 → τ≈99.
          Clipped at β=0.999 so τ stays finite.
        * ``beta_min``, ``beta_max``, ``beta_mean``: scalar summaries.

        Call once every N steps in the training loop to verify the
        mechanism is producing a nontrivial distribution over channels.
        """
        beta = self.current_beta_vec()
        if beta is None:
            return None
        beta = beta.float().cpu()
        clamped = beta.clamp(max=0.999)
        log_beta = torch.log(clamped).clamp(max=-1e-6)
        tau_steps = -1.0 / log_beta
        return {
            "beta_vec": beta,
            "tau_steps": tau_steps,
            "beta_min": float(beta.min()),
            "beta_max": float(beta.max()),
            "beta_mean": float(beta.mean()),
        }

    def _name_of(self, p: Tensor) -> str | None:
        return self._param_name_by_id.get(id(p))

    def _is_matrix_param(self, p: Tensor) -> bool:
        name = self._name_of(p)
        if self._matrix_param_names is not None and name is not None:
            return name in self._matrix_param_names
        return self._is_matrix_fn(p, name)

    @staticmethod
    def _broadcast_beta(
        beta_vec: Tensor, shape: torch.Size, axis: int
    ) -> Tensor:
        if axis < 0:
            axis = len(shape) + axis
        if axis < 0 or axis >= len(shape):
            raise ValueError(
                f"channel axis {axis} out of range for shape {tuple(shape)}"
            )
        if beta_vec.numel() != shape[axis]:
            raise ValueError(
                f"β has {beta_vec.numel()} channels, param axis {axis} "
                f"has {shape[axis]}"
            )
        view = [1] * len(shape)
        view[axis] = beta_vec.numel()
        return beta_vec.view(view)

    def _momentum_buffer_dtype(self, p: Tensor) -> torch.dtype:
        """Pick the momentum-buffer dtype.

        fp32 whenever a_param is bound (so per-channel β multiplies don't
        drop precision at β near 1). p.dtype otherwise, for exact Muon
        parity in the a_param_name=None reduction case.
        """
        if self._a_param_ref is not None:
            return torch.float32
        return p.dtype

    @torch.no_grad()
    def step(
        self, closure: Callable[[], Tensor] | None = None
    ) -> Tensor | None:
        loss: Tensor | None = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Guard against a misconfigured optimizer: a_param_name was set but
        # bind_param_names was never called. Silent reduction to Muon here
        # would masquerade as success; raise instead.
        if self._a_param_name is not None and self._a_param_ref is None:
            raise RuntimeError(
                f"a_param_name={self._a_param_name!r} is set but no A "
                f"parameter resolved — call bind_param_names() before step()"
            )

        # Compute β once per step from the resolved A, then cache the
        # device+dtype-cast views so each matrix param's broadcast doesn't
        # re-cast. beta_vec itself is fp32 on A's device.
        #
        # Ordering: β is read from A *before* this step's AdamW update to
        # A. Matrix params see pre-step β; A itself gets updated later in
        # the loop. This is the consistent ordering — a within-step
        # recompute would create a within-step ordering dependency
        # between matrix params and the A parameter.
        beta_vec: Tensor | None = None
        beta_cast_cache: dict[tuple[torch.device, torch.dtype], Tensor] = {}
        if self._a_param_ref is not None:
            beta_vec = self._beta_from_a(self._a_param_ref)

        for group in self.param_groups:
            lr = group["lr"]
            scalar_momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]
            beta1, beta2 = group["adamw_betas"]
            adamw_eps = group["adamw_eps"]
            adamw_lr = group["adamw_lr"]
            adamw_wd = group["adamw_weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                name = self._name_of(p)

                if self._is_matrix_param(p):
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(
                            p,
                            memory_format=torch.preserve_format,
                            dtype=self._momentum_buffer_dtype(p),
                        )
                    buf = state["momentum_buffer"]

                    if (
                        beta_vec is not None
                        and name is not None
                        and name in self._channel_map
                    ):
                        axis = self._channel_map[name]
                        cache_key = (buf.device, buf.dtype)
                        if cache_key not in beta_cast_cache:
                            beta_cast_cache[cache_key] = beta_vec.to(
                                device=buf.device, dtype=buf.dtype
                            )
                        beta: Tensor | float = self._broadcast_beta(
                            beta_cast_cache[cache_key], buf.shape, axis
                        )
                    else:
                        beta = scalar_momentum

                    # Promote grad to buf.dtype for the momentum update so
                    # fp32 bufs stay fp32 across a bf16 grad. This is a
                    # no-op when buf.dtype == grad.dtype (Muon parity case).
                    grad_for_buf = grad.to(dtype=buf.dtype)
                    buf.mul_(beta).add_(grad_for_buf)
                    if nesterov:
                        direction = (
                            grad_for_buf + beta * buf
                            if isinstance(beta, Tensor)
                            else grad_for_buf.add(buf, alpha=beta)
                        )
                    else:
                        direction = buf
                    update = newton_schulz_orthogonalize(
                        direction, steps=ns_steps,
                        compute_dtype=self._compute_dtype,
                    )
                    rows, cols = p.shape[-2], p.shape[-1]
                    scale = max(1.0, rows / cols) ** 0.5
                    if wd > 0.0:
                        p.data.mul_(1.0 - lr * wd)
                    p.data.add_(update.to(dtype=p.dtype), alpha=-lr * scale)
                else:
                    if "step" not in state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                        state["exp_avg_sq"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                    state["step"] += 1
                    t = state["step"]
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(
                        grad, grad, value=1.0 - beta2
                    )
                    bias1 = 1.0 - beta1 ** t
                    bias2 = 1.0 - beta2 ** t
                    denom = (exp_avg_sq.sqrt() / (bias2 ** 0.5)).add_(adamw_eps)
                    if adamw_wd > 0.0:
                        p.data.mul_(1.0 - adamw_lr * adamw_wd)
                    p.data.addcdiv_(
                        exp_avg, denom, value=-adamw_lr / bias1
                    )

        return loss
