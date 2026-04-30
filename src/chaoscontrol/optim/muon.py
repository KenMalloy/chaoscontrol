"""Muon optimizer with an AdamW fallback for non-matrix parameters.

Lifted from ``baselines/parameter_golf/sota/train_gpt.py`` with the
distributed reduce-scatter/all-gather plumbing stripped so this runs as a
plain single-process ``torch.optim.Optimizer``. Matrix-shaped weights
receive a 5-step Newton-Schulz orthogonalized-momentum update; scalar,
vector, and other non-matrix params (biases, LayerNorm gains, log-decay
scalars, 1D SSM state tensors) fall through to an inline decoupled AdamW
step so one optimizer owns the whole parameter set.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable

import torch
from torch import Tensor

# Quintic polynomial coefficients for the Newton-Schulz matrix-sign
# iteration; these are mathematical constants tuned for 5-step convergence.
_NS_COEFFS = (3.4445, -4.7750, 2.0315)


def newton_schulz_orthogonalize(grad: Tensor, steps: int = 5, eps: float = 1e-7,
                                 compute_dtype: torch.dtype | None = None) -> Tensor:
    """Quintic Newton-Schulz iteration pushing ``grad`` toward an orthogonal factor.

    Accepts a 2D matrix or a batch of matrices (leading dims treated as a
    batch axis). Runs in ``compute_dtype`` (defaults to bf16 on CUDA and
    float32 on CPU because CPU bf16 matmul is imprecise).
    """
    if compute_dtype is None:
        compute_dtype = torch.bfloat16 if grad.is_cuda else torch.float32
    a, b, c = _NS_COEFFS
    was_flat = grad.ndim == 2
    X = grad.unsqueeze(0) if was_flat else grad
    X = X.to(compute_dtype)
    # Transpose tall matrices so rows <= cols; NS5 converges on that side.
    tall = X.size(-2) > X.size(-1)
    if tall:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        XXt = X @ X.mT
        poly = b * XXt + c * (XXt @ XXt)
        X = a * X + poly @ X
    if tall:
        X = X.mT
    if was_flat:
        X = X.squeeze(0)
    return X


def _default_is_matrix(param: Tensor, name: str | None) -> bool:  # noqa: ARG001
    """Default classifier: treat exactly-2D tensors as matrix params."""
    if name and (
        name.endswith(".delta_proj.weight") or ".delta_proj." in name
    ):
        return False
    return param.ndim == 2


class Muon(torch.optim.Optimizer):
    """Single-process Muon with a decoupled AdamW fallback for non-matrix params."""

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
        adamw_lr: float | None = None,
        adamw_weight_decay: float | None = None,
        matrix_param_names: set[str] | None = None,
        is_matrix: Callable[[Tensor, str | None], bool] | None = None,
        fused: bool = False,
        compute_dtype: torch.dtype | None = None,
        log_a_beta_coupling: bool = False,
        log_a_beta_ema: float = 0.99,
        log_a_beta_min: float = 0.5,
    ) -> None:
        if lr <= 0.0 or ns_steps <= 0 or not 0.0 <= momentum < 1.0:
            raise ValueError(f"invalid Muon hparams: lr={lr} momentum={momentum} ns_steps={ns_steps}")
        if not 0.0 <= float(log_a_beta_min) < 1.0:
            raise ValueError(
                f"log_a_beta_min must be in [0, 1), got {log_a_beta_min}"
            )
        if not 0.0 <= float(log_a_beta_ema) < 1.0:
            raise ValueError(
                f"log_a_beta_ema must be in [0, 1), got {log_a_beta_ema}"
            )
        defaults = dict(
            lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
            weight_decay=weight_decay, adamw_betas=adamw_betas, adamw_eps=adamw_eps,
            adamw_lr=lr if adamw_lr is None else adamw_lr,
            adamw_weight_decay=weight_decay if adamw_weight_decay is None else adamw_weight_decay,
        )
        super().__init__(params, defaults)
        # Note: explicit `is not None` check — an empty set() is a legitimate
        # override meaning "no params are matrix params, run everything through
        # AdamW." Using plain truthiness would collapse set() to None and
        # silently route matrix-shaped params into the Newton-Schulz path.
        self._matrix_param_names = set(matrix_param_names) if matrix_param_names is not None else None
        self._is_matrix_fn = is_matrix if is_matrix is not None else _default_is_matrix
        self._fused = fused
        self._compute_dtype = compute_dtype
        # Id-keyed lookup from param tensor -> optional name for the classifier.
        self._param_name_by_id: dict[int, str] = {}
        self._plasticity_multiplier: Tensor | None = None
        self._plasticity_multiplier_views: dict[tuple[torch.device, torch.dtype], Tensor] = {}
        self._plasticity_step: int = -1
        self._plasticity_strength: float = 0.0
        self._plasticity_updates: int = 0
        self._plasticity_last_dim: int = 0
        self._plasticity_last_budget: Tensor | None = None
        self._plasticity_last_confidence: Tensor | None = None
        self._log_a_beta_coupling = bool(log_a_beta_coupling)
        self._log_a_beta_ema = float(log_a_beta_ema)
        self._log_a_beta_min = float(log_a_beta_min)
        self._log_a_beta_updates = 0
        self._log_a_beta_last: Tensor | None = None

    def bind_param_names(self, named_params: Iterable[tuple[str, Tensor]]) -> None:
        """Attach (name, param) pairs so the classifier can use names, if provided."""
        self._param_name_by_id = {id(p): n for n, p in named_params}

    def _is_matrix_param(self, p: Tensor) -> bool:
        name = self._param_name_by_id.get(id(p))
        if self._matrix_param_names is not None and name is not None:
            return name in self._matrix_param_names
        return self._is_matrix_fn(p, name)

    def _uses_log_a_beta(self, p: Tensor) -> bool:
        if not self._log_a_beta_coupling:
            return False
        name = self._param_name_by_id.get(id(p), "")
        return p.ndim == 1 and name.endswith(".log_a")

    def _log_a_beta_for(
        self,
        p: Tensor,
        state: dict[str, Any],
        beta1: float,
    ) -> tuple[Tensor, Tensor] | None:
        """Return per-channel beta1 and bias correction for log_a.

        The beta signal is derived from a slow EMA of log_a, not the
        instant parameter value. That keeps the optimizer's timescale
        match from forming a tight feedback loop with the same update it
        is about to apply.
        """
        if not self._uses_log_a_beta(p):
            return None
        ema = state.get("log_a_slow_ema")
        current = p.detach().float()
        if ema is None:
            ema = current.clone()
            state["log_a_slow_ema"] = ema
        else:
            ema.mul_(self._log_a_beta_ema).add_(
                current, alpha=1.0 - self._log_a_beta_ema
            )
        beta_floor = min(float(self._log_a_beta_min), float(beta1))
        beta_vec = beta_floor + (float(beta1) - beta_floor) * torch.sigmoid(ema)
        beta_vec = beta_vec.to(device=p.device, dtype=torch.float32)
        prod = state.get("log_a_beta1_prod")
        if prod is None:
            prod = beta_vec.clone()
            state["log_a_beta1_prod"] = prod
        else:
            prod.mul_(beta_vec)
        bias1 = (1.0 - prod).clamp_min(1e-12)
        self._log_a_beta_updates += 1
        self._log_a_beta_last = beta_vec.detach()
        return beta_vec.to(device=p.device, dtype=p.dtype), bias1.to(
            device=p.device, dtype=p.dtype
        )

    @torch.no_grad()
    def set_plasticity_budget(
        self,
        budget: Tensor,
        *,
        confidence: Tensor | None = None,
        strength: float = 0.25,
        step: int | None = None,
    ) -> None:
        """Install the latest optimizer-side ARM plasticity packet.

        ``budget`` is a per-channel trusted coverage gate in ``[0, 1]``.
        Channels with high trusted coverage receive up to
        ``1 + strength`` times the base optimizer update; missing packets
        leave the previously installed latest-complete packet in place.
        """
        b = budget.detach().float().reshape(-1).clamp(min=0.0, max=1.0)
        if confidence is None:
            c = torch.ones_like(b)
        else:
            c = confidence.detach().float().reshape(-1).clamp(min=0.0, max=1.0)
            if c.numel() != b.numel():
                raise ValueError(
                    "plasticity confidence must match budget shape; got "
                    f"{tuple(c.shape)} vs {tuple(b.shape)}"
                )
        effective = b
        self._plasticity_multiplier = 1.0 + float(strength) * effective
        self._plasticity_multiplier_views.clear()
        self._plasticity_step = -1 if step is None else int(step)
        self._plasticity_strength = float(strength)
        self._plasticity_updates += 1
        self._plasticity_last_dim = int(effective.numel())
        self._plasticity_last_budget = effective
        self._plasticity_last_confidence = c

    def plasticity_budget_trace(self) -> dict[str, float | int | bool | list[int] | list[float]]:
        mult = self._plasticity_multiplier
        if mult is None or int(mult.numel()) == 0:
            return {
                "enabled": False,
                "updates": int(self._plasticity_updates),
                "step": int(self._plasticity_step),
                "dim": 0,
                "strength": float(self._plasticity_strength),
                "budget_mean": 0.0,
                "budget_max": 0.0,
                "confidence_mean": 0.0,
                "lr_multiplier_min": 1.0,
                "lr_multiplier_mean": 1.0,
                "lr_multiplier_max": 1.0,
                "budget_nonzero_fraction": 0.0,
                "confidence_max": 0.0,
                "top_channels": [],
                "top_budget": [],
            }
        budget = self._plasticity_last_budget
        confidence = self._plasticity_last_confidence
        top_channels: list[int] = []
        top_budget: list[float] = []
        if budget is not None and int(budget.numel()) > 0:
            k = min(8, int(budget.numel()))
            vals, idx = torch.topk(budget.detach().float(), k=k)
            top_channels = [int(i) for i in idx.cpu().tolist()]
            top_budget = [float(v) for v in vals.cpu().tolist()]
        return {
            "enabled": True,
            "updates": int(self._plasticity_updates),
            "step": int(self._plasticity_step),
            "dim": int(self._plasticity_last_dim),
            "strength": float(self._plasticity_strength),
            "budget_mean": (
                float(self._plasticity_last_budget.mean().item())
                if self._plasticity_last_budget is not None
                else 0.0
            ),
            "budget_max": (
                float(self._plasticity_last_budget.max().item())
                if self._plasticity_last_budget is not None
                else 0.0
            ),
            "confidence_mean": (
                float(confidence.mean().item())
                if confidence is not None
                else 0.0
            ),
            "budget_nonzero_fraction": (
                float((budget > 0).float().mean().item())
                if budget is not None
                else 0.0
            ),
            "confidence_max": (
                float(confidence.max().item())
                if confidence is not None
                else 0.0
            ),
            "lr_multiplier_min": float(mult.min().item()),
            "lr_multiplier_mean": float(mult.mean().item()),
            "lr_multiplier_max": float(mult.max().item()),
            "top_channels": top_channels,
            "top_budget": top_budget,
        }

    def ssm_role_trace(self) -> dict[str, float | int | bool]:
        beta = self._log_a_beta_last
        if beta is None or int(beta.numel()) == 0:
            return {
                "log_a_beta_coupling": bool(self._log_a_beta_coupling),
                "log_a_beta_updates": int(self._log_a_beta_updates),
                "log_a_beta_ema": float(self._log_a_beta_ema),
                "log_a_beta_min_config": float(self._log_a_beta_min),
                "log_a_beta_min": 0.0,
                "log_a_beta_mean": 0.0,
                "log_a_beta_max": 0.0,
            }
        beta_cpu = beta.detach().float()
        return {
            "log_a_beta_coupling": bool(self._log_a_beta_coupling),
            "log_a_beta_updates": int(self._log_a_beta_updates),
            "log_a_beta_ema": float(self._log_a_beta_ema),
            "log_a_beta_min_config": float(self._log_a_beta_min),
            "log_a_beta_min": float(beta_cpu.min().item()),
            "log_a_beta_mean": float(beta_cpu.mean().item()),
            "log_a_beta_max": float(beta_cpu.max().item()),
        }

    def _plasticity_multiplier_for(self, p: Tensor) -> Tensor | None:
        mult = self._plasticity_multiplier
        if mult is None:
            return None
        name = self._param_name_by_id.get(id(p), "")
        if not name.startswith("layers.") or ".core." not in name:
            return None
        view_key = (p.device, p.dtype)
        m = self._plasticity_multiplier_views.get(view_key)
        if m is None:
            m = mult.to(device=p.device, dtype=p.dtype)
            self._plasticity_multiplier_views[view_key] = m
        d = int(m.numel())
        if p.ndim == 1 and p.numel() == d and name.endswith(
            (".log_a", ".log_r", ".theta")
        ):
            return m.reshape_as(p)
        if p.ndim != 2:
            return None
        if name.endswith(
            (
                ".in_proj.weight",
                ".select_proj.weight",
                ".gate_proj.weight",
                ".delta_proj.weight",
                ".delta_proj.up.weight",
            )
        ) and int(p.shape[0]) == d:
            return m.view(d, 1)
        if name.endswith(".delta_proj.down.weight") and int(p.shape[1]) == d:
            return m.view(1, d)
        if name.endswith(".out_proj.weight") and int(p.shape[1]) == d:
            return m.view(1, d)
        return None

    def _adamw_update_param(
        self,
        *,
        p: Tensor,
        state: dict[str, Any],
        grad: Tensor,
        beta1: float,
        beta2: float,
        adamw_eps: float,
        adamw_lr: float,
        adamw_wd: float,
    ) -> None:
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
        beta_pair = self._log_a_beta_for(p, state, beta1)
        if beta_pair is None:
            exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            bias1: float | Tensor = 1.0 - beta1 ** t
        else:
            beta_vec, bias1 = beta_pair
            exp_avg.mul_(beta_vec).add_(grad * (1.0 - beta_vec))
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
        bias2 = 1.0 - beta2 ** t
        denom = (exp_avg_sq.sqrt() / (bias2 ** 0.5)).add_(adamw_eps)
        if adamw_wd > 0.0:
            p.data.mul_(1.0 - adamw_lr * adamw_wd)
        mult = self._plasticity_multiplier_for(p)
        update = exp_avg / denom
        if isinstance(bias1, torch.Tensor):
            update = update / bias1
            step = update if mult is None else update * mult
            p.data.add_(step, alpha=-adamw_lr)
        elif mult is None:
            p.data.addcdiv_(exp_avg, denom, value=-adamw_lr / bias1)
        else:
            p.data.add_(update * mult, alpha=-adamw_lr / bias1)

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor] | None = None) -> Tensor | None:
        if self._fused:
            return self._step_fused(closure)

        loss: Tensor | None = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
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

                if self._is_matrix_param(p):
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(grad)
                    direction = grad.add(buf, alpha=momentum) if nesterov else buf
                    update = newton_schulz_orthogonalize(
                        direction, steps=ns_steps, compute_dtype=self._compute_dtype
                    )
                    # Rectangular-shape correction: square roots of the row/col ratio.
                    rows, cols = p.shape[-2], p.shape[-1]
                    scale = max(1.0, rows / cols) ** 0.5
                    if wd > 0.0:
                        p.data.mul_(1.0 - lr * wd)
                    mult = self._plasticity_multiplier_for(p)
                    if mult is not None:
                        update = update * mult
                    p.data.add_(update.to(dtype=p.dtype), alpha=-lr * scale)
                else:
                    # Decoupled AdamW fallback for scalar / vector / non-matrix params.
                    self._adamw_update_param(
                        p=p,
                        state=state,
                        grad=grad,
                        beta1=beta1,
                        beta2=beta2,
                        adamw_eps=adamw_eps,
                        adamw_lr=adamw_lr,
                        adamw_wd=adamw_wd,
                    )

        return loss

    @torch.no_grad()
    def _step_fused(self, closure: Callable[[], Tensor] | None = None) -> Tensor | None:
        """Fused Muon step.

        Matrix params are partitioned by exact ``(rows, cols)`` shape; each
        shape-group's grads and momentum buffers are stacked into a single
        ``[G, rows, cols]`` tensor and pushed through one Newton-Schulz
        call (``newton_schulz_orthogonalize`` already handles a leading
        batch dim; see ``muon.py:34-35``). Non-matrix params flatten into
        three shared buffers (exp_avg, exp_avg_sq, grad) for a coalesced
        in-place AdamW update.

        Per-param ``self.state[p]`` entries are preserved so
        ``optimizer.state_dict()`` serialization stays checkpoint-
        compatible with the unfused path.
        """
        loss: Tensor | None = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]
            beta1, beta2 = group["adamw_betas"]
            adamw_eps = group["adamw_eps"]
            adamw_lr = group["adamw_lr"]
            adamw_wd = group["adamw_weight_decay"]

            matrix_groups: dict[tuple[int, int], list[Tensor]] = {}
            nonmatrix_params: list[Tensor] = []

            for p in group["params"]:
                if p.grad is None:
                    continue
                if self._is_matrix_param(p):
                    key = (int(p.shape[-2]), int(p.shape[-1]))
                    matrix_groups.setdefault(key, []).append(p)
                else:
                    nonmatrix_params.append(p)

            # ----- Matrix branch: one NS call per (rows, cols) shape-group.
            for (rows, cols), params_in_group in matrix_groups.items():
                # Ensure per-param momentum_buffer state exists first so
                # per-param state dicts remain consistent for serialization.
                bufs: list[Tensor] = []
                grads: list[Tensor] = []
                for p in params_in_group:
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                    bufs.append(state["momentum_buffer"])
                    grads.append(p.grad)

                grad_stack = torch.stack(grads, dim=0)  # [G, rows, cols]
                buf_stack = torch.stack(bufs, dim=0)

                # Fused momentum update: buf = momentum * buf + grad
                buf_stack.mul_(momentum).add_(grad_stack)
                direction = (
                    grad_stack.add(buf_stack, alpha=momentum) if nesterov else buf_stack
                )
                # Module-level lookup so monkeypatch.setattr on the module
                # resolves to the patched callable (tests rely on this).
                update_stack = newton_schulz_orthogonalize(
                    direction, steps=ns_steps, compute_dtype=self._compute_dtype
                )

                # Rectangular-shape correction is constant across the group.
                scale = max(1.0, rows / cols) ** 0.5

                for i, p in enumerate(params_in_group):
                    # Write the updated momentum buffer back into per-param
                    # state so state_dict() keeps its per-param entries.
                    # copy_ rather than reassign so any external aliases on
                    # self.state[p]["momentum_buffer"] remain valid.
                    self.state[p]["momentum_buffer"].copy_(buf_stack[i])
                    if wd > 0.0:
                        p.data.mul_(1.0 - lr * wd)
                    update = update_stack[i]
                    mult = self._plasticity_multiplier_for(p)
                    if mult is not None:
                        update = update * mult
                    p.data.add_(update.to(dtype=p.dtype), alpha=-lr * scale)

            # ----- Non-matrix branch: coalesced AdamW over flat buffers.
            if not nonmatrix_params:
                continue

            # Initialize state for any uninitialized params and collect the
            # per-param tensors. We deliberately keep per-param state dicts
            # rather than one global (exp_avg, exp_avg_sq) buffer so
            # state_dict() structure matches the unfused path.
            exp_avgs: list[Tensor] = []
            exp_avg_sqs: list[Tensor] = []
            grads_flat: list[Tensor] = []
            steps_before: list[int] = []
            for p in nonmatrix_params:
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])
                grads_flat.append(p.grad)
                steps_before.append(state["step"])

            # If step counters disagree (hand-edited state_dict, or a
            # non-matrix param added mid-training), fall through to the
            # per-param path — the coalesced update assumes a single
            # shared `t`. Graceful degradation: math still matches the
            # unfused branch, we just lose the coalesce win for this
            # step.
            uniform_step = all(s == steps_before[0] for s in steps_before)
            adaptive_log_a = any(self._uses_log_a_beta(p) for p in nonmatrix_params)
            if not uniform_step or adaptive_log_a:
                for p in nonmatrix_params:
                    state = self.state[p]
                    self._adamw_update_param(
                        p=p,
                        state=state,
                        grad=p.grad,
                        beta1=beta1,
                        beta2=beta2,
                        adamw_eps=adamw_eps,
                        adamw_lr=adamw_lr,
                        adamw_wd=adamw_wd,
                    )
                continue

            # Uniform step-count path: coalesced update over flat buffers.
            # We flatten views into a contiguous 1D buffer via the stdlib
            # helper (reused from chaoscontrol.distributed.allreduce_grads'
            # coalesce pattern). In-place ops on the flat buffer write
            # through to the per-param tensors because _unflatten_dense_
            # tensors returns views on the flat buffer.
            from torch._utils import (
                _flatten_dense_tensors,
                _unflatten_dense_tensors,
            )

            # Flatten each tensor list. Each _flatten call concatenates a
            # detached copy of the input tensors, so updates on the flat
            # buffer will need to be copied back. This is the standard
            # coalesce pattern — the same one allreduce_grads uses.
            #
            # _flatten_dense_tensors silently upcasts mixed-dtype inputs
            # to the widest common dtype; the unfused AdamW branch
            # operates on each param in its own dtype. Assert uniformity
            # across grads AND state so a future mixed-dtype change
            # (e.g. a bf16 bias alongside an fp32 LayerNorm gain) fails
            # loud here instead of silently diverging from the unfused
            # path over thousands of steps.
            dtypes = (
                {t.dtype for t in grads_flat}
                | {t.dtype for t in exp_avgs}
                | {t.dtype for t in exp_avg_sqs}
            )
            assert len(dtypes) == 1, (
                f"fused AdamW coalesce requires uniform non-matrix dtypes "
                f"across grads, exp_avg, and exp_avg_sq; got {dtypes}. "
                f"Either add mixed-dtype support to _step_fused or fall "
                f"through to per-param AdamW for non-uniform groups."
            )
            exp_avg_flat = _flatten_dense_tensors(exp_avgs)
            exp_avg_sq_flat = _flatten_dense_tensors(exp_avg_sqs)
            grad_flat = _flatten_dense_tensors(grads_flat)

            t = steps_before[0] + 1
            exp_avg_flat.mul_(beta1).add_(grad_flat, alpha=1.0 - beta1)
            exp_avg_sq_flat.mul_(beta2).addcmul_(grad_flat, grad_flat, value=1.0 - beta2)
            bias1 = 1.0 - beta1 ** t
            bias2 = 1.0 - beta2 ** t
            denom_flat = (exp_avg_sq_flat.sqrt() / (bias2 ** 0.5)).add_(adamw_eps)

            # Copy the flat results back to per-param state, then apply the
            # parameter update per-param (weight decay + addcdiv). Applying
            # the update on the flat buffer would require matching dtypes
            # across every param; per-param copy-back preserves the
            # unfused path's dtype handling exactly.
            exp_avg_views = _unflatten_dense_tensors(exp_avg_flat, exp_avgs)
            exp_avg_sq_views = _unflatten_dense_tensors(exp_avg_sq_flat, exp_avg_sqs)
            denom_views = _unflatten_dense_tensors(denom_flat, exp_avgs)

            for i, p in enumerate(nonmatrix_params):
                state = self.state[p]
                state["exp_avg"].copy_(exp_avg_views[i])
                state["exp_avg_sq"].copy_(exp_avg_sq_views[i])
                state["step"] = t
                if adamw_wd > 0.0:
                    p.data.mul_(1.0 - adamw_lr * adamw_wd)
                mult = self._plasticity_multiplier_for(p)
                if mult is None:
                    p.data.addcdiv_(
                        state["exp_avg"], denom_views[i], value=-adamw_lr / bias1,
                    )
                else:
                    update = state["exp_avg"] / denom_views[i]
                    p.data.add_(update * mult, alpha=-adamw_lr / bias1)

        return loss
