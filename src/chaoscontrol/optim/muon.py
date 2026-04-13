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
        compute_dtype: torch.dtype | None = None,
    ) -> None:
        if lr <= 0.0 or ns_steps <= 0 or not 0.0 <= momentum < 1.0:
            raise ValueError(f"invalid Muon hparams: lr={lr} momentum={momentum} ns_steps={ns_steps}")
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
        self._compute_dtype = compute_dtype
        # Id-keyed lookup from param tensor -> optional name for the classifier.
        self._param_name_by_id: dict[int, str] = {}

    def bind_param_names(self, named_params: Iterable[tuple[str, Tensor]]) -> None:
        """Attach (name, param) pairs so the classifier can use names, if provided."""
        self._param_name_by_id = {id(p): n for n, p in named_params}

    def _is_matrix_param(self, p: Tensor) -> bool:
        name = self._param_name_by_id.get(id(p))
        if self._matrix_param_names is not None and name is not None:
            return name in self._matrix_param_names
        return self._is_matrix_fn(p, name)

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor] | None = None) -> Tensor | None:
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
                    p.data.add_(update.to(dtype=p.dtype), alpha=-lr * scale)
                else:
                    # Decoupled AdamW fallback for scalar / vector / non-matrix params.
                    if "step" not in state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["step"] += 1
                    t = state["step"]
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                    bias1 = 1.0 - beta1 ** t
                    bias2 = 1.0 - beta2 ** t
                    denom = (exp_avg_sq.sqrt() / (bias2 ** 0.5)).add_(adamw_eps)
                    if adamw_wd > 0.0:
                        p.data.mul_(1.0 - adamw_lr * adamw_wd)
                    p.data.addcdiv_(exp_avg, denom, value=-adamw_lr / bias1)

        return loss
