"""LAMB optimizer: layer-wise Adam with trust-ratio scaling.

Reference:
    You et al. (2019) "Large Batch Optimization for Deep Learning:
    Training BERT in 76 minutes." arXiv:1904.00962.

LAMB is Adam with a per-parameter-tensor trust ratio that scales each
update so it stays proportional to the parameter magnitude:

    m_t     = beta1 * m_{t-1} + (1 - beta1) * g_t
    v_t     = beta2 * v_{t-1} + (1 - beta2) * g_t^2
    m_hat   = m_t / (1 - beta1^t)          # optional, gated by bias_correction
    v_hat   = v_t / (1 - beta2^t)          # optional, gated by bias_correction
    r_t     = m_hat / (sqrt(v_hat) + eps) + weight_decay * w    # update direction
    phi     = clip(||w|| / ||r_t||, 0, trust_clip)              # trust ratio
    w_t+1   = w_t - lr * phi * r_t

Critical detail: LAMB folds weight decay INTO the update direction r_t
(classical L2-regularization style), then trust-ratio-scales the whole
thing. This differs from AdamW's decoupled weight decay and is the #1
source of silent bugs in reference impls. The original paper's algorithm
box (Algorithm 2) is explicit on this; we match it.

Guards against div-by-zero follow the canonical convention used by
NVIDIA's open-source reference and timm's pure-PyTorch impl: if either
||w|| or ||r_t|| is zero, the trust ratio falls back to 1.0, which makes
the step reduce to plain Adam for that tensor on that step.

Usage:
    from chaoscontrol.optim import LAMB
    optimizer = LAMB(model.parameters(), lr=2e-3, weight_decay=0.01)
"""
from __future__ import annotations

from typing import Any, Callable, Iterable

import torch
from torch.optim.optimizer import Optimizer

ParamsT = Iterable[torch.Tensor] | Iterable[dict[str, Any]]


class LAMB(Optimizer):
    """Layer-wise Adaptive Moments optimizer (LAMB).

    Args:
        params: iterable of parameters or param groups.
        lr: learning rate (default 1e-3).
        betas: (beta1, beta2) EMA decay for first and second moments.
        eps: term added to denominator for numerical stability.
        weight_decay: L2 coefficient folded into the update direction,
            not decoupled. Default 0.0.
        trust_clip: upper bound on the per-tensor trust ratio. Default
            10.0, which matches the original paper and common refs. Set
            to ``float('inf')`` to leave the ratio unclipped.
        bias_correction: whether to apply Adam's 1/(1-beta^t) bias
            correction. Default True. The original paper omits it; most
            modern impls enable it and find it stabilizes early steps.
        always_adapt: if True, the trust ratio is applied even to scalar
            or one-dimensional tensors (biases, norms). Default False,
            which matches timm/NVIDIA refs and skips trust-ratio scaling
            for 1-D tensors to avoid over-shrinking biases.
        trust_ratio_override: if not None, force the trust ratio to this
            constant value for every tensor. Intended for the
            "LAMB-with-trust-ratio=1-equals-Adam" parity test; leave as
            None in production.
    """

    def __init__(
        self,
        params: ParamsT,
        *,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        trust_clip: float = 10.0,
        bias_correction: bool = True,
        always_adapt: bool = False,
        trust_ratio_override: float | None = None,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"invalid lr: {lr}")
        if eps < 0.0:
            raise ValueError(f"invalid eps: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"invalid beta2: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"invalid weight_decay: {weight_decay}")
        if trust_clip <= 0.0:
            raise ValueError(f"invalid trust_clip: {trust_clip}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            trust_clip=trust_clip,
            bias_correction=bias_correction,
            always_adapt=always_adapt,
            trust_ratio_override=trust_ratio_override,
        )
        super().__init__(params, defaults)

    @staticmethod
    def _trust_ratio(
        w_norm: torch.Tensor,
        r_norm: torch.Tensor,
        *,
        trust_clip: float,
    ) -> torch.Tensor:
        """Compute the per-tensor trust ratio phi = clip(||w|| / ||r||, 0, trust_clip).

        When either norm is zero, phi falls back to 1.0 — this keeps the
        step well-defined (it reduces to a plain Adam step for that
        tensor on that update) and avoids division by zero. Returned as
        a 0-dim tensor on the same device/dtype as the inputs.
        """
        one = torch.ones((), dtype=w_norm.dtype, device=w_norm.device)
        clip = torch.full((), trust_clip, dtype=w_norm.dtype, device=w_norm.device)
        return torch.where(
            (w_norm > 0) & (r_norm > 0),
            torch.minimum(w_norm / r_norm, clip),
            one,
        )

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        """Perform a single optimization step."""
        loss: torch.Tensor | None = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr: float = group["lr"]
            eps: float = group["eps"]
            weight_decay: float = group["weight_decay"]
            trust_clip: float = group["trust_clip"]
            bias_correction: bool = group["bias_correction"]
            always_adapt: bool = group["always_adapt"]
            trust_ratio_override: float | None = group["trust_ratio_override"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("LAMB does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg: torch.Tensor = state["exp_avg"]
                exp_avg_sq: torch.Tensor = state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]

                # First and second moment EMAs (in-place, Adam-style).
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if bias_correction:
                    bc1 = 1.0 - beta1 ** t
                    bc2 = 1.0 - beta2 ** t
                    m_hat = exp_avg / bc1
                    v_hat = exp_avg_sq / bc2
                else:
                    m_hat = exp_avg
                    v_hat = exp_avg_sq

                # Update direction with L2-style weight decay folded in.
                # This is the LAMB convention (NOT AdamW-decoupled).
                update = m_hat / (v_hat.sqrt() + eps)
                if weight_decay != 0.0:
                    update = update.add(p, alpha=weight_decay)

                # Per-tensor trust ratio. Fall back to 1.0 when either
                # norm is zero, or when the tensor is 1-D and we're not
                # explicitly adapting everything.
                if trust_ratio_override is not None:
                    trust_ratio = torch.full(
                        (),
                        float(trust_ratio_override),
                        dtype=update.dtype,
                        device=update.device,
                    )
                elif not always_adapt and p.ndim <= 1:
                    trust_ratio = torch.ones(
                        (), dtype=update.dtype, device=update.device
                    )
                else:
                    trust_ratio = self._trust_ratio(
                        p.norm(p=2),
                        update.norm(p=2),
                        trust_clip=trust_clip,
                    )

                # NB: multiply the update by the scalar trust_ratio tensor
                # first so we stay device-resident (no Python-side sync)
                # even when the ratio is a GPU scalar.
                p.add_(update * trust_ratio, alpha=-lr)

        return loss
