"""SSM-aware parameter grouping for optimizer construction.

Motivation: S4/S5/HOPE primary sources converge on the recommendation that
SSM dynamics parameters (the A / Δ / B_dyn triplet) need different
optimizer hyperparameters from readout/MLP/embedding weights. The three
concrete asks that survive across sources:

    - smaller learning rate on spectral parameters (HOPE: state matrices
      trained "on a logarithmic scale with a very small learning rate")
    - zero weight decay on A's parameterization (S4 codebase hooks;
      S5 sweeps with wd=0 on the dynamics triplet)
    - zero weight decay on norm gains and biases (standard LLM practice)

Scope note: CareSSMCore has no standalone Δ parameter — its discretization
step is ``F.softplus(delta_proj(x))``, computed per-token from a plain
2D projection that lands in the ``main`` group with full weight decay.
Only A's parameterization (the seven suffixes below) gets the dynamics
treatment here.

Per-shape update path:

  * 1D dynamics params (``log_a``, ``log_r``, ``theta``, ``skew_params``,
    ``log_gamma``) → Muon/SemanticOptimizer/ScOpt's AdamW fallback at the
    dynamics group's ``lr`` / ``wd``.
  * 2D dynamics params (``U``, ``V`` in ``a_mode="full"``) → Muon's
    Newton-Schulz matrix path at the dynamics group's ``lr`` / ``wd``.
    This is deliberate: the SSM optimizer research flags low-rank
    corrections as exactly the place where structure-aware curvature
    (Shampoo/SOAP-style) outperforms diagonal adaptivity, and NS gives
    that orthogonalized update with the right directional structure.
    If a caller wants U/V through AdamW instead, pass
    ``matrix_param_names`` to the optimizer with U/V excluded.

Submission regime is ``a_mode="diag"``, so U/V don't exist in the current
locked config. This policy is relevant only to future full-mode work.

This module centralizes the classification so every optimizer in the
subpackage uses the same rule. Classification is shape-first with
name-based override for spectral parameters.
"""
from __future__ import annotations

from typing import Any, Iterable

from torch import Tensor


# Parameter-name suffixes that identify A-parameterization tensors across
# the CareSSMCore a_modes. All live under ``layers.{i}.core.<suffix>``:
#
#   diag mode:   log_a
#   paired mode: log_r, theta
#   full mode:   skew_params, log_gamma, U, V
#
# Any parameter whose name ends in one of these suffixes is treated as
# dynamics (smaller LR, zero WD). Matched on the last dot-segment only.
# The single-letter suffixes (``U``, ``V``) are specific to CareSSMCore's
# full-mode low-rank factor; if a future module adopts them for something
# unrelated, it would accidentally land in the dynamics group. No such
# collision exists today.
SPECTRAL_SUFFIXES: frozenset[str] = frozenset(
    {
        "log_a",
        "log_r",
        "theta",
        "skew_params",
        "log_gamma",
        "U",
        "V",
    }
)


def classify_param(name: str, param: Tensor) -> str:
    """Return one of ``"dynamics"``, ``"no_decay"``, ``"main"``.

    Rule:
      * suffix in ``SPECTRAL_SUFFIXES`` → ``"dynamics"``
      * otherwise ``param.ndim <= 1`` → ``"no_decay"`` (norm gains, biases)
      * otherwise → ``"main"`` (matrix weights)

    Shape-first with spectral-name override. A future 1D readout param
    that's neither spectral nor a norm would land in ``no_decay``, which
    is the conservative default for 1D tensors.
    """
    suffix = name.rsplit(".", 1)[-1]
    if suffix in SPECTRAL_SUFFIXES:
        return "dynamics"
    if param.ndim <= 1:
        return "no_decay"
    return "main"


def ssm_three_group_params(
    named_params: Iterable[tuple[str, Tensor]],
    *,
    base_lr: float,
    weight_decay: float,
    dynamics_lr_mul: float = 0.1,
) -> list[dict[str, Any]]:
    """Build three parameter groups for SSM-aware optimization.

    Returns a list of ``torch.optim.Optimizer`` param-group dicts:

      * ``dynamics``: spectral parameters (``log_a``, etc.).
        ``lr = base_lr * dynamics_lr_mul``, ``wd = 0``.
      * ``no_decay``: 1D non-spectral parameters (norm gains, biases).
        ``lr = base_lr``, ``wd = 0``.
      * ``main``: matrix weights. ``lr = base_lr``, ``wd = weight_decay``.

    Each group sets both ``weight_decay`` and ``adamw_weight_decay``
    (and ``lr`` / ``adamw_lr``) so optimizers with an AdamW fallback path
    (Muon, SemanticOptimizer, ScarcityAwareOptimizer) honor the split on
    both the matrix branch and the non-matrix branch. Empty groups are
    dropped so torch doesn't iterate over them.

    Parameters with ``requires_grad=False`` are silently skipped — they
    don't belong in any optimizer group.
    """
    buckets: dict[str, list[Tensor]] = {
        "dynamics": [],
        "no_decay": [],
        "main": [],
    }
    for name, param in named_params:
        if not param.requires_grad:
            continue
        buckets[classify_param(name, param)].append(param)

    dynamics_lr = float(base_lr) * float(dynamics_lr_mul)
    base_lr_f = float(base_lr)
    wd_f = float(weight_decay)

    groups: list[dict[str, Any]] = [
        {
            "name": "dynamics",
            "params": buckets["dynamics"],
            "lr": dynamics_lr,
            "adamw_lr": dynamics_lr,
            "weight_decay": 0.0,
            "adamw_weight_decay": 0.0,
        },
        {
            "name": "no_decay",
            "params": buckets["no_decay"],
            "lr": base_lr_f,
            "adamw_lr": base_lr_f,
            "weight_decay": 0.0,
            "adamw_weight_decay": 0.0,
        },
        {
            "name": "main",
            "params": buckets["main"],
            "lr": base_lr_f,
            "adamw_lr": base_lr_f,
            "weight_decay": wd_f,
            "adamw_weight_decay": wd_f,
        },
    ]
    return [group for group in groups if group["params"]]


def build_optimizer_params(
    named_params: Iterable[tuple[str, Tensor]],
    *,
    grouping: str,
    base_lr: float,
    weight_decay: float,
    dynamics_lr_mul: float = 0.1,
) -> list[Tensor] | list[dict[str, Any]]:
    """Dispatch on ``grouping``: ``"flat"`` returns a param list; any
    SSM-aware mode returns a param-group list.

    Supported modes:
      * ``"flat"`` — single group (back-compat with every Exp18–23 run)
      * ``"ssm_three_group"`` — dynamics / no_decay / main split

    Unknown modes raise ``ValueError`` rather than silently falling back,
    so a typo in a config file fails loudly before 8×H100 time is spent.
    """
    mode = str(grouping).strip().lower()
    if mode == "flat":
        return [param for _, param in named_params if param.requires_grad]
    if mode == "ssm_three_group":
        return ssm_three_group_params(
            named_params,
            base_lr=base_lr,
            weight_decay=weight_decay,
            dynamics_lr_mul=dynamics_lr_mul,
        )
    raise ValueError(
        f"unknown optimizer_param_grouping: {grouping!r}; "
        f"expected one of {{'flat', 'ssm_three_group'}}"
    )
