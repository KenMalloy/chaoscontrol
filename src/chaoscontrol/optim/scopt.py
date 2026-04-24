"""Scarcity-aware optimizer.

ScOpt is a Muon-compatible optimizer with optional rare-event state:

* detached event pressure from per-token CE,
* rare-gradient EMA per parameter,
* rare/common orthogonal decomposition against Muon's actual common
  Nesterov direction,
* row-scarcity pre-scaling for token-indexed matrices, and
* optional two-sided channel scarcity for matrix parameters.

When no scarcity state is supplied, the update intentionally reduces to
``Muon`` for matrices and the same inline AdamW fallback for non-matrices.
That reduction is important: ablations should isolate scarcity mechanics,
not a hidden optimizer rewrite.
"""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any, Callable

import torch
from torch import Tensor

from chaoscontrol.optim.muon import newton_schulz_orthogonalize


class FrequencyBucketBaseline:
    """Per-frequency-bucket running-mean CE baseline.

    Fallback baseline named in the design spec (line 197): bucket tokens
    by log-frequency, maintain an EMA of mean per-token CE per bucket,
    return that EMA indexed by target token. Deliberately simpler and
    noisier than the attention baseline — good enough to test the rest
    of the mechanism while the attention artifact is being built.

    Pressure sparsity under this baseline will not hit the design's
    5-25% target; the attention baseline is what calibrates sparsity.
    Sparsity is tracked as diagnostic telemetry regardless.
    """

    def __init__(
        self,
        token_frequencies: Tensor,
        *,
        num_buckets: int = 16,
        decay: float = 0.99,
        device: torch.device | None = None,
    ) -> None:
        if token_frequencies.ndim != 1:
            raise ValueError("token_frequencies must be 1D")
        if num_buckets < 1:
            raise ValueError(f"num_buckets must be >= 1, got {num_buckets}")
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must be in [0, 1), got {decay}")
        target_device = device if device is not None else token_frequencies.device
        log_freq = torch.log1p(
            token_frequencies.to(dtype=torch.float32).clamp_min(0.0)
        )
        min_lf = float(log_freq.min().item())
        max_lf = float(log_freq.max().item())
        span = max(max_lf - min_lf, 1e-6)
        edges = torch.linspace(
            min_lf,
            min_lf + span + 1e-6,
            num_buckets + 1,
            device=log_freq.device,
            dtype=torch.float32,
        )
        bucket = torch.bucketize(log_freq, edges[1:-1])
        bucket.clamp_(0, num_buckets - 1)
        self._token_bucket = bucket.to(device=target_device)
        self._ema = torch.zeros(
            num_buckets, device=target_device, dtype=torch.float32
        )
        self._initialized = torch.zeros(
            num_buckets, device=target_device, dtype=torch.bool
        )
        self._decay = float(decay)
        self._num_buckets = int(num_buckets)

    def baseline(self, targets: Tensor) -> Tensor:
        """Return baseline CE for each target position."""
        return self._ema[self._token_bucket[targets]]

    def update(self, ce: Tensor, targets: Tensor) -> None:
        """Update per-bucket running-mean CE from an observed batch."""
        with torch.no_grad():
            ce_flat = ce.detach().to(dtype=torch.float32, device=self._ema.device).reshape(-1)
            bucket_flat = self._token_bucket[targets.reshape(-1).to(device=self._ema.device)]
            bucket_sum = torch.zeros_like(self._ema)
            bucket_count = torch.zeros_like(self._ema)
            bucket_sum.scatter_add_(0, bucket_flat, ce_flat)
            bucket_count.scatter_add_(0, bucket_flat, torch.ones_like(ce_flat))
            seen = bucket_count > 0
            observed = torch.where(
                seen, bucket_sum / bucket_count.clamp_min(1.0), self._ema
            )
            # For buckets seen for the first time, replace EMA outright; for
            # others, smooth with the configured decay.
            first_seen = seen & ~self._initialized
            updated = torch.where(
                first_seen,
                observed,
                self._decay * self._ema + (1.0 - self._decay) * observed,
            )
            # Buckets never yet observed keep their prior value (zero at init).
            self._ema = torch.where(seen, updated, self._ema)
            self._initialized = self._initialized | seen

    def state_dict(self) -> dict[str, Any]:
        return {
            "ema": self._ema.detach().cpu().clone(),
            "initialized": self._initialized.detach().cpu().clone(),
            "token_bucket": self._token_bucket.detach().cpu().clone(),
            "decay": self._decay,
            "num_buckets": self._num_buckets,
        }


def scarcity_pressure_from_ce(
    ce: Tensor,
    targets: Tensor,
    *,
    token_frequencies: Tensor,
    baseline: Tensor | float | None = None,
    eps: float = 1e-8,
    upper_c: float | None = None,
    upper_floor: float = 1.0,
) -> Tensor:
    """Return detached rare-event pressure for unreduced CE.

    ``ce`` remains the differentiable loss tensor used by the caller.
    Pressure is a constant weight derived from ``ce.detach()`` so
    ``rare_loss = (ce * pressure).sum() / pressure.sum()`` backprops
    through CE only, not through the pressure heuristic itself.

    ``upper_c`` (optional) enables a Gerber-style upper clamp on
    pressure: ``pressure.clamp_max(max(upper_c * pressure_std, upper_floor))``.
    This is the pressure-level counterpart to the per-factor Gerber
    gates in the optimizer's ``_scarcity_factor``. Without it, a small
    number of exploded-CE tokens can drive the per-batch pressure far
    out of its typical range, destabilising the rare-gradient
    accumulation (observed in the 2026-04-24 ScOpt smoke, with
    ``pressure.max=172`` vs ``p95=21``). ``upper_c=None`` keeps the
    historical no-clamp behavior.
    """
    if token_frequencies.ndim != 1:
        raise ValueError(
            "token_frequencies must be a 1D tensor indexed by token id"
        )
    if ce.shape != targets.shape:
        raise ValueError(
            f"ce and targets must have matching shape, got "
            f"{tuple(ce.shape)} and {tuple(targets.shape)}"
        )
    with torch.no_grad():
        freqs = token_frequencies.to(device=targets.device, dtype=torch.float32)
        rarity = 1.0 / torch.log1p(freqs[targets].clamp_min(float(eps)))
        ce_detached = ce.detach().float()
        if baseline is None:
            baseline_tensor = torch.zeros_like(ce_detached)
        elif isinstance(baseline, Tensor):
            baseline_tensor = baseline.to(
                device=ce.device,
                dtype=torch.float32,
            )
            if baseline_tensor.ndim == 0:
                baseline_tensor = torch.full_like(
                    ce_detached,
                    float(baseline_tensor),
                )
            elif baseline_tensor.shape != ce.shape:
                raise ValueError(
                    f"baseline must match ce shape, got "
                    f"{tuple(baseline_tensor.shape)} and {tuple(ce.shape)}"
                )
        else:
            baseline_tensor = torch.full_like(ce_detached, float(baseline))
        excess = (ce_detached - baseline_tensor).clamp_min(0.0)
        pressure = rarity.to(dtype=torch.float32) * excess
        if upper_c is not None:
            # Robust scale: use 95%-winsorized std so the outliers we want
            # to clamp don't themselves inflate the threshold they'd be
            # caught by. For tiny tensors (<10 elements) fall back to the
            # plain std since quantile is ill-defined.
            if pressure.numel() >= 10:
                q95 = pressure.flatten().quantile(0.95)
                pressure_scale = pressure.clamp_max(q95).std(unbiased=False)
            elif pressure.numel() > 1:
                pressure_scale = pressure.std(unbiased=False)
            else:
                pressure_scale = pressure.new_zeros(())
            upper = torch.maximum(
                pressure_scale * float(upper_c),
                pressure.new_tensor(float(upper_floor)),
            )
            pressure = pressure.clamp_max(upper)
    return pressure.to(device=ce.device, dtype=torch.float32)


def _default_is_matrix(param: Tensor, name: str | None) -> bool:  # noqa: ARG001
    return param.ndim == 2


def _as_name_mapping(
    values: Mapping[str, Tensor] | Mapping[Tensor, Tensor],
    *,
    name_of: Callable[[Tensor], str | None],
) -> dict[str, Tensor]:
    mapped: dict[str, Tensor] = {}
    for key, value in values.items():
        if isinstance(key, str):
            mapped[key] = value
        elif isinstance(key, Tensor):
            name = name_of(key)
            if name is None:
                raise KeyError(
                    "tensor-keyed scarcity mappings require bind_param_names()"
                )
            mapped[name] = value
        else:
            raise TypeError(
                "scarcity mappings must be keyed by parameter name or Tensor"
            )
    return mapped


class ScarcityAwareOptimizer(torch.optim.Optimizer):
    """Muon with rare-event gradient preservation and scarcity geometry.

    The training loop owns the expensive autograd work. It should:

    1. compute common gradients and put them into ``p.grad``;
    2. periodically call :meth:`update_rare_grad_ema` with rare gradients;
    3. optionally update row/channel scarcity state; and
    4. call :meth:`step`.

    This keeps the optimizer reusable across the slow correctness path and
    future fused/streaming training paths.
    """

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        *,
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
        rare_ema_decay: float = 0.9,
        rare_orthogonal_weight: float = 1.0,
        rare_macro_c: float = 0.5,
        warmup_steps: int = 200,
        row_param_names: set[str] | None = None,
        row_scarcity_power: float = 0.5,
        matrix_scarcity_map: Mapping[str, tuple[str | None, str | None]] | None = None,
        recurrence_scarcity_map: Mapping[str, str] | None = None,
        recurrence_timescale: Mapping[str, Tensor] | None = None,
        recurrence_weight: float = 1.0,
        tau_std_scale: float = 0.5,
        tau_out_floor: float = 1e-4,
        tau_in_floor: float = 1e-4,
        tau_row_floor: float = 1e-4,
        eps: float = 1e-8,
    ) -> None:
        if lr <= 0.0 or ns_steps <= 0 or not 0.0 <= momentum < 1.0:
            raise ValueError(
                f"invalid ScOpt hparams: lr={lr} momentum={momentum} "
                f"ns_steps={ns_steps}"
            )
        if not 0.0 <= rare_ema_decay < 1.0:
            raise ValueError(
                f"rare_ema_decay must be in [0, 1), got {rare_ema_decay}"
            )
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            adamw_lr=lr if adamw_lr is None else adamw_lr,
            adamw_weight_decay=(
                weight_decay if adamw_weight_decay is None else adamw_weight_decay
            ),
        )
        super().__init__(params, defaults)
        self._matrix_param_names = (
            set(matrix_param_names) if matrix_param_names is not None else None
        )
        self._is_matrix_fn = is_matrix if is_matrix is not None else _default_is_matrix
        self._compute_dtype = compute_dtype
        self._rare_ema_decay = float(rare_ema_decay)
        self._rare_orthogonal_weight = float(rare_orthogonal_weight)
        self._rare_macro_c = float(rare_macro_c)
        if self._rare_macro_c < 0.0:
            raise ValueError(
                f"rare_macro_c must be >= 0, got {self._rare_macro_c}"
            )
        self._warmup_steps = int(warmup_steps)
        self._row_param_names = set(row_param_names or {"embed.weight", "lm_head.weight"})
        self._row_scarcity_power = float(row_scarcity_power)
        self._matrix_scarcity_map = dict(matrix_scarcity_map or {})
        self._recurrence_scarcity_map = dict(recurrence_scarcity_map or {})
        self._recurrence_timescale = dict(recurrence_timescale or {})
        self._recurrence_weight = float(recurrence_weight)
        self._tau_std_scale = float(tau_std_scale)
        self._tau_out_floor = float(tau_out_floor)
        self._tau_in_floor = float(tau_in_floor)
        self._tau_row_floor = float(tau_row_floor)
        self._eps = float(eps)
        self._param_name_by_id: dict[int, str] = {}
        self._row_pressure_ema: Tensor | None = None
        self._channel_pressure: dict[str, Tensor] = {}
        self._step_count = 0
        self._telemetry_accum: dict[str, list[float]] = defaultdict(list)
        self._pressure_stats: dict[str, float] | None = None
        self._clip_events = 0
        self._clip_observations = 0

    def bind_param_names(self, named_params: Iterable[tuple[str, Tensor]]) -> None:
        """Attach names so role maps can target parameters."""
        self._param_name_by_id = {id(p): n for n, p in named_params}

    def _name_of(self, p: Tensor) -> str | None:
        return self._param_name_by_id.get(id(p))

    def _is_matrix_param(self, p: Tensor) -> bool:
        name = self._name_of(p)
        if self._matrix_param_names is not None and name is not None:
            return name in self._matrix_param_names
        return self._is_matrix_fn(p, name)

    def _scarcity_enabled(self) -> bool:
        return self._step_count > self._warmup_steps

    def _writes_enabled(self) -> bool:
        """Gate for EMA/state writes.

        Writes happen BEFORE ``step()`` increments ``_step_count``, so the
        pre-step counter is ``step_count``. We suppress writes during the
        first ``warmup_steps`` training steps and begin accumulating at
        step ``warmup_steps + 1`` (when pre-step counter equals
        ``warmup_steps``). This mirrors the read gate's symmetry: by the
        time ``_scarcity_enabled`` turns True inside ``step()``, the EMA
        has seen exactly one warm entry.
        """
        return self._step_count >= self._warmup_steps

    def set_rare_grad_ema(
        self,
        rare_grads: Mapping[str, Tensor] | Mapping[Tensor, Tensor],
    ) -> None:
        """Set rare-gradient EMA buffers directly.

        Useful for tests, checkpoints, and offline probes where the caller
        has already smoothed rare gradients.
        """
        mapped = _as_name_mapping(rare_grads, name_of=self._name_of)
        for group in self.param_groups:
            for p in group["params"]:
                name = self._name_of(p)
                if name is None or name not in mapped:
                    continue
                self.state[p]["rare_grad_ema"] = mapped[name].detach().to(
                    device=p.device,
                    dtype=p.dtype,
                ).clone()

    def update_rare_grad_ema(
        self,
        rare_grads: Mapping[str, Tensor] | Mapping[Tensor, Tensor],
        *,
        skip: bool = False,
    ) -> None:
        """Update rare-gradient EMA from a split-step rare backward pass.

        Silently no-ops during warmup (see :meth:`_writes_enabled`) and
        when ``skip=True`` (caller's grad-clip fired). Either gate means
        the EMA shouldn't see the current batch's rare signal.
        """
        if skip or not self._writes_enabled():
            return
        mapped = _as_name_mapping(rare_grads, name_of=self._name_of)
        decay = self._rare_ema_decay
        for group in self.param_groups:
            for p in group["params"]:
                name = self._name_of(p)
                if name is None or name not in mapped:
                    continue
                rare = mapped[name].detach().to(device=p.device, dtype=p.dtype)
                state = self.state[p]
                if "rare_grad_ema" not in state:
                    state["rare_grad_ema"] = torch.zeros_like(
                        p,
                        memory_format=torch.preserve_format,
                    )
                state["rare_grad_ema"].mul_(decay).add_(rare, alpha=1.0 - decay)

    def set_row_pressure_ema(self, row_pressure: Tensor | None) -> None:
        """Set token-row pressure EMA directly.

        Warmup-gated for symmetry with :meth:`update_row_pressure_ema`;
        both the populate case and the explicit clear case are
        suppressed during warmup so a caller can't accidentally bypass
        the gate via either path.
        """
        if not self._writes_enabled():
            return
        self._row_pressure_ema = (
            None if row_pressure is None else row_pressure.detach().float().clone()
        )

    def update_row_pressure_ema(
        self,
        targets: Tensor,
        pressure: Tensor,
        *,
        vocab_size: int,
        decay: float | None = None,
    ) -> Tensor | None:
        """Scatter per-token pressure into a per-vocab-row EMA.

        Skipped during warmup so the EMA starts clean at step
        ``warmup_steps + 1``. Returns ``None`` when the write is
        suppressed so callers don't attempt to all-reduce a missing
        tensor.
        """
        if not self._writes_enabled():
            return None
        if targets.shape != pressure.shape:
            raise ValueError(
                f"targets and pressure must match, got "
                f"{tuple(targets.shape)} and {tuple(pressure.shape)}"
            )
        if self._row_pressure_ema is None:
            self._row_pressure_ema = torch.zeros(
                int(vocab_size),
                dtype=torch.float32,
                device=pressure.device,
            )
        elif self._row_pressure_ema.numel() != int(vocab_size):
            raise ValueError(
                f"existing row pressure has {self._row_pressure_ema.numel()} "
                f"rows, got vocab_size={vocab_size}"
            )
        row_sum = torch.zeros_like(self._row_pressure_ema, device=pressure.device)
        row_count = torch.zeros_like(row_sum)
        flat_targets = targets.reshape(-1).to(device=pressure.device)
        flat_pressure = pressure.detach().float().reshape(-1)
        row_sum.scatter_add_(0, flat_targets, flat_pressure)
        row_count.scatter_add_(0, flat_targets, torch.ones_like(flat_pressure))
        observed = row_sum / row_count.clamp_min(1.0)
        mask = row_count > 0
        ema_decay = self._rare_ema_decay if decay is None else float(decay)
        current = self._row_pressure_ema.to(device=pressure.device)
        current[mask] = current[mask] * ema_decay + observed[mask] * (1.0 - ema_decay)
        self._row_pressure_ema = current.detach()
        return self._row_pressure_ema

    def set_channel_pressure(
        self,
        channel_pressure: Mapping[str, Tensor],
    ) -> None:
        """Set per-channel pressure vectors used by matrix/recurrence maps.

        Warmup-gated to stay consistent with the rare-EMA write policy.
        """
        if not self._writes_enabled():
            return
        self._channel_pressure = {
            key: value.detach().float().clone()
            for key, value in channel_pressure.items()
        }

    def set_recurrence_timescale(
        self,
        recurrence_timescale: Mapping[str, Tensor],
    ) -> None:
        self._recurrence_timescale = {
            key: value.detach().float().clone()
            for key, value in recurrence_timescale.items()
        }

    def _scarcity_factor(
        self,
        pressure: Tensor,
        *,
        floor: float,
        device: torch.device,
        dtype: torch.dtype,
        telemetry_key: str | None = None,
    ) -> Tensor:
        pressure_f = pressure.detach().to(device=device, dtype=torch.float32).clamp_min(0.0)
        if pressure_f.numel() == 0:
            return pressure_f.to(dtype=dtype)
        std = pressure_f.std(unbiased=False)
        tau = torch.maximum(
            std * self._tau_std_scale,
            torch.tensor(float(floor), device=device, dtype=torch.float32),
        )
        factor = torch.tanh(pressure_f / tau.clamp_min(self._eps)) + 1.0
        if telemetry_key is not None:
            self._telemetry_accum[f"{telemetry_key}_min"].append(float(factor.min().item()))
            self._telemetry_accum[f"{telemetry_key}_median"].append(float(factor.median().item()))
            self._telemetry_accum[f"{telemetry_key}_max"].append(float(factor.max().item()))
        return factor.to(dtype=dtype)

    def _rare_adjusted_direction(
        self,
        *,
        p: Tensor,
        common: Tensor,
    ) -> Tensor:
        if not self._scarcity_enabled():
            return common
        rare = self.state[p].get("rare_grad_ema")
        if rare is None:
            return common
        orig_dtype = common.dtype
        common_f = common.to(dtype=torch.float32)
        rare_f = rare.to(device=common.device, dtype=torch.float32)
        common_sq = common_f.square().sum().clamp_min(self._eps)
        common_norm = common_sq.sqrt()
        rare_norm = rare_f.square().sum().clamp_min(self._eps).sqrt()
        parallel = (rare_f.mul(common_f).sum() / common_sq) * common_f
        orthogonal = rare_f - parallel
        orth_norm = orthogonal.square().sum().clamp_min(0.0).sqrt()

        # Record alignment diagnostics (fp32 domain) for null-result
        # interpretability — matches the design's diagnostics contract.
        cos_rc = float(
            (rare_f.mul(common_f).sum() / (rare_norm * common_norm))
            .clamp(-1.0, 1.0)
            .item()
        )
        self._telemetry_accum["cos_rare_common"].append(cos_rc)
        self._telemetry_accum["r_orth_over_common"].append(
            float((orth_norm / common_norm).item())
        )

        # Macro Gerber cap: bound the orthogonal contribution's norm to
        # ``c_macro * common_norm`` so a rare-grad EMA that outgrows the
        # common gradient can never dominate the update direction. The
        # existing per-factor Gerber gates sit inside ``_scarcity_factor``
        # and are irrelevant here — this path composes direction vectors,
        # not per-channel amplifiers. The missing guard at this site is
        # what drove the 2026-04-24 smoke to final_loss 158.
        #
        # ``rare_macro_c=0`` disables the cap (historical behavior).
        desired_weight = float(self._rare_orthogonal_weight)
        if self._rare_macro_c > 0.0 and desired_weight > 0.0:
            cap = (self._rare_macro_c * common_norm) / orth_norm.clamp_min(self._eps)
            effective_weight = torch.minimum(
                cap,
                cap.new_tensor(desired_weight),
            )
            ew_float = float(effective_weight.item())
            self._telemetry_accum["rare_macro_effective_weight"].append(ew_float)
            self._telemetry_accum["rare_macro_cap_fired"].append(
                1.0 if ew_float < desired_weight - 1e-6 else 0.0
            )
            direction = common_f + effective_weight * orthogonal
        else:
            direction = common_f + desired_weight * orthogonal
        return direction.to(dtype=orig_dtype)

    def _apply_matrix_scarcity(
        self, name: str | None, direction: Tensor
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        """Returns (scaled_direction, out_factor, in_factor)."""
        if not self._scarcity_enabled() or name is None:
            return direction, None, None
        out_key: str | None
        in_key: str | None
        out_key, in_key = self._matrix_scarcity_map.get(name, (None, None))
        out_factor = None
        in_factor = None
        if out_key is not None and out_key in self._channel_pressure:
            out_factor = self._scarcity_factor(
                self._channel_pressure[out_key],
                floor=self._tau_out_floor,
                device=direction.device,
                dtype=direction.dtype,
                telemetry_key="out_scarcity",
            )
            if out_factor.numel() != direction.size(0):
                raise ValueError(
                    f"out scarcity {out_key!r} has {out_factor.numel()} "
                    f"channels for {name!r} rows={direction.size(0)}"
                )
        if in_key is not None and in_key in self._channel_pressure:
            in_factor = self._scarcity_factor(
                self._channel_pressure[in_key],
                floor=self._tau_in_floor,
                device=direction.device,
                dtype=direction.dtype,
                telemetry_key="in_scarcity",
            )
            if in_factor.numel() != direction.size(1):
                raise ValueError(
                    f"in scarcity {in_key!r} has {in_factor.numel()} "
                    f"channels for {name!r} cols={direction.size(1)}"
                )

        scaled = direction
        if out_factor is not None:
            scaled = out_factor.sqrt()[:, None] * scaled
        if in_factor is not None:
            scaled = scaled * in_factor.sqrt()[None, :]
        return scaled, out_factor, in_factor

    def _apply_row_scarcity(
        self, name: str | None, direction: Tensor
    ) -> tuple[Tensor, Tensor | None]:
        """Returns (scaled_direction, row_factor)."""
        if (
            not self._scarcity_enabled()
            or name is None
            or name not in self._row_param_names
            or self._row_pressure_ema is None
        ):
            return direction, None
        if self._row_pressure_ema.numel() != direction.size(0):
            raise ValueError(
                f"row pressure has {self._row_pressure_ema.numel()} rows for "
                f"{name!r} rows={direction.size(0)}"
            )
        row_factor = self._scarcity_factor(
            self._row_pressure_ema,
            floor=self._tau_row_floor,
            device=direction.device,
            dtype=direction.dtype,
            telemetry_key="row_scarcity",
        ).pow(self._row_scarcity_power)
        return row_factor[:, None] * direction, row_factor

    def _record_energy_enrichment(
        self,
        update: Tensor,
        *,
        row_factor: Tensor | None,
        col_factor: Tensor | None,
    ) -> None:
        """Record post-NS scarce-vs-common row/col energy enrichment and
        log-space correlation between pre-NS scarcity factor and post-NS
        row/col norm.

        Spec lines 278-281: NS normalizes away diagonal amplitude, so we
        must measure energy on the NS *output* to verify scarce
        rows/columns actually carry disproportionate update energy.

        ``ns_row_factor_corr`` / ``ns_col_factor_corr`` answer a sharper
        question that the median-split enrichment ratio leaves
        underspecified: does the pre-NS multiplier *survive in
        magnitude* through Newton-Schulz, or does NS whiten it toward
        uniform? Correlation near 1 means pre-NS scaling is preserved
        and a post-NS pressure-selected tail bypass has nothing to
        recover. Correlation near 0 means NS flattens the signal and a
        magnitude-preserving bypass mechanism has a real lever.
        """
        if update.ndim != 2:
            return
        update_f = update.detach().to(dtype=torch.float32)
        frob_sq = update_f.square().sum().clamp_min(self._eps)
        if row_factor is not None and row_factor.numel() == update.size(0):
            factor = row_factor.detach().to(device=update.device, dtype=torch.float32)
            row_energy = update_f.square().sum(dim=1) / frob_sq
            median = factor.median()
            scarce = factor >= median
            common = ~scarce
            if scarce.any() and common.any():
                ratio = (
                    row_energy[scarce].mean() / row_energy[common].mean().clamp_min(self._eps)
                )
                self._telemetry_accum["ns_row_energy_enrichment"].append(float(ratio.item()))
            self._record_factor_corr(
                factor=factor,
                energy=row_energy,
                telemetry_key="ns_row_factor_corr",
            )
        if col_factor is not None and col_factor.numel() == update.size(1):
            factor = col_factor.detach().to(device=update.device, dtype=torch.float32)
            col_energy = update_f.square().sum(dim=0) / frob_sq
            median = factor.median()
            scarce = factor >= median
            common = ~scarce
            if scarce.any() and common.any():
                ratio = (
                    col_energy[scarce].mean() / col_energy[common].mean().clamp_min(self._eps)
                )
                self._telemetry_accum["ns_col_energy_enrichment"].append(float(ratio.item()))
            self._record_factor_corr(
                factor=factor,
                energy=col_energy,
                telemetry_key="ns_col_factor_corr",
            )

    def _record_factor_corr(
        self,
        *,
        factor: Tensor,
        energy: Tensor,
        telemetry_key: str,
    ) -> None:
        """Emit log-space Pearson correlation between pre-NS scarcity
        factor and post-NS row/col norm (sqrt of normalized energy).

        Uses log on both sides so the correlation reads as "what
        fraction of a multiplicative pre-NS scaling survived NS". Skips
        emission when either side has zero variance — correlation is
        undefined and would noise the telemetry.
        """
        log_factor = factor.clamp_min(self._eps).log()
        log_norm = energy.clamp_min(self._eps).log() * 0.5  # log(sqrt) = 0.5*log
        lf_std = log_factor.std(unbiased=False)
        ln_std = log_norm.std(unbiased=False)
        if lf_std <= self._eps or ln_std <= self._eps:
            return
        lf_centered = log_factor - log_factor.mean()
        ln_centered = log_norm - log_norm.mean()
        corr = (lf_centered * ln_centered).mean() / (lf_std * ln_std).clamp_min(self._eps)
        self._telemetry_accum[telemetry_key].append(
            float(corr.clamp(-1.0, 1.0).item())
        )

    def _apply_recurrence_scarcity(self, name: str | None, direction: Tensor) -> Tensor:
        if not self._scarcity_enabled() or name is None:
            return direction
        key = self._recurrence_scarcity_map.get(name)
        if key is None or key not in self._channel_pressure:
            return direction
        pressure = self._channel_pressure[key]
        factor = self._scarcity_factor(
            pressure,
            floor=self._tau_out_floor,
            device=direction.device,
            dtype=direction.dtype,
            telemetry_key="recurrence_scarcity",
        )
        timescale = self._recurrence_timescale.get(key)
        if timescale is not None:
            timescale = timescale.to(device=direction.device, dtype=direction.dtype)
            if timescale.numel() != factor.numel():
                raise ValueError(
                    f"recurrence timescale {key!r} has {timescale.numel()} "
                    f"channels but pressure has {factor.numel()}"
                )
            factor = 1.0 + self._recurrence_weight * (factor - 1.0) * timescale
        if factor.numel() != direction.numel() or factor.shape != direction.shape:
            raise ValueError(
                f"recurrence scarcity for {name!r} has shape "
                f"{tuple(factor.shape)} but direction has shape "
                f"{tuple(direction.shape)}; broadcast is not supported"
            )
        return direction * factor

    def record_pressure_stats(self, stats: Mapping[str, float]) -> None:
        """Record per-step pressure distribution stats from the runner.

        The optimizer itself never sees the raw per-token pressure tensor
        (that lives in the training loop), so the runner passes in a
        summary dict each split step. Keys we expect: ``min``,
        ``median``, ``p95``, ``max``, ``fraction_positive``.
        """
        self._pressure_stats = dict(stats)

    def record_clip_event(self, *, triggered: bool) -> None:
        """Record whether grad clip fired this step (for clip-rate telemetry)."""
        self._clip_observations += 1
        if triggered:
            self._clip_events += 1

    def scarcity_trace(self) -> dict[str, Any]:
        """Diagnostics snapshot aggregating the current accumulator.

        Called periodically by the training loop (every N steps). Resets
        the accumulator so each trace covers the window since the last
        call. Coverage matches the design's diagnostics contract:
        pressure distribution, rare/common alignment, r_orth magnitude,
        scarcity factor ranges per role, and clip rate.
        """
        trace: dict[str, Any] = {
            "step": self._step_count,
            "scarcity_enabled": self._scarcity_enabled(),
        }
        if self._row_pressure_ema is not None:
            row = self._row_pressure_ema.detach().float().cpu()
            trace["row_pressure"] = {
                "min": float(row.min()),
                "median": float(row.median()),
                "max": float(row.max()),
            }
        if self._channel_pressure:
            trace["channel_pressure_keys"] = sorted(self._channel_pressure)
        if self._pressure_stats is not None:
            trace["pressure_stats"] = dict(self._pressure_stats)

        def _stats(values: list[float]) -> dict[str, float]:
            if not values:
                return {}
            t = torch.tensor(values, dtype=torch.float32)
            return {
                "min": float(t.min()),
                "median": float(t.median()),
                "max": float(t.max()),
                "count": int(t.numel()),
            }

        for key, values in self._telemetry_accum.items():
            s = _stats(values)
            if s:
                trace[key] = s

        if self._clip_observations > 0:
            trace["clip_rate"] = self._clip_events / self._clip_observations

        self._telemetry_accum = defaultdict(list)
        self._clip_events = 0
        self._clip_observations = 0
        return trace

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor] | None = None) -> Tensor | None:
        loss: Tensor | None = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1

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
                name = self._name_of(p)

                if self._is_matrix_param(p):
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(
                            p,
                            memory_format=torch.preserve_format,
                        )
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(grad)
                    common = grad.add(buf, alpha=momentum) if nesterov else buf
                    direction = self._rare_adjusted_direction(p=p, common=common)
                    direction, out_factor, in_factor = self._apply_matrix_scarcity(
                        name, direction
                    )
                    direction, row_factor = self._apply_row_scarcity(name, direction)
                    update = newton_schulz_orthogonalize(
                        direction,
                        steps=ns_steps,
                        compute_dtype=self._compute_dtype,
                    )
                    # Scarce-row/col energy enrichment after NS — prefer
                    # row_scarcity when both are present (both target the
                    # same axis-0 row dimension).
                    effective_row = row_factor if row_factor is not None else out_factor
                    if effective_row is not None or in_factor is not None:
                        self._record_energy_enrichment(
                            update,
                            row_factor=effective_row,
                            col_factor=in_factor,
                        )
                    rows, cols = p.shape[-2], p.shape[-1]
                    scale = max(1.0, rows / cols) ** 0.5
                    if wd > 0.0:
                        p.data.mul_(1.0 - lr * wd)
                    p.data.add_(update.to(dtype=p.dtype), alpha=-lr * scale)
                    continue

                grad_for_adam = grad
                common = grad
                adjusted = self._rare_adjusted_direction(p=p, common=common)
                adjusted = self._apply_recurrence_scarcity(name, adjusted)
                grad_for_adam = adjusted
                if "step" not in state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        p,
                        memory_format=torch.preserve_format,
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p,
                        memory_format=torch.preserve_format,
                    )
                state["step"] += 1
                t = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(grad_for_adam, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(
                    grad_for_adam,
                    grad_for_adam,
                    value=1.0 - beta2,
                )
                bias1 = 1.0 - beta1 ** t
                bias2 = 1.0 - beta2 ** t
                denom = (exp_avg_sq.sqrt() / (bias2 ** 0.5)).add_(adamw_eps)
                if adamw_wd > 0.0:
                    p.data.mul_(1.0 - adamw_lr * adamw_wd)
                p.data.addcdiv_(exp_avg, denom, value=-adamw_lr / bias1)

        return loss
