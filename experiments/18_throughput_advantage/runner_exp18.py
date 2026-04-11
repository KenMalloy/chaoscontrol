#!/usr/bin/env python3
"""Single-run engine for Experiment 18.

Experiment-local by design: coverage planning, phase scheduling, frozen
rescoring, and subset selection live here instead of in shared training
infrastructure.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml

REPO = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments" / "15_chaospiece"))

from chaoscontrol.data import (  # noqa: E402
    batch_from_starts,
    build_lm_starts,
    choose_eval_starts,
    maybe_autocast,
    maybe_sync_cuda,
    resolve_device,
    resolve_param_dtype,
)
from chaoscontrol.core import verify_diag_recurrence  # noqa: E402
from chaoscontrol.evaluation import compute_bpb  # noqa: E402
from runner_exp15 import (  # noqa: E402
    build_model,
    build_sentencepiece_luts,
    evaluate_bpb_sp,
    load_sp_data,
)


DEFAULT_SWEEP_BUDGET_S = 400.0
DEFAULT_RESCORE_BUDGET_S = 45.0
DEFAULT_SELECTION_BUDGET_S = 5.0
DEFAULT_TOTAL_BUDGET_S = 600.0
DEFAULT_BASE_LR = 2e-3
LOW_COVERAGE_FLOOR = 0.25
SMOKE_TOTAL_BUDGET_S = 2.0
MIN_PYTHON = (3, 10)


@dataclass
class PhaseBudget:
    total_s: float
    sweep_s: float = 0.0
    rescore_s: float = 0.0
    subset_s: float = 0.0
    retarget_s: float = 0.0

    @property
    def remaining_s(self) -> float:
        return max(
            0.0,
            self.total_s - self.sweep_s - self.rescore_s - self.subset_s - self.retarget_s,
        )

    @property
    def selection_overhead_s(self) -> float:
        return self.rescore_s + self.subset_s


@dataclass
class CoveragePlan:
    total_windows: int
    planned_windows: int
    unique_targets: int
    coverage_frac: float
    low_coverage_regime: bool


def resolve_visible_cuda_devices(env: dict[str, str] | None = None) -> list[str]:
    env_map = os.environ if env is None else env
    mask = env_map.get("CUDA_VISIBLE_DEVICES", "").strip()
    if mask:
        return [piece.strip() for piece in mask.split(",") if piece.strip()]
    if torch.cuda.is_available():
        return [str(i) for i in range(torch.cuda.device_count())]
    return []


def validate_gpu_concurrency(num_gpus: int, env: dict[str, str] | None = None) -> list[str]:
    if num_gpus <= 0:
        raise ValueError(f"num_gpus must be positive, got {num_gpus}")
    visible = resolve_visible_cuda_devices(env)
    if not visible:
        raise RuntimeError(
            "No visible CUDA devices. Check the pod allocation, CUDA_VISIBLE_DEVICES, and driver/runtime setup."
        )
    if num_gpus > len(visible):
        raise RuntimeError(
            f"Requested num_gpus={num_gpus}, but only {len(visible)} CUDA slots are visible "
            f"({','.join(visible)})."
        )
    return visible


def build_child_env(
    *,
    gpu_slot: int | None,
    smoke: bool,
    base_env: dict[str, str] | None = None,
) -> dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    if smoke or gpu_slot is None:
        return env
    visible = resolve_visible_cuda_devices(env)
    if gpu_slot < 0 or gpu_slot >= len(visible):
        raise RuntimeError(
            f"GPU slot {gpu_slot} is out of range for visible CUDA devices {visible}."
        )
    env["CUDA_VISIBLE_DEVICES"] = visible[gpu_slot]
    return env


def assert_runtime_compatibility(
    *,
    smoke: bool,
    device: torch.device | None = None,
    require_sentencepiece: bool = False,
    sp_model_path: str | None = None,
) -> dict[str, Any]:
    if sys.version_info < MIN_PYTHON:
        raise RuntimeError(
            f"Experiment 18 requires Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+, "
            f"found {sys.version_info.major}.{sys.version_info.minor}."
        )

    info: dict[str, Any] = {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "diag_recurrence": verify_diag_recurrence(),
        "smoke": smoke,
    }
    if smoke:
        return info

    if device is not None and device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested, but torch.cuda.is_available() is false.")
        visible = resolve_visible_cuda_devices()
        info["visible_cuda_devices"] = visible
        info["cuda_device_count"] = torch.cuda.device_count()
        bf16_supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
        info["bf16_supported"] = bf16_supported
        if not bf16_supported:
            raise RuntimeError(
                "CUDA is visible, but torch reports bf16 is unsupported. "
                "Check the H100 pod image, PyTorch build, and driver/runtime compatibility."
            )

    if require_sentencepiece:
        try:
            import sentencepiece as spm
        except Exception as exc:
            raise RuntimeError(
                "sentencepiece is required for Experiment 18 Phase A, but import failed."
            ) from exc
        info["sentencepiece_import"] = "ok"
        if sp_model_path is not None:
            sp = spm.SentencePieceProcessor()
            loaded = sp.Load(sp_model_path)
            if not loaded:
                raise RuntimeError(f"Failed to load SentencePiece model: {sp_model_path}")
            info["sentencepiece_model"] = sp_model_path
    return info


def make_smoke_summary(
    *,
    total_budget_s: float = DEFAULT_TOTAL_BUDGET_S,
) -> dict[str, Any]:
    """Create a tiny synthetic phase-0 summary for smoke tests."""
    return {
        "phase": "phase0",
        "selected": {
            "tokenizer": "sp8192",
            "vocab_size": 64,
            "data_path": "__smoke__",
            "sp_model_path": None,
            "batch_size": 4,
            "base_lr": DEFAULT_BASE_LR,
            "selected_lr": DEFAULT_BASE_LR,
            "step_time_s": 0.01,
            "tokens_per_s": 4000.0,
            "sweep_budget_s": min(DEFAULT_SWEEP_BUDGET_S, total_budget_s * 0.67),
            "rescore_budget_s": min(DEFAULT_RESCORE_BUDGET_S, total_budget_s * 0.08),
            "selection_budget_s": min(DEFAULT_SELECTION_BUDGET_S, total_budget_s * 0.01),
            "retarget_budget_s": max(0.0, total_budget_s - DEFAULT_SWEEP_BUDGET_S - DEFAULT_RESCORE_BUDGET_S - DEFAULT_SELECTION_BUDGET_S),
            "projected_coverage_frac": 0.5,
            "projected_unique_windows": 32,
            "projected_unique_targets": 32 * 16,
            "low_coverage_regime": False,
            "model_config": {
                "model_type": "transformer",
                "vocab_size": 64,
                "model_dim": 16,
                "num_layers": 1,
                "ff_mult": 2,
                "seq_len": 16,
                "stride": 8,
                "batch_size": 4,
                "base_lr": DEFAULT_BASE_LR,
                "a_mode": "diag",
                "crit_target_coupling": 0.92,
            },
        },
    }


def build_unit_luts(vocab_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    base = torch.ones(vocab_size, dtype=torch.int16, device=device)
    has_space = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    is_boundary = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    if vocab_size > 0:
        is_boundary[0] = True
    return base, has_space, is_boundary


def make_smoke_tokens(vocab_size: int = 64, total_tokens: int = 512, seed: int = 1234) -> tuple[torch.Tensor, torch.Tensor]:
    rng = random.Random(seed)
    seq = [rng.randrange(vocab_size) for _ in range(total_tokens)]
    data = torch.tensor(seq, dtype=torch.long)
    split = int(total_tokens * 0.8)
    return data[:split], data[split:]


def generate_sweep_starts(num_tokens: int, seq_len: int, *, max_windows: int | None = None) -> list[int]:
    """Generate non-overlapping sweep starts.

    Coverage is defined as unique next-token prediction targets, so we stop at
    `num_tokens - seq_len - 1` and use stride=seq_len.
    """
    starts = build_lm_starts(num_tokens, seq_len, seq_len)
    if max_windows is not None:
        return starts[: max(max_windows, 0)]
    return starts


def count_unique_targets(starts: list[int], seq_len: int) -> int:
    return len(starts) * seq_len


def build_coverage_plan(
    num_tokens: int,
    *,
    seq_len: int,
    projected_windows: int | None = None,
) -> CoveragePlan:
    total_windows = len(generate_sweep_starts(num_tokens, seq_len))
    planned_windows = total_windows if projected_windows is None else min(projected_windows, total_windows)
    unique_targets = planned_windows * seq_len
    denom = max(total_windows * seq_len, 1)
    coverage_frac = unique_targets / denom
    return CoveragePlan(
        total_windows=total_windows,
        planned_windows=planned_windows,
        unique_targets=unique_targets,
        coverage_frac=coverage_frac,
        low_coverage_regime=coverage_frac < LOW_COVERAGE_FLOOR,
    )


def choose_tokenizer(phase0_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick a tokenizer using coverage-first, quality-second rules."""
    stable = [row for row in phase0_results if row.get("stable", False)]
    if not stable:
        raise ValueError("No stable tokenizer candidates available")

    stable = sorted(
        stable,
        key=lambda row: (
            float(row.get("projected_coverage_frac", 0.0)),
            float(row.get("tokens_per_s", 0.0)),
        ),
        reverse=True,
    )
    best = stable[0]
    if len(stable) == 1:
        return best

    runner_up = stable[1]
    cov_best = float(best.get("projected_coverage_frac", 0.0))
    cov_next = float(runner_up.get("projected_coverage_frac", 0.0))
    relative_gap = (cov_best - cov_next) / max(cov_best, 1e-9) if cov_best > 0 else 0.0

    best_low_cov = bool(best.get("low_coverage_regime", False))
    next_low_cov = bool(runner_up.get("low_coverage_regime", False))
    if best_low_cov != next_low_cov:
        return runner_up if best_low_cov else best

    if relative_gap < 0.10:
        best_quality = float(best.get("prior_mean_bpb", math.inf))
        next_quality = float(runner_up.get("prior_mean_bpb", math.inf))
        if next_quality < best_quality and runner_up.get("tokenizer") == "sp16384":
            return runner_up
        if best.get("tokenizer") == "sp16384":
            return best
    return best


def select_subset(
    scored_windows: list[dict[str, float]],
    *,
    fraction: float,
    mode: str,
    seed: int,
) -> list[int]:
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if not scored_windows:
        return []
    n_keep = max(1, int(len(scored_windows) * fraction))
    if mode == "top":
        ranked = sorted(scored_windows, key=lambda row: row["loss"], reverse=True)
        return [int(row["start"]) for row in ranked[:n_keep]]
    if mode == "random":
        rng = random.Random(seed)
        shuffled = list(scored_windows)
        rng.shuffle(shuffled)
        return [int(row["start"]) for row in shuffled[:n_keep]]
    raise ValueError(f"Unsupported subset mode: {mode}")


def _train_batch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    param_dtype: torch.dtype,
    grad_clip_norm: float,
) -> dict[str, float]:
    vocab_size = model.vocab_size
    optimizer.zero_grad(set_to_none=True)
    autocast_dtype = next(model.parameters()).dtype if device.type == "cuda" else torch.float32
    with maybe_autocast(device, autocast_dtype if device.type == "cuda" else param_dtype):
        out = model(inputs)
        loss = F.cross_entropy(out["logits"].reshape(-1, vocab_size), targets.reshape(-1))
    loss.backward()
    if grad_clip_norm > 0.0:
        torch.nn.utils.clip_grad_norm_(list(model.parameters()), grad_clip_norm)
    optimizer.step()
    return {
        "loss": float(loss.detach().cpu()),
        "tokens": float(targets.numel()),
    }


def train_on_starts(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    tokens: torch.Tensor,
    starts: list[int],
    seq_len: int,
    batch_size: int,
    device: torch.device,
    param_dtype: torch.dtype,
    time_budget_s: float,
    grad_clip_norm: float,
    mode: str,
    seed: int,
    wrap: bool = False,
) -> dict[str, Any]:
    if not starts or time_budget_s <= 0.0:
        return {"steps": 0, "elapsed_s": 0.0, "tokens": 0.0, "loss_ema": None, "seen_starts": []}

    rng = random.Random(seed)
    history: list[float] = []
    seen_starts: list[int] = []
    step_count = 0
    token_count = 0.0
    cursor = 0
    start_time = time.perf_counter()
    model.train()

    while True:
        elapsed = time.perf_counter() - start_time
        if elapsed >= time_budget_s and step_count > 0:
            break

        if mode == "random":
            batch_starts = [starts[rng.randrange(len(starts))] for _ in range(batch_size)]
        else:
            if cursor >= len(starts):
                if not wrap:
                    break
                cursor = 0
            batch_starts = starts[cursor : cursor + batch_size]
            cursor += len(batch_starts)
            if not batch_starts:
                break

        inputs, targets = batch_from_starts(tokens, batch_starts, seq_len, device)
        stats = _train_batch(
            model,
            optimizer,
            inputs=inputs,
            targets=targets,
            device=device,
            param_dtype=param_dtype,
            grad_clip_norm=grad_clip_norm,
        )
        history.append(stats["loss"])
        token_count += stats["tokens"]
        seen_starts.extend(batch_starts)
        step_count += 1

    maybe_sync_cuda(device)
    return {
        "steps": step_count,
        "elapsed_s": float(time.perf_counter() - start_time),
        "tokens": token_count,
        "loss_ema": float(sum(history) / len(history)) if history else None,
        "seen_starts": seen_starts,
    }


def score_windows_by_loss(
    model: torch.nn.Module,
    *,
    tokens: torch.Tensor,
    starts: list[int],
    seq_len: int,
    batch_size: int,
    device: torch.device,
    time_budget_s: float | None = None,
) -> dict[str, Any]:
    if not starts:
        return {"scored": [], "elapsed_s": 0.0, "coverage_scored_frac": 0.0}

    model.eval()
    vocab_size = model.vocab_size
    scored: list[dict[str, float]] = []
    begin = time.perf_counter()
    with torch.no_grad():
        for idx in range(0, len(starts), batch_size):
            if time_budget_s is not None and time.perf_counter() - begin >= time_budget_s and scored:
                break
            batch_starts = starts[idx : idx + batch_size]
            inputs, targets = batch_from_starts(tokens, batch_starts, seq_len, device)
            out = model(inputs)
            per_tok = F.cross_entropy(
                out["logits"].reshape(-1, vocab_size),
                targets.reshape(-1),
                reduction="none",
            ).reshape(inputs.size(0), seq_len)
            per_window = per_tok.mean(dim=1)
            for start, loss in zip(batch_starts, per_window.tolist(), strict=True):
                scored.append({"start": float(start), "loss": float(loss)})
    maybe_sync_cuda(device)
    elapsed_s = float(time.perf_counter() - begin)
    coverage_scored_frac = len(scored) / max(len(starts), 1)
    return {
        "scored": scored,
        "elapsed_s": elapsed_s,
        "coverage_scored_frac": coverage_scored_frac,
    }


def build_phase_a_conditions(
    phase0_summary: dict[str, Any],
    *,
    total_budget_s: float = DEFAULT_TOTAL_BUDGET_S,
) -> dict[str, dict[str, Any]]:
    selected = phase0_summary["selected"]
    model_config = dict(selected["model_config"])
    sweep_batch = int(selected["batch_size"])
    large_lr = float(selected.get("selected_lr", selected.get("base_lr", DEFAULT_BASE_LR)))
    sweep_budget_s = float(selected.get("sweep_budget_s", min(DEFAULT_SWEEP_BUDGET_S, total_budget_s * 0.67)))
    rescore_budget_s = float(selected.get("rescore_budget_s", min(DEFAULT_RESCORE_BUDGET_S, total_budget_s * 0.08)))
    selection_budget_s = float(selected.get("selection_budget_s", min(DEFAULT_SELECTION_BUDGET_S, total_budget_s * 0.01)))
    return {
        "baseline_b32": {
            "condition": "baseline_b32",
            "model_config": dict(model_config, batch_size=32, base_lr=DEFAULT_BASE_LR),
            "train_mode": "baseline",
            "total_budget_s": total_budget_s,
        },
        "sweep_only": {
            "condition": "sweep_only",
            "model_config": dict(model_config, batch_size=sweep_batch, base_lr=large_lr),
            "train_mode": "sweep_only",
            "total_budget_s": total_budget_s,
            "sweep_budget_s": total_budget_s,
        },
        "sweep_target_top10": {
            "condition": "sweep_target_top10",
            "model_config": dict(model_config, batch_size=sweep_batch, base_lr=large_lr),
            "train_mode": "target",
            "subset_fraction": 0.10,
            "subset_mode": "top",
            "total_budget_s": total_budget_s,
            "sweep_budget_s": sweep_budget_s,
            "rescore_budget_s": rescore_budget_s,
            "selection_budget_s": selection_budget_s,
        },
        "sweep_random_retrain": {
            "condition": "sweep_random_retrain",
            "model_config": dict(model_config, batch_size=sweep_batch, base_lr=large_lr),
            "train_mode": "target",
            "subset_fraction": 0.10,
            "subset_mode": "random",
            "total_budget_s": total_budget_s,
            "sweep_budget_s": sweep_budget_s,
            "rescore_budget_s": rescore_budget_s,
            "selection_budget_s": selection_budget_s,
        },
    }


def _load_phase_inputs(
    *,
    summary: dict[str, Any],
    data_path: str | None,
    sp_model_path: str | None,
    smoke: bool,
    device: torch.device,
) -> dict[str, Any]:
    selected = summary["selected"]
    vocab_size = int(selected["model_config"]["vocab_size"])
    if smoke:
        train_tokens, val_tokens = make_smoke_tokens(vocab_size=vocab_size)
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_unit_luts(vocab_size, device)
        return {
            "train_tokens": train_tokens,
            "val_tokens": val_tokens,
            "base_bytes_lut": base_bytes_lut,
            "has_leading_space_lut": has_leading_space_lut,
            "is_boundary_token_lut": is_boundary_token_lut,
            "sp_model_path": None,
            "data_path": "__smoke__",
        }

    resolved_data_path = data_path or selected.get("data_path")
    if resolved_data_path is None:
        raise ValueError("data path required (or stored in phase0 summary)")
    train_tokens, val_tokens, _ = load_sp_data(resolved_data_path, vocab_size)
    resolved_sp_path = sp_model_path or selected.get("sp_model_path")
    if resolved_sp_path is None:
        raise ValueError("SentencePiece model path required (or stored in phase0 summary)")
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.Load(resolved_sp_path)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, vocab_size, device)
    return {
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "base_bytes_lut": base_bytes_lut,
        "has_leading_space_lut": has_leading_space_lut,
        "is_boundary_token_lut": is_boundary_token_lut,
        "sp_model_path": resolved_sp_path,
        "data_path": resolved_data_path,
    }


def run_condition(
    config: dict[str, Any],
    *,
    phase0_summary: dict[str, Any],
    data_path: str | None = None,
    sp_model_path: str | None = None,
    output_json: str | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    selected = phase0_summary["selected"]
    model_config = dict(config["model_config"])
    device = resolve_device(model_config.get("device", "auto"))
    param_dtype = resolve_param_dtype(model_config.get("dtype", "bf16"), device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    runtime = assert_runtime_compatibility(
        smoke=smoke,
        device=device,
        require_sentencepiece=not smoke,
        sp_model_path=(sp_model_path or phase0_summary["selected"].get("sp_model_path")) if not smoke else None,
    )

    inputs = _load_phase_inputs(
        summary=phase0_summary,
        data_path=data_path,
        sp_model_path=sp_model_path,
        smoke=smoke,
        device=device,
    )
    train_tokens = inputs["train_tokens"]
    val_tokens = inputs["val_tokens"]

    seq_len = int(model_config["seq_len"])
    seed = int(model_config.get("seed", 1337))
    total_budget_s = float(config.get("total_budget_s", DEFAULT_TOTAL_BUDGET_S))
    grad_clip_norm = float(model_config.get("grad_clip_norm", 1.0))
    train_starts_all = build_lm_starts(int(train_tokens.numel()), seq_len, model_config.get("stride", seq_len // 2))
    eval_starts = choose_eval_starts(
        build_lm_starts(int(val_tokens.numel()), seq_len, seq_len),
        batch_size=min(32, model_config["batch_size"]),
        eval_batches=4 if smoke else 16,
        seed=seed,
    )

    sweep_plan = build_coverage_plan(
        int(train_tokens.numel()),
        seq_len=seq_len,
        projected_windows=int(selected.get("projected_unique_windows", 0)) or None,
    )
    sweep_starts = generate_sweep_starts(int(train_tokens.numel()), seq_len, max_windows=sweep_plan.planned_windows)

    model = build_model(model_config, device, param_dtype)
    optimizer = torch.optim.AdamW(
        list(model.parameters()),
        lr=float(model_config.get("base_lr", DEFAULT_BASE_LR)),
        weight_decay=float(model_config.get("weight_decay", 1e-2)),
    )

    budget = PhaseBudget(total_s=total_budget_s)
    train_mode = config["train_mode"]
    train_seed = seed + 17
    phase_results: dict[str, Any] = {}
    selected_subset: list[int] = []
    rescored: list[dict[str, float]] = []

    if train_mode == "baseline":
        baseline = train_on_starts(
            model,
            optimizer,
            tokens=train_tokens,
            starts=train_starts_all,
            seq_len=seq_len,
            batch_size=int(model_config["batch_size"]),
            device=device,
            param_dtype=param_dtype,
            time_budget_s=total_budget_s,
            grad_clip_norm=grad_clip_norm,
            mode="random",
            seed=train_seed,
            wrap=True,
        )
        budget.retarget_s = baseline["elapsed_s"]
        phase_results["baseline"] = baseline
    else:
        sweep = train_on_starts(
            model,
            optimizer,
            tokens=train_tokens,
            starts=sweep_starts,
            seq_len=seq_len,
            batch_size=int(model_config["batch_size"]),
            device=device,
            param_dtype=param_dtype,
            time_budget_s=float(config.get("sweep_budget_s", total_budget_s)),
            grad_clip_norm=grad_clip_norm,
            mode="sequential",
            seed=train_seed,
            wrap=(train_mode == "sweep_only"),
        )
        budget.sweep_s = sweep["elapsed_s"]
        phase_results["sweep"] = sweep

        if train_mode == "target":
            rescore = score_windows_by_loss(
                model,
                tokens=train_tokens,
                starts=list(dict.fromkeys(sweep["seen_starts"])),
                seq_len=seq_len,
                batch_size=int(model_config["batch_size"]),
                device=device,
                time_budget_s=float(config.get("rescore_budget_s", DEFAULT_RESCORE_BUDGET_S)),
            )
            budget.rescore_s = rescore["elapsed_s"]
            rescored = rescore["scored"]
            phase_results["rescore"] = {
                "elapsed_s": rescore["elapsed_s"],
                "coverage_scored_frac": rescore["coverage_scored_frac"],
                "scored_windows": len(rescored),
            }

            subset_begin = time.perf_counter()
            selected_subset = select_subset(
                rescored,
                fraction=float(config.get("subset_fraction", 0.10)),
                mode=str(config.get("subset_mode", "top")),
                seed=seed + 101,
            )
            budget.subset_s = min(float(config.get("selection_budget_s", DEFAULT_SELECTION_BUDGET_S)), float(time.perf_counter() - subset_begin))
            phase_results["selection"] = {
                "mode": config.get("subset_mode", "top"),
                "fraction": float(config.get("subset_fraction", 0.10)),
                "selected_windows": len(selected_subset),
            }

            remaining = budget.remaining_s
            retarget = train_on_starts(
                model,
                optimizer,
                tokens=train_tokens,
                starts=selected_subset,
                seq_len=seq_len,
                batch_size=32,
                device=device,
                param_dtype=param_dtype,
                time_budget_s=remaining,
                grad_clip_norm=grad_clip_norm,
                mode="random",
                seed=seed + 211,
                wrap=True,
            )
            budget.retarget_s = retarget["elapsed_s"]
            phase_results["retarget"] = retarget
        elif train_mode == "sweep_only" and budget.remaining_s > 0:
            # Sweep-only correctly spends the full budget on training.
            extra = train_on_starts(
                model,
                optimizer,
                tokens=train_tokens,
                starts=sweep_starts,
                seq_len=seq_len,
                batch_size=int(model_config["batch_size"]),
                device=device,
                param_dtype=param_dtype,
                time_budget_s=budget.remaining_s,
                grad_clip_norm=grad_clip_norm,
                mode="sequential",
                seed=train_seed + 1,
                wrap=True,
            )
            budget.retarget_s = extra["elapsed_s"]
            phase_results["sweep_continuation"] = extra

    eval_result = evaluate_bpb_sp(
        model,
        tokens=val_tokens,
        eval_starts=eval_starts,
        batch_size=min(32, int(model_config["batch_size"])),
        seq_len=seq_len,
        device=device,
        base_bytes_lut=inputs["base_bytes_lut"],
        has_leading_space_lut=inputs["has_leading_space_lut"],
        is_boundary_token_lut=inputs["is_boundary_token_lut"],
    )

    rescore_frac = budget.rescore_s / total_budget_s if total_budget_s > 0 else 0.0
    result = {
        "phase": "phaseA" if not smoke else "preflight",
        "condition": config["condition"],
        "tokenizer": selected["tokenizer"],
        "data_path": inputs["data_path"],
        "sp_model_path": inputs["sp_model_path"],
        "model_config": model_config,
        "coverage": {
            "total_windows": sweep_plan.total_windows,
            "planned_windows": sweep_plan.planned_windows,
            "planned_unique_targets": sweep_plan.unique_targets,
            "planned_coverage_frac": sweep_plan.coverage_frac,
            "low_coverage_regime": sweep_plan.low_coverage_regime,
        },
        "timings": {
            **asdict(budget),
            "selection_overhead_s": budget.selection_overhead_s,
            "rescore_frac_of_budget": rescore_frac,
        },
        "selection": {
            "selected_windows": len(selected_subset),
            "mode": config.get("subset_mode"),
            "scored_windows": len(rescored),
        },
        "runtime": runtime,
        "train_phases": phase_results,
        "eval": eval_result,
        "params": sum(p.numel() for p in model.parameters()),
        "artifact_bytes": model.artifact_bytes() if hasattr(model, "artifact_bytes") else sum(p.numel() for p in model.parameters()) * 2,
    }
    if output_json is not None:
        out_path = Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(result, indent=2, default=str))
        tmp_path.rename(out_path)
    return result


def _load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> None:
    p = argparse.ArgumentParser(description="Exp 18 single-run engine")
    p.add_argument("--config", help="YAML condition config path")
    p.add_argument("--phase0-summary", help="Phase 0 summary JSON path")
    p.add_argument("--data-path", help="Dataset path (overrides summary)")
    p.add_argument("--sp-model-path", help="SentencePiece model path (overrides summary)")
    p.add_argument("--output-json", help="Where to write run output")
    p.add_argument("--smoke", action="store_true", help="Run tiny synthetic preflight smoke test")
    args = p.parse_args()

    if args.smoke:
        if args.config and args.phase0_summary:
            summary = json.loads(Path(args.phase0_summary).read_text())
            config = _load_yaml(args.config)
        else:
            summary = make_smoke_summary(total_budget_s=SMOKE_TOTAL_BUDGET_S)
            config = build_phase_a_conditions(summary, total_budget_s=SMOKE_TOTAL_BUDGET_S)["sweep_target_top10"]
        result = run_condition(
            config,
            phase0_summary=summary,
            output_json=args.output_json,
            smoke=True,
        )
        print(json.dumps(result["timings"], indent=2))
        return

    if not args.config or not args.phase0_summary:
        raise SystemExit("--config and --phase0-summary are required unless --smoke is used")
    summary = json.loads(Path(args.phase0_summary).read_text())
    config = _load_yaml(args.config)
    result = run_condition(
        config,
        phase0_summary=summary,
        data_path=args.data_path,
        sp_model_path=args.sp_model_path,
        output_json=args.output_json,
        smoke=False,
    )
    print(
        f"{config['condition']}: bpb={result['eval']['bpb']:.4f} "
        f"coverage={result['coverage']['planned_coverage_frac']:.1%} "
        f"rescore_tax={result['timings']['rescore_frac_of_budget']:.1%}"
    )


if __name__ == "__main__":
    main()
