#!/usr/bin/env python3
"""Exp24 training-time bundle matrices and summaries."""

from __future__ import annotations

import copy
import statistics
from collections.abc import Sequence
from typing import Any


DEFAULT_CONTROL_SEEDS = (1337, 2674, 4011)
ARTIFACT_CHANGES_WEIGHTS_ONLY = "artifact_changes_weights_only"
ARTIFACT_TRAINING_ONLY = "artifact_training_only"


def _base_entry(
    *,
    speed_config: dict[str, Any],
    world_size: int,
    budget_seconds: float,
) -> dict[str, Any]:
    entry = copy.deepcopy(speed_config)
    entry.update(
        {
            "mode": "exp24_training_time_bundle",
            "world_size": int(world_size),
            "budget_seconds": float(budget_seconds),
            "eval_batches": int(entry.get("eval_batches", 0)),
            "train_sampling_mode": str(entry.get("train_sampling_mode", "random")),
            "artifact_impact": str(
                entry.get("artifact_impact", ARTIFACT_CHANGES_WEIGHTS_ONLY)
            ),
            "submit_valid": bool(entry.get("submit_valid", True)),
            "fast_slow_enabled": bool(entry.get("fast_slow_enabled", False)),
            "fast_slow_interval": int(entry.get("fast_slow_interval", 0)),
            "fast_slow_alpha": float(entry.get("fast_slow_alpha", 0.0)),
            "fast_slow_eval_copy": str(entry.get("fast_slow_eval_copy", "fast")),
            "spectral_reg_lambda_dead": float(
                entry.get("spectral_reg_lambda_dead", 0.0)
            ),
            "spectral_reg_lambda_sticky": float(
                entry.get("spectral_reg_lambda_sticky", 0.0)
            ),
            "spectral_reg_min_a": float(entry.get("spectral_reg_min_a", 0.05)),
            "spectral_reg_max_a": float(entry.get("spectral_reg_max_a", 0.98)),
            "predictive_aux_weight": float(
                entry.get("predictive_aux_weight", 0.0)
            ),
            "predictive_aux_horizon": int(
                entry.get("predictive_aux_horizon", 0)
            ),
            "predictive_aux_dim": int(entry.get("predictive_aux_dim", 0)),
            "embed_freeze_steps": int(entry.get("embed_freeze_steps", 0)),
            "semantic_layer_index": int(entry.get("semantic_layer_index", 0)),
            "semantic_momentum_min": float(
                entry.get("semantic_momentum_min", 0.5)
            ),
            "semantic_overhead_gate": float(
                entry.get("semantic_overhead_gate", 0.08)
            ),
        }
    )
    return entry


def _named_entry(
    *,
    base: dict[str, Any],
    phase: str,
    mechanism: str,
    arm: str,
    seed: int,
) -> dict[str, Any]:
    entry = copy.deepcopy(base)
    entry.update(
        {
            "name": f"exp24_{phase}_{arm}_s{int(seed)}",
            "seed": int(seed),
            "exp24_phase": phase,
            "exp24_mechanism": mechanism,
        }
    )
    return entry


def build_ring0_control_matrix(
    *,
    speed_config: dict[str, Any],
    world_size: int = 8,
    budget_seconds: float = 600.0,
    seeds: Sequence[int] = DEFAULT_CONTROL_SEEDS,
) -> list[dict[str, Any]]:
    base = _base_entry(
        speed_config=speed_config,
        world_size=world_size,
        budget_seconds=budget_seconds,
    )
    base["train_sampling_mode"] = "random"
    return [
        _named_entry(
            base=base,
            phase="ring0",
            mechanism="control",
            arm="control",
            seed=int(seed),
        )
        for seed in seeds
    ]


def build_phase_a_sampling_matrix(
    *,
    speed_config: dict[str, Any],
    world_size: int = 8,
    budget_seconds: float = 600.0,
    seeds: Sequence[int] = (1337,),
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for policy in ("random", "sequential_epoch", "shuffled_epoch"):
        base = _base_entry(
            speed_config=speed_config,
            world_size=world_size,
            budget_seconds=budget_seconds,
        )
        base["train_sampling_mode"] = policy
        for seed in seeds:
            entries.append(
                _named_entry(
                    base=base,
                    phase="phaseA",
                    mechanism="sampling_policy",
                    arm=policy,
                    seed=int(seed),
                )
            )
    return entries


def summarize_control_noise(results: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [
        {
            "seed": int(row.get("config", {}).get("seed")),
            "bpb": float(row.get("eval", {}).get("bpb")),
            "elapsed_s": float(row.get("train", {}).get("elapsed_s")),
            "tokens_per_sec": float(
                row.get("train", {}).get("aggregate_tokens_per_sec")
            ),
        }
        for row in results
    ]
    bpbs = [row["bpb"] for row in rows]
    tokens_per_sec_values = [row["tokens_per_sec"] for row in rows]
    return {
        "count": len(rows),
        "seeds": [row["seed"] for row in rows],
        "bpb_mean": float(statistics.fmean(bpbs)),
        "bpb_sample_std": float(statistics.stdev(bpbs)) if len(bpbs) > 1 else 0.0,
        "bpb_min": float(min(bpbs)),
        "bpb_max": float(max(bpbs)),
        "elapsed_s_mean": float(statistics.fmean(row["elapsed_s"] for row in rows)),
        "tokens_per_sec_mean": float(statistics.fmean(tokens_per_sec_values)),
    }
