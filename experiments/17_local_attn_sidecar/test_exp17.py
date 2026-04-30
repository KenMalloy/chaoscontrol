#!/usr/bin/env python3
"""Tests for the Experiment 17 scaffold."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chaoscontrol.model import CareSSMBlock, CareSSMHybridBlock
import run_exp17
from runner_exp17 import build_child_env, build_model, resolve_visible_cuda_devices, validate_gpu_concurrency


def test_build_model_bare_fast_ssm():
    config = {
        "vocab_size": 8192,
        "model_dim": 64,
        "num_layers": 4,
        "ff_mult": 2,
        "a_mode": "diag",
        "local_attn_window": 0,
    }
    model = build_model(config, torch.device("cpu"), torch.float32)
    assert model.local_attn_window == 0
    assert all(isinstance(layer, CareSSMBlock) for layer in model.layers)


def test_build_model_hybrid_top_block():
    config = {
        "vocab_size": 8192,
        "model_dim": 64,
        "num_layers": 4,
        "ff_mult": 2,
        "a_mode": "diag",
        "local_attn_window": 16,
        "local_attn_heads": 1,
        "local_attn_dim": 32,
    }
    model = build_model(config, torch.device("cpu"), torch.float32)
    assert isinstance(model.layers[-1], CareSSMHybridBlock)
    assert all(isinstance(layer, CareSSMBlock) for layer in model.layers[:-1])


def test_hybrid_model_forward_shape():
    config = {
        "vocab_size": 64,
        "model_dim": 32,
        "num_layers": 4,
        "ff_mult": 2,
        "a_mode": "diag",
        "local_attn_window": 8,
        "local_attn_heads": 1,
        "local_attn_dim": 16,
    }
    model = build_model(config, torch.device("cpu"), torch.float32)
    ids = torch.randint(0, 64, (2, 10))
    out = model(ids)
    assert out["logits"].shape == (2, 10, 64)


def test_build_child_env_respects_parent_mask():
    env = build_child_env(gpu_slot=1, base_env={"CUDA_VISIBLE_DEVICES": "2,4,6"})
    assert env["CUDA_VISIBLE_DEVICES"] == "4"
    assert resolve_visible_cuda_devices({"CUDA_VISIBLE_DEVICES": "2,4,6"}) == ["2", "4", "6"]


def test_validate_gpu_concurrency_uses_visible_mask():
    visible = validate_gpu_concurrency(2, {"CUDA_VISIBLE_DEVICES": "1,3,5"})
    assert visible == ["1", "3", "5"]


def test_summarize_results_prefers_best_passing_local(tmp_path):
    original_results = run_exp17.RESULTS
    run_exp17.RESULTS = tmp_path
    try:
        conditions = {
            "bare_fast_ssm": {"local_attn_window": 0},
            "local_w16": {"local_attn_window": 16},
            "local_w32": {"local_attn_window": 32},
        }
        bare_bpbs = [1.630, 1.632, 1.628]
        local_w16_bpbs = [1.598, 1.601, 1.600]
        local_w32_bpbs = [1.585, 1.586, 1.584]
        for seed, bare_bpb, w16_bpb, w32_bpb in zip(run_exp17.SWEEP_SEEDS[:3], bare_bpbs, local_w16_bpbs, local_w32_bpbs):
            (tmp_path / f"bare_fast_ssm_s{seed}.json").write_text(json.dumps({
                "eval": {"bpb": bare_bpb},
                "train": {"steps_per_second": 20.0},
                "artifact_bytes": 6_500_000,
            }))
            (tmp_path / f"local_w16_s{seed}.json").write_text(json.dumps({
                "eval": {"bpb": w16_bpb},
                "train": {"steps_per_second": 15.0},
                "artifact_bytes": 6_600_000,
            }))
            (tmp_path / f"local_w32_s{seed}.json").write_text(json.dumps({
                "eval": {"bpb": w32_bpb},
                "train": {"steps_per_second": 8.0},
                "artifact_bytes": 6_600_000,
            }))

        summary = run_exp17.summarize_results(conditions)
        assert summary["_decision"]["best_condition"] == "local_w16"
        assert summary["_decision"]["n_passing_conditions"] == 1
        assert summary["_decision"]["all_gates_pass"] is True
    finally:
        run_exp17.RESULTS = original_results
