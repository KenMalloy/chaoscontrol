#!/usr/bin/env python3
"""Tests for the Experiment 16 Phase A scaffold."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from runner_exp16 import (
    ContentOracleSelector,
    build_model,
    collect_oracle_examples,
    train_selector_probe,
)
import run_exp16


def test_build_model_shapes():
    config = {
        "vocab_size": 8192,
        "model_dim": 64,
        "num_layers": 2,
        "ff_mult": 2,
        "a_mode": "diag",
    }
    model = build_model(config, torch.device("cpu"), torch.float32)
    assert model.vocab_size == 8192
    assert model.embed.weight.shape == (8192, 64)


def test_collect_oracle_examples_returns_expected_shapes():
    config = {
        "vocab_size": 64,
        "model_dim": 16,
        "num_layers": 2,
        "ff_mult": 2,
        "a_mode": "diag",
    }
    model = build_model(config, torch.device("cpu"), torch.float32)
    tokens = torch.randint(0, 64, (256,), dtype=torch.long)
    examples = collect_oracle_examples(
        model,
        tokens=tokens,
        starts=[0, 32, 64],
        seq_len=32,
        device=torch.device("cpu"),
        layer_index=1,
        buffer_size=8,
        k=4,
        max_examples=32,
        query_source="x_state",
        write_source="x_state",
    )
    assert examples["queries"].ndim == 2
    assert examples["candidate_keys"].shape[1] == 8
    assert examples["target_probs"].shape == examples["mask"].shape
    assert examples["examples"] > 0


def test_selector_probe_solves_easy_identity_task():
    torch.manual_seed(0)
    n = 128
    buffer_size = 8
    dim = 12
    k = 2

    queries = torch.randn(n, dim)
    candidate_keys = torch.randn(n, buffer_size, dim)
    mask = torch.ones(n, buffer_size, dtype=torch.bool)
    target_probs = torch.zeros(n, buffer_size)

    for i in range(n):
        candidate_keys[i, 3] = queries[i]
        candidate_keys[i, 6] = queries[i] * 0.9
        target_probs[i, 3] = 0.7
        target_probs[i, 6] = 0.3

    metrics = train_selector_probe(
        {
            "queries": queries,
            "candidate_keys": candidate_keys,
            "target_probs": target_probs,
            "mask": mask,
        },
        k=k,
        selector_dim=dim,
        device=torch.device("cpu"),
        epochs=20,
        batch_size=32,
        lr=5e-3,
        seed=123,
    )
    assert metrics["mass_capture_at_k"] > 0.90
    assert metrics["recall_at_k"] > 0.90


def test_token_keyed_baseline_metrics_present():
    config = {
        "vocab_size": 64,
        "model_dim": 16,
        "num_layers": 2,
        "ff_mult": 2,
        "a_mode": "diag",
    }
    model = build_model(config, torch.device("cpu"), torch.float32)
    # Use tokens with repeats so token-keyed has matches to find
    tokens = torch.randint(0, 8, (256,), dtype=torch.long)
    examples = collect_oracle_examples(
        model,
        tokens=tokens,
        starts=[0, 32, 64],
        seq_len=32,
        device=torch.device("cpu"),
        layer_index=1,
        buffer_size=8,
        k=4,
        max_examples=32,
        query_source="x_state",
        write_source="x_state",
    )
    assert "token_keyed_recall_at_k" in examples
    assert "token_keyed_mass_capture_at_k" in examples
    assert 0.0 <= examples["token_keyed_recall_at_k"] <= 1.0
    assert 0.0 <= examples["token_keyed_mass_capture_at_k"] <= 1.0


def test_token_keyed_baseline_zero_when_no_repeats():
    config = {
        "vocab_size": 1024,
        "model_dim": 16,
        "num_layers": 2,
        "ff_mult": 2,
        "a_mode": "diag",
    }
    model = build_model(config, torch.device("cpu"), torch.float32)
    # All unique token IDs — no repeats in any buffer window
    tokens = torch.arange(256, dtype=torch.long)
    examples = collect_oracle_examples(
        model,
        tokens=tokens,
        starts=[0],
        seq_len=32,
        device=torch.device("cpu"),
        layer_index=1,
        buffer_size=8,
        k=4,
        max_examples=32,
        query_source="x_state",
        write_source="x_state",
    )
    assert examples["token_keyed_mass_capture_at_k"] == 0.0
    assert examples["token_keyed_recall_at_k"] == 0.0


def test_content_oracle_selector_masks_invalid_slots():
    selector = ContentOracleSelector(query_dim=4, key_dim=4, selector_dim=4)
    queries = torch.randn(2, 4)
    keys = torch.randn(2, 5, 4)
    mask = torch.tensor([[1, 1, 0, 0, 0], [1, 0, 0, 0, 0]], dtype=torch.bool)
    scores = selector(queries, keys, mask)
    assert torch.isfinite(scores[:, :2]).all()
    assert (scores[0, 2:] < -1e8).all()


def test_summarize_results_prefers_best_passing_condition(tmp_path):
    original_results = run_exp16.RESULTS
    run_exp16.RESULTS = tmp_path
    try:
        conditions = {
            "raw_mass_winner_fails_gate": {"sparse_attn_k": 8},
            "lower_mass_but_passes": {"sparse_attn_k": 8},
        }
        failing_probe = {
            "selector_mass_capture_at_k": 0.72,
            "selector_recall_at_k": 0.70,
            "effective_connections": 8.0,
            "recent_mass_capture_at_k": 0.60,
            "token_keyed_mass_capture_at_k": 0.75,
        }
        passing_probe = {
            "selector_mass_capture_at_k": 0.68,
            "selector_recall_at_k": 0.66,
            "effective_connections": 5.0,
            "recent_mass_capture_at_k": 0.50,
            "token_keyed_mass_capture_at_k": 0.52,
        }
        for seed in run_exp16.SWEEP_SEEDS[:3]:
            (tmp_path / f"raw_mass_winner_fails_gate_s{seed}.json").write_text(
                json.dumps({"oracle_probe": failing_probe})
            )
            (tmp_path / f"lower_mass_but_passes_s{seed}.json").write_text(
                json.dumps({"oracle_probe": passing_probe})
            )

        summary = run_exp16.summarize_results(conditions)
        assert summary["_decision"]["best_condition"] == "lower_mass_but_passes"
        assert summary["_decision"]["n_passing_conditions"] == 1
        assert summary["_decision"]["all_gates_pass"] is True
    finally:
        run_exp16.RESULTS = original_results
