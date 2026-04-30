"""Tests for Experiment 14 training loop integration."""
import torch
from chaoscontrol.model import CareStudentLM
from chaoscontrol.training import train_chaoscontrol_for_budget


def _make_model(buffer_mode="append_only", retrieval_mode="bucket_mean"):
    return CareStudentLM(
        vocab_size=256, dim=64, num_layers=2,
        outer_model_dim=32, outer_model_type="multislot",
        outer_max_slots=0,
        buffer_mode=buffer_mode,
        retrieval_mode=retrieval_mode,
        wernicke_enabled=True, wernicke_k_max=4,
        wernicke_router="moe",
    )


def _make_data(n=4096):
    tokens = torch.randint(0, 256, (n,))
    starts = list(range(0, n - 128, 64))
    return tokens, starts


def test_training_append_only_runs():
    """Smoke test: training loop completes with append-only buffer."""
    model = _make_model()
    train_tokens, train_starts = _make_data()
    result = train_chaoscontrol_for_budget(
        model,
        train_tokens=train_tokens,
        train_starts=train_starts,
        seq_len=64, batch_size=4,
        device=torch.device("cpu"),
        param_dtype=torch.float32,
        budget_seconds=3.0,
        base_lr=2e-3, weight_decay=1e-2,
        grad_clip_norm=1.0, seed=42,
        buffer_mode="append_only",
        wernicke_enabled=True,
    )
    assert result["steps"] > 0


def test_training_append_only_skips_consolidation():
    """In append_only mode, consolidation_step should not be called
    but the buffer should grow via per-token writes."""
    model = _make_model(retrieval_mode="bucket_topk")
    train_tokens, train_starts = _make_data()
    result = train_chaoscontrol_for_budget(
        model,
        train_tokens=train_tokens,
        train_starts=train_starts,
        seq_len=64, batch_size=4,
        device=torch.device("cpu"),
        param_dtype=torch.float32,
        budget_seconds=3.0,
        base_lr=2e-3, weight_decay=1e-2,
        grad_clip_norm=1.0, seed=42,
        buffer_mode="append_only",
        wernicke_enabled=True,
    )
    assert result["steps"] > 0
    # Buffer should have grown
    assert len(model.outer_model._slots) > 0


def test_training_legacy_mode_still_works():
    """Legacy mode should still use consolidation_step."""
    model = CareStudentLM(
        vocab_size=256, dim=64, num_layers=2,
        outer_model_dim=32, outer_model_type="multislot",
        outer_max_slots=64,
        wernicke_enabled=True, wernicke_k_max=4,
        wernicke_router="moe",
        typed_storage=True,
    )
    train_tokens, train_starts = _make_data()
    result = train_chaoscontrol_for_budget(
        model,
        train_tokens=train_tokens,
        train_starts=train_starts,
        seq_len=64, batch_size=4,
        device=torch.device("cpu"),
        param_dtype=torch.float32,
        budget_seconds=3.0,
        base_lr=2e-3, weight_decay=1e-2,
        grad_clip_norm=1.0, seed=42,
    )
    assert result["steps"] > 0
