"""Tests for Experiment 14 model integration."""
import torch
from chaoscontrol.model import ChaosStudentLM


def test_forward_append_only_mode():
    """Model creates with append_only buffer and produces logits."""
    model = ChaosStudentLM(
        vocab_size=256, dim=128, num_layers=2,
        outer_model_dim=64, outer_model_type="multislot",
        outer_max_slots=0,
        buffer_mode="append_only",
        retrieval_mode="bucket_topk", retrieval_k=4,
        wernicke_enabled=True, wernicke_k_max=8,
        wernicke_router="moe",
    )
    x = torch.randint(0, 256, (2, 32))
    out = model(x)
    assert "logits" in out
    assert out["logits"].shape == (2, 32, 256)


def test_forward_bucket_prototypes():
    """Model with bucket prototypes runs without error."""
    model = ChaosStudentLM(
        vocab_size=256, dim=128, num_layers=2,
        outer_model_dim=64, outer_model_type="multislot",
        outer_max_slots=0,
        buffer_mode="append_only",
        retrieval_mode="bucket_mean",
        bucket_prototypes=True,
        wernicke_enabled=True, wernicke_k_max=4,
        wernicke_router="moe",
    )
    x = torch.randint(0, 256, (2, 32))
    out = model(x)
    assert "logits" in out


def test_forward_hierarchical_wernicke():
    """Model with hierarchical Wernicke runs and reports bucket_ids."""
    model = ChaosStudentLM(
        vocab_size=256, dim=128, num_layers=2,
        outer_model_dim=64, outer_model_type="multislot",
        outer_max_slots=0,
        buffer_mode="append_only",
        retrieval_mode="bucket_topk", retrieval_k=4,
        wernicke_enabled=True, wernicke_layers=2,
        wernicke_k_max=8, wernicke_k_max_fine=8,
        wernicke_router="moe",
    )
    x = torch.randint(0, 256, (2, 32))
    out = model(x)
    assert "logits" in out
    assert "bucket_ids" in out
    assert out["bucket_ids"].max() < 64  # 8 * 8


def test_forward_side_effect_free_by_default():
    """Fix 2: forward() with default memory_write_mode='none' must not
    modify the buffer."""
    model = ChaosStudentLM(
        vocab_size=256, dim=128, num_layers=2,
        outer_model_dim=64, outer_model_type="multislot",
        outer_max_slots=0,
        buffer_mode="append_only",
        retrieval_mode="bucket_mean",
        wernicke_enabled=True, wernicke_k_max=4,
        wernicke_router="moe",
    )
    assert len(model.outer_model._slots) == 0
    x = torch.randint(0, 256, (2, 32))
    # Default: no writes
    model(x)
    assert len(model.outer_model._slots) == 0


def test_forward_append_only_writes():
    """When memory_write_mode='append_only', buffer should grow."""
    model = ChaosStudentLM(
        vocab_size=256, dim=128, num_layers=2,
        outer_model_dim=64, outer_model_type="multislot",
        outer_max_slots=0,
        buffer_mode="append_only",
        retrieval_mode="bucket_mean",
        wernicke_enabled=True, wernicke_k_max=4,
        wernicke_router="moe",
    )
    assert len(model.outer_model._slots) == 0
    x = torch.randint(0, 256, (2, 32))
    model(x, memory_write_mode="append_only")
    assert len(model.outer_model._slots) > 0


def test_forward_per_token_writes():
    """Fix 3: writes must be per-token, not one dominant bucket per batch."""
    model = ChaosStudentLM(
        vocab_size=256, dim=128, num_layers=2,
        outer_model_dim=64, outer_model_type="multislot",
        outer_max_slots=0,
        buffer_mode="append_only",
        retrieval_mode="bucket_mean",
        wernicke_enabled=True, wernicke_k_max=4,
        wernicke_router="moe",
    )
    x = torch.randint(0, 256, (2, 32))
    out = model(x, memory_write_mode="append_only")
    # With batch=2, seq=32, we should have 2*32 = 64 entries
    assert len(model.outer_model._slots) == 2 * 32


def test_forward_legacy_mode_unchanged():
    """Legacy buffer mode should use the old read() path."""
    model = ChaosStudentLM(
        vocab_size=256, dim=128, num_layers=2,
        outer_model_dim=64, outer_model_type="multislot",
        outer_max_slots=64,
        wernicke_enabled=True, wernicke_k_max=4,
        wernicke_router="moe",
    )
    x = torch.randint(0, 256, (2, 32))
    out = model(x)
    assert "logits" in out
    assert out["logits"].shape == (2, 32, 256)


def test_forward_returns_bucket_ids():
    model = ChaosStudentLM(
        vocab_size=256, dim=128, num_layers=2,
        outer_model_dim=64, outer_model_type="multislot",
        outer_max_slots=0,
        buffer_mode="append_only",
        retrieval_mode="bucket_mean",
        wernicke_enabled=True, wernicke_k_max=4,
        wernicke_router="moe",
    )
    x = torch.randint(0, 256, (2, 32))
    out = model(x)
    assert "bucket_ids" in out
    assert out["bucket_ids"].shape == (2, 32)
