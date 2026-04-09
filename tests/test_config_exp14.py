"""Tests for Experiment 14 config fields."""
from chaoscontrol.config import ChaosControlConfig


def test_buffer_mode_default():
    cfg = ChaosControlConfig(data_path="dummy")
    assert cfg.buffer_mode == "legacy"


def test_buffer_mode_append():
    cfg = ChaosControlConfig(data_path="dummy", buffer_mode="append_only")
    assert cfg.buffer_mode == "append_only"


def test_retrieval_mode_default():
    cfg = ChaosControlConfig(data_path="dummy")
    assert cfg.retrieval_mode == "softmax_all"


def test_retrieval_modes():
    for mode in ("softmax_all", "bucket_mean", "bucket_recent", "bucket_topk"):
        cfg = ChaosControlConfig(data_path="dummy", retrieval_mode=mode)
        assert cfg.retrieval_mode == mode


def test_retrieval_k_default():
    cfg = ChaosControlConfig(data_path="dummy")
    assert cfg.retrieval_k == 8


def test_hierarchical_wernicke_default():
    cfg = ChaosControlConfig(data_path="dummy")
    assert cfg.wernicke_layers == 1


def test_hierarchical_wernicke():
    cfg = ChaosControlConfig(
        data_path="dummy",
        wernicke_layers=2,
        wernicke_k_max=8,
        wernicke_k_max_fine=32,
    )
    assert cfg.wernicke_layers == 2
    assert cfg.wernicke_k_max_fine == 32


def test_bucket_prototypes_default():
    cfg = ChaosControlConfig(data_path="dummy")
    assert cfg.bucket_prototypes is False


def test_max_slots_unlimited():
    cfg = ChaosControlConfig(data_path="dummy", outer_max_slots=0)
    assert cfg.outer_max_slots == 0  # 0 = unlimited


def test_prototype_dim_default():
    cfg = ChaosControlConfig(data_path="dummy")
    assert cfg.prototype_dim == 64


def test_prototype_update_rate_default():
    cfg = ChaosControlConfig(data_path="dummy")
    assert cfg.prototype_update_rate == 0.1
