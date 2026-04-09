import pytest

from chaoscontrol.config import ChaosControlConfig

def test_defaults():
    cfg = ChaosControlConfig(data_path="/tmp/data")
    assert cfg.a_mode == "diag"
    assert cfg.rich_b_mode == "none"
    assert cfg.outer_model_dim == 0
    assert cfg.model_type == "ssm"
    assert cfg.semantic_tier_bases == 0
    assert cfg.generation_mode == "noise"
    assert cfg.retrieval_mode == "softmax_all"
    assert cfg.posterior_mode == "none"
    assert cfg.posterior_lr == 0.01
    assert cfg.residual_cache_k == 4

def test_all_a_modes():
    for mode in ("diag", "paired", "full"):
        cfg = ChaosControlConfig(data_path="/tmp", a_mode=mode)
        assert cfg.a_mode == mode

def test_all_rich_b_modes():
    for mode in ("none", "nn", "hub", "assembly", "hybrid"):
        cfg = ChaosControlConfig(data_path="/tmp", rich_b_mode=mode)
        assert cfg.rich_b_mode == mode

def test_model_type():
    cfg = ChaosControlConfig(data_path="/tmp", model_type="transformer")
    assert cfg.model_type == "transformer"


def test_valid_retrieval_modes():
    for mode in ("softmax_all", "bucket_mean", "bucket_recent", "bucket_topk"):
        cfg = ChaosControlConfig(enwik8_path="/tmp", retrieval_mode=mode)
        assert cfg.retrieval_mode == mode


def test_invalid_retrieval_mode():
    with pytest.raises(ValueError, match="retrieval_mode must be one of"):
        ChaosControlConfig(enwik8_path="/tmp", retrieval_mode="invalid_mode")


def test_valid_posterior_modes():
    for mode in ("none", "global_delta", "bucket_delta", "residual_cache"):
        cfg = ChaosControlConfig(enwik8_path="/tmp", posterior_mode=mode)
        assert cfg.posterior_mode == mode


def test_invalid_posterior_mode():
    with pytest.raises(ValueError, match="posterior_mode must be one of"):
        ChaosControlConfig(enwik8_path="/tmp", posterior_mode="bad")
