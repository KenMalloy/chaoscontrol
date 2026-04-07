from chaoscontrol.config import ChaosControlConfig

def test_defaults():
    cfg = ChaosControlConfig(enwik8_path="/tmp/enwik8")
    assert cfg.a_mode == "diag"
    assert cfg.rich_b_mode == "none"
    assert cfg.outer_model_dim == 0
    assert cfg.model_type == "ssm"
    assert cfg.semantic_tier_bases == 0
    assert cfg.generation_mode == "noise"

def test_all_a_modes():
    for mode in ("diag", "paired", "full"):
        cfg = ChaosControlConfig(enwik8_path="/tmp", a_mode=mode)
        assert cfg.a_mode == mode

def test_all_rich_b_modes():
    for mode in ("none", "nn", "hub", "assembly", "hybrid"):
        cfg = ChaosControlConfig(enwik8_path="/tmp", rich_b_mode=mode)
        assert cfg.rich_b_mode == mode

def test_model_type():
    cfg = ChaosControlConfig(enwik8_path="/tmp", model_type="transformer")
    assert cfg.model_type == "transformer"


def test_data_format_defaults():
    """New data_path / data_format fields default to enwik8 behavior."""
    cfg = ChaosControlConfig(enwik8_path="/tmp/enwik8")
    assert cfg.data_path == ""
    assert cfg.data_format == "enwik8"


def test_data_format_fineweb():
    cfg = ChaosControlConfig(
        enwik8_path="/tmp/enwik8",
        data_path="/tmp/fineweb",
        data_format="fineweb_bytes",
    )
    assert cfg.data_path == "/tmp/fineweb"
    assert cfg.data_format == "fineweb_bytes"
