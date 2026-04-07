from chaoscontrol.config import ChaosControlConfig

def test_defaults():
    cfg = ChaosControlConfig(data_path="/tmp/data")
    assert cfg.a_mode == "diag"
    assert cfg.rich_b_mode == "none"
    assert cfg.outer_model_dim == 0
    assert cfg.model_type == "ssm"
    assert cfg.semantic_tier_bases == 0
    assert cfg.generation_mode == "noise"

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

def test_tokenizer_fields():
    cfg = ChaosControlConfig(
        data_path="/tmp",
        tokenizer_type="fixed_stride",
        tokenizer_codebook_size=512,
    )
    assert cfg.tokenizer_type == "fixed_stride"
    assert cfg.tokenizer_codebook_size == 512
