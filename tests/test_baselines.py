import importlib
import pytest
import torch
from chaoscontrol.baselines import SimpleTransformerLM

def test_forward_shape():
    model = SimpleTransformerLM(vocab_size=256, dim=64, num_layers=2, num_heads=4)
    ids = torch.randint(0, 256, (2, 16))
    out = model(ids)
    assert isinstance(out, dict)
    assert out["logits"].shape == (2, 16, 256)

def test_param_budget():
    model = SimpleTransformerLM(vocab_size=256, dim=128, num_layers=4, num_heads=4)
    assert model.artifact_bytes() < 16 * 1024 * 1024

def test_causal_no_future_leakage():
    model = SimpleTransformerLM(vocab_size=256, dim=32, num_layers=1, num_heads=2)
    x = torch.randint(0, 256, (1, 8))
    x.requires_grad_(False)
    embed = model.embed(x)
    embed.requires_grad_(True)
    embed.retain_grad()
    logits = model.final_norm(model.layers[0](embed))
    logits[0, 0].sum().backward()
    # Gradient at position 0 should not flow from future positions
    assert embed.grad[0, 1:].abs().sum() == 0.0

def test_deterministic():
    model = SimpleTransformerLM(vocab_size=256, dim=32, num_layers=2, num_heads=2)
    ids = torch.randint(0, 256, (2, 8))
    out1 = model(ids)["logits"]
    out2 = model(ids)["logits"]
    assert torch.allclose(out1, out2)


# ---------- Mamba2LM tests ----------

_has_mamba = importlib.util.find_spec("mamba_ssm") is not None


def test_mamba2_import_error():
    """Mamba2LM raises a clear ImportError when mamba-ssm is not installed."""
    if _has_mamba:
        pytest.skip("mamba-ssm is installed; cannot test ImportError path")
    from chaoscontrol.baselines import Mamba2LM
    with pytest.raises(ImportError, match="mamba-ssm"):
        Mamba2LM()


@pytest.mark.skipif(not _has_mamba, reason="mamba-ssm not installed")
def test_mamba2_forward_shape():
    from chaoscontrol.baselines import Mamba2LM
    model = Mamba2LM(vocab_size=256, dim=64, num_layers=2)
    ids = torch.randint(0, 256, (2, 16))
    out = model(ids)
    assert isinstance(out, dict)
    assert out["logits"].shape == (2, 16, 256)
    assert out["hidden"].shape == (2, 16, 64)


@pytest.mark.skipif(not _has_mamba, reason="mamba-ssm not installed")
def test_mamba2_param_budget():
    from chaoscontrol.baselines import Mamba2LM
    model = Mamba2LM(vocab_size=256, dim=128, num_layers=4)
    assert model.artifact_bytes() < 16 * 1024 * 1024


@pytest.mark.skipif(not _has_mamba, reason="mamba-ssm not installed")
def test_mamba2_jacobian_stats():
    from chaoscontrol.baselines import Mamba2LM
    model = Mamba2LM(vocab_size=256, dim=64, num_layers=2)
    ids = torch.randint(0, 256, (1, 8))
    out = model(ids, return_jacobian_stats=True)
    assert "jacobian_stats" in out
