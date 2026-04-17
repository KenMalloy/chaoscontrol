"""Unit tests for the modded-NanoGPT lean transformer variant (Exp 21)."""
import torch

from chaoscontrol.baselines_nanogpt_lean import NanoGPTLeanLM


def test_nanogpt_lean_param_count_v8192():
    """Design-spec parameter budget: ~10.49M at V=8192."""
    model = NanoGPTLeanLM(
        vocab_size=8192, d_model=256, n_head=4, n_layer=8, ffn_mult=4
    )
    n_params = sum(p.numel() for p in model.parameters())
    assert 10_300_000 <= n_params <= 10_700_000, f"got {n_params}"


def test_nanogpt_lean_forward_shape():
    torch.manual_seed(0)
    model = NanoGPTLeanLM(
        vocab_size=8192, d_model=256, n_head=4, n_layer=8, ffn_mult=4
    )
    tokens = torch.randint(0, 8192, (2, 64))
    out = model(tokens)
    assert isinstance(out, dict)
    assert out["logits"].shape == (2, 64, 8192)
    assert out["hidden"].shape == (2, 64, 256)


def test_nanogpt_lean_untied_embed_and_lm_head():
    """Embed and LM-head must be separate parameters (untied)."""
    model = NanoGPTLeanLM(
        vocab_size=1024, d_model=64, n_head=4, n_layer=2, ffn_mult=2
    )
    assert (
        model.embed.weight.data_ptr() != model.lm_head.weight.data_ptr()
    ), "embed and lm_head must be untied (separate parameters)"
    # Also verify they are distinct nn.Parameter instances, not just views.
    assert model.embed.weight is not model.lm_head.weight
