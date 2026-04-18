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


def test_nanogpt_lean_encode_matches_forward_hidden():
    """encode() must return the same pre-final_norm hidden as forward()["hidden"].

    train_ssm_for_budget relies on this contract: it runs model.encode() once,
    detaches, then loops chunked forward/backward through model.final_norm +
    model.lm_head. If encode() diverges from forward()'s hidden, the chunked
    path computes different gradients than the frozen forward path.
    """
    torch.manual_seed(0)
    model = NanoGPTLeanLM(
        vocab_size=1024, d_model=64, n_head=4, n_layer=2, ffn_mult=2
    )
    model.eval()
    tokens = torch.randint(0, 1024, (2, 32))
    with torch.no_grad():
        forward_out = model(tokens)
        encoded = model.encode(tokens)
    torch.testing.assert_close(encoded, forward_out["hidden"])


def test_nanogpt_lean_encode_plus_head_matches_forward_logits():
    """encode() + final_norm + lm_head must reproduce forward()'s logits.

    End-to-end check that chunked LM-head backward over encode() output
    sees the same function as forward()."""
    torch.manual_seed(0)
    model = NanoGPTLeanLM(
        vocab_size=1024, d_model=64, n_head=4, n_layer=2, ffn_mult=2
    )
    model.eval()
    tokens = torch.randint(0, 1024, (2, 32))
    with torch.no_grad():
        forward_out = model(tokens)
        hidden = model.encode(tokens)
        logits = model.lm_head(model.final_norm(hidden))
    torch.testing.assert_close(logits, forward_out["logits"])
