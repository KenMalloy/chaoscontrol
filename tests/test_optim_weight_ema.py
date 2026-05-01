"""Tests for rank-0 fp32 weight EMA shadow."""
import pytest
import torch
from chaoscontrol.optim.weight_ema import WeightEMA


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.outer_model = torch.nn.Linear(4, 4)  # simulated memory component


def test_ema_init_excludes_named_prefixes():
    model = _Tiny()
    ema = WeightEMA(model, decay=0.997, exclude_prefixes=("outer_model.",))
    assert "linear.weight" in ema.shadow
    assert "linear.bias" in ema.shadow
    assert not any(name.startswith("outer_model.") for name in ema.shadow)


def test_ema_excludes_registered_buffers():
    """Buffers like running stats are not gradient-trained — keep them out
    of the shadow even when they appear in state_dict()."""
    class _WithBuffer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4)
            self.register_buffer("running_stat", torch.zeros(4))

    model = _WithBuffer()
    ema = WeightEMA(model, decay=0.997, exclude_prefixes=())
    assert "linear.weight" in ema.shadow
    assert "running_stat" not in ema.shadow


def test_ema_excludes_non_trainable_params():
    """Frozen parameters should not be in the shadow."""
    model = _Tiny()
    model.linear.bias.requires_grad_(False)
    ema = WeightEMA(model, decay=0.997, exclude_prefixes=())
    assert "linear.weight" in ema.shadow
    assert "linear.bias" not in ema.shadow


def test_ema_shadow_is_fp32_regardless_of_model_dtype():
    model = _Tiny().to(torch.bfloat16)
    ema = WeightEMA(model, decay=0.997, exclude_prefixes=())
    for tensor in ema.shadow.values():
        assert tensor.dtype == torch.float32


def test_ema_update_blends_decay():
    model = _Tiny()
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    ema = WeightEMA(model, decay=0.9, exclude_prefixes=())
    initial = ema.shadow["linear.weight"].clone()
    with torch.no_grad():
        model.linear.weight.fill_(1.0)
    ema.update(model)
    expected = 0.9 * initial + 0.1 * torch.ones_like(initial)
    assert torch.allclose(ema.shadow["linear.weight"], expected)


def test_ema_update_is_noop_when_shadow_excluded():
    """Excluded parameters never appear in shadow — update should not error."""
    model = _Tiny()
    ema = WeightEMA(model, decay=0.9, exclude_prefixes=("outer_model.",))
    # Mutate the excluded params; should not affect shadow.
    with torch.no_grad():
        model.outer_model.weight.fill_(99.0)
    ema.update(model)
    assert "outer_model.weight" not in ema.shadow


def test_ema_swap_and_restore():
    model = _Tiny()
    with torch.no_grad():
        model.linear.weight.fill_(2.0)
    ema = WeightEMA(model, decay=0.9, exclude_prefixes=())
    ema.shadow["linear.weight"].fill_(7.0)
    pre_swap = model.linear.weight.detach().clone()
    with ema.applied(model):
        assert torch.allclose(model.linear.weight, torch.full_like(pre_swap, 7.0))
    # Original weights restored after context exit.
    assert torch.allclose(model.linear.weight, pre_swap)
