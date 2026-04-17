"""Exp 21: config-driven embedding-init hook in runner.build_model.

Verifies that an `embed_init_path` pointing to a (vocab_size, model_dim)
tensor is loaded and copied into model.embed.weight, and that leaving it
None preserves the model's default random init.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from chaoscontrol.config import ChaosControlConfig
from chaoscontrol.data import resolve_device, resolve_param_dtype
from chaoscontrol.runner import build_model


def _small_cfg(**overrides) -> ChaosControlConfig:
    kwargs = dict(
        data_path="/tmp",
        model_type="ssm",
        vocab_size=64,
        model_dim=16,
        num_layers=2,
    )
    kwargs.update(overrides)
    return ChaosControlConfig(**kwargs)


def test_embed_init_path_loads_weights() -> None:
    torch.manual_seed(0)
    vocab, dim = 64, 16
    known = torch.randn(vocab, dim)

    with tempfile.TemporaryDirectory() as tmp:
        pt_path = Path(tmp) / "embed.pt"
        torch.save(known, pt_path)

        cfg = _small_cfg(vocab_size=vocab, model_dim=dim, embed_init_path=str(pt_path))
        device = resolve_device("cpu")
        dtype = resolve_param_dtype("fp32", device)
        model = build_model(cfg, device, dtype)

        loaded = model.embed.weight.detach().to(known.dtype).cpu()
        assert loaded.shape == known.shape
        assert torch.allclose(loaded, known)


def test_embed_init_path_none_uses_random() -> None:
    torch.manual_seed(0)
    cfg = _small_cfg()  # embed_init_path defaults to None
    device = resolve_device("cpu")
    dtype = resolve_param_dtype("fp32", device)
    model = build_model(cfg, device, dtype)

    w = model.embed.weight.detach()
    zeros = torch.zeros_like(w)
    assert not torch.allclose(w, zeros), "expected random init, got all-zeros"
