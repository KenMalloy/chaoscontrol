"""Unit tests for the Exp 21 runner's local build_model dispatch and
embed_init_path hook.

The Exp 21 ablation runs two architectures (NanoGPTLeanLM + CareStudentLM)
through the same training path. runner_exp18_ssm.py's build_model is
hardcoded to CareStudentLM, so Exp 21 has a sibling runner with its own
dispatching build_model. These tests cover that dispatch without touching
the DDP loop.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "21_sgns_tokenizer"))

from runner_exp21 import _apply_embed_init, build_model  # noqa: E402


def _default_ssm_config(vocab_size: int = 256, model_dim: int = 32) -> dict:
    return {
        "vocab_size": vocab_size,
        "model_dim": model_dim,
        "num_layers": 2,
        "ff_mult": 2,
        "a_mode": "diag",
        "a_full_rank": 8,
        "a_full_gamma": 0.05,
    }


def _default_transformer_config(vocab_size: int = 256, model_dim: int = 64) -> dict:
    return {
        "model_type": "transformer_nanogpt_lean",
        "vocab_size": vocab_size,
        "model_dim": model_dim,
        "num_layers": 2,
        "ff_mult": 2,
    }


def test_build_model_default_returns_bare_ssm():
    """Default (no model_type) builds CareStudentLM bare-SSM.

    Check by ``type(model).__name__`` rather than ``isinstance``: when the
    full test suite runs, sys.path edits from multiple conftest/runner
    modules can cause ``chaoscontrol.model`` to be loaded under different
    module-cache entries, giving distinct class objects with the same
    qualified name. ``isinstance`` would then spuriously fail even though
    the model is in fact a CareStudentLM. Name-based identity is
    import-order-independent.
    """
    model = build_model(
        _default_ssm_config(), torch.device("cpu"), torch.float32
    )
    cls = type(model)
    assert cls.__name__ == "CareStudentLM"
    assert cls.__module__ == "chaoscontrol.model"


def test_build_model_transformer_nanogpt_lean_dispatches():
    """model_type='transformer_nanogpt_lean' dispatches to NanoGPTLeanLM."""
    model = build_model(
        _default_transformer_config(), torch.device("cpu"), torch.float32
    )
    cls = type(model)
    assert cls.__name__ == "NanoGPTLeanLM"
    assert cls.__module__ == "chaoscontrol.baselines_nanogpt_lean"
    assert model.embed.num_embeddings == 256
    assert model.embed.embedding_dim == 64


def test_apply_embed_init_none_is_noop():
    """embed_init_path=None leaves weights untouched."""
    model = build_model(
        _default_transformer_config(), torch.device("cpu"), torch.float32
    )
    snapshot = model.embed.weight.detach().clone()
    _apply_embed_init(model, {"embed_init_path": None}, torch.device("cpu"))
    torch.testing.assert_close(model.embed.weight, snapshot)


def test_apply_embed_init_loads_weights(tmp_path):
    """Saved tensor with matching shape is copied into model.embed.weight."""
    model = build_model(
        _default_transformer_config(), torch.device("cpu"), torch.float32
    )
    target = torch.randn(256, 64)
    path = tmp_path / "init.pt"
    torch.save(target, path)

    _apply_embed_init(
        model, {"embed_init_path": str(path)}, torch.device("cpu")
    )
    torch.testing.assert_close(
        model.embed.weight, target.to(dtype=model.embed.weight.dtype)
    )


def test_apply_embed_init_rejects_shape_mismatch(tmp_path):
    """Wrong-shape init tensor fails fast with a clear message."""
    model = build_model(
        _default_transformer_config(), torch.device("cpu"), torch.float32
    )
    wrong = torch.randn(128, 64)  # vocab_size mismatch
    path = tmp_path / "wrong.pt"
    torch.save(wrong, path)

    with pytest.raises(AssertionError, match="shape mismatch"):
        _apply_embed_init(
            model, {"embed_init_path": str(path)}, torch.device("cpu")
        )


def test_apply_embed_init_works_on_ssm_arm(tmp_path):
    """The hook must work for the bare-SSM arm (cells C/D use it too)."""
    model = build_model(
        _default_ssm_config(vocab_size=256, model_dim=32),
        torch.device("cpu"),
        torch.float32,
    )
    target = torch.randn(256, 32)
    path = tmp_path / "ssm_init.pt"
    torch.save(target, path)

    _apply_embed_init(
        model, {"embed_init_path": str(path)}, torch.device("cpu")
    )
    torch.testing.assert_close(
        model.embed.weight, target.to(dtype=model.embed.weight.dtype)
    )
