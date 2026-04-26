"""Cache wiring tests for scripts/run_exp20_fast_score.py.

The score-only fast path is used by Exp20/Exp24 validation pilots. When a
matrix config enables eval-side episodic cache, fast_score must load the same
checkpoint cache payload and construct the same cache-aware LegalityController
shape as scripts/run_exp20_eval.py.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch

from chaoscontrol.optim.episodic_cache import EpisodicCache

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_exp20_fast_score.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_exp20_fast_score_cache_test",
        SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cache_args(**overrides):
    values = {
        "episodic_cache_enabled": True,
        "episodic_cache_capacity": 4096,
        "episodic_span_length": 4,
        "episodic_key_rep_dim": -1,
        "episodic_grace_steps": 1000,
        "episodic_fingerprint_window": 12,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_fast_score_loads_checkpoint_cache_into_legality_controller(tmp_path):
    """Loaded cache payload must survive into the fast_score controller."""
    mod = _load_module()
    from chaoscontrol.model import ChaosStudentLM

    model = ChaosStudentLM(
        vocab_size=32,
        dim=16,
        num_layers=1,
        block_type="ssm",
        a_mode="diag",
    )
    cache = EpisodicCache(
        capacity=4096,
        span_length=4,
        key_rep_dim=16,
        grace_steps=1000,
        fingerprint_window=12,
    )
    cache.append(
        key_fp=123456,
        key_rep=torch.ones(16),
        value_tok_ids=torch.tensor([1, 2, 3, 4], dtype=torch.int64),
        value_anchor_id=1,
        current_step=9,
        embedding_version=0,
        pressure_at_write=2.5,
        source_write_id=77,
        write_bucket=3,
    )
    ckpt_blob = {
        "model": model.state_dict(),
        "config": {
            "vocab_size": 32,
            "dim": 16,
            "num_layers": 1,
            "block_type": "ssm",
            "a_mode": "diag",
        },
        "episodic_cache": cache.to_dict(),
    }
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save(ckpt_blob, ckpt_path)

    loaded_model, ckpt_cfg, loaded_blob = mod._build_model_with_blob(ckpt_path)
    controller, loaded_cache, source = mod._build_legality_controller(
        loaded_model,
        args=_cache_args(),
        ckpt_cfg=ckpt_cfg,
        ckpt_blob=loaded_blob,
    )

    assert source == "loaded"
    assert controller.cache is loaded_cache
    assert loaded_cache is not None
    assert loaded_cache.capacity == 4096
    assert loaded_cache.span_length == 4
    assert loaded_cache.key_rep_dim == 16
    assert loaded_cache.grace_steps == 1000
    assert loaded_cache.fingerprint_window == 12
    assert int(loaded_cache.occupied.sum().item()) == 1
    entry = loaded_cache.query(123456)
    assert entry is not None
    assert entry.source_write_id == 77
    assert entry.write_bucket == 3
    assert controller.fingerprint_window == 12
