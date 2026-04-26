"""Tests for run_exp20_eval.py episodic cache wiring.

Covers the W-task integration into the Exp 20 driver:
  - When ``episodic_cache_enabled=True``, the driver loads the cache from
    ``ckpt['episodic_cache']`` if present, OR instantiates a fresh empty
    cache when the checkpoint has none (the "fresh-cache TTT" path used
    by the matrix's Arm D).
  - When ``episodic_cache_reset_per_doc=True``, the driver calls
    ``cache.reset()`` at the per-doc boundary so cross-document leakage
    in the retrieval index is structurally impossible.
  - When ``episodic_cache_enabled=False`` (default), behavior is bit-
    identical to the pre-cache driver — back-compat for non-cache TTT.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from chaoscontrol.eval_stream.types import RunConfig


REPO = Path(__file__).resolve().parents[1]


def test_run_config_exposes_episodic_cache_fields():
    """The driver reads two new fields off RunConfig:
      - episodic_cache_enabled (bool, default False) — opt-in toggle
      - episodic_cache_reset_per_doc (bool, default False) — per-doc reset
    Both default to False so existing configs deserialize without
    silently flipping semantics.
    """
    cfg = RunConfig()
    assert hasattr(cfg, "episodic_cache_enabled")
    assert cfg.episodic_cache_enabled is False
    assert hasattr(cfg, "episodic_cache_reset_per_doc")
    assert cfg.episodic_cache_reset_per_doc is False
    # Constructable with explicit overrides.
    cfg2 = RunConfig(
        episodic_cache_enabled=True, episodic_cache_reset_per_doc=True,
    )
    assert cfg2.episodic_cache_enabled is True
    assert cfg2.episodic_cache_reset_per_doc is True


def _make_tiny_corpus_and_ckpt(tmp_path: Path, *, with_cache: bool = False):
    """Stand up a 1-doc-per-line JSONL, a SP model trained on it, and a
    tiny ChaosStudentLM checkpoint. Returns (jsonl, sp_model_path, ckpt_path).
    Saves ``episodic_cache`` into the checkpoint when ``with_cache=True``.
    """
    import sentencepiece as spm
    from chaoscontrol.model import ChaosStudentLM
    from chaoscontrol.optim.episodic_cache import EpisodicCache

    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "\n".join(["alpha beta gamma", "delta epsilon zeta"] * 50)
    )
    sp_prefix = tmp_path / "sp"
    spm.SentencePieceTrainer.Train(
        input=str(corpus), model_prefix=str(sp_prefix),
        vocab_size=64, character_coverage=1.0, model_type="bpe",
    )

    jsonl = tmp_path / "docs.jsonl"
    with jsonl.open("w") as fh:
        for t in [
            "hello world this is a doc",
            "another small doc",
            "and a third one",
        ]:
            fh.write(json.dumps({"text": t}) + "\n")

    m = ChaosStudentLM(
        vocab_size=64, dim=16, num_layers=2,
        block_type="ssm", a_mode="diag",
    )
    ckpt_path = tmp_path / "ckpt.pt"
    blob = {
        "model": m.state_dict(),
        "config": {
            "vocab_size": 64, "dim": 16, "num_layers": 2,
            "block_type": "ssm", "a_mode": "diag",
        },
    }
    if with_cache:
        cache = EpisodicCache(capacity=4, span_length=4, key_rep_dim=8)
        # Pre-populate so we can later confirm the cache loaded.
        cache.append(
            key_fp=12345,
            key_rep=torch.zeros(8),
            value_tok_ids=torch.tensor([1, 2, 3, 4], dtype=torch.int64),
            value_anchor_id=4,
            current_step=0, embedding_version=0,
        )
        # Use the canonical to_dict so this fixture stays in sync with the
        # save/load contract — hand-rolled payloads here would re-introduce
        # the schema-drift the from_dict strictness exists to catch.
        blob["episodic_cache"] = cache.to_dict()
    torch.save(blob, ckpt_path)
    return jsonl, f"{sp_prefix}.model", ckpt_path


def _write_cfg(
    tmp_path: Path,
    *,
    jsonl: Path,
    sp_model_path: str,
    ckpt_path: Path,
    out_path: Path,
    extra: dict | None = None,
) -> Path:
    base = {
        "adapt_set": "none", "persistence_mode": "reset",
        "chunk_size": 32, "steps_per_chunk": 0,
        "max_docs": 3, "seed": 0,
        "jsonl_paths": [str(jsonl)],
        "sp_model_path": sp_model_path,
        "checkpoint_path": str(ckpt_path),
        "output_path": str(out_path),
    }
    if extra:
        base.update(extra)
    cfg_path = tmp_path / "run.json"
    cfg_path.write_text(json.dumps(base))
    return cfg_path


def test_driver_runs_with_cache_disabled_unchanged_default(tmp_path):
    """Default config (no episodic_cache_enabled override) must produce a
    successful run with output identical in shape to the pre-cache driver.
    """
    jsonl, sp_model, ckpt = _make_tiny_corpus_and_ckpt(tmp_path)
    out_path = tmp_path / "metrics.jsonl"
    cfg_path = _write_cfg(
        tmp_path, jsonl=jsonl, sp_model_path=sp_model,
        ckpt_path=ckpt, out_path=out_path,
    )
    result = subprocess.run(
        [sys.executable, "scripts/run_exp20_eval.py",
         "--config", str(cfg_path)],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert result.returncode == 0, result.stderr
    lines = out_path.read_text().strip().splitlines()
    assert len(lines) == 3


def test_driver_loads_cache_from_checkpoint_when_enabled(tmp_path):
    """When ``episodic_cache_enabled=True`` and the checkpoint carries an
    ``episodic_cache`` payload, the driver must construct a cache from it
    and pass it through to the controller. Smoke-tested by running the
    driver end-to-end with TTT enabled and confirming a clean exit.
    """
    jsonl, sp_model, ckpt = _make_tiny_corpus_and_ckpt(
        tmp_path, with_cache=True,
    )
    out_path = tmp_path / "metrics.jsonl"
    cfg_path = _write_cfg(
        tmp_path, jsonl=jsonl, sp_model_path=sp_model,
        ckpt_path=ckpt, out_path=out_path,
        extra={
            "episodic_cache_enabled": True,
            # Light TTT so the controller exercises both score and adapt
            # under the cache-enabled controller construction.
            "adapt_set": "lm_head", "steps_per_chunk": 1,
        },
    )
    result = subprocess.run(
        [sys.executable, "scripts/run_exp20_eval.py",
         "--config", str(cfg_path)],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert result.returncode == 0, result.stderr
    # Subprocess emits a marker on the cache load path so we can confirm
    # the load branch fired (vs the fresh-cache fallback).
    assert "episodic_cache: loaded from checkpoint" in result.stderr or \
           "episodic_cache: loaded from checkpoint" in result.stdout, (
        f"expected 'episodic_cache: loaded from checkpoint' marker; "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )


def test_driver_constructs_fresh_cache_when_checkpoint_has_none(tmp_path):
    """Arm D path: episodic_cache_enabled=True but the checkpoint was
    produced by a non-cache training run (``ckpt.get('episodic_cache')`` is
    None). Driver must instantiate a fresh empty cache and run anyway.
    """
    jsonl, sp_model, ckpt = _make_tiny_corpus_and_ckpt(
        tmp_path, with_cache=False,
    )
    out_path = tmp_path / "metrics.jsonl"
    cfg_path = _write_cfg(
        tmp_path, jsonl=jsonl, sp_model_path=sp_model,
        ckpt_path=ckpt, out_path=out_path,
        extra={
            "episodic_cache_enabled": True,
            "adapt_set": "lm_head", "steps_per_chunk": 1,
        },
    )
    result = subprocess.run(
        [sys.executable, "scripts/run_exp20_eval.py",
         "--config", str(cfg_path)],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert result.returncode == 0, result.stderr
    assert "episodic_cache: fresh empty cache" in result.stderr or \
           "episodic_cache: fresh empty cache" in result.stdout, (
        f"expected 'episodic_cache: fresh empty cache' marker; "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )


def test_driver_resets_cache_per_doc_when_configured(tmp_path):
    """``episodic_cache_reset_per_doc=True`` triggers ``cache.reset()`` on
    every per-doc boundary. Smoke: the run completes and the marker fires
    once per document scored.
    """
    jsonl, sp_model, ckpt = _make_tiny_corpus_and_ckpt(
        tmp_path, with_cache=True,
    )
    out_path = tmp_path / "metrics.jsonl"
    cfg_path = _write_cfg(
        tmp_path, jsonl=jsonl, sp_model_path=sp_model,
        ckpt_path=ckpt, out_path=out_path,
        extra={
            "episodic_cache_enabled": True,
            "episodic_cache_reset_per_doc": True,
            "adapt_set": "lm_head", "steps_per_chunk": 1,
        },
    )
    result = subprocess.run(
        [sys.executable, "scripts/run_exp20_eval.py",
         "--config", str(cfg_path)],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert result.returncode == 0, result.stderr
    combined = result.stdout + result.stderr
    # One reset per scored doc — 3 docs in the corpus.
    assert combined.count("episodic_cache: reset for doc") == 3, (
        f"expected 3 per-doc reset markers; got: {combined!r}"
    )
