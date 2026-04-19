"""Unit tests for the Exp 21b embedding-prior dissection runner manifest."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "21_sgns_tokenizer"))

import runner_exp21b  # noqa: E402


def test_exp21b_mechanism_conditions_are_ssm_only_and_named():
    conditions = runner_exp21b.build_conditions()

    assert set(conditions) == {
        "ssm_norm_only",
        "ssm_norm_only_shuffled",
        "ssm_freq_bucket_shuffle",
        "ssm_class_bucket_shuffle",
        "ssm_fullcov_shuffled",
        "ssm_random_fullcov",
    }
    for name, cfg in conditions.items():
        assert cfg["model_type"] == "ssm", name
        assert cfg["base_lr"] == runner_exp21b.SSM_LR, name
        assert cfg["embed_init_path"] == runner_exp21b.MECHANISM_ARTIFACTS[name]


def test_exp21b_condition_subset_preserves_manifest_paths():
    conditions = runner_exp21b.build_conditions([
        "ssm_norm_only",
        "ssm_random_fullcov",
    ])

    assert list(conditions) == ["ssm_norm_only", "ssm_random_fullcov"]
    assert (
        conditions["ssm_norm_only"]["embed_init_path"]
        == "artifacts/sgns_init_norm_only.pt"
    )
    assert (
        conditions["ssm_random_fullcov"]["embed_init_path"]
        == "artifacts/sgns_init_random_fullcov.pt"
    )


def test_exp21b_rejects_unknown_condition_name():
    try:
        runner_exp21b.build_conditions(["ssm_nope"])
    except ValueError as exc:
        assert "unknown Exp 21b condition" in str(exc)
    else:
        raise AssertionError("expected unknown condition to fail")


def test_exp21b_preflight_rejects_bad_artifact_shape(tmp_path, monkeypatch):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    bad_path = artifacts / "bad.pt"
    torch.save(torch.randn(3, 5), bad_path)
    monkeypatch.setattr(runner_exp21b, "REPO", tmp_path)

    conditions = {
        "ssm_bad": {
            "embed_init_path": "artifacts/bad.pt",
            "vocab_size": 3,
            "model_dim": 4,
        },
    }
    with pytest.raises(ValueError, match="shape mismatch"):
        runner_exp21b.validate_embed_artifacts(conditions)
