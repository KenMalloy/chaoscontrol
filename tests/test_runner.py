"""Tests for the YAML config experiment dispatcher."""
from __future__ import annotations

import os
import tempfile
import unittest

import yaml

from chaoscontrol.config import ChaosControlConfig
from chaoscontrol.data import resolve_device, resolve_param_dtype
from chaoscontrol.runner import load_config, build_model


class TestLoadConfig(unittest.TestCase):
    def test_load_config_from_yaml(self) -> None:
        cfg_dict = {"model_dim": 64, "a_mode": "diag"}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(cfg_dict, f)
            path = f.name
        cfg = load_config(path, enwik8_path="/tmp/enwik8")
        assert cfg.model_dim == 64
        assert cfg.a_mode == "diag"
        assert cfg.enwik8_path == "/tmp/enwik8"
        os.unlink(path)

    def test_load_config_budget_override(self) -> None:
        cfg_dict = {"budget_seconds": 60}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(cfg_dict, f)
            path = f.name
        cfg = load_config(path, enwik8_path="/tmp/enwik8", budget_seconds=999)
        assert cfg.budget_seconds == 999
        os.unlink(path)


class TestBuildModel(unittest.TestCase):
    def test_build_model_ssm(self) -> None:
        cfg = ChaosControlConfig(enwik8_path="/tmp", model_type="ssm", model_dim=16, num_layers=2)
        device = resolve_device("cpu")
        dtype = resolve_param_dtype("fp32", device)
        model = build_model(cfg, device, dtype)
        assert hasattr(model, "vocab_size")


if __name__ == "__main__":
    unittest.main()
