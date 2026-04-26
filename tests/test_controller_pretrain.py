"""Phase D4 -- synthetic-dataset convergence test for the CPU SSM
controller pretrain pipeline.

Trains the controller on a synthetic dataset where the hidden labeling
function is a known linear-then-argmax map of features. Asserts the
policy head reaches non-trivial accuracy and the value head reaches
non-trivial R^2 within 100 epochs.

When real heuristic-trace data lands (D1+D2+D3 harvested from a
training run), the same ``train`` function will operate on real data
via a different ``dataset_fn`` -- only the data source changes.

The experiment package directory is digit-prefixed
(``experiments/25_controller_pretrain``) so a normal ``import`` would
fail. Match the convention used by ``tests/test_exp24_training_bundle.py``
and load the module via ``importlib.util.spec_from_file_location``.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[1]
PRETRAIN_PATH = (
    REPO / "experiments" / "25_controller_pretrain" / "pretrain_controller.py"
)


def _load_pretrain_module():
    module_name = "controller_pretrain_for_tests"
    spec = importlib.util.spec_from_file_location(module_name, PRETRAIN_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so ``@dataclass`` (which resolves
    # ``cls.__module__`` via ``sys.modules`` at decoration time on
    # Python 3.14+) sees the module. Skipping this raises
    # ``AttributeError: 'NoneType' object has no attribute '__dict__'``
    # from ``dataclasses._is_type``.
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_pretrain_converges_on_synthetic_data():
    mod = _load_pretrain_module()
    cfg = mod.PretrainConfig(epochs=100, seed=1337)
    result = mod.train(cfg)
    assert result["final_policy_acc"] > 0.30, (
        f"policy_acc {result['final_policy_acc']:.3f} -- expected > 0.30 "
        f"(random would be 1/n_slots ~= 0.0625)"
    )
    assert result["final_value_r2"] > 0.40, (
        f"value_r2 {result['final_value_r2']:.3f} -- expected > 0.40 "
        f"(constant predictor would give 0.0)"
    )


def test_synthetic_dataset_shapes():
    mod = _load_pretrain_module()
    cfg = mod.PretrainConfig()
    features, target_slot, target_reward = mod.synthetic_dataset(100, cfg)
    assert features.shape == (100, cfg.feature_dim)
    assert target_slot.shape == (100,)
    assert target_reward.shape == (100,)
    assert target_slot.dtype == torch.int64
