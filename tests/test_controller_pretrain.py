"""Phase S4 -- simplex controller behavior-cloning convergence test.

Trains the simplex policy on synthetic queries where the heuristic
target is ``argmax(V[:, 0])``. Asserts the trained policy reaches
>= 0.7 accuracy on a held-out batch within the configured number of
batches. The threshold matches the design doc -- perfect mimicry is
not the goal; the controller should mostly match the heuristic but
keep slack to deviate during online learning.

The experiment package directory is digit-prefixed
(``experiments/25_controller_pretrain``) so a normal ``import`` would
fail. Match the convention in ``tests/test_exp24_training_bundle.py``
and load the module via ``importlib.util.spec_from_file_location``.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[1]
PRETRAIN_DIR = REPO / "experiments" / "25_controller_pretrain"
PRETRAIN_PATH = PRETRAIN_DIR / "pretrain_controller.py"


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


def test_synthetic_batch_shapes_and_target():
    mod = _load_pretrain_module()
    cfg = mod.PretrainConfig()
    gen = torch.Generator().manual_seed(7)
    V, E, sf, target = mod.synthetic_batch(8, cfg, gen)
    assert V.shape == (8, cfg.n_vertices, cfg.k_v)
    assert E.shape == (8, cfg.n_vertices, cfg.n_vertices)
    assert sf.shape == (8, cfg.k_s)
    assert target.shape == (8,)
    assert target.dtype == torch.int64
    # Edge matrix is symmetric and has unit diagonal (cosine of a
    # vector with itself).
    diag = torch.diagonal(E, dim1=-2, dim2=-1)
    torch.testing.assert_close(diag, torch.ones_like(diag), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(E, E.transpose(-1, -2), atol=1e-6, rtol=1e-6)
    # Target is argmax of column 0 (utility) by construction.
    torch.testing.assert_close(target, V[..., 0].argmax(dim=-1))


def test_simplex_policy_forward_shapes():
    mod = _load_pretrain_module()
    cfg = mod.PretrainConfig()
    torch.manual_seed(0)
    model = mod.SimplexPolicy(cfg)
    gen = torch.Generator().manual_seed(0)
    V, E, sf, _ = mod.synthetic_batch(4, cfg, gen)
    logits = model(V, E, sf)
    assert logits.shape == (4, cfg.n_vertices)
    p = model.predict_probs(V, E, sf)
    assert p.shape == (4, cfg.n_vertices)
    sums = p.sum(dim=-1)
    torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5)


def test_simplex_pretrain_synthetic_convergence():
    mod = _load_pretrain_module()
    # 1000 batches with the default LR is enough for the policy to
    # cleanly fit argmax(V[:, 0]); we gate at 0.7 to leave headroom
    # for online learning to deviate from the heuristic. Random
    # baseline is 1/16 = 0.0625.
    cfg = mod.PretrainConfig(n_batches=1000, seed=1337)
    result = mod.train(cfg)
    acc = result["final_policy_acc"]
    assert acc >= 0.7, (
        f"policy_acc {acc:.4f} below 0.7 floor "
        f"(loss={result['final_loss']:.4f})"
    )


def test_simplex_pretrain_is_deterministic_under_seed():
    mod = _load_pretrain_module()
    # Short run is enough to detect non-determinism without paying for
    # 1000 batches twice.
    cfg = mod.PretrainConfig(n_batches=20, seed=1337)
    a = mod.train(cfg)
    b = mod.train(cfg)
    assert abs(a["final_loss"] - b["final_loss"]) < 1e-6, (
        f"loss diverged across runs: {a['final_loss']} vs {b['final_loss']}"
    )
    assert abs(a["final_policy_acc"] - b["final_policy_acc"]) < 1e-6
