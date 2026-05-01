import os
import pathlib

import pytest

_REPO_ROOT = str(pathlib.Path(__file__).parent.parent.parent)
os.environ.setdefault("CHAOSCONTROL_ROOT", _REPO_ROOT)

from chaoscontrol.public.engine_entry import build_arm_config, init_arm_topology, run_arm_submission  # noqa: E402


class _FakeHyperparams:
    """Minimal stand-in for a hyperparams object with no required attributes."""


def test_init_arm_topology_8gpu():
    role5 = init_arm_topology(rank=5, world_size=8)
    role6 = init_arm_topology(rank=6, world_size=8)
    role7 = init_arm_topology(rank=7, world_size=8)
    role0 = init_arm_topology(rank=0, world_size=8)
    assert role5.is_train_rank
    assert role6.is_packet_rank
    assert not role6.is_maintenance_rank
    assert role7.is_maintenance_rank
    assert not role7.is_packet_rank
    assert role0.is_train_rank
    assert role6.split_memory_ranks
    assert role7.split_memory_ranks


def test_init_arm_topology_4gpu():
    role2 = init_arm_topology(rank=2, world_size=4)
    role3 = init_arm_topology(rank=3, world_size=4)
    role0 = init_arm_topology(rank=0, world_size=4)
    assert role2.is_train_rank
    assert role3.is_packet_rank
    assert role3.is_maintenance_rank
    assert not role3.is_train_rank
    assert role0.is_train_rank
    assert not role3.split_memory_ranks


def test_init_arm_topology_single_gpu():
    role = init_arm_topology(rank=0, world_size=1)
    assert role.is_train_rank
    assert not role.is_packet_rank
    assert not role.is_maintenance_rank


def test_init_arm_topology_6gpu():
    # rank=5 is the shared packet+maintenance rank for world_size=6 (split=False);
    # the spec listed rank=4, which is a train rank — treating 5 as the intended rank.
    role5 = init_arm_topology(rank=5, world_size=6)
    assert role5.is_packet_rank
    assert role5.is_maintenance_rank
    assert not role5.split_memory_ranks


def test_packet_rank_value_8gpu():
    role = init_arm_topology(rank=0, world_size=8)
    assert role.packet_rank == 6
    assert role.maintenance_rank == 7


def test_build_arm_config_telemetry_tuned_defaults():
    cfg = build_arm_config(_FakeHyperparams())
    assert cfg["crct_memory_write_tokens_per_step"] == 256
    assert cfg["online_episodic_write_tokens_per_chunk"] == 64
    assert abs(cfg["crct_target_write_rate"] - 0.25) < 1e-6


def test_build_arm_config_eval_routing():
    cfg = build_arm_config(_FakeHyperparams())
    assert cfg.get("calc_types") == ["packet_online_cache"]
    assert cfg.get("headline_calc_type") == "packet_online_cache"


def test_build_arm_config_hyperparams_forwarded():
    hp = _FakeHyperparams()
    hp.model_dim = 384
    cfg = build_arm_config(hp)
    assert cfg.get("model_dim") == 384


def test_telemetry_overrides_win_over_lock(monkeypatch):
    import sys
    _exp26_dir = str(pathlib.Path(__file__).parent.parent.parent / "experiments" / "26_arm")
    if _exp26_dir not in sys.path:
        sys.path.insert(0, _exp26_dir)
    import exp26
    monkeypatch.setattr(exp26, "_crct_lock", lambda: {"crct_target_write_rate": 99.0})
    cfg = build_arm_config(_FakeHyperparams())
    assert abs(cfg["crct_target_write_rate"] - 0.25) < 1e-6


def test_run_arm_submission_import_chain():
    """Verify runner_fast_path can be imported and run_condition is callable."""
    import sys
    _exp_dir = str(pathlib.Path(__file__).parent.parent.parent / "experiments" / "23_fast_path")
    if _exp_dir not in sys.path:
        sys.path.insert(0, _exp_dir)
    from runner_fast_path import run_condition
    assert callable(run_condition)


def test_run_arm_submission_signature():
    """Verify run_arm_submission accepts the expected keyword arguments."""
    import inspect
    sig = inspect.signature(run_arm_submission)
    params = sig.parameters
    for key in ("config", "data_path", "sp_model_path", "budget_seconds", "output_json", "val_cache_dir"):
        assert key in params, f"Missing parameter: {key}"
