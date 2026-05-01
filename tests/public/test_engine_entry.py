import os
import pathlib

import pytest

_REPO_ROOT = str(pathlib.Path(__file__).parent.parent.parent)
os.environ.setdefault("CHAOSCONTROL_ROOT", _REPO_ROOT)

from chaoscontrol.public.engine_entry import build_arm_config, init_arm_topology, run_arm_submission  # noqa: E402


class _FakeHyperparams:
    """Minimal stand-in for a submission hyperparams object."""

    vocab_size = 16384
    model_dim = 384
    num_layers = 8
    seq_len = 512
    batch_size = 896


def test_init_arm_topology_8gpu():
    role6 = init_arm_topology(rank=6, world_size=8)
    role7 = init_arm_topology(rank=7, world_size=8)
    role0 = init_arm_topology(rank=0, world_size=8)
    assert role6.is_train_rank
    assert role7.is_maintenance_rank
    assert role7.is_packet_rank
    assert not role7.is_train_rank
    assert role0.is_train_rank
    assert not role6.split_memory_ranks
    assert not role7.split_memory_ranks


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
    assert role.packet_rank == 7
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
    hp.lm_head_tile_size = 2048
    hp.crct_teacher_param_sync_interval_steps = 6
    hp.max_steps = 1
    hp.eval_only = False
    cfg = build_arm_config(hp)
    assert cfg.get("model_dim") == 384
    assert cfg.get("lm_head_tile_size") == 2048
    assert cfg.get("crct_teacher_param_sync_interval_steps") == 6
    assert cfg.get("max_steps") == 1
    assert cfg.get("eval_only") is False


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
    for key in (
        "config",
        "data_path",
        "sp_model_path",
        "budget_seconds",
        "output_json",
        "output_ckpt",
        "val_cache_dir",
    ):
        assert key in params, f"Missing parameter: {key}"


def test_run_arm_submission_delegates_to_run_condition(monkeypatch):
    """Verify the wrapper actually delegates to run_condition with expected args."""
    import sys
    _exp_dir = str(pathlib.Path(__file__).parent.parent.parent / "experiments" / "23_fast_path")
    if _exp_dir not in sys.path:
        sys.path.insert(0, _exp_dir)
    import runner_fast_path

    calls = []

    def _fake_run_condition(config, **kwargs):
        calls.append({"config": config, "kwargs": kwargs})
        return {"train": {}, "eval": {"bpb": 1.5}, "artifact": {}}

    monkeypatch.setattr(runner_fast_path, "run_condition", _fake_run_condition)

    result = run_arm_submission(
        {"model_dim": 384},
        data_path="/data",
        sp_model_path="/tok.model",
        budget_seconds=600,
        output_json="/out.json",
        output_ckpt="/out.pt",
        val_cache_dir="/val",
    )

    assert len(calls) == 1
    call = calls[0]
    assert call["config"]["model_dim"] == 384
    assert call["kwargs"]["data_path"] == "/data"
    assert call["kwargs"]["budget_seconds"] == 600.0
    assert call["kwargs"]["output_ckpt"] == "/out.pt"
    assert "bpb" in result.get("eval", {})
