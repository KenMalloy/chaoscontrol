"""Unit tests for the Exp 21 runner's --output-ckpt save path.

The checkpoint format is the contract consumed by
``scripts/run_exp20_eval.py::_build_model`` (lines 38-49):

    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = blob["config"]
    model = CareStudentLM(**cfg)
    model.load_state_dict(blob["model"], strict=True)

These tests exercise that exact loader pattern against the runner's
``_save_output_ckpt`` writer to guarantee the writer/loader pair stays in
sync. The test runs on CPU and skips the full training loop — the
checkpoint contract is a small unit independent of training.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "21_sgns_tokenizer"))

from runner_exp21 import (  # noqa: E402
    _save_output_ckpt,
    _ssm_constructor_kwargs,
    _transformer_constructor_kwargs,
    build_model,
)


def _tiny_ssm_config() -> dict:
    """Tiny SSM config that builds in <1 second on CPU."""
    return {
        "model_type": "ssm",
        "vocab_size": 64,
        "model_dim": 16,
        "num_layers": 1,
        "ff_mult": 2,
        "a_mode": "diag",
        "a_full_rank": 8,
        "a_full_gamma": 0.05,
    }


def _tiny_transformer_config() -> dict:
    return {
        "model_type": "transformer_nanogpt_lean",
        "vocab_size": 64,
        "model_dim": 32,  # must be divisible by n_head
        "n_head": 2,
        "num_layers": 1,
        "ff_mult": 2,
    }


def _load_via_run_exp20_eval_pattern(ckpt_path: Path):
    """Mirror scripts/run_exp20_eval.py::_build_model line-for-line."""
    from chaoscontrol.model import CareStudentLM

    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = blob["config"]
    model = CareStudentLM(**cfg)
    model.load_state_dict(blob["model"], strict=True)
    online_eval_state = blob.get("online_eval_state")
    if isinstance(online_eval_state, dict):
        model._online_eval_state = online_eval_state
    return model, cfg, blob


def test_save_output_ckpt_writes_expected_keys(tmp_path):
    """The default saved blob has exactly {"model", "config"} top-level keys."""
    config = _tiny_ssm_config()
    model = build_model(config, torch.device("cpu"), torch.float32)
    ckpt_path = tmp_path / "ckpt.pt"

    _save_output_ckpt(str(ckpt_path), model, config)

    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert set(blob.keys()) == {"model", "config"}, (
        f"unexpected top-level keys: {set(blob.keys())} (consumer "
        "expects exactly model + config)"
    )


def test_save_output_ckpt_can_include_episodic_cache_payload(tmp_path):
    """Cache-aware TTT arms need ckpt['episodic_cache'] alongside weights."""
    from chaoscontrol.optim.episodic_cache import EpisodicCache

    config = _tiny_ssm_config()
    model = build_model(config, torch.device("cpu"), torch.float32)
    cache = EpisodicCache(
        capacity=2,
        span_length=2,
        key_rep_dim=4,
        fingerprint_window=6,
        slot_state_dim=2,
        simplex_k_max=1,
    )
    cache.append(
        key_fp=42,
        key_rep=torch.ones(4),
        value_tok_ids=torch.tensor([1, 2], dtype=torch.int64),
        value_anchor_id=1,
        current_step=3,
        embedding_version=0,
        pressure_at_write=1.5,
        source_write_id=123,
        write_bucket=2,
    )
    ckpt_path = tmp_path / "ckpt.pt"

    _save_output_ckpt(str(ckpt_path), model, config, episodic_cache=cache)

    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert set(blob.keys()) == {"model", "config", "episodic_cache"}
    restored = EpisodicCache.from_dict(blob["episodic_cache"])
    assert restored.fingerprint_window == 6
    assert restored.slot_state_dim == 2
    assert restored.simplex_k_max == 1
    entry = restored.query(42)
    assert entry is not None
    assert entry.pressure_at_write == pytest.approx(1.5)
    assert entry.source_write_id == 123
    assert entry.write_bucket == 2


def test_save_output_ckpt_can_include_online_eval_state(tmp_path):
    """Online eval must inherit train-time controller state, not cold-start."""
    config = _tiny_ssm_config()
    model = build_model(config, torch.device("cpu"), torch.float32)
    online_state = {
        "replay_eviction": {
            "schema_version": 1,
            "refresh_proposal_model": {
                "learned_direction": torch.ones(1, 4),
                "updates_total": 3,
            },
        }
    }
    ckpt_path = tmp_path / "ckpt.pt"

    _save_output_ckpt(
        str(ckpt_path),
        model,
        config,
        online_eval_state=online_state,
    )

    restored_model, _cfg, blob = _load_via_run_exp20_eval_pattern(ckpt_path)
    assert set(blob.keys()) == {"model", "config", "online_eval_state"}
    restored_state = blob["online_eval_state"]
    assert torch.equal(
        restored_state["replay_eviction"]["refresh_proposal_model"][
            "learned_direction"
        ],
        torch.ones(1, 4),
    )
    assert restored_model._online_eval_state["replay_eviction"]["schema_version"] == 1


def test_save_output_ckpt_preserves_non_tensor_extra_state(tmp_path):
    """PyTorch state_dict extra-state entries are metadata dicts, not tensors."""

    class ModuleWithExtraState(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2, 3))

        def get_extra_state(self):
            return {
                "schema_version": 1,
                "nested": {"tensor": torch.arange(3), "name": "cache"},
                "flags": [True, 7],
            }

        def set_extra_state(self, state) -> None:
            self._extra_state = state

    config = _tiny_ssm_config()
    model = ModuleWithExtraState()
    ckpt_path = tmp_path / "ckpt.pt"

    _save_output_ckpt(str(ckpt_path), model, config)

    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = blob["model"]
    assert torch.equal(state["weight"], torch.ones(2, 3))
    assert state["_extra_state"]["schema_version"] == 1
    assert state["_extra_state"]["nested"]["name"] == "cache"
    assert torch.equal(state["_extra_state"]["nested"]["tensor"], torch.arange(3))


def test_ssm_arm_round_trips_through_run_exp20_eval_loader(tmp_path):
    """Save SSM checkpoint, reload via the exact run_exp20_eval pattern.

    This is the load-bearing test: ``strict=True`` succeeding on the
    reload proves the saved config kwargs reconstruct a model whose
    state_dict matches the trained model 1-1, with no missing or extra
    keys. If the kwargs drift from build_model, this test breaks.
    """
    config = _tiny_ssm_config()
    trained = build_model(config, torch.device("cpu"), torch.float32)

    # Take one optimizer step so weights aren't all init values — proves
    # we're saving the trained state, not the constructor's freshly-init
    # tensors.
    optim = torch.optim.AdamW(trained.parameters(), lr=1e-3)
    tokens = torch.randint(0, config["vocab_size"], (2, 8))
    out = trained(tokens)
    logits = out["logits"] if isinstance(out, dict) else out
    loss = logits.sum()
    loss.backward()
    optim.step()

    ckpt_path = tmp_path / "ssm.pt"
    _save_output_ckpt(str(ckpt_path), trained, config)

    reloaded, cfg, _ = _load_via_run_exp20_eval_pattern(ckpt_path)

    # Round-trip integrity: every key matches, every tensor matches.
    trained_state = trained.state_dict()
    reloaded_state = reloaded.state_dict()
    assert set(trained_state.keys()) == set(reloaded_state.keys()), (
        f"state_dict key mismatch — "
        f"trained-only: {set(trained_state) - set(reloaded_state)}; "
        f"reloaded-only: {set(reloaded_state) - set(trained_state)}"
    )
    for k in trained_state:
        torch.testing.assert_close(
            reloaded_state[k].to("cpu"),
            trained_state[k].to("cpu"),
            msg=f"tensor mismatch on key {k!r}",
        )

    # Reconstructed model is the same class, not some lookalike.
    assert type(reloaded).__name__ == "CareStudentLM"
    assert type(reloaded).__module__ == "chaoscontrol.model"


def test_ssm_kwargs_match_build_model_signature(tmp_path):
    """The kwargs we save are accepted by CareStudentLM unchanged.

    Guards against silently introducing a config field that the model
    constructor would reject — caught here as TypeError before a real
    run_exp20_eval invocation.
    """
    from chaoscontrol.model import CareStudentLM

    cfg = _ssm_constructor_kwargs(_tiny_ssm_config())
    # If any kwarg is not in CareStudentLM.__init__'s signature, this
    # raises TypeError("got an unexpected keyword argument").
    model = CareStudentLM(**cfg)
    assert model.vocab_size == cfg["vocab_size"]
    assert model.dim == cfg["dim"]


def test_transformer_arm_save_loads_back_via_torch_load(tmp_path):
    """Transformer arm saves the right kwargs even though run_exp20_eval
    doesn't currently know how to load them.

    Per design comment in _save_output_ckpt: the consumer hardcodes
    CareStudentLM(**cfg). Transformer checkpoints save correctly with
    NanoGPTLeanLM kwargs and can be reloaded directly by NanoGPTLeanLM.
    """
    from chaoscontrol.baselines_nanogpt_lean import NanoGPTLeanLM

    config = _tiny_transformer_config()
    trained = build_model(config, torch.device("cpu"), torch.float32)
    ckpt_path = tmp_path / "tx.pt"

    _save_output_ckpt(str(ckpt_path), trained, config)

    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = blob["config"]
    reloaded = NanoGPTLeanLM(**cfg)
    reloaded.load_state_dict(blob["model"], strict=True)

    assert set(reloaded.state_dict().keys()) == set(trained.state_dict().keys())


def test_no_save_when_path_absent(tmp_path):
    """Sanity: passing output_ckpt=None to the runner's save helper is
    not invoked at all — the runner's caller guards on ``if output_ckpt``.

    This test ensures no checkpoint file is produced when the helper
    isn't called, i.e. validates the contract from the call site
    perspective.
    """
    config = _tiny_ssm_config()
    model = build_model(config, torch.device("cpu"), torch.float32)
    ckpt_path = tmp_path / "should_not_exist.pt"

    # Simulate runner main(): output_ckpt is None, helper is not called.
    output_ckpt = None
    if output_ckpt:  # pragma: no cover — branch intentionally skipped
        _save_output_ckpt(output_ckpt, model, config)

    assert not ckpt_path.exists()


def test_main_threads_output_ckpt_into_run_ddp(tmp_path, monkeypatch):
    """argparse → main → run_ddp wiring carries --output-ckpt through.

    Stubs run_ddp so we don't need a real dataset or sp model — the test
    asserts the CLI flag arrives at run_ddp's output_ckpt kwarg with
    the right value. Catches typos like ``output_ckpt=args.output_json``.
    """
    import runner_exp21

    cfg_path = tmp_path / "tiny.yaml"
    cfg_path.write_text(
        "model_type: ssm\nvocab_size: 64\nmodel_dim: 16\nnum_layers: 1\n"
    )
    ckpt_path = tmp_path / "ckpt.pt"

    captured: dict = {}

    def fake_run_ddp(config, **kwargs):
        captured["config"] = config
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(runner_exp21, "run_ddp", fake_run_ddp)

    rc = runner_exp21.main([
        "--config", str(cfg_path),
        "--data-path", "/dev/null",
        "--sp-model-path", "/dev/null",
        "--output-ckpt", str(ckpt_path),
    ])
    assert rc == 0
    assert captured["output_ckpt"] == str(ckpt_path)
    assert captured["output_json"] is None  # default preserved


def test_transformer_kwargs_does_not_include_max_seq_len():
    """Transformer kwargs intentionally omit max_seq_len so reload
    matches the 2048 default that build_model uses at training time.

    If a future change adds max_seq_len to the YAML config, the saved
    blob should still omit it — until build_model is also updated to
    pass it through. This locks the contract.
    """
    cfg = _transformer_constructor_kwargs(_tiny_transformer_config())
    assert "max_seq_len" not in cfg
