from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from chaoscontrol.eval.packet_online_cache_compare import (
    load_and_run_packet_online_cache_compare,
    run_packet_online_cache_compare,
)
from chaoscontrol.eval.ttt_eval import CalcTypeResult
from chaoscontrol.eval_stream.val_cache import DOC_DTYPE, TOKEN_DTYPE, ValCache


class _TinyCompareModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vocab_size = 8
        self.lm_head = nn.Linear(4, self.vocab_size, bias=False)
        self.outer_model = type("Outer", (), {})()
        self.outer_model._slots = []
        self._compare_artifact_path = None
        self._runs: list[bool] = []

    def encode(
        self,
        input_ids: torch.Tensor,
        *,
        memory_mode: str = "packet",
        initial_states=None,
        return_final_states: bool = False,
        episodic_residual=None,
        episodic_gate=None,
    ):
        self._runs.append(episodic_residual is not None)
        hidden = torch.zeros(input_ids.shape[0], input_ids.shape[1], 4)
        final_state = torch.zeros(input_ids.shape[0], 4)
        if return_final_states:
            return hidden, [final_state]
        return hidden

    def append_memory_from_hidden(self, hidden: torch.Tensor, **_kwargs):
        self.outer_model._slots.append(hidden.detach())
        return [{"slot_id": len(self.outer_model._slots) - 1}]


def _val_cache(tmp_path: Path) -> ValCache:
    tokens = np.asarray([1, 2, 3, 4], dtype=TOKEN_DTYPE)
    docs = np.asarray([(0, 0, 4, 4)], dtype=DOC_DTYPE)
    return ValCache(cache_dir=tmp_path, manifest={"schema_version": 1}, tokens=tokens, docs=docs)


def test_compare_writes_summary_json(tmp_path, monkeypatch):
    model = _TinyCompareModel()
    cache = _val_cache(tmp_path / "cache")
    out = tmp_path / "summary.json"

    def fake_packet_online_cache(ctx):
        seeded = bool(ctx.config.get("seeded", True))
        return CalcTypeResult(
            bpb=1.5 if seeded else 2.0,
            loss=3.0 if seeded else 4.0,
            docs_scored=1,
            tokens_scored=3,
            raw_bytes=4,
            hyperparams={"seeded": seeded},
            extra={"seeded": seeded},
        )

    monkeypatch.setattr(
        "chaoscontrol.eval.packet_online_cache_compare.packet_online_cache",
        fake_packet_online_cache,
    )

    result = run_packet_online_cache_compare(
        model=model,
        val_cache=cache,
        device=torch.device("cpu"),
        compare_config={"chunk_tokens": 4},
        output_json=out,
    )

    assert out.is_file()
    written = json.loads(out.read_text())
    assert written["compare_type"] == "packet_online_cache"
    assert written["seeded"]["bpb"] == 1.5
    assert written["empty"]["bpb"] == 2.0
    assert written["delta_bpb"] == 0.5
    assert result["delta_loss"] == 1.0


def test_compare_snapshots_empty_model_before_seeded_write(monkeypatch, tmp_path):
    """The empty run must not inherit cache writes from the seeded run."""
    model = _TinyCompareModel()
    cache = _val_cache(tmp_path / "cache")

    def fake_packet_online_cache(ctx):
        seeded = bool(ctx.config.get("seeded", True))
        slot_count = len(getattr(ctx.model.outer_model, "_slots", []))
        if seeded:
            ctx.model.outer_model._slots.append(torch.ones(1, 4))
        return CalcTypeResult(
            bpb=1.0 + slot_count,
            loss=2.0 + slot_count,
            docs_scored=1,
            tokens_scored=3,
            raw_bytes=4,
            hyperparams={"seeded": seeded},
            extra={"slot_count": slot_count},
        )

    monkeypatch.setattr(
        "chaoscontrol.eval.packet_online_cache_compare.packet_online_cache",
        fake_packet_online_cache,
    )

    result = run_packet_online_cache_compare(
        model=model,
        val_cache=cache,
        device=torch.device("cpu"),
        compare_config={"chunk_tokens": 4},
    )

    assert result["seeded"]["extra"]["slot_count"] == 0
    assert result["empty"]["extra"]["slot_count"] == 0
    assert result["delta_bpb"] == 0.0


def test_load_and_run_packet_online_cache_compare_persists(monkeypatch, tmp_path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"unused")
    cache_dir = tmp_path / "cache"
    cache = _val_cache(cache_dir)
    out = tmp_path / "compare" / "summary.json"

    model = _TinyCompareModel()
    model._compare_artifact_path = str(artifact)

    def fake_load_artifact(path, device):
        assert Path(path) == artifact
        return model, None, None

    def fake_load_val_cache(path):
        assert Path(path) == cache_dir
        return cache

    def fake_packet_online_cache(ctx):
        seeded = bool(ctx.config.get("seeded", True))
        return CalcTypeResult(
            bpb=10.0 if seeded else 12.0,
            loss=1.0 if seeded else 2.0,
            docs_scored=1,
            tokens_scored=3,
            raw_bytes=4,
            hyperparams={},
            extra={},
        )

    monkeypatch.setattr(
        "chaoscontrol.eval.packet_online_cache_compare.load_artifact",
        fake_load_artifact,
    )
    monkeypatch.setattr(
        "chaoscontrol.eval.packet_online_cache_compare.load_val_cache",
        fake_load_val_cache,
    )
    monkeypatch.setattr(
        "chaoscontrol.eval.packet_online_cache_compare.packet_online_cache",
        fake_packet_online_cache,
    )

    result = load_and_run_packet_online_cache_compare(
        artifact_path=artifact,
        val_cache_dir=cache_dir,
        device="cpu",
        compare_config={"chunk_tokens": 4},
        output_json=out,
    )

    assert out.is_file()
    assert result["seeded"]["bpb"] == 10.0
    assert result["empty"]["bpb"] == 12.0
