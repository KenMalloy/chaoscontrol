"""Phase S4 -- CSWG v3 self-describing export round-trip tests."""
from __future__ import annotations

import importlib.util
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


REPO = Path(__file__).resolve().parents[1]
PRETRAIN_DIR = REPO / "experiments" / "25_controller_pretrain"
PRETRAIN_PATH = PRETRAIN_DIR / "pretrain_controller.py"
DUMP_PATH = PRETRAIN_DIR / "dump_to_cpp.py"


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_pretrain():
    return _load_module("controller_pretrain_for_dump_tests", PRETRAIN_PATH)


def _load_dumper():
    return _load_module("controller_dump_to_cpp_test", DUMP_PATH)


def _trained_model(n_batches: int = 20) -> tuple:
    pretrain = _load_pretrain()
    cfg = pretrain.PretrainConfig(n_batches=n_batches, seed=1337)
    result = pretrain.train(cfg)
    return pretrain, cfg, result["model"]


def test_cswg_v3_header_and_manifest_layout(tmp_path: Path):
    _pretrain, cfg, model = _trained_model(n_batches=5)
    dumper = _load_dumper()
    out = tmp_path / "simplex_policy_v1.cswg"
    manifest = dumper.dump_model_to_cswg_v3(model, out)

    raw = out.read_bytes()
    magic, version, dtype_code, manifest_nbytes, reserved = struct.unpack(
        "<4sIIII", raw[: dumper.HEADER_SIZE]
    )
    assert magic == b"CSWG"
    assert version == 3
    assert dtype_code == dumper.DTYPE_FP16
    assert reserved == 0
    assert manifest_nbytes == manifest["manifest_bytes"]
    manifest_json = json.loads(
        raw[dumper.HEADER_SIZE: dumper.HEADER_SIZE + manifest_nbytes]
        .decode("utf-8")
    )
    assert manifest_json["version"] == 3
    assert manifest_json["dims"]["n_vertices"] == cfg.n_vertices == 16
    assert manifest_json["dims"]["n_heads"] == cfg.n_heads == 2
    assert {t["name"] for t in manifest_json["tensors"]} == set(
        dumper.TENSOR_NAMES
    )


def test_cswg_v3_roundtrip_matches_fp16(tmp_path: Path):
    _pretrain, _cfg, model = _trained_model(n_batches=10)
    dumper = _load_dumper()
    out = tmp_path / "simplex_policy_v1.cswg"
    dumper.dump_model_to_cswg_v3(model, out)

    parsed = dumper.load_cswg_v3(out)
    expected_state = {
        name: tensor.detach().to(torch.float16).to(torch.float32).contiguous()
        for name, tensor in model.state_dict().items()
    }
    for name in dumper.TENSOR_NAMES:
        loaded = parsed["tensors"][name]
        ref = expected_state[name]
        torch.testing.assert_close(
            loaded.reshape(-1).float(),
            ref.reshape(-1).float(),
            atol=1e-3,
            rtol=1e-3,
            msg=f"tensor {name!r} mismatched after fp16 round-trip",
        )


def test_cswg_v3_rejects_non_v3_payload(tmp_path: Path):
    dumper = _load_dumper()
    bad = tmp_path / "fake_v2.cswg"
    bad.write_bytes(struct.pack("<4sIIII", b"CSWG", 2, 1, 2, 0) + b"{}")
    with pytest.raises(ValueError, match=r"version"):
        dumper.load_cswg_v3(bad)


def test_cswg_v3_rejects_bad_magic(tmp_path: Path):
    dumper = _load_dumper()
    bad = tmp_path / "bad.cswg"
    bad.write_bytes(b"NOPE" + b"\x00" * 64)
    with pytest.raises(ValueError, match=r"magic"):
        dumper.load_cswg_v3(bad)


def test_cswg_v3_manifest_is_self_describing_for_hxh(tmp_path: Path):
    _pretrain, cfg, model = _trained_model(n_batches=5)
    dumper = _load_dumper()
    out = tmp_path / "simplex_policy_v1.cswg"
    manifest = dumper.dump_model_to_cswg_v3(model, out)

    entries = {entry["name"]: entry for entry in manifest["tensors"]}
    assert entries["W_q"]["shape"] == [cfg.n_heads, cfg.h, cfg.h]
    assert entries["W_k"]["shape"] == [cfg.n_heads, cfg.h, cfg.h]
    assert entries["W_v"]["shape"] == [cfg.n_heads, cfg.h, cfg.h]
    assert entries["W_o"]["shape"] == [cfg.n_heads, cfg.h]
    assert entries["W_e"]["shape"] == [cfg.n_heads, cfg.k_e]
    assert entries["lambda_hxh"]["shape"] == []
    assert manifest["payload_bytes"] == sum(
        int(entry["nbytes"]) for entry in manifest["tensors"]
    )


def test_cswg_v3_payload_byte_equality_with_state_dict(tmp_path: Path):
    _pretrain, _cfg, model = _trained_model(n_batches=5)
    dumper = _load_dumper()
    out = tmp_path / "simplex_policy_v1.cswg"
    dumper.dump_model_to_cswg_v3(model, out)

    raw = out.read_bytes()
    magic, version, dtype_code, manifest_nbytes, reserved = struct.unpack(
        "<4sIIII", raw[: dumper.HEADER_SIZE]
    )
    assert magic == b"CSWG" and version == 3 and dtype_code == dumper.DTYPE_FP16
    manifest = json.loads(
        raw[dumper.HEADER_SIZE: dumper.HEADER_SIZE + manifest_nbytes]
        .decode("utf-8")
    )
    payload = raw[dumper.HEADER_SIZE + manifest_nbytes:]
    state = model.state_dict()
    for entry in manifest["tensors"]:
        name = entry["name"]
        tensor = state[name].detach().to(torch.float32).contiguous()
        offset = int(entry["offset"])
        nbytes = int(entry["nbytes"])
        slab = payload[offset: offset + nbytes]
        loaded = torch.from_numpy(
            np.frombuffer(slab, dtype=np.float16).copy().astype(np.float32)
        ).reshape(tensor.shape)
        ref = tensor.to(torch.float16).to(torch.float32)
        torch.testing.assert_close(loaded, ref, atol=0.0, rtol=0.0)
