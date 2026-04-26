"""Phase S4 -- CSWG v2 export round-trip + budget tests.

Verifies that the simplex controller pretrain dump:
  1. Writes a 32-byte header with magic ``CSWG``, version 2, and the
     locked simplex dimensions.
  2. Round-trips the trained tensors through fp16 and matches the
     in-memory model state at fp16 precision.
  3. Rejects v1-formatted payloads with a clear error.
  4. Stores ~647 fp16 tensor values, matching the design-doc artifact
     budget.

The C++ V2 loader is part of Phase S2; this file only exercises the
Python writer + parser pair.
"""
from __future__ import annotations

import importlib.util
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


def _trained_model(n_batches: int = 50) -> tuple:
    """Run a short training pass so the dumped tensors are non-trivial.

    A short run is enough to verify the export path; convergence is
    pinned by ``test_simplex_pretrain_synthetic_convergence``.
    """
    pretrain = _load_pretrain()
    cfg = pretrain.PretrainConfig(n_batches=n_batches, seed=1337)
    result = pretrain.train(cfg)
    return pretrain, cfg, result["model"]


def test_cswg_v2_header_layout(tmp_path: Path):
    pretrain, cfg, model = _trained_model(n_batches=10)
    dumper = _load_dumper()
    out = tmp_path / "simplex_policy_v1.cswg"
    manifest = dumper.dump_model_to_cswg_v2(model, out)

    assert manifest["header_bytes"] == 32
    assert manifest["version"] == 2
    raw = out.read_bytes()
    assert len(raw) == manifest["file_bytes"]

    (magic, version, n_vertices, k_v, k_e, k_s, h, dtype_code, reserved) = (
        struct.unpack("<4sIIIIIIHH", raw[:32])
    )
    assert magic == b"CSWG"
    assert version == 2
    assert n_vertices == cfg.n_vertices == 16
    assert k_v == cfg.k_v == 16
    assert k_e == cfg.k_e == 1
    assert k_s == cfg.k_s == 4
    assert h == cfg.h == 32
    assert dtype_code == dumper.DTYPE_FP16
    assert reserved == 0


def test_cswg_v2_roundtrip_matches_fp16(tmp_path: Path):
    pretrain, cfg, model = _trained_model(n_batches=20)
    dumper = _load_dumper()
    out = tmp_path / "simplex_policy_v1.cswg"
    dumper.dump_model_to_cswg_v2(model, out)

    parsed = dumper.load_cswg_v2(out)
    expected_state = {
        name: tensor.detach().to(torch.float16).to(torch.float32).contiguous()
        for name, tensor in model.state_dict().items()
    }
    for name, _shape_expr in dumper.TENSOR_ORDER:
        loaded = parsed["tensors"][name]
        ref = expected_state[name]
        # Scalars (b_lh, alpha, temperature) are stored as 0-d tensors;
        # state_dict() may return either 0-d or 1-d (1,) for buffers.
        # Compare flattened to dodge the shape mismatch.
        torch.testing.assert_close(
            loaded.reshape(-1).float(),
            ref.reshape(-1).float(),
            atol=1e-3,
            rtol=1e-3,
            msg=f"tensor {name!r} mismatched after fp16 round-trip",
        )


def test_cswg_v2_rejects_v1_payload(tmp_path: Path):
    """Feed a v1-formatted header to load_cswg_v2 and expect a clear error."""
    dumper = _load_dumper()
    v1_path = tmp_path / "fake_v1.cswg"
    # v1 header layout: 4-byte magic + 6 uint32 = 28 bytes.
    v1_header = struct.pack(
        "<4sIIIIII",
        b"CSWG",
        1,        # version=1
        4,        # n_layers
        128,      # d_global
        16,       # d_slot
        16,       # feature_dim
        1,        # dtype
    )
    # Pad with arbitrary bytes so the file is large enough to attempt
    # a payload read; the parser must bail on the version mismatch
    # before getting that far.
    v1_path.write_bytes(v1_header + b"\x00" * 1024)

    with pytest.raises(ValueError, match=r"version"):
        dumper.load_cswg_v2(v1_path)


def test_cswg_v2_rejects_bad_magic(tmp_path: Path):
    dumper = _load_dumper()
    bad_path = tmp_path / "bad.cswg"
    bad_path.write_bytes(b"NOPE" + b"\x00" * 64)
    with pytest.raises(ValueError, match=r"magic"):
        dumper.load_cswg_v2(bad_path)


def test_cswg_v2_param_count_matches_design_budget(tmp_path: Path):
    """Total fp16 element count is the locked artifact-budget number."""
    pretrain, cfg, model = _trained_model(n_batches=5)
    dumper = _load_dumper()
    out = tmp_path / "simplex_policy_v1.cswg"
    manifest = dumper.dump_model_to_cswg_v2(model, out)

    # Design doc Artifact budget table:
    #   W_vp(K_v=16, H=32)  = 512
    #   b_vp(H=32)          = 32
    #   W_lh(H=32)          = 32
    #   b_lh()              = 1
    #   W_sb(K_s=4)         = 4
    #   alpha()             = 1
    #   temperature()       = 1
    #   bucket_embed(8, 8)  = 64
    # ------------------------------
    #   total               = 647 fp16 = 1294 bytes payload
    expected_floats = 512 + 32 + 32 + 1 + 4 + 1 + 1 + 64
    assert expected_floats == 647
    assert manifest["payload_bytes"] == expected_floats * 2 == 1294
    assert manifest["file_bytes"] == 32 + 1294 == 1326


def test_cswg_v2_payload_byte_equality_with_state_dict(tmp_path: Path):
    """Bypass the parser and verify raw fp16 bytes == state_dict.fp16."""
    pretrain, cfg, model = _trained_model(n_batches=10)
    dumper = _load_dumper()
    out = tmp_path / "simplex_policy_v1.cswg"
    dumper.dump_model_to_cswg_v2(model, out)

    raw = out.read_bytes()
    cursor = 32  # skip header
    state = model.state_dict()
    for name, _ in dumper.TENSOR_ORDER:
        tensor = state[name].detach().to(torch.float32).contiguous()
        n_elems = tensor.numel()
        n_bytes = n_elems * 2
        slab = raw[cursor : cursor + n_bytes]
        cursor += n_bytes
        loaded = (
            torch.from_numpy(np.frombuffer(slab, dtype=np.float16).copy().astype(np.float32))
            .reshape(tensor.shape)
            .contiguous()
        )
        # Compare against the source cast to fp16 then back: the file
        # is the authoritative quantized form.
        ref = tensor.to(torch.float16).to(torch.float32)
        torch.testing.assert_close(loaded, ref, atol=0.0, rtol=0.0)
    assert cursor == len(raw)
