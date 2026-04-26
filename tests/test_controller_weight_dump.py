"""Phase D5 -- PyTorch controller pretrain weights -> C++ binary dump."""
from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path

import pytest
import torch

from chaoscontrol.kernels import _cpu_ssm_controller as _ext


REPO = Path(__file__).resolve().parents[1]
PRETRAIN_PATH = (
    REPO / "experiments" / "25_controller_pretrain" / "pretrain_controller.py"
)
DUMP_PATH = REPO / "experiments" / "25_controller_pretrain" / "dump_to_cpp.py"


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _tiny_model():
    pretrain = _load_module("controller_pretrain_dump_test", PRETRAIN_PATH)
    cfg = pretrain.PretrainConfig(
        d_global=4,
        n_layers=2,
        feature_dim=3,
        n_slots_per_query=5,
        seed=17,
    )
    model = pretrain.ControllerPretrainModel(cfg)
    with torch.no_grad():
        for offset, (_name, param) in enumerate(model.named_parameters()):
            values = torch.linspace(
                -0.20,
                0.20,
                steps=param.numel(),
                dtype=torch.float32,
            ).reshape_as(param)
            param.copy_(values + 0.01 * offset)
    model.eval()
    return pretrain, cfg, model


def test_pretrain_checkpoint_dumps_binary_and_loads_through_cpp(tmp_path: Path):
    if _ext._C is None:
        pytest.skip("cpu_ssm_controller C++ extension not built")

    pretrain, cfg, model = _tiny_model()
    checkpoint_path = tmp_path / "controller_pretrain.pt"
    dump_path = tmp_path / "controller_pretrain.cswg"
    torch.save(
        {
            "config": cfg.__dict__,
            "weights": {k: v.detach().clone() for k, v in model.state_dict().items()},
        },
        checkpoint_path,
    )

    dumper = _load_module("controller_dump_to_cpp_test", DUMP_PATH)
    manifest = dumper.dump_checkpoint_to_cpp(checkpoint_path, dump_path)

    header = dump_path.read_bytes()[:28]
    magic, version, n_layers, d_global, d_slot, feature_dim, dtype_code = (
        struct.unpack(
            "<4sIIIIII",
            header,
        )
    )
    assert magic == b"CSWG"
    assert version == 1
    assert n_layers == cfg.n_layers
    assert d_global == cfg.d_global
    assert d_slot == cfg.n_slots_per_query
    assert feature_dim == cfg.feature_dim
    assert dtype_code == dumper.DTYPE_FP16
    assert manifest["tensor_order"] == dumper.TENSOR_ORDER
    assert manifest["fp32_vs_fp16_max_abs_drift"] > 0.0

    loaded = _ext.load_weights_from_path(str(dump_path))
    loaded_state = {key: loaded[key] for key in dumper.TENSOR_ORDER}
    expected_state = {
        key: value.detach().to(torch.float16).to(torch.float32).contiguous()
        for key, value in model.state_dict().items()
    }
    for key in dumper.TENSOR_ORDER:
        torch.testing.assert_close(loaded_state[key], expected_state[key])

    quantized_model = pretrain.ControllerPretrainModel(cfg)
    quantized_model.load_state_dict(expected_state)
    quantized_model.eval()

    x = torch.tensor(
        [[0.25, -0.50, 1.00], [-0.75, 0.125, 0.50]],
        dtype=torch.float32,
    )
    with torch.no_grad():
        ref_logits, ref_value = quantized_model(x)
        cpp_logits, cpp_value = _ext.forward_pretrain_model(x, loaded)
        fp32_logits, fp32_value = model(x)

    torch.testing.assert_close(cpp_logits, ref_logits, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(cpp_value, ref_value, atol=1e-5, rtol=1e-5)

    measured_drift = max(
        (fp32_logits - cpp_logits).abs().max().item(),
        (fp32_value - cpp_value).abs().max().item(),
    )
    assert measured_drift > 0.0
    print(f"fp32_vs_fp16_forward_max_abs_drift={measured_drift:.8g}")
